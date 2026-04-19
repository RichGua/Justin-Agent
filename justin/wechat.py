from __future__ import annotations

import asyncio
import json
import os
import random
import ssl
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import qrcode

from .config import AgentConfig
from .runtime import JustinRuntime

ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
EP_GET_BOT_QR = "ilink/bot/get_bot_qrcode"
EP_GET_QR_STATUS = "ilink/bot/get_qrcode_status"
EP_GET_UPDATES = "ilink/bot/getupdates"
EP_SEND_MESSAGE = "ilink/bot/sendmessage"

QR_TIMEOUT_MS = 15000
POLL_TIMEOUT_MS = 35000

_runtime: JustinRuntime | None = None
_config: AgentConfig | None = None


def _make_ssl_connector() -> aiohttp.TCPConnector:
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return aiohttp.TCPConnector(ssl=ctx)


async def _api_get(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    timeout_ms: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    app_id = _config.wechat_app_id if _config else "ilink_app_1"
    headers = {
        "iLink-App-Id": app_id,
        "iLink-App-ClientVersion": "1.0.0",
    }
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000.0)
    async with session.get(url, headers=headers, timeout=timeout) as resp:
        if not resp.ok:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
        return await resp.json(content_type=None)


async def _api_post(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_ms: int,
    token: str = "",
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    # Minimal base info required by iLink
    body_obj = {**payload, "base_info": {"channel_version": "1.0.0"}}
    body = json.dumps(body_obj).encode("utf-8")

    app_id = _config.wechat_app_id if _config else "ilink_app_1"
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "iLink-App-Id": app_id,
        "iLink-App-ClientVersion": "1.0.0",
        "X-WECHAT-UIN": str(random.randint(1000000, 9999999)),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["ilink_bot_token"] = token

    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000.0)
    async with session.post(url, data=body, headers=headers, timeout=timeout) as resp:
        if not resp.ok:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
        return await resp.json(content_type=None)


async def qr_login() -> dict[str, str] | None:
    print("\nStarting iLink Bot API QR Login...")
    async with aiohttp.ClientSession(connector=_make_ssl_connector()) as session:
        try:
            qr_resp = await _api_get(
                session,
                base_url=ILINK_BASE_URL,
                endpoint=f"{EP_GET_BOT_QR}?bot_type=3",
                timeout_ms=QR_TIMEOUT_MS,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[WeChat] Failed to fetch QR code: {exc}")
            return None

        qrcode_value = str(qr_resp.get("qrcode") or "")
        qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
        if not qrcode_value:
            print("[WeChat] QR response missing qrcode token")
            return None

        # WeChat requires the raw `qrcode_value` (the short token) to be encoded in the QR for scanning.
        # `qrcode_img_content` is sometimes provided as a base64 image or a long URL, but scanning it 
        # often leads to a network error because the iLink Bot device flow expects the raw token.
        qr_scan_data = qrcode_value

        print("\nPlease scan the following QR code with WeChat:")
        if qrcode_url:
            print(qrcode_url)
            
        try:
            qr = qrcode.QRCode()
            qr.add_data(qr_scan_data)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except Exception as _qr_exc:
            print(f"(Terminal QR render failed: {_qr_exc}, please open the URL above)")

        deadline = time.time() + 480
        current_base_url = ILINK_BASE_URL
        refresh_count = 0

        while time.time() < deadline:
            try:
                status_resp = await _api_get(
                    session,
                    base_url=current_base_url,
                    endpoint=f"{EP_GET_QR_STATUS}?qrcode={qrcode_value}",
                    timeout_ms=QR_TIMEOUT_MS,
                )
            except asyncio.TimeoutError:
                await asyncio.sleep(1)
                continue
            except Exception as exc:
                print(f"[WeChat] QR poll error: {exc}")
                await asyncio.sleep(1)
                continue

            status = str(status_resp.get("status") or "wait")
            if status == "wait":
                print(".", end="", flush=True)
            elif status == "scaned":
                print("\nScanned! Please confirm on your phone...")
            elif status == "scaned_but_redirect":
                redirect_host = str(status_resp.get("redirect_host") or "")
                if redirect_host:
                    current_base_url = f"https://{redirect_host}"
            elif status == "expired":
                refresh_count += 1
                if refresh_count > 3:
                    print("\nQR code expired multiple times. Please restart.")
                    return None
                print(f"\nQR code expired, refreshing... ({refresh_count}/3)")
                try:
                    qr_resp = await _api_get(
                        session,
                        base_url=ILINK_BASE_URL,
                        endpoint=f"{EP_GET_BOT_QR}?bot_type=3",
                        timeout_ms=QR_TIMEOUT_MS,
                    )
                    qrcode_value = str(qr_resp.get("qrcode") or "")
                    qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
                    qr_scan_data = qrcode_value
                    if qrcode_url:
                        print(qrcode_url)
                    try:
                        qr = qrcode.QRCode()
                        qr.add_data(qr_scan_data)
                        qr.make(fit=True)
                        qr.print_ascii(invert=True)
                    except Exception:
                        pass
                except Exception as exc:
                    print(f"[WeChat] QR refresh failed: {exc}")
                    return None
            elif status == "confirmed":
                account_id = str(status_resp.get("ilink_bot_id") or "")
                token = str(status_resp.get("bot_token") or "")
                base_url = str(status_resp.get("baseurl") or ILINK_BASE_URL)
                if not account_id or not token:
                    print("[WeChat] QR confirmed but payload incomplete")
                    return None
                print(f"\nWeChat connected successfully! Account ID: {account_id}")
                return {
                    "account_id": account_id,
                    "token": token,
                    "base_url": base_url,
                }
            await asyncio.sleep(1)

        print("\nWeChat login timed out.")
        return None


async def _send_message(
    session: aiohttp.ClientSession,
    base_url: str,
    token: str,
    to_user: str,
    content: str,
    context_token: str = "",
) -> None:
    payload = {
        "bot_id": "bot_id", # iLink ignores this but requires the field
        "msg_type": 1,      # Text
        "content": content,
        "to_user_id": to_user,
    }
    if context_token:
        payload["context_token"] = context_token

    try:
        resp = await _api_post(
            session,
            base_url,
            EP_SEND_MESSAGE,
            payload,
            timeout_ms=10000,
            token=token,
        )
        if resp.get("errcode", 0) != 0:
            print(f"[WeChat] Send error: {resp}")
    except Exception as exc:
        print(f"[WeChat] Failed to send message: {exc}")


async def _poll_loop(
    account_id: str,
    token: str,
    base_url: str,
    runtime: JustinRuntime,
) -> None:
    sync_buf = ""
    context_tokens: dict[str, str] = {}  # In-memory context token store for outbound replies
    
    print("\nListening for WeChat messages via iLink API...")
    async with aiohttp.ClientSession(connector=_make_ssl_connector()) as session:
        while True:
            payload = {
                "limit": 50,
                "get_updates_buf": sync_buf,
                "longpolling_timeout_ms": POLL_TIMEOUT_MS,
            }
            try:
                resp = await _api_post(
                    session,
                    base_url,
                    EP_GET_UPDATES,
                    payload,
                    timeout_ms=POLL_TIMEOUT_MS + 5000,
                    token=token,
                )
                errcode = resp.get("errcode", 0)
                if errcode == -14:
                    print("\n[WeChat] Session expired. Please restart to scan QR again.")
                    return
                if errcode != 0:
                    print(f"[WeChat] Poll errcode {errcode}: {resp.get('errmsg')}")
                    await asyncio.sleep(5)
                    continue

                sync_buf = str(resp.get("get_updates_buf") or sync_buf)
                msgs = resp.get("msg_list") or []
                
                for msg in msgs:
                    msg_type = msg.get("msg_type")
                    if msg_type != 1:  # Only process text for now
                        continue
                        
                    content = str(msg.get("content") or "")
                    from_user = str(msg.get("from_user_id") or "")
                    ctx_token = str(msg.get("context_token") or "")
                    
                    if from_user and ctx_token:
                        context_tokens[from_user] = ctx_token
                    
                    if content and from_user:
                        print(f"\n[WeChat] Received from {from_user}: {content}")
                        # Process in Justin
                        try:
                            started_at = time.perf_counter()
                            result = runtime.send_message(content=content, session_id=from_user)
                            elapsed = time.perf_counter() - started_at
                            print(f"[WeChat] Replied to {from_user} in {elapsed:.1f}s")
                            
                            # Send reply back to WeChat
                            reply_text = result.assistant_message.content
                            prefix = _config.wechat_auto_reply_prefix if _config and _config.wechat_auto_reply_prefix else ""
                            if prefix and not reply_text.startswith(prefix):
                                reply_text = f"{prefix}{reply_text}"
                                
                            await _send_message(
                                session,
                                base_url,
                                token,
                                from_user,
                                reply_text,
                                context_token=context_tokens.get(from_user, "")
                            )
                        except Exception as exc:
                            print(f"[WeChat] Agent error: {exc}")

            except asyncio.TimeoutError:
                continue # Normal long-polling timeout
            except Exception as exc:
                print(f"[WeChat] Poll error: {exc}")
                await asyncio.sleep(5)


def start_wechat_bot(runtime: JustinRuntime, config: AgentConfig) -> None:
    """Entry point for the Justin CLI `wechat` command."""
    global _runtime, _config
    _runtime = runtime
    _config = config
    
    try:
        credentials = asyncio.run(qr_login())
        if not credentials:
            sys.exit(1)
            
        asyncio.run(_poll_loop(
            account_id=credentials["account_id"],
            token=credentials["token"],
            base_url=credentials["base_url"],
            runtime=runtime,
        ))
    except KeyboardInterrupt:
        print("\nWeChat Bot stopped.")
