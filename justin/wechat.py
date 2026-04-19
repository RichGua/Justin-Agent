from __future__ import annotations

import asyncio
import json
import random
import ssl
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import qrcode

from .config import (
    AgentConfig,
    WECHAT_ACCESS_ALLOWLIST,
    WECHAT_ACCESS_DISABLED,
)
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


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _account_credentials_path(config: AgentConfig, account_id: str) -> Path:
    return config.wechat_accounts_dir() / f"{account_id}.json"


def _account_context_tokens_path(config: AgentConfig, account_id: str) -> Path:
    return config.wechat_accounts_dir() / f"{account_id}.context.json"


def _active_account_path(config: AgentConfig) -> Path:
    return config.wechat_active_account_path()


def _list_paired_accounts(config: AgentConfig) -> list[dict[str, Any]]:
    accounts: list[dict[str, Any]] = []
    for path in sorted(config.wechat_accounts_dir().glob("*.json")):
        if path.name.endswith(".context.json"):
            continue
        payload = _safe_read_json(path)
        account_id = str(payload.get("account_id") or path.stem)
        token = str(payload.get("token") or "")
        if account_id and token:
            payload["account_id"] = account_id
            accounts.append(payload)
    return accounts


def _set_active_account(config: AgentConfig, account_id: str) -> None:
    _safe_write_json(_active_account_path(config), {"account_id": account_id})


def _get_active_account_id(config: AgentConfig) -> str | None:
    active = _safe_read_json(_active_account_path(config))
    account_id = str(active.get("account_id") or "").strip()
    return account_id or None


def load_saved_credentials(config: AgentConfig, account_id: str | None = None) -> dict[str, Any] | None:
    config.ensure_directories()
    picked_account_id = account_id or _get_active_account_id(config)
    if picked_account_id:
        payload = _safe_read_json(_account_credentials_path(config, picked_account_id))
        if payload.get("token"):
            payload["account_id"] = picked_account_id
            return payload

    accounts = _list_paired_accounts(config)
    if not accounts:
        return None
    picked = accounts[0]
    _set_active_account(config, str(picked["account_id"]))
    return picked


def save_gateway_credentials(config: AgentConfig, credentials: dict[str, Any]) -> dict[str, Any]:
    account_id = str(credentials.get("account_id") or "").strip()
    if not account_id:
        raise ValueError("Missing account_id for WeChat credentials.")

    payload = {
        "account_id": account_id,
        "token": str(credentials.get("token") or ""),
        "base_url": str(credentials.get("base_url") or ILINK_BASE_URL),
        "paired_at": credentials.get("paired_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _safe_write_json(_account_credentials_path(config, account_id), payload)
    _set_active_account(config, account_id)
    return payload


def _load_context_tokens(config: AgentConfig, account_id: str) -> dict[str, str]:
    payload = _safe_read_json(_account_context_tokens_path(config, account_id))
    raw_tokens = payload.get("context_tokens", {})
    if not isinstance(raw_tokens, dict):
        return {}
    return {
        str(user_id): str(token)
        for user_id, token in raw_tokens.items()
        if str(user_id).strip() and str(token).strip()
    }


def _save_context_tokens(config: AgentConfig, account_id: str, context_tokens: dict[str, str]) -> None:
    _safe_write_json(
        _account_context_tokens_path(config, account_id),
        {
            "account_id": account_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "context_tokens": context_tokens,
        },
    )


def is_wechat_sender_allowed(config: AgentConfig, user_id: str) -> bool:
    normalized = str(user_id).strip()
    if not normalized or not config.wechat_enabled:
        return False
    if config.wechat_access_policy == WECHAT_ACCESS_DISABLED:
        return False
    if config.wechat_access_policy != WECHAT_ACCESS_ALLOWLIST:
        return True

    allowed = set(config.parse_wechat_allowed_users())
    if config.wechat_admin_user:
        allowed.add(config.wechat_admin_user.strip())
    return normalized in allowed


def describe_gateway_status(config: AgentConfig) -> dict[str, Any]:
    config.ensure_directories()
    accounts = _list_paired_accounts(config)
    active_account_id = _get_active_account_id(config)
    if not active_account_id and accounts:
        active_account_id = str(accounts[0]["account_id"])
    return {
        "enabled": config.wechat_enabled,
        "app_id": config.wechat_app_id,
        "auto_reply_prefix": config.wechat_auto_reply_prefix,
        "access_policy": config.wechat_access_policy,
        "admin_user": config.wechat_admin_user,
        "allowed_users": config.parse_wechat_allowed_users(),
        "paired_account_id": active_account_id,
        "saved_accounts_count": len(accounts),
        "has_saved_credentials": bool(accounts),
        "active_account_path": str(_active_account_path(config)) if _active_account_path(config).exists() else None,
    }


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
            print(f"[WeChat] Failed to fetch QR code: {exc}")
            return None

        qrcode_value = str(qr_resp.get("qrcode") or "")
        qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
        if not qrcode_value:
            print("[WeChat] QR response missing qrcode token")
            return None

        print("\nPlease scan the following QR code with WeChat:")
        if qrcode_url:
            print(qrcode_url)

        try:
            qr = qrcode.QRCode()
            qr.add_data(qrcode_value)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except Exception as qr_exc:
            print(f"(Terminal QR render failed: {qr_exc}, please open the URL above)")

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
                    if qrcode_url:
                        print(qrcode_url)
                    try:
                        qr = qrcode.QRCode()
                        qr.add_data(qrcode_value)
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
                    "paired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            await asyncio.sleep(1)

        print("\nWeChat login timed out.")
        return None


def pair_wechat_gateway(config: AgentConfig) -> dict[str, Any] | None:
    global _config

    _config = config
    credentials = asyncio.run(qr_login())
    if not credentials:
        return None
    return save_gateway_credentials(config, credentials)


async def _send_message(
    session: aiohttp.ClientSession,
    base_url: str,
    token: str,
    to_user: str,
    content: str,
    context_token: str = "",
) -> None:
    payload = {
        "bot_id": "bot_id",
        "msg_type": 1,
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
) -> str:
    sync_buf = ""
    context_tokens = _load_context_tokens(runtime.config, account_id)

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
                    print("\n[WeChat] Session expired. Please pair again.")
                    return "expired"
                if errcode != 0:
                    print(f"[WeChat] Poll errcode {errcode}: {resp.get('errmsg')}")
                    await asyncio.sleep(5)
                    continue

                sync_buf = str(resp.get("get_updates_buf") or sync_buf)
                msgs = resp.get("msg_list") or []

                for msg in msgs:
                    if msg.get("msg_type") != 1:
                        continue

                    content = str(msg.get("content") or "")
                    from_user = str(msg.get("from_user_id") or "")
                    ctx_token = str(msg.get("context_token") or "")
                    if from_user and ctx_token:
                        existing = context_tokens.get(from_user)
                        if existing != ctx_token:
                            context_tokens[from_user] = ctx_token
                            _save_context_tokens(runtime.config, account_id, context_tokens)

                    if not content or not from_user:
                        continue
                    if not is_wechat_sender_allowed(runtime.config, from_user):
                        print(f"\n[WeChat] Ignored message from {from_user} due to access policy.")
                        continue

                    print(f"\n[WeChat] Received from {from_user}: {content}")
                    try:
                        started_at = time.perf_counter()
                        result = runtime.send_message(content=content, session_id=from_user)
                        elapsed = time.perf_counter() - started_at
                        print(f"[WeChat] Replied to {from_user} in {elapsed:.1f}s")

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
                            context_token=context_tokens.get(from_user, ""),
                        )
                    except Exception as exc:
                        print(f"[WeChat] Agent error: {exc}")

            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                print(f"[WeChat] Poll error: {exc}")
                await asyncio.sleep(5)


def start_wechat_bot(
    runtime: JustinRuntime,
    config: AgentConfig,
    pair_if_needed: bool = False,
) -> None:
    global _runtime, _config

    _runtime = runtime
    _config = config
    config.ensure_directories()

    if not config.wechat_enabled:
        print("WeChat gateway is disabled. Run `justin gateway setup` first.")
        sys.exit(1)

    credentials = load_saved_credentials(config)
    if not credentials:
        if not pair_if_needed:
            print("No paired WeChat account found. Run `justin gateway setup` first.")
            sys.exit(1)
        credentials = pair_wechat_gateway(config)
        if not credentials:
            sys.exit(1)

    try:
        while True:
            outcome = asyncio.run(
                _poll_loop(
                    account_id=str(credentials["account_id"]),
                    token=str(credentials["token"]),
                    base_url=str(credentials.get("base_url") or ILINK_BASE_URL),
                    runtime=runtime,
                )
            )
            if outcome != "expired":
                return
            if not pair_if_needed:
                print("Saved WeChat session expired. Run `justin gateway setup` to re-pair.")
                sys.exit(1)
            print("Saved WeChat session expired. Re-pairing via QR login...")
            credentials = pair_wechat_gateway(config)
            if not credentials:
                sys.exit(1)
    except KeyboardInterrupt:
        print("\nWeChat Bot stopped.")
