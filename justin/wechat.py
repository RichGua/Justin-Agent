from __future__ import annotations

import asyncio
import base64
import json
import secrets
import ssl
import struct
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import qrcode

from .config import (
    AgentConfig,
    WECHAT_DEFAULT_BASE_URL,
    WECHAT_POLICY_ALLOWLIST,
    WECHAT_POLICY_DISABLED,
    WECHAT_POLICY_OPEN,
)
from .runtime import JustinRuntime

ILINK_BASE_URL = WECHAT_DEFAULT_BASE_URL
ILINK_APP_ID = "bot"
ILINK_APP_CLIENT_VERSION = (2 << 16) | (2 << 8) | 0
CHANNEL_VERSION = "2.2.0"

EP_GET_BOT_QR = "ilink/bot/get_bot_qrcode"
EP_GET_QR_STATUS = "ilink/bot/get_qrcode_status"
EP_GET_UPDATES = "ilink/bot/getupdates"
EP_SEND_MESSAGE = "ilink/bot/sendmessage"

ITEM_TEXT = 1
MSG_TYPE_USER = 1
MSG_TYPE_BOT = 2
MSG_STATE_FINISH = 2

QR_TIMEOUT_MS = 15000
POLL_TIMEOUT_MS = 35000
API_TIMEOUT_MS = 10000

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


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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


def _render_terminal_qr(scan_data: str) -> None:
    qr = qrcode.QRCode()
    qr.add_data(scan_data)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def _random_wechat_uin() -> str:
    value = struct.unpack(">I", secrets.token_bytes(4))[0]
    return base64.b64encode(str(value).encode("utf-8")).decode("ascii")


def _base_info() -> dict[str, Any]:
    return {"channel_version": CHANNEL_VERSION}


def _build_ilink_headers(
    *,
    token: str = "",
    include_json_headers: bool = False,
    content_length: int | None = None,
) -> dict[str, str]:
    headers = {
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    if include_json_headers:
        headers["Content-Type"] = "application/json"
        headers["AuthorizationType"] = "ilink_bot_token"
        headers["X-WECHAT-UIN"] = _random_wechat_uin()
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["ilink_bot_token"] = token
        if include_json_headers:
            headers["AuthorizationType"] = "ilink_bot_token"
    return headers


def _account_credentials_path(config: AgentConfig, account_id: str) -> Path:
    return config.wechat_accounts_dir() / f"{account_id}.json"


def _account_context_tokens_path(config: AgentConfig, account_id: str) -> Path:
    return config.wechat_accounts_dir() / f"{account_id}.context-tokens.json"


def _account_sync_buf_path(config: AgentConfig, account_id: str) -> Path:
    return config.wechat_accounts_dir() / f"{account_id}.sync.json"


def _active_account_path(config: AgentConfig) -> Path:
    return config.wechat_active_account_path()


def _list_paired_accounts(config: AgentConfig) -> list[dict[str, Any]]:
    accounts: list[dict[str, Any]] = []
    for path in sorted(config.wechat_accounts_dir().glob("*.json")):
        if path.name.endswith(".context-tokens.json") or path.name.endswith(".sync.json"):
            continue
        payload = _safe_read_json(path)
        account_id = str(payload.get("account_id") or path.stem).strip()
        token = str(payload.get("token") or "").strip()
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


def _load_context_tokens(config: AgentConfig, account_id: str) -> dict[str, str]:
    payload = _safe_read_json(_account_context_tokens_path(config, account_id))
    raw_tokens = payload.get("context_tokens", {})
    if not isinstance(raw_tokens, dict):
        return {}
    return {
        str(chat_id): str(token)
        for chat_id, token in raw_tokens.items()
        if str(chat_id).strip() and str(token).strip()
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


def _load_sync_buf(config: AgentConfig, account_id: str) -> str:
    payload = _safe_read_json(_account_sync_buf_path(config, account_id))
    return str(payload.get("get_updates_buf") or "")


def _save_sync_buf(config: AgentConfig, account_id: str, sync_buf: str) -> None:
    _safe_write_json(
        _account_sync_buf_path(config, account_id),
        {
            "account_id": account_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "get_updates_buf": sync_buf,
        },
    )


def load_saved_credentials(config: AgentConfig, account_id: str | None = None) -> dict[str, Any] | None:
    config.ensure_directories()
    requested_account_id = (account_id or config.wechat_account_id or _get_active_account_id(config) or "").strip()

    if requested_account_id:
        payload = _safe_read_json(_account_credentials_path(config, requested_account_id))
        if payload.get("token"):
            payload["account_id"] = requested_account_id
            return payload
        if config.wechat_account_id == requested_account_id and config.wechat_token:
            return {
                "account_id": requested_account_id,
                "token": config.wechat_token,
                "base_url": config.wechat_base_url or WECHAT_DEFAULT_BASE_URL,
            }

    if config.wechat_account_id and config.wechat_token:
        return {
            "account_id": config.wechat_account_id,
            "token": config.wechat_token,
            "base_url": config.wechat_base_url or WECHAT_DEFAULT_BASE_URL,
        }

    accounts = _list_paired_accounts(config)
    if not accounts:
        return None
    picked = accounts[0]
    _set_active_account(config, str(picked["account_id"]))
    return picked


def save_gateway_credentials(config: AgentConfig, credentials: dict[str, Any]) -> dict[str, Any]:
    account_id = str(credentials.get("account_id") or "").strip()
    token = str(credentials.get("token") or "").strip()
    if not account_id or not token:
        raise ValueError("Missing account_id or token for WeChat credentials.")

    payload = {
        "account_id": account_id,
        "token": token,
        "base_url": str(credentials.get("base_url") or WECHAT_DEFAULT_BASE_URL),
        "paired_at": credentials.get("paired_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _safe_write_json(_account_credentials_path(config, account_id), payload)
    _set_active_account(config, account_id)
    config.wechat_account_id = account_id
    config.wechat_token = token
    config.wechat_base_url = payload["base_url"]
    return payload


def _guess_chat_type(message: dict[str, Any], account_id: str) -> tuple[str, str]:
    room_id = str(message.get("room_id") or message.get("chat_room_id") or "").strip()
    to_user_id = str(message.get("to_user_id") or "").strip()
    is_group = bool(room_id) or (
        to_user_id and account_id and to_user_id != account_id and message.get("msg_type") == MSG_TYPE_USER
    )
    if is_group:
        return "group", room_id or to_user_id or str(message.get("from_user_id") or "")
    return "dm", str(message.get("from_user_id") or "")


def _extract_text_content(message: dict[str, Any]) -> str:
    item_list = message.get("item_list") or []
    if isinstance(item_list, list):
        parts: list[str] = []
        for item in item_list:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type") or item.get("message_type")
            if item_type != ITEM_TEXT:
                continue
            text_block = item.get("text_item") or item.get("text") or {}
            if isinstance(text_block, dict):
                text = str(text_block.get("text") or text_block.get("content") or "").strip()
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(message.get("content") or "").strip()


def is_wechat_sender_allowed(
    config: AgentConfig,
    user_id: str,
    *,
    chat_type: str = "dm",
    group_id: str | None = None,
) -> bool:
    if not config.wechat_enabled:
        return False
    if chat_type == "group":
        if config.wechat_group_policy == WECHAT_POLICY_DISABLED:
            return False
        if config.wechat_group_policy == WECHAT_POLICY_OPEN:
            return True
        allowed_groups = set(config.parse_wechat_group_allowed_users())
        return bool(group_id and (group_id in allowed_groups or "*" in allowed_groups))

    normalized = str(user_id).strip()
    if not normalized:
        return False
    if config.wechat_dm_policy == WECHAT_POLICY_DISABLED:
        return False
    if config.wechat_dm_policy == WECHAT_POLICY_OPEN:
        return True
    allowed_users = set(config.parse_wechat_allowed_users())
    return normalized in allowed_users or "*" in allowed_users


def describe_gateway_status(config: AgentConfig) -> dict[str, Any]:
    config.ensure_directories()
    accounts = _list_paired_accounts(config)
    active_account_id = _get_active_account_id(config) or config.wechat_account_id
    if not active_account_id and accounts:
        active_account_id = str(accounts[0]["account_id"])
    return {
        "enabled": config.wechat_enabled,
        "account_id": config.wechat_account_id,
        "base_url": config.wechat_base_url or WECHAT_DEFAULT_BASE_URL,
        "auto_reply_prefix": config.wechat_auto_reply_prefix,
        "dm_policy": config.wechat_dm_policy,
        "group_policy": config.wechat_group_policy,
        "allowed_users": config.parse_wechat_allowed_users(),
        "allowed_groups": config.parse_wechat_group_allowed_users(),
        "paired_account_id": active_account_id,
        "saved_accounts_count": len(accounts),
        "has_saved_credentials": bool(accounts or (config.wechat_account_id and config.wechat_token)),
        "active_account_path": str(_active_account_path(config)) if _active_account_path(config).exists() else None,
    }


async def _api_get(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    timeout_ms: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000.0)
    async with session.get(url, headers=headers, timeout=timeout) as response:
        raw = await response.text()
        if not response.ok:
            raise RuntimeError(f"iLink GET {endpoint} HTTP {response.status}: {raw[:200]}")
        return json.loads(raw)


async def _api_post(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    token: str,
    timeout_ms: int,
) -> dict[str, Any]:
    body = _json_dumps({**payload, "base_info": _base_info()})
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = _build_ilink_headers(
        token=token,
        include_json_headers=True,
        content_length=len(body.encode("utf-8")),
    )
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000.0)
    async with session.post(url, data=body, headers=headers, timeout=timeout) as response:
        raw = await response.text()
        if not response.ok:
            raise RuntimeError(f"iLink POST {endpoint} HTTP {response.status}: {raw[:200]}")
        return json.loads(raw)


async def _get_updates(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    token: str,
    sync_buf: str,
    timeout_ms: int,
) -> dict[str, Any]:
    try:
        return await _api_post(
            session,
            base_url=base_url,
            endpoint=EP_GET_UPDATES,
            payload={"get_updates_buf": sync_buf},
            token=token,
            timeout_ms=timeout_ms,
        )
    except asyncio.TimeoutError:
        return {"ret": 0, "msgs": [], "get_updates_buf": sync_buf}


async def qr_login() -> dict[str, str] | None:
    print("\nStarting iLink Bot API QR Login...")
    async with aiohttp.ClientSession(connector=_make_ssl_connector(), trust_env=True) as session:
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

        qrcode_value = str(qr_resp.get("qrcode") or "").strip()
        qrcode_url = str(qr_resp.get("qrcode_img_content") or "").strip()
        if not qrcode_value and not qrcode_url:
            print("[WeChat] QR response missing qrcode payload")
            return None

        qr_scan_data = qrcode_url or qrcode_value
        print("\nPlease scan the following QR code with WeChat:")
        if qrcode_url:
            print(qrcode_url)
        try:
            _render_terminal_qr(qr_scan_data)
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
                redirect_host = str(status_resp.get("redirect_host") or "").strip()
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
                    qrcode_value = str(qr_resp.get("qrcode") or "").strip()
                    qrcode_url = str(qr_resp.get("qrcode_img_content") or "").strip()
                    qr_scan_data = qrcode_url or qrcode_value
                    if qrcode_url:
                        print(qrcode_url)
                    _render_terminal_qr(qr_scan_data)
                except Exception as exc:
                    print(f"[WeChat] QR refresh failed: {exc}")
                    return None
            elif status == "confirmed":
                account_id = str(status_resp.get("ilink_bot_id") or "").strip()
                token = str(status_resp.get("bot_token") or "").strip()
                base_url = str(status_resp.get("baseurl") or ILINK_BASE_URL).strip()
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
    *,
    base_url: str,
    token: str,
    to: str,
    text: str,
    context_token: str | None,
    client_id: str,
) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("_send_message: text must not be empty")
    message: dict[str, Any] = {
        "from_user_id": "",
        "to_user_id": to,
        "client_id": client_id,
        "message_type": MSG_TYPE_BOT,
        "message_state": MSG_STATE_FINISH,
        "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}],
    }
    if context_token:
        message["context_token"] = context_token
    return await _api_post(
        session,
        base_url=base_url,
        endpoint=EP_SEND_MESSAGE,
        payload={"msg": message},
        token=token,
        timeout_ms=API_TIMEOUT_MS,
    )


async def _poll_loop(
    account_id: str,
    token: str,
    base_url: str,
    runtime: JustinRuntime,
) -> str:
    sync_buf = _load_sync_buf(runtime.config, account_id)
    context_tokens = _load_context_tokens(runtime.config, account_id)

    print("\nListening for WeChat messages via iLink API...")
    async with aiohttp.ClientSession(connector=_make_ssl_connector(), trust_env=True) as session:
        while True:
            try:
                resp = await _get_updates(
                    session,
                    base_url=base_url,
                    token=token,
                    sync_buf=sync_buf,
                    timeout_ms=POLL_TIMEOUT_MS,
                )
            except Exception as exc:
                print(f"[WeChat] Poll error: {exc}")
                await asyncio.sleep(5)
                continue

            ret = int(resp.get("ret", resp.get("errcode", 0)) or 0)
            if ret == -14:
                print("\n[WeChat] Session expired. Please pair again.")
                return "expired"
            if ret != 0:
                print(f"[WeChat] Poll errcode {ret}: {resp.get('msg') or resp.get('errmsg')}")
                await asyncio.sleep(5)
                continue

            sync_buf = str(resp.get("get_updates_buf") or sync_buf)
            _save_sync_buf(runtime.config, account_id, sync_buf)
            msgs = resp.get("msgs") or resp.get("msg_list") or []

            for message in msgs:
                if not isinstance(message, dict):
                    continue
                chat_type, chat_id = _guess_chat_type(message, account_id)
                from_user = str(message.get("from_user_id") or "").strip()
                context_token = str(message.get("context_token") or "").strip()
                if context_token:
                    if context_tokens.get(chat_id) != context_token:
                        context_tokens[chat_id] = context_token
                        _save_context_tokens(runtime.config, account_id, context_tokens)

                if not is_wechat_sender_allowed(
                    runtime.config,
                    from_user,
                    chat_type=chat_type,
                    group_id=chat_id if chat_type == "group" else None,
                ):
                    print(f"\n[WeChat] Ignored {chat_type} message from {from_user or chat_id} due to policy.")
                    continue

                content = _extract_text_content(message)
                if not content:
                    continue

                print(f"\n[WeChat] Received {chat_type} message from {from_user or chat_id}: {content}")
                try:
                    started_at = time.perf_counter()
                    session_id = f"wechat:{chat_type}:{chat_id}"
                    result = runtime.send_message(content=content, session_id=session_id)
                    elapsed = time.perf_counter() - started_at
                    print(f"[WeChat] Replied to {chat_id} in {elapsed:.1f}s")

                    reply_text = result.assistant_message.content
                    prefix = _config.wechat_auto_reply_prefix if _config and _config.wechat_auto_reply_prefix else ""
                    if prefix and not reply_text.startswith(prefix):
                        reply_text = f"{prefix}{reply_text}"

                    await _send_message(
                        session,
                        base_url=base_url,
                        token=token,
                        to=chat_id,
                        text=reply_text,
                        context_token=context_tokens.get(chat_id),
                        client_id=f"justin-{int(time.time() * 1000)}",
                    )
                except Exception as exc:
                    print(f"[WeChat] Agent error: {exc}")


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
            print("No WeChat credentials found. Run `justin gateway setup` first.")
            sys.exit(1)
        credentials = pair_wechat_gateway(config)
        if not credentials:
            sys.exit(1)
        config.save_settings()

    try:
        while True:
            outcome = asyncio.run(
                _poll_loop(
                    account_id=str(credentials["account_id"]),
                    token=str(credentials["token"]),
                    base_url=str(credentials.get("base_url") or WECHAT_DEFAULT_BASE_URL),
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
            config.save_settings()
    except KeyboardInterrupt:
        print("\nWeChat Bot stopped.")
