from __future__ import annotations

import sys
import time
from typing import Any

from .config import AgentConfig
from .runtime import JustinRuntime

try:
    import itchat
    from itchat.content import TEXT
    ITCHAT_AVAILABLE = True
except ImportError:
    ITCHAT_AVAILABLE = False


_runtime: JustinRuntime | None = None
_config: AgentConfig | None = None


def _get_reply(msg: Any) -> str:
    if not _runtime:
        return "Agent is not initialized."
    
    user_id = msg.user.userName
    content = msg.text
    
    print(f"\n[WeChat] Received from {user_id}: {content}")
    try:
        started_at = time.perf_counter()
        result = _runtime.send_message(content=content, session_id=user_id)
        elapsed = time.perf_counter() - started_at
        
        print(f"[WeChat] Replied to {user_id} in {elapsed:.1f}s")
        return result.assistant_message.content
    except Exception as exc:
        print(f"[WeChat] Error processing message: {exc}")
        return f"Sorry, I encountered an error: {exc}"


def start_wechat_bot(runtime: JustinRuntime, config: AgentConfig) -> None:
    if not ITCHAT_AVAILABLE:
        print("itchat-uos is not installed. Please run `uv add itchat-uos`")
        sys.exit(1)
        
    global _runtime, _config
    _runtime = runtime
    _config = config
    
    @itchat.msg_register(TEXT, isFriendChat=True, isGroupChat=False)
    def text_reply(msg: Any) -> str:
        return _get_reply(msg)

    print("Starting WeChat Bot (itchat-uos)...")
    print("Please scan the QR code to log in.")
    
    try:
        itchat.auto_login(enableCmdQR=2, hotReload=True)
        print("\nWeChat login successful! Listening for messages...")
        itchat.run()
    except KeyboardInterrupt:
        print("\nWeChat Bot stopped.")
    except Exception as exc:
        print(f"\nWeChat login failed: {exc}")
