from __future__ import annotations

import asyncio
import unittest
import uuid
from pathlib import Path
from shutil import rmtree
from unittest.mock import patch

from justin.config import (
    AgentConfig,
    WECHAT_POLICY_ALLOWLIST,
    WECHAT_POLICY_DISABLED,
    WECHAT_POLICY_OPEN,
)
from justin.wechat import (
    _build_ilink_headers,
    describe_gateway_status,
    is_wechat_sender_allowed,
    load_saved_credentials,
    qr_login,
    save_gateway_credentials,
)


class WeChatGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path.cwd() / ".tmp_tests"
        root.mkdir(exist_ok=True)
        self.temp_dir = root / f"wechat_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir()
        self.config = AgentConfig(
            home_dir=self.temp_dir,
            database_path=self.temp_dir / "agent.db",
            settings_path=self.temp_dir / "settings.json",
            wechat_enabled=True,
            wechat_dm_policy=WECHAT_POLICY_OPEN,
        )
        self.config.ensure_directories()

    def tearDown(self) -> None:
        rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_gateway_credentials(self) -> None:
        save_gateway_credentials(
            self.config,
            {
                "account_id": "wx_bot_1",
                "token": "secret-token",
                "base_url": "https://example.com",
                "paired_at": "2026-04-19T00:00:00Z",
            },
        )

        loaded = load_saved_credentials(self.config)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["account_id"], "wx_bot_1")
        self.assertEqual(loaded["token"], "secret-token")
        self.assertEqual(loaded["base_url"], "https://example.com")

    def test_describe_gateway_status_reports_paired_account(self) -> None:
        save_gateway_credentials(
            self.config,
            {
                "account_id": "wx_bot_2",
                "token": "another-token",
                "base_url": "https://example.com",
            },
        )

        status = describe_gateway_status(self.config)

        self.assertTrue(status["has_saved_credentials"])
        self.assertEqual(status["paired_account_id"], "wx_bot_2")
        self.assertEqual(status["saved_accounts_count"], 1)

    def test_ilink_headers_use_hermes_bot_app_id(self) -> None:
        headers = _build_ilink_headers(token="secret", include_json_headers=True, content_length=3)

        self.assertEqual(headers["iLink-App-Id"], "bot")
        self.assertEqual(headers["AuthorizationType"], "ilink_bot_token")
        self.assertIn("X-WECHAT-UIN", headers)

    def test_sender_policy_open_allowlist_and_disabled(self) -> None:
        self.assertTrue(is_wechat_sender_allowed(self.config, "alice"))

        self.config.wechat_dm_policy = WECHAT_POLICY_ALLOWLIST
        self.config.wechat_allowed_users = "alice,bob"
        self.assertTrue(is_wechat_sender_allowed(self.config, "alice"))
        self.assertFalse(is_wechat_sender_allowed(self.config, "mallory"))

        self.config.wechat_dm_policy = WECHAT_POLICY_DISABLED
        self.assertFalse(is_wechat_sender_allowed(self.config, "alice"))

    def test_group_policy_respects_allowlist(self) -> None:
        self.config.wechat_group_policy = WECHAT_POLICY_ALLOWLIST
        self.config.wechat_group_allowed_users = "room-1,room-2"

        self.assertTrue(is_wechat_sender_allowed(self.config, "alice", chat_type="group", group_id="room-1"))
        self.assertFalse(is_wechat_sender_allowed(self.config, "alice", chat_type="group", group_id="room-9"))

    def test_qr_login_renders_url_payload_when_available(self) -> None:
        scan_payloads: list[str] = []

        class FakeQRCode:
            def add_data(self, data: str) -> None:
                scan_payloads.append(data)

            def make(self, fit: bool = True) -> None:
                return None

            def print_ascii(self, invert: bool = True) -> None:
                return None

        class FakeSession:
            async def __aenter__(self):
                return object()

            async def __aexit__(self, exc_type, exc, tb):
                return False

        responses = [
            {
                "qrcode": "short-token",
                "qrcode_img_content": "https://liteapp.weixin.qq.com/q/demo",
            },
            {
                "status": "confirmed",
                "ilink_bot_id": "wx_bot_3",
                "bot_token": "token-3",
                "baseurl": "https://wechat.example.com",
            },
        ]

        async def fake_api_get(*args, **kwargs):
            return responses.pop(0)

        with (
            patch("justin.wechat._api_get", side_effect=fake_api_get),
            patch("justin.wechat.qrcode.QRCode", return_value=FakeQRCode()),
            patch("justin.wechat.aiohttp.ClientSession", return_value=FakeSession()),
            patch("justin.wechat.asyncio.sleep"),
        ):
            credentials = asyncio.run(qr_login())

        self.assertIsNotNone(credentials)
        self.assertIn("https://liteapp.weixin.qq.com/q/demo", scan_payloads)


if __name__ == "__main__":
    unittest.main()
