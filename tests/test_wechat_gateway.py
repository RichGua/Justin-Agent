from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from shutil import rmtree

from justin.config import (
    AgentConfig,
    WECHAT_ACCESS_ALLOWLIST,
    WECHAT_ACCESS_DISABLED,
    WECHAT_ACCESS_OPEN,
)
from justin.wechat import (
    describe_gateway_status,
    is_wechat_sender_allowed,
    load_saved_credentials,
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
            wechat_access_policy=WECHAT_ACCESS_OPEN,
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

    def test_sender_policy_open_allowlist_and_disabled(self) -> None:
        self.assertTrue(is_wechat_sender_allowed(self.config, "alice"))

        self.config.wechat_access_policy = WECHAT_ACCESS_ALLOWLIST
        self.config.wechat_admin_user = "wx_admin"
        self.config.wechat_allowed_users = "alice,bob"
        self.assertTrue(is_wechat_sender_allowed(self.config, "alice"))
        self.assertTrue(is_wechat_sender_allowed(self.config, "wx_admin"))
        self.assertFalse(is_wechat_sender_allowed(self.config, "mallory"))

        self.config.wechat_access_policy = WECHAT_ACCESS_DISABLED
        self.assertFalse(is_wechat_sender_allowed(self.config, "alice"))


if __name__ == "__main__":
    unittest.main()
