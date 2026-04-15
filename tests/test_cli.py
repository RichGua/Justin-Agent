from __future__ import annotations

import io
import unittest
import uuid
from pathlib import Path
from shutil import rmtree
from unittest.mock import MagicMock, patch

from justin.cli import _run_chat, main, run_setup_wizard
from justin.config import AgentConfig, PROVIDER_OPENAI


class CLITests(unittest.TestCase):
    def test_main_without_args_defaults_to_interactive_chat(self) -> None:
        config = object()
        runtime = MagicMock()

        with (
            patch("justin.cli.AgentConfig.from_env", return_value=config),
            patch("justin.cli._maybe_prompt_first_run_setup", return_value=config),
            patch("justin.cli.build_runtime_bundle", return_value=object()),
            patch("justin.cli.JustinRuntime", return_value=runtime),
            patch("justin.cli._run_chat") as run_chat,
        ):
            main([])

        run_chat.assert_called_once_with(runtime, None, None)
        runtime.close.assert_called_once()

    def test_setup_wizard_configures_openai(self) -> None:
        root = Path.cwd() / ".tmp_tests"
        root.mkdir(exist_ok=True)
        temp_dir = root / f"setup_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir()
        config = AgentConfig(
            home_dir=temp_dir,
            database_path=temp_dir / "agent.db",
            settings_path=temp_dir / "settings.json",
        )

        try:
            # openai -> choose default base/model -> provide key
            with patch("builtins.input", side_effect=["1", "", "", "sk-test-123"]):
                updated = run_setup_wizard(config)
        finally:
            rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(updated.model_provider, PROVIDER_OPENAI)
        self.assertEqual(updated.api_base, "https://api.openai.com/v1")
        self.assertEqual(updated.model_name, "gpt-4.1-mini")
        self.assertEqual(updated.api_key, "sk-test-123")

    def test_run_chat_prints_timeout_hint(self) -> None:
        runtime = MagicMock()
        runtime.send_message.side_effect = TimeoutError("The read operation timed out")

        with (
            patch("sys.stderr", new=io.StringIO()) as stderr,
            patch("sys.stdout", new=io.StringIO()),
        ):
            _run_chat(runtime, None, "hello")

        output = stderr.getvalue()
        self.assertIn("Justin is thinking...", output)
        self.assertIn("timed out", output.lower())
        self.assertIn("/new", output)

    def test_run_chat_prints_remote_disconnect_hint(self) -> None:
        runtime = MagicMock()
        runtime.send_message.side_effect = RuntimeError("Remote end closed connection without response")

        with (
            patch("sys.stderr", new=io.StringIO()) as stderr,
            patch("sys.stdout", new=io.StringIO()),
        ):
            _run_chat(runtime, None, "hello")

        output = stderr.getvalue().lower()
        self.assertIn("remote server closed the connection", output)


if __name__ == "__main__":
    unittest.main()
