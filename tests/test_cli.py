from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from justin.cli import main


class CLITests(unittest.TestCase):
    def test_main_without_args_defaults_to_interactive_chat(self) -> None:
        runtime = MagicMock()

        with (
            patch("justin.cli.AgentConfig.from_env", return_value=object()),
            patch("justin.cli.build_runtime_bundle", return_value=object()),
            patch("justin.cli.JustinRuntime", return_value=runtime),
            patch("justin.cli._run_chat") as run_chat,
        ):
            main([])

        run_chat.assert_called_once_with(runtime, None, None)
        runtime.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
