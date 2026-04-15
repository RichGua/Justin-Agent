from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from shutil import rmtree
from unittest.mock import patch

from justin.config import AgentConfig
from justin.runtime import JustinRuntime, build_runtime_bundle
from justin.tools import ToolResult


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path.cwd() / ".tmp_tests"
        root.mkdir(exist_ok=True)
        self.temp_dir = root / f"runtime_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir()
        config = AgentConfig(
            home_dir=self.temp_dir,
            database_path=self.temp_dir / "agent.db",
        )
        bundle = build_runtime_bundle(config)
        self.runtime = JustinRuntime(bundle)

    def tearDown(self) -> None:
        self.runtime.close()
        rmtree(self.temp_dir, ignore_errors=True)

    def test_candidate_lifecycle_and_recall(self) -> None:
        first_turn = self.runtime.send_message("记住我喜欢简洁输出和深色终端")
        self.assertEqual(len(first_turn.candidates), 1)

        candidate = first_turn.candidates[0]
        memory = self.runtime.confirm_candidate(candidate.id)
        self.assertEqual(memory.kind, "preference")

        second_turn = self.runtime.send_message("你知道我喜欢什么吗？", session_id=first_turn.session.id)
        self.assertTrue(second_turn.recalled_memories)
        self.assertIn("简洁输出和深色终端", second_turn.assistant_message.content)

    def test_search_hint_records_tool_event_and_citations(self) -> None:
        fake_result = ToolResult(
            ok=True,
            output={
                "query": "latest python release",
                "results": [
                    {
                        "title": "Python Release",
                        "url": "https://example.com/python-release",
                        "snippet": "Python 3.x release notes",
                        "source": "duckduckgo",
                        "fetched_at": "2026-04-15T00:00:00Z",
                        "confidence": 0.9,
                    }
                ],
            },
            summary="Found 1 search result for latest python release.",
            source="builtin",
            meta={"latency_ms": 12},
        )

        with patch.object(self.runtime.tool_registry, "execute", return_value=fake_result):
            turn = self.runtime.send_message("latest python release notes")

        self.assertEqual(len(turn.tool_events), 1)
        self.assertEqual(turn.tool_events[0].tool_name, "search_web")
        self.assertEqual(len(turn.citations), 1)
        self.assertEqual(turn.citations[0].url, "https://example.com/python-release")
        self.assertIsNotNone(turn.context_telemetry)

        stored_events = self.runtime.list_tool_events(turn.session.id)
        self.assertEqual(len(stored_events), 1)
        self.assertEqual(stored_events[0].tool_name, "search_web")


if __name__ == "__main__":
    unittest.main()
