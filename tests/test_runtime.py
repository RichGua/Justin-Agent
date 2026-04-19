from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from shutil import rmtree
from unittest.mock import patch

from justin.config import AgentConfig
from justin.runtime import JustinRuntime, build_runtime_bundle
from justin.tools import ToolResult
from justin.types import ChatResponse, ChatToolCall


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
        first_turn = self.runtime.send_message("I like concise output and dark terminals.")
        self.assertEqual(len(first_turn.candidates), 1)

        candidate = first_turn.candidates[0]
        memory = self.runtime.confirm_candidate(candidate.id)
        self.assertEqual(memory.kind, "preference")

        second_turn = self.runtime.send_message(
            "Do you remember that I like concise output and dark terminals?",
            session_id=first_turn.session.id,
        )
        self.assertTrue(second_turn.recalled_memories)
        self.assertIn("likes concise output and dark terminals", second_turn.assistant_message.content)

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

        fake_responses = [
            ChatResponse(
                content="",
                tool_calls=[
                    ChatToolCall(
                        id="call_123",
                        name="search_web",
                        arguments='{"query": "latest python release notes"}',
                    )
                ],
            ),
            ChatResponse(content="Here are the release notes."),
        ]

        with patch.object(self.runtime.tool_registry, "execute", return_value=fake_result):
            with patch.object(self.runtime.chat_provider, "generate", side_effect=fake_responses):
                turn = self.runtime.send_message("latest python release notes")

        self.assertEqual(len(turn.tool_events), 1)
        self.assertEqual(turn.tool_events[0].tool_name, "search_web")
        self.assertEqual(len(turn.citations), 1)
        self.assertEqual(turn.citations[0].url, "https://example.com/python-release")
        self.assertIsNotNone(turn.context_telemetry)

        stored_events = self.runtime.list_tool_events(turn.session.id)
        self.assertEqual(len(stored_events), 1)
        self.assertEqual(stored_events[0].tool_name, "search_web")

    def test_compact_preserves_existing_summary_and_appends_new_archived_messages(self) -> None:
        session = self.runtime.store.create_session("summary-test")
        for index in range(6):
            role = "user" if index % 2 == 0 else "assistant"
            self.runtime.store.add_message(session.id, role, f"msg-{index}")

        builder = self.runtime.context_builder

        with (
            patch.object(
                builder,
                "_summarize_messages",
                side_effect=lambda messages: "|".join(message.content for message in messages),
            ),
            patch.object(
                builder,
                "_extend_summary",
                side_effect=lambda summary, messages: summary
                + "|"
                + "|".join(message.content for message in messages),
            ),
        ):
            summary = builder.compact(session.id)
            self.assertEqual(summary, "msg-0|msg-1|msg-2|msg-3")

            stored = self.runtime.store.get_session_summary(session.id)
            self.assertIsNotNone(stored)
            self.assertEqual(stored.source_message_count, 4)

            for index in range(6, 16):
                role = "user" if index % 2 == 0 else "assistant"
                self.runtime.store.add_message(session.id, role, f"msg-{index}")

            refreshed = builder._refresh_summary(
                session.id,
                self.runtime.store.list_messages(session.id, limit=None),
            )

        self.assertEqual(
            refreshed,
            "msg-0|msg-1|msg-2|msg-3|msg-4|msg-5|msg-6|msg-7|msg-8|msg-9|msg-10|msg-11",
        )


if __name__ == "__main__":
    unittest.main()
