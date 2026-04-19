from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from shutil import rmtree
from unittest.mock import patch

from justin.embeddings import LocalHashEmbeddingProvider
from justin.storage import AgentStore


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path.cwd() / ".tmp_tests"
        root.mkdir(exist_ok=True)
        self.temp_dir = root / f"storage_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir()
        self.store = AgentStore(
            database_path=self.temp_dir / "agent.db",
            embedder=LocalHashEmbeddingProvider(),
        )

    def tearDown(self) -> None:
        self.store.close()
        rmtree(self.temp_dir, ignore_errors=True)

    def test_confirmed_memory_is_searchable(self) -> None:
        session = self.store.create_session("test")
        candidate = self.store.create_candidate(
            kind="preference",
            content="likes structured output",
            evidence="I like structured output",
            confidence=0.9,
            source_session_id=session.id,
        )
        memory = self.store.confirm_candidate(candidate.id)
        self.assertEqual(memory.kind, "preference")

        results = self.store.search_memories("structured output")
        self.assertTrue(results)
        self.assertEqual(results[0].id, memory.id)

    def test_confirm_candidate_is_idempotent(self) -> None:
        session = self.store.create_session("test")
        candidate = self.store.create_candidate(
            kind="fact",
            content="prefers concise answers",
            evidence="User prefers concise answers",
            confidence=0.8,
            source_session_id=session.id,
        )

        first_memory = self.store.confirm_candidate(candidate.id)
        second_memory = self.store.confirm_candidate(candidate.id)

        self.assertEqual(first_memory.id, second_memory.id)
        self.assertEqual(len(self.store.list_memories()), 1)

    def test_message_order_is_stable_when_timestamps_match(self) -> None:
        session = self.store.create_session("test")

        with patch("justin.storage.now_iso", return_value="2026-04-19T00:00:00Z"):
            self.store.add_message(session.id, "user", "first")
            self.store.add_message(session.id, "assistant", "second")
            self.store.add_message(session.id, "user", "third")

        self.assertEqual(
            [message.content for message in self.store.list_messages(session.id, limit=None)],
            ["first", "second", "third"],
        )

        self.store.delete_old_messages(session.id, keep_latest=1)
        self.assertEqual(
            [message.content for message in self.store.list_messages(session.id, limit=None)],
            ["third"],
        )


if __name__ == "__main__":
    unittest.main()
