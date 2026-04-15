from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from shutil import rmtree

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
            content="喜欢结构化输出",
            evidence="我喜欢结构化输出",
            confidence=0.9,
            source_session_id=session.id,
        )
        memory = self.store.confirm_candidate(candidate.id)
        self.assertEqual(memory.kind, "preference")

        results = self.store.search_memories("结构化 输出")
        self.assertTrue(results)
        self.assertEqual(results[0].id, memory.id)


if __name__ == "__main__":
    unittest.main()
