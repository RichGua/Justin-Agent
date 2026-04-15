from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from shutil import rmtree

from justin.config import AgentConfig
from justin.runtime import JustinRuntime, build_runtime_bundle


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


if __name__ == "__main__":
    unittest.main()
