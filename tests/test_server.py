from __future__ import annotations

import unittest
from pathlib import Path

from justin.server import _resolve_static_file


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.web_dir = Path(__file__).resolve().parents[1] / "justin" / "web"

    def test_resolve_static_file_allows_known_asset(self) -> None:
        file_path = _resolve_static_file(self.web_dir, "/app.js")

        self.assertIsNotNone(file_path)
        self.assertEqual(file_path.name, "app.js")

    def test_resolve_static_file_rejects_path_traversal(self) -> None:
        self.assertIsNone(_resolve_static_file(self.web_dir, "/../README.md"))
        self.assertIsNone(_resolve_static_file(self.web_dir, "/..%2FREADME.md"))


if __name__ == "__main__":
    unittest.main()
