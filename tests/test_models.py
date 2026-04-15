from __future__ import annotations

import unittest

from justin.models import OpenAICompatibleChatProvider


class ModelParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = OpenAICompatibleChatProvider(
            model_name="test-model",
            api_base="https://example.com/v1",
            api_key="your_api_key_here",
        )

    def test_extracts_standard_message_content(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello world",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        text = self.provider._extract_response_text(payload)
        self.assertEqual(text, "hello world")

    def test_extracts_reasoning_content_when_content_missing_and_truncated(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "partial reasoning text",
                    },
                    "finish_reason": "length",
                }
            ]
        }
        text = self.provider._extract_response_text(payload)
        self.assertIn("finish_reason=length", text)
        self.assertIn("partial reasoning text", text)


if __name__ == "__main__":
    unittest.main()
