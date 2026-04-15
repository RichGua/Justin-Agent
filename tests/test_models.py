from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from justin.models import OpenAICompatibleChatProvider
from justin.types import ChatRequest


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


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

    def test_raises_when_only_reasoning_and_truncated(self) -> None:
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
        with self.assertRaises(RuntimeError) as ctx:
            self.provider._extract_response_text(payload)
        self.assertIn("JUSTIN_MODEL_MAX_TOKENS", str(ctx.exception))

    def test_generate_retries_when_length_without_final_content(self) -> None:
        provider = OpenAICompatibleChatProvider(
            model_name="z-ai/glm4.7",
            api_base="https://integrate.api.nvidia.com/v1",
            api_key="your_api_key_here",
            max_tokens=64,
            retry_max_tokens=4096,
        )
        request_payload = ChatRequest(
            system_prompt="You are Justin.",
            conversation=[{"role": "user", "content": "你好"}],
            memory_snippets=[],
            latest_user_message="你好",
        )

        calls: list[dict] = []
        first_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "partial reasoning",
                    },
                    "finish_reason": "length",
                }
            ]
        }
        second_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "你好，我在。",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        def _fake_urlopen(http_request, timeout=0):
            calls.append(json.loads(http_request.data.decode("utf-8")))
            if len(calls) == 1:
                return _FakeHTTPResponse(first_response)
            return _FakeHTTPResponse(second_response)

        with patch("justin.models.request.urlopen", side_effect=_fake_urlopen):
            text = provider.generate(request_payload)

        self.assertEqual(text, "你好，我在。")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["max_tokens"], 64)
        self.assertGreater(calls[1]["max_tokens"], calls[0]["max_tokens"])


if __name__ == "__main__":
    unittest.main()
