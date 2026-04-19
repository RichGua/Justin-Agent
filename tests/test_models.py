from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APIConnectionError, APITimeoutError

from justin.models import CompletionOutcome, LocalFallbackChatProvider, OpenAICompatibleChatProvider
from justin.types import ChatRequest, ChatResponse


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://example.com/v1/chat/completions")


class ModelProviderTests(unittest.TestCase):
    def test_local_fallback_profile_response_is_readable(self) -> None:
        provider = LocalFallbackChatProvider()
        payload = ChatRequest(
            system_prompt="You are Justin.",
            conversation=[],
            memory_snippets=["likes concise output"],
            latest_user_message="What do you know about me?",
        )

        result = provider.generate(payload)

        self.assertIn("long-term", result.content.lower())
        self.assertIn("likes concise output", result.content)

    def test_consume_non_stream_extracts_content_and_tool_calls(self) -> None:
        provider = OpenAICompatibleChatProvider(
            model_name="test-model",
            api_base="https://example.com/v1",
            api_key="test-key",
        )
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=[SimpleNamespace(text="hello world")],
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(name="search_web", arguments='{"q":"hello"}'),
                            )
                        ],
                    ),
                )
            ]
        )

        outcome = provider._consume_non_stream(completion)

        self.assertEqual(outcome.finish_reason, "stop")
        self.assertEqual(outcome.response.content, "hello world")
        self.assertEqual(len(outcome.response.tool_calls), 1)
        self.assertEqual(outcome.response.tool_calls[0].name, "search_web")

    def test_generate_raises_when_truncated_reasoning_has_no_final_content(self) -> None:
        provider = OpenAICompatibleChatProvider(
            model_name="test-model",
            api_base="https://example.com/v1",
            api_key="test-key",
            max_tokens=64,
            retry_max_tokens=64,
        )
        payload = ChatRequest(
            system_prompt="You are Justin.",
            conversation=[{"role": "user", "content": "hello"}],
            memory_snippets=[],
            latest_user_message="hello",
        )

        with patch.object(
            provider,
            "_request_chat_completion",
            return_value=CompletionOutcome(
                response=ChatResponse(content="", reasoning_content="partial reasoning"),
                finish_reason="length",
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                provider.generate(payload)

        self.assertIn("JUSTIN_MODEL_MAX_TOKENS", str(ctx.exception))

    def test_generate_retries_with_higher_token_limit_after_length_finish(self) -> None:
        provider = OpenAICompatibleChatProvider(
            model_name="test-model",
            api_base="https://example.com/v1",
            api_key="test-key",
            max_tokens=64,
            retry_max_tokens=4096,
        )
        payload = ChatRequest(
            system_prompt="You are Justin.",
            conversation=[{"role": "user", "content": "hello"}],
            memory_snippets=[],
            latest_user_message="hello",
        )
        calls: list[tuple[int, bool, bool]] = []

        def side_effect(*args, **kwargs):
            calls.append((kwargs["max_tokens"], kwargs["stream"], kwargs["trust_env"]))
            if len(calls) == 1:
                return CompletionOutcome(
                    response=ChatResponse(content="", reasoning_content="partial reasoning"),
                    finish_reason="length",
                )
            return CompletionOutcome(
                response=ChatResponse(content="hello"),
                finish_reason="stop",
            )

        with patch.object(provider, "_request_chat_completion", side_effect=side_effect):
            result = provider.generate(payload, chunk_callback=lambda _chunk, _is_reasoning: None)

        self.assertEqual(result.content, "hello")
        self.assertEqual(calls[0], (64, True, True))
        self.assertEqual(calls[1], (1024, False, True))

    def test_generate_retries_with_lower_token_limit_after_timeout(self) -> None:
        provider = OpenAICompatibleChatProvider(
            model_name="test-model",
            api_base="https://example.com/v1",
            api_key="test-key",
            max_tokens=1024,
            timeout_seconds=60,
        )
        payload = ChatRequest(
            system_prompt="You are Justin.",
            conversation=[
                {"role": "user", "content": "msg-1"},
                {"role": "assistant", "content": "msg-2"},
                {"role": "user", "content": "msg-3"},
            ],
            memory_snippets=[],
            latest_user_message="msg-3",
        )
        calls: list[tuple[int, int]] = []

        def side_effect(*args, **kwargs):
            calls.append((kwargs["max_tokens"], kwargs["timeout_seconds"]))
            if len(calls) == 1:
                raise APITimeoutError(request=_request())
            return CompletionOutcome(
                response=ChatResponse(content="ok"),
                finish_reason="stop",
            )

        with (
            patch.object(provider, "_request_chat_completion", side_effect=side_effect),
            patch("justin.models.time.sleep"),
        ):
            result = provider.generate(payload)

        self.assertEqual(result.content, "ok")
        self.assertEqual(calls[0][0], 1024)
        self.assertEqual(calls[1][0], 512)
        self.assertGreater(calls[0][1], 60)

    def test_generate_retries_tls_stream_errors_without_env_proxy(self) -> None:
        provider = OpenAICompatibleChatProvider(
            model_name="test-model",
            api_base="https://example.com/v1",
            api_key="test-key",
            max_tokens=512,
        )
        payload = ChatRequest(
            system_prompt="You are Justin.",
            conversation=[{"role": "user", "content": "hello"}],
            memory_snippets=[],
            latest_user_message="hello",
        )
        calls: list[tuple[bool, bool]] = []

        def side_effect(*args, **kwargs):
            calls.append((kwargs["stream"], kwargs["trust_env"]))
            if len(calls) == 1:
                raise APIConnectionError(
                    message="stream disconnected before completion: tls handshake eof",
                    request=_request(),
                )
            return CompletionOutcome(
                response=ChatResponse(content="hello"),
                finish_reason="stop",
            )

        with (
            patch.object(provider, "_request_chat_completion", side_effect=side_effect),
            patch("justin.models.time.sleep"),
        ):
            result = provider.generate(payload, chunk_callback=lambda _chunk, _is_reasoning: None)

        self.assertEqual(result.content, "hello")
        self.assertEqual(calls, [(True, True), (False, False)])


if __name__ == "__main__":
    unittest.main()
