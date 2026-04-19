from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .types import ChatRequest, ChatResponse, ChatToolCall


class ChatProvider:
    def generate(
        self,
        payload: ChatRequest,
        chunk_callback: Callable[[str, bool], None] | None = None,
    ) -> ChatResponse:
        raise NotImplementedError


@dataclass(slots=True)
class LocalFallbackChatProvider(ChatProvider):
    def generate(
        self,
        payload: ChatRequest,
        chunk_callback: Callable[[str, bool], None] | None = None,
    ) -> ChatResponse:
        latest = payload.latest_user_message.strip()
        memories = payload.memory_snippets[:3]

        if self._is_profile_question(latest):
            if memories:
                joined = "\n".join(f"- {item}" for item in memories)
                return ChatResponse(
                    content="Here are the long-term details I currently remember about you:\n"
                    f"{joined}\n\n"
                    "If any of this is wrong, correct it and I will store the updated version as a candidate memory."
                )
            return ChatResponse(
                content=(
                    "I do not have any confirmed long-term memories about you yet. "
                    "Tell me your preferences, background, or goals and I can turn them into candidate memories for review."
                )
            )

        if memories:
            memory_context = "I considered these confirmed memories: " + "; ".join(memories)
        else:
            memory_context = (
                "I did not find enough confirmed long-term memories for this turn, "
                "so I am answering only from the current conversation."
            )

        suggestion = 'If this is worth keeping, say "remember ..." or approve it from the candidate memories list.'
        return ChatResponse(content=f"{memory_context}\n\nYou said: {latest}\n\n{suggestion}")

    def _is_profile_question(self, latest: str) -> bool:
        normalized = latest.lower()
        prompts = [
            "what do you know about me",
            "who am i",
            "你记得我什么",
            "你了解我什么",
            "你知道我什么",
        ]
        return any(prompt in normalized for prompt in prompts)


@dataclass(slots=True)
class CompletionOutcome:
    response: ChatResponse
    finish_reason: str | None = None


@dataclass(slots=True)
class OpenAICompatibleChatProvider(ChatProvider):
    model_name: str
    api_base: str
    api_key: str | None = None
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 1024
    timeout_seconds: int = 60
    retry_max_tokens: int = 8192

    def _next_retry_max_tokens(self, current_tokens: int) -> int:
        current = max(int(current_tokens), 1)
        candidate = max(current * 4, 1024)
        return min(candidate, max(int(self.retry_max_tokens), current))

    def _fallback_max_tokens_on_timeout(self, current_tokens: int) -> int:
        current = int(current_tokens)
        if current <= 256:
            return current
        return max(256, current // 2)

    def _compute_timeout_seconds(self, messages: list[dict[str, Any]]) -> int:
        base = max(int(self.timeout_seconds), 1)
        extra = max(len(messages) - 2, 0) * 8
        return base + min(extra, 120)

    def _timeout_message(self, timeout_seconds: int) -> str:
        return (
            f"Request timed out after {timeout_seconds}s. "
            "Provider/network may be slow or blocked. Check API settings and retry."
        )

    def generate(
        self,
        payload: ChatRequest,
        chunk_callback: Callable[[str, bool], None] | None = None,
    ) -> ChatResponse:
        import openai

        messages = [{"role": "system", "content": payload.system_prompt}, *payload.conversation]
        current_max_tokens = self.max_tokens
        effective_timeout = self._compute_timeout_seconds(messages)
        stream = chunk_callback is not None
        trust_env = True
        timeout_retry_used = False
        connection_retries = 0

        while True:
            try:
                outcome = self._request_chat_completion(
                    openai_module=openai,
                    messages=messages,
                    max_tokens=current_max_tokens,
                    tools=payload.tools,
                    timeout_seconds=effective_timeout,
                    stream=stream,
                    trust_env=trust_env,
                    chunk_callback=chunk_callback,
                )

                if outcome.finish_reason == "length" and self.retry_max_tokens > 0:
                    next_tokens = self._next_retry_max_tokens(current_max_tokens)
                    if next_tokens > max(current_max_tokens, 0):
                        current_max_tokens = next_tokens
                        stream = False
                        continue

                if (
                    outcome.finish_reason == "length"
                    and not outcome.response.content
                    and not outcome.response.tool_calls
                    and outcome.response.reasoning_content
                ):
                    raise RuntimeError(
                        "Model returned reasoning text but no final assistant content "
                        "(finish_reason=length). Increase JUSTIN_MODEL_MAX_TOKENS and retry."
                    )

                return outcome.response

            except openai.APITimeoutError as exc:
                fallback_tokens = self._fallback_max_tokens_on_timeout(current_max_tokens)
                if not timeout_retry_used and fallback_tokens < current_max_tokens:
                    timeout_retry_used = True
                    current_max_tokens = fallback_tokens
                    stream = False
                    time.sleep(0.25)
                    continue
                raise RuntimeError(self._timeout_message(effective_timeout)) from exc
            except openai.APIConnectionError as exc:
                detail = self._format_connection_error(exc)
                if connection_retries < 2 and self._is_transient_network_error(detail):
                    connection_retries += 1
                    stream = False
                    if trust_env and self._should_retry_without_env(detail):
                        trust_env = False
                    time.sleep(0.25)
                    continue
                raise RuntimeError(detail) from exc
            except Exception as exc:
                detail = str(exc).strip() or exc.__class__.__name__
                if connection_retries < 2 and self._is_transient_network_error(detail):
                    connection_retries += 1
                    stream = False
                    if trust_env and self._should_retry_without_env(detail):
                        trust_env = False
                    time.sleep(0.25)
                    continue
                raise RuntimeError(f"Model request failed: {detail}") from exc

    def _request_chat_completion(
        self,
        openai_module: Any,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        timeout_seconds: int,
        stream: bool,
        trust_env: bool,
        chunk_callback: Callable[[str, bool], None] | None,
    ) -> CompletionOutcome:
        client = openai_module.OpenAI(
            base_url=self.api_base,
            api_key=self.api_key or "dummy",
            max_retries=0,
            timeout=timeout_seconds,
            http_client=openai_module.DefaultHttpxClient(trust_env=trust_env),
        )
        try:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if max_tokens > 0:
                kwargs["max_tokens"] = max_tokens
            if tools:
                kwargs["tools"] = tools

            completion = client.chat.completions.create(stream=stream, **kwargs)
            if stream:
                return self._consume_stream(completion, chunk_callback)
            return self._consume_non_stream(completion)
        finally:
            client.close()

    def _consume_stream(
        self,
        completion: Any,
        chunk_callback: Callable[[str, bool], None] | None,
    ) -> CompletionOutcome:
        full_content: list[str] = []
        full_reasoning: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None

        for chunk in completion:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason

            reasoning = self._normalize_message_content(getattr(delta, "reasoning_content", None))
            if reasoning:
                full_reasoning.append(reasoning)
                if chunk_callback is not None:
                    chunk_callback(reasoning, True)

            content = self._normalize_message_content(getattr(delta, "content", None))
            if content:
                full_content.append(content)
                if chunk_callback is not None:
                    chunk_callback(content, False)

            for tool_call in getattr(delta, "tool_calls", None) or []:
                index = getattr(tool_call, "index", 0)
                entry = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                tool_function = getattr(tool_call, "function", None)
                if getattr(tool_call, "id", None):
                    entry["id"] += tool_call.id
                if tool_function and getattr(tool_function, "name", None):
                    entry["name"] += tool_function.name
                if tool_function and getattr(tool_function, "arguments", None):
                    entry["arguments"] += tool_function.arguments

        return CompletionOutcome(
            response=ChatResponse(
                content="".join(full_content),
                tool_calls=[
                    ChatToolCall(id=item["id"], name=item["name"], arguments=item["arguments"])
                    for _, item in sorted(tool_calls.items())
                ],
                reasoning_content="".join(full_reasoning) or None,
            ),
            finish_reason=finish_reason,
        )

    def _consume_non_stream(self, completion: Any) -> CompletionOutcome:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise RuntimeError("Unexpected model response: no choices returned.")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise RuntimeError("Unexpected model response: missing message payload.")

        tool_calls: list[ChatToolCall] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            tool_function = getattr(tool_call, "function", None)
            tool_calls.append(
                ChatToolCall(
                    id=getattr(tool_call, "id", "") or "",
                    name=getattr(tool_function, "name", "") or "",
                    arguments=getattr(tool_function, "arguments", "") or "",
                )
            )

        return CompletionOutcome(
            response=ChatResponse(
                content=self._normalize_message_content(getattr(message, "content", None)) or "",
                tool_calls=tool_calls,
                reasoning_content=self._normalize_message_content(
                    getattr(message, "reasoning_content", None)
                ),
            ),
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def _normalize_message_content(self, raw_content: Any) -> str | None:
        if isinstance(raw_content, str):
            cleaned = raw_content.strip()
            return cleaned or None

        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        parts.append(cleaned)
                    continue
                if isinstance(item, dict):
                    for key in ("text", "content"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            parts.append(value.strip())
                            break
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n".join(parts)

        return None

    def _format_connection_error(self, exc: Exception) -> str:
        detail = " | ".join(self._exception_messages(exc))
        lowered = detail.lower()

        if "winerror 10013" in lowered:
            return (
                "Connection failed: outbound HTTPS was blocked by the OS or network policy "
                "(WinError 10013). Check firewall, VPN, or proxy settings, or switch providers."
            )

        if "tls handshake eof" in lowered or "stream disconnected before completion" in lowered:
            return (
                "Connection failed: stream disconnected before completion during TLS handshake. "
                "This is often caused by a broken HTTPS proxy, VPN, or an incompatible API base."
            )

        if detail:
            return f"Connection failed: {detail}"
        return "Connection failed."

    def _exception_messages(self, exc: BaseException) -> list[str]:
        messages: list[str] = []
        seen: set[int] = set()
        current: BaseException | None = exc

        while current is not None and id(current) not in seen:
            seen.add(id(current))
            text = str(current).strip()
            if text and text not in messages:
                messages.append(text)
            current = current.__cause__ or current.__context__

        return messages

    def _is_transient_network_error(self, detail: str) -> bool:
        lowered = detail.lower()
        markers = (
            "stream disconnected before completion",
            "tls handshake eof",
            "tls handshake failed",
            "unexpected eof while reading",
            "connection reset",
            "connection aborted",
            "connection closed by remote server",
            "remote end closed connection",
            "server disconnected",
            "temporarily unavailable",
        )
        return any(marker in lowered for marker in markers)

    def _should_retry_without_env(self, detail: str) -> bool:
        lowered = detail.lower()
        markers = (
            "tls handshake eof",
            "unexpected eof while reading",
            "proxy",
            "tunnel",
        )
        return any(marker in lowered for marker in markers)
