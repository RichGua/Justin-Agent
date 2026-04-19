from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from .types import ChatRequest, ChatResponse, ChatToolCall

class ChatProvider:
    def generate(self, payload: ChatRequest) -> ChatResponse:
        raise NotImplementedError

@dataclass(slots=True)
class LocalFallbackChatProvider(ChatProvider):
    def generate(self, payload: ChatRequest) -> ChatResponse:
        latest = payload.latest_user_message.strip()
        memories = payload.memory_snippets[:3]

        if self._is_profile_question(latest):
            if memories:
                joined = "\n".join(f"- {item}" for item in memories)
                return ChatResponse(
                    content="这是我当前能确认的长期记忆线索：\n"
                    f"{joined}\n\n"
                    "如果其中有不准确的地方，你可以直接纠正，我会把新的说法作为候选记忆等待你确认。"
                )
            return ChatResponse(content="我还没有足够的已确认长期记忆。你可以直接告诉我你的偏好、身份背景或长期目标，我会先生成候选记忆给你审核。")

        if memories:
            memory_context = "我参考了这些已确认记忆：" + "；".join(memories)
        else:
            memory_context = "这次没有检索到足够的已确认长期记忆，所以我只根据当前对话来回答。"

        suggestion = "如果这条信息值得长期保留，你可以直接说“记住……”或者在候选记忆里确认它。"
        return ChatResponse(content=f"{memory_context}\n\n你刚才说的是：{latest}\n\n我建议把接下来要长期复用的信息沉淀成已确认记忆。{suggestion}")

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
class OpenAICompatibleChatProvider(ChatProvider):
    model_name: str
    api_base: str
    api_key: str | None = None
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 1024
    timeout_seconds: int = 60
    retry_max_tokens: int = 8192

    def generate(self, payload: ChatRequest) -> ChatResponse:
        messages = [{"role": "system", "content": payload.system_prompt}, *payload.conversation]
        payload_json = self._request_with_timeout_fallback(messages=messages, max_tokens=self.max_tokens, tools=payload.tools)
        if self._should_retry_for_length(payload_json):
            retry_tokens = self._next_retry_max_tokens(self.max_tokens)
            payload_json = self._request_with_timeout_fallback(messages=messages, max_tokens=retry_tokens, tools=payload.tools)

        return self._extract_response(payload_json)

    def _request_with_timeout_fallback(self, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict] | None = None) -> dict:
        try:
            return self._request_chat_completion(messages=messages, max_tokens=max_tokens, tools=tools)
        except RuntimeError as exc:
            if not self._is_timeout_error(exc):
                if self._is_transient_network_error(exc):
                    time.sleep(0.25)
                    return self._request_chat_completion(messages=messages, max_tokens=max_tokens, tools=tools)
                raise
            fallback_tokens = self._fallback_max_tokens_on_timeout(max_tokens)
            if fallback_tokens >= max_tokens:
                if self._is_transient_network_error(exc):
                    time.sleep(0.25)
                    return self._request_chat_completion(messages=messages, max_tokens=max_tokens, tools=tools)
                raise
            time.sleep(0.25)
            return self._request_chat_completion(messages=messages, max_tokens=fallback_tokens, tools=tools)

    def _request_chat_completion(self, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict] | None = None) -> dict:
        body_obj: dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            body_obj["tools"] = tools
        if 0 < self.top_p <= 1:
            body_obj["top_p"] = self.top_p
        if max_tokens > 0:
            body_obj["max_tokens"] = max_tokens

        body = json.dumps(body_obj).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = request.Request(
            url=f"{self.api_base.rstrip('/')}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        timeout = self._compute_timeout_seconds(messages)
        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(self._timeout_message(timeout)) from exc
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model request failed with HTTP {exc.code}: {detail}") from exc
        except http.client.RemoteDisconnected as exc:
            raise RuntimeError("Model connection closed by remote server before response.") from exc
        except error.URLError as exc:
            raise RuntimeError(self._format_network_error(exc)) from exc

    def _compute_timeout_seconds(self, messages: list[dict[str, str]]) -> int:
        base = max(int(self.timeout_seconds), 1)
        # Multi-turn sessions usually carry more context and can take longer.
        extra = max(len(messages) - 2, 0) * 8
        return base + min(extra, 120)

    def _fallback_max_tokens_on_timeout(self, current_tokens: int) -> int:
        current = max(int(current_tokens), 1)
        if current <= 256:
            return current
        return max(256, current // 2)

    def _is_timeout_error(self, exc: Exception) -> bool:
        lowered = str(exc).lower()
        return "timeout" in lowered or "timed out" in lowered

    def _is_transient_network_error(self, exc: Exception) -> bool:
        lowered = str(exc).lower()
        markers = (
            "remote end closed connection",
            "connection closed by remote server",
            "tls handshake failed",
            "unexpected eof while reading",
            "connection reset",
            "temporarily unavailable",
        )
        return any(marker in lowered for marker in markers)

    def _should_retry_for_length(self, payload_json: dict) -> bool:
        if self.retry_max_tokens <= 0:
            return False
        choices = payload_json.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        if first_choice.get("finish_reason") != "length":
            return False

        next_tokens = self._next_retry_max_tokens(self.max_tokens)
        return next_tokens > max(self.max_tokens, 0)

    def _next_retry_max_tokens(self, current_tokens: int) -> int:
        current = max(int(current_tokens), 1)
        candidate = max(current * 4, 1024)
        return min(candidate, max(int(self.retry_max_tokens), current))

    def _choice_has_final_text(self, choice: dict) -> bool:
        message = choice.get("message")
        if isinstance(message, dict):
            if self._normalize_message_content(message.get("content")):
                return True
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return True
        return False

    def _format_network_error(self, exc: error.URLError) -> str:
        reason = exc.reason
        reason_text = str(reason)
        lowered = reason_text.lower()

        if isinstance(reason, TimeoutError) or "timeout" in lowered or "timed out" in lowered:
            return self._timeout_message()

        winerror = getattr(reason, "winerror", None)
        if winerror == 10013:
            return (
                "Model request blocked by OS/network policy (WinError 10013). "
                "Allow outbound HTTPS for your Python runtime, or switch provider via 'Justin setup' "
                "to 'local'/'ollama' for offline use."
            )

        if "SSL: UNEXPECTED_EOF_WHILE_READING" in reason_text:
            return (
                "TLS handshake failed while connecting to model API. "
                "Check JUSTIN_API_BASE (domain typo/common proxy issue) and network policy."
            )

        return f"Model request failed: {reason_text}"

    def _timeout_message(self, timeout_seconds: int | None = None) -> str:
        effective_timeout = timeout_seconds if timeout_seconds is not None else max(int(self.timeout_seconds), 1)
        return (
            f"Model request timed out after {effective_timeout} seconds. "
            "Provider/network may be slow or blocked. Check API settings and retry."
        )

    def _extract_response(self, payload_json: dict) -> ChatResponse:
        choices = payload_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Unexpected model response: {payload_json}")

        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message")
        finish_reason = first_choice.get("finish_reason")

        if isinstance(message, dict):
            content = self._normalize_message_content(message.get("content")) or ""
            
            tool_calls_data = message.get("tool_calls")
            chat_tool_calls = []
            if isinstance(tool_calls_data, list):
                for tc in tool_calls_data:
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        chat_tool_calls.append(ChatToolCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            arguments=func.get("arguments", "")
                        ))

            reasoning = message.get("reasoning_content")
            reasoning_str = reasoning if isinstance(reasoning, str) and reasoning.strip() else None

            if content or chat_tool_calls:
                return ChatResponse(
                    content=content, 
                    tool_calls=chat_tool_calls,
                    reasoning_content=reasoning_str
                )

            # Some OpenAI-compatible providers (for example reasoning models on NIM)
            # may return only reasoning_content when output hits token limits.
            if reasoning_str:
                if finish_reason == "length":
                    raise RuntimeError(
                        "Model returned reasoning text but no final assistant content "
                        "(finish_reason=length). Increase JUSTIN_MODEL_MAX_TOKENS and retry."
                    )
                # Model returned ONLY reasoning content without tool calls or content
                return ChatResponse(content="", reasoning_content=reasoning_str)

        text = first_choice.get("text")
        if isinstance(text, str) and text.strip():
            return ChatResponse(content=text.strip())

        if finish_reason == "length":
            raise RuntimeError(
                "Model response was truncated before a final assistant message "
                "(finish_reason=length). If your provider supports it, raise the token limit and retry."
            )

        raise RuntimeError(f"Unexpected model response: {payload_json}")

    def _normalize_message_content(self, raw_content) -> str | None:
        if isinstance(raw_content, str):
            cleaned = raw_content.strip()
            return cleaned or None

        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, str):
                    chunk = item.strip()
                    if chunk:
                        parts.append(chunk)
                    continue
                if isinstance(item, dict):
                    for key in ("text", "content"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            parts.append(value.strip())
                            break
            if parts:
                return "\n".join(parts)

        return None
