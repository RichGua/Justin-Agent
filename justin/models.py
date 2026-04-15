from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request

from .types import ChatRequest


class ChatProvider:
    def generate(self, payload: ChatRequest) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class LocalFallbackChatProvider(ChatProvider):
    def generate(self, payload: ChatRequest) -> str:
        latest = payload.latest_user_message.strip()
        memories = payload.memory_snippets[:3]

        if self._is_profile_question(latest):
            if memories:
                joined = "\n".join(f"- {item}" for item in memories)
                return (
                    "这是我当前能确认的长期记忆线索：\n"
                    f"{joined}\n\n"
                    "如果其中有不准确的地方，你可以直接纠正，我会把新的说法作为候选记忆等待你确认。"
                )
            return "我还没有足够的已确认长期记忆。你可以直接告诉我你的偏好、身份背景或长期目标，我会先生成候选记忆给你审核。"

        if memories:
            memory_context = "我参考了这些已确认记忆：" + "；".join(memories)
        else:
            memory_context = "这次没有检索到足够的已确认长期记忆，所以我只根据当前对话来回答。"

        suggestion = "如果这条信息值得长期保留，你可以直接说“记住……”或者在候选记忆里确认它。"
        return f"{memory_context}\n\n你刚才说的是：{latest}\n\n我建议把接下来要长期复用的信息沉淀成已确认记忆。{suggestion}"

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

    def generate(self, payload: ChatRequest) -> str:
        messages = [{"role": "system", "content": payload.system_prompt}, *payload.conversation]
        body = json.dumps(
            {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.3,
            }
        ).encode("utf-8")
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
        try:
            with request.urlopen(http_request, timeout=60) as response:
                payload_json = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(self._timeout_message()) from exc
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(self._format_network_error(exc)) from exc

        try:
            return self._extract_response_text(payload_json)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected model response: {payload_json}") from exc

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

    def _timeout_message(self) -> str:
        return (
            "Model request timed out after 60 seconds. "
            "Provider/network may be slow or blocked. Check API settings and retry."
        )

    def _extract_response_text(self, payload_json: dict) -> str:
        choices = payload_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Unexpected model response: {payload_json}")

        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message")
        finish_reason = first_choice.get("finish_reason")

        if isinstance(message, dict):
            content = self._normalize_message_content(message.get("content"))
            if content:
                return content

            # Some OpenAI-compatible providers (for example reasoning models on NIM)
            # may return only reasoning_content when output hits token limits.
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                if finish_reason == "length":
                    return (
                        "Model response was truncated before final assistant content "
                        "(finish_reason=length). If your provider supports it, raise the token limit and retry.\n\n"
                        f"[partial reasoning]\n{reasoning.strip()}"
                    )
                return reasoning.strip()

        text = first_choice.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

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
