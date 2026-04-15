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
    api_key: str

    def generate(self, payload: ChatRequest) -> str:
        messages = [{"role": "system", "content": payload.system_prompt}, *payload.conversation]
        body = json.dumps(
            {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.3,
            }
        ).encode("utf-8")
        http_request = request.Request(
            url=f"{self.api_base.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=60) as response:
                payload_json = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Model request failed: {exc.reason}") from exc

        try:
            return payload_json["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected model response: {payload_json}") from exc
