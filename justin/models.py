from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error, request

from .types import ChatRequest, ChatResponse, ChatToolCall

class ChatProvider:
    def generate(
        self, 
        payload: ChatRequest, 
        chunk_callback: Callable[[str, bool], None] | None = None
    ) -> ChatResponse:
        raise NotImplementedError

@dataclass(slots=True)
class LocalFallbackChatProvider(ChatProvider):
    def generate(
        self, 
        payload: ChatRequest, 
        chunk_callback: Callable[[str, bool], None] | None = None
    ) -> ChatResponse:
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

    def generate(
        self, 
        payload: ChatRequest, 
        chunk_callback: Callable[[str, bool], None] | None = None
    ) -> ChatResponse:
        import openai
        
        client = openai.OpenAI(
            base_url=self.api_base,
            api_key=self.api_key or "dummy",
            max_retries=3,
            timeout=self.timeout_seconds,
        )
        
        messages = [{"role": "system", "content": payload.system_prompt}, *payload.conversation]
        
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.max_tokens > 0:
                kwargs["max_tokens"] = self.max_tokens
            if payload.tools:
                kwargs["tools"] = payload.tools

            # Decide whether to stream
            stream = chunk_callback is not None
            
            completion = client.chat.completions.create(
                stream=stream,
                **kwargs
            )
            
            if stream:
                full_content = []
                full_reasoning = []
                tool_calls_dict = {}
                
                for chunk in completion:
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        full_reasoning.append(reasoning)
                        chunk_callback(reasoning, True)
                        
                    content = delta.content
                    if content:
                        full_content.append(content)
                        chunk_callback(content, False)
                        
                    if getattr(delta, "tool_calls", None):
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_dict:
                                tool_calls_dict[idx] = {"id": tc.id or "", "name": tc.function.name or "", "arguments": tc.function.arguments or ""}
                            else:
                                if tc.id: tool_calls_dict[idx]["id"] += tc.id
                                if tc.function.name: tool_calls_dict[idx]["name"] += tc.function.name
                                if tc.function.arguments: tool_calls_dict[idx]["arguments"] += tc.function.arguments
                                
                chat_tool_calls = [
                    ChatToolCall(id=v["id"], name=v["name"], arguments=v["arguments"]) 
                    for k, v in sorted(tool_calls_dict.items())
                ]
                
                content_str = "".join(full_content)
                reasoning_str = "".join(full_reasoning)
                
                return ChatResponse(
                    content=content_str,
                    tool_calls=chat_tool_calls,
                    reasoning_content=reasoning_str if reasoning_str else None
                )
            else:
                # Non-streaming fallback
                choice = completion.choices[0]
                message = choice.message
                
                content = message.content or ""
                reasoning = getattr(message, "reasoning_content", None)
                
                chat_tool_calls = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        chat_tool_calls.append(ChatToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments
                        ))
                
                return ChatResponse(
                    content=content,
                    tool_calls=chat_tool_calls,
                    reasoning_content=reasoning if reasoning else None
                )

        except openai.APIConnectionError as exc:
            raise RuntimeError(f"Connection failed: {exc}") from exc
        except openai.APITimeoutError as exc:
            raise RuntimeError(f"Request timed out after {self.timeout_seconds}s") from exc
        except Exception as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc
