from __future__ import annotations

import re
from dataclasses import dataclass

from .types import ContextTelemetry, Message, RetrievedMemory, StoredToolFact, ToolEvent


@dataclass(slots=True)
class ContextBudgetPolicy:
    max_input_tokens: int = 3200
    reserved_output_tokens: int = 800
    recent_messages: int = 8
    summary_trigger_messages: int = 10
    tool_fact_limit: int = 4
    compression_tier: str = "balanced"


@dataclass(slots=True)
class ContextAssembly:
    system_prompt: str
    conversation: list[dict[str, str]]
    summary: str
    telemetry: ContextTelemetry


class ConversationContextBuilder:
    def __init__(self, store, policy: ContextBudgetPolicy) -> None:
        self.store = store
        self.policy = policy

    def build(
        self,
        session_id: str,
        latest_user_message: str,
        messages: list[Message],
        recalled_memories: list[RetrievedMemory],
        tool_events: list[ToolEvent],
        tool_facts: list[StoredToolFact],
        activated_skill_block: str,
        citation_block: str,
    ) -> ContextAssembly:
        summary = self._refresh_summary(session_id, messages)
        memory_block = self._format_memories(recalled_memories)
        tool_fact_block = self._format_tool_facts(tool_facts[: self.policy.tool_fact_limit])
        tool_event_block = self._format_tool_events(tool_events)
        recent_messages = self._select_recent_messages(messages, summary, memory_block, tool_fact_block, tool_event_block)
        system_prompt = self._build_system_prompt(
            memory_block=memory_block,
            summary=summary,
            tool_fact_block=tool_fact_block,
            tool_event_block=tool_event_block,
            activated_skill_block=activated_skill_block,
            citation_block=citation_block,
        )

        raw_context_text = "\n".join(message.content for message in messages)
        final_context_text = system_prompt + "\n" + "\n".join(message["content"] for message in recent_messages)
        before_tokens = estimate_tokens(raw_context_text) + estimate_tokens(memory_block)
        after_tokens = estimate_tokens(final_context_text)
        telemetry = ContextTelemetry(
            input_tokens_estimate=after_tokens,
            output_tokens_estimate=0,
            context_tokens_before=before_tokens,
            context_tokens_after=after_tokens,
            saved_tokens=max(before_tokens - after_tokens, 0),
            tool_tokens_saved=0,
        )
        return ContextAssembly(
            system_prompt=system_prompt,
            conversation=recent_messages,
            summary=summary,
            telemetry=telemetry,
        )

    def _refresh_summary(self, session_id: str, messages: list[Message]) -> str:
        if len(messages) <= self.policy.summary_trigger_messages:
            stored = self.store.get_session_summary(session_id)
            return stored.summary if stored else ""

        preserved = min(self.policy.recent_messages, max(len(messages) // 3, 4))
        older_messages = messages[:-preserved]
        if not older_messages:
            stored = self.store.get_session_summary(session_id)
            return stored.summary if stored else ""

        existing = self.store.get_session_summary(session_id)
        if existing and existing.source_message_count == len(older_messages):
            return existing.summary

        summary = self._summarize_messages(older_messages)
        self.store.upsert_session_summary(session_id, summary, source_message_count=len(older_messages))
        return summary

    def _summarize_messages(self, messages: list[Message]) -> str:
        bullets: list[str] = []
        for message in messages[-10:]:
            content = " ".join(message.content.split())
            if not content:
                continue
            bullets.append(f"- {message.role}: {content[:140]}")
        return "\n".join(bullets[:8])

    def _select_recent_messages(
        self,
        messages: list[Message],
        summary: str,
        memory_block: str,
        tool_fact_block: str,
        tool_event_block: str,
    ) -> list[dict[str, str]]:
        fixed_cost = estimate_tokens("\n".join([summary, memory_block, tool_fact_block, tool_event_block]))
        token_budget = max(self.policy.max_input_tokens - self.policy.reserved_output_tokens - fixed_cost, 200)
        selected: list[Message] = []
        running = 0
        for message in reversed(messages):
            message_tokens = estimate_tokens(message.content)
            if selected and running + message_tokens > token_budget:
                break
            selected.append(message)
            running += message_tokens
            if len(selected) >= self.policy.recent_messages:
                break
        selected.reverse()
        return [{"role": message.role, "content": message.content} for message in selected]

    def _build_system_prompt(
        self,
        *,
        memory_block: str,
        summary: str,
        tool_fact_block: str,
        tool_event_block: str,
        activated_skill_block: str,
        citation_block: str,
    ) -> str:
        blocks = [
            "You are Justin, a practical local-first agent.",
            "Prefer concise answers. Use tool evidence when available instead of speculating.",
            "If current-turn evidence contains source labels like [S1], cite them inline when relevant.",
            f"Approved memories:\n{memory_block}",
        ]
        if summary:
            blocks.append(f"Compressed earlier session context:\n{summary}")
        if tool_fact_block:
            blocks.append(f"Reusable tool facts:\n{tool_fact_block}")
        if tool_event_block:
            blocks.append(f"Current tool evidence:\n{tool_event_block}")
        if activated_skill_block:
            blocks.append(f"Activated skills:\n{activated_skill_block}")
        if citation_block:
            blocks.append(f"Available citations:\n{citation_block}")
        return "\n\n".join(blocks)

    def _format_memories(self, memories: list[RetrievedMemory]) -> str:
        if not memories:
            return "- none"
        return "\n".join(f"- {memory.content}" for memory in memories[:5])

    def _format_tool_facts(self, facts: list[StoredToolFact]) -> str:
        if not facts:
            return ""
        return "\n".join(f"- [{fact.fact_type}] {fact.content}" for fact in facts)

    def _format_tool_events(self, events: list[ToolEvent]) -> str:
        if not events:
            return ""
        lines: list[str] = []
        for event in events[:6]:
            marker = "ok" if event.ok else "error"
            lines.append(f"- {event.tool_name} ({marker}): {event.summary}")
        return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    wide_chars = len(text) - ascii_chars
    words = len(re.findall(r"[A-Za-z0-9_]+", text))
    estimate = (ascii_chars / 4.0) + (wide_chars * 1.1) + (words * 0.25)
    return max(int(estimate), 1)
