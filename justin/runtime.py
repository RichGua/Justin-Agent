from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from .config import (
    AgentConfig,
    PROVIDER_LOCAL,
    PROVIDER_NVIDIA_NIM,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPATIBLE,
)
from .embeddings import LocalHashEmbeddingProvider
from .extractor import HeuristicMemoryExtractor
from .models import ChatProvider, LocalFallbackChatProvider, OpenAICompatibleChatProvider
from .storage import AgentStore
from .types import AgentTurnResult, ChatRequest, MemoryCandidate


@dataclass(slots=True)
class RuntimeBundle:
    config: AgentConfig
    store: AgentStore
    extractor: HeuristicMemoryExtractor
    chat_provider: ChatProvider


def build_chat_provider(config: AgentConfig) -> ChatProvider:
    provider = (config.model_provider or PROVIDER_LOCAL).lower()
    if provider == PROVIDER_LOCAL:
        return LocalFallbackChatProvider()

    if provider in {PROVIDER_OPENAI, PROVIDER_OPENAI_COMPATIBLE}:
        api_base = (config.api_base or "https://api.openai.com/v1").strip()
        _validate_api_base(provider, api_base)
        api_key = (config.api_key or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI provider requires JUSTIN_API_KEY.")
        model_name = config.model_name or "gpt-4.1-mini"
        return OpenAICompatibleChatProvider(model_name=model_name, api_base=api_base, api_key=api_key)

    if provider == PROVIDER_OLLAMA:
        api_base = (config.api_base or "http://localhost:11434/v1").strip()
        _validate_api_base(provider, api_base)
        model_name = config.model_name or "llama3.1"
        api_key = (config.api_key or "").strip() or None
        return OpenAICompatibleChatProvider(model_name=model_name, api_base=api_base, api_key=api_key)

    if provider == PROVIDER_NVIDIA_NIM:
        api_base = (config.api_base or "https://integrate.api.nvidia.com/v1").strip()
        _validate_api_base(provider, api_base)
        api_key = (config.api_key or "").strip()
        if not api_key:
            raise RuntimeError("NVIDIA NIM provider requires JUSTIN_API_KEY.")
        model_name = config.model_name or "meta/llama-3.1-70b-instruct"
        return OpenAICompatibleChatProvider(model_name=model_name, api_base=api_base, api_key=api_key)

    raise RuntimeError(
        f"Unsupported JUSTIN_MODEL_PROVIDER '{config.model_provider}'. "
        f"Use one of: local, openai, ollama, nvidia-nim."
    )


def _validate_api_base(provider: str, api_base: str) -> None:
    parsed = urlparse(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            f"Invalid JUSTIN_API_BASE '{api_base}'. "
            "Expected a full URL, e.g. 'https://api.openai.com/v1'."
        )

    host = parsed.netloc.lower()
    if "nidia.com" in host:
        raise RuntimeError(
            "JUSTIN_API_BASE appears to contain a typo: 'nidia.com'. "
            "Did you mean 'nvidia.com'?"
        )

    if provider == PROVIDER_NVIDIA_NIM and "nvidia.com" not in host:
        raise RuntimeError(
            "NVIDIA NIM provider expects JUSTIN_API_BASE to point to NVIDIA, "
            "for example 'https://integrate.api.nvidia.com/v1'."
        )


def build_runtime_bundle(config: AgentConfig | None = None) -> RuntimeBundle:
    config = config or AgentConfig.from_env()
    config.ensure_directories()
    embedder = LocalHashEmbeddingProvider(dimensions=config.embedding_dimensions)
    store = AgentStore(config.database_path, embedder)
    extractor = HeuristicMemoryExtractor()
    chat_provider = build_chat_provider(config)

    return RuntimeBundle(config=config, store=store, extractor=extractor, chat_provider=chat_provider)


class JustinRuntime:
    def __init__(self, bundle: RuntimeBundle) -> None:
        self.config = bundle.config
        self.store = bundle.store
        self.extractor = bundle.extractor
        self.chat_provider = bundle.chat_provider

    def send_message(self, content: str, session_id: str | None = None) -> AgentTurnResult:
        session = self.store.get_session(session_id)
        if session is None:
            title = content.strip()[:40] or "New session"
            session = self.store.create_session(title=title)

        self.store.add_message(session.id, "user", content)
        recalled_memories = self.store.search_memories(content, top_k=self.config.retrieval_top_k)
        recent_messages = self.store.list_messages(session.id, limit=self.config.context_messages)

        payload = ChatRequest(
            system_prompt=self._build_system_prompt(recalled_memories),
            conversation=[{"role": message.role, "content": message.content} for message in recent_messages],
            memory_snippets=[memory.content for memory in recalled_memories],
            latest_user_message=content,
        )

        response_text = self.chat_provider.generate(payload)
        candidate_records = self._create_candidates(content, session.id)

        assistant_message = self.store.add_message(
            session.id,
            "assistant",
            response_text,
            metadata={
                "recalled_memory_ids": [memory.id for memory in recalled_memories],
                "candidate_ids": [candidate.id for candidate in candidate_records],
            },
        )

        return AgentTurnResult(
            session=session,
            assistant_message=assistant_message,
            recalled_memories=recalled_memories,
            candidates=candidate_records,
        )

    def _create_candidates(self, content: str, session_id: str) -> list[MemoryCandidate]:
        drafts = self.extractor.extract(content)
        candidates: list[MemoryCandidate] = []
        for draft in drafts:
            candidate = self.store.create_candidate(
                kind=draft.kind,
                content=draft.content,
                evidence=draft.evidence,
                confidence=draft.confidence,
                source_session_id=session_id,
            )
            if candidate.status == "pending":
                candidates.append(candidate)
        return candidates

    def confirm_candidate(self, candidate_id: str):
        return self.store.confirm_candidate(candidate_id)

    def reject_candidate(self, candidate_id: str, note: str | None = None):
        return self.store.reject_candidate(candidate_id, review_note=note)

    def list_sessions(self):
        return self.store.list_sessions()

    def list_messages(self, session_id: str):
        return self.store.list_messages(session_id)

    def list_candidates(self, status: str | None = None):
        return self.store.list_candidates(status=status)

    def list_memories(self, kind: str | None = None):
        return self.store.list_memories(kind=kind)

    def search_memories(self, query: str, top_k: int | None = None):
        return self.store.search_memories(query, top_k=top_k or self.config.retrieval_top_k)

    def close(self) -> None:
        self.store.close()

    def apply_config(self, config: AgentConfig) -> None:
        # Keep runtime state but switch model/config behavior in-process.
        self.config.model_provider = config.model_provider
        self.config.model_name = config.model_name
        self.config.api_base = config.api_base
        self.config.api_key = config.api_key
        self.config.retrieval_top_k = config.retrieval_top_k
        self.config.context_messages = config.context_messages
        self.chat_provider = build_chat_provider(self.config)

    def _build_system_prompt(self, recalled_memories) -> str:
        memory_block = "\n".join(f"- {memory.content}" for memory in recalled_memories[:5]) or "- none"
        return (
            "You are Justin, a local-first agent.\n"
            "Use recalled memories only as helpful background, not as unquestionable truth.\n"
            "Never claim that unapproved candidate memories are confirmed facts.\n"
            "Prefer concise, practical answers.\n"
            "If the user states stable facts, preferences, goals, or project context, they can become candidate memories for later review.\n"
            f"Relevant approved memories:\n{memory_block}"
        )
