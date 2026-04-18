from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .config import (
    AgentConfig,
    PROVIDER_LOCAL,
    PROVIDER_NVIDIA_NIM,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPATIBLE,
)
from .context import ContextBudgetPolicy, ConversationContextBuilder
from .embeddings import LocalHashEmbeddingProvider
from .extensions import ExtensionRegistry
from .extractor import HeuristicMemoryExtractor
from .models import ChatProvider, LocalFallbackChatProvider, OpenAICompatibleChatProvider
from .search import DuckDuckGoHTMLSearchProvider, SearchItem, SearchService, WikipediaSearchProvider
from .skills import SkillManager
from .storage import AgentStore
from .tools import ExecutionPolicy, ToolContext, ToolRegistry, build_default_tool_registry
from .types import AgentTurnResult, ChatRequest, Citation, MemoryCandidate, ToolEvent, to_plain_dict


@dataclass(slots=True)
class RuntimeBundle:
    config: AgentConfig
    store: AgentStore
    extractor: HeuristicMemoryExtractor
    chat_provider: ChatProvider
    context_builder: ConversationContextBuilder
    search_service: SearchService | None
    tool_registry: ToolRegistry
    skill_manager: SkillManager
    extensions: ExtensionRegistry


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
        return OpenAICompatibleChatProvider(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=config.model_temperature,
            top_p=config.model_top_p,
            max_tokens=config.model_max_tokens,
            timeout_seconds=config.model_timeout_seconds,
            retry_max_tokens=config.model_retry_max_tokens,
        )

    if provider == PROVIDER_OLLAMA:
        api_base = (config.api_base or "http://localhost:11434/v1").strip()
        _validate_api_base(provider, api_base)
        model_name = config.model_name or "llama3.1"
        api_key = (config.api_key or "").strip() or None
        return OpenAICompatibleChatProvider(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=config.model_temperature,
            top_p=config.model_top_p,
            max_tokens=config.model_max_tokens,
            timeout_seconds=config.model_timeout_seconds,
            retry_max_tokens=config.model_retry_max_tokens,
        )

    if provider == PROVIDER_NVIDIA_NIM:
        api_base = (config.api_base or "https://integrate.api.nvidia.com/v1").strip()
        _validate_api_base(provider, api_base)
        api_key = (config.api_key or "").strip()
        if not api_key:
            raise RuntimeError("NVIDIA NIM provider requires JUSTIN_API_KEY.")
        model_name = config.model_name or "meta/llama-3.1-70b-instruct"
        return OpenAICompatibleChatProvider(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=config.model_temperature,
            top_p=config.model_top_p,
            max_tokens=config.model_max_tokens,
            timeout_seconds=config.model_timeout_seconds,
            retry_max_tokens=config.model_retry_max_tokens,
        )

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
    context_builder = _build_context_builder(config, store, chat_provider)
    search_service = _build_search_service(config, store)
    tool_registry = _build_tool_registry(config, search_service)
    skill_manager = SkillManager(config.skills_dir, store)
    extensions = ExtensionRegistry()

    return RuntimeBundle(
        config=config,
        store=store,
        extractor=extractor,
        chat_provider=chat_provider,
        context_builder=context_builder,
        search_service=search_service,
        tool_registry=tool_registry,
        skill_manager=skill_manager,
        extensions=extensions,
    )


def _build_context_builder(config: AgentConfig, store: AgentStore, chat_provider: ChatProvider) -> ConversationContextBuilder:
    policy = ContextBudgetPolicy(
        max_input_tokens=config.context_max_input_tokens,
        reserved_output_tokens=config.context_reserved_output_tokens,
        recent_messages=config.context_messages,
        summary_trigger_messages=config.context_summary_trigger_messages,
        tool_fact_limit=config.context_tool_fact_limit,
        compression_tier=config.context_policy,
    )
    return ConversationContextBuilder(store=store, policy=policy, chat_provider=chat_provider)


def _build_search_service(config: AgentConfig, store: AgentStore) -> SearchService | None:
    if not config.network_tools_enabled:
        return None

    providers = []
    requested = [item.strip().lower() for item in config.search_provider_order.split(",") if item.strip()]
    for name in requested:
        if name == "duckduckgo":
            providers.append(DuckDuckGoHTMLSearchProvider())
        elif name == "wikipedia":
            providers.append(WikipediaSearchProvider())
    if not providers:
        providers = [DuckDuckGoHTMLSearchProvider(), WikipediaSearchProvider()]

    return SearchService(
        store=store,
        providers=providers,
        cache_ttl_hours=config.search_cache_ttl_hours,
    )


def _build_tool_registry(config: AgentConfig, search_service: SearchService | None) -> ToolRegistry:
    allowed_programs = {
        item.strip() for item in config.shell_allowed_programs.split(",") if item.strip()
    } or {"git", "rg", "where"}
    policy = ExecutionPolicy(
        workspace_root=Path.cwd().resolve(),
        home_dir=config.home_dir.resolve(),
        allowed_programs=allowed_programs,
        network_enabled=config.network_tools_enabled,
        max_output_chars=config.tool_max_output_chars,
    )
    return build_default_tool_registry(policy=policy, search_service=search_service)


class JustinRuntime:
    def __init__(self, bundle: RuntimeBundle) -> None:
        self.config = bundle.config
        self.store = bundle.store
        self.extractor = bundle.extractor
        self.chat_provider = bundle.chat_provider
        self.context_builder = bundle.context_builder
        self.search_service = bundle.search_service
        self.tool_registry = bundle.tool_registry
        self.skill_manager = bundle.skill_manager
        self.extensions = bundle.extensions

    def send_message(self, content: str, session_id: str | None = None) -> AgentTurnResult:
        session = self.store.get_session(session_id)
        if session is None:
            title = content.strip()[:40] or "New session"
            session = self.store.create_session(title=title)

        self.store.add_message(session.id, "user", content)
        recalled_memories = self.store.search_memories(content, top_k=self.config.retrieval_top_k)
        activated_skills = self.skill_manager.match_for_query(content)
        tool_facts = self.store.search_tool_facts(
            content,
            session_id=session.id,
            top_k=self.config.context_tool_fact_limit,
        )
        messages = self.store.list_messages(
            session.id,
            limit=max(
                self.config.context_messages * 3,
                self.config.context_summary_trigger_messages + self.config.context_messages + 4,
            ),
        )

        turn_tool_events: list[ToolEvent] = []
        citations: list[Citation] = []
        
        tools_schema = self._build_tools_schema()
        intermediate_messages = []

        for i in range(25):  # Max tool call iterations
            if i == 20:
                intermediate_messages.append({
                    "role": "user",
                    "content": "[SYSTEM WARNING] You have reached iteration 20. You MUST stop using tools and provide a final answer immediately to avoid an infinite loop."
                })
            context = self.context_builder.build(
                session_id=session.id,
                latest_user_message=content,
                messages=messages,
                recalled_memories=recalled_memories,
                tool_events=turn_tool_events,
                tool_facts=tool_facts,
                activated_skill_block=self.skill_manager.build_activation_block(activated_skills),
                citation_block=_format_citation_block(citations),
            )

            # Combine DB messages with intermediate tool_call and tool messages
            full_conversation = context.conversation + intermediate_messages

            payload = ChatRequest(
                system_prompt=context.system_prompt,
                conversation=full_conversation,
                memory_snippets=[memory.content for memory in recalled_memories],
                latest_user_message=content,
                citations=citations,
                tool_events=turn_tool_events,
                activated_skills=activated_skills,
                tools=tools_schema,
            )

            response = self.chat_provider.generate(payload)

            if not response.tool_calls:
                response_text = response.content
                break

            # Execute tool calls
            tool_calls_dicts = []
            for tc in response.tool_calls:
                tool_calls_dicts.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                    }
                })
            
            # Append assistant message with tool_calls
            intermediate_messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": tool_calls_dicts
            })
            
            # Let's execute the tools
            for tc in response.tool_calls:
                import json
                try:
                    arguments = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                result = self.tool_registry.execute(tc.name, arguments, self._tool_context(session.id))
                output = result.output if isinstance(result.output, dict) else {"value": result.output}
                event = self.store.add_tool_event(
                    session_id=session.id,
                    tool_name=tc.name,
                    arguments=arguments,
                    ok=result.ok,
                    output=output,
                    summary=result.summary,
                    error=result.error,
                    latency_ms=int(result.meta.get("latency_ms", 0)),
                    source=result.source,
                )
                turn_tool_events.append(event)
                self._persist_tool_facts(event)
                citations.extend(self._citations_for_event(event))

                # Append tool result message
                intermediate_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": json.dumps(output, ensure_ascii=False) if result.ok else str(result.error)
                })
            
            citations = _dedupe_citations(citations)
            
        else:
            response_text = response.content or "Tool call limit reached."

        candidate_records = self._create_candidates(content, session.id)

        assistant_message = self.store.add_message(
            session.id,
            "assistant",
            response_text,
            metadata={
                "recalled_memory_ids": [memory.id for memory in recalled_memories],
                "candidate_ids": [candidate.id for candidate in candidate_records],
                "tool_event_ids": [event.id for event in turn_tool_events],
                "citation_urls": [citation.url for citation in citations],
                "activated_skills": [skill.name for skill in activated_skills],
                "context_telemetry": to_plain_dict(context.telemetry),
            },
        )

        return AgentTurnResult(
            session=session,
            assistant_message=assistant_message,
            recalled_memories=recalled_memories,
            candidates=candidate_records,
            tool_events=turn_tool_events,
            citations=citations,
            activated_skills=activated_skills,
            context_telemetry=context.telemetry,
        )

    def _build_tools_schema(self) -> list[dict[str, object]]:
        schema = []
        for spec in self.tool_registry.list_specs():
            schema.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema
                }
            })
        return schema

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



    def _tool_context(self, session_id: str) -> ToolContext:
        cwd = Path.cwd().resolve()
        return ToolContext(
            session_id=session_id,
            workspace_root=cwd,
            home_dir=self.config.home_dir.resolve(),
            cwd=cwd,
        )



    def _persist_tool_facts(self, event: ToolEvent) -> None:
        if not event.ok:
            return

        if event.tool_name == "search_web":
            results = event.output.get("results", [])
            if not isinstance(results, list):
                return
            for index, item in enumerate(results[: self.config.context_tool_fact_limit], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                snippet = str(item.get("snippet", "")).strip()
                url = str(item.get("url", "")).strip()
                fact = " | ".join(part for part in [title, snippet, url] if part)
                if fact:
                    self.store.add_tool_fact(
                        session_id=event.session_id,
                        tool_event_id=event.id,
                        fact_type="search_result",
                        content=fact,
                        weight=max(0.1, 1.1 - (index * 0.1)),
                    )
            return

        if event.tool_name == "page_extract":
            url = str(event.output.get("url", "")).strip()
            title = str(event.output.get("title", "")).strip()
            text = str(event.output.get("text", "")).strip()
            fact = " | ".join(part for part in [title, text[:240], url] if part)
            if fact:
                self.store.add_tool_fact(
                    session_id=event.session_id,
                    tool_event_id=event.id,
                    fact_type="page_extract",
                    content=fact,
                    weight=1.0,
                )

    def _citations_for_event(self, event: ToolEvent) -> list[Citation]:
        if event.tool_name == "search_web" and self.search_service is not None:
            results = event.output.get("results", [])
            items = []
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        try:
                            items.append(SearchItem(**item))
                        except TypeError:
                            continue
            return self.search_service.citations_for(items)

        if event.tool_name == "page_extract":
            url = str(event.output.get("url", "")).strip()
            if not url:
                return []
            title = str(event.output.get("title", "")).strip() or url
            snippet = str(event.output.get("text", "")).strip()[:240]
            return [
                Citation(
                    id="s1",
                    label="S1",
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=event.source,
                )
            ]

        return []

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

    def list_tool_events(self, session_id: str, limit: int = 20):
        return self.store.list_tool_events(session_id, limit=limit)

    def list_installed_skills(self):
        return self.skill_manager.list_installed()

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
        self.config.model_temperature = config.model_temperature
        self.config.model_top_p = config.model_top_p
        self.config.model_max_tokens = config.model_max_tokens
        self.config.model_timeout_seconds = config.model_timeout_seconds
        self.config.model_retry_max_tokens = config.model_retry_max_tokens
        self.config.retrieval_top_k = config.retrieval_top_k
        self.config.context_messages = config.context_messages
        self.config.skills_dir = config.skills_dir
        self.config.context_max_input_tokens = config.context_max_input_tokens
        self.config.context_reserved_output_tokens = config.context_reserved_output_tokens
        self.config.context_summary_trigger_messages = config.context_summary_trigger_messages
        self.config.context_tool_fact_limit = config.context_tool_fact_limit
        self.config.context_policy = config.context_policy
        self.config.search_provider_order = config.search_provider_order
        self.config.search_top_k = config.search_top_k
        self.config.search_cache_ttl_hours = config.search_cache_ttl_hours
        self.config.search_locale = config.search_locale
        self.config.network_tools_enabled = config.network_tools_enabled
        self.config.shell_allowed_programs = config.shell_allowed_programs
        self.config.tool_max_output_chars = config.tool_max_output_chars

        self.chat_provider = build_chat_provider(self.config)
        self.context_builder = _build_context_builder(self.config, self.store)
        self.search_service = _build_search_service(self.config, self.store)
        self.tool_registry = _build_tool_registry(self.config, self.search_service)
        self.skill_manager = SkillManager(self.config.skills_dir, self.store)


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    unique: dict[str, Citation] = {}
    for citation in citations:
        key = citation.url.strip().lower() or citation.title.strip().lower()
        if key and key not in unique:
            unique[key] = citation

    normalized: list[Citation] = []
    for index, citation in enumerate(unique.values(), start=1):
        normalized.append(
            Citation(
                id=f"s{index}",
                label=f"S{index}",
                title=citation.title,
                url=citation.url,
                snippet=citation.snippet,
                source=citation.source,
            )
        )
    return normalized


def _format_citation_block(citations: list[Citation]) -> str:
    if not citations:
        return ""
    return "\n".join(
        f"- [{citation.label}] {citation.title} | {citation.url} | {citation.snippet[:180]}".strip()
        for citation in citations
    )
