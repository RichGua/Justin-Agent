from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class MemoryKind(StrEnum):
    FACT = "fact"
    IDENTITY = "identity"
    PREFERENCE = "preference"
    GOAL = "goal"
    PROJECT = "project"


class CandidateStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(slots=True)
class Session:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(slots=True)
class Message:
    id: str
    session_id: str
    role: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedMemory:
    id: str
    kind: str
    content: str
    summary: str
    confidence: float
    score: float
    source_session_id: str | None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryCandidate:
    id: str
    kind: str
    content: str
    evidence: str
    confidence: float
    source_session_id: str | None
    status: str
    created_at: str
    updated_at: str
    review_note: str | None = None


@dataclass(slots=True)
class MemoryRecord:
    id: str
    kind: str
    content: str
    summary: str
    confidence: float
    source_session_id: str | None
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class ActionProposal:
    id: str
    label: str
    risk_level: str
    status: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    id: str
    label: str
    title: str
    url: str
    snippet: str
    source: str


@dataclass(slots=True)
class ToolEvent:
    id: str
    session_id: str
    tool_name: str
    arguments: dict[str, Any]
    ok: bool
    output: dict[str, Any]
    summary: str
    error: str | None
    latency_ms: int
    source: str
    created_at: str


@dataclass(slots=True)
class StoredToolFact:
    id: str
    session_id: str
    tool_event_id: str | None
    fact_type: str
    content: str
    weight: float
    created_at: str


@dataclass(slots=True)
class SessionSummary:
    session_id: str
    summary: str
    source_message_count: int
    updated_at: str


@dataclass(slots=True)
class InstalledSkill:
    name: str
    version: str
    source: str
    install_path: str
    summary: str
    description: str
    entry: str
    triggers: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ContextTelemetry:
    input_tokens_estimate: int
    output_tokens_estimate: int
    context_tokens_before: int
    context_tokens_after: int
    saved_tokens: int
    tool_tokens_saved: int = 0


@dataclass(slots=True)
class AgentTurnResult:
    session: Session
    assistant_message: Message
    recalled_memories: list[RetrievedMemory]
    candidates: list[MemoryCandidate]
    action_proposals: list[ActionProposal] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    activated_skills: list[InstalledSkill] = field(default_factory=list)
    context_telemetry: ContextTelemetry | None = None


@dataclass(slots=True)
class ChatRequest:
    system_prompt: str
    conversation: list[dict[str, str]]
    memory_snippets: list[str]
    latest_user_message: str
    citations: list[Citation] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    activated_skills: list[InstalledSkill] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value
