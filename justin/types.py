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
class AgentTurnResult:
    session: Session
    assistant_message: Message
    recalled_memories: list[RetrievedMemory]
    candidates: list[MemoryCandidate]
    action_proposals: list[ActionProposal] = field(default_factory=list)


@dataclass(slots=True)
class ChatRequest:
    system_prompt: str
    conversation: list[dict[str, str]]
    memory_snippets: list[str]
    latest_user_message: str


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
