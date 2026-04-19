from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .embeddings import LocalHashEmbeddingProvider, cosine_similarity, tokenize
from .types import (
    CandidateStatus,
    InstalledSkill,
    MemoryCandidate,
    MemoryRecord,
    Message,
    RetrievedMemory,
    Session,
    SessionSummary,
    StoredToolFact,
    ToolEvent,
    now_iso,
    to_plain_dict,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


class AgentStore:
    def __init__(self, database_path: Path, embedder: LocalHashEmbeddingProvider) -> None:
        self.database_path = database_path
        self.embedder = embedder
        self._lock = threading.RLock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.database_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._enable_pragmas()
        self._initialize_schema()

    def _enable_pragmas(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    def _initialize_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_candidates (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                    status TEXT NOT NULL,
                    review_note TEXT,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS action_proposals (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    arguments_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL,
                    source_message_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_events (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    output_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    error TEXT,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'builtin',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_facts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    tool_event_id TEXT REFERENCES tool_events(id) ON DELETE SET NULL,
                    fact_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS search_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    query TEXT NOT NULL,
                    locale TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS installed_skills (
                    name TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    source TEXT NOT NULL,
                    install_path TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    description TEXT NOT NULL,
                    entry TEXT NOT NULL,
                    triggers_json TEXT NOT NULL DEFAULT '[]',
                    required_tools_json TEXT NOT NULL DEFAULT '[]',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    installed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_fts()

    def _ensure_fts(self) -> None:
        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    content,
                    summary,
                    tags,
                    tokenize = 'unicode61'
                );
                """
            )
        except sqlite3.OperationalError:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories_fts_fallback (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags TEXT NOT NULL
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create_session(self, title: str) -> Session:
        timestamp = now_iso()
        session = Session(id=_new_id("sess"), title=title, created_at=timestamp, updated_at=timestamp)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session.id, session.title, session.created_at, session.updated_at),
            )
        return session

    def get_session(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[Session]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def add_message(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> Message:
        timestamp = now_iso()
        message = Message(
            id=_new_id("msg"),
            session_id=session_id,
            role=role,
            content=content,
            created_at=timestamp,
            metadata=metadata or {},
        )
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    message.role,
                    message.content,
                    json.dumps(message.metadata, ensure_ascii=False),
                    message.created_at,
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (timestamp, session_id),
            )
        return message

    def list_messages(self, session_id: str, limit: int = 100) -> list[Message]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def delete_old_messages(self, session_id: str, keep_latest: int = 2) -> None:
        with self._lock, self._conn:
            rows = self._conn.execute(
                """
                SELECT id FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT -1 OFFSET ?
                """,
                (session_id, keep_latest),
            ).fetchall()
            
            if rows:
                ids = [row["id"] for row in rows]
                placeholders = ",".join(["?"] * len(ids))
                self._conn.execute(
                    f"DELETE FROM messages WHERE id IN ({placeholders})",
                    ids
                )

    def create_candidate(
        self,
        kind: str,
        content: str,
        evidence: str,
        confidence: float,
        source_session_id: str | None,
    ) -> MemoryCandidate:
        existing = self.find_duplicate_candidate_or_memory(kind, content)
        if existing is not None:
            return existing

        timestamp = now_iso()
        candidate = MemoryCandidate(
            id=_new_id("cand"),
            kind=kind,
            content=content,
            evidence=evidence,
            confidence=confidence,
            source_session_id=source_session_id,
            status=CandidateStatus.PENDING,
            created_at=timestamp,
            updated_at=timestamp,
        )
        embedding = self.embedder.embed(content)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO memory_candidates (
                    id, kind, content, evidence, confidence, source_session_id, status,
                    review_note, embedding_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.kind,
                    candidate.content,
                    candidate.evidence,
                    candidate.confidence,
                    candidate.source_session_id,
                    candidate.status,
                    json.dumps(embedding),
                    candidate.created_at,
                    candidate.updated_at,
                ),
            )
        return candidate

    def list_candidates(self, status: str | None = None, limit: int = 100) -> list[MemoryCandidate]:
        query = "SELECT * FROM memory_candidates"
        params: list[object] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def confirm_candidate(self, candidate_id: str) -> MemoryRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown candidate: {candidate_id}")

        candidate = self._row_to_candidate(row)
        timestamp = now_iso()
        memory = MemoryRecord(
            id=_new_id("mem"),
            kind=candidate.kind,
            content=candidate.content,
            summary=self._summarize(candidate.content),
            confidence=candidate.confidence,
            source_session_id=candidate.source_session_id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        embedding = self.embedder.embed(memory.content)

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO memories (
                    id, kind, content, summary, confidence, source_session_id,
                    tags_json, embedding_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.kind,
                    memory.content,
                    memory.summary,
                    memory.confidence,
                    memory.source_session_id,
                    json.dumps(embedding),
                    memory.created_at,
                    memory.updated_at,
                ),
            )
            self._conn.execute(
                """
                UPDATE memory_candidates
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (CandidateStatus.APPROVED, timestamp, candidate_id),
            )
            self._sync_memory_fts(memory.id, memory.content, memory.summary, [])
        return memory

    def reject_candidate(self, candidate_id: str, review_note: str | None = None) -> MemoryCandidate:
        timestamp = now_iso()
        with self._lock, self._conn:
            updated = self._conn.execute(
                """
                UPDATE memory_candidates
                SET status = ?, review_note = ?, updated_at = ?
                WHERE id = ?
                """,
                (CandidateStatus.REJECTED, review_note, timestamp, candidate_id),
            )
            if updated.rowcount == 0:
                raise KeyError(f"Unknown candidate: {candidate_id}")
            row = self._conn.execute(
                "SELECT * FROM memory_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
        return self._row_to_candidate(row)

    def list_memories(self, kind: str | None = None, limit: int = 100) -> list[MemoryRecord]:
        query = "SELECT * FROM memories"
        params: list[object] = []
        if kind:
            query += " WHERE kind = ?"
            params.append(kind)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_memories(self, query: str, top_k: int = 5, kinds: list[str] | None = None) -> list[RetrievedMemory]:
        base_memories = self.list_memories(limit=500)
        if kinds:
            base_memories = [memory for memory in base_memories if memory.kind in kinds]
        if not query.strip():
            return [
                RetrievedMemory(
                    id=memory.id,
                    kind=memory.kind,
                    content=memory.content,
                    summary=memory.summary,
                    confidence=memory.confidence,
                    score=0.0,
                    source_session_id=memory.source_session_id,
                    tags=memory.tags,
                )
                for memory in base_memories[:top_k]
            ]

        lexical_scores = self._fts_scores(query)
        query_embedding = self.embedder.embed(query)
        combined: dict[str, RetrievedMemory] = {}
        for memory in base_memories:
            semantic_score = cosine_similarity(query_embedding, self._embedding_for_memory(memory.id))
            lexical_score = max(lexical_scores.get(memory.id, 0.0), self._manual_keyword_score(query, memory))
            score = (semantic_score * 0.6) + (lexical_score * 0.4)
            if score <= 0:
                continue
            combined[memory.id] = RetrievedMemory(
                id=memory.id,
                kind=memory.kind,
                content=memory.content,
                summary=memory.summary,
                confidence=memory.confidence,
                score=score,
                source_session_id=memory.source_session_id,
                tags=memory.tags,
            )
        ranked = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    def find_duplicate_candidate_or_memory(self, kind: str, content: str) -> MemoryCandidate | None:
        normalized = content.strip().lower()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM memory_candidates
                WHERE lower(content) = ? AND kind = ? AND status = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (normalized, kind, CandidateStatus.PENDING),
            ).fetchone()
            if row is not None:
                return self._row_to_candidate(row)

            row = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE lower(content) = ? AND kind = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (normalized, kind),
            ).fetchone()
        if row is not None:
            memory = self._row_to_memory(row)
            return MemoryCandidate(
                id=memory.id,
                kind=memory.kind,
                content=memory.content,
                evidence="already approved memory",
                confidence=memory.confidence,
                source_session_id=memory.source_session_id,
                status=CandidateStatus.APPROVED,
                created_at=memory.created_at,
                updated_at=memory.updated_at,
            )
        return None

    def upsert_session_summary(self, session_id: str, summary: str, source_message_count: int) -> SessionSummary:
        record = SessionSummary(
            session_id=session_id,
            summary=summary,
            source_message_count=source_message_count,
            updated_at=now_iso(),
        )
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO session_summaries (session_id, summary, source_message_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id)
                DO UPDATE SET summary = excluded.summary,
                              source_message_count = excluded.source_message_count,
                              updated_at = excluded.updated_at
                """,
                (
                    record.session_id,
                    record.summary,
                    record.source_message_count,
                    record.updated_at,
                ),
            )
        return record

    def get_session_summary(self, session_id: str) -> SessionSummary | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_session_summary(row) if row else None

    def add_tool_event(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, object],
        ok: bool,
        output: dict[str, object],
        summary: str,
        error: str | None,
        latency_ms: int,
        source: str = "builtin",
    ) -> ToolEvent:
        event = ToolEvent(
            id=_new_id("tool"),
            session_id=session_id,
            tool_name=tool_name,
            arguments={str(key): value for key, value in arguments.items()},
            ok=ok,
            output=to_plain_dict(output),
            summary=summary,
            error=error,
            latency_ms=int(latency_ms),
            source=source,
            created_at=now_iso(),
        )
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO tool_events (
                    id, session_id, tool_name, arguments_json, ok, output_json, summary,
                    error, latency_ms, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.session_id,
                    event.tool_name,
                    json.dumps(event.arguments, ensure_ascii=False),
                    1 if event.ok else 0,
                    json.dumps(event.output, ensure_ascii=False),
                    event.summary,
                    event.error,
                    event.latency_ms,
                    event.source,
                    event.created_at,
                ),
            )
        return event

    def list_tool_events(self, session_id: str, limit: int = 20) -> list[ToolEvent]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM tool_events
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [self._row_to_tool_event(row) for row in rows]

    def add_tool_fact(
        self,
        session_id: str,
        tool_event_id: str | None,
        fact_type: str,
        content: str,
        weight: float = 1.0,
    ) -> StoredToolFact:
        record = StoredToolFact(
            id=_new_id("fact"),
            session_id=session_id,
            tool_event_id=tool_event_id,
            fact_type=fact_type,
            content=content,
            weight=weight,
            created_at=now_iso(),
        )
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO tool_facts (id, session_id, tool_event_id, fact_type, content, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.session_id,
                    record.tool_event_id,
                    record.fact_type,
                    record.content,
                    record.weight,
                    record.created_at,
                ),
            )
        return record

    def search_tool_facts(self, query: str, session_id: str | None = None, top_k: int = 5) -> list[StoredToolFact]:
        sql = "SELECT * FROM tool_facts"
        params: list[object] = []
        if session_id:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at DESC LIMIT 100"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        facts = [self._row_to_tool_fact(row) for row in rows]
        if not query.strip():
            return facts[:top_k]
        query_tokens = tokenize(query)
        scored: list[tuple[float, StoredToolFact]] = []
        for fact in facts:
            haystack = fact.content.lower()
            matched = sum(1 for token in query_tokens if token and token in haystack)
            if matched <= 0:
                continue
            score = (matched / max(len(query_tokens), 1)) + fact.weight
            scored.append((score, fact))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [pair[1] for pair in scored[:top_k]]

    def save_search_cache(
        self,
        cache_key: str,
        provider: str,
        query: str,
        locale: str,
        results: list[object],
        ttl_hours: int,
    ) -> None:
        created_at = _utc_now()
        expires_at = created_at + timedelta(hours=max(int(ttl_hours), 1))
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO search_cache (cache_key, provider, query, locale, results_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key)
                DO UPDATE SET results_json = excluded.results_json,
                              created_at = excluded.created_at,
                              expires_at = excluded.expires_at
                """,
                (
                    cache_key,
                    provider,
                    query,
                    locale,
                    json.dumps(to_plain_dict(results), ensure_ascii=False),
                    created_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )

    def get_search_cache(self, cache_key: str) -> list[dict[str, object]] | None:
        now = _utc_now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM search_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at <= now:
            with self._lock, self._conn:
                self._conn.execute("DELETE FROM search_cache WHERE cache_key = ?", (cache_key,))
            return None
        return json.loads(row["results_json"])

    def save_installed_skill(self, skill: InstalledSkill) -> InstalledSkill:
        timestamp = now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO installed_skills (
                    name, version, source, install_path, summary, description, entry,
                    triggers_json, required_tools_json, tags_json, installed_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name)
                DO UPDATE SET version = excluded.version,
                              source = excluded.source,
                              install_path = excluded.install_path,
                              summary = excluded.summary,
                              description = excluded.description,
                              entry = excluded.entry,
                              triggers_json = excluded.triggers_json,
                              required_tools_json = excluded.required_tools_json,
                              tags_json = excluded.tags_json,
                              updated_at = excluded.updated_at
                """,
                (
                    skill.name,
                    skill.version,
                    skill.source,
                    skill.install_path,
                    skill.summary,
                    skill.description,
                    skill.entry,
                    json.dumps(skill.triggers, ensure_ascii=False),
                    json.dumps(skill.required_tools, ensure_ascii=False),
                    json.dumps(skill.tags, ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
        return skill

    def list_installed_skills(self) -> list[InstalledSkill]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM installed_skills ORDER BY updated_at DESC, name ASC"
            ).fetchall()
        return [self._row_to_installed_skill(row) for row in rows]

    def get_installed_skill(self, name: str) -> InstalledSkill | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM installed_skills WHERE name = ?",
                (name,),
            ).fetchone()
        return self._row_to_installed_skill(row) if row else None

    def remove_installed_skill(self, name: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM installed_skills WHERE name = ?", (name,))

    def _sync_memory_fts(self, memory_id: str, content: str, summary: str, tags: list[str]) -> None:
        try:
            self._conn.execute(
                "INSERT INTO memories_fts(memory_id, content, summary, tags) VALUES (?, ?, ?, ?)",
                (memory_id, content, summary, " ".join(tags)),
            )
        except sqlite3.OperationalError:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories_fts_fallback(memory_id, content, summary, tags)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, content, summary, " ".join(tags)),
            )

    def _fts_scores(self, query: str) -> dict[str, float]:
        tokens = [token for token in query.replace('"', " ").split() if token]
        if not tokens:
            return {}

        search_query = " OR ".join(tokens)
        try:
            rows = self._conn.execute(
                """
                SELECT memory_id, bm25(memories_fts) AS rank
                FROM memories_fts
                WHERE memories_fts MATCH ?
                ORDER BY rank ASC
                LIMIT 20
                """,
                (search_query,),
            ).fetchall()
            return {str(row["memory_id"]): 1 / (1 + abs(float(row["rank"]))) for row in rows}
        except sqlite3.OperationalError:
            if not self._table_exists("memories_fts_fallback"):
                return {}
            like_term = f"%{query.lower()}%"
            rows = self._conn.execute(
                """
                SELECT memory_id
                FROM memories_fts_fallback
                WHERE lower(content) LIKE ? OR lower(summary) LIKE ? OR lower(tags) LIKE ?
                LIMIT 20
                """,
                (like_term, like_term, like_term),
            ).fetchall()
            return {str(row["memory_id"]): 0.75 for row in rows}

    def _embedding_for_memory(self, memory_id: str) -> list[float]:
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding_json FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return []
        return json.loads(row["embedding_json"])

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    def _row_to_candidate(self, row: sqlite3.Row) -> MemoryCandidate:
        return MemoryCandidate(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            evidence=row["evidence"],
            confidence=float(row["confidence"]),
            source_session_id=row["source_session_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            review_note=row["review_note"],
        )

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            summary=row["summary"],
            confidence=float(row["confidence"]),
            source_session_id=row["source_session_id"],
            tags=json.loads(row["tags_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_session_summary(self, row: sqlite3.Row) -> SessionSummary:
        return SessionSummary(
            session_id=row["session_id"],
            summary=row["summary"],
            source_message_count=int(row["source_message_count"]),
            updated_at=row["updated_at"],
        )

    def _row_to_tool_event(self, row: sqlite3.Row) -> ToolEvent:
        return ToolEvent(
            id=row["id"],
            session_id=row["session_id"],
            tool_name=row["tool_name"],
            arguments=json.loads(row["arguments_json"]),
            ok=bool(row["ok"]),
            output=json.loads(row["output_json"]),
            summary=row["summary"],
            error=row["error"],
            latency_ms=int(row["latency_ms"]),
            source=row["source"],
            created_at=row["created_at"],
        )

    def _row_to_tool_fact(self, row: sqlite3.Row) -> StoredToolFact:
        return StoredToolFact(
            id=row["id"],
            session_id=row["session_id"],
            tool_event_id=row["tool_event_id"],
            fact_type=row["fact_type"],
            content=row["content"],
            weight=float(row["weight"]),
            created_at=row["created_at"],
        )

    def _row_to_installed_skill(self, row: sqlite3.Row) -> InstalledSkill:
        return InstalledSkill(
            name=row["name"],
            version=row["version"],
            source=row["source"],
            install_path=row["install_path"],
            summary=row["summary"],
            description=row["description"],
            entry=row["entry"],
            triggers=json.loads(row["triggers_json"]),
            required_tools=json.loads(row["required_tools_json"]),
            tags=json.loads(row["tags_json"]),
        )

    def _summarize(self, content: str) -> str:
        if len(content) <= 80:
            return content
        return content[:77] + "..."

    def _manual_keyword_score(self, query: str, memory: MemoryRecord) -> float:
        haystack = f"{memory.content} {memory.summary}".lower()
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0
        matched = sum(1 for token in query_tokens if token and token in haystack)
        if matched == 0:
            return 0.0
        return matched / len(query_tokens)

    def _table_exists(self, table_name: str) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None
