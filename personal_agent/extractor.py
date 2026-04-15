from __future__ import annotations

import re
from dataclasses import dataclass

from .types import MemoryKind


@dataclass(slots=True)
class CandidateDraft:
    kind: str
    content: str
    evidence: str
    confidence: float


class HeuristicMemoryExtractor:
    """Extract stable memory candidates from direct self-description."""

    def extract(self, text: str) -> list[CandidateDraft]:
        content = text.strip()
        if not content:
            return []

        candidates: list[CandidateDraft] = []
        candidates.extend(self._extract_explicit_memory_requests(content))
        candidates.extend(self._extract_identity(content))
        candidates.extend(self._extract_preferences(content))
        candidates.extend(self._extract_goals(content))
        candidates.extend(self._extract_projects(content))

        unique: dict[tuple[str, str], CandidateDraft] = {}
        for candidate in candidates:
            key = (candidate.kind, candidate.content.strip().lower())
            unique[key] = candidate
        deduped = list(unique.values())

        specific_candidates = [candidate for candidate in deduped if candidate.kind != MemoryKind.FACT]
        if not specific_candidates:
            return deduped

        filtered: list[CandidateDraft] = []
        for candidate in deduped:
            if candidate.kind != MemoryKind.FACT:
                filtered.append(candidate)
                continue
            overlaps = any(
                candidate.content.replace(" ", "").endswith(other.content.replace(" ", ""))
                or other.content.replace(" ", "") in candidate.content.replace(" ", "")
                for other in specific_candidates
            )
            if not overlaps:
                filtered.append(candidate)
        return filtered

    def _extract_explicit_memory_requests(self, text: str) -> list[CandidateDraft]:
        patterns = [
            (MemoryKind.FACT, re.compile(r"记住(?P<body>.+)", re.IGNORECASE)),
            (MemoryKind.FACT, re.compile(r"remember(?:\s+that)?\s+(?P<body>.+)", re.IGNORECASE)),
        ]
        matches: list[CandidateDraft] = []
        for kind, pattern in patterns:
            found = pattern.search(text)
            if not found:
                continue
            body = found.group("body").strip(" 。.!?，,")
            lowered = body.lower()
            if lowered.startswith(("我喜欢", "我不喜欢", "我偏好", "i like ", "i don't like ", "i prefer ")):
                continue
            if body:
                matches.append(CandidateDraft(kind=kind, content=body, evidence=text, confidence=0.95))
        return matches

    def _extract_identity(self, text: str) -> list[CandidateDraft]:
        patterns = [
            re.compile(r"我叫(?P<body>[^，。.!?]+)"),
            re.compile(r"我是(?P<body>[^，。.!?]+)"),
            re.compile(r"\bmy name is (?P<body>[^,.!?]+)", re.IGNORECASE),
            re.compile(r"\bi am (?P<body>[^,.!?]+)", re.IGNORECASE),
        ]
        return self._capture(patterns, MemoryKind.IDENTITY, text, confidence=0.86)

    def _extract_preferences(self, text: str) -> list[CandidateDraft]:
        patterns = [
            re.compile(r"我喜欢(?P<body>[^，。.!?]+)"),
            re.compile(r"我不喜欢(?P<body>[^，。.!?]+)"),
            re.compile(r"我偏好(?P<body>[^，。.!?]+)"),
            re.compile(r"\bi prefer (?P<body>[^,.!?]+)", re.IGNORECASE),
            re.compile(r"\bi like (?P<body>[^,.!?]+)", re.IGNORECASE),
            re.compile(r"\bi don't like (?P<body>[^,.!?]+)", re.IGNORECASE),
        ]
        results: list[CandidateDraft] = []
        for pattern in patterns:
            found = pattern.search(text)
            if not found:
                continue
            body = found.group("body").strip()
            if not body:
                continue
            if "不喜欢" in pattern.pattern or "don't like" in pattern.pattern:
                prefix = "不喜欢 "
            elif "偏好" in pattern.pattern or "prefer" in pattern.pattern:
                prefix = "偏好 "
            else:
                prefix = "喜欢 "
            results.append(
                CandidateDraft(
                    kind=MemoryKind.PREFERENCE,
                    content=f"{prefix}{body}".strip(),
                    evidence=text,
                    confidence=0.84,
                )
            )
        return results

    def _extract_goals(self, text: str) -> list[CandidateDraft]:
        patterns = [
            re.compile(r"我的目标是(?P<body>[^，。.!?]+)"),
            re.compile(r"我想要(?P<body>[^，。.!?]+)"),
            re.compile(r"\bmy goal is (?P<body>[^,.!?]+)", re.IGNORECASE),
            re.compile(r"\bi want to (?P<body>[^,.!?]+)", re.IGNORECASE),
        ]
        return self._capture(patterns, MemoryKind.GOAL, text, confidence=0.8)

    def _extract_projects(self, text: str) -> list[CandidateDraft]:
        patterns = [
            re.compile(r"我在做(?P<body>[^，。.!?]+)"),
            re.compile(r"我正在开发(?P<body>[^，。.!?]+)"),
            re.compile(r"\bi am working on (?P<body>[^,.!?]+)", re.IGNORECASE),
            re.compile(r"\bmy project is (?P<body>[^,.!?]+)", re.IGNORECASE),
        ]
        return self._capture(patterns, MemoryKind.PROJECT, text, confidence=0.82)

    def _capture(
        self,
        patterns: list[re.Pattern[str]],
        kind: str,
        text: str,
        confidence: float,
    ) -> list[CandidateDraft]:
        results: list[CandidateDraft] = []
        for pattern in patterns:
            found = pattern.search(text)
            if not found:
                continue
            body = found.group("body").strip()
            if len(body) <= 1:
                continue
            results.append(CandidateDraft(kind=kind, content=body, evidence=text, confidence=confidence))
        return results
