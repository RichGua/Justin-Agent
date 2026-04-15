from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ExtensionPoint:
    slot: str
    name: str
    description: str


class ExtensionRegistry:
    DEFAULT_SLOTS = ("planner", "subagent", "cron", "browser_automation", "tool")

    def __init__(self) -> None:
        self._handlers: dict[str, dict[str, Callable[..., Any]]] = {
            slot: {} for slot in self.DEFAULT_SLOTS
        }
        self._descriptions: dict[tuple[str, str], str] = {}

    def register(
        self,
        slot: str,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
    ) -> None:
        if slot not in self._handlers:
            self._handlers[slot] = {}
        self._handlers[slot][name] = handler
        self._descriptions[(slot, name)] = description

    def get(self, slot: str, name: str) -> Callable[..., Any] | None:
        return self._handlers.get(slot, {}).get(name)

    def list_extensions(self, slot: str | None = None) -> list[ExtensionPoint]:
        slots = [slot] if slot else list(self._handlers)
        points: list[ExtensionPoint] = []
        for current_slot in slots:
            for name in sorted(self._handlers.get(current_slot, {})):
                points.append(
                    ExtensionPoint(
                        slot=current_slot,
                        name=name,
                        description=self._descriptions.get((current_slot, name), ""),
                    )
                )
        return points
