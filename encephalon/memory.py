"""Small bounded, source-preserving M+ ledger, not learned long-term memory.

No cross-session/persona sharing, no vector search, no automatic factual upgrade,
and no edits to initial canonical biography M0. Version history is retained.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from .contracts import ContractError, Header, Scope, ValidityStatus, integer, nonempty


class AssertionKind(str, Enum):
    OBSERVED = "observed"
    REPORTED = "reported"
    INFERRED = "inferred"
    HYPOTHETICAL = "hypothetical"
    CORRECTED = "corrected"


@dataclass(frozen=True)
class MemoryEvent:
    header: Header
    memory_id: str
    event_id: str
    summary: str
    assertion_kind: AssertionKind

    def __post_init__(self) -> None:
        if not isinstance(self.header, Header):
            raise ContractError("memory header must be Header")
        for name in ("memory_id", "event_id", "summary"):
            nonempty(getattr(self, name), name)
        try:
            object.__setattr__(self, "assertion_kind", AssertionKind(self.assertion_kind))
        except (ValueError, TypeError) as exc:
            raise ContractError("invalid assertion_kind") from exc
        if self.header.validity.status is not ValidityStatus.VALID:
            raise ContractError("cannot write invalid observation as a memory event")


class SessionMemory:
    def __init__(self, scope: Scope, max_versions: int = 2048):
        self._scope = scope
        self.max_versions = integer(max_versions, "max_versions", 1)
        self._events: dict[str, list[MemoryEvent]] = {}
        self._event_ids: dict[str, str] = {}
        self._last_write_ms = 0

    @property
    def last_observed_ms(self) -> int:
        return self._last_write_ms

    @property
    def scope(self) -> Scope:
        return self._scope

    def put(self, event: MemoryEvent, now_ms: int) -> bool:
        integer(now_ms, "now_ms")
        if now_ms < self._last_write_ms:
            raise ContractError("memory write clock cannot move backward")
        if event.header.scope != self.scope:
            raise ContractError("memory namespace mismatch")
        if event.header.available_at_ms > now_ms:
            raise ContractError("cannot remember a future/unavailable event")
        if event.event_id in self._event_ids and self._event_ids[event.event_id] != event.memory_id:
            raise ContractError("same source event cannot create duplicate memories")
        versions = self._events.get(event.memory_id, [])
        if versions:
            if versions[0].event_id != event.event_id:
                raise ContractError("memory_id cannot be reassigned to a different event")
            for old in versions:
                if old.header.revision == event.header.revision:
                    if old != event:
                        raise ContractError("same memory revision has conflicting content")
                    return False
            previous = max(versions, key=lambda e: e.header.revision)
            if event.header.revision <= previous.header.revision or event.header.available_at_ms < previous.header.available_at_ms:
                raise ContractError("memory correction must advance revision and not reverse availability")
            if not set(previous.header.source_ids + previous.header.provenance) <= set(event.header.source_ids + event.header.provenance):
                raise ContractError("memory correction lost original provenance")
        if sum(map(len, self._events.values())) >= self.max_versions:
            raise ContractError("memory ledger full; persist/archive explicitly")
        self._events.setdefault(event.memory_id, []).append(event)
        self._event_ids[event.event_id] = event.memory_id
        self._last_write_ms = now_ms
        return True

    def get(self, memory_id: str, *, scope: Scope, now_ms: int) -> MemoryEvent | None:
        integer(now_ms, "now_ms")
        if scope != self.scope:
            raise ContractError("memory access requires exact persona/version/session scope")
        visible = [e for e in self._events.get(memory_id, []) if e.header.available_at_ms <= now_ms]
        return max(visible, key=lambda e: e.header.revision) if visible else None

    def history(self, memory_id: str, *, scope: Scope, now_ms: int) -> tuple[MemoryEvent, ...]:
        self.get(memory_id, scope=scope, now_ms=now_ms)  # validate scope/time
        return tuple(e for e in self._events.get(memory_id, []) if e.header.available_at_ms <= now_ms)

    def snapshot(self, *, scope: Scope, now_ms: int) -> tuple[MemoryEvent, ...]:
        """All visible versions, for explicit transactional archive/export."""
        integer(now_ms, "now_ms")
        if scope != self.scope:
            raise ContractError("memory snapshot scope mismatch")
        return tuple(event for memory_id in sorted(self._events) for event in self._events[memory_id]
                     if event.header.available_at_ms <= now_ms)
