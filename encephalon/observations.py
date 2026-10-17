"""Causal revision-aware buffer. No future revision may hide a visible old one."""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import ContractError, Observation, Scope, ValidityStatus, integer


class CapacityError(ContractError):
    """Explicit bounded-buffer backpressure, not silent loss of observations."""


@dataclass(frozen=True)
class ObservationDelta:
    observation: Observation
    change: str  # inserted or revised; stable observation_id is the dedup key
    valid_new_event: bool


class ObservationBuffer:
    def __init__(self, scope: Scope, max_records: int = 4096):
        self._scope = scope
        self.max_records = integer(max_records, "max_records", 1)
        self._records: dict[str, dict[int, Observation]] = {}
        self._consumed: dict[str, int] = {}
        self._last_now = 0
        self._stimulated: set[str] = set()

    @property
    def last_observed_ms(self) -> int:
        return self._last_now

    @property
    def scope(self) -> Scope:
        return self._scope

    def add(self, observation: Observation) -> bool:
        if observation.header.scope != self.scope:
            raise ContractError("observation belongs to a different persona/version/session")
        revisions = self._records.get(observation.observation_id, {})
        prior = revisions.get(observation.header.revision)
        if prior is not None:
            if prior.to_dict() != observation.to_dict():
                raise ContractError("same observation/revision has conflicting content")
            return False
        if sum(map(len, self._records.values())) >= self.max_records:
            raise CapacityError("observation buffer full; archive/prune explicitly before ingesting")
        for old in revisions.values():
            if (old.modality, old.header.event_start_ms, old.header.event_end_ms, old.header.source_ids) != (
                    observation.modality, observation.header.event_start_ms, observation.header.event_end_ms, observation.header.source_ids):
                raise ContractError("revisions must retain modality, source and event window identity")
            old_order = (old.header.revision > observation.header.revision) - (old.header.revision < observation.header.revision)
            time_order = (old.header.available_at_ms > observation.header.available_at_ms) - (old.header.available_at_ms < observation.header.available_at_ms)
            if old_order * time_order < 0:
                raise ContractError("higher revision cannot have earlier availability")
        self._records.setdefault(observation.observation_id, {})[observation.header.revision] = observation
        return True

    def _check_time(self, now_ms: int) -> None:
        integer(now_ms, "now_ms")
        if now_ms < self._last_now:
            raise ContractError("buffer cannot be read with a backward clock")
        self._last_now = now_ms

    def visible(self, now_ms: int) -> tuple[Observation, ...]:
        self._check_time(now_ms)
        selected = []
        for revisions in self._records.values():
            available = [obs for obs in revisions.values() if obs.header.available_at_ms <= now_ms]
            if available:
                selected.append(max(available, key=lambda obs: obs.header.revision))
        return tuple(sorted(selected, key=lambda obs: (obs.header.available_at_ms, obs.observation_id)))

    def consume(self, now_ms: int) -> tuple[ObservationDelta, ...]:
        changes = []
        for obs in self.visible(now_ms):
            previous = self._consumed.get(obs.observation_id)
            if previous is not None and previous >= obs.header.revision:
                continue
            is_new = obs.header.validity.status is ValidityStatus.VALID and obs.observation_id not in self._stimulated
            changes.append(ObservationDelta(obs, "inserted" if previous is None else "revised", is_new))
            if is_new:
                self._stimulated.add(obs.observation_id)
            self._consumed[obs.observation_id] = obs.header.revision
        return tuple(changes)

    def prune_before(self, available_before_ms: int) -> int:
        """Explicit retention; retain one latest consumed revision per event.

        Future/unconsumed records are never removed. Complete session history
        belongs in a durable archive, which stage 0 deliberately does not fake.
        """
        integer(available_before_ms, "available_before_ms")
        removed = 0
        for event_id, revisions in self._records.items():
            consumed_revision = self._consumed.get(event_id, -1)
            for revision, obs in list(revisions.items()):
                if revision < consumed_revision and obs.header.available_at_ms < available_before_ms:
                    del revisions[revision]
                    removed += 1
        return removed
