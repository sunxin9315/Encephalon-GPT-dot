"""Strict canonical runtime contracts. Times are session-monotonic milliseconds.

README in docs/design is authoritative over older examples. Canonical validity
is an object with status valid/missing/stale/corrupt, quality and age_ms. The age
is measured at available_at_ms, never a prediction of future availability.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA_VERSION = "encephalon.runtime.v1"


class ContractError(ValueError):
    """A supplied record violates a public data contract."""


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    return value


def strings(value: Any, name: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{name} must be a list of strings")
    result = tuple(nonempty(x, name) for x in value)
    if len(set(result)) != len(result) or (not allow_empty and not result):
        raise ContractError(f"{name} must contain unique strings" + (" and not be empty" if not allow_empty else ""))
    return result


def freeze_json(value: Any) -> Any:
    """Validate and detach JSON data, recursively making it immutable."""
    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise ContractError("JSON object keys must be strings")
        return MappingProxyType({k: freeze_json(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(v) for v in value)
    raise ContractError("payload must be finite JSON, without bytes or tensor objects")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: thaw_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(v) for v in value]
    return value


def exact_keys(value: Mapping[str, Any], required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, Mapping):
        raise ContractError("record must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing or unknown:
        raise ContractError(f"missing fields: {sorted(missing)}; unknown fields: {sorted(unknown)}")


class ValidityStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    STALE = "stale"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class Validity:
    status: ValidityStatus
    quality: float
    age_ms: int
    reason: str | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", ValidityStatus(self.status))
        except (ValueError, TypeError) as exc:
            raise ContractError("validity.status must be valid/missing/stale/corrupt") from exc
        if type(self.quality) not in (int, float) or not math.isfinite(self.quality) or not 0 <= self.quality <= 1:
            raise ContractError("validity.quality must be a finite number in [0, 1]")
        integer(self.age_ms, "validity.age_ms")
        if self.reason is not None:
            nonempty(self.reason, "validity.reason")
        if self.status is not ValidityStatus.VALID and self.reason is None:
            raise ContractError("non-valid data requires a specific reason")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Validity:
        exact_keys(data, {"status", "quality", "age_ms"}, {"reason"})
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "quality": self.quality, "age_ms": self.age_ms, "reason": self.reason}


def normalize_legacy_validity(status: str, *, quality: float, age_ms: int,
                              reason: str | None = None, failed_as: str | None = None) -> Validity:
    """Explicit opt-in migration; failed is ambiguous and never silently mapped."""
    if status == "present":
        status = "valid"
    elif status == "failed":
        if failed_as not in ("missing", "corrupt") or not reason:
            raise ContractError("legacy failed requires failed_as=missing/corrupt and a reason")
        status = failed_as
    return Validity(status, quality, age_ms, reason)


@dataclass(frozen=True)
class Scope:
    persona_id: str
    persona_version: str
    session_id: str

    def __post_init__(self) -> None:
        for item in fields(self):
            nonempty(getattr(self, item.name), item.name)

    def to_dict(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class Header:
    persona_id: str
    persona_version: str
    session_id: str
    state_version: int
    event_start_ms: int
    event_end_ms: int
    available_at_ms: int
    source_ids: tuple[str, ...]
    provenance: tuple[str, ...]
    validity: Validity
    revision: int = 0
    request_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        Scope(self.persona_id, self.persona_version, self.session_id)
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported schema_version: {self.schema_version}")
        for name in ("state_version", "event_start_ms", "event_end_ms", "available_at_ms", "revision"):
            integer(getattr(self, name), name)
        if not self.event_start_ms <= self.event_end_ms <= self.available_at_ms:
            raise ContractError("require event_start_ms <= event_end_ms <= available_at_ms")
        if not isinstance(self.validity, Validity):
            raise ContractError("validity must be a validated Validity object")
        if self.validity.age_ms != self.available_at_ms - self.event_end_ms:
            raise ContractError("validity.age_ms must equal available_at_ms - event_end_ms")
        object.__setattr__(self, "source_ids", strings(self.source_ids, "source_ids", False))
        object.__setattr__(self, "provenance", strings(self.provenance, "provenance"))
        if self.request_id is not None:
            nonempty(self.request_id, "request_id")

    @property
    def scope(self) -> Scope:
        return Scope(self.persona_id, self.persona_version, self.session_id)

    def age_at(self, now_ms: int) -> int:
        integer(now_ms, "now_ms")
        if now_ms < self.available_at_ms:
            raise ContractError("cannot inspect age of unavailable input")
        return now_ms - self.event_end_ms

    def to_dict(self) -> dict[str, Any]:
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        result.update(source_ids=list(self.source_ids), provenance=list(self.provenance), validity=self.validity.to_dict())
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Header:
        exact_keys(data, {f.name for f in fields(cls)})
        clean = dict(data)
        clean["validity"] = Validity.from_dict(clean["validity"])
        return cls(**clean)


@dataclass(frozen=True)
class Observation:
    header: Header
    observation_id: str
    modality: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.header, Header):
            raise ContractError("header must be a Header")
        nonempty(self.observation_id, "observation_id")
        if not isinstance(self.modality, str) or self.modality not in {"text", "audio", "speech_semantic", "vision", "environment", "speaker", "fast_acoustic", "retrieval"}:
            raise ContractError("unsupported observation modality")
        if not isinstance(self.payload, Mapping):
            raise ContractError("payload must be an object")
        object.__setattr__(self, "payload", freeze_json(self.payload))
        if self.header.request_id is not None:
            raise ContractError("ordinary Observation request_id must be null; use AsyncResult for replies")

    def to_dict(self) -> dict[str, Any]:
        return {**self.header.to_dict(), "record_type": "observation", "observation_id": self.observation_id,
                "modality": self.modality, "payload": thaw_json(self.payload)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Observation:
        names = {"record_type", "observation_id", "modality", "payload"}
        exact_keys(data, {f.name for f in fields(Header)} | names)
        if data["record_type"] != "observation":
            raise ContractError("record_type must be observation")
        return cls(Header.from_dict({k: v for k, v in data.items() if k not in names}),
                   data["observation_id"], data["modality"], data["payload"])


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    PENDING = "pending"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Claim:
    header: Header
    claim_id: str
    text: str
    evidence_status: EvidenceStatus
    evidence_ids: tuple[str, ...]
    requires_cloud: bool
    valid_until_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.header, Header):
            raise ContractError("header must be a Header")
        nonempty(self.claim_id, "claim_id")
        nonempty(self.text, "text")
        try:
            object.__setattr__(self, "evidence_status", EvidenceStatus(self.evidence_status))
        except (ValueError, TypeError) as exc:
            raise ContractError("unsupported evidence_status") from exc
        object.__setattr__(self, "evidence_ids", strings(self.evidence_ids, "evidence_ids"))
        if type(self.requires_cloud) is not bool:
            raise ContractError("requires_cloud must be bool")
        integer(self.valid_until_ms, "valid_until_ms")
        if self.valid_until_ms < self.header.available_at_ms:
            raise ContractError("claim expired before availability")
        if self.evidence_status is EvidenceStatus.SUPPORTED and not self.evidence_ids:
            raise ContractError("supported claim requires evidence_ids; transport success is not evidence")

    def can_assert(self, now_ms: int) -> bool:
        integer(now_ms, "now_ms")
        return (self.header.available_at_ms <= now_ms <= self.valid_until_ms
                and self.header.validity.status is ValidityStatus.VALID
                and self.evidence_status is EvidenceStatus.SUPPORTED)


@dataclass(frozen=True)
class AsyncRequest:
    header: Header
    request_version: int
    question_version: int
    claim_ids: tuple[str, ...]
    premise_versions: Mapping[str, int]
    valid_until_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.header, Header) or self.header.request_id is None:
            raise ContractError("AsyncRequest needs Header.request_id")
        for name in ("request_version", "question_version", "valid_until_ms"):
            integer(getattr(self, name), name)
        if self.valid_until_ms < self.header.available_at_ms:
            raise ContractError("request expiry precedes availability")
        object.__setattr__(self, "claim_ids", strings(self.claim_ids, "claim_ids", False))
        if not isinstance(self.premise_versions, Mapping):
            raise ContractError("premise_versions must be an object")
        for key, value in self.premise_versions.items():
            nonempty(key, "premise ID")
            integer(value, "premise version")
        object.__setattr__(self, "premise_versions", freeze_json(self.premise_versions))


@dataclass(frozen=True)
class AsyncResult:
    header: Header
    request_version: int
    question_version: int
    claim_ids: tuple[str, ...]
    premise_versions: Mapping[str, int]
    valid_until_ms: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Validation shares precisely the correlation contract with the request.
        request = AsyncRequest(self.header, self.request_version, self.question_version,
                               self.claim_ids, self.premise_versions, self.valid_until_ms)
        object.__setattr__(self, "claim_ids", request.claim_ids)
        object.__setattr__(self, "premise_versions", request.premise_versions)
        if not isinstance(self.payload, Mapping):
            raise ContractError("payload must be an object")
        object.__setattr__(self, "payload", freeze_json(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {**self.header.to_dict(), "record_type": "async_result", "request_version": self.request_version,
                "question_version": self.question_version, "claim_ids": list(self.claim_ids),
                "premise_versions": thaw_json(self.premise_versions), "valid_until_ms": self.valid_until_ms,
                "payload": thaw_json(self.payload)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AsyncResult:
        names = {"record_type", "request_version", "question_version", "claim_ids", "premise_versions", "valid_until_ms", "payload"}
        exact_keys(data, {f.name for f in fields(Header)} | names)
        if data["record_type"] != "async_result":
            raise ContractError("record_type must be async_result")
        return cls(Header.from_dict({k: v for k, v in data.items() if k not in names}),
                   **{k: data[k] for k in names - {"record_type"}})
