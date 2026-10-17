"""Local request bookkeeping and applicability checks; no network client.

Transport success never promotes a Claim to supported. QUEUED means our own
local queue; RUNNING means in flight, not a remote progress estimate.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from .contracts import AsyncRequest, AsyncResult, ContractError, Scope, ValidityStatus, integer


class RequestStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"


@dataclass
class RequestEntry:
    request: AsyncRequest
    status: RequestStatus
    created_at_ms: int
    dispatched_at_ms: int | None = None
    finished_at_ms: int | None = None


@dataclass(frozen=True)
class ResultDecision:
    accepted: bool
    reason: str
    request_id: str
    request_version: int


class RequestRegistry:
    def __init__(self, scope: Scope, max_requests: int = 1024):
        self._scope = scope
        self.max_requests = integer(max_requests, "max_requests", 1)
        self._entries: dict[str, RequestEntry] = {}
        self._last_now = 0

    @property
    def last_observed_ms(self) -> int:
        return self._last_now

    @property
    def scope(self) -> Scope:
        return self._scope

    def _time(self, now_ms: int) -> None:
        integer(now_ms, "now_ms")
        if now_ms < self._last_now:
            raise ContractError("request registry clock cannot go backward")
        self._last_now = now_ms

    def register(self, request: AsyncRequest, *, now_ms: int, state_version: int,
                 question_version: int, premise_versions: Mapping[str, int], active_claim_ids: set[str]) -> None:
        self._time(now_ms)
        if request.header.scope != self.scope:
            raise ContractError("request scope mismatch")
        if request.header.available_at_ms > now_ms or request.valid_until_ms < now_ms:
            raise ContractError("request not available or already expired")
        if request.header.validity.status is not ValidityStatus.VALID:
            raise ContractError("request must have valid input")
        if request.header.state_version != state_version or request.question_version != question_version:
            raise ContractError("request must capture current state and question versions")
        if not set(request.claim_ids) <= active_claim_ids:
            raise ContractError("request targets inactive claim")
        if any(premise_versions.get(k) != v for k, v in request.premise_versions.items()):
            raise ContractError("request premise mismatch")
        request_id = request.header.request_id
        old = self._entries.get(request_id)
        if old:
            if old.status in {RequestStatus.QUEUED, RequestStatus.RUNNING}:
                raise ContractError("cannot replace a pending request without cancelling it")
            if request.request_version <= old.request.request_version:
                raise ContractError("reused request_id must increase request_version")
        elif len(self._entries) >= self.max_requests:
            raise ContractError("request registry full; archive terminal requests explicitly")
        self._entries[request_id] = RequestEntry(request, RequestStatus.QUEUED, now_ms)

    def dispatch(self, request_id: str, now_ms: int) -> None:
        self._time(now_ms)
        entry = self._get(request_id)
        if entry.status is not RequestStatus.QUEUED:
            raise ContractError("only a locally queued request can be dispatched")
        if now_ms > entry.request.valid_until_ms:
            raise ContractError("cannot dispatch expired request")
        entry.status = RequestStatus.RUNNING
        entry.dispatched_at_ms = now_ms

    def _get(self, request_id: str) -> RequestEntry:
        if request_id not in self._entries:
            raise ContractError(f"unknown request_id: {request_id}")
        return self._entries[request_id]

    def fail(self, request_id: str, status: RequestStatus | str, now_ms: int) -> None:
        self._time(now_ms)
        try:
            status = RequestStatus(status)
        except (ValueError, TypeError) as exc:
            raise ContractError("invalid request failure state") from exc
        if status not in {RequestStatus.FAILED, RequestStatus.TIMEOUT, RequestStatus.UNAVAILABLE, RequestStatus.CANCELLED}:
            raise ContractError("fail accepts only failure/cancellation states")
        entry = self._get(request_id)
        if entry.status not in {RequestStatus.QUEUED, RequestStatus.RUNNING}:
            raise ContractError("request already terminal")
        entry.status, entry.finished_at_ms = status, now_ms

    def cancel_all(self, now_ms: int) -> None:
        self._time(now_ms)
        for entry in self._entries.values():
            if entry.status in {RequestStatus.QUEUED, RequestStatus.RUNNING}:
                entry.status, entry.finished_at_ms = RequestStatus.CANCELLED, now_ms

    def status_view(self, request_id: str, now_ms: int) -> dict:
        self._time(now_ms)
        entry = self._get(request_id)
        end = entry.finished_at_ms if entry.finished_at_ms is not None else now_ms
        return {"request_id": request_id, "status": entry.status.value,
                "elapsed_ms": end - entry.created_at_ms,
                "request_version": entry.request.request_version,
                "request_state_version": entry.request.header.state_version,
                "claim_ids": list(entry.request.claim_ids)}

    def receive(self, result: AsyncResult, *, now_ms: int, state_version: int,
                question_version: int, premise_versions: Mapping[str, int], active_claim_ids: set[str]) -> ResultDecision:
        self._time(now_ms)
        request_id = result.header.request_id

        def reject(reason: str) -> ResultDecision:
            return ResultDecision(False, reason, request_id, result.request_version)

        if result.header.scope != self.scope:
            return reject("scope_mismatch")
        if result.header.available_at_ms > now_ms:
            return reject("not_yet_available")
        entry = self._entries.get(request_id)
        if entry is None:
            return reject("unknown_request")
        request = entry.request
        if result.request_version != request.request_version:
            return reject("request_version_mismatch")
        if entry.status is not RequestStatus.RUNNING:
            return reject("request_not_running")
        if result.header.available_at_ms < entry.dispatched_at_ms:
            return reject("result_predates_dispatch")
        if result.header.state_version != request.header.state_version:
            return reject("request_state_mismatch")
        if result.question_version != request.question_version or question_version != request.question_version:
            return reject("question_changed")
        if not set(result.claim_ids) <= set(request.claim_ids) or not set(result.claim_ids) <= active_claim_ids:
            return reject("claim_inactive_or_unrequested")
        if dict(result.premise_versions) != dict(request.premise_versions):
            return reject("result_premise_mismatch")
        if any(premise_versions.get(k) != v for k, v in request.premise_versions.items()):
            return reject("premise_changed")
        if state_version < request.header.state_version:
            return reject("future_request_state")
        if state_version != request.header.state_version and not request.premise_versions:
            return reject("state_changed_without_dependencies")
        if now_ms > min(request.valid_until_ms, result.valid_until_ms):
            return reject("expired")
        if result.header.validity.status is not ValidityStatus.VALID:
            return reject("invalid_result")
        if not set(request.header.source_ids) <= set(result.header.source_ids + result.header.provenance):
            return reject("missing_request_provenance")
        # This changes transport state only; evidence/expressibility is separate.
        entry.status, entry.finished_at_ms = RequestStatus.SUCCEEDED, now_ms
        return ResultDecision(True, "applicable_not_fact_verified", request_id, result.request_version)
