"""Single-session coordinator for deterministic engineering tests.

This owns bookkeeping only. Cache slots are opaque placeholders for FUTURE
neural integrations; they never contain random neural outputs. Real microphone,
AEC, GPU, model inference and physical playback remain unimplemented.
"""
from __future__ import annotations
from dataclasses import asdict
from copy import deepcopy
from threading import RLock
from types import MappingProxyType
from .async_requests import RequestRegistry, ResultDecision
from .clock import ReplayClock
from .contracts import AsyncRequest, AsyncResult, Claim, ContractError, Scope, integer, nonempty
from .memory import SessionMemory
from .observations import ObservationBuffer
from .playback import PlaybackLedger


class SessionRuntime:
    def __init__(self, scope: Scope, voice_profile: str, *, max_journal: int = 1024, max_claims: int = 8, max_utterances: int = 4096):
        self._lock = RLock()
        self.max_journal = integer(max_journal, "max_journal", 1)
        self.max_claims = integer(max_claims, "max_claims", 1)
        self.max_utterances = integer(max_utterances, "max_utterances", 1)
        self._used_session_ids: set[str] = {scope.session_id}
        self.journal: list[dict] = []
        self._archived_sessions: dict[str, dict] = {}
        self._activate(scope, voice_profile)

    @property
    def scope(self) -> Scope:
        return self._scope

    @property
    def state_version(self) -> int:
        return self._state_version

    @property
    def question_version(self) -> int:
        return self._question_version

    @property
    def premise_versions(self):
        return MappingProxyType(self._premise_versions)

    @property
    def claims(self):
        return MappingProxyType(self._claims)

    def _activate(self, scope: Scope, voice_profile: str) -> None:
        self._scope = scope
        self.voice_profile = nonempty(voice_profile, "voice_profile")
        self.clock = ReplayClock()
        self._state_version = 0
        self._question_version = 0
        self._premise_versions: dict[str, int] = {}
        self._claims: dict[str, Claim] = {}
        self.observations = ObservationBuffer(scope)
        self.requests = RequestRegistry(scope)
        self.memory = SessionMemory(scope)
        self.playback: PlaybackLedger | None = None
        self._used_utterance_ids: set[str] = set()
        self._closed_utterances: list[dict] = []
        self.cache_handles: dict[str, object] = {}  # explicit invalidation test hook, not tensors
        self.switch_pending = False

    def advance_to(self, now_ms: int) -> None:
        with self._lock:
            self.clock.advance_to(now_ms)

    def advance_state(self) -> None:
        with self._lock:
            self._state_version += 1

    def set_premise(self, premise_id: str, revision: int) -> None:
        with self._lock:
            nonempty(premise_id, "premise_id")
            integer(revision, "revision")
            old = self._premise_versions.get(premise_id)
            if old is not None and revision <= old:
                raise ContractError("premise revision must increase")
            self._premise_versions[premise_id] = revision
            self.advance_state()

    def change_question(self) -> None:
        with self._lock:
            self._question_version += 1
            self.advance_state()
            self._claims.clear()

    def add_claim(self, claim: Claim) -> None:
        with self._lock:
            if self.switch_pending:
                raise ContractError("cannot add expression plans during persona switch")
            if claim.header.scope != self.scope or claim.header.state_version != self._state_version:
                raise ContractError("claim must belong to current scope/state")
            if claim.header.available_at_ms > self.clock.now_ms or claim.valid_until_ms < self.clock.now_ms:
                raise ContractError("claim not available or expired")
            if claim.claim_id in self._claims:
                raise ContractError("duplicate claim_id; use a new claim ID for revised content")
            if len(self._claims) >= self.max_claims:
                raise ContractError("claim slot limit reached; retire a claim explicitly")
            self._claims[claim.claim_id] = claim

    def retire_claim(self, claim_id: str) -> None:
        with self._lock:
            if claim_id not in self._claims:
                raise ContractError("unknown claim_id")
            del self._claims[claim_id]
            self.advance_state()

    def active_claim_ids(self) -> set[str]:
        return {key for key, claim in self._claims.items() if claim.valid_until_ms >= self.clock.now_ms}

    def register_request(self, request: AsyncRequest) -> None:
        with self._lock:
            if self.switch_pending:
                raise ContractError("cannot register request during persona switch")
            self.requests.register(request, now_ms=self.clock.now_ms, state_version=self._state_version,
                                   question_version=self._question_version, premise_versions=self._premise_versions,
                                   active_claim_ids=self.active_claim_ids())

    def receive_result(self, result: AsyncResult) -> ResultDecision:
        with self._lock:
            if len(self.journal) >= self.max_journal:
                raise ContractError("result journal full; export/archive before accepting more results")
            decision = self.requests.receive(result, now_ms=self.clock.now_ms, state_version=self._state_version,
                                             question_version=self._question_version, premise_versions=self._premise_versions,
                                             active_claim_ids=self.active_claim_ids())
            self.journal.append({**asdict(decision), "result_scope": result.header.scope.to_dict(),
                                 "received_at_ms": self.clock.now_ms})
            # Do not mutate claims or memory, and never auto-queue returned text.
            return decision

    def start_utterance(self, utterance_id: str, **kwargs) -> PlaybackLedger:
        with self._lock:
            if self.switch_pending:
                raise ContractError("cannot generate during persona switch")
            if self.playback is not None and not self.playback.stopped:
                raise ContractError("close/flush old utterance before starting another")
            if len(self._used_utterance_ids) >= self.max_utterances:
                raise ContractError("utterance anti-replay limit reached; activate a fresh session")
            if utterance_id in self._used_utterance_ids:
                raise ContractError("utterance_id must not be reused within a session")
            self.snapshot()  # do not archive a component timestamp ahead of the session
            new_playback = PlaybackLedger(self.scope, utterance_id, **kwargs)
            if self.playback is not None:
                self._closed_utterances.append(self.playback.snapshot())
            self.playback = new_playback
            self._used_utterance_ids.add(utterance_id)
            return self.playback

    def begin_switch(self) -> None:
        with self._lock:
            self.switch_pending = True
            self.requests.cancel_all(self.clock.now_ms)
            if self.playback is not None and not self.playback.stopped:
                self.playback.request_stop(scope=self.scope, now_ms=self.clock.now_ms)

    def commit_switch(self, scope: Scope, voice_profile: str) -> dict:
        """Atomic identity reset after actual flush acknowledgement.

        Require a fresh session_id, even when switching back to the same person,
        so old callbacks cannot resurrect when A -> B -> A. Time restarts at 0.
        """
        with self._lock:
            if not self.switch_pending:
                raise ContractError("begin_switch must run before commit_switch")
            if self.playback is not None and not self.playback.stopped:
                raise ContractError("await final playback cursor and device flush acknowledgement")
            if scope.session_id in self._used_session_ids:
                raise ContractError("each activation requires a fresh session_id")
            nonempty(voice_profile, "voice_profile")
            if len(self._archived_sessions) >= 64:
                raise ContractError("session archive full; export explicitly before switching")
            from .serialization import record_to_dict
            previous = self.snapshot()
            previous["memory_versions"] = [record_to_dict(event) for event in self.memory.snapshot(scope=self.scope, now_ms=self.clock.now_ms)]
            self._archived_sessions[self.scope.session_id] = previous
            self._used_session_ids.add(scope.session_id)
            self._activate(scope, voice_profile)
            return deepcopy(previous)

    def archived_session(self, session_id: str) -> dict:
        if session_id not in self._archived_sessions:
            raise ContractError("unknown archived session")
        return deepcopy(self._archived_sessions[session_id])

    def snapshot(self) -> dict:
        with self._lock:
            component_times = [self.observations.last_observed_ms, self.requests.last_observed_ms,
                               self.memory.last_observed_ms]
            if self.playback is not None:
                component_times.append(self.playback.last_observed_ms)
            if max(component_times) > self.clock.now_ms:
                raise ContractError("component clock is ahead of runtime; advance the shared session clock before callbacks")
            return {**self.scope.to_dict(), "voice_profile": self.voice_profile, "now_ms": self.clock.now_ms,
                    "state_version": self._state_version, "question_version": self._question_version,
                    "premise_versions": dict(self._premise_versions), "claim_ids": sorted(self._claims),
                    "switch_pending": self.switch_pending, "cache_handle_count": len(self.cache_handles),
                    "playback": self.playback.snapshot() if self.playback else None,
                    "closed_utterances": deepcopy(self._closed_utterances)}
