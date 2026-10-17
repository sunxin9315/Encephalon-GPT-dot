"""Generated/queued/actually-played ledger; no audio backend is implemented.

Offsets are utterance-relative audio milliseconds, not observation timestamps.
Stopping has two phases: block new queueing, then accept the final device cursor
and flush acknowledgement. Already audible material is never un-said.
"""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import ContractError, Header, Scope, ValidityStatus, integer, nonempty


@dataclass(frozen=True)
class SpeechSpan:
    start_ms: int
    end_ms: int
    text: str
    claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from .contracts import strings
        integer(self.start_ms, "span.start_ms")
        integer(self.end_ms, "span.end_ms")
        if self.start_ms >= self.end_ms:
            raise ContractError("speech span must have positive duration")
        nonempty(self.text, "span.text")
        object.__setattr__(self, "claim_ids", strings(self.claim_ids, "span.claim_ids"))


@dataclass(frozen=True)
class OutputPacket:
    header: Header
    utterance_id: str
    sequence_id: int
    audio_start_ms: int
    audio_end_ms: int
    spans: tuple[SpeechSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.header, Header):
            raise ContractError("header must be a Header")
        nonempty(self.utterance_id, "utterance_id")
        for name in ("sequence_id", "audio_start_ms", "audio_end_ms"):
            integer(getattr(self, name), name)
        if self.audio_start_ms >= self.audio_end_ms:
            raise ContractError("audio packet must have positive duration")
        if self.header.validity.status is not ValidityStatus.VALID:
            raise ContractError("cannot generate corrupt, missing or stale audio")
        if not isinstance(self.spans, (tuple, list)) or any(not isinstance(s, SpeechSpan) for s in self.spans):
            raise ContractError("spans must contain SpeechSpan values")
        object.__setattr__(self, "spans", tuple(self.spans))
        last_end = self.audio_start_ms
        for span in self.spans:
            if not last_end <= span.start_ms < span.end_ms <= self.audio_end_ms:
                raise ContractError("spans must be ordered, nonoverlapping and contained in the packet")
            last_end = span.end_ms


class PlaybackLedger:
    def __init__(self, scope: Scope, utterance_id: str, *, max_queue_ms: int = 320, max_packets: int = 4096):
        self._scope = scope
        self._utterance_id = nonempty(utterance_id, "utterance_id")
        self.max_queue_ms = integer(max_queue_ms, "max_queue_ms", 1)
        self.max_packets = integer(max_packets, "max_packets", 1)
        self._generated_until_ms = 0
        self._queued_until_ms = 0
        self._played_until_ms = 0
        self._stop_requested = False
        self._stopped = False
        self._packets: list[OutputPacket] = []
        self._queued_count = 0
        self._last_now = 0
        self._playback_started_ms: int | None = None
        self._last_ack_ms = 0
        self._earliest_play_intervals: list[tuple[int, int]] = []

    @property
    def utterance_id(self) -> str:
        return self._utterance_id

    @property
    def last_observed_ms(self) -> int:
        return self._last_now

    @property
    def scope(self) -> Scope:
        return self._scope

    @property
    def generated_until_ms(self) -> int:
        return self._generated_until_ms

    @property
    def queued_until_ms(self) -> int:
        return self._queued_until_ms

    @property
    def played_until_ms(self) -> int:
        return self._played_until_ms

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def stopped(self) -> bool:
        return self._stopped

    def _check(self, scope: Scope, now_ms: int) -> None:
        if scope != self.scope:
            raise ContractError("playback callback scope mismatch")
        integer(now_ms, "now_ms")
        if now_ms < self._last_now:
            raise ContractError("playback clock cannot move backward")
        self._last_now = now_ms

    def generate(self, packet: OutputPacket, now_ms: int) -> None:
        self._check(packet.header.scope, now_ms)
        if self._stop_requested or self._stopped:
            raise ContractError("generation forbidden after stop requested")
        if packet.utterance_id != self.utterance_id:
            raise ContractError("utterance mismatch")
        if packet.header.available_at_ms > now_ms:
            raise ContractError("audio packet not yet available")
        if packet.sequence_id != len(self._packets) or packet.audio_start_ms != self._generated_until_ms:
            raise ContractError("output packets must be contiguous and sequential")
        if len(self._packets) >= self.max_packets:
            raise ContractError("output ledger full; close/archive utterance")
        self._packets.append(packet)
        self._generated_until_ms = packet.audio_end_ms

    def queue(self, sequence_id: int, *, scope: Scope, now_ms: int) -> None:
        self._check(scope, now_ms)
        integer(sequence_id, "sequence_id")
        if self._stop_requested or self._stopped:
            raise ContractError("queueing forbidden after stop requested")
        if sequence_id != self._queued_count or sequence_id >= len(self._packets):
            raise ContractError("must queue the next generated packet exactly once")
        packet = self._packets[sequence_id]
        if packet.audio_end_ms - self._played_until_ms > self.max_queue_ms:
            raise ContractError("playback queue duration limit exceeded")
        earliest_start = max(now_ms, self._earliest_play_intervals[-1][1] if self._earliest_play_intervals else now_ms)
        self._earliest_play_intervals.append((earliest_start, earliest_start + packet.audio_end_ms - packet.audio_start_ms))
        self._queued_until_ms = packet.audio_end_ms
        self._queued_count += 1
        if self._playback_started_ms is None:
            self._playback_started_ms = now_ms

    def acknowledge_played(self, until_ms: int, *, scope: Scope, utterance_id: str, now_ms: int) -> None:
        self._check(scope, now_ms)
        integer(until_ms, "until_ms")
        if utterance_id != self.utterance_id:
            raise ContractError("playback callback utterance mismatch")
        if self._stopped:
            raise ContractError("late callback after final stop acknowledgement")
        if not self._played_until_ms <= until_ms <= self._queued_until_ms:
            raise ContractError("played cursor must advance monotonically within queued audio")
        for packet, (earliest_start, _) in zip(self._packets, self._earliest_play_intervals):
            covered_ms = min(until_ms, packet.audio_end_ms) - packet.audio_start_ms
            if covered_ms > 0 and now_ms < earliest_start + covered_ms:
                raise ContractError("played cursor exceeds feasible playback since packet queue time")
        self._played_until_ms = until_ms
        self._last_ack_ms = now_ms

    def request_stop(self, *, scope: Scope, now_ms: int) -> None:
        self._check(scope, now_ms)
        self._stop_requested = True

    def confirm_stopped(self, final_played_until_ms: int, *, scope: Scope, utterance_id: str, now_ms: int) -> None:
        if not self._stop_requested:
            raise ContractError("must request stop before confirming device flush")
        self.acknowledge_played(final_played_until_ms, scope=scope, utterance_id=utterance_id, now_ms=now_ms)
        self._stopped = True

    @property
    def said_spans(self) -> tuple[SpeechSpan, ...]:
        """Only fully played aligned spans; a partial word is not a full claim."""
        return tuple(span for packet in self._packets for span in packet.spans if span.end_ms <= self._played_until_ms)

    @property
    def said_claim_ids(self) -> tuple[str, ...]:
        return tuple(sorted({claim for span in self.said_spans for claim in span.claim_ids}))

    def snapshot(self) -> dict:
        return {**self.scope.to_dict(), "utterance_id": self.utterance_id,
                "generated_until_ms": self._generated_until_ms, "queued_until_ms": self._queued_until_ms,
                "played_until_ms": self._played_until_ms, "stop_requested": self._stop_requested,
                "stopped": self._stopped, "active_queue_ms": 0 if self._stopped else self._queued_until_ms - self._played_until_ms,
                "discarded_unplayed_ms": self._generated_until_ms - self._played_until_ms if self._stopped else 0,
                "said_text": [s.text for s in self.said_spans], "said_claim_ids": list(self.said_claim_ids)}
