"""Adversarial stage-0 invariants, using synthetic bookkeeping only.

No device cursor in these fixtures comes from real hardware. No model output,
training result, or measured performance is claimed by any passing test.
"""
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
import unittest

from encephalon.async_requests import RequestRegistry, RequestStatus
from encephalon.clock import ReplayClock
from encephalon.contracts import (
    AsyncRequest, AsyncResult, Claim, ContractError, Header, Observation, Scope,
    Validity,
)
from encephalon.memory import MemoryEvent, SessionMemory
from encephalon.observations import CapacityError, ObservationBuffer
from encephalon.playback import OutputPacket, PlaybackLedger, SpeechSpan
from encephalon.runtime import SessionRuntime


SCOPE = Scope("fixture-persona", "fixture-v1", "fixture-session")


def header(*, at=0, end=0, start=0, revision=0, state=0, scope=SCOPE,
           request_id=None, status="valid", sources=("synthetic-source",), provenance=()):
    return Header(**scope.to_dict(), state_version=state, event_start_ms=start,
                  event_end_ms=end, available_at_ms=at, source_ids=sources,
                  provenance=provenance,
                  validity=Validity(status, 1.0, at - end,
                                    None if status == "valid" else "synthetic test condition"),
                  revision=revision, request_id=request_id)


def observation(*, at=0, revision=0, event_id="event", text="fixture", **kwargs):
    return Observation(header(at=at, revision=revision, **kwargs), event_id,
                       "speech_semantic", {"text": text})


def request(*, at=0, state=0, version=0, question=0, premises=None,
            claims=("claim",), until=1000, scope=SCOPE, request_id="request"):
    return AsyncRequest(header(at=at, state=state, scope=scope, request_id=request_id),
                        version, question, claims, {} if premises is None else premises, until)


def result(*, at=10, state=0, version=0, question=0, premises=None,
           claims=("claim",), until=1000, scope=SCOPE, request_id="request",
           status="valid", sources=("synthetic-source",), provenance=()):
    return AsyncResult(header(at=at, state=state, scope=scope, request_id=request_id,
                              status=status, sources=sources, provenance=provenance),
                       version, question, claims, {} if premises is None else premises,
                       until, {"text": "synthetic, not fact-verified"})


def claim(*, state=0, at=0, until=1000, claim_id="claim", status="pending"):
    return Claim(header(state=state, at=at), claim_id, "A synthetic proposition",
                 status, ("synthetic-evidence",) if status == "supported" else (), True, until)


def registry(req=None, dispatch_at=0):
    req = request() if req is None else req
    reg = RequestRegistry(req.header.scope)
    reg.register(req, now_ms=req.header.available_at_ms,
                 state_version=req.header.state_version, question_version=req.question_version,
                 premise_versions=dict(req.premise_versions), active_claim_ids=set(req.claim_ids))
    reg.dispatch(req.header.request_id, dispatch_at)
    return reg


def receive(reg, value, *, now=10, state=0, question=0, premises=None, claims=None):
    return reg.receive(value, now_ms=now, state_version=state, question_version=question,
                       premise_versions={} if premises is None else premises,
                       active_claim_ids={"claim"} if claims is None else claims)


def packet(sequence=0, start=0, end=100, *, at=0, scope=SCOPE, spans=None):
    if spans is None:
        spans = (SpeechSpan(start, end, f"synthetic span {sequence}", (f"claim-{sequence}",)),)
    return OutputPacket(header(at=at, scope=scope), "utterance", sequence, start, end, spans)


class ContractAdversarialTests(unittest.TestCase):
    def test_nested_payload_is_detached_and_immutable(self):
        raw = {"items": [{"token": "old"}], "flags": [True]}
        obs = Observation(header(), "event", "text", raw)
        raw["items"][0]["token"] = "changed"
        raw["flags"].append(False)
        self.assertEqual(obs.payload["items"][0]["token"], "old")
        self.assertEqual(obs.payload["flags"], (True,))
        with self.assertRaises(TypeError):
            obs.payload["items"][0]["token"] = "illegal"
        with self.assertRaises(FrozenInstanceError):
            obs.header.available_at_ms = 100
        exported = obs.to_dict()
        exported["payload"]["items"][0]["token"] = "changed-again"
        self.assertEqual(obs.payload["items"][0]["token"], "old")

    def test_request_dependency_input_mutation_cannot_change_request(self):
        premises = {"premise": 1}
        req = request(premises=premises)
        premises["premise"] = 9
        self.assertEqual(dict(req.premise_versions), {"premise": 1})
        with self.assertRaises(TypeError):
            req.premise_versions["premise"] = 10

    def test_boolean_integer_fields_are_rejected(self):
        for field in ("state_version", "event_start_ms", "event_end_ms", "available_at_ms", "revision"):
            with self.subTest(field=field), self.assertRaises(ContractError):
                replace(header(), **{field: True})
        for value in (True, -1, 1.0):
            with self.subTest(value=value), self.assertRaises(ContractError):
                ReplayClock(value)

    def test_non_finite_payloads_and_non_json_keys_are_rejected(self):
        for payload in ({"v": float("nan")}, {"v": float("inf")}, {1: "bad"}, {"v": b"bytes"}):
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                Observation(header(), "event", "text", payload)

    def test_canonical_roundtrip_is_strict_and_preserves_availability(self):
        obs = observation(at=80, end=40, revision=2)
        self.assertEqual(Observation.from_dict(obs.to_dict()), obs)
        data = obs.to_dict()
        data["arrival_ms"] = 0
        with self.assertRaises(ContractError):
            Observation.from_dict(data)
        data = obs.to_dict()
        data["validity"] = "present"
        with self.assertRaises(ContractError):
            Observation.from_dict(data)

    def test_header_age_cannot_claim_future_availability_is_current(self):
        h = header(at=100, end=20)
        with self.assertRaises(ContractError):
            h.age_at(99)
        self.assertEqual(h.age_at(150), 130)
        with self.assertRaises(ContractError):
            replace(h, validity=Validity("valid", 1, 0))

    def test_supported_claim_needs_evidence_and_time_validity(self):
        with self.assertRaises(ContractError):
            Claim(header(), "claim", "unsupported", "supported", (), False, 10)
        supported = claim(at=10, until=20, status="supported")
        self.assertFalse(supported.can_assert(9))
        self.assertTrue(supported.can_assert(10))
        self.assertTrue(supported.can_assert(20))
        self.assertFalse(supported.can_assert(21))
        self.assertFalse(claim(status="pending").can_assert(10))


class ObservationAdversarialTests(unittest.TestCase):
    def test_future_revision_never_hides_visible_original_for_any_ingest_order(self):
        versions = [observation(at=10, revision=0, text="first"),
                    observation(at=20, revision=1, text="second"),
                    observation(at=30, revision=2, text="third")]
        for order in permutations(versions):
            with self.subTest(order=[v.header.revision for v in order]):
                buf = ObservationBuffer(SCOPE)
                for obs in order:
                    buf.add(obs)
                self.assertEqual(buf.visible(9), ())
                self.assertEqual(buf.visible(10), (versions[0],))
                self.assertEqual(buf.visible(29), (versions[1],))
                self.assertEqual(buf.visible(30), (versions[2],))

    def test_duplicate_and_revisions_do_not_repeat_consumed_content(self):
        buf = ObservationBuffer(SCOPE)
        original = observation(at=10)
        self.assertTrue(buf.add(original))
        self.assertFalse(buf.add(original))
        self.assertEqual(buf.consume(9), ())
        delta, = buf.consume(10)
        self.assertEqual(delta.change, "inserted")
        self.assertTrue(delta.valid_new_event)
        self.assertEqual(buf.consume(10), ())
        buf.add(observation(at=20, revision=1))
        delta, = buf.consume(20)
        self.assertEqual(delta.change, "revised")
        self.assertFalse(delta.valid_new_event)
        self.assertEqual(buf.consume(20), ())

    def test_missing_corrupt_stale_are_not_valid_new_events(self):
        for status in ("missing", "stale", "corrupt"):
            with self.subTest(status=status):
                buf = ObservationBuffer(SCOPE)
                buf.add(observation(status=status))
                delta, = buf.consume(0)
                self.assertFalse(delta.valid_new_event)

    def test_recovery_from_missing_is_a_valid_new_event_exactly_once(self):
        buf = ObservationBuffer(SCOPE)
        for rev, status in enumerate(("missing", "corrupt", "valid", "stale", "valid")):
            buf.add(observation(at=rev * 10, revision=rev, status=status))
            delta, = buf.consume(rev * 10)
            self.assertEqual(delta.valid_new_event, rev == 2)
            self.assertEqual(delta.change, "inserted" if rev == 0 else "revised")

    def test_pruning_old_revisions_does_not_reset_event_stimulation(self):
        buf = ObservationBuffer(SCOPE)
        for rev in range(3):
            buf.add(observation(at=rev * 10, revision=rev))
            delta, = buf.consume(rev * 10)
            self.assertEqual(delta.valid_new_event, rev == 0)
            buf.prune_before(1000)

    def test_conflicts_and_revisions_rewriting_event_identity_are_rejected(self):
        buf = ObservationBuffer(SCOPE)
        buf.add(observation(at=10))
        for candidate in (observation(at=10, text="conflicting"),
                          observation(at=20, end=1, revision=1),
                          observation(at=20, revision=1, sources=("another-source",)),
                          observation(at=5, revision=1)):
            with self.subTest(candidate=candidate), self.assertRaises(ContractError):
                buf.add(candidate)

    def test_scope_includes_persona_version_and_session_not_just_persona_id(self):
        for scope in (Scope("another", SCOPE.persona_version, SCOPE.session_id),
                      Scope(SCOPE.persona_id, "v2", SCOPE.session_id),
                      Scope(SCOPE.persona_id, SCOPE.persona_version, "another-session")):
            with self.subTest(scope=scope), self.assertRaises(ContractError):
                ObservationBuffer(SCOPE).add(observation(scope=scope))

    def test_prune_cannot_delete_unconsumed_or_latest_consumed_revision(self):
        buf = ObservationBuffer(SCOPE, max_records=3)
        versions = [observation(at=10 * (i + 1), revision=i) for i in range(3)]
        for obs in versions:
            buf.add(obs)
        self.assertEqual(buf.prune_before(100), 0)
        buf.consume(10)
        self.assertEqual(buf.prune_before(100), 0)
        buf.consume(20)
        self.assertEqual(buf.prune_before(100), 1)
        self.assertEqual(buf.visible(20), (versions[1],))
        self.assertEqual(buf.consume(30)[0].observation, versions[2])

    def test_capacity_is_backpressure_not_silent_eviction(self):
        buf = ObservationBuffer(SCOPE, max_records=1)
        original = observation()
        buf.add(original)
        with self.assertRaises(CapacityError):
            buf.add(observation(event_id="other"))
        self.assertEqual(buf.visible(0), (original,))


class RequestAdversarialTests(unittest.TestCase):
    def test_result_predating_dispatch_is_rejected(self):
        reg = registry(request(at=100), dispatch_at=200)
        decision = receive(reg, result(at=199), now=200)
        self.assertFalse(decision.accepted, "A result cannot exist before this request was dispatched")

    def test_not_yet_available_result_does_not_end_request(self):
        reg = registry()
        future = result(at=20)
        self.assertFalse(receive(reg, future, now=10).accepted)
        self.assertEqual(reg.status_view("request", 10)["status"], "running")
        self.assertTrue(receive(reg, future, now=20).accepted)

    def test_result_versions_scope_and_targets_fail_closed(self):
        cases = [result(version=1), result(state=1), result(question=1),
                 result(scope=Scope("other", "fixture-v1", "fixture-session")),
                 result(scope=Scope("fixture-persona", "other", "fixture-session")),
                 result(scope=Scope("fixture-persona", "fixture-v1", "other")),
                 result(claims=("unrequested",)), result(request_id="unknown"),
                 result(premises={"undeclared": 1})]
        for value in cases:
            with self.subTest(value=value):
                reg = registry()
                self.assertFalse(receive(reg, value).accepted)
                self.assertEqual(reg.status_view("request", 10)["status"], "running")

    def test_changed_question_or_inactive_claim_rejects_old_result(self):
        for kwargs in ({"question": 1}, {"claims": set()}):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(receive(registry(), result(), **kwargs).accepted)

    def test_changed_state_requires_unchanged_explicit_dependencies(self):
        self.assertFalse(receive(registry(), result(), state=1).accepted)
        req = request(state=1, premises={"p": 3})
        value = result(state=1, premises={"p": 3})
        self.assertTrue(receive(registry(req), value, state=2, premises={"p": 3}).accepted)
        self.assertFalse(receive(registry(req), value, state=2, premises={"p": 4}).accepted)
        self.assertFalse(receive(registry(req), value, state=0, premises={"p": 3}).accepted)

    def test_cancelled_and_replaced_request_reject_old_return(self):
        reg = registry()
        reg.fail("request", RequestStatus.CANCELLED, 5)
        self.assertFalse(receive(reg, result()).accepted)
        replacement = request(at=10, version=1)
        reg.register(replacement, now_ms=10, state_version=0, question_version=0,
                     premise_versions={}, active_claim_ids={"claim"})
        reg.dispatch("request", 10)
        self.assertFalse(receive(reg, result()).accepted)
        self.assertTrue(receive(reg, result(version=1)).accepted)
        self.assertFalse(receive(reg, result(version=1)).accepted)

    def test_invalid_expired_and_source_laundered_results_are_rejected(self):
        for value, now in ((result(status="missing"), 10), (result(status="stale"), 10),
                           (result(status="corrupt"), 10), (result(until=10), 11),
                           (result(sources=("unrelated",)), 10)):
            with self.subTest(value=value, now=now):
                self.assertFalse(receive(registry(), value, now=now).accepted)
        self.assertTrue(receive(registry(), result(sources=("derived",),
                                                   provenance=("synthetic-source",))).accepted)

    def test_expired_request_cannot_be_extended_by_return(self):
        reg = registry(request(until=10))
        self.assertFalse(receive(reg, result(until=1000), now=11).accepted)

    def test_status_does_not_leak_simulator_future_schedule(self):
        view = registry().status_view("request", 10)
        self.assertEqual(view["elapsed_ms"], 10)
        self.assertEqual(view["status"], "running")
        for forbidden in ("eta", "remaining_ms", "completion_percent", "remote_queue_position", "valid_until_ms"):
            self.assertNotIn(forbidden, view)


class PlaybackAdversarialTests(unittest.TestCase):
    def test_partial_span_does_not_commit_full_claim_and_stop_preserves_it(self):
        ledger = PlaybackLedger(SCOPE, "utterance")
        spans = (SpeechSpan(0, 50, "heard", ("heard-claim",)),
                 SpeechSpan(50, 100, "not fully heard", ("unheard-claim",)))
        ledger.generate(packet(spans=spans), 0)
        ledger.queue(0, scope=SCOPE, now_ms=0)
        ledger.acknowledge_played(75, scope=SCOPE, utterance_id="utterance", now_ms=75)
        self.assertEqual(ledger.said_claim_ids, ("heard-claim",))
        ledger.request_stop(scope=SCOPE, now_ms=75)
        ledger.confirm_stopped(80, scope=SCOPE, utterance_id="utterance", now_ms=80)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["said_text"], ["heard"])
        self.assertEqual(snapshot["discarded_unplayed_ms"], 20)
        self.assertEqual(snapshot["active_queue_ms"], 0)
        with self.assertRaises(ContractError):
            ledger.acknowledge_played(100, scope=SCOPE, utterance_id="utterance", now_ms=100)

    def test_underrun_wait_time_cannot_make_fresh_queue_already_played(self):
        ledger = PlaybackLedger(SCOPE, "utterance")
        ledger.generate(packet(), 0)
        ledger.queue(0, scope=SCOPE, now_ms=0)
        ledger.acknowledge_played(100, scope=SCOPE, utterance_id="utterance", now_ms=100)
        ledger.generate(packet(1, 100, 200, at=1000), 1000)
        ledger.queue(1, scope=SCOPE, now_ms=1000)
        with self.assertRaises(ContractError):
            ledger.acknowledge_played(200, scope=SCOPE, utterance_id="utterance", now_ms=1000)
        self.assertEqual(ledger.played_until_ms, 100)
        ledger.acknowledge_played(200, scope=SCOPE, utterance_id="utterance", now_ms=1100)

    def test_late_generated_audio_cannot_claim_earlier_elapsed_playback(self):
        ledger = PlaybackLedger(SCOPE, "utterance")
        ledger.generate(packet(), 0)
        ledger.queue(0, scope=SCOPE, now_ms=0)
        # No cursor was reported during the idle gap. Queue timestamps still bound playback.
        ledger.generate(packet(1, 100, 200, at=1000), 1000)
        ledger.queue(1, scope=SCOPE, now_ms=1000)
        with self.assertRaises(ContractError):
            ledger.acknowledge_played(150, scope=SCOPE, utterance_id="utterance", now_ms=1000)

    def test_generated_unqueued_audio_can_never_count_as_said(self):
        ledger = PlaybackLedger(SCOPE, "utterance")
        ledger.generate(packet(), 0)
        with self.assertRaises(ContractError):
            ledger.acknowledge_played(1, scope=SCOPE, utterance_id="utterance", now_ms=1000)
        self.assertEqual(ledger.said_spans, ())

    def test_stop_blocks_further_generation_and_queueing(self):
        ledger = PlaybackLedger(SCOPE, "utterance")
        ledger.generate(packet(), 0)
        ledger.request_stop(scope=SCOPE, now_ms=0)
        with self.assertRaises(ContractError):
            ledger.queue(0, scope=SCOPE, now_ms=0)
        with self.assertRaises(ContractError):
            ledger.generate(packet(1, 100, 200), 0)
        ledger.confirm_stopped(0, scope=SCOPE, utterance_id="utterance", now_ms=0)
        self.assertEqual(ledger.snapshot()["discarded_unplayed_ms"], 100)

    def test_queue_limit_cannot_be_bypassed_by_generation(self):
        ledger = PlaybackLedger(SCOPE, "utterance", max_queue_ms=100)
        ledger.generate(packet(), 0)
        ledger.generate(packet(1, 100, 200), 0)
        ledger.queue(0, scope=SCOPE, now_ms=0)
        with self.assertRaises(ContractError):
            ledger.queue(1, scope=SCOPE, now_ms=0)
        self.assertEqual(ledger.queued_until_ms, 100)

    def test_spans_cannot_overlap_or_escape_packet_boundaries(self):
        for spans in ((SpeechSpan(0, 101, "escape"),),
                      (SpeechSpan(0, 60, "first"), SpeechSpan(50, 100, "overlap"))):
            with self.subTest(spans=spans), self.assertRaises(ContractError):
                packet(spans=spans)

    def test_public_cursor_assignment_cannot_bypass_acknowledgement(self):
        ledger = PlaybackLedger(SCOPE, "utterance")
        ledger.generate(packet(), 0)
        for attribute, value in (("played_until_ms", 100), ("queued_until_ms", 100),
                                 ("generated_until_ms", 1000), ("stopped", True),
                                 ("stop_requested", True)):
            with self.subTest(attribute=attribute), self.assertRaises(AttributeError):
                setattr(ledger, attribute, value)
        self.assertEqual(ledger.said_spans, ())

    def test_callbacks_from_prior_utterance_with_same_scope_are_rejected(self):
        ledger = PlaybackLedger(SCOPE, "utterance")
        ledger.generate(packet(), 0)
        ledger.queue(0, scope=SCOPE, now_ms=0)
        with self.assertRaises(ContractError):
            ledger.acknowledge_played(50, scope=SCOPE, utterance_id="old-utterance", now_ms=50)
        self.assertEqual(ledger.played_until_ms, 0)
        ledger.request_stop(scope=SCOPE, now_ms=50)
        with self.assertRaises(ContractError):
            ledger.confirm_stopped(50, scope=SCOPE, utterance_id="old-utterance", now_ms=50)
        self.assertFalse(ledger.stopped)

    def test_playback_callbacks_require_exact_scope(self):
        ledger = PlaybackLedger(SCOPE, "utterance")
        ledger.generate(packet(), 0)
        ledger.queue(0, scope=SCOPE, now_ms=0)
        wrong = Scope(SCOPE.persona_id, "other-version", SCOPE.session_id)
        with self.assertRaises(ContractError):
            ledger.acknowledge_played(10, scope=wrong, utterance_id="utterance", now_ms=10)
        self.assertEqual(ledger.played_until_ms, 0)


class MemoryAndRuntimeAdversarialTests(unittest.TestCase):
    def test_public_scope_reassignment_cannot_relabel_old_persona_data(self):
        wrong = Scope("other-persona", "v2", "other-session")
        scoped = (ObservationBuffer(SCOPE), RequestRegistry(SCOPE), SessionMemory(SCOPE),
                  PlaybackLedger(SCOPE, "utterance"), SessionRuntime(SCOPE, "synthetic-voice"))
        for obj in scoped:
            with self.subTest(type=type(obj).__name__), self.assertRaises(AttributeError):
                obj.scope = wrong
            self.assertEqual(obj.scope, SCOPE)
        ledger = scoped[3]
        with self.assertRaises(AttributeError):
            ledger.utterance_id = "relabelled"

    def test_utterance_id_cannot_be_reused_for_stale_same_session_callback(self):
        runtime = SessionRuntime(SCOPE, "synthetic-voice")
        old = runtime.start_utterance("utterance")
        old.request_stop(scope=SCOPE, now_ms=0)
        old.confirm_stopped(0, scope=SCOPE, utterance_id="utterance", now_ms=0)
        with self.assertRaises(ContractError):
            runtime.start_utterance("utterance")
        self.assertEqual(runtime.start_utterance("next-utterance").utterance_id, "next-utterance")

    def test_memory_revision_does_not_leak_into_historical_read(self):
        memory = SessionMemory(SCOPE)
        first = MemoryEvent(header(at=10), "memory", "event", "reported, not observed", "reported")
        correction = MemoryEvent(header(at=20, revision=1, provenance=("correction-source",)),
                                 "memory", "event", "corrected report", "corrected")
        memory.put(first, 10)
        memory.put(correction, 20)
        self.assertIsNone(memory.get("memory", scope=SCOPE, now_ms=9))
        self.assertEqual(memory.get("memory", scope=SCOPE, now_ms=10), first)
        self.assertEqual(memory.history("memory", scope=SCOPE, now_ms=19), (first,))
        self.assertEqual(memory.get("memory", scope=SCOPE, now_ms=20), correction)

    def test_memory_cannot_duplicate_event_or_erase_provenance(self):
        memory = SessionMemory(SCOPE)
        first = MemoryEvent(header(), "memory", "event", "report", "reported")
        self.assertTrue(memory.put(first, 0))
        self.assertFalse(memory.put(first, 0))
        with self.assertRaises(ContractError):
            memory.put(replace(first, memory_id="duplicate"), 0)
        with self.assertRaises(ContractError):
            memory.put(replace(first, header=header(at=10, revision=1, sources=("replacement",))), 10)

    def test_memory_rejects_invalid_future_and_wrong_namespace_inputs(self):
        with self.assertRaises(ContractError):
            MemoryEvent(header(status="stale"), "memory", "event", "stale", "observed")
        future = MemoryEvent(header(at=10), "memory", "event", "future", "reported")
        memory = SessionMemory(SCOPE)
        with self.assertRaises(ContractError):
            memory.put(future, 9)
        with self.assertRaises(ContractError):
            memory.get("memory", scope=Scope(SCOPE.persona_id, "v2", SCOPE.session_id), now_ms=10)

    def test_transport_success_does_not_promote_claim_memory_or_speech(self):
        runtime = SessionRuntime(SCOPE, "synthetic-voice")
        pending = claim()
        runtime.add_claim(pending)
        runtime.register_request(request())
        runtime.requests.dispatch("request", 0)
        runtime.advance_to(10)
        self.assertTrue(runtime.receive_result(result()).accepted)
        self.assertIs(runtime.claims["claim"], pending)
        self.assertFalse(pending.can_assert(10))
        self.assertIsNone(runtime.playback)
        self.assertIsNone(runtime.memory.get("claim", scope=SCOPE, now_ms=10))

    def test_switch_requires_flush_and_fresh_session_then_clears_state(self):
        runtime = SessionRuntime(SCOPE, "synthetic-voice")
        runtime.cache_handles["fake-only"] = object()
        runtime.add_claim(claim())
        runtime.register_request(request())
        runtime.requests.dispatch("request", 0)
        old_ledger = runtime.start_utterance("utterance")
        old_ledger.generate(packet(), 0)
        old_ledger.queue(0, scope=SCOPE, now_ms=0)
        runtime.advance_to(50)
        runtime.begin_switch()
        next_scope = Scope("new-persona", "v2", "new-session")
        with self.assertRaises(ContractError):
            runtime.commit_switch(next_scope, "new-voice")
        old_ledger.confirm_stopped(50, scope=SCOPE, utterance_id="utterance", now_ms=50)
        runtime.commit_switch(next_scope, "new-voice")
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["state_version"], 0)
        self.assertEqual(snapshot["question_version"], 0)
        self.assertEqual(snapshot["now_ms"], 0)
        self.assertEqual(snapshot["claim_ids"], [])
        self.assertEqual(snapshot["cache_handle_count"], 0)
        self.assertIsNone(snapshot["playback"])
        self.assertFalse(runtime.receive_result(result()).accepted)
        runtime.begin_switch()
        with self.assertRaises(ContractError):
            runtime.commit_switch(SCOPE, "synthetic-voice")

    def test_question_change_clears_plans_and_rejects_late_return(self):
        runtime = SessionRuntime(SCOPE, "synthetic-voice")
        runtime.add_claim(claim())
        runtime.register_request(request())
        runtime.requests.dispatch("request", 0)
        runtime.change_question()
        runtime.advance_to(10)
        self.assertFalse(runtime.receive_result(result()).accepted)
        self.assertEqual(runtime.active_claim_ids(), set())

    def test_journal_backpressure_does_not_accept_an_unrecorded_result(self):
        runtime = SessionRuntime(SCOPE, "synthetic-voice", max_journal=1)
        runtime.add_claim(claim())
        runtime.register_request(request())
        runtime.requests.dispatch("request", 0)
        runtime.advance_to(10)
        self.assertFalse(runtime.receive_result(result(request_id="unknown")).accepted)
        with self.assertRaises(ContractError):
            runtime.receive_result(result())
        self.assertEqual(runtime.requests.status_view("request", 10)["status"], "running")


if __name__ == "__main__":
    unittest.main()
