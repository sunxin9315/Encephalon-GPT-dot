import unittest
from dataclasses import replace
from encephalon.contracts import AsyncRequest, AsyncResult, Claim, ContractError, Scope
from encephalon.memory import MemoryEvent, SessionMemory
from encephalon.playback import OutputPacket, SpeechSpan
from encephalon.runtime import SessionRuntime
from tests.helpers import SCOPE, header


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = SessionMemory(SCOPE)
        self.event = MemoryEvent(header(at=10), "m", "source-event", "用户说 X", "reported")

    def test_no_implicit_truth_upgrade_or_duplicate_event(self):
        self.memory.put(self.event, 10)
        self.assertEqual(self.memory.get("m", scope=SCOPE, now_ms=10).assertion_kind.value, "reported")
        self.assertFalse(self.memory.put(self.event, 10))
        with self.assertRaises(ContractError):
            self.memory.put(replace(self.event, memory_id="m-duplicate"), 10)

    def test_correction_preserves_versions_and_original_provenance(self):
        self.memory.put(self.event, 10)
        correction = replace(self.event, header=header(at=20, revision=1), summary="用户改称 Y", assertion_kind="corrected")
        self.memory.put(correction, 20)
        self.assertEqual(self.memory.get("m", scope=SCOPE, now_ms=15).summary, "用户说 X")
        self.assertEqual(len(self.memory.history("m", scope=SCOPE, now_ms=20)), 2)
        bad = replace(correction, header=header(at=30, revision=2, source_ids=("new-only",)))
        with self.assertRaises(ContractError):
            self.memory.put(bad, 30)

    def test_scope_and_temporal_boundaries(self):
        with self.assertRaises(ContractError):
            self.memory.put(self.event, 9)
        self.memory.put(self.event, 10)
        for scope in (Scope("a2", "0.1", "s-a"), Scope("demo-a", "0.2", "s-a"), Scope("demo-a", "0.1", "s-b")):
            with self.assertRaises(ContractError):
                self.memory.get("m", scope=scope, now_ms=10)
        self.assertIsNone(self.memory.get("m", scope=SCOPE, now_ms=9))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SessionRuntime(SCOPE, "synthetic-voice-a")

    def test_switch_cancels_requests_and_preserves_played_and_memory(self):
        runtime = self.runtime
        runtime.add_claim(Claim(header(), "c", "尚未核验", "pending", (), True, 1000))
        req = AsyncRequest(header(request_id="r"), 0, 0, ("c",), {}, 1000)
        runtime.register_request(req)
        runtime.requests.dispatch("r", 0)
        runtime.memory.put(MemoryEvent(header(), "m", "e", "用户报告 X", "reported"), 0)
        runtime.cache_handles["semantic_kv"] = object()
        playback = runtime.start_utterance("u")
        playback.generate(OutputPacket(header(), "u", 0, 0, 80, (SpeechSpan(0, 80, "已说", ("c",)),)), 0)
        playback.queue(0, scope=SCOPE, now_ms=0)
        runtime.advance_to(80)
        runtime.begin_switch()
        with self.assertRaises(ContractError):
            runtime.commit_switch(Scope("demo-b", "0.1", "s-b"), "synthetic-voice-b")
        playback.confirm_stopped(80, scope=SCOPE, utterance_id="u", now_ms=80)
        saved = runtime.commit_switch(Scope("demo-b", "0.1", "s-b"), "synthetic-voice-b")
        self.assertEqual(saved["playback"]["said_text"], ["已说"])
        self.assertEqual(saved["memory_versions"][0]["summary"], "用户报告 X")
        saved["memory_versions"].clear()
        self.assertEqual(len(runtime.archived_session("s-a")["memory_versions"]), 1)
        self.assertEqual(runtime.cache_handles, {})
        self.assertEqual(runtime.clock.now_ms, 0)
        self.assertEqual(runtime.claims, {})
        result = AsyncResult(header(request_id="r"), 0, 0, ("c",), {}, 1000, {"text": "旧人物结论"})
        self.assertEqual(runtime.receive_result(result).reason, "scope_mismatch")

    def test_switch_requires_fresh_session_even_when_returning_to_a(self):
        runtime = self.runtime
        runtime.begin_switch()
        runtime.commit_switch(Scope("demo-b", "0.1", "s-b"), "voice-b")
        runtime.begin_switch()
        with self.assertRaises(ContractError):
            runtime.commit_switch(SCOPE, "voice-a")
        runtime.commit_switch(Scope("demo-a", "0.1", "s-a-new"), "voice-a")

    def test_success_does_not_promote_pending_claim_or_speak(self):
        runtime = self.runtime
        runtime.add_claim(Claim(header(), "c", "候选", "pending", (), True, 100))
        runtime.register_request(AsyncRequest(header(request_id="r"), 0, 0, ("c",), {}, 100))
        runtime.requests.dispatch("r", 0)
        result = AsyncResult(header(request_id="r"), 0, 0, ("c",), {}, 100, {"text": "candidate"})
        self.assertTrue(runtime.receive_result(result).accepted)
        self.assertEqual(runtime.claims["c"].evidence_status.value, "pending")
        self.assertIsNone(runtime.playback)

    def test_question_change_rejects_old_return(self):
        runtime = self.runtime
        runtime.add_claim(Claim(header(), "c", "候选", "pending", (), True, 100))
        runtime.register_request(AsyncRequest(header(request_id="r"), 0, 0, ("c",), {}, 100))
        runtime.requests.dispatch("r", 0)
        runtime.change_question()
        result = AsyncResult(header(request_id="r"), 0, 0, ("c",), {}, 100, {})
        self.assertEqual(runtime.receive_result(result).reason, "question_changed")

    def test_claim_slots_bounded_and_retirement_explicit(self):
        runtime = self.runtime
        for i in range(8):
            runtime.add_claim(Claim(header(), str(i), "候选", "pending", (), True, 100))
        with self.assertRaises(ContractError):
            runtime.add_claim(Claim(header(), "overflow", "候选", "pending", (), True, 100))
        runtime.retire_claim("0")
        runtime.add_claim(Claim(header(state=1), "new", "候选", "pending", (), True, 100))
        self.assertEqual(len(runtime.claims), 8)

    def test_public_versions_and_maps_are_read_only(self):
        with self.assertRaises(AttributeError):
            self.runtime.state_version = 100
        with self.assertRaises(TypeError):
            self.runtime.premise_versions["p"] = 10
        with self.assertRaises(TypeError):
            self.runtime.claims["foreign"] = object()

    def test_snapshot_rejects_component_time_ahead_of_runtime(self):
        ledger = self.runtime.start_utterance("u")
        ledger.generate(OutputPacket(header(), "u", 0, 0, 80), 0)
        ledger.queue(0, scope=SCOPE, now_ms=0)
        ledger.acknowledge_played(80, scope=SCOPE, utterance_id="u", now_ms=80)
        with self.assertRaises(ContractError):
            self.runtime.snapshot()
        self.runtime.advance_to(80)
        self.assertEqual(self.runtime.snapshot()["playback"]["played_until_ms"], 80)

    def test_utterance_anti_replay_state_bounded(self):
        runtime = SessionRuntime(SCOPE, "synthetic-voice", max_utterances=1)
        ledger = runtime.start_utterance("u1")
        ledger.request_stop(scope=SCOPE, now_ms=0)
        ledger.confirm_stopped(0, scope=SCOPE, utterance_id="u1", now_ms=0)
        with self.assertRaises(ContractError):
            runtime.start_utterance("u2")

    def test_closed_utterance_history_preserved_without_mutation(self):
        runtime = self.runtime
        first = runtime.start_utterance("u1")
        first.generate(OutputPacket(header(), "u1", 0, 0, 80, (SpeechSpan(0, 80, "first"),)), 0)
        first.queue(0, scope=SCOPE, now_ms=0)
        runtime.advance_to(80)
        first.request_stop(scope=SCOPE, now_ms=80)
        first.confirm_stopped(80, scope=SCOPE, utterance_id="u1", now_ms=80)
        second = runtime.start_utterance("u2")
        self.assertEqual(runtime.snapshot()["closed_utterances"][0]["said_text"], ["first"])
        runtime.snapshot()["closed_utterances"][0]["said_text"].clear()
        self.assertEqual(runtime.snapshot()["closed_utterances"][0]["said_text"], ["first"])
        runtime.begin_switch()
        second.confirm_stopped(0, scope=SCOPE, utterance_id="u2", now_ms=80)
        archived = runtime.commit_switch(Scope("b", "1", "s-b"), "voice-b")
        self.assertEqual(archived["closed_utterances"][0]["said_text"], ["first"])
