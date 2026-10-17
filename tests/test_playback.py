import unittest
from encephalon.contracts import ContractError, Scope
from encephalon.playback import OutputPacket, PlaybackLedger, SpeechSpan
from tests.helpers import SCOPE, header


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.ledger = PlaybackLedger(SCOPE, "u", max_queue_ms=160)
        self.args = {"scope": SCOPE, "utterance_id": "u"}
        for seq in range(3):
            self.ledger.generate(OutputPacket(header(), "u", seq, seq*80, (seq+1)*80,
                                 (SpeechSpan(seq*80, (seq+1)*80, str(seq), (f"c{seq}",)),)), 0)

    def test_three_distinct_boundaries(self):
        self.ledger.queue(0, scope=SCOPE, now_ms=0)
        self.assertEqual((self.ledger.generated_until_ms, self.ledger.queued_until_ms, self.ledger.played_until_ms), (240, 80, 0))
        self.assertEqual(self.ledger.said_spans, ())
        self.ledger.acknowledge_played(40, now_ms=40, **self.args)
        self.assertEqual(self.ledger.said_spans, ())
        self.ledger.acknowledge_played(80, now_ms=80, **self.args)
        self.assertEqual(self.ledger.said_claim_ids, ("c0",))

    def test_stop_discards_only_unplayed_and_waits_for_flush(self):
        self.ledger.queue(0, scope=SCOPE, now_ms=0)
        self.ledger.queue(1, scope=SCOPE, now_ms=0)
        self.ledger.acknowledge_played(80, now_ms=80, **self.args)
        self.ledger.request_stop(scope=SCOPE, now_ms=80)
        with self.assertRaises(ContractError):
            self.ledger.queue(2, scope=SCOPE, now_ms=80)
        # Device cache drained another 40 ms; count reality, not the stop request.
        self.ledger.confirm_stopped(120, now_ms=120, **self.args)
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot["generated_until_ms"], 240)
        self.assertEqual(snapshot["queued_until_ms"], 160)
        self.assertEqual(snapshot["played_until_ms"], 120)
        self.assertEqual(snapshot["discarded_unplayed_ms"], 120)
        self.assertEqual(snapshot["said_text"], ["0"])
        with self.assertRaises(ContractError):
            self.ledger.acknowledge_played(160, now_ms=160, **self.args)

    def test_over_queue_and_out_of_order(self):
        with self.assertRaises(ContractError):
            self.ledger.queue(1, scope=SCOPE, now_ms=0)
        self.ledger.queue(0, scope=SCOPE, now_ms=0)
        self.ledger.queue(1, scope=SCOPE, now_ms=0)
        with self.assertRaises(ContractError):
            self.ledger.queue(2, scope=SCOPE, now_ms=0)

    def test_cannot_ack_unqueued_or_too_fast_or_backwards(self):
        with self.assertRaises(ContractError):
            self.ledger.acknowledge_played(1, now_ms=0, **self.args)
        self.ledger.queue(0, scope=SCOPE, now_ms=0)
        with self.assertRaises(ContractError):
            self.ledger.acknowledge_played(80, now_ms=1, **self.args)
        self.ledger.acknowledge_played(80, now_ms=80, **self.args)
        with self.assertRaises(ContractError):
            self.ledger.acknowledge_played(40, now_ms=81, **self.args)

    def test_old_identity_and_utterance_callbacks_rejected(self):
        self.ledger.queue(0, scope=SCOPE, now_ms=0)
        with self.assertRaises(ContractError):
            self.ledger.acknowledge_played(80, scope=Scope("b", "0.1", "s-a"), utterance_id="u", now_ms=80)
        with self.assertRaises(ContractError):
            self.ledger.acknowledge_played(80, scope=SCOPE, utterance_id="old-u", now_ms=80)

    def test_cursors_read_only(self):
        with self.assertRaises(AttributeError):
            self.ledger.played_until_ms = 240

    def test_output_span_and_sequence_validation(self):
        with self.assertRaises(ContractError):
            OutputPacket(header(), "u", 0, 0, 80, (SpeechSpan(0, 100, "X"),))
        with self.assertRaises(ContractError):
            self.ledger.generate(OutputPacket(header(), "u", 4, 240, 320), 0)
