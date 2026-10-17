import unittest
from encephalon.clock import ReplayClock, SessionClock
from encephalon.contracts import ContractError, Observation, Scope
from encephalon.observations import CapacityError, ObservationBuffer
from tests.helpers import SCOPE, header


class ObservationTests(unittest.TestCase):
    def obs(self, revision=0, at=100, status="valid", event_id="win", **kwargs):
        return Observation(header(at=at, end=80, revision=revision, status=status,
                                  reason=None if status == "valid" else "simulated fault", **kwargs), event_id, "speech_semantic", {"text": str(revision)})

    def test_availability_not_event_time(self):
        buffer = ObservationBuffer(SCOPE)
        buffer.add(self.obs())
        self.assertEqual(buffer.visible(80), ())
        self.assertEqual(buffer.visible(99), ())
        self.assertEqual(len(buffer.visible(100)), 1)

    def test_future_revision_does_not_hide_old_visible_revision(self):
        buffer = ObservationBuffer(SCOPE)
        buffer.add(self.obs(1, 200))
        buffer.add(self.obs(0, 100))
        self.assertEqual(buffer.visible(100)[0].header.revision, 0)
        self.assertEqual(buffer.visible(200)[0].header.revision, 1)

    def test_repeat_refresh_does_not_retrigger_event(self):
        buffer = ObservationBuffer(SCOPE)
        self.assertTrue(buffer.add(self.obs()))
        self.assertFalse(buffer.add(self.obs()))
        first = buffer.consume(100)
        self.assertEqual(first[0].change, "inserted")
        self.assertTrue(first[0].valid_new_event)
        self.assertEqual(buffer.consume(101), ())
        buffer.add(self.obs(1, 200))
        self.assertEqual(buffer.consume(200)[0].change, "revised")
        self.assertEqual(buffer.consume(201), ())

    def test_corrupt_revision_invalidates_old_valid_without_stimulus(self):
        buffer = ObservationBuffer(SCOPE)
        buffer.add(self.obs())
        buffer.consume(100)
        buffer.add(self.obs(1, 120, "corrupt"))
        delta = buffer.consume(120)[0]
        self.assertFalse(delta.valid_new_event)
        self.assertEqual(delta.observation.header.validity.status.value, "corrupt")

    def test_identity_and_conflicting_revision_rejected(self):
        buffer = ObservationBuffer(SCOPE)
        buffer.add(self.obs())
        with self.assertRaises(ContractError):
            buffer.add(self.obs(scope=Scope("other", "0.1", "s-a")))
        with self.assertRaises(ContractError):
            buffer.add(self.obs(at=101))

    def test_revision_cannot_precede_previous_availability(self):
        buffer = ObservationBuffer(SCOPE)
        buffer.add(self.obs(1, 200))
        with self.assertRaises(ContractError):
            buffer.add(self.obs(2, 150))

    def test_buffer_bounded_and_pruning_preserves_future(self):
        buffer = ObservationBuffer(SCOPE, max_records=2)
        buffer.add(self.obs())
        buffer.add(self.obs(1, 200))
        buffer.consume(100)
        self.assertEqual(buffer.prune_before(300), 0)
        with self.assertRaises(CapacityError):
            buffer.add(self.obs(2, 300))
        buffer.consume(200)
        self.assertEqual(buffer.prune_before(300), 1)
        buffer.add(self.obs(2, 300))

    def test_backward_clock_rejected(self):
        clock = ReplayClock()
        clock.advance_to(10)
        with self.assertRaises(ContractError):
            clock.advance_to(9)
        buffer = ObservationBuffer(SCOPE)
        buffer.visible(10)
        with self.assertRaises(ContractError):
            buffer.visible(9)
        self.assertGreaterEqual(SessionClock().now_ms, 0)
