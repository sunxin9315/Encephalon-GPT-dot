import json
import unittest
from dataclasses import FrozenInstanceError, replace

from encephalon.contracts import (AsyncRequest, AsyncResult, Claim, ContractError, Observation,
                                 Scope, Validity, normalize_legacy_validity)
from encephalon.serialization import dumps_record, record_from_dict, strict_loads
from encephalon.playback import OutputPacket, SpeechSpan
from encephalon.memory import MemoryEvent
from tests.helpers import header


class ContractsTests(unittest.TestCase):
    def test_observation_json_roundtrip(self):
        observation = Observation(header(at=10, end=5), "window-1", "audio", {"energy": 0, "samples_uri": "fixture.wav"})
        self.assertEqual(record_from_dict(strict_loads(dumps_record(observation))), observation)

    def test_all_record_types_roundtrip(self):
        request_header = header(request_id="r")
        records = [Claim(header(), "c", "待核验", "pending", (), True, 100),
                   AsyncRequest(request_header, 0, 0, ("c",), {"input": 1}, 100),
                   AsyncResult(request_header, 0, 0, ("c",), {"input": 1}, 100, {"text": "候选"}),
                   OutputPacket(header(), "u", 0, 0, 80, (SpeechSpan(0, 80, "你好", ("c",)),)),
                   MemoryEvent(header(), "m", "e", "用户说 X", "reported")]
        for record in records:
            with self.subTest(record=type(record).__name__):
                self.assertEqual(record_from_dict(strict_loads(dumps_record(record))), record)

    def test_canonical_only_and_explicit_legacy_migration(self):
        with self.assertRaises(ContractError):
            Validity("present", 1, 0)
        self.assertEqual(normalize_legacy_validity("present", quality=1, age_ms=0).status.value, "valid")
        with self.assertRaises(ContractError):
            normalize_legacy_validity("failed", quality=0, age_ms=0)
        validity = normalize_legacy_validity("failed", quality=0, age_ms=0, failed_as="corrupt", reason="checksum mismatch")
        self.assertEqual(validity.status.value, "corrupt")

    def test_valid_silence_is_not_missing(self):
        silent = Observation(header(), "silence", "audio", {"energy": 0.0})
        self.assertEqual(silent.header.validity.status.value, "valid")

    def test_monotonic_time_fields(self):
        for kwargs in ({"start": 10, "end": 9}, {"at": 4, "end": 5}, {"at": -1}, {"at": True}, {"at": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                header(**kwargs)

    def test_strict_validity_quality_and_reason(self):
        for quality in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.subTest(quality=quality), self.assertRaises(ContractError):
                Validity("valid", quality, 0)
        with self.assertRaises(ContractError):
            Validity("missing", 0, 0)
        with self.assertRaises(ContractError):
            replace(header(), validity=Validity("valid", 1, 20))

    def test_payload_is_detached_and_immutable(self):
        data = {"items": [{"v": 1}]}
        observation = Observation(header(), "o", "text", data)
        data["items"][0]["v"] = 2
        self.assertEqual(observation.payload["items"][0]["v"], 1)
        with self.assertRaises(TypeError):
            observation.payload["items"][0]["v"] = 3
        with self.assertRaises(FrozenInstanceError):
            observation.modality = "audio"

    def test_non_json_payload_rejected(self):
        for payload in ({"bad": float("nan")}, {"binary": b"abc"}, {1: "key"}, {"set": {1, 2}}):
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                Observation(header(), "o", "text", payload)

    def test_unknown_alias_and_schema_rejected(self):
        record = Observation(header(), "o", "text", {}).to_dict()
        record["arrival_ms"] = 0
        with self.assertRaises(ContractError):
            record_from_dict(record)
        del record["arrival_ms"]
        record["schema_version"] = "other"
        with self.assertRaises(ContractError):
            record_from_dict(record)

    def test_json_duplicate_keys_nan_and_infinity_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{]'):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                strict_loads(raw)

    def test_supported_claim_needs_evidence_and_expiry(self):
        with self.assertRaises(ContractError):
            Claim(header(), "c", "X", "supported", (), False, 100)
        claim = Claim(header(at=10), "c", "X", "supported", ("source",), False, 100)
        self.assertFalse(claim.can_assert(9))
        self.assertTrue(claim.can_assert(10))
        self.assertTrue(claim.can_assert(100))
        self.assertFalse(claim.can_assert(101))

    def test_scope_and_sources_nonempty(self):
        with self.assertRaises(ContractError):
            Scope("", "1", "s")
        with self.assertRaises(ContractError):
            header(source_ids=())
        with self.assertRaises(ContractError):
            header(source_ids=("s", "s"))

    def test_async_needs_request_id_and_unique_claims(self):
        with self.assertRaises(ContractError):
            AsyncRequest(header(), 0, 0, ("c",), {}, 100)
        with self.assertRaises(ContractError):
            AsyncRequest(header(request_id="r"), 0, 0, ("c", "c"), {}, 100)

    def test_observation_request_id_forbidden(self):
        with self.assertRaises(ContractError):
            Observation(header(request_id="r"), "o", "text", {})
