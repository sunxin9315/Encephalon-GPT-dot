import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from encephalon.cli import main
from encephalon.contracts import ContractError
from encephalon.replay import run_replay
from encephalon.serialization import record_from_dict, strict_loads

ROOT = Path(__file__).resolve().parents[1]


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = strict_loads((ROOT / "fixtures/offline_replay.json").read_text())

    def test_deterministic_complete_replay(self):
        one = run_replay(self.fixture)
        two = run_replay(copy.deepcopy(self.fixture))
        self.assertEqual(json.dumps(one, sort_keys=True), json.dumps(two, sort_keys=True))
        self.assertTrue(one["passed"])
        self.assertEqual(one["scenario_count"], 4)
        self.assertEqual(one["steps"], 38)
        self.assertEqual(one["assertions"], 44)
        self.assertFalse(one["model_inference_performed"])
        self.assertFalse(one["physical_audio_played"])

    def test_fixture_must_not_claim_real_training_evidence(self):
        for key, value in (("mock", False), ("synthetic", False), ("purpose", "training_data")):
            fixture = copy.deepcopy(self.fixture)
            fixture[key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                run_replay(fixture)

    def test_incorrect_expected_checkpoint_fails_not_silent_pass(self):
        self.fixture["scenarios"][0]["commands"][2]["expect"]["visible"] = ["future leak"]
        with self.assertRaises(ContractError):
            run_replay(self.fixture)

    def test_canonical_fixture_records_all_validate(self):
        lines = (ROOT / "fixtures/canonical_records.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 6)
        for line in lines:
            record_from_dict(strict_loads(line))

    def test_numeric_overflow_and_unhashable_record_types_are_contract_errors(self):
        with self.assertRaises(ContractError):
            strict_loads('{"x":1e9999}')
        for kind in ([], {}, None):
            with self.subTest(kind=kind), self.assertRaises(ContractError):
                record_from_dict({"record_type": kind})


class CliTests(unittest.TestCase):
    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(args)
        return code, out.getvalue(), err.getvalue()

    def test_doctor_default_does_not_probe_gpu(self):
        code, output, _ = self.invoke(["doctor"])
        self.assertEqual(code, 0)
        data = json.loads(output)
        self.assertEqual(data["gpu"]["status"], "not_probed")

    def test_template_manifest_intentionally_blocks(self):
        code, output, _ = self.invoke(["check-manifest", str(ROOT / "configs/models.lock.template.json")])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output)["structurally_ready"])
        self.assertFalse(json.loads(output)["assets_verified"])

    def test_validate_and_replay(self):
        code, output, _ = self.invoke(["validate", str(ROOT / "fixtures/canonical_records.jsonl")])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["records"], 6)
        code, output, _ = self.invoke(["replay", str(ROOT / "fixtures/offline_replay.json")])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output)["passed"])

    def test_malformed_input_fails_nonzero_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"record_type":[]}', '{"a":1,"a":2}', '{"x":1e9999}', '[]', '{]'):
                path.write_text(text)
                code, output, err = self.invoke(["validate", str(path)])
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertIn("error:", err)
                self.assertNotIn("Traceback", err)

    def test_module_entrypoint(self):
        result = subprocess.run([sys.executable, "-m", "encephalon", "--help"], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn("doctor", result.stdout)

    def test_default_resources_are_packaged_and_in_sync(self):
        for source, target in (("fixtures/offline_replay.json", "offline_replay.json"),
                               ("configs/models.lock.template.json", "models.lock.template.json"),
                               ("configs/rtx5090.json", "rtx5090.json")):
            self.assertEqual((ROOT / source).read_bytes(), (ROOT / "encephalon/data" / target).read_bytes())
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            import os
            old = os.getcwd()
            try:
                os.chdir(directory)
                code, output, _ = self.invoke(["replay"])
                self.assertEqual(code, 0)
                code, output, _ = self.invoke(["check-manifest"])
                self.assertEqual(code, 2)
            finally:
                os.chdir(old)

    def test_malformed_replay_commands_are_controlled_errors(self):
        fixture = strict_loads((ROOT / "fixtures/offline_replay.json").read_text())
        for command in ({"action": "queue", "at_ms": 0},
                        {"action": "tick", "at_ms": 0, "future_completion": 100},
                        {"action": [], "at_ms": 0}):
            fixture["scenarios"][0]["commands"] = [command]
            with self.subTest(command=command), self.assertRaises(ContractError):
                run_replay(fixture)
