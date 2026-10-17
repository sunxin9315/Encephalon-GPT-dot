"""Offline checks: no GPU, downloaded models, or optional packages required."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from encephalon.diagnostics import (
    GPU_OUTPUT_LIMIT_CHARACTERS,
    GPU_PROBE_TIMEOUT_SECONDS,
    collect_diagnostics,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
# Synthetic structural-test values only; never written into a model lock.
TEST_REVISION = hashlib.sha1(b"offline test revision, not an actual model commit").hexdigest()
TEST_DIGEST = hashlib.sha256(b"offline synthetic asset").hexdigest()


def resolved_manifest_fixture():
    """A structurally resolved fixture, not evidence of a real downloaded asset."""
    return {
        "schema_version": 1,
        "manifest_status": "resolved",
        "profile_id": "rtx5090",
        "models": [{
            "id": "synthetic_test_only", "source": "synthetic offline fixture",
            "revision": TEST_REVISION,
            "weights": [{"path": "model.safetensors", "sha256": TEST_DIGEST}],
            "processor": {"required": True, "revision": TEST_REVISION,
                          "files": [{"path": "processor.json", "sha256": TEST_DIGEST}]},
            "tokenizer": {"required": True, "revision": TEST_REVISION,
                          "files": [{"path": "tokenizer.json", "sha256": TEST_DIGEST}]},
            "codec": {"required": False, "reason": "Synthetic text-only fixture"},
            "license": {"identifier": "test-only", "verification_status": "verified",
                        "files": [{"path": "licenses/test/LICENSE", "sha256": TEST_DIGEST}]},
        }],
    }


class DiagnosticsTests(unittest.TestCase):
    def test_cpu_only_default_does_not_look_up_or_launch_gpu_command(self):
        with patch("encephalon.diagnostics.shutil.which") as which, \
                patch("encephalon.diagnostics.subprocess.run") as run:
            report = collect_diagnostics()
        which.assert_not_called()
        run.assert_not_called()
        self.assertEqual(report["gpu"]["status"], "not_probed")
        self.assertFalse(report["gpu"]["probe_requested"])
        self.assertEqual(report["target"]["operating_system"], "Ubuntu Linux")
        self.assertTrue(report["target"]["target_is_not_detected_hardware"])
        self.assertIn("version", report["python"])
        self.assertIn("system", report["platform"])
        self.assertIn("logical_cpu_count", report["resources"])
        self.assertTrue(all(value is False for value in report["scope"].values()))
        json.dumps(report)

    def test_profile_is_not_substituted_for_detected_environment(self):
        with patch("encephalon.diagnostics.platform.system", return_value="Darwin"):
            report = collect_diagnostics()
        self.assertEqual(report["platform"]["system"], "Darwin")
        self.assertEqual(report["target"]["operating_system"], "Ubuntu Linux")

    def test_resource_failures_produce_unknown_not_false_zero(self):
        with patch("encephalon.diagnostics.os.sysconf", side_effect=OSError("unavailable")), \
                patch("encephalon.diagnostics.shutil.disk_usage", side_effect=OSError("unavailable")):
            report = collect_diagnostics()
        self.assertIsNone(report["resources"]["physical_memory_bytes"])
        self.assertIsNone(report["resources"]["working_filesystem_free_bytes"])

    def test_missing_nvidia_smi_is_not_reported_as_no_gpu(self):
        with patch("encephalon.diagnostics.shutil.which", return_value=None), \
                patch("encephalon.diagnostics.subprocess.run") as run:
            report = collect_diagnostics(probe_gpu=True)
        run.assert_not_called()
        self.assertEqual(report["gpu"]["status"], "unavailable")

    def test_gpu_probe_uses_one_fixed_read_only_bounded_command(self):
        result = subprocess.CompletedProcess([], 0, "NVIDIA GeForce RTX 5090, 32607, 580.99\n", "")
        with patch("encephalon.diagnostics.shutil.which", return_value="/usr/bin/nvidia-smi"), \
                patch("encephalon.diagnostics.subprocess.run", return_value=result) as run:
            report = collect_diagnostics(probe_gpu=True)
        run.assert_called_once_with(
            ["/usr/bin/nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=GPU_PROBE_TIMEOUT_SECONDS,
            check=False, shell=False,
        )
        self.assertEqual(report["gpu"]["status"], "observed")
        self.assertEqual(report["gpu"]["devices"][0]["memory_total_mib"], 32607)
        self.assertIn("does not establish", report["gpu"]["note"])

    def test_gpu_probe_timeout(self):
        with patch("encephalon.diagnostics.shutil.which", return_value="nvidia-smi"), \
                patch("encephalon.diagnostics.subprocess.run",
                      side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            report = collect_diagnostics(probe_gpu=True)
        self.assertEqual(report["gpu"]["status"], "timeout")

    def test_gpu_probe_launch_error(self):
        with patch("encephalon.diagnostics.shutil.which", return_value="nvidia-smi"), \
                patch("encephalon.diagnostics.subprocess.run", side_effect=PermissionError("denied")):
            report = collect_diagnostics(probe_gpu=True)
        self.assertEqual(report["gpu"]["status"], "error")

    def test_gpu_probe_nonzero_exit_and_stderr_are_bounded(self):
        result = subprocess.CompletedProcess([], 1, "", "x" * (GPU_OUTPUT_LIMIT_CHARACTERS + 100))
        with patch("encephalon.diagnostics.shutil.which", return_value="nvidia-smi"), \
                patch("encephalon.diagnostics.subprocess.run", return_value=result):
            report = collect_diagnostics(probe_gpu=True)
        self.assertEqual(report["gpu"]["status"], "error")
        self.assertEqual(len(report["gpu"]["reason"]), GPU_OUTPUT_LIMIT_CHARACTERS)

    def test_gpu_probe_malformed_or_oversized_results_fail_closed(self):
        for output in ("", "a,b,c,d\n", "GPU, N/A, 580.99\n", "GPU, -1, 580.99\n",
                       "GPU, 0, 580.99\n", "x" * (GPU_OUTPUT_LIMIT_CHARACTERS + 1),
                       "GPU, 100, driver\n" * 65):
            with self.subTest(output=output[:40]):
                result = subprocess.CompletedProcess([], 0, output, "")
                with patch("encephalon.diagnostics.shutil.which", return_value="nvidia-smi"), \
                        patch("encephalon.diagnostics.subprocess.run", return_value=result):
                    report = collect_diagnostics(probe_gpu=True)
                self.assertEqual(report["gpu"]["status"], "error")
                self.assertEqual(report["gpu"]["devices"], [])


class ManifestValidationTests(unittest.TestCase):
    def test_template_is_unready_and_all_pins_remain_explicitly_unresolved(self):
        manifest = json.loads((ROOT / "configs/models.lock.template.json").read_text(encoding="utf-8"))
        blockers = validate_manifest(manifest)
        self.assertTrue(any("template" in blocker for blocker in blockers))
        self.assertTrue(any("revision" in blocker for blocker in blockers))
        self.assertTrue(any("sha256" in blocker for blocker in blockers))
        self.assertTrue(any("license" in blocker for blocker in blockers))
        self.assertEqual(len(manifest["models"]), 8)
        for model in manifest["models"]:
            self.assertEqual(model["revision"], "UNRESOLVED")
            self.assertEqual(model["weights"][0]["sha256"], "UNRESOLVED")
            self.assertEqual(model["license"]["files"][0]["path"], "UNRESOLVED")

    def test_structurally_resolved_fixture_has_no_blockers(self):
        fixture = resolved_manifest_fixture()
        original = copy.deepcopy(fixture)
        self.assertEqual(validate_manifest(fixture), [])
        self.assertEqual(fixture, original)

    def test_floating_or_placeholder_revisions_block(self):
        for value in ("main", "master", "latest", "v1.0.0", "refs/tags/v1", "abcdef0",
                      "UNRESOLVED", "0" * 40, "deadbeef" * 5, None, 123):
            with self.subTest(value=value):
                fixture = resolved_manifest_fixture()
                fixture["models"][0]["revision"] = value
                self.assertTrue(any(".revision" in item for item in validate_manifest(fixture)))

    def test_invalid_or_placeholder_sha256_block(self):
        for value in ("UNRESOLVED", "<sha256>", "0" * 64, "a" * 64, "deadbeef" * 8,
                      "g" * 64, TEST_DIGEST[:-1], None, 123):
            with self.subTest(value=value):
                fixture = resolved_manifest_fixture()
                fixture["models"][0]["weights"][0]["sha256"] = value
                self.assertTrue(any("sha256" in item for item in validate_manifest(fixture)))

    def test_all_required_components_need_their_own_pins(self):
        for component in ("processor", "tokenizer", "codec"):
            with self.subTest(component=component):
                fixture = resolved_manifest_fixture()
                fixture["models"][0][component] = {"required": True, "revision": "main", "files": []}
                blockers = validate_manifest(fixture)
                self.assertTrue(any(component + ".revision" in item for item in blockers))
                self.assertTrue(any(component + ".files" in item for item in blockers))

    def test_omitted_component_or_unexplained_exemption_blocks(self):
        fixture = resolved_manifest_fixture()
        del fixture["models"][0]["processor"]
        fixture["models"][0]["codec"] = {"required": False}
        blockers = validate_manifest(fixture)
        self.assertTrue(any("processor" in item for item in blockers))
        self.assertTrue(any("codec.reason" in item for item in blockers))

    def test_license_verification_and_archived_file_are_required(self):
        fixture = resolved_manifest_fixture()
        fixture["models"][0]["license"] = {"identifier": "MIT", "verification_status": "UNRESOLVED"}
        blockers = validate_manifest(fixture)
        self.assertTrue(any("verification_status" in item for item in blockers))
        self.assertTrue(any("license.files" in item for item in blockers))

    def test_asset_paths_cannot_escape_asset_directory(self):
        for path in ("/etc/passwd", "../weights", "a/../../weights", "C:\\weights", ".", "bad\x00name"):
            with self.subTest(path=path):
                fixture = resolved_manifest_fixture()
                fixture["models"][0]["weights"][0]["path"] = path
                self.assertTrue(any(".path" in item for item in validate_manifest(fixture)))

    def test_duplicate_model_ids_and_file_paths_block(self):
        fixture = resolved_manifest_fixture()
        fixture["models"].append(copy.deepcopy(fixture["models"][0]))
        fixture["models"][0]["weights"] *= 2
        blockers = validate_manifest(fixture)
        self.assertTrue(any("id is duplicated" in item for item in blockers))
        self.assertTrue(any("path duplicates" in item for item in blockers))

    def test_malformed_manifest_inputs_return_blockers_instead_of_crashing(self):
        for fixture in (None, [], "bad", {}, {"models": []},
                        {"schema_version": True, "models": [None]},
                        {"models": [{"id": [], "source": {}, "weights": [None]}]}):
            with self.subTest(fixture=fixture):
                self.assertTrue(validate_manifest(fixture))

    def test_profile_distinguishes_ubuntu_target_and_design_assumptions(self):
        profile = json.loads((ROOT / "configs/rtx5090.json").read_text(encoding="utf-8"))
        self.assertEqual(profile["profile_status"], "design_only")
        self.assertFalse(profile["provenance"]["measured_performance"])
        self.assertFalse(profile["provenance"]["hardware_observed"])
        self.assertEqual(profile["target"]["operating_system"], "Ubuntu Linux")
        self.assertEqual(profile["target"]["gpu_memory_gb_vendor_spec"], 32)
        self.assertEqual(profile["state"]["slots"], 64)
        self.assertEqual(profile["state"]["width"], 512)
        self.assertEqual(profile["state"]["tick_ms"], 80)
        self.assertEqual(profile["audio"]["codebooks"], 8)
        self.assertEqual(profile["audio"]["frame_rate_hz"], 12.5)
        self.assertEqual(profile["audio"]["sample_rate_hz"], 24000)
        self.assertEqual(profile["audio"]["committed_history_cursor"], "played")
        self.assertFalse(profile["runtime"]["automatic_model_downloads"])
        self.assertEqual(profile["budgets"]["status"], "planning_not_measured")


if __name__ == "__main__":
    unittest.main()
