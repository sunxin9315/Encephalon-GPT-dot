"""Offline environment reporting and structural model-lock validation.

Neither entry point imports a tensor framework, downloads assets, initializes
CUDA, or contacts a network service. A successful lock check is structural only:
it does not prove that a hash matches bytes or that a model license permits use.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import os
from pathlib import PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


GPU_PROBE_TIMEOUT_SECONDS = 5
GPU_OUTPUT_LIMIT_CHARACTERS = 16_384
GPU_MAX_DEVICES = 64
_HEX_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")
_HEX_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_PLACEHOLDERS = {"unresolved", "unknown", "todo", "tbd", "placeholder", "none", "null"}


def _memory_info() -> dict[str, Any]:
    """Report OS physical memory, not an assertion about container allocation."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages <= 0 or page_size <= 0:
            raise ValueError("non-positive physical-memory result")
        return {
            "physical_memory_bytes": pages * page_size,
            "memory_source": "os.sysconf physical pages; not cgroup/quota adjusted",
        }
    except (AttributeError, OSError, ValueError):
        return {"physical_memory_bytes": None, "memory_source": "unavailable"}


def _gpu_diagnostics() -> dict[str, Any]:
    """Run one fixed read-only query, with a timeout and retained-output bounds."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"probe_requested": True, "status": "unavailable", "devices": [],
                "reason": "nvidia-smi was not found on PATH"}
    command = [executable, "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=GPU_PROBE_TIMEOUT_SECONDS,
            check=False, shell=False,
        )
    except subprocess.TimeoutExpired:
        return {"probe_requested": True, "status": "timeout", "devices": [],
                "reason": "nvidia-smi exceeded the 5-second probe timeout"}
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        return {"probe_requested": True, "status": "error", "devices": [],
                "reason": str(exc)[:GPU_OUTPUT_LIMIT_CHARACTERS]}
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        return {"probe_requested": True, "status": "error", "devices": [],
                "returncode": result.returncode,
                "reason": stderr[:GPU_OUTPUT_LIMIT_CHARACTERS] or "nvidia-smi failed"}
    if len(stdout) > GPU_OUTPUT_LIMIT_CHARACTERS:
        return {"probe_requested": True, "status": "error", "devices": [],
                "reason": "nvidia-smi output exceeds the diagnostic size limit"}
    devices = []
    try:
        rows = list(csv.reader(io.StringIO(stdout)))
        if not rows or len(rows) > GPU_MAX_DEVICES:
            raise ValueError("unexpected number of GPU records")
        for row in rows:
            if len(row) != 3:
                raise ValueError("expected name, memory.total, driver_version")
            name, memory_mib, driver = (value.strip() for value in row)
            if not name or not driver or not memory_mib.isdecimal() or int(memory_mib) <= 0:
                raise ValueError("invalid GPU name, memory size, or driver version")
            devices.append({"name": name, "memory_total_mib": int(memory_mib),
                            "driver_version": driver})
    except (csv.Error, ValueError) as exc:
        return {"probe_requested": True, "status": "error", "devices": [],
                "reason": "Could not parse nvidia-smi output: " + str(exc)}
    return {"probe_requested": True, "status": "observed", "devices": devices,
            "source": "nvidia-smi read-only query",
            "note": "Device visibility does not establish CUDA, PyTorch, or kernel compatibility."}


def collect_diagnostics(probe_gpu: bool = False) -> dict[str, Any]:
    """Return actual local observations, separate from the Ubuntu/5090 target.

    The default is CPU-only. Callers must explicitly opt into the local GPU
    query, for example with a documented ``--probe-gpu`` CLI option.
    """
    resources: dict[str, Any] = {"logical_cpu_count": os.cpu_count(), **_memory_info()}
    try:
        usage = shutil.disk_usage(os.getcwd())
        resources.update({"working_filesystem_total_bytes": usage.total,
                          "working_filesystem_free_bytes": usage.free})
    except OSError:
        resources.update({"working_filesystem_total_bytes": None,
                          "working_filesystem_free_bytes": None})
    gpu = _gpu_diagnostics() if probe_gpu else {
        "probe_requested": False, "status": "not_probed", "devices": [],
        "reason": "CPU-only default; use --probe-gpu to request a local read-only query",
    }
    return {
        "schema_version": 1,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {"profile_id": "rtx5090", "operating_system": "Ubuntu Linux",
                   "gpu": "NVIDIA GeForce RTX 5090", "target_is_not_detected_hardware": True},
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation(),
                   "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "resources": resources,
        "gpu": gpu,
        "scope": {"network_access": False, "model_downloads": False,
                  "tensor_framework_imports": False, "model_forward": False,
                  "performance_benchmark": False},
        "limitations": [
            "This report is not a model-readiness or RTX 5090 compatibility certificate.",
            "Target profile dimensions, budgets, and latency goals are design assumptions, not observations.",
            "Memory and CPU counts may describe the host rather than container limits.",
        ],
    }


def _is_placeholder(value: Any) -> bool:
    return (not isinstance(value, str) or not value.strip()
            or value.strip().lower() in _PLACEHOLDERS
            or "unresolved" in value.lower()
            or "<" in value or ">" in value)


def _valid_hex(value: Any, pattern: re.Pattern[str]) -> bool:
    # Obvious dummy digests such as 000... or deadbeef repeated are not pins.
    if not isinstance(value, str) or not pattern.fullmatch(value):
        return False
    value = value.lower()
    return not any(len(value) % length == 0 and value == value[:length] * (len(value) // length)
                   for length in range(1, 9))


def _validate_files(files: Any, label: str, blockers: list[str]) -> None:
    if not isinstance(files, list) or not files:
        blockers.append(f"{label} must contain at least one pinned file")
        return
    seen_paths: set[str] = set()
    for index, item in enumerate(files):
        key = f"{label}[{index}]"
        if not isinstance(item, dict):
            blockers.append(f"{key} must be an object")
            continue
        path = item.get("path")
        if _is_placeholder(path):
            blockers.append(f"{key}.path is unresolved")
        elif (PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
              or "\\" in path or ":" in path or "\x00" in path or path == "."):
            blockers.append(f"{key}.path must be a relative asset path without traversal")
        elif path in seen_paths:
            blockers.append(f"{key}.path duplicates another file")
        else:
            seen_paths.add(path)
        if not _valid_hex(item.get("sha256"), _HEX_SHA256):
            blockers.append(f"{key}.sha256 must be a resolved 64-character SHA256, not a placeholder")


def _validate_revision(revision: Any, label: str, blockers: list[str]) -> None:
    if not _valid_hex(revision, _HEX_REVISION):
        blockers.append(f"{label} must be an immutable full 40- or 64-character commit hash; "
                        "floating branches/tags and UNRESOLVED are not accepted")


def validate_manifest(manifest: Any) -> list[str]:
    """Return all structural blockers to considering a manifest resolved.

    Templates are intentionally blocked. Branches and tags (even release tags)
    are not immutable revisions. All required processor/tokenizer/codec assets
    and archived license files need independent file hashes. Non-applicable
    components require an explicit reason, rather than silently missing pins.
    This function does not open files or verify hashes against model bytes.
    """
    blockers: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
        blockers.append("schema_version must be the integer 1")
    status = manifest.get("manifest_status")
    if status == "template":
        blockers.append("manifest_status is template, not a resolved model lock")
    elif status != "resolved":
        blockers.append("manifest_status must be resolved (or template for an unready template)")
    if manifest.get("profile_id") != "rtx5090":
        blockers.append("profile_id must be rtx5090")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        blockers.append("models must be a nonempty list")
        return blockers
    ids: set[str] = set()
    for index, model in enumerate(models):
        label = f"models[{index}]"
        if not isinstance(model, dict):
            blockers.append(f"{label} must be an object")
            continue
        model_id = model.get("id")
        if _is_placeholder(model_id):
            blockers.append(f"{label}.id is required")
        elif model_id in ids:
            blockers.append(f"{label}.id is duplicated")
        else:
            ids.add(model_id)
        source = model.get("source")
        if _is_placeholder(source):
            blockers.append(f"{label}.source is unresolved")
        _validate_revision(model.get("revision"), f"{label}.revision", blockers)
        _validate_files(model.get("weights"), f"{label}.weights", blockers)
        for name in ("processor", "tokenizer", "codec"):
            component = model.get(name)
            component_label = f"{label}.{name}"
            if not isinstance(component, dict) or type(component.get("required")) is not bool:
                blockers.append(f"{component_label} needs an explicit required boolean")
                continue
            if component["required"]:
                _validate_revision(component.get("revision"), f"{component_label}.revision", blockers)
                _validate_files(component.get("files"), f"{component_label}.files", blockers)
            elif _is_placeholder(component.get("reason")):
                blockers.append(f"{component_label}.reason must explain why it is not applicable")
        license_record = model.get("license")
        if not isinstance(license_record, dict):
            blockers.append(f"{label}.license must archive the weight license")
            continue
        if license_record.get("verification_status") != "verified":
            blockers.append(f"{label}.license.verification_status must be verified against the imported weights")
        if _is_placeholder(license_record.get("identifier")):
            blockers.append(f"{label}.license.identifier is unresolved")
        _validate_files(license_record.get("files"), f"{label}.license.files", blockers)
    return blockers
