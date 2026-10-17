"""Offline-first commands; optional GPU probe never initializes a model."""
from __future__ import annotations
import argparse
import json
from importlib import resources
from pathlib import Path
import sys
from .contracts import ContractError
from .diagnostics import collect_diagnostics, validate_manifest
from .replay import run_replay
from .serialization import record_from_dict, strict_loads


def _load(path: str):
    return strict_loads(Path(path).read_text(encoding="utf-8"))


def _load_resource(path: str | None, filename: str):
    if path is not None:
        return _load(path)
    return strict_loads(resources.files("encephalon").joinpath("data", filename).read_text(encoding="utf-8"))


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Encephalon S0a: offline engineering foundation, not model inference")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="CPU-only environment diagnostics")
    doctor.add_argument("--probe-gpu", action="store_true", help="opt in to bounded read-only nvidia-smi; no CUDA/model execution")
    replay = sub.add_parser("replay", help="run synthetic deterministic replay")
    replay.add_argument("fixture", nargs="?", default=None)
    validate = sub.add_parser("validate", help="validate canonical JSON or JSONL records")
    validate.add_argument("records")
    manifest = sub.add_parser("check-manifest", help="structural lock readiness check, not asset-byte/license verification")
    manifest.add_argument("manifest", nargs="?", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            _print(collect_diagnostics(probe_gpu=args.probe_gpu))
        elif args.command == "replay":
            _print(run_replay(_load_resource(args.fixture, "offline_replay.json")))
        elif args.command == "validate":
            path = Path(args.records)
            if path.suffix.lower() == ".jsonl":
                records = [strict_loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                data = _load(args.records)
                records = data if isinstance(data, list) else [data]
            if not records:
                raise ContractError("no records to validate")
            for index, record in enumerate(records):
                try:
                    record_from_dict(record)
                except ContractError as exc:
                    raise ContractError(f"record {index}: {exc}") from exc
            _print({"valid": True, "records": len(records), "scope": "structure_only_not_fact_verification"})
        elif args.command == "check-manifest":
            blockers = validate_manifest(_load_resource(args.manifest, "models.lock.template.json"))
            _print({"structurally_ready": not blockers, "assets_verified": False,
                    "models_loaded": False, "blocker_count": len(blockers), "blockers": blockers})
            return 2 if blockers else 0
    except (OSError, ContractError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
