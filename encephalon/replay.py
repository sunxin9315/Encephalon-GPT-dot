"""Deterministic synthetic replay, with no models, networks or audio devices.

The harness owns the future command schedule and expected outputs. Runtime
methods see only a current command/record, never the future completion schedule
or expected labels. Playback acknowledgements here are explicitly simulated.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Mapping
from .contracts import ContractError, Scope, exact_keys, integer
from .runtime import SessionRuntime
from .serialization import record_from_dict


class ReplayFailure(ContractError):
    pass


def _scope(value: Mapping[str, Any]) -> Scope:
    exact_keys(value, {"persona_id", "persona_version", "session_id"})
    return Scope(**value)


def _execute(runtime: SessionRuntime, command: dict) -> dict:
    action = command.get("action")
    arguments = {
        "observe": {"record"}, "tick": set(), "premise": {"premise_id", "revision"},
        "advance_state": set(), "change_question": set(), "claim": {"record"},
        "request": {"record"}, "dispatch": {"request_id"}, "result": {"record"},
        "status": {"request_id"}, "remember": {"record"}, "start_utterance": {"utterance_id"},
        "generate": {"record"}, "queue": {"sequence_id"}, "played": {"until_ms", "utterance_id"},
        "stop": set(), "stopped": {"until_ms", "utterance_id"}, "begin_switch": set(),
        "switch": {"scope", "voice_profile"}, "snapshot": set(),
    }
    if action not in arguments:
        raise ReplayFailure(f"unsupported replay action: {action}")
    exact_keys(command, {"action", "at_ms"} | arguments[action], {"expect"})
    runtime.advance_to(command["at_ms"])
    now = runtime.clock.now_ms
    record_actions = {"observe", "claim", "request", "result", "generate", "remember"}
    record = record_from_dict(command["record"]) if action in record_actions else None
    expected_type = {"observe": "Observation", "claim": "Claim", "request": "AsyncRequest",
                     "result": "AsyncResult", "generate": "OutputPacket", "remember": "MemoryEvent"}
    if action in record_actions and type(record).__name__ != expected_type[action]:
        raise ReplayFailure("action and record_type disagree")
    if action == "observe":
        return {"added": runtime.observations.add(record)}
    if action == "tick":
        deltas = runtime.observations.consume(now)
        visible = runtime.observations.visible(now)
        return {"visible": [[o.observation_id, o.header.revision, o.header.validity.status.value] for o in visible],
                "delta_ids": [d.observation.observation_id for d in deltas],
                "valid_new_event": [d.valid_new_event for d in deltas]}
    if action == "premise":
        runtime.set_premise(command["premise_id"], command["revision"])
        return {"state_version": runtime.state_version}
    if action == "advance_state":
        runtime.advance_state()
        return {"state_version": runtime.state_version}
    if action == "change_question":
        runtime.change_question()
        return {"question_version": runtime.question_version}
    if action == "claim":
        runtime.add_claim(record)
        return {"claim_ids": sorted(runtime.claims)}
    if action == "request":
        runtime.register_request(record)
        return runtime.requests.status_view(record.header.request_id, now)
    if action == "dispatch":
        runtime.requests.dispatch(command["request_id"], now)
        return runtime.requests.status_view(command["request_id"], now)
    if action == "result":
        return asdict(runtime.receive_result(record))
    if action == "status":
        return runtime.requests.status_view(command["request_id"], now)
    if action == "remember":
        return {"added": runtime.memory.put(record, now)}
    if action == "start_utterance":
        runtime.start_utterance(command["utterance_id"])
        return runtime.playback.snapshot()
    if action in {"generate", "queue", "played", "stop", "stopped"}:
        playback = runtime.playback
        if playback is None:
            raise ReplayFailure("playback action without utterance")
        if action == "generate":
            playback.generate(record, now)
        elif action == "queue":
            playback.queue(command["sequence_id"], scope=runtime.scope, now_ms=now)
        elif action == "played":
            playback.acknowledge_played(command["until_ms"], scope=runtime.scope,
                                        utterance_id=command["utterance_id"], now_ms=now)
        elif action == "stop":
            playback.request_stop(scope=runtime.scope, now_ms=now)
        elif action == "stopped":
            playback.confirm_stopped(command["until_ms"], scope=runtime.scope,
                                      utterance_id=command["utterance_id"], now_ms=now)
        return playback.snapshot()
    if action == "begin_switch":
        runtime.begin_switch()
        return runtime.snapshot()
    if action == "switch":
        archived = runtime.commit_switch(_scope(command["scope"]), command["voice_profile"])
        return {"active_scope": runtime.scope.to_dict(), "state_version": runtime.state_version,
                "cache_handle_count": len(runtime.cache_handles), "archived_memory_count": len(archived["memory_versions"]),
                "archived_playback": archived["playback"]}
    if action == "snapshot":
        return runtime.snapshot()
    raise ReplayFailure(f"unsupported replay action: {action}")


def run_replay(fixture: Mapping[str, Any]) -> dict:
    exact_keys(fixture, {"schema_version", "mock", "synthetic", "purpose", "scenarios"})
    if fixture["schema_version"] != "encephalon.replay.v1" or fixture["mock"] is not True or fixture["synthetic"] is not True:
        raise ReplayFailure("replay fixture must explicitly identify itself as synthetic/mock v1")
    if fixture["purpose"] != "engineering_contract_test_not_training_data":
        raise ReplayFailure("synthetic fixture must not masquerade as training or evaluation evidence")
    if not isinstance(fixture["scenarios"], list) or not fixture["scenarios"]:
        raise ReplayFailure("at least one scenario is required")
    outputs, names = [], set()
    total_steps = total_assertions = 0
    for scenario in fixture["scenarios"]:
        exact_keys(scenario, {"scenario_id", "scope", "voice_profile", "commands"})
        name = scenario["scenario_id"]
        if not isinstance(name, str) or not name or name in names:
            raise ReplayFailure("scenario_id must be nonempty and unique")
        names.add(name)
        if not isinstance(scenario["commands"], list) or not scenario["commands"]:
            raise ReplayFailure("scenario needs nonempty commands")
        runtime = SessionRuntime(_scope(scenario["scope"]), scenario["voice_profile"])
        checkpoints = []
        assertions = 0
        for index, command in enumerate(scenario["commands"]):
            if not isinstance(command, dict) or "at_ms" not in command or "action" not in command:
                raise ReplayFailure(f"{name}[{index}]: invalid replay command")
            integer(command["at_ms"], "at_ms")
            if not isinstance(command["action"], str):
                raise ReplayFailure("action must be a string")
            actual = _execute(runtime, command)
            expected = command.get("expect", {})
            if not isinstance(expected, dict):
                raise ReplayFailure("expect must be an object")
            for key, value in expected.items():
                if key not in actual or actual[key] != value:
                    raise ReplayFailure(f"{name}[{index}] {key}: expected {value!r}, got {actual.get(key)!r}")
                assertions += 1
            if expected:
                checkpoints.append({"step": index, "action": command["action"], "verified": expected})
        if assertions == 0:
            raise ReplayFailure("scenario must contain explicit checks")
        total_steps += len(scenario["commands"])
        total_assertions += assertions
        outputs.append({"scenario_id": name, "steps": len(scenario["commands"]), "assertions": assertions,
                        "checkpoints": checkpoints, "final_state": runtime.snapshot()})
    return {"schema_version": "encephalon.replay_report.v1", "passed": True, "synthetic": True,
            "model_inference_performed": False, "physical_audio_played": False,
            "scenario_count": len(outputs), "steps": total_steps, "assertions": total_assertions,
            "scenarios": outputs}
