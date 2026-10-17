"""Strict JSON boundary for the public runtime records.

Unknown keys, duplicate JSON keys, nonfinite numbers and local aliases are
rejected. Importers must explicitly map old field names before this boundary.
"""
from __future__ import annotations
from dataclasses import fields
import json
import math
from typing import Any, Mapping
from .contracts import (AsyncRequest, AsyncResult, Claim, ContractError, Header, Observation,
                        exact_keys, thaw_json)
from .memory import MemoryEvent
from .playback import OutputPacket, SpeechSpan


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ContractError(f"nonfinite JSON number: {value}")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ContractError("JSON numeric overflow")
    return number


def strict_loads(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON: {exc.msg} (line {exc.lineno})") from exc


def record_from_dict(data: Mapping[str, Any]):
    if not isinstance(data, Mapping):
        raise ContractError("record must be an object")
    kind = data.get("record_type")
    if not isinstance(kind, str):
        raise ContractError("record_type must be a string")
    if kind == "observation":
        return Observation.from_dict(data)
    if kind == "async_result":
        return AsyncResult.from_dict(data)
    record_types = {"claim": Claim, "async_request": AsyncRequest, "output_packet": OutputPacket, "memory_event": MemoryEvent}
    if kind not in record_types:
        raise ContractError(f"unsupported record_type: {kind}")
    cls = record_types[kind]
    extra = {f.name for f in fields(cls)} - {"header"}
    header_keys = {f.name for f in fields(Header)}
    exact_keys(data, extra | header_keys | {"record_type"})
    header = Header.from_dict({k: data[k] for k in header_keys})
    values = {k: data[k] for k in extra}
    if kind == "output_packet":
        if not isinstance(values["spans"], list):
            raise ContractError("spans must be a JSON array")
        spans = []
        for span in values["spans"]:
            exact_keys(span, {f.name for f in fields(SpeechSpan)})
            spans.append(SpeechSpan(**span))
        values["spans"] = tuple(spans)
    return cls(header=header, **values)


def record_to_dict(record) -> dict[str, Any]:
    if isinstance(record, (Observation, AsyncResult)):
        return record.to_dict()
    types = {Claim: "claim", AsyncRequest: "async_request", OutputPacket: "output_packet", MemoryEvent: "memory_event"}
    if type(record) not in types:
        raise ContractError("unsupported record object")
    values = {f.name: getattr(record, f.name) for f in fields(record) if f.name != "header"}
    if isinstance(record, Claim):
        values["evidence_status"] = record.evidence_status.value
    if isinstance(record, MemoryEvent):
        values["assertion_kind"] = record.assertion_kind.value
    if isinstance(record, OutputPacket):
        values["spans"] = [{f.name: getattr(s, f.name) for f in fields(s)} for s in record.spans]
    return thaw_json({**record.header.to_dict(), "record_type": types[type(record)], **values})


def dumps_record(record) -> str:
    return json.dumps(record_to_dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False)
