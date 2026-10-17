"""Separate online monotonic elapsed time from manually advanced offline replay."""
from __future__ import annotations
import time
from .contracts import ContractError, integer


class ReplayClock:
    def __init__(self, now_ms: int = 0):
        self._now_ms = integer(now_ms, "now_ms")

    @property
    def now_ms(self) -> int:
        return self._now_ms

    def advance_to(self, now_ms: int) -> int:
        integer(now_ms, "now_ms")
        if now_ms < self._now_ms:
            raise ContractError("session clock cannot move backward")
        self._now_ms = now_ms
        return now_ms


class SessionClock:
    def __init__(self):
        self._origin_ns = time.monotonic_ns()

    @property
    def now_ms(self) -> int:
        return (time.monotonic_ns() - self._origin_ns) // 1_000_000
