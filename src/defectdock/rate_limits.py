"""Bounded, process-local request budgets for the single-worker network API."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

RateGroup = Literal["requests", "mutations", "auth_failures"]
RATE_LIMIT_ENV = {
    "requests": "DEFECTDOCK_REQUESTS_PER_MINUTE",
    "mutations": "DEFECTDOCK_MUTATIONS_PER_MINUTE",
    "auth_failures": "DEFECTDOCK_AUTH_FAILURES_PER_MINUTE",
}


@dataclass(frozen=True)
class RateLimitSettings:
    requests: int = 120
    mutations: int = 30
    auth_failures: int = 20

    def __post_init__(self) -> None:
        for group, variable in RATE_LIMIT_ENV.items():
            value = getattr(self, group)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1_000_000:
                raise ValueError(f"{variable} must be an integer between 1 and 1000000")

    @classmethod
    def from_env(cls, environment: Mapping[str, str]) -> "RateLimitSettings":
        values = {}
        for group, variable in RATE_LIMIT_ENV.items():
            if variable in environment:
                try:
                    values[group] = int(environment[variable])
                except ValueError as exc:
                    raise ValueError(f"{variable} must be an integer") from exc
        return cls(**values)

    def to_dict(self) -> dict[str, int]:
        return {group: getattr(self, group) for group in RATE_LIMIT_ENV}


@dataclass
class _Bucket:
    capacity: int
    tokens: float
    updated_at: float


class RequestRateLimiter:
    """Three token buckets, shared across clients, paths and forwarded headers.

    Capacity equals the per-minute budget; tokens refill continuously. A write
    must reserve both request and mutation tokens atomically. No client/path
    dictionary is retained, so attacker-controlled keys cannot grow memory.
    """

    def __init__(
        self, settings: RateLimitSettings, *, clock: Callable[[], float] | None = None
    ) -> None:
        self._clock = clock or time.monotonic
        now = self._clock()
        self._buckets = {
            group: _Bucket(capacity, float(capacity), now)
            for group, capacity in settings.to_dict().items()
        }
        self._lock = threading.Lock()

    def acquire(self, *groups: RateGroup) -> int:
        """Return 0 on admission, otherwise whole seconds until budget is available."""
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("Specify distinct rate-limit groups")
        with self._lock:
            buckets = [self._buckets[group] for group in groups]
            now = self._clock()
            retry_after = 0.0
            for bucket in buckets:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.capacity / 60)
                bucket.updated_at = max(now, bucket.updated_at)
                if bucket.tokens < 1:
                    retry_after = max(retry_after, (1 - bucket.tokens) * 60 / bucket.capacity)
            if retry_after > 0:
                return max(1, math.ceil(retry_after))
            for bucket in buckets:
                bucket.tokens -= 1
            return 0
