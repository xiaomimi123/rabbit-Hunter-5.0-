"""Retry-with-backoff wrapper for exchange REST calls.

OKX (and every other exchange) periodically drops requests — rate limits,
transient 5xx, network flake. Callers that swallow those and return None
end up with silent gaps in the parquet history; callers that raise abort
long-running fetches. The right primitive is bounded retry with
exponential backoff.

Design:
  - Attempts is `max_attempts` inclusive of the first try.
  - Base delay doubles each attempt (0.5s, 1s, 2s, ...) capped at
    `max_delay_seconds`.
  - Only retries on Exception types listed in `retry_on` (default: any).
    A caller that needs to skip retries on, say, a 4xx auth error can
    pass a narrower tuple.
  - Passes structured retry telemetry to `on_retry` (attempt, error) so
    the caller can log / count without the wrapper picking a logger.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn()`; on `retry_on` exceptions, sleep + retry up to `max_attempts`.

    Raises the last exception after exhausting retries. `sleep` is injectable
    for tests — a test can pass a lambda that records durations instead of
    actually sleeping.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be ≥1")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retry_on as e:
            last_error = e
            if attempt >= max_attempts:
                raise
            delay = min(base_delay_seconds * (2 ** (attempt - 1)),
                        max_delay_seconds)
            if on_retry is not None:
                on_retry(attempt, e)
            sleep(delay)
    # unreachable — the loop either returns or raises
    assert last_error is not None
    raise last_error
