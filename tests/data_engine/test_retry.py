"""Tests for the call_with_retry helper."""
from __future__ import annotations

import pytest

from rabbit_hunter.data_engine.retry import call_with_retry


class _Flaky:
    def __init__(self, fails_before_success: int, exc: type[Exception] = RuntimeError):
        self.fails = fails_before_success
        self.calls = 0
        self.exc = exc

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc(f"attempt {self.calls} boom")
        return f"ok after {self.calls} tries"


def test_returns_on_first_success():
    f = _Flaky(fails_before_success=0)
    sleeps: list[float] = []
    result = call_with_retry(f, max_attempts=3, sleep=sleeps.append)
    assert result == "ok after 1 tries"
    assert sleeps == []
    assert f.calls == 1


def test_retries_then_succeeds():
    f = _Flaky(fails_before_success=2)
    sleeps: list[float] = []
    result = call_with_retry(
        f, max_attempts=3, base_delay_seconds=0.1,
        max_delay_seconds=10.0, sleep=sleeps.append,
    )
    assert result == "ok after 3 tries"
    # Two retries → two sleeps
    assert len(sleeps) == 2
    # First sleep 0.1, second sleep 0.2 (exponential)
    assert sleeps[0] == pytest.approx(0.1)
    assert sleeps[1] == pytest.approx(0.2)


def test_raises_after_exhausting_attempts():
    f = _Flaky(fails_before_success=99)
    sleeps: list[float] = []
    with pytest.raises(RuntimeError):
        call_with_retry(f, max_attempts=3, sleep=sleeps.append)
    assert f.calls == 3
    assert len(sleeps) == 2


def test_delay_capped_by_max():
    f = _Flaky(fails_before_success=99)
    sleeps: list[float] = []
    with pytest.raises(RuntimeError):
        call_with_retry(
            f, max_attempts=5, base_delay_seconds=1.0,
            max_delay_seconds=2.0, sleep=sleeps.append,
        )
    # sleeps schedule: 1, 2, 2, 2 (all cap at max=2)
    assert sleeps == [1.0, 2.0, 2.0, 2.0]


def test_on_retry_hook_invoked_with_attempt_and_error():
    f = _Flaky(fails_before_success=2)
    calls: list[tuple[int, str]] = []
    call_with_retry(
        f, max_attempts=3, base_delay_seconds=0.1,
        sleep=lambda _s: None,
        on_retry=lambda attempt, err: calls.append((attempt, str(err))),
    )
    assert [c[0] for c in calls] == [1, 2]
    assert "boom" in calls[0][1]


def test_does_not_retry_on_excluded_exception():
    class _NoRetry(Exception): pass
    def fn():
        raise _NoRetry("nope")
    sleeps: list[float] = []
    with pytest.raises(_NoRetry):
        call_with_retry(
            fn, max_attempts=3,
            retry_on=(RuntimeError,), sleep=sleeps.append,
        )
    assert sleeps == []


def test_max_attempts_lt_one_rejected():
    with pytest.raises(ValueError):
        call_with_retry(lambda: 1, max_attempts=0)
