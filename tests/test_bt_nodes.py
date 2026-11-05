"""Behavior-tree node semantics using scripted leaves (no simulator)."""

from __future__ import annotations

import pytest

from robot_skills.bt import (
    Blackboard,
    Condition,
    Fallback,
    Node,
    Retry,
    Selector,
    Sequence,
    TickContext,
    Timeout,
)
from robot_skills.core import ErrorCode, SkillError
from robot_skills.core.status import Status

S, F, R = Status.SUCCESS, Status.FAILURE, Status.RUNNING


class Scripted(Node):
    """Returns a scripted status per tick; FAILURE uses `code`."""

    def __init__(self, name, script, code=ErrorCode.GRASP_NOT_VERIFIED, repeat_last=True):
        super().__init__(name)
        self.script, self.code, self.repeat_last = list(script), code, repeat_last
        self.ticks = 0
        self.halted = 0

    def _tick(self, ctx):
        i = min(self.ticks, len(self.script) - 1)
        self.ticks += 1
        s = self.script[i]
        if s is F:
            return self._fail(SkillError(self.code, f"{self.name} failed", self.name))
        return s

    def halt(self, ctx):
        self.halted += 1
        super().halt(ctx)


def run_until_done(node, ctx, max_ticks=100, dt=0.1):
    for _ in range(max_ticks):
        s = node.tick(ctx)
        if s is not R:
            return s
        ctx.now = round(ctx.now + dt, 9)
    raise AssertionError("tree did not terminate")


@pytest.fixture
def ctx():
    return TickContext(now=0.0, blackboard=Blackboard())


def test_sequence_success_and_failure(ctx):
    assert Sequence("s", [Scripted("a", [S]), Scripted("b", [S])]).tick(ctx) is S
    seq = Sequence("s", [Scripted("a", [S]), Scripted("b", [F], ErrorCode.NAV_ABORTED), Scripted("c", [S])])
    assert seq.tick(ctx) is F
    assert seq.error.code is ErrorCode.NAV_ABORTED


def test_sequence_resumes_running_child_without_reticking_earlier(ctx):
    a, b = Scripted("a", [S]), Scripted("b", [R, R, S])
    seq = Sequence("s", [a, b])
    assert run_until_done(seq, ctx) is S
    assert a.ticks == 1 and b.ticks == 3


def test_selector_returns_first_success_or_last_error(ctx):
    assert Selector("sel", [Scripted("a", [F]), Scripted("b", [S])]).tick(ctx) is S
    sel = Selector("sel", [Scripted("a", [F], ErrorCode.NAV_ABORTED), Scripted("b", [F], ErrorCode.OBJECT_NOT_FOUND)])
    assert sel.tick(ctx) is F
    assert sel.error.code is ErrorCode.OBJECT_NOT_FOUND


def test_empty_composites_rejected():
    with pytest.raises(ValueError):
        Sequence("s", [])
    with pytest.raises(ValueError):
        Selector("s", [])


def test_retry_succeeds_after_failures(ctx):
    child = Scripted("c", [F, F, S])
    r = Retry("r", child, max_attempts=3)
    assert run_until_done(r, ctx) is S
    assert r.attempt == 3


def test_retry_exhaustion_is_bounded_and_keeps_cause(ctx):
    child = Scripted("c", [F])
    r = Retry("r", child, max_attempts=4)
    assert run_until_done(r, ctx) is F
    assert child.ticks == 4
    assert r.error.code is ErrorCode.RETRIES_EXHAUSTED
    assert r.error.cause.code is ErrorCode.GRASP_NOT_VERIFIED


def test_retry_does_not_retry_non_retryable(ctx):
    child = Scripted("c", [F], ErrorCode.INVALID_TARGET)
    r = Retry("r", child, max_attempts=5)
    assert run_until_done(r, ctx) is F
    assert child.ticks == 1
    assert r.error.code is ErrorCode.INVALID_TARGET


def test_retry_only_retries_listed_codes(ctx):
    child = Scripted("c", [F], ErrorCode.OUT_OF_REACH)
    r = Retry("r", child, max_attempts=5, retry_on=frozenset({ErrorCode.GRASP_NOT_VERIFIED}))
    assert run_until_done(r, ctx) is F
    assert child.ticks == 1 and r.error.code is ErrorCode.OUT_OF_REACH


def test_retry_runs_recovery_between_attempts(ctx):
    child = Scripted("c", [F, S])
    rec = Scripted("rec", [R, S])
    r = Retry("r", child, max_attempts=3, recovery=rec)
    assert run_until_done(r, ctx) is S
    assert rec.ticks == 2
    assert ctx.blackboard["recovery_request"].error.code is ErrorCode.GRASP_NOT_VERIFIED


def test_retry_fails_when_recovery_fails(ctx):
    r = Retry("r", Scripted("c", [F]), max_attempts=3, recovery=Scripted("rec", [F], ErrorCode.UNRECOVERABLE))
    assert run_until_done(r, ctx) is F
    assert r.error.code is ErrorCode.RECOVERY_FAILED
    assert r.error.cause.code is ErrorCode.UNRECOVERABLE


@pytest.mark.parametrize("bad", [0, -1, 1.5, None])
def test_retry_rejects_unbounded_or_invalid_budget(bad):
    with pytest.raises(ValueError):
        Retry("r", Scripted("c", [S]), max_attempts=bad)


def test_timeout_halts_child(ctx):
    child = Scripted("c", [R])
    t = Timeout("t", child, seconds=1.0, code=ErrorCode.NAV_TIMEOUT)
    assert run_until_done(t, ctx) is F
    assert t.error.code is ErrorCode.NAV_TIMEOUT
    assert child.halted == 1
    assert ctx.now == pytest.approx(1.0)


def test_timeout_passes_through_fast_child(ctx):
    assert run_until_done(Timeout("t", Scripted("c", [R, S]), seconds=5.0), ctx) is S


def test_timeout_rejects_nonpositive():
    with pytest.raises(ValueError):
        Timeout("t", Scripted("c", [S]), seconds=0)


def test_fallback_only_handles_listed_errors(ctx):
    fb = Fallback("fb", Scripted("p", [F], ErrorCode.GRASP_PLAN_FAILED), Scripted("alt", [S]),
                  frozenset({ErrorCode.GRASP_PLAN_FAILED}))
    assert fb.tick(ctx) is S
    alt = Scripted("alt", [S])
    fb2 = Fallback("fb", Scripted("p", [F], ErrorCode.OUT_OF_REACH), alt, frozenset({ErrorCode.GRASP_PLAN_FAILED}))
    assert fb2.tick(ctx) is F
    assert fb2.error.code is ErrorCode.OUT_OF_REACH
    assert alt.ticks == 0


def test_condition(ctx):
    assert Condition("ok", lambda c: None).tick(ctx) is S
    c = Condition("bad", lambda c: SkillError(ErrorCode.NOT_HOLDING, "no"))
    assert c.tick(ctx) is F and c.error.code is ErrorCode.NOT_HOLDING


def test_failure_without_error_becomes_internal_error(ctx):
    class Silent(Node):
        def _tick(self, ctx):
            return Status.FAILURE

    n = Silent("silent")
    assert n.tick(ctx) is F
    assert n.error.code is ErrorCode.INTERNAL_ERROR


def test_blackboard_require_errors():
    from robot_skills.core import SkillInputError

    bb = Blackboard(x=1)
    assert bb.require("x", int) == 1
    with pytest.raises(SkillInputError):
        bb.require("missing", int)
    with pytest.raises(SkillInputError):
        bb.require("x", str)
