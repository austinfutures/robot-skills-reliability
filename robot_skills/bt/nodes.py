"""Minimal behavior-tree executive.

Semantics:
  * Every node returns SUCCESS, FAILURE or RUNNING from `tick(ctx)`.
  * A node that returns FAILURE always carries a `SkillError` in `.error`
    (enforced in `Node.tick`) - there are no silent failures.
  * Composite nodes have memory: a RUNNING child is resumed on the next tick
    rather than re-ticking earlier children (appropriate for long-running
    robot actions).
  * `halt(ctx)` stops RUNNING descendants (cancels robot actions).

Implemented here instead of using py_trees to keep zero extra dependencies
and to make error propagation / bounded retry explicit and testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from ..core.errors import NON_RETRYABLE, ErrorCode, SkillError, SkillInputError
from ..core.status import Status
from ..skills.base import Skill
from ..skills.contracts import RecoverInput

T = TypeVar("T")


class Blackboard(dict[str, Any]):
    """Shared key/value store between nodes with typed access."""

    def require(self, key: str, typ: type[T]) -> T:
        if key not in self:
            raise SkillInputError(ErrorCode.INVALID_INPUT, f"blackboard key '{key}' is missing")
        val = self[key]
        if not isinstance(val, typ):
            raise SkillInputError(
                ErrorCode.INVALID_INPUT,
                f"blackboard key '{key}' has type {type(val).__name__}, expected {typ.__name__}",
            )
        return val


class _NullLogger:
    def log(self, event: str, **fields: Any) -> None:
        pass


@dataclass
class TickContext:
    now: float
    blackboard: Blackboard = field(default_factory=Blackboard)
    logger: Any = field(default_factory=_NullLogger)


class Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status: Status | None = None
        self.error: SkillError | None = None
        self.parent: Node | None = None

    @property
    def children(self) -> list["Node"]:
        return []

    def _adopt(self, *nodes: "Node") -> None:
        for n in nodes:
            n.parent = self

    def tick(self, ctx: TickContext) -> Status:
        if self.status is not Status.RUNNING:
            self.error = None
            self.on_enter(ctx)
        status = self._tick(ctx)
        if status is Status.FAILURE and self.error is None:
            self.error = SkillError(ErrorCode.INTERNAL_ERROR, f"node '{self.name}' failed without a reason", self.name)
        if status is not Status.FAILURE:
            self.error = None if status is Status.SUCCESS else self.error
        self.status = status
        return status

    def on_enter(self, ctx: TickContext) -> None:  # noqa: B027
        pass

    def _tick(self, ctx: TickContext) -> Status:
        raise NotImplementedError

    def _fail(self, error: SkillError | None) -> Status:
        self.error = error
        return Status.FAILURE

    def halt(self, ctx: TickContext) -> None:
        for c in self.children:
            if c.status is Status.RUNNING:
                c.halt(ctx)
        self.reset()

    def reset(self) -> None:
        self.status = None
        for c in self.children:
            c.reset()

    def retry_attempt(self) -> int:
        """Attempt number of the nearest enclosing Retry (1 if none)."""
        n = self.parent
        while n is not None:
            if isinstance(n, Retry):
                return n.attempt
            n = n.parent
        return 1

    def iter_tree(self, depth: int = 0) -> list[tuple[int, "Node"]]:
        out = [(depth, self)]
        for c in self.children:
            out.extend(c.iter_tree(depth + 1))
        return out


# --------------------------------------------------------------- leaves
class Condition(Node):
    """Instant check. `check` returns None on success or a SkillError."""

    def __init__(self, name: str, check: Callable[[TickContext], SkillError | None]) -> None:
        super().__init__(name)
        self._check = check

    def _tick(self, ctx: TickContext) -> Status:
        err = self._check(ctx)
        return Status.SUCCESS if err is None else self._fail(err)


class SkillNode(Node, Generic[T]):
    """Leaf that runs a Skill. Builds the skill input from the blackboard,
    enforces the skill's contract and writes the output back."""

    def __init__(
        self,
        skill: Skill,  # type: ignore[type-arg]
        make_input: Callable[[Blackboard], Any],
        output_key: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name or skill.name)
        self.skill = skill
        self.make_input = make_input
        self.output_key = output_key
        self._t0 = 0.0
        self._wall = 0.0
        self._start_error: SkillError | None = None

    def on_enter(self, ctx: TickContext) -> None:
        self._t0 = ctx.now
        self._wall = 0.0
        self._start_error = None
        w0 = time.perf_counter()
        try:
            inp = self.make_input(ctx.blackboard)
            self.skill.start(inp, ctx.now)
        except SkillInputError as e:
            self._start_error = SkillError(e.code, str(e), self.skill.name)
        except Exception as e:  # defensive: never let a skill crash the executive
            self._start_error = SkillError(ErrorCode.INTERNAL_ERROR, f"{type(e).__name__}: {e}", self.skill.name)
        self._wall += time.perf_counter() - w0
        ctx.logger.log("skill_start", skill=self.skill.name, node=self.name, retries=self.retry_attempt() - 1)

    def _tick(self, ctx: TickContext) -> Status:
        if self._start_error is not None:
            return self._finish(ctx, Status.FAILURE, self._contract(self._start_error))
        w0 = time.perf_counter()
        try:
            res = self.skill.tick(ctx.now)
        except SkillInputError as e:
            res = self.skill.fail(e.code, str(e))
        except Exception as e:
            res = self.skill.fail(ErrorCode.INTERNAL_ERROR, f"{type(e).__name__}: {e}")
        self._wall += time.perf_counter() - w0
        if res.status is Status.RUNNING:
            return Status.RUNNING
        if res.status is Status.SUCCESS:
            if not isinstance(res.output, self.skill.output_type):
                err = SkillError(
                    ErrorCode.CONTRACT_VIOLATION,
                    f"{self.skill.name} returned {type(res.output).__name__}, "
                    f"contract says {self.skill.output_type.__name__}",
                    self.skill.name,
                )
                return self._finish(ctx, Status.FAILURE, err)
            if self.output_key:
                ctx.blackboard[self.output_key] = res.output
            return self._finish(ctx, Status.SUCCESS, None)
        return self._finish(ctx, Status.FAILURE, self._contract(res.error))

    def _contract(self, err: SkillError | None) -> SkillError:
        if err is None:
            return SkillError(ErrorCode.CONTRACT_VIOLATION, f"{self.skill.name} failed without an error", self.skill.name)
        if err.code not in self.skill.errors and err.code is not ErrorCode.INTERNAL_ERROR:
            return SkillError(
                ErrorCode.CONTRACT_VIOLATION,
                f"{self.skill.name} returned undeclared error {err.code.value}",
                self.skill.name,
                cause=err,
            )
        return err

    def _finish(self, ctx: TickContext, status: Status, err: SkillError | None) -> Status:
        self.error = err
        ctx.logger.log(
            "skill_end",
            skill=self.skill.name,
            node=self.name,
            status=status.value,
            duration_s=round(ctx.now - self._t0, 6),
            wall_ms=round(self._wall * 1000.0, 4),
            retries=self.retry_attempt() - 1,
            error_code=err.code.value if err else None,
            error_reason=err.message if err else None,
        )
        return status

    def halt(self, ctx: TickContext) -> None:
        if self.status is Status.RUNNING:
            self.skill.halt()
            ctx.logger.log(
                "skill_halted",
                skill=self.skill.name,
                node=self.name,
                status="HALTED",
                duration_s=round(ctx.now - self._t0, 6),
                wall_ms=round(self._wall * 1000.0, 4),
                retries=self.retry_attempt() - 1,
            )
        self.reset()


# ----------------------------------------------------------- composites
class Sequence(Node):
    """Run children in order; fail on the first failure."""

    def __init__(self, name: str, children: list[Node]) -> None:
        super().__init__(name)
        if not children:
            raise ValueError(f"Sequence '{name}' needs at least one child")
        self._children = children
        self._idx = 0
        self._adopt(*children)

    @property
    def children(self) -> list[Node]:
        return self._children

    def on_enter(self, ctx: TickContext) -> None:
        self._idx = 0

    def _tick(self, ctx: TickContext) -> Status:
        while self._idx < len(self._children):
            child = self._children[self._idx]
            s = child.tick(ctx)
            if s is Status.RUNNING:
                return s
            if s is Status.FAILURE:
                return self._fail(child.error)
            self._idx += 1
        return Status.SUCCESS


class Selector(Node):
    """Try children in order; succeed on the first success. If all fail,
    report the last child's error (the most specific attempt)."""

    def __init__(self, name: str, children: list[Node]) -> None:
        super().__init__(name)
        if not children:
            raise ValueError(f"Selector '{name}' needs at least one child")
        self._children = children
        self._idx = 0
        self._adopt(*children)

    @property
    def children(self) -> list[Node]:
        return self._children

    def on_enter(self, ctx: TickContext) -> None:
        self._idx = 0

    def _tick(self, ctx: TickContext) -> Status:
        last: SkillError | None = None
        while self._idx < len(self._children):
            child = self._children[self._idx]
            s = child.tick(ctx)
            if s is not Status.FAILURE:
                return s
            last = child.error
            self._idx += 1
        return self._fail(last)


class Fallback(Node):
    """Primary/alternative pair with error-aware gating.

    The alternative only runs when the primary fails with an error code in
    `handles`; any other failure propagates unchanged. This keeps fallbacks
    from masking unrelated faults (unlike a plain Selector).
    """

    def __init__(self, name: str, primary: Node, alternative: Node, handles: frozenset[ErrorCode]) -> None:
        super().__init__(name)
        self.primary, self.alternative, self.handles = primary, alternative, handles
        self._using_alt = False
        self._adopt(primary, alternative)

    @property
    def children(self) -> list[Node]:
        return [self.primary, self.alternative]

    def on_enter(self, ctx: TickContext) -> None:
        self._using_alt = False

    def _tick(self, ctx: TickContext) -> Status:
        if not self._using_alt:
            s = self.primary.tick(ctx)
            if s is not Status.FAILURE:
                return s
            err = self.primary.error
            if err is None or err.code not in self.handles:
                return self._fail(err)
            ctx.logger.log("fallback", node=self.name, error_code=err.code.value, error_reason=err.message)
            self._using_alt = True
        s = self.alternative.tick(ctx)
        if s is Status.FAILURE:
            return self._fail(self.alternative.error)
        return s


# ----------------------------------------------------------- decorators
class Timeout(Node):
    """Fail (and halt the child) if it is still RUNNING after `seconds`."""

    def __init__(self, name: str, child: Node, seconds: float, code: ErrorCode = ErrorCode.TIMEOUT) -> None:
        super().__init__(name)
        if seconds <= 0:
            raise ValueError(f"Timeout '{name}' must be positive, got {seconds}")
        self.child, self.seconds, self.code = child, seconds, code
        self._start = 0.0
        self._adopt(child)

    @property
    def children(self) -> list[Node]:
        return [self.child]

    def on_enter(self, ctx: TickContext) -> None:
        self._start = ctx.now

    def _tick(self, ctx: TickContext) -> Status:
        s = self.child.tick(ctx)
        if s is Status.FAILURE:
            return self._fail(self.child.error)
        if s is Status.RUNNING and ctx.now - self._start >= self.seconds - 1e-9:
            self.child.halt(ctx)
            ctx.logger.log("timeout", node=self.name, limit_s=self.seconds, error_code=self.code.value)
            return self._fail(
                SkillError(self.code, f"'{self.child.name}' exceeded {self.seconds:.1f}s", source=self.name)
            )
        return s


@dataclass(frozen=True)
class RecoveryRequest:
    error: SkillError
    stage: str
    attempt: int


RECOVERY_KEY = "recovery_request"


class Retry(Node):
    """Bounded retry with optional recovery between attempts.

    * Retries only error codes in `retry_on` (None = any retryable code);
      codes in NON_RETRYABLE always fail fast.
    * Between attempts, runs `recovery` (if given). A failed recovery ends
      the stage with RECOVERY_FAILED.
    * After `max_attempts` failures the stage fails with RETRIES_EXHAUSTED,
      keeping the last error as `cause`.
    """

    def __init__(
        self,
        name: str,
        child: Node,
        max_attempts: int,
        retry_on: frozenset[ErrorCode] | None = None,
        recovery: Node | None = None,
    ) -> None:
        super().__init__(name)
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError(f"Retry '{name}' max_attempts must be an int >= 1, got {max_attempts!r}")
        self.child, self.max_attempts, self.retry_on, self.recovery = child, max_attempts, retry_on, recovery
        self.attempt = 1
        self._recovering = False
        self._adopt(child)
        if recovery is not None:
            self._adopt(recovery)

    @property
    def children(self) -> list[Node]:
        return [self.child] + ([self.recovery] if self.recovery else [])

    def on_enter(self, ctx: TickContext) -> None:
        self.attempt = 1
        self._recovering = False

    def is_retryable(self, code: ErrorCode) -> bool:
        if code in NON_RETRYABLE:
            return False
        return self.retry_on is None or code in self.retry_on

    def _tick(self, ctx: TickContext) -> Status:
        if not self._recovering:
            s = self.child.tick(ctx)
            if s is not Status.FAILURE:
                return s
            err = self.child.error or SkillError(ErrorCode.INTERNAL_ERROR, "child failed without error", self.name)
            if not self.is_retryable(err.code):
                ctx.logger.log("stage_failed", node=self.name, attempt=self.attempt,
                               error_code=err.code.value, error_reason="non-retryable: " + err.message)
                return self._fail(err)
            if self.attempt >= self.max_attempts:
                ctx.logger.log("retries_exhausted", node=self.name, attempt=self.attempt,
                               max_attempts=self.max_attempts, error_code=err.code.value)
                return self._fail(
                    SkillError(
                        ErrorCode.RETRIES_EXHAUSTED,
                        f"stage '{self.name}' failed {self.max_attempts}/{self.max_attempts} attempts",
                        source=self.name,
                        cause=err,
                    )
                )
            ctx.logger.log("retry", node=self.name, attempt=self.attempt, max_attempts=self.max_attempts,
                           error_code=err.code.value, error_reason=err.message)
            if self.recovery is None:
                self._next_attempt()
                return Status.RUNNING
            ctx.blackboard[RECOVERY_KEY] = RecoveryRequest(err, self.name, self.attempt)
            self.recovery.reset()
            self._recovering = True
        s = self.recovery.tick(ctx)  # type: ignore[union-attr]
        if s is Status.RUNNING:
            return s
        if s is Status.FAILURE:
            rerr = self.recovery.error  # type: ignore[union-attr]
            return self._fail(
                SkillError(ErrorCode.RECOVERY_FAILED, f"recovery for stage '{self.name}' failed", self.name, cause=rerr)
            )
        self._recovering = False
        self._next_attempt()
        return Status.RUNNING  # retry on the next tick so time always advances

    def _next_attempt(self) -> None:
        self.attempt += 1
        self.child.reset()


def recovery_input(bb: Blackboard) -> RecoverInput:
    """Build recover_from_failure's input from the Retry node's request."""
    req = bb.require(RECOVERY_KEY, RecoveryRequest)
    return RecoverInput(req.error.code, req.stage, req.attempt)
