"""Skill base class.

A skill is a small, reusable, stateful unit of robot behavior with:
  * a typed input dataclass (validated on construction),
  * a typed output dataclass,
  * a declared set of error codes it may return,
  * a start/tick/halt lifecycle compatible with behavior-tree execution.

Skills depend only on `RobotAPI` (interfaces), never on the simulator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

from ..core.errors import ErrorCode, SkillError, SkillInputError
from ..core.interfaces import RobotAPI
from ..core.status import Status

I = TypeVar("I")
O = TypeVar("O")


@dataclass(frozen=True)
class SkillResult(Generic[O]):
    status: Status
    output: O | None = None
    error: SkillError | None = None


class Skill(ABC, Generic[I, O]):
    name: ClassVar[str]
    input_type: ClassVar[type]
    output_type: ClassVar[type]
    errors: ClassVar[frozenset[ErrorCode]]

    def __init__(self, robot: RobotAPI) -> None:
        self.robot = robot
        self.inp: I | None = None
        self.started_at = 0.0

    # lifecycle -----------------------------------------------------------
    def start(self, inp: I, now: float) -> None:
        """Validate and accept a new request. Raises SkillInputError."""
        if not isinstance(inp, self.input_type):
            raise SkillInputError(
                ErrorCode.INVALID_INPUT,
                f"{self.name} expects {self.input_type.__name__}, got {type(inp).__name__}",
            )
        self.inp = inp
        self.started_at = now
        self.on_start(inp, now)

    def on_start(self, inp: I, now: float) -> None:  # noqa: B027 - optional hook
        """Send commands to the robot. May raise SkillInputError."""

    @abstractmethod
    def tick(self, now: float) -> SkillResult[O]:
        """Advance the skill. Must not block."""

    def halt(self) -> None:  # noqa: B027 - optional hook
        """Stop any in-flight robot action (called on timeout / preemption)."""

    # helpers --------------------------------------------------------------
    def ok(self, output: O) -> SkillResult[O]:
        return SkillResult(Status.SUCCESS, output=output)

    def fail(self, code: ErrorCode, message: str, **details: Any) -> SkillResult[O]:
        return SkillResult(Status.FAILURE, error=SkillError(code, message, source=self.name, details=details))

    @staticmethod
    def running() -> SkillResult[Any]:
        return SkillResult(Status.RUNNING)

    @classmethod
    def describe_contract(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "input": cls.input_type.__name__,
            "output": cls.output_type.__name__,
            "errors": sorted(e.value for e in cls.errors),
        }
