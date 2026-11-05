"""Error taxonomy. Every failure in the system carries one of these codes.

The executive relies on the code (not the message) to decide whether a
failure is retryable, so codes are part of each skill's contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    # Input / contract problems: never retried.
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_TARGET = "INVALID_TARGET"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
    # Navigation
    NAV_ABORTED = "NAV_ABORTED"
    NAV_TIMEOUT = "NAV_TIMEOUT"
    # Perception
    SENSOR_DROPOUT = "SENSOR_DROPOUT"
    OBJECT_NOT_FOUND = "OBJECT_NOT_FOUND"
    # Manipulation
    GRASP_PLAN_FAILED = "GRASP_PLAN_FAILED"
    OUT_OF_REACH = "OUT_OF_REACH"
    GRASP_EXECUTION_FAILED = "GRASP_EXECUTION_FAILED"
    GRASP_NOT_VERIFIED = "GRASP_NOT_VERIFIED"
    NOT_HOLDING = "NOT_HOLDING"
    PLACE_FAILED = "PLACE_FAILED"
    # Executive-level
    TIMEOUT = "TIMEOUT"
    RETRIES_EXHAUSTED = "RETRIES_EXHAUSTED"
    RECOVERY_FAILED = "RECOVERY_FAILED"
    UNRECOVERABLE = "UNRECOVERABLE"
    TASK_DEADLINE_EXCEEDED = "TASK_DEADLINE_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Codes that indicate the request itself is wrong; retrying cannot help.
NON_RETRYABLE: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.INVALID_INPUT,
        ErrorCode.INVALID_TARGET,
        ErrorCode.CONTRACT_VIOLATION,
        ErrorCode.INTERNAL_ERROR,
        ErrorCode.UNRECOVERABLE,
    }
)


@dataclass(frozen=True)
class SkillError:
    """A structured failure reason. `cause` keeps the root cause when an
    executive node wraps a lower-level error (e.g. RETRIES_EXHAUSTED)."""

    code: ErrorCode
    message: str
    source: str = ""
    cause: "SkillError | None" = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> "SkillError":
        err = self
        while err.cause is not None:
            err = err.cause
        return err

    def describe(self) -> str:
        text = f"{self.code.value}: {self.message}"
        if self.cause is not None:
            text += f" (caused by {self.cause.describe()})"
        return text

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code.value, "message": self.message, "source": self.source}
        if self.cause is not None:
            d["cause"] = self.cause.to_dict()
        return d


class SkillInputError(ValueError):
    """Raised when a skill input fails contract validation."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
