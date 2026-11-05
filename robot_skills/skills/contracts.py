"""Explicit input/output contracts for every skill.

Inputs validate themselves in `__post_init__` and raise `SkillInputError`
with a specific ErrorCode, so a malformed request fails immediately with a
meaningful reason instead of producing undefined robot behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.errors import ErrorCode, SkillInputError
from ..core.types import GraspStrategy, Pose2D

MAX_GRIPPER_OPENING_M = 0.14


def _check(cond: bool, msg: str, code: ErrorCode = ErrorCode.INVALID_INPUT) -> None:
    if not cond:
        raise SkillInputError(code, msg)


def _check_name(field: str, value: object) -> None:
    _check(isinstance(value, str) and value.strip() != "", f"{field} must be a non-empty string, got {value!r}")


def _check_number(field: str, value: object, lo: float, hi: float, lo_inclusive: bool = True) -> None:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if ok:
        ok = (value >= lo if lo_inclusive else value > lo) and value <= hi
    bracket = "[" if lo_inclusive else "("
    _check(ok, f"{field} must be a finite number in {bracket}{lo}, {hi}], got {value!r}")


def _check_pose(field: str, value: object) -> None:
    _check(isinstance(value, Pose2D), f"{field} must be a Pose2D, got {type(value).__name__}")


# --------------------------------------------------------------- navigate
@dataclass(frozen=True)
class NavigateInput:
    target_name: str
    standoff_m: float = 0.0
    """Stop this far short of the target, facing it (0 = drive onto it)."""
    tolerance_m: float = 0.05

    def __post_init__(self) -> None:
        _check_name("target_name", self.target_name)
        _check_number("standoff_m", self.standoff_m, 0.0, 2.0)
        _check_number("tolerance_m", self.tolerance_m, 0.0, 1.0, lo_inclusive=False)


@dataclass(frozen=True)
class NavigateOutput:
    goal: Pose2D
    final_pose: Pose2D
    position_error_m: float


# ----------------------------------------------------------------- detect
@dataclass(frozen=True)
class DetectInput:
    object_id: str
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        _check_name("object_id", self.object_id)
        _check_number("min_confidence", self.min_confidence, 0.0, 1.0)


@dataclass(frozen=True)
class DetectOutput:
    object_id: str
    pose: Pose2D
    width_m: float
    confidence: float
    distance_m: float


# ------------------------------------------------------------- plan grasp
@dataclass(frozen=True)
class PlanGraspInput:
    object_id: str
    object_pose: Pose2D
    object_width_m: float
    strategy: GraspStrategy
    supported_strategies: tuple[GraspStrategy, ...]

    def __post_init__(self) -> None:
        _check_name("object_id", self.object_id)
        _check_pose("object_pose", self.object_pose)
        _check_number("object_width_m", self.object_width_m, 0.0, 1.0, lo_inclusive=False)
        _check(isinstance(self.strategy, GraspStrategy), f"strategy must be a GraspStrategy, got {self.strategy!r}")
        _check(len(self.supported_strategies) > 0, "supported_strategies must not be empty")


@dataclass(frozen=True)
class GraspPlan:
    object_id: str
    grasp_pose: Pose2D
    gripper_width_m: float
    approach_heading: float
    strategy: GraspStrategy

    def __post_init__(self) -> None:
        _check_name("object_id", self.object_id)
        _check_pose("grasp_pose", self.grasp_pose)
        _check_number("gripper_width_m", self.gripper_width_m, 0.0, MAX_GRIPPER_OPENING_M, lo_inclusive=False)
        _check_number("approach_heading", self.approach_heading, -math.pi - 1e-9, math.pi + 1e-9)


# ---------------------------------------------------------- execute grasp
@dataclass(frozen=True)
class ExecuteGraspOutput:
    object_id: str
    gripper_closed: bool


# ----------------------------------------------------------- verify grasp
@dataclass(frozen=True)
class VerifyGraspInput:
    object_id: str

    def __post_init__(self) -> None:
        _check_name("object_id", self.object_id)


@dataclass(frozen=True)
class VerifyGraspOutput:
    object_id: str
    holding: bool


# ------------------------------------------------------------------ place
@dataclass(frozen=True)
class PlaceInput:
    object_id: str
    target_name: str

    def __post_init__(self) -> None:
        _check_name("object_id", self.object_id)
        _check_name("target_name", self.target_name)


@dataclass(frozen=True)
class PlaceOutput:
    object_id: str
    target_name: str
    placed_pose: Pose2D


# --------------------------------------------------------------- recovery
@dataclass(frozen=True)
class RecoverInput:
    error_code: ErrorCode
    failed_stage: str
    attempt: int

    def __post_init__(self) -> None:
        _check(isinstance(self.error_code, ErrorCode), f"error_code must be an ErrorCode, got {self.error_code!r}")
        _check_name("failed_stage", self.failed_stage)
        _check(
            isinstance(self.attempt, int) and not isinstance(self.attempt, bool) and self.attempt >= 1,
            f"attempt must be an int >= 1, got {self.attempt!r}",
        )


@dataclass(frozen=True)
class RecoverOutput:
    action: str
    duration_s: float
