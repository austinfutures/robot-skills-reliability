"""Hardware-abstraction interfaces used by skills.

Skills only ever talk to these Protocols, never to the simulator. The shape
deliberately mirrors ROS 2 action semantics (send goal -> poll status ->
cancel) so a future adapter can wrap Nav2 / MoveIt / a perception node:

    MobileBase   ~ nav2_msgs/action/NavigateToPose + tf lookup
    Perception   ~ a detection topic / service
    Manipulator  ~ control_msgs gripper + MoveIt pick/place actions
    SemanticMap  ~ a map/landmark server

Time is never read from the wall clock inside skills; the executive passes
`now` (seconds) so the same code runs under simulated or real time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .types import Pose2D


class ActionState(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    SUCCEEDED = "SUCCEEDED"
    ABORTED = "ABORTED"
    CANCELED = "CANCELED"


@dataclass(frozen=True)
class Detection:
    object_id: str
    pose: Pose2D
    width_m: float
    confidence: float


@dataclass(frozen=True)
class PerceptionReading:
    """Result of one perception query. `healthy=False` means the sensor
    produced no valid frame (dropout) - distinct from 'nothing detected'."""

    healthy: bool
    detections: tuple[Detection, ...] = ()


@dataclass(frozen=True)
class GripperState:
    closed: bool
    holding_object_id: str | None


@runtime_checkable
class MobileBase(Protocol):
    def get_pose(self) -> Pose2D: ...
    def send_nav_goal(self, goal: Pose2D, tolerance_m: float) -> None: ...
    def nav_state(self) -> ActionState: ...
    def cancel_nav(self) -> None: ...
    def clear_costmap(self) -> None: ...


@runtime_checkable
class Perception(Protocol):
    def read(self) -> PerceptionReading: ...
    def reset_sensor(self) -> None: ...
    def max_range_m(self) -> float: ...


@runtime_checkable
class Manipulator(Protocol):
    def reach_m(self) -> float: ...
    def send_grasp(self, object_pose: Pose2D, gripper_width_m: float, approach_heading: float) -> None: ...
    def send_place(self, pose: Pose2D) -> None: ...
    def arm_state(self) -> ActionState: ...
    def gripper_state(self) -> GripperState: ...
    def open_gripper(self) -> None: ...


@runtime_checkable
class SemanticMap(Protocol):
    def lookup(self, name: str) -> Pose2D | None:
        """Prior-map location of a named place or object (may be stale)."""
        ...

    def known_names(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class RobotAPI:
    """Bundle of interfaces handed to every skill."""

    base: MobileBase
    perception: Perception
    manipulator: Manipulator
    semantic_map: SemanticMap
