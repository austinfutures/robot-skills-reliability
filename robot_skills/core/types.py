"""Plain value types shared by the simulator, skills and executive.

All types are immutable dataclasses so they can be logged, compared and
replayed without hidden state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


def wrap_angle(theta: float) -> float:
    """Wrap an angle to the interval [-pi, pi)."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _require_finite(name: str, *values: float) -> None:
    for v in values:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name} must be finite numbers, got {values!r}")


@dataclass(frozen=True)
class Pose2D:
    """Planar pose: position in metres, heading in radians (world frame)."""

    x: float
    y: float
    theta: float = 0.0

    def __post_init__(self) -> None:
        _require_finite("Pose2D", self.x, self.y, self.theta)

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def bearing_to(self, other: "Pose2D") -> float:
        """Heading (world frame) that points from self towards other."""
        return math.atan2(other.y - self.y, other.x - self.x)

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 6), "y": round(self.y, 6), "theta": round(self.theta, 6)}

    @staticmethod
    def from_dict(d: dict[str, float]) -> "Pose2D":
        return Pose2D(float(d["x"]), float(d["y"]), float(d.get("theta", 0.0)))


class GraspStrategy(str, Enum):
    TOP = "top"
    SIDE = "side"


@dataclass(frozen=True)
class ObjectSpec:
    """Static description of a manipulable object in a scene."""

    object_id: str
    pose: Pose2D
    width_m: float = 0.05
    grasp_strategies: tuple[GraspStrategy, ...] = (GraspStrategy.TOP, GraspStrategy.SIDE)

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "pose": self.pose.to_dict(),
            "width_m": self.width_m,
            "grasp_strategies": [s.value for s in self.grasp_strategies],
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> "ObjectSpec":
        return ObjectSpec(
            object_id=str(d["object_id"]),
            pose=Pose2D.from_dict(d["pose"]),  # type: ignore[arg-type]
            width_m=float(d.get("width_m", 0.05)),  # type: ignore[arg-type]
            grasp_strategies=tuple(GraspStrategy(s) for s in d.get("grasp_strategies", ["top", "side"])),  # type: ignore[union-attr]
        )


@dataclass(frozen=True)
class SceneSpec:
    """Complete, serialisable description of a simulated scene."""

    robot_start: Pose2D
    objects: tuple[ObjectSpec, ...]
    targets: dict[str, Pose2D] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "robot_start": self.robot_start.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
            "targets": {k: v.to_dict() for k, v in sorted(self.targets.items())},
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> "SceneSpec":
        return SceneSpec(
            robot_start=Pose2D.from_dict(d["robot_start"]),  # type: ignore[arg-type]
            objects=tuple(ObjectSpec.from_dict(o) for o in d["objects"]),  # type: ignore[union-attr]
            targets={k: Pose2D.from_dict(v) for k, v in d["targets"].items()},  # type: ignore[union-attr]
        )


def default_scene() -> SceneSpec:
    """The reference scene used by `run`, tests and the benchmark."""
    return SceneSpec(
        robot_start=Pose2D(0.0, 0.0, 0.0),
        objects=(
            ObjectSpec("red_cube", Pose2D(2.5, 1.0), 0.05, (GraspStrategy.TOP, GraspStrategy.SIDE)),
            ObjectSpec("tall_bottle", Pose2D(3.0, -1.5), 0.07, (GraspStrategy.SIDE,)),
            ObjectSpec("blue_box", Pose2D(-1.5, -2.0), 0.12, (GraspStrategy.TOP,)),
        ),
        targets={
            "bin_a": Pose2D(-1.0, 2.5),
            "shelf_b": Pose2D(4.0, 2.5),
        },
    )
