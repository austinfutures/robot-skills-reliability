from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.types import SceneSpec, default_scene

MAX_ATTEMPTS_LIMIT = 10  # hard ceiling: recovery budgets are always bounded


@dataclass(frozen=True)
class RecoveryPolicy:
    """Attempt budgets and timeouts per task stage."""

    nav_attempts: int = 3
    grasp_attempts: int = 3
    place_attempts: int = 2
    nav_timeout_s: float = 30.0
    approach_standoff_m: float = 0.45

    def __post_init__(self) -> None:
        for name in ("nav_attempts", "grasp_attempts", "place_attempts"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= MAX_ATTEMPTS_LIMIT:
                raise ValueError(f"{name} must be an int in [1, {MAX_ATTEMPTS_LIMIT}], got {v!r}")
        if not (math.isfinite(self.nav_timeout_s) and 0 < self.nav_timeout_s <= 600):
            raise ValueError(f"nav_timeout_s must be in (0, 600], got {self.nav_timeout_s!r}")
        if not 0.0 <= self.approach_standoff_m <= 2.0:
            raise ValueError(f"approach_standoff_m must be in [0, 2], got {self.approach_standoff_m!r}")


@dataclass(frozen=True)
class TaskSpec:
    """Everything needed to reproduce a pick-and-place task (with FaultConfig)."""

    task_id: str
    object_id: str = "red_cube"
    place_target: str = "bin_a"
    scene: SceneSpec = field(default_factory=default_scene)
    policy: RecoveryPolicy = field(default_factory=RecoveryPolicy)
    dt: float = 0.1
    deadline_s: float = 180.0

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id must be a non-empty string")
        if not isinstance(self.place_target, str) or not self.place_target.strip():
            raise ValueError("place_target must be a non-empty string")
        if not (math.isfinite(self.dt) and 0.0 < self.dt <= 1.0):
            raise ValueError(f"dt must be in (0, 1], got {self.dt!r}")
        if not (math.isfinite(self.deadline_s) and 0.0 < self.deadline_s <= 3600.0):
            raise ValueError(f"deadline_s must be in (0, 3600], got {self.deadline_s!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "object_id": self.object_id,
            "place_target": self.place_target,
            "scene": self.scene.to_dict(),
            "policy": asdict(self.policy),
            "dt": self.dt,
            "deadline_s": self.deadline_s,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TaskSpec":
        return TaskSpec(
            task_id=d["task_id"],
            object_id=d["object_id"],
            place_target=d["place_target"],
            scene=SceneSpec.from_dict(d["scene"]),
            policy=RecoveryPolicy(**d.get("policy", {})),
            dt=float(d.get("dt", 0.1)),
            deadline_s=float(d.get("deadline_s", 180.0)),
        )
