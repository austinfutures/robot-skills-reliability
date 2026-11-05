"""Deterministic fault injection.

Faults are described by an immutable `FaultConfig` (recorded in every log so
a run can be replayed) and applied by a stateful `FaultInjector` owned by the
simulator. Probabilistic faults draw from a seeded `random.Random`, so the
same config + seed always produces the same sequence of faults.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FaultType(str, Enum):
    GRASP_FAILURE = "grasp_failure"
    MISSING_OBJECT = "missing_object"
    NAVIGATION_TIMEOUT = "navigation_timeout"
    SENSOR_DROPOUT = "sensor_dropout"
    INVALID_TARGET = "invalid_target"
    REPEATED_FAILURE = "repeated_failure"


FAULT_DESCRIPTIONS: dict[FaultType, str] = {
    FaultType.GRASP_FAILURE: "first grasp attempt closes on nothing (object slips); recoverable",
    FaultType.MISSING_OBJECT: "task object is absent from the world although the map lists it",
    FaultType.NAVIGATION_TIMEOUT: "base is blocked on its first navigation goal until cancelled",
    FaultType.SENSOR_DROPOUT: "perception returns no valid frames for 0.6 s after first use",
    FaultType.INVALID_TARGET: "task requests a place target that does not exist",
    FaultType.REPEATED_FAILURE: "every grasp attempt slips; retries must be exhausted, not looped",
}


@dataclass(frozen=True)
class FaultConfig:
    grasp_failures: int = 0
    """Number of initial grasp attempts that deterministically slip."""
    grasp_failure_prob: float = 0.0
    """Additional per-attempt slip probability (seeded)."""
    missing_task_object: bool = False
    nav_stuck_goals: int = 0
    """Number of initial navigation goals on which the base makes no progress."""
    sensor_dropout_s: float = 0.0
    """Length of a dropout window that starts at the first perception read."""
    corrupt_place_target: bool = False
    seed: int = 0
    enabled: tuple[str, ...] = field(default=())
    """Names of the presets that produced this config (informational)."""

    def __post_init__(self) -> None:
        if self.grasp_failures < 0 or self.nav_stuck_goals < 0:
            raise ValueError("fault counts must be >= 0")
        if not 0.0 <= self.grasp_failure_prob <= 1.0:
            raise ValueError("grasp_failure_prob must be in [0, 1]")
        if self.sensor_dropout_s < 0.0:
            raise ValueError("sensor_dropout_s must be >= 0")

    @staticmethod
    def from_names(names: list[str] | tuple[str, ...], seed: int = 0) -> "FaultConfig":
        """Build a config from preset names, e.g. ["grasp_failure"]."""
        kw: dict[str, Any] = {"seed": seed, "enabled": tuple(names)}
        for raw in names:
            try:
                ft = FaultType(raw)
            except ValueError:
                valid = ", ".join(f.value for f in FaultType)
                raise ValueError(f"unknown fault '{raw}'. Valid faults: {valid}") from None
            if ft is FaultType.GRASP_FAILURE:
                kw["grasp_failures"] = max(kw.get("grasp_failures", 0), 1)
            elif ft is FaultType.REPEATED_FAILURE:
                kw["grasp_failures"] = 10_000
            elif ft is FaultType.MISSING_OBJECT:
                kw["missing_task_object"] = True
            elif ft is FaultType.NAVIGATION_TIMEOUT:
                kw["nav_stuck_goals"] = 1
            elif ft is FaultType.SENSOR_DROPOUT:
                kw["sensor_dropout_s"] = 0.6
            elif ft is FaultType.INVALID_TARGET:
                kw["corrupt_place_target"] = True
        return FaultConfig(**kw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["enabled"] = list(self.enabled)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FaultConfig":
        d = dict(d)
        d["enabled"] = tuple(d.get("enabled", ()))
        return FaultConfig(**d)


@dataclass(frozen=True)
class FaultEvent:
    fault: str
    t_sim: float
    detail: str


class FaultInjector:
    """Stateful view of a FaultConfig during one run."""

    def __init__(self, config: FaultConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._grasp_attempts = 0
        self._nav_goals = 0
        self._dropout_start: float | None = None
        self.events: list[FaultEvent] = []

    def _record(self, fault: FaultType, t: float, detail: str) -> None:
        self.events.append(FaultEvent(fault.value, round(t, 6), detail))

    def grasp_should_slip(self, t: float) -> bool:
        self._grasp_attempts += 1
        slip = self._grasp_attempts <= self.config.grasp_failures
        # Always draw so the RNG stream does not depend on which branch ran.
        draw = self._rng.random()
        if not slip and self.config.grasp_failure_prob > 0.0:
            slip = draw < self.config.grasp_failure_prob
        if slip:
            self._record(FaultType.GRASP_FAILURE, t, f"grasp attempt {self._grasp_attempts} slipped")
        return slip

    def nav_goal_should_stick(self, t: float) -> bool:
        self._nav_goals += 1
        stuck = self._nav_goals <= self.config.nav_stuck_goals
        if stuck:
            self._record(FaultType.NAVIGATION_TIMEOUT, t, f"base blocked on nav goal {self._nav_goals}")
        return stuck

    def sensor_dropped(self, t: float) -> bool:
        if self.config.sensor_dropout_s <= 0.0:
            return False
        if self._dropout_start is None:
            self._dropout_start = t
            self._record(FaultType.SENSOR_DROPOUT, t, f"dropout for {self.config.sensor_dropout_s:.2f}s")
        return t < self._dropout_start + self.config.sensor_dropout_s - 1e-9
