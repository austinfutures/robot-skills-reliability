"""Task runner: ticks the tree against the simulator with a hard deadline."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..bt.nodes import Blackboard, TickContext
from ..core.errors import ErrorCode, SkillError
from ..core.status import Status
from ..events import SCHEMA_VERSION, EventLogger
from ..sim.faults import FaultConfig
from ..sim.simulator import Simulator2D
from .task import TaskSpec
from .trees import build_pick_and_place_tree

INVALID_TARGET_NAME = "nonexistent_zone"


@dataclass
class SkillStat:
    skill: str
    status: str
    duration_s: float
    wall_ms: float
    retries: int


@dataclass
class TaskResult:
    task_id: str
    status: str
    error_code: str | None
    root_error_code: str | None
    error_reason: str | None
    sim_duration_s: float
    ticks: int
    retries: int
    recoveries_attempted: int
    recoveries_succeeded: int
    object_at_target: bool
    faults_injected: list[str]
    skill_stats: list[SkillStat] = field(default_factory=list)
    wall_s: float = 0.0
    log_path: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def succeeded(self) -> bool:
        return self.status == Status.SUCCESS.value

    def summary(self) -> dict[str, Any]:
        per_skill: dict[str, int] = defaultdict(int)
        for s in self.skill_stats:
            per_skill[s.skill] += 1
        return {
            "status": self.status,
            "error_code": self.error_code,
            "root_error_code": self.root_error_code,
            "error_reason": self.error_reason,
            "sim_duration_s": self.sim_duration_s,
            "ticks": self.ticks,
            "retries": self.retries,
            "recoveries_attempted": self.recoveries_attempted,
            "recoveries_succeeded": self.recoveries_succeeded,
            "object_at_target": self.object_at_target,
            "faults_injected": self.faults_injected,
            "skill_invocations": dict(sorted(per_skill.items())),
        }


def run_task(
    spec: TaskSpec,
    faults: FaultConfig | None = None,
    log_path: str | Path | None = None,
    echo: Callable[[dict[str, Any]], None] | None = None,
) -> TaskResult:
    faults = faults or FaultConfig()
    wall0 = time.perf_counter()
    sim = Simulator2D(spec.scene, faults, task_object_id=spec.object_id)
    robot = sim.api()
    logger = EventLogger(spec.task_id, lambda: sim.now, log_path, echo)
    try:
        logger.log(
            "task_start",
            schema=SCHEMA_VERSION,
            task_spec=spec.to_dict(),
            fault_config=faults.to_dict(),
        )
        place_target = spec.place_target
        if faults.corrupt_place_target:
            place_target = INVALID_TARGET_NAME
            logger.log("fault_injected", fault="invalid_target", detail=f"place target replaced by '{place_target}'")
        for ev in sim.drain_fault_events():
            logger.log("fault_injected", fault=ev.fault, detail=ev.detail)

        tree = build_pick_and_place_tree(robot, spec, place_target)
        ctx = TickContext(now=sim.now, blackboard=Blackboard(), logger=logger)
        ticks = 0
        error: SkillError | None = None
        while True:
            status = tree.tick(ctx)
            ticks += 1
            for ev in sim.drain_fault_events():
                logger.log("fault_injected", fault=ev.fault, detail=ev.detail)
            if status is not Status.RUNNING:
                error = tree.error if status is Status.FAILURE else None
                break
            if sim.now + 1e-9 >= spec.deadline_s:
                tree.halt(ctx)
                status = Status.FAILURE
                error = SkillError(
                    ErrorCode.TASK_DEADLINE_EXCEEDED, f"task exceeded deadline of {spec.deadline_s:.1f}s", "runner"
                )
                break
            sim.step(spec.dt)
            ctx.now = sim.now
            # Faults raised inside the physics step (e.g. a grasp slip) are
            # logged immediately so they appear before the skill that sees them.
            for ev in sim.drain_fault_events():
                logger.log("fault_injected", fault=ev.fault, detail=ev.detail)

        if status is Status.FAILURE and error is None:  # belt and braces: never fail silently
            error = SkillError(ErrorCode.INTERNAL_ERROR, "task failed without an error reason", "runner")

        tgt = robot.semantic_map.lookup(spec.place_target)
        obj = sim.objects.get(spec.object_id)
        at_target = bool(tgt and obj and not obj.held and obj.pose.distance_to(tgt) < 0.05)

        result = _collect(spec, status, error, sim.now, ticks, at_target, logger.records)
        result.wall_s = round(time.perf_counter() - wall0, 6)
        result.log_path = str(log_path) if log_path else None
        logger.log("task_end", wall_ms=round(result.wall_s * 1000, 3), **result.summary())
        result.events = logger.records
        return result
    finally:
        logger.close()


def _collect(
    spec: TaskSpec,
    status: Status,
    error: SkillError | None,
    sim_time: float,
    ticks: int,
    at_target: bool,
    records: list[dict[str, Any]],
) -> TaskResult:
    stats: list[SkillStat] = []
    retries = rec_att = rec_ok = 0
    faults: list[str] = []
    for r in records:
        ev = r["event"]
        if ev in ("skill_end", "skill_halted"):
            stats.append(SkillStat(r["skill"], r["status"], r["duration_s"], r["wall_ms"], r["retries"]))
            if r["skill"] == "recover_from_failure" and ev == "skill_end":
                rec_att += 1
                rec_ok += r["status"] == Status.SUCCESS.value
        elif ev == "retry":
            retries += 1
        elif ev == "fault_injected":
            faults.append(r["fault"])
    return TaskResult(
        task_id=spec.task_id,
        status=status.value,
        error_code=error.code.value if error else None,
        root_error_code=error.root.code.value if error else None,
        error_reason=error.describe() if error else None,
        sim_duration_s=round(sim_time, 6),
        ticks=ticks,
        retries=retries,
        recoveries_attempted=rec_att,
        recoveries_succeeded=rec_ok,
        object_at_target=at_target,
        faults_injected=sorted(set(faults)),
        skill_stats=stats,
    )
