"""Reliability benchmark over nominal and faulted scenarios.

Scenario variation (object choice, object position jitter, place target) is
drawn from a seeded NumPy generator, so the whole benchmark is reproducible
from `--seed`. Reported times: `sim` = simulated seconds (deterministic),
`wall` = host CPU time spent in skill code (machine dependent).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .core.types import Pose2D, default_scene
from .executive.runner import TaskResult, run_task
from .executive.task import TaskSpec
from .sim.faults import FaultConfig


@dataclass(frozen=True)
class Scenario:
    name: str
    faults: tuple[str, ...] = ()
    grasp_failure_prob: float = 0.0


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("nominal"),
    Scenario("grasp_failure", ("grasp_failure",)),
    Scenario("missing_object", ("missing_object",)),
    Scenario("navigation_timeout", ("navigation_timeout",)),
    Scenario("sensor_dropout", ("sensor_dropout",)),
    Scenario("invalid_target", ("invalid_target",)),
    Scenario("repeated_failure", ("repeated_failure",)),
    Scenario("stochastic_grasp_35pct", (), 0.35),
    Scenario("combined", ("grasp_failure", "sensor_dropout", "navigation_timeout")),
)

GRASPABLE = ("red_cube", "tall_bottle")
TARGETS = ("bin_a", "shelf_b")


@dataclass
class ScenarioStats:
    name: str
    results: list[TaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        n = len(self.results)
        ok = sum(r.succeeded for r in self.results)
        with_rec = [r for r in self.results if r.recoveries_attempted > 0]
        retries = np.array([r.retries for r in self.results], dtype=float)
        dur = np.array([r.sim_duration_s for r in self.results], dtype=float)
        return {
            "trials": n,
            "task_success_rate": round(ok / n, 4) if n else 0.0,
            "recovery_success_rate": round(sum(r.succeeded for r in with_rec) / len(with_rec), 4) if with_rec else None,
            "tasks_needing_recovery": len(with_rec),
            "retries_mean": round(float(retries.mean()), 3) if n else 0.0,
            "retries_max": int(retries.max()) if n else 0,
            "sim_duration_mean_s": round(float(dur.mean()), 3) if n else 0.0,
            "failures_by_code": dict(sorted(Counter(r.error_code for r in self.results if not r.succeeded).items())),
            "failures_by_root_cause": dict(
                sorted(Counter(r.root_error_code for r in self.results if not r.succeeded).items())
            ),
        }


def _make_spec(rng: np.random.Generator, task_id: str) -> TaskSpec:
    base = default_scene()
    obj_id = GRASPABLE[int(rng.integers(len(GRASPABLE)))]
    target = TARGETS[int(rng.integers(len(TARGETS)))]
    dx, dy = (float(v) for v in rng.uniform(-0.4, 0.4, size=2))
    objs = tuple(
        replace(o, pose=Pose2D(round(o.pose.x + dx, 4), round(o.pose.y + dy, 4))) if o.object_id == obj_id else o
        for o in base.objects
    )
    scene = replace(base, objects=objs)
    return TaskSpec(task_id=task_id, object_id=obj_id, place_target=target, scene=scene)


def run_benchmark(trials: int = 10, seed: int = 0, log_dir: str | Path | None = None) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    rng = np.random.default_rng(seed)
    per_scenario: dict[str, ScenarioStats] = {}
    skill_sim: dict[str, list[float]] = defaultdict(list)
    skill_wall: dict[str, list[float]] = defaultdict(list)
    skill_fail: Counter[str] = Counter()
    rec_actions = rec_ok = 0
    for sc in SCENARIOS:
        stats = ScenarioStats(sc.name)
        for i in range(trials):
            task_id = f"bench-{sc.name}-{i:03d}"
            spec = _make_spec(rng, task_id)
            fseed = int(rng.integers(2**31 - 1))
            fc = FaultConfig.from_names(list(sc.faults), seed=fseed)
            if sc.grasp_failure_prob:
                fc = replace(fc, grasp_failure_prob=sc.grasp_failure_prob)
            log_path = Path(log_dir) / f"{task_id}.jsonl" if log_dir else None
            res = run_task(spec, fc, log_path=log_path)
            stats.results.append(res)
            rec_actions += res.recoveries_attempted
            rec_ok += res.recoveries_succeeded
            for s in res.skill_stats:
                skill_sim[s.skill].append(s.duration_s)
                skill_wall[s.skill].append(s.wall_ms)
                if s.status != "SUCCESS":
                    skill_fail[s.skill] += 1
        per_scenario[sc.name] = stats

    all_results = [r for s in per_scenario.values() for r in s.results]
    overall = ScenarioStats("overall", all_results).to_dict()
    overall["recovery_action_success_rate"] = round(rec_ok / rec_actions, 4) if rec_actions else None
    skills = {}
    for name in sorted(skill_sim):
        sim_a, wall_a = np.array(skill_sim[name]), np.array(skill_wall[name])
        skills[name] = {
            "invocations": int(sim_a.size),
            "failures": int(skill_fail[name]),
            "sim_mean_s": round(float(sim_a.mean()), 4),
            "sim_p95_s": round(float(np.percentile(sim_a, 95)), 4),
            "wall_mean_ms": round(float(wall_a.mean()), 4),
            "wall_p95_ms": round(float(np.percentile(wall_a, 95)), 4),
        }
    return {
        "seed": seed,
        "trials_per_scenario": trials,
        "overall": overall,
        "scenarios": {k: v.to_dict() for k, v in per_scenario.items()},
        "skills": skills,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Benchmark: {report['trials_per_scenario']} trials/scenario, seed={report['seed']} (simulation only)",
        "",
        f"{'scenario':<24}{'success':>9}{'recovery':>10}{'retries':>9}{'sim_s':>8}  failures (root cause)",
        "-" * 96,
    ]

    def pct(v: float | None) -> str:
        return "   n/a" if v is None else f"{v * 100:5.1f}%"

    rows = list(report["scenarios"].items()) + [("OVERALL", report["overall"])]
    for name, s in rows:
        fails = ", ".join(f"{k}={v}" for k, v in s["failures_by_root_cause"].items()) or "-"
        lines.append(
            f"{name:<24}{pct(s['task_success_rate']):>9}{pct(s['recovery_success_rate']):>10}"
            f"{s['retries_mean']:>9.2f}{s['sim_duration_mean_s']:>8.1f}  {fails}"
        )
    lines += [
        "",
        f"recovery action success rate: {pct(report['overall']['recovery_action_success_rate']).strip()}",
        "",
        f"{'skill':<22}{'calls':>7}{'fails':>7}{'sim_mean_s':>12}{'sim_p95_s':>11}{'wall_mean_ms':>14}{'wall_p95_ms':>13}",
        "-" * 86,
    ]
    for name, s in report["skills"].items():
        lines.append(
            f"{name:<22}{s['invocations']:>7}{s['failures']:>7}{s['sim_mean_s']:>12.3f}{s['sim_p95_s']:>11.3f}"
            f"{s['wall_mean_ms']:>14.4f}{s['wall_p95_ms']:>13.4f}"
        )
    return "\n".join(lines)


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


__all__ = ["SCENARIOS", "Scenario", "format_report", "run_benchmark", "save_report"]
