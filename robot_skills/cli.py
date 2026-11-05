"""Command-line interface: python -m robot_skills <command>."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .benchmark import format_report, run_benchmark, save_report
from .core.types import default_scene
from .executive.runner import TaskResult, run_task
from .executive.task import TaskSpec
from .executive.trees import build_pick_and_place_tree, render_tree
from .replay import ReplayError, replay_log
from .sim.faults import FAULT_DESCRIPTIONS, FaultConfig, FaultType
from .sim.simulator import Simulator2D
from .skills import SKILL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
_PRINTED = {"skill_end", "skill_halted", "retry", "retries_exhausted", "stage_failed", "timeout", "fallback",
            "fault_injected"}


def _echo(rec: dict[str, Any]) -> None:
    ev = rec["event"]
    if ev not in _PRINTED:
        return
    t = f"[t={rec['t_sim']:7.2f}s]"
    if ev in ("skill_end", "skill_halted"):
        line = f"{t} {rec['node']:<22} {rec['status']:<8} {rec['duration_s']:6.2f}s  retries={rec['retries']}"
        if rec.get("error_code"):
            line += f"  {rec['error_code']}: {rec.get('error_reason')}"
    elif ev == "fault_injected":
        line = f"{t} ** FAULT {rec['fault']}: {rec['detail']}"
    elif ev == "retry":
        line = f"{t} -> retry {rec['node']} (attempt {rec['attempt']}/{rec['max_attempts']} failed: {rec['error_code']})"
    elif ev == "timeout":
        line = f"{t} -> timeout {rec['node']} after {rec['limit_s']:.1f}s"
    elif ev == "fallback":
        line = f"{t} -> fallback {rec['node']} ({rec['error_code']})"
    else:
        line = f"{t} -> {ev} {rec['node']} ({rec.get('error_code')})"
    print(line, flush=True)


def _print_result(res: TaskResult) -> None:
    print("-" * 72)
    print(f"task {res.task_id}: {res.status}")
    print(f"  sim duration      : {res.sim_duration_s:.2f} s ({res.ticks} ticks), wall {res.wall_s * 1000:.1f} ms")
    print(f"  retries           : {res.retries}")
    print(f"  recoveries        : {res.recoveries_succeeded}/{res.recoveries_attempted} succeeded")
    print(f"  object at target  : {res.object_at_target}")
    if res.faults_injected:
        print(f"  faults injected   : {', '.join(res.faults_injected)}")
    if res.error_reason:
        print(f"  error             : {res.error_reason}")
    if res.log_path:
        print(f"  event log         : {res.log_path}")


def cmd_run(args: argparse.Namespace) -> int:
    task_id = args.task_id or f"task-{uuid.uuid4().hex[:8]}"
    try:
        faults = FaultConfig.from_names(args.fault or [], seed=args.seed)
        spec = TaskSpec(task_id=task_id, object_id=args.object, place_target=args.target)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.show_tree:
        sim = Simulator2D(spec.scene)
        print(render_tree(build_pick_and_place_tree(sim.api(), spec)))
        print()
    log_path = Path(args.log) if args.log else Path(args.log_dir) / f"{task_id}.jsonl"
    print(f"Running pick-and-place '{spec.object_id}' -> '{spec.place_target}' in 2D simulation "
          f"(faults: {', '.join(args.fault) if args.fault else 'none'})")
    res = run_task(spec, faults, log_path=log_path, echo=None if args.quiet else _echo)
    if args.json:
        print(json.dumps(res.summary(), indent=2))
    else:
        _print_result(res)
    return 0 if res.succeeded else 1


def cmd_test(args: argparse.Namespace) -> int:
    tests = REPO_ROOT / "tests"
    if not tests.is_dir():
        print(f"error: tests directory not found at {tests}", file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "pytest", str(tests), *args.pytest_args]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def cmd_replay(args: argparse.Namespace) -> int:
    try:
        rep = replay_log(args.log, out_path=args.out)
    except (FileNotFoundError, ReplayError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"replayed {rep.log_path} (task {rep.task_id})")
    print(f"  recorded: {rep.recorded_events} events, status {rep.recorded_status}")
    print(f"  replayed: {rep.replayed_events} events, status {rep.replayed_status}")
    if rep.match:
        print("  result  : MATCH - event stream identical (ignoring wall-clock fields)")
        return 0
    print("  result  : DIVERGED")
    print(json.dumps(rep.first_divergence, indent=2, default=str))
    return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    report = run_benchmark(trials=args.trials, seed=args.seed, log_dir=args.log_dir)
    print(format_report(report))
    out = save_report(report, args.out)
    print(f"\nfull report: {out}")
    return 0


def cmd_faults(args: argparse.Namespace) -> int:
    for ft in FaultType:
        print(f"{ft.value:<20} {FAULT_DESCRIPTIONS[ft]}")
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    for name, cls in SKILL_REGISTRY.items():
        c = cls.describe_contract()
        print(f"{name:<22} {c['input']} -> {c['output']}")
        print(f"{'':<22} errors: {', '.join(c['errors'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m robot_skills", description="Robot skills reliability toolkit (2D sim)")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one pick-and-place task")
    r.add_argument("--fault", action="append", choices=[f.value for f in FaultType],
                   help="inject a fault (repeatable)")
    names = sorted(o.object_id for o in default_scene().objects)
    r.add_argument("--object", default="red_cube", help=f"object to pick ({', '.join(names)})")
    r.add_argument("--target", default="bin_a", help="place target (bin_a, shelf_b)")
    r.add_argument("--seed", type=int, default=0, help="fault RNG seed")
    r.add_argument("--task-id", default=None)
    r.add_argument("--log", default=None, help="event log path (default logs/<task_id>.jsonl)")
    r.add_argument("--log-dir", default="logs")
    r.add_argument("--quiet", action="store_true", help="do not stream events")
    r.add_argument("--json", action="store_true", help="print the summary as JSON")
    r.add_argument("--show-tree", action="store_true", help="print the behavior tree first")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("test", help="run the pytest suite (extra args go to pytest)")
    t.add_argument("pytest_args", nargs=argparse.REMAINDER)
    t.set_defaults(func=cmd_test)

    rp = sub.add_parser("replay", help="re-run a task from its log and check determinism")
    rp.add_argument("--log", required=True, help="JSONL log written by `run`")
    rp.add_argument("--out", default=None, help="optional path for the replayed log")
    rp.set_defaults(func=cmd_replay)

    b = sub.add_parser("benchmark", help="success/recovery statistics across fault scenarios")
    b.add_argument("--trials", type=int, default=10, help="trials per scenario")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="results/benchmark.json")
    b.add_argument("--log-dir", default=None, help="also write per-task logs here")
    b.set_defaults(func=cmd_benchmark)

    sub.add_parser("faults", help="list injectable faults").set_defaults(func=cmd_faults)
    sub.add_parser("skills", help="list skill contracts").set_defaults(func=cmd_skills)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
