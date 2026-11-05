"""Deterministic replay: re-run a task from the inputs recorded in its log
and verify the new event stream matches the original exactly (excluding
wall-clock fields)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .events import SCHEMA_VERSION, deterministic_view, read_jsonl
from .executive.runner import TaskResult, run_task
from .executive.task import TaskSpec
from .sim.faults import FaultConfig


class ReplayError(ValueError):
    pass


@dataclass
class ReplayReport:
    log_path: str
    task_id: str
    recorded_events: int
    replayed_events: int
    recorded_status: str | None
    replayed_status: str
    match: bool
    first_divergence: dict[str, Any] | None
    result: TaskResult


def load_inputs(records: list[dict[str, Any]]) -> tuple[TaskSpec, FaultConfig]:
    if not records:
        raise ReplayError("log is empty")
    start = records[0]
    if start.get("event") != "task_start":
        raise ReplayError(f"first event must be 'task_start', got {start.get('event')!r}")
    if start.get("schema") != SCHEMA_VERSION:
        raise ReplayError(f"unsupported log schema {start.get('schema')!r} (expected {SCHEMA_VERSION})")
    try:
        return TaskSpec.from_dict(start["task_spec"]), FaultConfig.from_dict(start["fault_config"])
    except (KeyError, TypeError, ValueError) as e:
        raise ReplayError(f"task_start record is malformed: {e}") from None


def replay_log(path: str | Path, out_path: str | Path | None = None) -> ReplayReport:
    recorded = read_jsonl(path)
    spec, faults = load_inputs(recorded)
    result = run_task(spec, faults, log_path=out_path)
    a, b = deterministic_view(recorded), deterministic_view(result.events)
    divergence: dict[str, Any] | None = None
    for i in range(max(len(a), len(b))):
        ra = a[i] if i < len(a) else None
        rb = b[i] if i < len(b) else None
        if ra != rb:
            keys = sorted(set(ra or {}) | set(rb or {}))
            diff = {k: {"recorded": (ra or {}).get(k), "replayed": (rb or {}).get(k)}
                    for k in keys if (ra or {}).get(k) != (rb or {}).get(k)}
            divergence = {"index": i, "fields": diff}
            break
    end = next((r for r in reversed(recorded) if r.get("event") == "task_end"), None)
    return ReplayReport(
        log_path=str(path),
        task_id=spec.task_id,
        recorded_events=len(a),
        replayed_events=len(b),
        recorded_status=end.get("status") if end else None,
        replayed_status=result.status,
        match=divergence is None,
        first_divergence=divergence,
        result=result,
    )
