"""Structured JSONL event logging.

Each line is one JSON object with at least:
  seq, ts (wall clock, UTC ISO-8601), t_sim (simulated seconds), task_id, event
plus event-specific fields (skill, node, status, duration_s, wall_ms,
retries, error_code, error_reason, ...).

`ts` and `wall_ms` are the only non-deterministic fields; replay ignores them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable

SCHEMA_VERSION = 1
NON_DETERMINISTIC_FIELDS = frozenset({"ts", "wall_ms"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventLogger:
    def __init__(
        self,
        task_id: str,
        sim_time: Callable[[], float],
        path: str | Path | None = None,
        echo: Callable[[dict[str, Any]], None] | None = None,
        wall_time: Callable[[], str] = _utc_now,
    ) -> None:
        self.task_id = task_id
        self._sim_time = sim_time
        self._wall_time = wall_time
        self._echo = echo
        self.records: list[dict[str, Any]] = []
        self.path = Path(path) if path is not None else None
        self._fh: IO[str] | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8", newline="\n")

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "seq": len(self.records),
            "ts": self._wall_time(),
            "t_sim": round(self._sim_time(), 6),
            "task_id": self.task_id,
            "event": event,
        }
        rec.update(fields)
        self.records.append(rec)
        if self._fh is not None:
            self._fh.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
            self._fh.flush()  # keep the log useful even if the process dies
        if self._echo is not None:
            self._echo(rec)
        return rec

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"log file not found: {p}")
    records = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSON ({e.msg})") from None
    return records


def deterministic_view(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip wall-clock fields and normalise through JSON (tuples -> lists)
    so an in-memory run and a loaded log can be compared exactly."""
    out = []
    for r in records:
        norm = json.loads(json.dumps(r, default=str))
        out.append({k: v for k, v in norm.items() if k not in NON_DETERMINISTIC_FIELDS})
    return out
