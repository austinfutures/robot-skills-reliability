from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow `pytest` from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot_skills.executive import TaskSpec, run_task  # noqa: E402
from robot_skills.sim import FaultConfig  # noqa: E402


@pytest.fixture
def run():
    """run(faults=[...], **spec_kwargs) -> TaskResult"""

    def _run(faults: list[str] | None = None, log_path=None, seed: int = 0, **kw):
        kw.setdefault("task_id", "test-task")
        spec = TaskSpec(**kw)
        return run_task(spec, FaultConfig.from_names(faults or [], seed=seed), log_path=log_path)

    return _run


def events(result, name: str) -> list[dict]:
    return [e for e in result.events if e["event"] == name]
