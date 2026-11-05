"""JSONL logs, deterministic replay, benchmark metrics and CLI commands."""

from __future__ import annotations

import json

import pytest

from robot_skills.benchmark import SCENARIOS, run_benchmark
from robot_skills.cli import main
from robot_skills.events import deterministic_view, read_jsonl
from robot_skills.executive import RecoveryPolicy, TaskSpec
from robot_skills.replay import ReplayError, replay_log

REQUIRED = {"seq", "ts", "t_sim", "task_id", "event"}


def test_jsonl_log_schema(run, tmp_path):
    path = tmp_path / "log.jsonl"
    res = run(["grasp_failure"], log_path=path)
    recs = read_jsonl(path)
    assert len(recs) == len(res.events)
    assert recs[0]["event"] == "task_start" and recs[-1]["event"] == "task_end"
    for i, r in enumerate(recs):
        assert REQUIRED <= set(r)
        assert r["seq"] == i and r["task_id"] == "test-task"
    ends = [r for r in recs if r["event"] == "skill_end"]
    assert ends
    for r in ends:
        for k in ("skill", "duration_s", "wall_ms", "status", "retries", "error_code", "error_reason"):
            assert k in r
    failed = [r for r in ends if r["status"] == "FAILURE"]
    assert failed and all(r["error_code"] and r["error_reason"] for r in failed)
    assert any(r["retries"] == 1 for r in ends)


def test_runs_are_deterministic(run):
    a, b = run(["sensor_dropout", "grasp_failure"]), run(["sensor_dropout", "grasp_failure"])
    assert deterministic_view(a.events) == deterministic_view(b.events)


@pytest.mark.parametrize("faults", [[], ["grasp_failure"], ["repeated_failure"], ["navigation_timeout"],
                                    ["invalid_target"]])
def test_replay_matches(run, tmp_path, faults):
    path = tmp_path / "run.jsonl"
    original = run(faults, log_path=path)
    rep = replay_log(path, out_path=tmp_path / "replay.jsonl")
    assert rep.match, rep.first_divergence
    assert rep.replayed_status == original.status == rep.recorded_status


def test_replay_detects_divergence(run, tmp_path):
    path = tmp_path / "run.jsonl"
    run(["grasp_failure"], log_path=path)
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = next(i for i, ln in enumerate(lines) if '"event":"skill_end"' in ln)
    rec = json.loads(lines[idx])
    rec["status"] = "TAMPERED"
    lines[idx] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = replay_log(path)
    assert not rep.match
    assert rep.first_divergence["index"] == idx
    assert "status" in rep.first_divergence["fields"]


def test_replay_rejects_bad_logs(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay_log(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"event": "skill_end"}\n', encoding="utf-8")
    with pytest.raises(ReplayError):
        replay_log(bad)
    garbage = tmp_path / "garbage.jsonl"
    garbage.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        replay_log(garbage)


def test_benchmark_metrics():
    rep = run_benchmark(trials=2, seed=1)
    assert set(rep["scenarios"]) == {s.name for s in SCENARIOS}
    sc = rep["scenarios"]
    assert sc["nominal"]["task_success_rate"] == 1.0
    assert sc["grasp_failure"]["task_success_rate"] == 1.0
    assert sc["grasp_failure"]["recovery_success_rate"] == 1.0
    assert sc["invalid_target"]["task_success_rate"] == 0.0
    assert sc["invalid_target"]["failures_by_code"] == {"INVALID_TARGET": 2}
    assert sc["repeated_failure"]["failures_by_root_cause"] == {"GRASP_NOT_VERIFIED": 2}
    assert sc["missing_object"]["failures_by_root_cause"] == {"OBJECT_NOT_FOUND": 2}
    for s in rep["skills"].values():
        assert s["invocations"] > 0 and s["sim_p95_s"] >= 0 and s["wall_mean_ms"] >= 0
    assert rep["overall"]["trials"] == 2 * len(SCENARIOS)


def test_benchmark_reproducible_except_wall_time():
    def strip(r):
        return {k: v for k, v in r.items() if k != "skills"}

    assert strip(run_benchmark(trials=1, seed=7)) == strip(run_benchmark(trials=1, seed=7))


@pytest.mark.parametrize("kw", [{"nav_attempts": 0}, {"grasp_attempts": 11}, {"nav_timeout_s": 0.0},
                                {"place_attempts": 2.5}])
def test_policy_rejects_unbounded_values(kw):
    with pytest.raises(ValueError):
        RecoveryPolicy(**kw)


@pytest.mark.parametrize("kw", [{"task_id": ""}, {"task_id": "t", "dt": 0.0}, {"task_id": "t", "deadline_s": -1.0},
                                {"task_id": "t", "object_id": ""}])
def test_taskspec_rejects_invalid(kw):
    with pytest.raises(ValueError):
        TaskSpec(**kw)


# -------------------------------------------------------------------- CLI
def test_cli_run_and_replay(tmp_path, capsys):
    log = tmp_path / "cli.jsonl"
    assert main(["run", "--quiet", "--task-id", "cli-1", "--log", str(log)]) == 0
    assert "SUCCESS" in capsys.readouterr().out
    assert main(["replay", "--log", str(log)]) == 0
    assert "MATCH" in capsys.readouterr().out


def test_cli_run_with_faults(tmp_path, capsys):
    assert main(["run", "--fault", "grasp_failure", "--log", str(tmp_path / "a.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "FAULT grasp_failure" in out and "retry" in out
    assert main(["run", "--quiet", "--fault", "repeated_failure", "--log", str(tmp_path / "b.jsonl")]) == 1
    assert "RETRIES_EXHAUSTED" in capsys.readouterr().out


def test_cli_rejects_unknown_fault():
    with pytest.raises(SystemExit) as exc:
        main(["run", "--fault", "gremlins"])
    assert exc.value.code == 2


def test_cli_replay_missing_file(tmp_path, capsys):
    assert main(["replay", "--log", str(tmp_path / "nope.jsonl")]) == 2


def test_cli_benchmark_writes_report(tmp_path, capsys):
    out = tmp_path / "bench.json"
    assert main(["benchmark", "--trials", "1", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "overall" in data and "skills" in data
    assert "OVERALL" in capsys.readouterr().out


def test_cli_listing_commands(capsys):
    assert main(["faults"]) == 0 and "sensor_dropout" in capsys.readouterr().out
    assert main(["skills"]) == 0 and "recover_from_failure" in capsys.readouterr().out
