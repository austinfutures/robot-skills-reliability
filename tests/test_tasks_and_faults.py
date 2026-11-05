"""End-to-end task execution: nominal run, every fault, recovery success and
exhaustion, timeouts, deadlines."""

from __future__ import annotations

from conftest import events

from robot_skills.executive import RecoveryPolicy


def test_nominal_pick_and_place_succeeds(run):
    res = run()
    assert res.status == "SUCCESS"
    assert res.error_code is None
    assert res.retries == 0
    assert res.recoveries_attempted == 0
    assert res.object_at_target
    skills = [s.skill for s in res.skill_stats]
    for name in ("navigate_to_target", "detect_object", "plan_grasp", "execute_grasp", "verify_grasp", "place_object"):
        assert name in skills


def test_nominal_side_grasp_object_uses_fallback(run):
    res = run(object_id="tall_bottle", place_target="shelf_b")
    assert res.status == "SUCCESS"
    assert res.object_at_target
    fb = events(res, "fallback")
    assert len(fb) == 1 and fb[0]["error_code"] == "GRASP_PLAN_FAILED"


def test_grasp_failure_recovers(run):
    res = run(["grasp_failure"])
    assert res.status == "SUCCESS"
    assert res.object_at_target
    assert res.retries == 1
    assert res.recoveries_attempted == 1 and res.recoveries_succeeded == 1
    assert "grasp_failure" in res.faults_injected
    verify = [s for s in res.skill_stats if s.skill == "verify_grasp"]
    assert [v.status for v in verify] == ["FAILURE", "SUCCESS"]
    assert verify[1].retries == 1


def test_repeated_failure_exhausts_bounded_retries(run):
    res = run(["repeated_failure"])
    assert res.status == "FAILURE"
    assert res.error_code == "RETRIES_EXHAUSTED"
    assert res.root_error_code == "GRASP_NOT_VERIFIED"
    attempts = RecoveryPolicy().grasp_attempts
    assert len([s for s in res.skill_stats if s.skill == "execute_grasp"]) == attempts
    assert res.retries == attempts - 1
    assert len(events(res, "retries_exhausted")) == 1
    assert not res.object_at_target


def test_repeated_failure_respects_custom_budget(run):
    res = run(["repeated_failure"], policy=RecoveryPolicy(grasp_attempts=5))
    assert res.error_code == "RETRIES_EXHAUSTED"
    assert len([s for s in res.skill_stats if s.skill == "execute_grasp"]) == 5


def test_missing_object_fails_with_root_cause(run):
    res = run(["missing_object"])
    assert res.status == "FAILURE"
    assert res.error_code == "RETRIES_EXHAUSTED"
    assert res.root_error_code == "OBJECT_NOT_FOUND"
    assert "red_cube" in res.error_reason
    # never attempted a grasp on an object that was not detected
    assert not any(s.skill == "execute_grasp" for s in res.skill_stats)


def test_navigation_timeout_recovers(run):
    res = run(["navigation_timeout"])
    assert res.status == "SUCCESS"
    assert res.object_at_target
    to = events(res, "timeout")
    assert len(to) == 1 and to[0]["error_code"] == "NAV_TIMEOUT"
    assert len(events(res, "skill_halted")) == 1
    assert res.retries == 1
    assert res.sim_duration_s >= RecoveryPolicy().nav_timeout_s


def test_navigation_timeout_exhaustion(run):
    from robot_skills.executive import TaskSpec, run_task
    from robot_skills.sim import FaultConfig

    spec = TaskSpec(task_id="t", policy=RecoveryPolicy(nav_attempts=2, nav_timeout_s=5.0))
    res = run_task(spec, FaultConfig(nav_stuck_goals=10))
    assert res.status == "FAILURE"
    assert res.error_code == "RETRIES_EXHAUSTED"
    assert res.root_error_code == "NAV_TIMEOUT"
    assert len(events(res, "timeout")) == 2


def test_sensor_dropout_recovers(run):
    res = run(["sensor_dropout"])
    assert res.status == "SUCCESS"
    detect = [s for s in res.skill_stats if s.skill == "detect_object"]
    assert detect[0].status == "FAILURE"
    first_fail = next(e for e in res.events if e["event"] == "skill_end" and e["skill"] == "detect_object")
    assert first_fail["error_code"] == "SENSOR_DROPOUT"
    rec = [s for s in res.skill_stats if s.skill == "recover_from_failure"]
    assert len(rec) == 1 and rec[0].status == "SUCCESS"


def test_invalid_target_fails_fast_without_retries(run):
    res = run(["invalid_target"])
    assert res.status == "FAILURE"
    assert res.error_code == "INVALID_TARGET"
    assert res.retries == 0
    assert res.skill_stats == []  # rejected before any robot motion
    assert res.ticks == 1


def test_unknown_place_target_in_spec_fails_fast(run):
    res = run(place_target="kitchen")
    assert res.error_code == "INVALID_TARGET"
    assert "kitchen" in res.error_reason


def test_unknown_object_fails_fast(run):
    res = run(object_id="banana")
    assert res.status == "FAILURE"
    assert res.error_code == "INVALID_INPUT"
    assert res.skill_stats == []


def test_combined_faults_recover(run):
    res = run(["grasp_failure", "sensor_dropout", "navigation_timeout"])
    assert res.status == "SUCCESS"
    assert res.object_at_target
    assert set(res.faults_injected) >= {"grasp_failure", "sensor_dropout", "navigation_timeout"}
    assert res.recoveries_attempted == res.recoveries_succeeded == 3


def test_task_deadline_halts_and_fails(run):
    res = run(deadline_s=3.0)
    assert res.status == "FAILURE"
    assert res.error_code == "TASK_DEADLINE_EXCEEDED"
    assert res.sim_duration_s <= 3.0 + 1e-6
    assert len(events(res, "skill_halted")) == 1


def test_every_failure_has_a_reason(run):
    for faults in (["missing_object"], ["repeated_failure"], ["invalid_target"]):
        res = run(faults)
        assert res.status == "FAILURE"
        assert res.error_code and res.error_reason
        end = events(res, "task_end")[0]
        assert end["error_code"] == res.error_code


def test_stochastic_faults_are_seed_deterministic():
    from robot_skills.events import deterministic_view
    from robot_skills.executive import TaskSpec, run_task
    from robot_skills.sim import FaultConfig

    fc = FaultConfig(grasp_failure_prob=0.5, seed=42)
    a = run_task(TaskSpec(task_id="s"), fc)
    b = run_task(TaskSpec(task_id="s"), fc)
    assert deterministic_view(a.events) == deterministic_view(b.events)
