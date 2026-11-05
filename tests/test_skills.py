"""Skill contracts, input validation, and simulator independence."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from robot_skills.bt import Blackboard, SkillNode, TickContext
from robot_skills.core import (
    ActionState,
    ErrorCode,
    GraspStrategy,
    GripperState,
    MobileBase,
    PerceptionReading,
    Pose2D,
    RobotAPI,
    SkillInputError,
    default_scene,
)
from robot_skills.core.status import Status
from robot_skills.sim import FaultConfig, Simulator2D
from robot_skills.skills import (
    SKILL_REGISTRY,
    DetectInput,
    DetectObject,
    NavigateInput,
    NavigateToTarget,
    PlaceInput,
    PlaceObject,
    PlanGrasp,
    PlanGraspInput,
    RecoverFromFailure,
    RecoverInput,
    Skill,
    SkillResult,
    VerifyGrasp,
    VerifyGraspInput,
)


def drive(node, sim, ctx, max_ticks=500):
    for _ in range(max_ticks):
        s = node.tick(ctx)
        if s is not Status.RUNNING:
            return s
        sim.step(0.1)
        ctx.now = sim.now
    raise AssertionError("did not terminate")


@pytest.fixture
def sim():
    return Simulator2D(default_scene())


def node_for(skill, inp, key="out"):
    return SkillNode(skill, lambda bb: inp, key)


# ---------------------------------------------------------- input validation
@pytest.mark.parametrize(
    "factory",
    [
        lambda: NavigateInput(""),
        lambda: NavigateInput("bin_a", standoff_m=-1.0),
        lambda: NavigateInput("bin_a", tolerance_m=0.0),
        lambda: NavigateInput("bin_a", standoff_m=float("nan")),
        lambda: DetectInput("   "),
        lambda: DetectInput("red_cube", min_confidence=1.5),
        lambda: PlanGraspInput("red_cube", "not a pose", 0.05, GraspStrategy.TOP, (GraspStrategy.TOP,)),  # type: ignore[arg-type]
        lambda: PlanGraspInput("red_cube", Pose2D(0, 0), -0.1, GraspStrategy.TOP, (GraspStrategy.TOP,)),
        lambda: PlanGraspInput("red_cube", Pose2D(0, 0), 0.05, GraspStrategy.TOP, ()),
        lambda: VerifyGraspInput(""),
        lambda: PlaceInput("red_cube", ""),
        lambda: RecoverInput(ErrorCode.NAV_TIMEOUT, "stage", 0),
        lambda: RecoverInput("NAV_TIMEOUT", "stage", 1),  # type: ignore[arg-type]
        lambda: RecoverInput(ErrorCode.NAV_TIMEOUT, "stage", True),  # type: ignore[arg-type]
    ],
)
def test_invalid_inputs_rejected(factory):
    with pytest.raises(SkillInputError) as exc:
        factory()
    assert exc.value.code is ErrorCode.INVALID_INPUT


def test_pose_rejects_non_finite():
    with pytest.raises(ValueError):
        Pose2D(math.inf, 0.0)


def test_unknown_fault_name_rejected():
    with pytest.raises(ValueError, match="unknown fault"):
        FaultConfig.from_names(["laser_eyes"])


def test_wrong_input_type_fails_via_skillnode(sim):
    ctx = TickContext(now=0.0)
    n = node_for(NavigateToTarget(sim.api()), DetectInput("red_cube"))
    assert n.tick(ctx) is Status.FAILURE
    assert n.error.code is ErrorCode.INVALID_INPUT


def test_input_builder_exception_becomes_invalid_input(sim):
    ctx = TickContext(now=0.0, blackboard=Blackboard())
    n = SkillNode(VerifyGrasp(sim.api()), lambda bb: bb.require("nope", VerifyGraspInput))
    assert n.tick(ctx) is Status.FAILURE
    assert n.error.code is ErrorCode.INVALID_INPUT


# ------------------------------------------------------------- skills on sim
def test_navigate_reaches_standoff_facing_target(sim):
    ctx = TickContext(now=0.0)
    n = node_for(NavigateToTarget(sim.api()), NavigateInput("red_cube", standoff_m=0.45))
    assert drive(n, sim, ctx) is Status.SUCCESS
    out = ctx.blackboard["out"]
    target = default_scene().objects[0].pose
    assert out.position_error_m <= 0.05
    assert abs(sim.pose.distance_to(target) - 0.45) <= 0.06


def test_navigate_unknown_target(sim):
    ctx = TickContext(now=0.0)
    n = node_for(NavigateToTarget(sim.api()), NavigateInput("narnia"))
    assert n.tick(ctx) is Status.FAILURE
    assert n.error.code is ErrorCode.INVALID_TARGET


def test_detect_out_of_view_and_dropout():
    s = Simulator2D(default_scene())
    ctx = TickContext(now=0.0)
    # blue_box is behind the robot (outside FOV)
    n = node_for(DetectObject(s.api()), DetectInput("blue_box"))
    assert n.tick(ctx) is Status.FAILURE and n.error.code is ErrorCode.OBJECT_NOT_FOUND

    s2 = Simulator2D(default_scene(), FaultConfig(sensor_dropout_s=0.5))
    n2 = node_for(DetectObject(s2.api()), DetectInput("red_cube"))
    assert n2.tick(ctx) is Status.FAILURE and n2.error.code is ErrorCode.SENSOR_DROPOUT


def test_plan_grasp_out_of_reach(sim):
    ctx = TickContext(now=0.0)
    inp = PlanGraspInput("red_cube", Pose2D(2.5, 1.0), 0.05, GraspStrategy.TOP, (GraspStrategy.TOP,))
    n = node_for(PlanGrasp(sim.api()), inp)
    assert n.tick(ctx) is Status.FAILURE and n.error.code is ErrorCode.OUT_OF_REACH


def test_plan_grasp_unsupported_strategy(sim):
    ctx = TickContext(now=0.0)
    inp = PlanGraspInput("tall_bottle", Pose2D(0.3, 0.0), 0.07, GraspStrategy.TOP, (GraspStrategy.SIDE,))
    n = node_for(PlanGrasp(sim.api()), inp)
    assert n.tick(ctx) is Status.FAILURE and n.error.code is ErrorCode.GRASP_PLAN_FAILED


def test_verify_and_place_require_holding(sim):
    ctx = TickContext(now=0.0)
    v = node_for(VerifyGrasp(sim.api()), VerifyGraspInput("red_cube"))
    assert v.tick(ctx) is Status.FAILURE and v.error.code is ErrorCode.GRASP_NOT_VERIFIED
    p = node_for(PlaceObject(sim.api()), PlaceInput("red_cube", "bin_a"))
    assert p.tick(ctx) is Status.FAILURE and p.error.code is ErrorCode.NOT_HOLDING
    p2 = node_for(PlaceObject(sim.api()), PlaceInput("red_cube", "moon"))
    assert p2.tick(ctx) is Status.FAILURE and p2.error.code is ErrorCode.INVALID_TARGET


def test_recovery_unrecoverable_code(sim):
    ctx = TickContext(now=0.0)
    n = node_for(RecoverFromFailure(sim.api()), RecoverInput(ErrorCode.OUT_OF_REACH, "grasp", 1))
    assert n.tick(ctx) is Status.FAILURE and n.error.code is ErrorCode.UNRECOVERABLE


def test_recovery_sensor_gives_up_when_sensor_stays_down():
    s = Simulator2D(default_scene(), FaultConfig(sensor_dropout_s=100.0))
    s.api().perception.read()  # start the dropout window
    ctx = TickContext(now=0.0)
    n = node_for(RecoverFromFailure(s.api()), RecoverInput(ErrorCode.SENSOR_DROPOUT, "grasp", 1))
    assert drive(n, s, ctx) is Status.FAILURE
    assert n.error.code is ErrorCode.RECOVERY_FAILED


# --------------------------------------------------------- contract checks
class _BadSkill(Skill[VerifyGraspInput, object]):
    name = "bad_skill"
    input_type = VerifyGraspInput
    output_type = int
    errors = frozenset({ErrorCode.INVALID_INPUT})

    def __init__(self, robot, mode):
        super().__init__(robot)
        self.mode = mode

    def tick(self, now):
        if self.mode == "undeclared":
            return self.fail(ErrorCode.NAV_ABORTED, "not in contract")
        if self.mode == "wrong_type":
            return SkillResult(Status.SUCCESS, output="a string")
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    "mode,code",
    [("undeclared", ErrorCode.CONTRACT_VIOLATION), ("wrong_type", ErrorCode.CONTRACT_VIOLATION),
     ("crash", ErrorCode.INTERNAL_ERROR)],
)
def test_skillnode_enforces_contract(sim, mode, code):
    ctx = TickContext(now=0.0)
    n = node_for(_BadSkill(sim.api(), mode), VerifyGraspInput("x"))
    assert n.tick(ctx) is Status.FAILURE
    assert n.error.code is code


def test_all_skills_declare_contracts():
    expected = {"navigate_to_target", "detect_object", "plan_grasp", "execute_grasp", "verify_grasp",
                "place_object", "recover_from_failure"}
    assert set(SKILL_REGISTRY) == expected
    for cls in SKILL_REGISTRY.values():
        c = cls.describe_contract()
        assert c["input"] and c["output"] and "INVALID_INPUT" in c["errors"]


# ------------------------------------------------- simulator independence
def test_skills_package_never_imports_simulator():
    skills_dir = Path(__file__).resolve().parent.parent / "robot_skills" / "skills"
    for f in skills_dir.glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "sim" not in (node.module or "").split("."), f"{f.name} imports {node.module}"
            if isinstance(node, ast.Import):
                assert all("sim" not in a.name.split(".") for a in node.names)


class _FakeBase:
    """Stand-in for a ROS 2 adapter: proves skills run on any RobotAPI."""

    def __init__(self):
        self.pose = Pose2D(0, 0)
        self.state = ActionState.IDLE

    def get_pose(self):
        return self.pose

    def send_nav_goal(self, goal, tolerance_m):
        self.pose, self.state = goal, ActionState.SUCCEEDED

    def nav_state(self):
        return self.state

    def cancel_nav(self):
        self.state = ActionState.CANCELED

    def clear_costmap(self):
        pass


class _FakePerception:
    def read(self):
        return PerceptionReading(True, ())

    def reset_sensor(self):
        pass

    def max_range_m(self):
        return 1.0


class _FakeArm:
    def reach_m(self):
        return 0.5

    def send_grasp(self, *a):
        pass

    def send_place(self, *a):
        pass

    def arm_state(self):
        return ActionState.IDLE

    def gripper_state(self):
        return GripperState(False, None)

    def open_gripper(self):
        pass


class _FakeMap:
    def lookup(self, name):
        return Pose2D(1.0, 1.0) if name == "dock" else None

    def known_names(self):
        return ("dock",)


def test_skills_run_against_non_simulator_backend():
    base = _FakeBase()
    assert isinstance(base, MobileBase)
    robot = RobotAPI(base, _FakePerception(), _FakeArm(), _FakeMap())
    ctx = TickContext(now=0.0)
    n = node_for(NavigateToTarget(robot), NavigateInput("dock"))
    assert n.tick(ctx) is Status.SUCCESS
    d = node_for(DetectObject(robot), DetectInput("thing"))
    assert d.tick(ctx) is Status.FAILURE and d.error.code is ErrorCode.OBJECT_NOT_FOUND
