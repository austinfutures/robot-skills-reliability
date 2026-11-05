"""Task trees built from reusable skills.

pick_and_place
├── validate_task                      (Condition: fail fast on bad targets)
├── ensure_holding                     (Selector: skip pick if already held)
│   ├── already_holding                (Condition)
│   └── pick                           (Sequence)
│       ├── approach_object            (Retry x3, recovery between attempts)
│       │   └── approach_timeout       (Timeout -> NAV_TIMEOUT)
│       │       └── navigate_to_object
│       └── grasp_object               (Retry x3, recovery between attempts)
│           └── grasp_sequence         (Sequence)
│               ├── detect_object
│               ├── plan_grasp_with_fallback (Fallback on GRASP_PLAN_FAILED)
│               │   ├── plan_grasp[top]
│               │   └── plan_grasp[side]
│               ├── execute_grasp
│               └── verify_grasp
├── transport                          (Retry x3 + Timeout, recovery)
│   └── transport_timeout -> navigate_to_place
└── place                              (Retry x2)
    └── place_object
"""

from __future__ import annotations

from typing import Callable

from ..bt.nodes import (
    Blackboard,
    Condition,
    Fallback,
    Node,
    Retry,
    Selector,
    Sequence,
    SkillNode,
    TickContext,
    Timeout,
    recovery_input,
)
from ..core.errors import ErrorCode, SkillError
from ..core.interfaces import RobotAPI
from ..core.types import GraspStrategy
from ..skills import (
    DetectInput,
    DetectObject,
    DetectOutput,
    ExecuteGrasp,
    GraspPlan,
    NavigateInput,
    NavigateToTarget,
    PlaceInput,
    PlaceObject,
    PlanGrasp,
    PlanGraspInput,
    RecoverFromFailure,
    VerifyGrasp,
    VerifyGraspInput,
)
from .task import TaskSpec

NAV_RETRY_ON = frozenset({ErrorCode.NAV_TIMEOUT, ErrorCode.NAV_ABORTED})
GRASP_RETRY_ON = frozenset(
    {
        ErrorCode.SENSOR_DROPOUT,
        ErrorCode.OBJECT_NOT_FOUND,
        ErrorCode.GRASP_NOT_VERIFIED,
        ErrorCode.GRASP_EXECUTION_FAILED,
    }
)


def build_pick_and_place_tree(robot: RobotAPI, spec: TaskSpec, place_target: str | None = None) -> Node:
    target = place_target if place_target is not None else spec.place_target
    pol = spec.policy
    catalog = {o.object_id: o for o in spec.scene.objects}
    obj_model = catalog.get(spec.object_id)

    def validate(ctx: TickContext) -> SkillError | None:
        if obj_model is None:
            return SkillError(
                ErrorCode.INVALID_INPUT,
                f"object '{spec.object_id}' is not in the object catalog {sorted(catalog)}",
                "validate_task",
            )
        if robot.semantic_map.lookup(spec.object_id) is None:
            return SkillError(ErrorCode.INVALID_TARGET, f"object '{spec.object_id}' has no map location", "validate_task")
        if robot.semantic_map.lookup(target) is None:
            known = [n for n in robot.semantic_map.known_names() if n not in catalog]
            return SkillError(
                ErrorCode.INVALID_TARGET, f"place target '{target}' not in map (known targets: {known})", "validate_task"
            )
        return None

    def already_holding(ctx: TickContext) -> SkillError | None:
        held = robot.manipulator.gripper_state().holding_object_id
        if held == spec.object_id:
            return None
        return SkillError(ErrorCode.NOT_HOLDING, f"not holding '{spec.object_id}'", "already_holding")

    def recover(stage: str) -> SkillNode:  # type: ignore[type-arg]
        return SkillNode(RecoverFromFailure(robot), recovery_input, "last_recovery", name=f"recover[{stage}]")

    def plan_input(strategy: GraspStrategy) -> Callable[[Blackboard], PlanGraspInput]:
        def make(bb: Blackboard) -> PlanGraspInput:
            det = bb.require("detection", DetectOutput)
            assert obj_model is not None  # guaranteed by validate_task
            return PlanGraspInput(det.object_id, det.pose, det.width_m, strategy, obj_model.grasp_strategies)

        return make

    approach = Retry(
        "approach_object",
        Timeout(
            "approach_timeout",
            SkillNode(
                NavigateToTarget(robot),
                lambda bb: NavigateInput(spec.object_id, pol.approach_standoff_m),
                "approach_result",
                name="navigate_to_object",
            ),
            pol.nav_timeout_s,
            ErrorCode.NAV_TIMEOUT,
        ),
        pol.nav_attempts,
        NAV_RETRY_ON,
        recover("approach_object"),
    )

    grasp = Retry(
        "grasp_object",
        Sequence(
            "grasp_sequence",
            [
                SkillNode(DetectObject(robot), lambda bb: DetectInput(spec.object_id), "detection"),
                Fallback(
                    "plan_grasp_with_fallback",
                    SkillNode(PlanGrasp(robot), plan_input(GraspStrategy.TOP), "grasp_plan", name="plan_grasp[top]"),
                    SkillNode(PlanGrasp(robot), plan_input(GraspStrategy.SIDE), "grasp_plan", name="plan_grasp[side]"),
                    frozenset({ErrorCode.GRASP_PLAN_FAILED}),
                ),
                SkillNode(ExecuteGrasp(robot), lambda bb: bb.require("grasp_plan", GraspPlan), "grasp_exec"),
                SkillNode(VerifyGrasp(robot), lambda bb: VerifyGraspInput(spec.object_id), "grasp_verified"),
            ],
        ),
        pol.grasp_attempts,
        GRASP_RETRY_ON,
        recover("grasp_object"),
    )

    transport = Retry(
        "transport",
        Timeout(
            "transport_timeout",
            SkillNode(
                NavigateToTarget(robot),
                lambda bb: NavigateInput(target, pol.approach_standoff_m),
                "transport_result",
                name="navigate_to_place",
            ),
            pol.nav_timeout_s,
            ErrorCode.NAV_TIMEOUT,
        ),
        pol.nav_attempts,
        NAV_RETRY_ON,
        recover("transport"),
    )

    place = Retry(
        "place",
        SkillNode(PlaceObject(robot), lambda bb: PlaceInput(spec.object_id, target), "place_result"),
        pol.place_attempts,
        frozenset({ErrorCode.PLACE_FAILED}),
    )

    return Sequence(
        "pick_and_place",
        [
            Condition("validate_task", validate),
            Selector("ensure_holding", [Condition("already_holding", already_holding), Sequence("pick", [approach, grasp])]),
            transport,
            place,
        ],
    )


def render_tree(root: Node) -> str:
    """ASCII rendering (Windows-console safe)."""
    lines = []
    for depth, node in root.iter_tree():
        lines.append(f"{'  ' * depth}- {node.name} [{type(node).__name__}]")
    return "\n".join(lines)
