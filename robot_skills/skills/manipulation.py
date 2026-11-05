from __future__ import annotations

from ..core.errors import ErrorCode, SkillInputError
from ..core.interfaces import ActionState
from ..core.types import GraspStrategy, wrap_angle
from .base import Skill, SkillResult
from .contracts import (
    MAX_GRIPPER_OPENING_M,
    ExecuteGraspOutput,
    GraspPlan,
    PlaceInput,
    PlaceOutput,
    PlanGraspInput,
    VerifyGraspInput,
    VerifyGraspOutput,
)

GRIPPER_CLEARANCE_M = 0.02


class PlanGrasp(Skill[PlanGraspInput, GraspPlan]):
    """Compute a grasp for a detected object with the requested strategy.

    Pure computation (no robot motion). A real implementation would call a
    grasp planner; here feasibility is: strategy supported by the object
    model, object within arm reach, object fits the gripper.
    """

    name = "plan_grasp"
    input_type = PlanGraspInput
    output_type = GraspPlan
    errors = frozenset({ErrorCode.INVALID_INPUT, ErrorCode.GRASP_PLAN_FAILED, ErrorCode.OUT_OF_REACH})

    def tick(self, now: float) -> SkillResult[GraspPlan]:
        inp = self.inp
        assert inp is not None
        if inp.strategy not in inp.supported_strategies:
            return self.fail(
                ErrorCode.GRASP_PLAN_FAILED,
                f"strategy '{inp.strategy.value}' not supported for {inp.object_id}",
            )
        width = inp.object_width_m + GRIPPER_CLEARANCE_M
        if width > MAX_GRIPPER_OPENING_M:
            return self.fail(
                ErrorCode.GRASP_PLAN_FAILED,
                f"{inp.object_id} width {inp.object_width_m:.3f} m exceeds gripper opening",
            )
        base = self.robot.base.get_pose()
        dist = base.distance_to(inp.object_pose)
        reach = self.robot.manipulator.reach_m()
        if dist > reach:
            return self.fail(ErrorCode.OUT_OF_REACH, f"object {dist:.3f} m away, reach {reach:.3f} m")
        heading = base.theta if inp.strategy is GraspStrategy.TOP else base.bearing_to(inp.object_pose)
        return self.ok(GraspPlan(inp.object_id, inp.object_pose, round(width, 6), wrap_angle(heading), inp.strategy))


class ExecuteGrasp(Skill[GraspPlan, ExecuteGraspOutput]):
    """Run the arm to the grasp pose and close the gripper.

    SUCCESS means the motion completed and the gripper closed. It does NOT
    mean the object is held - that is verify_grasp's job (objects slip).
    """

    name = "execute_grasp"
    input_type = GraspPlan
    output_type = ExecuteGraspOutput
    errors = frozenset({ErrorCode.INVALID_INPUT, ErrorCode.GRASP_EXECUTION_FAILED})

    def on_start(self, inp: GraspPlan, now: float) -> None:
        if self.robot.manipulator.gripper_state().holding_object_id is not None:
            raise SkillInputError(ErrorCode.GRASP_EXECUTION_FAILED, "gripper already holding an object")
        self.robot.manipulator.send_grasp(inp.grasp_pose, inp.gripper_width_m, inp.approach_heading)

    def tick(self, now: float) -> SkillResult[ExecuteGraspOutput]:
        assert self.inp is not None
        state = self.robot.manipulator.arm_state()
        if state is ActionState.ACTIVE:
            return self.running()
        if state is ActionState.SUCCEEDED and self.robot.manipulator.gripper_state().closed:
            return self.ok(ExecuteGraspOutput(self.inp.object_id, True))
        return self.fail(ErrorCode.GRASP_EXECUTION_FAILED, f"arm action ended in state {state.value}")

    def halt(self) -> None:
        if self.robot.manipulator.arm_state() is ActionState.ACTIVE:
            self.robot.manipulator.open_gripper()


class VerifyGrasp(Skill[VerifyGraspInput, VerifyGraspOutput]):
    """Confirm the expected object is in the gripper (gripper object sensor)."""

    name = "verify_grasp"
    input_type = VerifyGraspInput
    output_type = VerifyGraspOutput
    errors = frozenset({ErrorCode.INVALID_INPUT, ErrorCode.GRASP_NOT_VERIFIED})

    def tick(self, now: float) -> SkillResult[VerifyGraspOutput]:
        assert self.inp is not None
        gs = self.robot.manipulator.gripper_state()
        if gs.holding_object_id == self.inp.object_id:
            return self.ok(VerifyGraspOutput(self.inp.object_id, True))
        return self.fail(
            ErrorCode.GRASP_NOT_VERIFIED,
            f"expected '{self.inp.object_id}' in gripper, found {gs.holding_object_id!r} (closed={gs.closed})",
        )


class PlaceObject(Skill[PlaceInput, PlaceOutput]):
    """Place the held object at a named target location."""

    name = "place_object"
    input_type = PlaceInput
    output_type = PlaceOutput
    errors = frozenset(
        {
            ErrorCode.INVALID_INPUT,
            ErrorCode.INVALID_TARGET,
            ErrorCode.NOT_HOLDING,
            ErrorCode.OUT_OF_REACH,
            ErrorCode.PLACE_FAILED,
        }
    )

    def on_start(self, inp: PlaceInput, now: float) -> None:
        loc = self.robot.semantic_map.lookup(inp.target_name)
        if loc is None:
            raise SkillInputError(ErrorCode.INVALID_TARGET, f"unknown place target '{inp.target_name}'")
        held = self.robot.manipulator.gripper_state().holding_object_id
        if held != inp.object_id:
            raise SkillInputError(ErrorCode.NOT_HOLDING, f"cannot place '{inp.object_id}': holding {held!r}")
        dist = self.robot.base.get_pose().distance_to(loc)
        if dist > self.robot.manipulator.reach_m():
            raise SkillInputError(ErrorCode.OUT_OF_REACH, f"place target {dist:.3f} m away exceeds reach")
        self._loc = loc
        self.robot.manipulator.send_place(loc)

    def tick(self, now: float) -> SkillResult[PlaceOutput]:
        assert self.inp is not None
        state = self.robot.manipulator.arm_state()
        if state is ActionState.ACTIVE:
            return self.running()
        if state is ActionState.SUCCEEDED:
            return self.ok(PlaceOutput(self.inp.object_id, self.inp.target_name, self._loc))
        return self.fail(ErrorCode.PLACE_FAILED, f"place action ended in state {state.value}")
