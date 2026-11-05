from __future__ import annotations

import math

from ..core.errors import ErrorCode, SkillInputError
from ..core.interfaces import ActionState
from ..core.types import Pose2D
from .base import Skill, SkillResult
from .contracts import NavigateInput, NavigateOutput


class NavigateToTarget(Skill[NavigateInput, NavigateOutput]):
    """Drive the base to a named location from the semantic map.

    Preconditions: target exists in the map.
    Postconditions (SUCCESS): base within tolerance of the goal, facing the target.
    Timeouts are applied by the executive (Timeout node), which calls halt().
    """

    name = "navigate_to_target"
    input_type = NavigateInput
    output_type = NavigateOutput
    errors = frozenset({ErrorCode.INVALID_INPUT, ErrorCode.INVALID_TARGET, ErrorCode.NAV_ABORTED})

    def __init__(self, robot) -> None:  # type: ignore[no-untyped-def]
        super().__init__(robot)
        self._goal: Pose2D | None = None

    def on_start(self, inp: NavigateInput, now: float) -> None:
        loc = self.robot.semantic_map.lookup(inp.target_name)
        if loc is None:
            raise SkillInputError(ErrorCode.INVALID_TARGET, f"unknown navigation target '{inp.target_name}'")
        self._goal = self.compute_goal(self.robot.base.get_pose(), loc, inp.standoff_m)
        self.robot.base.send_nav_goal(self._goal, inp.tolerance_m)

    @staticmethod
    def compute_goal(current: Pose2D, location: Pose2D, standoff_m: float) -> Pose2D:
        """Pose `standoff_m` short of `location` along the line from the robot,
        oriented to face the location."""
        if standoff_m <= 0.0 or current.distance_to(location) < 1e-9:
            heading = current.bearing_to(location) if current.distance_to(location) > 1e-9 else current.theta
            return Pose2D(location.x, location.y, heading)
        back = location.bearing_to(current)
        gx = location.x + standoff_m * math.cos(back)
        gy = location.y + standoff_m * math.sin(back)
        goal_xy = Pose2D(gx, gy)
        return Pose2D(gx, gy, goal_xy.bearing_to(location))

    def tick(self, now: float) -> SkillResult[NavigateOutput]:
        state = self.robot.base.nav_state()
        assert self._goal is not None
        if state is ActionState.ACTIVE:
            return self.running()
        if state is ActionState.SUCCEEDED:
            pose = self.robot.base.get_pose()
            return self.ok(NavigateOutput(self._goal, pose, round(pose.distance_to(self._goal), 6)))
        return self.fail(ErrorCode.NAV_ABORTED, f"navigation ended in state {state.value}")

    def halt(self) -> None:
        if self.robot.base.nav_state() is ActionState.ACTIVE:
            self.robot.base.cancel_nav()
