from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ErrorCode
from .base import Skill, SkillResult
from .contracts import RecoverInput, RecoverOutput


@dataclass(frozen=True)
class RecoveryAction:
    name: str
    min_duration_s: float
    max_duration_s: float


# Error code -> recovery behavior. Anything not listed is UNRECOVERABLE:
# the executive must fail loudly rather than guess.
RECOVERY_TABLE: dict[ErrorCode, RecoveryAction] = {
    ErrorCode.GRASP_NOT_VERIFIED: RecoveryAction("open_gripper_and_rescan", 0.3, 0.3),
    ErrorCode.GRASP_EXECUTION_FAILED: RecoveryAction("open_gripper_and_rescan", 0.3, 0.3),
    ErrorCode.NAV_TIMEOUT: RecoveryAction("cancel_and_clear_costmap", 0.2, 0.2),
    ErrorCode.NAV_ABORTED: RecoveryAction("cancel_and_clear_costmap", 0.2, 0.2),
    ErrorCode.SENSOR_DROPOUT: RecoveryAction("reset_sensor_and_wait", 0.1, 1.5),
    ErrorCode.OBJECT_NOT_FOUND: RecoveryAction("wait_and_rescan", 0.5, 0.5),
}


class RecoverFromFailure(Skill[RecoverInput, RecoverOutput]):
    """Bring the robot back to a state from which the failed stage can be
    retried. Bounded in time; never retries anything itself - the executive's
    Retry node owns the attempt budget."""

    name = "recover_from_failure"
    input_type = RecoverInput
    output_type = RecoverOutput
    errors = frozenset({ErrorCode.INVALID_INPUT, ErrorCode.RECOVERY_FAILED, ErrorCode.UNRECOVERABLE})

    def __init__(self, robot) -> None:  # type: ignore[no-untyped-def]
        super().__init__(robot)
        self._action: RecoveryAction | None = None

    def on_start(self, inp: RecoverInput, now: float) -> None:
        self._action = RECOVERY_TABLE.get(inp.error_code)
        if self._action is None:
            return
        name = self._action.name
        if name == "open_gripper_and_rescan":
            self.robot.manipulator.open_gripper()
        elif name == "cancel_and_clear_costmap":
            self.robot.base.cancel_nav()
            self.robot.base.clear_costmap()
        elif name == "reset_sensor_and_wait":
            self.robot.perception.reset_sensor()

    def tick(self, now: float) -> SkillResult[RecoverOutput]:
        inp = self.inp
        assert inp is not None
        if self._action is None:
            return self.fail(ErrorCode.UNRECOVERABLE, f"no recovery defined for {inp.error_code.value}")
        elapsed = round(now - self.started_at, 9)
        act = self._action
        if elapsed + 1e-9 < act.min_duration_s:
            return self.running()
        if act.name == "reset_sensor_and_wait":
            if self.robot.perception.read().healthy:
                return self.ok(RecoverOutput(act.name, elapsed))
            if elapsed + 1e-9 >= act.max_duration_s:
                return self.fail(ErrorCode.RECOVERY_FAILED, f"sensor still unhealthy after {elapsed:.2f}s")
            return self.running()
        return self.ok(RecoverOutput(act.name, elapsed))
