from __future__ import annotations

from ..core.errors import ErrorCode
from .base import Skill, SkillResult
from .contracts import DetectInput, DetectOutput


class DetectObject(Skill[DetectInput, DetectOutput]):
    """Locate a specific object with the onboard sensor (single frame).

    Distinguishes a sensor fault (SENSOR_DROPOUT: no valid frame) from a
    valid frame that does not contain the object (OBJECT_NOT_FOUND), because
    the right recovery is different.
    """

    name = "detect_object"
    input_type = DetectInput
    output_type = DetectOutput
    errors = frozenset({ErrorCode.INVALID_INPUT, ErrorCode.SENSOR_DROPOUT, ErrorCode.OBJECT_NOT_FOUND})

    def tick(self, now: float) -> SkillResult[DetectOutput]:
        assert self.inp is not None
        reading = self.robot.perception.read()
        if not reading.healthy:
            return self.fail(ErrorCode.SENSOR_DROPOUT, "perception returned no valid frame")
        pose = self.robot.base.get_pose()
        for det in reading.detections:
            if det.object_id == self.inp.object_id and det.confidence >= self.inp.min_confidence:
                return self.ok(
                    DetectOutput(det.object_id, det.pose, det.width_m, det.confidence, round(pose.distance_to(det.pose), 6))
                )
        seen = sorted(d.object_id for d in reading.detections)
        return self.fail(
            ErrorCode.OBJECT_NOT_FOUND,
            f"'{self.inp.object_id}' not detected (visible: {seen or 'none'})",
        )
