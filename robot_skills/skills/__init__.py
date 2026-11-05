from .base import Skill, SkillResult
from .contracts import (
    DetectInput,
    DetectOutput,
    ExecuteGraspOutput,
    GraspPlan,
    NavigateInput,
    NavigateOutput,
    PlaceInput,
    PlaceOutput,
    PlanGraspInput,
    RecoverInput,
    RecoverOutput,
    VerifyGraspInput,
    VerifyGraspOutput,
)
from .manipulation import ExecuteGrasp, PlaceObject, PlanGrasp, VerifyGrasp
from .navigation import NavigateToTarget
from .perception import DetectObject
from .recovery import RECOVERY_TABLE, RecoverFromFailure

SKILL_REGISTRY: dict[str, type[Skill]] = {  # type: ignore[type-arg]
    cls.name: cls
    for cls in (
        NavigateToTarget, DetectObject, PlanGrasp, ExecuteGrasp, VerifyGrasp, PlaceObject, RecoverFromFailure,
    )
}

__all__ = [
    "DetectInput", "DetectObject", "DetectOutput", "ExecuteGrasp", "ExecuteGraspOutput", "GraspPlan",
    "NavigateInput", "NavigateOutput", "NavigateToTarget", "PlaceInput", "PlaceObject", "PlaceOutput",
    "PlanGrasp", "PlanGraspInput", "RECOVERY_TABLE", "RecoverFromFailure", "RecoverInput", "RecoverOutput",
    "SKILL_REGISTRY", "Skill", "SkillResult", "VerifyGrasp", "VerifyGraspInput", "VerifyGraspOutput",
]
