"""Lightweight deterministic 2D mobile-manipulator simulator.

This is a kinematic simulation (no dynamics, contacts or physics). It exists
to exercise skills, the executive and fault handling - it is NOT a model of
real hardware. It implements the interfaces in `robot_skills.core.interfaces`
so skills never import it directly.

Time only advances through `step(dt)`, which makes every run reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.interfaces import (
    ActionState,
    Detection,
    GripperState,
    PerceptionReading,
    RobotAPI,
)
from ..core.types import Pose2D, SceneSpec, wrap_angle
from .faults import FaultConfig, FaultEvent, FaultInjector


@dataclass
class SimParams:
    max_linear_speed: float = 0.5  # m/s
    max_angular_speed: float = 1.2  # rad/s
    heading_tolerance: float = 0.05  # rad
    sensor_range: float = 2.5  # m
    sensor_fov: float = math.radians(120.0)
    arm_reach: float = 0.65  # m, from base centre
    grasp_duration: float = 1.0  # s
    place_duration: float = 0.8  # s


@dataclass
class _ObjectState:
    object_id: str
    pose: Pose2D
    width_m: float
    held: bool = False


@dataclass
class _ArmAction:
    kind: str  # "grasp" | "place"
    start: float
    duration: float
    target: Pose2D
    width: float = 0.0


class Simulator2D:
    def __init__(
        self,
        scene: SceneSpec,
        faults: FaultConfig | None = None,
        params: SimParams | None = None,
        task_object_id: str | None = None,
    ) -> None:
        self.scene = scene
        self.params = params or SimParams()
        self.faults = FaultInjector(faults or FaultConfig())
        self.t = 0.0
        self.pose = scene.robot_start
        self.objects: dict[str, _ObjectState] = {
            o.object_id: _ObjectState(o.object_id, o.pose, o.width_m) for o in scene.objects
        }
        if self.faults.config.missing_task_object and task_object_id in self.objects:
            del self.objects[task_object_id]
            self.faults.events.append(
                FaultEvent("missing_object", 0.0, f"{task_object_id} removed from world")
            )
        # navigation
        self._nav_goal: Pose2D | None = None
        self._nav_tol = 0.05
        self._nav_state = ActionState.IDLE
        self._nav_stuck = False
        # manipulation
        self._arm: _ArmAction | None = None
        self._arm_state = ActionState.IDLE
        self._gripper_closed = False
        self._held: str | None = None
        self._sensor_reset_count = 0
        self.costmap_clears = 0

    # ------------------------------------------------------------------ time
    @property
    def now(self) -> float:
        return self.t

    def step(self, dt: float) -> None:
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"dt must be positive and finite, got {dt}")
        self.t = round(self.t + dt, 9)
        self._step_nav(dt)
        self._step_arm()
        if self._held is not None:
            self.objects[self._held].pose = Pose2D(self.pose.x, self.pose.y, 0.0)

    def drain_fault_events(self) -> list[FaultEvent]:
        ev, self.faults.events = self.faults.events, []
        return ev

    def api(self) -> RobotAPI:
        return RobotAPI(
            base=_Base(self),
            perception=_Perception(self),
            manipulator=_Manipulator(self),
            semantic_map=_Map(self),
        )

    # ------------------------------------------------------------ navigation
    def _step_nav(self, dt: float) -> None:
        if self._nav_state is not ActionState.ACTIVE or self._nav_goal is None:
            return
        if self._nav_stuck:
            return  # blocked: no progress until the goal is cancelled
        p, g, prm = self.pose, self._nav_goal, self.params
        dist = p.distance_to(g)
        if dist > self._nav_tol:
            err = wrap_angle(p.bearing_to(g) - p.theta)
            max_turn = prm.max_angular_speed * dt
            turn = max(-max_turn, min(max_turn, err))
            theta = wrap_angle(p.theta + turn)
            v = 0.0 if abs(err) > 0.3 else min(prm.max_linear_speed, dist / dt)
            step = v * dt
            self.pose = Pose2D(p.x + step * math.cos(theta), p.y + step * math.sin(theta), theta)
            return
        err = wrap_angle(g.theta - p.theta)
        if abs(err) <= prm.heading_tolerance:
            self._nav_state = ActionState.SUCCEEDED
            return
        max_turn = prm.max_angular_speed * dt
        self.pose = Pose2D(p.x, p.y, wrap_angle(p.theta + max(-max_turn, min(max_turn, err))))

    # ---------------------------------------------------------- manipulation
    def _step_arm(self) -> None:
        a = self._arm
        if a is None or self._arm_state is not ActionState.ACTIVE:
            return
        if self.t + 1e-9 < a.start + a.duration:
            return
        self._arm = None
        reach = self.params.arm_reach
        if a.kind == "grasp":
            if self.pose.distance_to(a.target) > reach:
                self._arm_state = ActionState.ABORTED
                return
            self._gripper_closed = True
            obj = self._object_near(a.target, tol=0.08)
            if obj is None or obj.width_m > a.width + 1e-9 or self.faults.grasp_should_slip(self.t):
                self._held = None  # closed on nothing
            else:
                obj.held = True
                self._held = obj.object_id
            self._arm_state = ActionState.SUCCEEDED
        else:  # place
            if self._held is None or self.pose.distance_to(a.target) > reach:
                self._arm_state = ActionState.ABORTED
                return
            obj = self.objects[self._held]
            obj.held = False
            obj.pose = Pose2D(a.target.x, a.target.y, 0.0)
            self._held = None
            self._gripper_closed = False
            self._arm_state = ActionState.SUCCEEDED

    def _object_near(self, pose: Pose2D, tol: float) -> _ObjectState | None:
        for obj in sorted(self.objects.values(), key=lambda o: o.object_id):
            if not obj.held and obj.pose.distance_to(pose) <= tol:
                return obj
        return None


# ---------------------------------------------------------------------------
# Interface adapters (thin views over the simulator state)
# ---------------------------------------------------------------------------
class _Base:
    def __init__(self, sim: Simulator2D) -> None:
        self._s = sim

    def get_pose(self) -> Pose2D:
        return self._s.pose

    def send_nav_goal(self, goal: Pose2D, tolerance_m: float) -> None:
        s = self._s
        s._nav_goal, s._nav_tol = goal, tolerance_m
        s._nav_state = ActionState.ACTIVE
        s._nav_stuck = s.faults.nav_goal_should_stick(s.t)

    def nav_state(self) -> ActionState:
        return self._s._nav_state

    def cancel_nav(self) -> None:
        s = self._s
        if s._nav_state is ActionState.ACTIVE:
            s._nav_state = ActionState.CANCELED
        s._nav_goal, s._nav_stuck = None, False

    def clear_costmap(self) -> None:
        self._s.costmap_clears += 1


class _Perception:
    def __init__(self, sim: Simulator2D) -> None:
        self._s = sim

    def max_range_m(self) -> float:
        return self._s.params.sensor_range

    def reset_sensor(self) -> None:
        self._s._sensor_reset_count += 1

    def read(self) -> PerceptionReading:
        s = self._s
        if s.faults.sensor_dropped(s.t):
            return PerceptionReading(healthy=False)
        dets: list[Detection] = []
        for obj in sorted(s.objects.values(), key=lambda o: o.object_id):
            if obj.held:
                continue
            d = s.pose.distance_to(obj.pose)
            bearing = wrap_angle(s.pose.bearing_to(obj.pose) - s.pose.theta)
            if d <= s.params.sensor_range and abs(bearing) <= s.params.sensor_fov / 2:
                conf = round(1.0 - 0.3 * d / s.params.sensor_range, 4)
                dets.append(Detection(obj.object_id, obj.pose, obj.width_m, conf))
        return PerceptionReading(healthy=True, detections=tuple(dets))


class _Manipulator:
    def __init__(self, sim: Simulator2D) -> None:
        self._s = sim

    def reach_m(self) -> float:
        return self._s.params.arm_reach

    def send_grasp(self, object_pose: Pose2D, gripper_width_m: float, approach_heading: float) -> None:
        s = self._s
        s._arm = _ArmAction("grasp", s.t, s.params.grasp_duration, object_pose, gripper_width_m)
        s._arm_state = ActionState.ACTIVE

    def send_place(self, pose: Pose2D) -> None:
        s = self._s
        s._arm = _ArmAction("place", s.t, s.params.place_duration, pose)
        s._arm_state = ActionState.ACTIVE

    def arm_state(self) -> ActionState:
        return self._s._arm_state

    def gripper_state(self) -> GripperState:
        return GripperState(self._s._gripper_closed, self._s._held)

    def open_gripper(self) -> None:
        s = self._s
        if s._held is not None:
            s.objects[s._held].held = False
        s._held, s._gripper_closed = None, False
        s._arm, s._arm_state = None, ActionState.IDLE


class _Map:
    def __init__(self, sim: Simulator2D) -> None:
        self._s = sim
        self._prior = dict(sim.scene.targets)
        for o in sim.scene.objects:
            self._prior[o.object_id] = o.pose  # prior map, may be stale

    def lookup(self, name: str) -> Pose2D | None:
        return self._prior.get(name)

    def known_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._prior))
