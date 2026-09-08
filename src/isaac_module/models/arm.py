"""viam:isaac-sim-devin:arm - a simulated arm.

Attributes:
  world (string, required)   - name of the viam:isaac-sim-devin:world component
  asset (string)             - known robot, e.g. "ur20", "ur10", "franka"
  usd_path (string)          - explicit USD to spawn instead of a known asset
  prim_path (string)         - where to place it (default /World/<name>), or
                               an existing articulation in the stage
  position ([x,y,z] meters)  - spawn position
  end_effector_prim (string) - prim whose pose, in the arm base frame, is
                               reported by GetEndPosition (default
                               <arm prim>/wrist_3_link for UR assets)
  move_timeout_sec (float)   - max time to wait for a move (default 30)
  kinematics_url (string)    - where to fetch the kinematics file served by
                               GetKinematics (.json = SVA, .urdf = URDF;
                               file:// URLs work). Known assets with official
                               viam kinematics (ur3e/ur5e/ur20) fetch them
                               automatically.

Note: GetEndPosition reports the end effector pose in the arm base frame
(not world frame) as of this release.

Move completion (ARM-12/ARM-13, FINDINGS R-7/R-8): moves settle via
ArmHandle.wait_for_settle - no wall-clock polling - and raise one of:
  JointTargetOutOfLimitsError (ValueError, INVALID_ARGUMENT)  - a target is
    outside the SVA's declared joint limits, or the joint count doesn't
    match the arm's DOF count.
  ArmMoveStalledError (ABORTED)      - the arm stopped moving before
    reaching its target (e.g. blocked by an obstacle).
  ArmMoveTimeoutError (TimeoutError, DEADLINE_EXCEEDED) - the move deadline
    (move_timeout_sec, capped by the SDK's timeout= kwarg) passed while the
    arm was still converging.
move_through_joint_positions honours MoveOptions.max_vel_degs_per_sec_joints
(the min across joints) when set, else max_vel_degs_per_sec; the
acceleration fields and max_tcp_speed are logged once and not honoured.
DoCommand "all_dof_names" returns every DOF of the articulation (arm joints
plus anything attached under it, e.g. a gripper); "dof_names" stays just the
arm's named joints.

close() releases the handle and its post-reset hooks (XC-4); the prim stays
in the stage. A reconfigure that changes a spawn attribute (asset, usd_path,
prim_path, position, or the frame it derives from) after the arm is already
attached raises ValueError - restart the module to apply it.
"""

import asyncio
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from grpclib import Status
from typing_extensions import Self
from viam.components.arm import Arm, JointPositions, KinematicsFileFormat, Pose
from viam.errors import MethodNotImplementedError, ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.proto.component.arm import MoveOptions
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE
from ..sim_manager import (
    KNOWN_ASSETS,
    SETTLE_TOL_RAD,
    ArmHandle,
    SettleOutcome,
    SimManager,
    _prim_name,
)
from ..spatial import quat_to_ov
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component

_TOLERANCE_RAD = SETTLE_TOL_RAD


class JointTargetOutOfLimitsError(ViamGRPCError, ValueError):
    """A commanded joint target is outside the SVA's declared limits, or the
    number of joint values doesn't match the arm's DOF count (ARM-13)."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.INVALID_ARGUMENT)
        Exception.__init__(self, message)


class ArmMoveStalledError(ViamGRPCError):
    """The arm stopped moving (velocities settled) before reaching its
    commanded target - e.g. blocked by an obstacle (ARM-12/ARM-13)."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.ABORTED)
        Exception.__init__(self, message)


class ArmMoveTimeoutError(ViamGRPCError, TimeoutError):
    """The move deadline passed while the arm was still converging on its
    target (ARM-12/ARM-13)."""

    def __init__(self, message: str) -> None:
        ViamGRPCError.__init__(self, message, Status.DEADLINE_EXCEEDED)
        Exception.__init__(self, message)


def _stuck_joint_detail(
    current: Sequence[float], targets: Sequence[float], tolerance_rad: float
) -> str:
    """ "jN: at <deg> want <deg>" for every joint outside tolerance."""
    return ", ".join(
        f"j{i}: at {math.degrees(c):.1f} want {math.degrees(t):.1f}"
        for i, (c, t) in enumerate(zip(current, targets, strict=True))
        if abs(c - t) > tolerance_rad
    )


class IsaacArm(Arm, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "arm")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: ArmHandle | None = None
        self._attrs: dict[str, Any] = {}
        self._move_timeout = 30.0
        self._kinematics: tuple[KinematicsFileFormat.ValueType, bytes] | None = None
        self._kinematics_load_has_failed = False
        self._has_warned_options = False

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        arm = cls(config.name)
        arm.reconfigure(config, dependencies)
        return arm

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        return validate_sim_component(config)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        self._move_timeout = float(attrs.get("move_timeout_sec", 30.0))
        self._attrs = attrs
        self._handle = SimManager.get().create_arm(self.name, attrs)

    async def close(self) -> None:
        """XC-4: release the handle (hooks, callbacks); the prim stays attached."""
        SimManager.get().release_handle(self.name)
        self._handle = None

    def _h(self) -> ArmHandle:
        if self._handle is None:
            raise RuntimeError(f"arm {self.name} is not attached to the sim")
        return self._handle

    def _deadline_s(self, timeout: float | None) -> float:
        """move_timeout_sec, capped by the SDK's timeout= kwarg when given."""
        return self._move_timeout if timeout is None else min(self._move_timeout, timeout)

    async def get_end_position(self, **kwargs) -> Pose:
        (x, y, z), quat = await asyncio.to_thread(self._h().get_end_pose)
        ox, oy, oz, theta = quat_to_ov(quat)
        return Pose(
            x=x * 1000.0,
            y=y * 1000.0,
            z=z * 1000.0,
            o_x=ox,
            o_y=oy,
            o_z=oz,
            theta=math.degrees(theta),
        )

    async def move_to_position(self, pose: Pose, **kwargs) -> None:
        # IK and motion planning are Viam's job, not the module's: the motion
        # service does this via GetKinematics + move_to_joint_positions.
        raise MethodNotImplementedError("move_to_position")

    async def _joint_limits_deg(self) -> list[tuple[str, float, float]] | None:
        """(joint id, min_deg, max_deg) per joint from the SVA kinematics, in
        SVA joint order - the order set_joint_targets/move_to_joint_positions
        already expect. None when limits aren't available: no kinematics
        configured, URDF format, or the file failed to load (logged once)."""
        kinematics = self._kinematics
        if kinematics is None:
            url = self._kinematics_url()
            if not url:
                return None
            try:
                kinematics = await asyncio.to_thread(self._load_kinematics)
                self._kinematics = kinematics
            except Exception:
                if not self._kinematics_load_has_failed:
                    self.logger.warning(
                        "arm %s: could not load kinematics for joint-limit checking; skipping",
                        self.name,
                    )
                    self._kinematics_load_has_failed = True
                return None

        fmt, data = kinematics
        if fmt != KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA:
            return None
        try:
            joints = json.loads(data).get("joints", [])
            return [
                (j.get("id", f"joint{i}"), float(j["min"]), float(j["max"]))
                for i, j in enumerate(joints)
            ]
        except Exception:
            if not self._kinematics_load_has_failed:
                self.logger.warning(
                    "arm %s: could not parse SVA kinematics for joint-limit checking; skipping",
                    self.name,
                )
                self._kinematics_load_has_failed = True
            return None

    async def _check_joint_targets(self, targets_rad: Sequence[float]) -> None:
        """Raise JointTargetOutOfLimitsError if a target is outside the SVA's
        declared limits (ARM-13). A no-op when limits aren't available."""
        limits = await self._joint_limits_deg()
        if limits is None:
            return
        for i, t in enumerate(targets_rad):
            if i >= len(limits):
                break
            joint_id, min_deg, max_deg = limits[i]
            deg = math.degrees(t)
            if deg < min_deg or deg > max_deg:
                raise JointTargetOutOfLimitsError(
                    f"arm {self.name}: joint {joint_id} target {deg:.2f} deg "
                    f"out of range [{min_deg:.2f}, {max_deg:.2f}]"
                )

    async def _settle_or_raise(
        self,
        handle: ArmHandle,
        targets: Sequence[float],
        deadline_s: float,
        tolerance_rad: float,
        *,
        detail_prefix: str,
    ) -> None:
        outcome = await asyncio.to_thread(handle.wait_for_settle, deadline_s, tolerance_rad)
        if outcome is SettleOutcome.REACHED:
            return
        # Hold where we are: leaving the drive target at an unreachable pose
        # keeps the arm pushing into whatever blocked it (GPU run 15 launched
        # the block and wound the elbow up that way).
        await asyncio.to_thread(handle.stop)
        current = await asyncio.to_thread(handle.get_joint_positions)
        detail = _stuck_joint_detail(current, targets, tolerance_rad)
        if outcome is SettleOutcome.STALLED:
            raise ArmMoveStalledError(f"{detail_prefix} stalled (stuck joints: {detail})")
        raise ArmMoveTimeoutError(
            f"{detail_prefix} did not reach target within {deadline_s:.1f}s "
            f"(stuck joints: {detail})"
        )

    async def move_to_joint_positions(
        self, positions: JointPositions, *, timeout: float | None = None, **kwargs
    ) -> None:
        targets = [math.radians(v) for v in positions.values]
        handle = self._h()
        current = await asyncio.to_thread(handle.get_joint_positions)
        if len(current) != len(targets):
            raise JointTargetOutOfLimitsError(
                f"arm {self.name}: expected {len(current)} joint values, got {len(targets)}"
            )
        await self._check_joint_targets(targets)
        await asyncio.to_thread(handle.set_joint_targets, targets, None)

        await self._settle_or_raise(
            handle,
            targets,
            self._deadline_s(timeout),
            _TOLERANCE_RAD,
            detail_prefix=f"arm {self.name}",
        )

    def _max_vel_rad_s(self, options: MoveOptions | None) -> float | None:
        """The per-joint max_vel_degs_per_sec_joints, when non-empty, is the
        ONLY velocity limit honoured (viam.md: the scalar is ignored in that
        case); the handle only takes one scalar, so the min across joints is
        used. Otherwise MoveOptions.max_vel_degs_per_sec applies when set.
        None = the drive's own limit. Acceleration fields and max_tcp_speed
        follow the same per-joint-wins precedence but aren't honoured at
        all - logged once (ARM-13)."""
        if options is None:
            return None

        has_acceleration_limit = len(options.max_acc_degs_per_sec2_joints) > 0 or (
            options.HasField("max_acc_degs_per_sec2")
        )
        if not self._has_warned_options and (
            has_acceleration_limit or options.HasField("max_tcp_speed")
        ):
            self.logger.info(
                "arm %s: MoveOptions acceleration limits and max_tcp_speed are not honoured",
                self.name,
            )
            self._has_warned_options = True

        if len(options.max_vel_degs_per_sec_joints) > 0:
            return math.radians(min(options.max_vel_degs_per_sec_joints))
        if options.HasField("max_vel_degs_per_sec"):
            return math.radians(options.max_vel_degs_per_sec)
        return None

    async def move_through_joint_positions(
        self,
        positions: Sequence[JointPositions],
        options: MoveOptions | None = None,
        *,
        extra: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """Execute a trajectory - this is what the motion service calls to run
        its planned paths. Intermediate waypoints use a loose tolerance so the
        arm flows through them and a short deadline that warns and continues
        on timeout (an obstacle blocking the path raises, since it won't
        clear itself); the final waypoint settles tight against the move
        deadline."""
        handle = self._h()
        waypoints = list(positions)
        if not waypoints:
            return
        max_vel_rad_s = self._max_vel_rad_s(options)
        move_deadline_s = self._deadline_s(timeout)
        current = await asyncio.to_thread(handle.get_joint_positions)
        path: list[list[float]] = []
        for wp in waypoints:
            targets = [math.radians(v) for v in wp.values]
            if len(current) != len(targets):
                raise JointTargetOutOfLimitsError(
                    f"arm {self.name}: expected {len(current)} joint values, got {len(targets)}"
                )
            await self._check_joint_targets(targets)
            path.append(targets)
        # one continuous path, settled once at the end (see ArmHandle.follow_joint_path)
        await asyncio.to_thread(handle.follow_joint_path, path, max_vel_rad_s)
        outcome = await asyncio.to_thread(handle.wait_for_settle, move_deadline_s, _TOLERANCE_RAD)
        if outcome is SettleOutcome.REACHED:
            return
        progress = await asyncio.to_thread(handle.path_progress)
        where = (
            f"at segment {progress[0] + 1}/{progress[1]}" if progress else "at the final waypoint"
        )
        current = await asyncio.to_thread(handle.get_joint_positions)
        detail = _stuck_joint_detail(current, path[-1], _TOLERANCE_RAD)
        # hold here rather than keep pushing at the unreachable target
        await asyncio.to_thread(handle.stop)
        if outcome is SettleOutcome.STALLED:
            raise ArmMoveStalledError(
                f"arm {self.name} stalled {where} of {len(waypoints)} waypoints "
                f"(stuck joints: {detail})"
            )
        raise ArmMoveTimeoutError(
            f"arm {self.name} did not reach the final waypoint within {move_deadline_s:.1f}s "
            f"({where}; stuck joints: {detail})"
        )

    async def get_joint_positions(self, **kwargs) -> JointPositions:
        radians = await asyncio.to_thread(self._h().get_joint_positions)
        return JointPositions(values=[math.degrees(r) for r in radians])

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    def _kinematics_url(self) -> str | None:
        url = self._attrs.get("kinematics_url")
        if url:
            return str(url)
        asset = self._attrs.get("asset")
        if asset and asset in KNOWN_ASSETS:
            return KNOWN_ASSETS[asset].get("kinematics")
        return None

    def _load_kinematics(self) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        url = self._kinematics_url()
        if not url:
            raise NotImplementedError(
                f"no kinematics file known for arm {self.name}; set the "
                '"kinematics_url" attribute (SVA .json or .urdf)'
            )
        ext = os.path.splitext(url)[1].lower()
        fmt = (
            KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF
            if ext in (".urdf", ".xml", ".xacro")
            else KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
        )

        cache_dir = os.environ.get("VIAM_MODULE_DATA") or tempfile.gettempdir()
        cache = os.path.join(
            cache_dir,
            f"kinematics-{hashlib.sha1(url.encode()).hexdigest()[:12]}{ext}",
        )
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return fmt, f.read()

        self.logger.info("fetching kinematics for %s from %s", self.name, url)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = cache + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, cache)
        except OSError:
            pass  # caching is best-effort
        return fmt, data

    async def get_kinematics(self, **kwargs) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            self._kinematics = await asyncio.to_thread(self._load_kinematics)
        return self._kinematics

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        # Deliberately empty: rdk builds arm geometry from GetKinematics (the
        # SVA already carries the link capsules) and never calls Geometries
        # for arms.
        return []

    # Abstract on viam-sdk "main" but absent at the 0.80.0 floor pinned in
    # requirements.txt; implementing it keeps IsaacArm instantiable across
    # that bound.
    async def get_3d_models(self, **kwargs) -> Mapping[str, Any]:
        return {}

    def _default_ee_prim_path(self) -> str:
        """The EE prim GetEndPosition/prim_world_pose fall back to when no
        explicit prim_path is given, matching the normalisation SimManager
        uses to spawn/mock the arm's prim."""
        prim_path = self._attrs.get("prim_path") or f"/World/{_prim_name(self.name)}"
        return f"{prim_path}/wrist_3_link"

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        if cmd == "get_joint_positions_radians":
            return {"values": list(await asyncio.to_thread(self._h().get_joint_positions))}
        if cmd == "dof_names":
            names: list[ValueTypes] = list(await asyncio.to_thread(self._h().dof_names))
            return {"dof_names": names}
        if cmd == "all_dof_names":
            all_names: list[ValueTypes] = list(await asyncio.to_thread(self._h().all_dof_names))
            return {"dof_names": all_names}
        if cmd == "joint_state":
            state = await asyncio.to_thread(self._h().joint_state)
            joints: list[ValueTypes] = [
                {
                    "name": entry["name"],
                    "named": entry["named"],
                    "position_deg": math.degrees(entry["position"]),
                    "velocity_deg_s": math.degrees(entry["velocity"]),
                    "target_deg": (
                        None if entry["target"] is None else math.degrees(entry["target"])
                    ),
                }
                for entry in state
            ]
            return {"joints": joints}
        if cmd == "path_trace":
            trace: list[ValueTypes] = list(await asyncio.to_thread(self._h().path_trace))
            return {"path_trace": trace}
        if cmd == "prim_world_pose":
            prim_path = command.get("prim_path") or self._default_ee_prim_path()
            if not isinstance(prim_path, str):
                raise ValueError(f"prim_path must be a string, got {prim_path!r}")
            prim_path = prim_path.strip()
            (x, y, z), quat = await asyncio.to_thread(self._h().get_prim_world_pose, prim_path)
            ox, oy, oz, theta = quat_to_ov(quat)
            return {
                "prim_path": prim_path,
                "position_mm": [x * 1000.0, y * 1000.0, z * 1000.0],
                "quaternion_wxyz": list(quat),
                "orientation_vector": {
                    "o_x": ox,
                    "o_y": oy,
                    "o_z": oz,
                    "theta_deg": math.degrees(theta),
                },
            }
        raise ValueError(f"unknown command: {command}")
