"""viam:isaac-sim-devin:gripper - a simulated parallel-jaw gripper riding an arm.

Attributes:
  world (string, required)        - name of the viam:isaac-sim-devin:world component
  arm (string, required)          - name of the viam:isaac-sim-devin:arm it is bolted to
  asset (string)                  - known gripper asset, default "robotiq_2f_85"
  parent_prim (string)            - link it is bolted to, default <arm prim>/wrist_3_link
  local_position ([x,y,z] m)      - mount pose of the gripper's base_link on parent_prim
  local_orientation_rpy_deg       - (defaults: identity - the 2F-85 base sits on the flange)
  tcp_offset_m (float)            - flange -> tool centre point along tool +Z, default 0.134
                                    = the fingertip pad centre measured on the GPU (OQ-7; the
                                    pads span 115-153 mm, W15's paper value 115 was their near edge)
  open_deg / closed_deg (float)   - drive-joint angles for open / fully closed; defaults 0
                                    and the Isaac-release value from compat.caps()
                                    (47 on 5.0, 45 on 4.5 - R-9)
  grab_timeout_sec (float)        - how long grab() waits for a stall or full closure, default 5
  holding_tolerance_deg (float)   - commanded-vs-measured gap that counts as holding, default 2
  holding_effort_min_nm (float)   - Isaac only: measured drive effort (N m) at which the jaw counts
                                    as holding; unset = the stall predicate
  mock_object_width_m (float)     - mock only: width of the object between the jaws
                                    (unset = nothing to grab, so grab() returns False)

Frame - the gripper's frame is its TCP, so the motion service plans the TCP
(not the flange) onto the block (W15; DEC-12):

    "frame": {"parent": "<arm>", "translation": {"x": 0, "y": 0, "z": <tcp_offset_m * 1000>}}

Unlike a mounted camera, the frame does NOT place the prim: base_link bolts to
parent_prim at local_position / local_orientation_rpy_deg, and the frame's
translation is the TCP the planner uses. validate_config requires
frame.parent == arm.

API mapping (viam-sdk 0.80.0 Gripper, all eight abstract methods - ARM-5):
  open / stop / is_moving          -> the handle
  grab() -> bool                   -> close, wait <= grab_timeout_sec for stall-or-closed,
                                      return is_holding()
  is_holding_something()           -> HoldingStatus(is_holding, meta={jaw angles, degrees})
  get_current_inputs() / go_to_inputs([v])  -> one value in [0, 1]: 0 = open, 1 = closed (DEC-12)
  get_kinematics()                 -> 1-link / 0-joint SVA whose link is the 36 x 146 x 153 mm
                                      gripper box spanning flange to fingertips, centred 57.5 mm
                                      behind the TCP (ARM-6; R-6: a gripper whose Kinematics
                                      fails is silently dropped from the frame system)
  get_geometries()                 -> that same single box (rdk keeps only [0])
  close()                          -> SimManager.release_handle (XC-4)
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from grpclib import Status
from typing_extensions import Self
from viam.components.gripper import Gripper
from viam.errors import ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    Geometry,
    KinematicsFileFormat,
    Pose,
    RectangularPrism,
    ResourceName,
    Vector3,
)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE
from ..sim_manager import KNOWN_ASSETS, GripperHandle, SimManager, _prim_name
from ..spatial import quat_rotate
from .utils import get_attrs

DEFAULT_GRIPPER_ASSET = "robotiq_2f_85"
DEFAULT_TCP_OFFSET_M = float(KNOWN_ASSETS[DEFAULT_GRIPPER_ASSET]["tcp_offset_m"])
DEFAULT_GRAB_TIMEOUT_S = 5.0
GRAB_POLL_INTERVAL_S = 1.0 / 120.0  # matches MockArmHandle.STEP_S; the sim's "physics step"
JAW_CLOSED_TOLERANCE_RAD = 1e-3
GRAB_REGRIPS = 3  # extra attempts when the jaw stalls short of closed without holding: nudge, back off + close, nudge
GRAB_REGRIP_BACKOFF_RAD = math.radians(8.0)
GRAB_NUDGE_RAD = math.radians(5.0)
GRAB_JAM_STILL_S = 0.25  # still and short of closed this long without holding = jammed
JAW_BOX_MM: tuple[float, float, float] = KNOWN_ASSETS[DEFAULT_GRIPPER_ASSET]["jaw_box_mm"]
# The box spans flange -> fingertips; in the gripper frame (origin = TCP) its
# centre sits reach/2 - tcp behind the TCP (measured: 76.5 - 134 = -57.5 mm).
GRIPPER_BOX_CENTRE_Z_MM = (
    KNOWN_ASSETS[DEFAULT_GRIPPER_ASSET]["fingertip_reach_m"] / 2.0 - DEFAULT_TCP_OFFSET_M
) * 1000.0


def default_parent_prim(arm_name: str) -> str:
    """The flange link of the arm's prim, matching SimManager's prim naming."""
    return f"/World/{_prim_name(arm_name)}/wrist_3_link"


def _gripper_sva(
    link_id: str, box_mm: tuple[float, float, float], box_centre_z_mm: float
) -> dict[str, Any]:
    """The gripper's kinematics: one link, no joints, whose geometry is the
    box_mm RectangularPrism spanning flange to fingertips - centred
    box_centre_z_mm along the tool axis from the TCP, the frame origin (ARM-6)."""
    box_x_mm, box_y_mm, box_z_mm = box_mm
    return {
        "name": link_id,
        "kinematic_param_type": "SVA",
        "links": [
            {
                "id": link_id,
                "parent": "world",
                "translation": {"x": 0, "y": 0, "z": 0},
                "orientation": {
                    "type": "ov_degrees",
                    "value": {"x": 0, "y": 0, "z": 1, "th": 0},
                },
                "geometry": {
                    "x": box_x_mm,
                    "y": box_y_mm,
                    "z": box_z_mm,
                    "translation": {"x": 0, "y": 0, "z": box_centre_z_mm},
                },
            }
        ],
        "joints": [],
    }


class IsaacGripper(Gripper, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "gripper")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: GripperHandle | None = None
        self._attrs: dict[str, Any] = {}
        self._grab_timeout = DEFAULT_GRAB_TIMEOUT_S
        self._tcp_offset_m = DEFAULT_TCP_OFFSET_M
        self._kinematics: tuple[KinematicsFileFormat.ValueType, bytes] | None = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        gripper = cls(config.name)
        gripper.reconfigure(config, dependencies)
        return gripper

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        """Requires world + arm; when a frame is set its parent must be the
        arm (the frame is the TCP in the arm's tool frame). Returns both as
        dependencies so viam-server builds the arm before the gripper."""
        attrs = get_attrs(config)
        world = attrs.get("world")
        if not world or not isinstance(world, str):
            raise ValueError(
                f'{config.name}: set the "world" attribute to the name of your '
                f"{NAMESPACE}:{FAMILY}:world component"
            )
        arm = attrs.get("arm")
        if not arm or not isinstance(arm, str):
            raise ValueError(
                f'{config.name}: set the "arm" attribute to the name of the '
                f"{NAMESPACE}:{FAMILY}:arm component this gripper is attached to"
            )
        if config.HasField("frame") and config.frame.parent.split(":")[0] != arm:
            raise ValueError(
                f"{config.name}: frame.parent must be the arm {arm!r} (the frame is the "
                f"gripper's TCP in the arm's tool frame), got {config.frame.parent!r}"
            )
        return [world, arm], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = get_attrs(config)
        attrs.setdefault("asset", DEFAULT_GRIPPER_ASSET)
        attrs.setdefault("parent_prim", default_parent_prim(str(attrs["arm"])))
        self._grab_timeout = float(attrs.get("grab_timeout_sec", DEFAULT_GRAB_TIMEOUT_S))
        self._tcp_offset_m = float(attrs.get("tcp_offset_m", DEFAULT_TCP_OFFSET_M))
        self._attrs = attrs
        self._kinematics = None
        self._handle = SimManager.get().create_gripper(self.name, attrs)

    async def close(self) -> None:
        """XC-4: release the handle (hooks, callbacks); the prim stays attached."""
        SimManager.get().release_handle(self.name)
        self._handle = None

    def _h(self) -> GripperHandle:
        if self._handle is None:
            raise RuntimeError(f"gripper {self.name} is not attached to the sim")
        return self._handle

    # -- the eight abstract methods (viam.md Q5) --------------------------

    async def open(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().open)

    async def grab(self, **kwargs) -> bool:
        """Close, and if the jaw stalls short of closed without holding (the
        2F-85 linkage jams in a low-force state on about half of otherwise
        identical closes: jaw 13.2-13.5 deg at 0.017 N m against 14.7 deg at
        2.4+ when it bites, measured 2026-09-04), back the jaw off a few
        degrees and close again, up to GRAB_REGRIPS times, all within
        grab_timeout_sec. A real gripper controller regrips the same way."""
        handle = self._h()
        deadline = time.monotonic() + self._grab_timeout
        _, closed_rad = await asyncio.to_thread(handle.jaw_limits)
        for attempt in range(GRAB_REGRIPS + 1):
            if attempt > 0:
                jaw = await asyncio.to_thread(handle.get_jaw)
                effort = await asyncio.to_thread(handle.finger_effort)
                self.logger.info(
                    "gripper %s: jaw stalled at %.1f deg without holding (effort %s); regrip %d/%d",
                    self.name, math.degrees(jaw), None if effort is None else round(effort, 3),
                    attempt, GRAB_REGRIPS,
                )
                if attempt % 2 == 1:
                    # nudge: a target a few degrees past the jam breaks it where a
                    # full close did not (an agent proved it: 11 deg jammed at no
                    # effort, commanded 16 deg, stalled at 15.9 deg with 5.8 N m)
                    await asyncio.to_thread(handle.set_jaw, jaw + GRAB_NUDGE_RAD)
                    await self._wait_still(handle, deadline)
                    if await asyncio.to_thread(handle.is_holding):
                        return True
                    continue
                await asyncio.to_thread(handle.set_jaw, jaw - GRAB_REGRIP_BACKOFF_RAD)
                await self._wait_still(handle, deadline)
            await asyncio.to_thread(handle.close)
            await self._wait_still(handle, deadline)
            still_since: float | None = None
            while time.monotonic() < deadline:
                if await asyncio.to_thread(handle.is_holding):
                    return True
                jaw = await asyncio.to_thread(handle.get_jaw)
                if abs(jaw - closed_rad) <= JAW_CLOSED_TOLERANCE_RAD:
                    return await asyncio.to_thread(handle.is_holding)  # fully closed: nothing there
                if await asyncio.to_thread(handle.is_moving):
                    still_since = None
                else:
                    still_since = still_since or time.monotonic()
                    if time.monotonic() - still_since >= GRAB_JAM_STILL_S:
                        break  # still, short of closed, not holding: jammed, regrip
                await asyncio.sleep(GRAB_POLL_INTERVAL_S)
            if time.monotonic() >= deadline:
                break
        return await asyncio.to_thread(handle.is_holding)

    async def _wait_still(self, handle: GripperHandle, deadline: float) -> None:
        while time.monotonic() < deadline and await asyncio.to_thread(handle.is_moving):
            await asyncio.sleep(GRAB_POLL_INTERVAL_S)

    async def is_holding_something(self, **kwargs) -> Gripper.HoldingStatus:
        handle = self._h()
        open_rad, closed_rad = await asyncio.to_thread(handle.jaw_limits)
        jaw_rad = await asyncio.to_thread(handle.get_jaw)
        is_holding = await asyncio.to_thread(handle.is_holding)
        effort = await asyncio.to_thread(handle.finger_effort)
        span = closed_rad - open_rad
        input_value = (jaw_rad - open_rad) / span if span else 0.0
        meta: dict[str, ValueTypes] = {
            "jaw_deg": math.degrees(jaw_rad),
            "open_deg": math.degrees(open_rad),
            "closed_deg": math.degrees(closed_rad),
            "input": min(max(input_value, 0.0), 1.0),
        }
        if effort is not None:
            meta["finger_effort_nm"] = effort
        return Gripper.HoldingStatus(is_holding_something=is_holding, meta=meta)

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_kinematics(self, **kwargs) -> tuple[KinematicsFileFormat.ValueType, bytes]:
        if self._kinematics is None:
            sva = _gripper_sva(self.name, JAW_BOX_MM, GRIPPER_BOX_CENTRE_Z_MM)
            self._kinematics = (
                KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA,
                json.dumps(sva).encode(),
            )
        return self._kinematics

    async def get_current_inputs(self, **kwargs) -> list[float]:
        handle = self._h()
        open_rad, closed_rad = await asyncio.to_thread(handle.jaw_limits)
        jaw_rad = await asyncio.to_thread(handle.get_jaw)
        span = closed_rad - open_rad
        value = (jaw_rad - open_rad) / span if span else 0.0
        return [min(max(value, 0.0), 1.0)]

    async def go_to_inputs(self, values: list[float], **kwargs) -> None:
        if len(values) != 1:
            raise ViamGRPCError(
                f"gripper {self.name}: go_to_inputs expects exactly one value in [0, 1], "
                f"got {len(values)}",
                Status.INVALID_ARGUMENT,
            )
        value = values[0]
        if not 0.0 <= value <= 1.0:
            raise ViamGRPCError(
                f"gripper {self.name}: go_to_inputs value must be in [0, 1], got {value}",
                Status.INVALID_ARGUMENT,
            )

        handle = self._h()
        open_rad, closed_rad = await asyncio.to_thread(handle.jaw_limits)
        await asyncio.to_thread(handle.set_jaw, open_rad + value * (closed_rad - open_rad))

        deadline = time.monotonic() + self._grab_timeout
        while time.monotonic() < deadline and await asyncio.to_thread(handle.is_moving):
            await asyncio.sleep(GRAB_POLL_INTERVAL_S)

    # -- non-abstract, overridden on purpose -------------------------------

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return [
            Geometry(
                center=Pose(x=0, y=0, z=GRIPPER_BOX_CENTRE_Z_MM, o_x=0, o_y=0, o_z=1, theta=0),
                box=RectangularPrism(
                    dims_mm=Vector3(x=JAW_BOX_MM[0], y=JAW_BOX_MM[1], z=JAW_BOX_MM[2])
                ),
                label=self.name,
            )
        ]

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        if cmd == "dof_names":
            names: list[ValueTypes] = list(self._h().dof_names())
            return {"dof_names": names}
        if cmd == "jaw_deg":
            open_rad, closed_rad = self._h().jaw_limits()
            return {
                "jaw_deg": math.degrees(self._h().get_jaw()),
                "open_deg": math.degrees(open_rad),
                "closed_deg": math.degrees(closed_rad),
            }
        if cmd == "tcp_pose":
            return await asyncio.to_thread(self._tcp_pose)
        if cmd == "contacts":
            contacts: list[ValueTypes] = list(await asyncio.to_thread(self._h().contacts))
            return {"contacts": contacts}
        if cmd == "collision_shapes":
            prim = command.get("prim")
            shapes: list[ValueTypes] = list(
                await asyncio.to_thread(self._h().collision_shapes, str(prim) if prim else None)
            )
            return {"collision_shapes": shapes}
        raise ValueError(f"unknown command: {command}")

    def _tcp_pose(self) -> dict[str, ValueTypes]:
        """GPU checklist item 4 / OQ-7: the fingertip midpoint's offset from
        the mount link along the tool +Z, in mm, next to the configured
        tcp_offset_m - so the TCP is corrected in one place if they differ."""
        poses = self._h().link_world_poses()
        out: dict[str, ValueTypes] = {"jaw_deg": math.degrees(self._h().get_jaw())}
        out |= {
            key: {
                "position_mm": [v * 1000.0 for v in pos],
                "quaternion_wxyz": list(quat),
            }
            for key, (pos, quat) in poses.items()
        }
        parent = poses.get("parent")
        if parent is None:
            out["error"] = "mount link pose unavailable"
            return out
        tool_axis = quat_rotate(parent[1], (0.0, 0.0, 1.0))

        def along_tool(point: tuple[float, ...]) -> float:
            return sum((q - p) * a for q, p, a in zip(point, parent[0], tool_axis, strict=True))

        left = poses.get("left_inner_finger")
        right = poses.get("right_inner_finger")
        if left is not None and right is not None:
            origin_mid = tuple((a + b) / 2.0 for a, b in zip(left[0], right[0], strict=True))
            # informational: this asset authors link frames at the base
            out["inner_finger_origin_offset_mm"] = along_tool(origin_mid) * 1000.0

        bounds = self._h().fingertip_world_bounds()
        out["fingertips"] = {
            side: {
                "min_mm": [v * 1000.0 for v in low],
                "max_mm": [v * 1000.0 for v in high],
                "center_mm": [(a + b) * 500.0 for a, b in zip(low, high, strict=True)],
            }
            for side, (low, high) in bounds.items()
        }
        if "left" not in bounds or "right" not in bounds:
            out["error"] = "fingertip pad meshes not found under the gripper"
            return out
        centers = [
            tuple((a + b) / 2.0 for a, b in zip(low, high, strict=True))
            for low, high in (bounds["left"], bounds["right"])
        ]
        pad_mid = tuple((a + b) / 2.0 for a, b in zip(centers[0], centers[1], strict=True))
        corners = [
            corner
            for low, high in bounds.values()
            for corner in (
                (x, y, z)
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            )
        ]
        measured = along_tool(pad_mid)
        out["pad_center_midpoint_mm"] = [v * 1000.0 for v in pad_mid]
        out["fingertip_reach_mm"] = max(along_tool(c) for c in corners) * 1000.0
        out["jaw_gap_mm"] = (
            math.dist(centers[0], centers[1]) * 1000.0
        )  # pad-centre to pad-centre, across the jaw
        out["measured_tcp_offset_mm"] = measured * 1000.0
        out["configured_tcp_offset_mm"] = self._tcp_offset_m * 1000.0
        out["delta_mm"] = (measured - self._tcp_offset_m) * 1000.0
        return out
