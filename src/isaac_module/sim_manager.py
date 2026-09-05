"""The singleton that owns Isaac Sim.

Isaac Sim (Omniverse Kit) wants to be created and stepped from a single
thread, so the module runs it on the process main thread (see main.py) and
everything else - the Viam module server, component handlers - submits work
to that thread through a queue. Handles returned by create_arm/create_camera/
create_base wrap that queue so component models can stay simple.

A "mock" backend (world attribute: {"mock": true}) implements the same
handle interfaces with plain python so the module can run and be tested on
machines without Isaac Sim installed.
"""

import concurrent.futures
import math
import queue
import random
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple, Optional

import numpy as np
from viam.logging import getLogger

from . import FAMILY, NAMESPACE
from .camera_base import CameraHandle, Frame, NoFrameYetError
from .compat import caps, import_isaac, isaac_version
from .encoding import Intrinsics
from .errors import CameraInitError, PrimNotFoundError, SimNotBootedError, SimTimeoutError
from .mock_camera import MockCameraHandle
from .physics import ARM_SOLVER_POSITION_ITERATIONS, apply_prop_physics
from .spatial import (
    Quat,
    Vec3,
    look_at_quat,
    quat_conj,
    quat_from_euler_deg,
    quat_mul,
    quat_rotate,
    to_vec3,
)

LOGGER = getLogger("viam-isaac-sim")

# Assets shipped on the Isaac Sim nucleus/content server, addressable by a
# short name in component config. Paths are relative to the assets root;
# where isaac 5.0 moved an asset, the 5.0 path is listed first with the 4.x
# path as a fallback - the first candidate that exists is used.
_UR_KINEMATICS = (
    "https://raw.githubusercontent.com/viam-modules/universal-robots/main/src/kinematics"
)

# the 6 UR joints in SVA (spatial vector algebra) order - the order the arm
# component's kinematics/motion planning expects, which need not match the
# articulation's PhysX dof order (FINDINGS ARM-1; R-3).
UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

KNOWN_ASSETS: dict[str, dict[str, Any]] = {
    "ur3e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur3e.json",
        "joint_names": UR_JOINT_NAMES,
    },
    "ur5e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur5e.json",
        # verified (FINDINGS XC-1/ARM-10, W7/W9): the usd asset's own base
        # link is rotated 180deg about Z relative to the kinematics frame.
        "base_frame_correction": (0.0, 0.0, 0.0, 1.0),
        "joint_names": UR_JOINT_NAMES,
    },
    "ur10": {"usd": ["/Isaac/Robots/UniversalRobots/ur10/ur10.usd"], "joint_names": UR_JOINT_NAMES},
    "ur10e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"],
        "joint_names": UR_JOINT_NAMES,
    },
    "ur16e": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur16e/ur16e.usd"],
        "joint_names": UR_JOINT_NAMES,
    },
    # ur3e/ur10*/ur16e correction is unchecked; deliberately no entry
    # (identity) until verified.
    "ur20": {
        "usd": ["/Isaac/Robots/UniversalRobots/ur20/ur20.usd"],
        "kinematics": f"{_UR_KINEMATICS}/ur20.json",
        "base_frame_correction": (0.0, 0.0, 0.0, 1.0),
        "joint_names": UR_JOINT_NAMES,
    },
    "franka": {
        "usd": [
            "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            "/Isaac/Robots/Franka/franka.usd",
        ]
    },
    # FINDINGS W12-W15 / DEC-2: a gripper asset, only ever referenced UNDER an
    # arm prim by create_gripper (never spawned free-standing). closed_deg is
    # per Isaac release - compat.caps().gripper_closed_deg (R-9). Never use the
    # ur5e.usd "Gripper" variant; do not hard-code sibling filenames beyond
    # Robotiq_2F_85_edit.usd (ARM-2).
    "robotiq_2f_85": {
        "kind": "gripper",
        "usd": ["/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"],
        "drive_joint": "finger_joint",
        # W13's paper value; at attach the drive joint's authored lower limit
        # wins when readable (~7.8 deg on the 5.0 asset)
        "open_deg": 0.0,
        # flange -> TCP along tool +Z: the fingertip PAD CENTRE, measured on the GPU
        # (OQ-7, 2026-08-28): pads span 115-153 mm, centre 134; W15's paper value was 115
        "tcp_offset_m": 0.134,
        # the single box GetGeometries / the SVA carry, spanning flange -> fingertips.
        # Measured on the GPU (OQ-7): pads reach 0.153 m; W15's 150 "centred on
        # the TCP" put 75 mm of virtual gripper below the fingertips.
        "jaw_box_mm": (36.0, 146.0, 153.0),
        "fingertip_reach_m": 0.153,
    },
    "jetbot": {
        "usd": [
            "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
            "/Isaac/Robots/Jetbot/jetbot.usd",
        ],
        "wheel_joints": ["left_wheel_joint", "right_wheel_joint"],
        "wheel_radius": 0.03,
        "wheel_base": 0.1125,
    },
}


@dataclass
class SimConfig:
    mock: bool = False
    headless: bool = True
    livestream: bool = True
    usd_stage: str | None = None
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    boot_timeout: float = 300.0
    # IP the livestream advertises to clients; auto-detected if empty
    livestream_public_ip: str = ""
    # props to spawn into the scene at boot; each entry:
    #   {"type": "cube"|"usd", "name": ..., "position": [x,y,z] (m),
    #    "size": edge_m, "scale": [sx,sy,sz], "color": [r,g,b] 0-1,
    #    "fixed": bool, "usd_path": ...}
    props: list[dict[str, Any]] = field(default_factory=list)
    # kit console verbosity (verbose/info/warning/error). Kit prints thousands
    # of lines at info, and viam-server records the module's stderr as
    # error-level logs, so default to warning.
    kit_log_level: str = "warning"
    # scene lighting (FINDINGS SCN-9 / W30). None = leave the stage's lights
    # alone. Shape: {"dome": {"intensity": 1000, "color": [1, 1, 1]},
    # "sphere_intensity": 30000}; Isaac creates a DomeLight prim and rescales
    # /World/SphereLight, the mock records the config for tests.
    lighting: dict[str, Any] | None = None
    # render-cost levers (CAM-12/FINDINGS R-11). None = leave the renderer's
    # defaults alone. Shape: {"motion_bvh": bool, "disable_viewport_updates":
    # bool}. Isaac applies both via carb settings, best-effort; the mock
    # records the config for tests.
    render: dict[str, Any] | None = None


# FINDINGS W30 lighting defaults: the DomeLight the module adds, and the prim
# paths in default_environment.usd it adjusts.
# a camera frequency must divide the render rate (IS-3); slack for float error
FREQUENCY_DIVISOR_TOLERANCE = 1e-6
DEFAULT_DOME_INTENSITY = 1000.0
DEFAULT_DOME_COLOR = (1.0, 1.0, 1.0)
DOME_LIGHT_PRIM_PATH = "/World/DomeLight"
SPHERE_LIGHT_PRIM_PATH = "/World/SphereLight"


def _as_quat(values: Sequence[float]) -> Quat:
    """Four numbers -> a (w, x, y, z) quaternion tuple (validates arity)."""
    w, x, y, z = (float(v) for v in values)
    return (w, x, y, z)


# create_camera attrs contract defaults (camera_base.py module docstring,
# FINDINGS W18/W19).
# A freshly booted renderer creates a render product's SDG pipeline nodes
# only after render ticks, so Camera.initialize()'s immediate node lookup can
# die with KeyError('/Render/PostProcess/SDGPipeline/..._LdrColorSDhostPtr')
# - observed on the first cold boot of a fresh install (empty shader cache,
# every component building at once). 5 attempts x 30 ticks is ~2 s of
# stepping at 60 Hz, well inside create_camera's 120 s budget even with
# shader compilation on top.
CAMERA_INIT_ATTEMPTS = 5
CAMERA_INIT_RENDER_TICKS = 30

DEFAULT_CAMERA_WIDTH = 848
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FOV_DEG = 90.5
DEFAULT_CLIP_NEAR_M = 0.05
DEFAULT_CLIP_FAR_M = 10.0


def _camera_prim_path(name: str, attrs: dict[str, Any]) -> str:
    """Prim path for a to-be-created camera: parented under ``parent_prim``,
    else an explicit ``prim_path``, else ``/World/<name>``."""
    parent = attrs.get("parent_prim")
    if parent:
        return f"{parent.rstrip('/')}/{_prim_name(name)}"
    return attrs.get("prim_path") or f"/World/{_prim_name(name)}"


def _place_camera(cam: Any, attrs: dict[str, Any]) -> None:
    """Pose a just-initialized camera per the create_camera attrs contract
    (camera_base.py module docstring). ``parent_prim`` rides a (possibly
    moving) link; ``local_orientation_wxyz`` - derived from the Viam frame -
    is the source of truth (CAM-10) and is applied in ROS-optical axes so the
    camera's +Z is the frame's forward axis. Absent that, the legacy
    ``local_orientation_rpy_deg`` pose (usd axes, 180 deg about X to flip the
    usd camera's -Z forward) still applies. Free-standing ``orientation_wxyz``
    is world axes (+X forward) unless ``orientation_axes`` says "ros" - the
    camera model sets that when the quat came from a Viam frame, whose
    convention is ROS-optical (+Z forward), so a frame-configured fixed
    camera aims where the frame system believes it aims (GPU phase-4 run 1:
    the side camera's world-axes read of a ROS quat measured the backdrop
    at 7994 mm)."""
    parent = attrs.get("parent_prim")
    if parent:
        local_position = list(to_vec3(attrs.get("local_position"), default=(0.0, 0.0, 0.05)))
        if attrs.get("local_orientation_wxyz") is not None:
            quat = list(_as_quat(attrs["local_orientation_wxyz"]))
            cam.set_local_pose(local_position, quat, camera_axes="ros")
        else:
            r, p, y = to_vec3(attrs.get("local_orientation_rpy_deg"), default=(180.0, 0.0, 0.0))
            quat = list(quat_from_euler_deg(r, p, y))
            cam.set_local_pose(local_position, quat, camera_axes="usd")
    elif attrs.get("target") is not None:
        # aim at a target point (world axes: +X forward, +Z up)
        position = to_vec3(attrs.get("position"), default=(3.0, 3.0, 2.5))
        world_quat = look_at_quat(position, to_vec3(attrs.get("target")))
        cam.set_world_pose(list(position), list(world_quat), camera_axes="world")
    elif attrs.get("orientation_wxyz") is not None:
        position = to_vec3(attrs.get("position"))
        world_quat = _as_quat(attrs["orientation_wxyz"])
        camera_axes = str(attrs.get("orientation_axes", "world"))
        cam.set_world_pose(list(position), list(world_quat), camera_axes=camera_axes)


def _configure_camera_optics(
    cam: Any, attrs: dict[str, Any], rendering_dt: float = 1.0 / 60.0
) -> None:
    """Focal length from ``fov_deg`` (CAM-4's aperture cancels usd's unit
    convention so this is unit-safe on both Isaac versions), a matching
    vertical aperture so pixels stay square (CAM-4), the clipping range
    (CAM-3 - OpenUSD's unauthored default is a 1 m near clip), the depth
    annotator (CAM-1) and an optional capture rate."""
    width, height = cam.get_resolution()

    # newly created cameras default to a 90.5 degree horizontal FOV; a
    # camera bound to an existing prim (explicit prim_path) keeps that
    # prim's authored FOV unless fov_deg overrides it.
    if not attrs.get("prim_path") or attrs.get("fov_deg"):
        fov = float(attrs.get("fov_deg", DEFAULT_CAMERA_FOV_DEG))
        horizontal_aperture = cam.get_horizontal_aperture()
        cam.set_focal_length(horizontal_aperture / (2.0 * math.tan(math.radians(fov) / 2.0)))

    horizontal_aperture = cam.get_horizontal_aperture()
    cam.set_vertical_aperture(horizontal_aperture * height / width)

    clip_near = float(attrs.get("clip_near", DEFAULT_CLIP_NEAR_M))
    clip_far = float(attrs.get("clip_far", DEFAULT_CLIP_FAR_M))
    cam.set_clipping_range(clip_near, clip_far)
    LOGGER.info(
        "camera %s clipping range %s",
        getattr(cam, "name", "<camera>"),
        cam.get_clipping_range(),
    )

    if attrs.get("depth"):
        cam.add_distance_to_image_plane_to_frame()

    frequency = attrs.get("frequency")
    if frequency:
        frequency = float(frequency)
        ticks_per_capture = (1.0 / rendering_dt) / frequency
        if abs(ticks_per_capture - round(ticks_per_capture)) > FREQUENCY_DIVISOR_TOLERANCE:
            LOGGER.warning(
                "camera %s frequency %s Hz is not an integer divisor of the "
                "render rate (%.4f Hz); the effective capture rate will differ",
                getattr(cam, "name", "<camera>"),
                frequency,
                1.0 / rendering_dt,
            )
        cam.set_frequency(frequency)


def spawn_orientation(attrs: dict[str, Any], meta: dict[str, Any]) -> Quat:
    """The (w,x,y,z) quaternion to spawn an arm's articulation with: the
    configured frame/orientation composed with the known asset's
    base_frame_correction (frame first), if any (FINDINGS XC-1/ARM-10)."""
    q_frame: Quat = (
        _as_quat(attrs["orientation_wxyz"])
        if attrs.get("orientation_wxyz") is not None
        else (1.0, 0.0, 0.0, 0.0)
    )
    correction = meta.get("base_frame_correction")
    if correction is not None:
        return quat_mul(q_frame, _as_quat(correction))
    return q_frame


def pose_in_frame(base_pos: Vec3, base_quat: Quat, pos: Vec3, quat: Quat) -> tuple[Vec3, Quat]:
    """Express a world pose (pos, quat) in the frame defined by
    (base_pos, base_quat) - both (w,x,y,z)."""
    base_quat_conj = quat_conj(base_quat)
    relative_position = quat_rotate(
        base_quat_conj,
        (pos[0] - base_pos[0], pos[1] - base_pos[1], pos[2] - base_pos[2]),
    )
    relative_orientation = quat_mul(base_quat_conj, quat)
    return relative_position, relative_orientation


def compose_pose(
    parent_pos: Vec3, parent_quat: Quat, local_pos: Vec3, local_quat: Quat
) -> tuple[Vec3, Quat]:
    """Inverse of pose_in_frame: express a pose (local_pos, local_quat) given
    in the frame (parent_pos, parent_quat) back in the parent's frame."""
    world_position = (
        parent_pos[0] + quat_rotate(parent_quat, local_pos)[0],
        parent_pos[1] + quat_rotate(parent_quat, local_pos)[1],
        parent_pos[2] + quat_rotate(parent_quat, local_pos)[2],
    )
    world_orientation = quat_mul(parent_quat, local_quat)
    return world_position, world_orientation


def viam_base_frame(root_pos: Vec3, root_quat: Quat, correction: Quat) -> tuple[Vec3, Quat]:
    """Recover Viam's arm frame from the Isaac articulation root's world
    pose: spawn composed root = frame * correction (FINDINGS ARM-10/XC-1),
    so frame = root * correction^-1."""
    return root_pos, quat_mul(root_quat, quat_conj(correction))


def anchor_fixed_joint_frame(
    spawn_pos: Vec3, spawn_quat: Quat, authored_pos: Vec3, authored_quat: Quat
) -> tuple[Vec3, Quat]:
    """Re-express a world-anchored fixed-base joint frame (authored_pos,
    authored_quat) so it matches an articulation spawned at (spawn_pos,
    spawn_quat). The UR assets' base FixedJoint has an empty body0 (= world
    frame) with localPos0/localRot0 authored in world coordinates, so PhysX
    resyncs the root xform to that joint frame on world.reset() and undoes
    any spawn pose passed to SingleArticulation (FINDINGS ARM-9/XC-1)."""
    return (
        (
            spawn_pos[0] + quat_rotate(spawn_quat, authored_pos)[0],
            spawn_pos[1] + quat_rotate(spawn_quat, authored_pos)[1],
            spawn_pos[2] + quat_rotate(spawn_quat, authored_pos)[2],
        ),
        quat_mul(spawn_quat, authored_quat),
    )


class SimManager:
    """Owns the sim thread. Get the process-wide instance via SimManager.get()."""

    _instance: Optional["SimManager"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "SimManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = SimManager()
            return cls._instance

    def __init__(self) -> None:
        self._tasks: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._boot_requested = threading.Event()
        self._booted = threading.Event()
        self._boot_error: BaseException | None = None
        self._stop = threading.Event()
        self._sim_thread_id: int | None = None

        self.cfg: SimConfig | None = None
        self.mock = False
        # Isaac objects are created at boot; typed Any because the isaacsim
        # modules are not importable (or type-checkable) outside Isaac Sim.
        self._sim_app: Any = None
        self.world: Any = None
        self._isaac: Any = None  # lazily-populated namespace of isaac imports
        self._step_callbacks: dict[str, Callable[[float], None]] = {}
        # scene lighting config from the world component (FINDINGS SCN-9/W30);
        # stored so status() and tests can read it even in mock mode.
        self.lighting: dict[str, Any] | None = None
        # render-cost levers config from the world component (CAM-12);
        # stored so status() and tests can read it even in mock mode.
        self.render: dict[str, Any] | None = None
        # hooks fired (in registration order) after every world reset -
        # XC-5, so component handles can re-anchor state that resets undo.
        # Each is (owner component name or None, hook); the owner lets
        # release_handle drop a closed component's hooks (XC-4).
        self._post_reset_hooks: list[tuple[str | None, Callable[[], None]]] = []
        self._post_reset_lock = threading.Lock()
        # the seam the world component's verbs drive the scene through
        # (SCN-16); created at boot (Isaac or Mock flavour)
        self._world_handle: WorldHandle | None = None
        # spawn spec per registered prop (sanitized name -> attrs), the
        # Isaac-side scene registry (SCN-5/SCN-11)
        self._prop_specs: dict[str, dict[str, Any]] = {}
        # component name -> (spawn attrs, handle). viam-server rebuilds
        # resources on config change, but prims can't be re-spawned without
        # restarting kit, so handles are cached per component name.
        self._handles: dict[str, tuple[dict[str, Any], Any]] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def ensure_booted(self, cfg: SimConfig) -> None:
        """Called by the world component's reconfigure. Boots the sim on the
        sim thread the first time; subsequent calls with a different config
        log that a module restart is required (Kit can't be re-created)."""
        if self._booted.is_set():
            if self.cfg != cfg:
                LOGGER.warning(
                    "isaac sim is already running; changes to world config "
                    "(stage/headless/etc) require restarting the module"
                )
            return
        if self._boot_error is not None:
            raise RuntimeError(f"isaac sim failed to boot previously: {self._boot_error}")

        self.cfg = cfg
        self._boot_requested.set()
        if not self._booted.wait(timeout=cfg.boot_timeout):
            raise SimTimeoutError(f"isaac sim did not boot within {cfg.boot_timeout}s")
        if self._boot_error is not None:
            raise RuntimeError(f"isaac sim failed to boot: {self._boot_error}")

    def request_stop(self) -> None:
        self._stop.set()

    def register_post_reset(self, fn: Callable[[], None], owner: str | None = None) -> None:
        """Register a hook that fires (in registration order) after every
        world reset - boot, a component spawn, or an explicit reset command -
        on whichever thread performs the reset (the sim thread in practice).
        ``owner`` is the component name the hook belongs to, so closing that
        component (XC-4) can drop it; None = lives for the process."""
        with self._post_reset_lock:
            self._post_reset_hooks.append((owner, fn))

    def unregister_post_reset(self, owner: str) -> None:
        """Drop every hook registered under ``owner`` (XC-4)."""
        with self._post_reset_lock:
            self._post_reset_hooks = [(o, fn) for o, fn in self._post_reset_hooks if o != owner]

    def _reset_world(self) -> None:
        """The single chokepoint for resetting the isaac world: resets it
        (skipped in mock mode) then runs every registered post-reset hook,
        isolating each hook's failures so one can't block the rest."""
        if not self.mock:
            self.world.reset()
        with self._post_reset_lock:
            hooks = [fn for _owner, fn in self._post_reset_hooks]
        for hook in hooks:
            try:
                hook()
            except Exception:
                LOGGER.exception("post-reset hook failed")

    def main_loop(self) -> None:
        """Run forever on the owning (main) thread: wait for a boot request,
        boot, then step the sim while draining queued tasks."""
        self._sim_thread_id = threading.get_ident()

        while not self._stop.is_set() and not self._boot_requested.wait(timeout=0.1):
            self._drain_tasks()

        if self._stop.is_set():
            return

        try:
            self._boot()
        except BaseException as e:  # SimulationApp failures can be SystemExit
            LOGGER.exception("failed to boot isaac sim")
            self._boot_error = e
            self._booted.set()
            return
        self._booted.set()

        last = time.monotonic()
        while not self._stop.is_set():
            self._drain_tasks()
            now = time.monotonic()
            dt = now - last
            last = now
            if self.mock:
                for cb in list(self._step_callbacks.values()):
                    cb(dt)
                time.sleep(0.01)
            else:
                self.world.step(render=True)

        if self._sim_app is not None:
            try:
                self._sim_app.close()
            except Exception:
                LOGGER.exception("error closing isaac sim")

    def _drain_tasks(self) -> None:
        while True:
            try:
                fn, fut = self._tasks.get_nowait()
            except queue.Empty:
                return
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except BaseException as e:
                    fut.set_exception(e)

    def run(self, fn: Callable[[], Any], timeout: float = 30.0) -> Any:
        """Run fn on the sim thread and return its result."""
        if threading.get_ident() == self._sim_thread_id:
            return fn()
        fut: Future = Future()
        self._tasks.put((fn, fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise SimTimeoutError(f"sim-thread call timed out after {timeout}s") from exc

    # ------------------------------------------------------------------
    # boot
    # ------------------------------------------------------------------

    def _boot(self) -> None:
        cfg = self.cfg
        assert cfg is not None
        self.lighting = cfg.lighting
        self.render = cfg.render
        if cfg.mock:
            LOGGER.info("booting in MOCK mode - no isaac sim")
            self.mock = True
            self._world_handle = MockWorldHandle(self, cfg.props)
            self._reset_world()
            return

        LOGGER.info("booting isaac sim (headless=%s)...", cfg.headless)
        try:
            from isaacsim import SimulationApp  # isaac sim >= 4.5
        except ImportError:
            from omni.isaac.kit import SimulationApp  # older releases

        # quiet kit's console stream; unknown argv entries are forwarded to kit
        level = cfg.kit_log_level.capitalize()
        sys.argv.append(f"--/log/outputStreamLevel={level}")

        self._sim_app = SimulationApp({"headless": cfg.headless})

        try:
            import carb.settings

            carb.settings.get_settings().set("/log/outputStreamLevel", level)
        except Exception:
            pass

        if cfg.livestream and cfg.headless:
            try:
                try:
                    from isaacsim.core.utils.extensions import enable_extension
                except ImportError:
                    from omni.isaac.core.utils.extensions import enable_extension

                ip = cfg.livestream_public_ip or _local_ip()
                self._sim_app.set_setting("/app/livestream/enabled", True)
                if ip:
                    self._sim_app.set_setting("/app/livestream/publicEndpointAddress", ip)
                enable_extension("omni.kit.livestream.webrtc")
                LOGGER.info(
                    "livestream enabled - connect the 'Isaac Sim WebRTC Streaming "
                    "Client' app to %s (TCP 49100 + UDP 47998 must be reachable)",
                    ip or "<this machine's IP>",
                )
            except Exception:
                LOGGER.exception("could not enable livestream; continuing without it")

        self._isaac = import_isaac()

        if cfg.usd_stage:
            LOGGER.info("opening stage %s", cfg.usd_stage)
            self._isaac.open_stage(cfg.usd_stage)

        self.world = self._isaac.World(
            physics_dt=cfg.physics_dt,
            rendering_dt=cfg.rendering_dt,
            stage_units_in_meters=1.0,
        )
        if not cfg.usd_stage:
            self.world.scene.add_default_ground_plane()
        for prop in cfg.props:
            try:
                self._spawn_prop(prop)
            except Exception:
                LOGGER.exception("failed to spawn prop %s", prop.get("name"))
        if cfg.lighting is not None:
            self._apply_lighting(cfg.lighting)
        if cfg.render is not None:
            self._apply_render(cfg.render)
        self._world_handle = IsaacWorldHandle(self)
        self._reset_world()
        LOGGER.info("isaac sim world ready")

    def _apply_lighting(self, lighting: dict[str, Any]) -> None:
        """Configure scene lights per FINDINGS SCN-9/W30. Best-effort: never
        raises, so bad/unavailable lighting config can't block boot."""
        try:
            import omni.usd
            from pxr import Gf, UsdLux

            stage = omni.usd.get_context().get_stage()

            dome = lighting.get("dome")
            if dome is not None:
                dome_light = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PRIM_PATH)
                dome_light.CreateIntensityAttr(float(dome.get("intensity", DEFAULT_DOME_INTENSITY)))
                color = dome.get("color", DEFAULT_DOME_COLOR)
                dome_light.CreateColorAttr(Gf.Vec3f(*[float(v) for v in color]))

            sphere_intensity = lighting.get("sphere_intensity")
            if sphere_intensity is not None:
                sphere_prim = stage.GetPrimAtPath(SPHERE_LIGHT_PRIM_PATH)
                if sphere_prim.IsValid():
                    UsdLux.SphereLight(sphere_prim).GetIntensityAttr().Set(float(sphere_intensity))
        except Exception:
            LOGGER.exception("failed to apply scene lighting")

    def _apply_render(self, render: dict[str, Any]) -> None:
        """Render-cost levers per FINDINGS CAM-12/R-11. Best-effort, like
        _apply_lighting: a bad/unavailable carb setting can't block boot.

        motion_bvh -> "/renderer/raytracingMotion/enabled" (FINDINGS CAM-12
        names this exact carb path). disable_viewport_updates has no carb
        path in any input doc (only that 5.0's SimulationApp launcher config
        gained a same-named key, RESEARCH row 241); "/app/viewport/
        disableViewportUpdates" is this module's best guess and needs GPU
        verification."""
        try:
            import carb.settings

            settings = carb.settings.get_settings()

            motion_bvh = render.get("motion_bvh")
            if motion_bvh is not None:
                settings.set("/renderer/raytracingMotion/enabled", bool(motion_bvh))

            disable_viewport_updates = render.get("disable_viewport_updates")
            if disable_viewport_updates is not None:
                settings.set("/app/viewport/disableViewportUpdates", bool(disable_viewport_updates))
        except Exception:
            LOGGER.exception("failed to apply render levers")

    def _spawn_prop(self, prop: dict[str, Any]) -> None:
        """Add a configured prop to the scene (runs on the sim thread,
        before the initial world.reset)."""
        import numpy as np

        from .spatial import to_vec3

        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        name = _prim_name(str(prop["name"]))
        prim_path = f"/World/{name}"
        position = list(to_vec3(prop.get("position")))
        kind = str(prop.get("type", "cube"))
        orientation = prop_spawn_orientation(prop)

        if kind == "usd":
            usd_path = prop.get("usd_path")
            if not usd_path:
                raise ValueError(f"prop {name}: type 'usd' needs usd_path")
            if self._usd_exists(usd_path) is False:
                raise ValueError(f"prop {name}: usd not found: {usd_path}")
            self._isaac.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            self._isaac.SingleXFormPrim(prim_path).set_world_pose(
                position=position, orientation=list(orientation)
            )
            apply_prop_physics(self._isaac, self.world, prim_path, prop)
            self._prop_specs[name] = {
                **prop,
                "name": name,
                "position": tuple(position),
                "spawn_orientation": orientation,
            }
            return

        if kind != "cube":
            raise ValueError(f"prop {name}: unknown type {kind!r} (cube or usd)")

        kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            position=np.array(position),
            orientation=np.array(orientation),
            size=float(prop.get("size", 0.05)),
        )
        if prop.get("scale") is not None:
            kwargs["scale"] = np.array([float(v) for v in prop["scale"]])
        if prop.get("color") is not None:
            kwargs["color"] = np.array([float(v) for v in prop["color"]])
        cls = self._isaac.FixedCuboid if prop.get("fixed") else self._isaac.DynamicCuboid
        self.world.scene.add(cls(**kwargs))
        # SCN-6 / W27: explicit material + offsets when the prop names them
        apply_prop_physics(self._isaac, self.world, prim_path, prop)
        self._prop_specs[name] = {
            **prop,
            "name": name,
            "position": tuple(position),
            "spawn_orientation": orientation,
        }

    def _require_booted(self) -> None:
        if not self._booted.is_set():
            raise SimNotBootedError(
                "isaac sim world is not running - configure a "
                f"{NAMESPACE}:{FAMILY}:world component and depend on it"
            )
        if self._boot_error is not None:
            raise SimNotBootedError(f"isaac sim failed to boot: {self._boot_error}")

    # ------------------------------------------------------------------
    # world controls (used by the world component's DoCommand)
    # ------------------------------------------------------------------

    def play(self) -> None:
        self._require_booted()
        if not self.mock:
            self.run(lambda: self.world.play())

    def pause(self) -> None:
        self._require_booted()
        if not self.mock:
            self.run(lambda: self.world.pause())

    def reset(self) -> None:
        self._require_booted()
        self.run(lambda: self._reset_world())

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "booted": self._booted.is_set(),
            "mock": self.mock,
            "error": str(self._boot_error) if self._boot_error else "",
            "lighting": self.lighting,
            "render": self.render,
            # OQ-14 / GPU checklist item 6: None in mock or when no probe answers
            "isaac_version": _version_string(isaac_version()),
        }
        if self._booted.is_set() and not self.mock:
            out["playing"] = self.run(lambda: bool(self.world.is_playing()))
            out["sim_time"] = self.run(lambda: float(self.world.current_time))
        return out

    def world_handle(self) -> "WorldHandle":
        """The WorldHandle the world component's DoCommand verbs drive
        the scene through (SCN-16). Every booted world has one."""
        self._require_booted()
        assert self._world_handle is not None
        return self._world_handle

    def add_usd_reference(
        self,
        usd_path: str,
        prim_path: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_wxyz: "Quat | None" = None,
    ) -> None:
        self._require_booted()
        if self.mock:
            return

        def _add():
            self._isaac.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            prim = self._isaac.SingleXFormPrim(prim_path)
            kwargs: dict[str, Any] = {"position": list(position)}
            if orientation_wxyz is not None:
                kwargs["orientation"] = list(orientation_wxyz)
            prim.set_world_pose(**kwargs)

        self.run(_add, timeout=60.0)

    # ------------------------------------------------------------------
    # component factories
    # ------------------------------------------------------------------

    # attributes that only affect the viam-side model, not the spawned prim -
    # they re-apply on reconfigure without a restart (XC-4)
    _RUNTIME_KEYS = frozenset(
        {
            "world",
            "move_timeout_sec",
            "max_linear_mps",
            "max_angular_rps",
            # gripper (models/gripper.py)
            "arm",
            "tcp_offset_m",
            "open_deg",
            "closed_deg",
            "grab_timeout_sec",
            "holding_tolerance_deg",
            "holding_effort_min_nm",
            "mock_object_width_m",
            # arm mock test knob
            "mock_stall_fraction",
        }
    )

    def _cached_handle(
        self, kind: str, name: str, attrs: dict[str, Any], factory: Callable[[], Any]
    ) -> Any:
        """One handle per component name for the life of the process (XC-4).

        viam-server re-runs reconfigure -> create_* on every config change,
        but a prim cannot be re-spawned without restarting Kit, so a change to
        any SPAWN attribute (anything outside _RUNTIME_KEYS, including the
        pose that the frame config folds in) is REJECTED with ValueError -
        the component shows the error instead of silently running a stale
        prim (the failure class phase 1 fought). Runtime attributes re-apply.
        After release_handle (the model's close()) the name is forgotten and
        the next create_* re-runs the factory, which re-attaches to the
        existing prim."""
        if name in self._handles:
            old_attrs, handle = self._handles[name]

            def strip(attrs):
                return {k: v for k, v in attrs.items() if k not in self._RUNTIME_KEYS}

            if strip(old_attrs) != strip(attrs):
                raise ValueError(
                    f"{kind} {name!r}: spawn config changed but the prim is already in "
                    "the stage; restart the module to apply (or revert the change). "
                    "Only runtime attributes "
                    f"({', '.join(sorted(self._RUNTIME_KEYS))}) apply without a restart"
                )
            return handle
        handle = factory()
        self._handles[name] = (dict(attrs), handle)
        return handle

    def release_handle(self, name: str) -> None:
        """XC-4: called from a model's close(). Forgets the cached handle,
        drops the post-reset hooks registered under ``name`` and calls
        handle.release(). The prim stays in the stage (Kit cannot un-spawn),
        so a later create_* for the same name re-attaches to it. Idempotent."""
        entry = self._handles.pop(name, None)
        self.unregister_post_reset(name)
        if entry is None:
            return
        _attrs, handle = entry
        try:
            handle.release()
        except Exception:
            LOGGER.exception("%r: handle.release() failed", name)

    def _usd_exists(self, path: str) -> bool | None:
        """True/False if we can check, None if omni.client is unavailable."""
        client = getattr(self._isaac, "client", None)
        if client is None:
            return None
        try:
            result, _ = client.stat(path)
            return result == client.Result.OK
        except Exception:
            return None

    def _resolve_usd(self, attrs: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """Return (absolute usd path or None, known-asset metadata)."""
        meta: dict[str, Any] = {}
        usd = attrs.get("usd_path")
        asset = attrs.get("asset")
        if asset:
            if asset not in KNOWN_ASSETS:
                raise ValueError(
                    f"unknown asset {asset!r}; known: {sorted(KNOWN_ASSETS)} "
                    "(or set usd_path directly)"
                )
            meta = KNOWN_ASSETS[asset]
            if not usd:
                root = self._isaac.get_assets_root_path()
                if root is None:
                    raise RuntimeError("could not reach the isaac sim assets server")
                candidates = meta["usd"]
                for rel in candidates:
                    if self._usd_exists(root + rel) is not False:
                        usd = root + rel
                        break
                if usd is None:
                    raise ValueError(
                        f"asset {asset!r}: none of {candidates} exist under {root}; "
                        "the asset layout may have changed in this isaac release"
                    )
        # a USD reference to a missing file "succeeds" but leaves an empty
        # prim, which later fails with confusing physics-tensor errors -
        # catch it here instead
        if usd and self._usd_exists(usd) is False:
            raise ValueError(f"usd not found: {usd}")
        return usd, meta

    def create_arm(self, name: str, attrs: dict[str, Any]) -> "ArmHandle":
        self._require_booted()
        if self.mock:

            def factory():
                return MockArmHandle(name, attrs)
        else:

            def factory():
                def _spawn():
                    handle = self._create_arm_isaac(name, attrs)
                    # ARM-15/ARM-16: snapshot the controller gains once and
                    # apply the solver iteration count, so post_reset (below)
                    # can re-apply both after a world.reset() undoes them.
                    handle._solver_iterations = ARM_SOLVER_POSITION_ITERATIONS
                    handle._art.set_solver_position_iteration_count(ARM_SOLVER_POSITION_ITERATIONS)
                    handle._gains = handle._art.get_articulation_controller().get_gains()
                    return handle

                handle = self.run(_spawn, timeout=120.0)
                self.register_post_reset(lambda: handle.post_reset(), owner=name)
                return handle

        return self._cached_handle("arm", name, attrs, factory)

    def _place_root_xform(self, prim_path: str, position: Vec3, orientation: Quat) -> bool:
        """Author the spawn pose as plain USD xform ops on the referenced asset's
        root prim, BEFORE any Isaac prim wrapper exists for it.

        Passing position/orientation through SingleArticulation does not work
        once a physics sim view exists (the world was reset at boot): the
        wrapper routes the write to a physics handle that has not parsed the
        new articulation yet, drops it, and captures identity as the default
        state (Isaac 5.0 xform_prim.py:150-175). Writing the ops here makes the
        USD pose the truth PhysX parses on the next world.reset(). Never raises."""
        try:
            from pxr import Gf, UsdGeom

            stage = self._isaac.get_prim_at_path(prim_path).GetStage()
            prim = stage.GetPrimAtPath(prim_path)
            xformable = UsdGeom.Xformable(prim)
            # ClearXformOpOrder drops the order, not the op attributes: a
            # referenced asset may already carry xformOp:orient as quatd or
            # quatf, and AddOrientOp raises if the requested precision differs.
            # Match whatever precision is already authored (default double).
            xformable.ClearXformOpOrder()
            double = UsdGeom.XformOp.PrecisionDouble
            single = UsdGeom.XformOp.PrecisionFloat
            translate_attr = prim.GetAttribute("xformOp:translate")
            translate_is_float = (
                bool(translate_attr) and str(translate_attr.GetTypeName()) == "float3"
            )
            orient_attr = prim.GetAttribute("xformOp:orient")
            orient_is_float = bool(orient_attr) and str(orient_attr.GetTypeName()) == "quatf"
            scale_attr = prim.GetAttribute("xformOp:scale")
            scale_is_double = bool(scale_attr) and str(scale_attr.GetTypeName()) == "double3"

            px, py, pz = (float(v) for v in position)
            translate_op = xformable.AddTranslateOp(single if translate_is_float else double)
            translate_op.Set(Gf.Vec3f(px, py, pz) if translate_is_float else Gf.Vec3d(px, py, pz))

            w, x, y, z = (float(v) for v in orientation)
            orient_op = xformable.AddOrientOp(single if orient_is_float else double)
            orient_op.Set(
                Gf.Quatf(w, Gf.Vec3f(x, y, z))
                if orient_is_float
                else Gf.Quatd(w, Gf.Vec3d(x, y, z))
            )

            scale_op = xformable.AddScaleOp(double if scale_is_double else single)
            scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0) if scale_is_double else Gf.Vec3f(1.0, 1.0, 1.0))
            LOGGER.info(
                "placed %s via usd xform ops: position=%s orientation=%s",
                prim_path,
                position,
                orientation,
            )
            return True
        except Exception:
            LOGGER.exception("failed to author spawn pose on %s", prim_path)
            return False

    def _anchor_fixed_base(self, prim_path: str, position: Vec3, orientation: Quat) -> bool:
        """Re-anchor a world-anchored fixed-base joint under prim_path to the
        spawn pose (position, orientation). The UR assets fix their base to
        the world frame with a FixedJoint whose localPos0/localRot0 are
        authored in world coordinates; PhysX re-syncs the root xform to that
        joint frame on world.reset(), silently undoing the spawn pose passed
        to SingleArticulation (FINDINGS ARM-9/XC-1). Never raises: a failure
        here should not fail the spawn, only leave the pose un-anchored."""
        try:
            from pxr import Gf, Sdf, Usd, UsdPhysics

            stage = self._isaac.get_prim_at_path(prim_path).GetStage()
            root_prim = stage.GetPrimAtPath(prim_path)
            for prim in Usd.PrimRange(root_prim):
                if not prim.IsA(UsdPhysics.FixedJoint):
                    continue
                joint = UsdPhysics.Joint(prim)
                if joint.GetBody0Rel().GetTargets():
                    continue

                pos_attr = prim.GetAttribute("physics:localPos0")
                rot_attr = prim.GetAttribute("physics:localRot0")
                authored_pos_gf = pos_attr.Get() if pos_attr else None
                authored_rot_gf = rot_attr.Get() if rot_attr else None
                authored_pos: Vec3 = (
                    (authored_pos_gf[0], authored_pos_gf[1], authored_pos_gf[2])
                    if authored_pos_gf is not None
                    else (0.0, 0.0, 0.0)
                )
                authored_quat: Quat = (
                    (
                        authored_rot_gf.GetReal(),
                        authored_rot_gf.GetImaginary()[0],
                        authored_rot_gf.GetImaginary()[1],
                        authored_rot_gf.GetImaginary()[2],
                    )
                    if authored_rot_gf is not None
                    else (1.0, 0.0, 0.0, 0.0)
                )

                new_pos, new_quat = anchor_fixed_joint_frame(
                    position, orientation, authored_pos, authored_quat
                )
                if pos_attr is None:
                    pos_attr = prim.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f)
                if rot_attr is None:
                    rot_attr = prim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf)
                pos_attr.Set(Gf.Vec3f(*new_pos))
                rot_attr.Set(Gf.Quatf(new_quat[0], Gf.Vec3f(*new_quat[1:])))
                LOGGER.info(
                    "re-anchored fixed base joint %s to position=%s orientation=%s",
                    prim.GetPath(),
                    new_pos,
                    new_quat,
                )
                return True
            LOGGER.warning(
                "no world-anchored fixed joint found under %s; spawn pose relies on the prim xform",
                prim_path,
            )
            return False
        except Exception:
            LOGGER.exception("failed to re-anchor fixed base joint under %s", prim_path)
            return False

    def _create_arm_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacArmHandle":
        from .spatial import to_vec3

        usd, meta = self._resolve_usd(attrs)
        prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
        position = to_vec3(attrs.get("position"))
        orientation = spawn_orientation(attrs, meta)
        if usd:
            self._isaac.add_reference_to_stage(usd_path=usd, prim_path=prim_path)
            # Spawn pose goes into USD first (see _place_root_xform) and the
            # world-anchored base joint is moved with it; the wrapper below is
            # constructed WITHOUT a pose on purpose.
            self._place_root_xform(prim_path, position, orientation)
            self._anchor_fixed_base(prim_path, position, orientation)

        art = self._isaac.SingleArticulation(prim_path=prim_path, name=name)
        self.world.scene.add(art)
        self._reset_world()
        try:
            root_pos, root_quat = art.get_world_pose()
            LOGGER.info(
                "arm %r root pose after reset: position=%s orientation_wxyz=%s (requested %s / %s)",
                name,
                [round(float(v), 4) for v in root_pos],
                [round(float(v), 4) for v in root_quat],
                position,
                orientation,
            )
        except Exception:
            LOGGER.exception("could not read root pose for arm %r after reset", name)

        ee = None
        asset = attrs.get("asset")
        ee_path = attrs.get("end_effector_prim") or (
            f"{prim_path}/wrist_3_link"
            if isinstance(asset, str) and asset.startswith("ur")
            else None
        )
        if ee_path:
            ee = self._isaac.SingleXFormPrim(ee_path)
        correction = (
            _as_quat(meta["base_frame_correction"])
            if meta.get("base_frame_correction") is not None
            else (1.0, 0.0, 0.0, 0.0)
        )
        return IsaacArmHandle(
            self, art, ee, meta.get("joint_names"), base_correction=correction, prim_path=prim_path
        )

    def create_camera(self, name: str, attrs: dict[str, Any]) -> "CameraHandle":
        self._require_booted()
        # CAM-17: wired once per handle (not in the model) so both backends
        # drop their cache / re-arm acquisition after every world.reset().
        # Registered inside factory() - which _cached_handle only calls on
        # first construction - because viam-server re-runs reconfigure ->
        # create_camera on every config change and _cached_handle returns the
        # same handle each time; registering outside factory() would append
        # a duplicate hook per reconfigure. Dispatched dynamically (not a
        # bound-method reference captured now) so tests can monkeypatch
        # handle.post_reset after creation.
        if self.mock:

            def factory():
                handle = MockCameraHandle(name, attrs)
                self.register_post_reset(lambda: handle.post_reset(), owner=name)
                return handle
        else:

            def factory():
                handle = self.run(lambda: self._create_camera_isaac(name, attrs), timeout=120.0)
                self.register_post_reset(lambda: handle.post_reset(), owner=name)
                return handle

        return self._cached_handle("camera", name, attrs, factory)

    def _create_camera_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacCameraHandle":
        parent = attrs.get("parent_prim")
        if parent:
            self._require_prim(parent)
        prim_path = _camera_prim_path(name, attrs)
        width = int(attrs.get("width", DEFAULT_CAMERA_WIDTH))
        height = int(attrs.get("height", DEFAULT_CAMERA_HEIGHT))

        kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            resolution=(width, height),
        )
        if attrs.get("position") is not None:
            kwargs["position"] = list(to_vec3(attrs.get("position")))
        if attrs.get("orientation_rpy_deg") is not None:
            r, p, y = to_vec3(attrs.get("orientation_rpy_deg"))
            kwargs["orientation"] = list(quat_from_euler_deg(r, p, y))

        annotator_device = attrs.get("annotator_device")
        if annotator_device is not None:
            if caps().camera_supports_annotator_device:
                kwargs["annotator_device"] = annotator_device
            else:
                LOGGER.info(
                    "camera %s: annotator_device %r ignored - the running isaac "
                    "release has no GPU-resident annotator path (CAM-12, 5.0 only)",
                    name,
                    annotator_device,
                )

        cam = self._isaac.Camera(**kwargs)
        # 4.5: get_resolution()/apertures only read back correctly once the
        # render product exists (IS-1), so initialize() must come first.
        self._initialize_camera(name, cam)

        _place_camera(cam, attrs)
        rendering_dt = self.cfg.rendering_dt if self.cfg is not None else 1.0 / 60.0
        _configure_camera_optics(cam, attrs, rendering_dt)

        return IsaacCameraHandle(
            self,
            cam,
            depth_enabled=bool(attrs.get("depth")),
            image_format=attrs.get("image_format", "png"),
            frequency=attrs.get("frequency"),
        )

    def _initialize_camera(self, name: str, cam: Any) -> None:
        """Bounded retry around ``Camera.initialize()`` for a cold renderer
        (CAMERA_INIT_ATTEMPTS's comment has the failure). Runs on the sim
        thread, where the loop is paused while we execute, so the render
        ticks between attempts are stepped here. On final failure the
        half-created render product is destroyed so the next resource build
        starts clean instead of wrapping the corpse."""
        last_error: Exception | None = None
        for attempt in range(CAMERA_INIT_ATTEMPTS):
            if attempt:
                for _ in range(CAMERA_INIT_RENDER_TICKS):
                    self.world.step(render=True)
            try:
                cam.initialize()
                return
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "camera %s: initialize attempt %d/%d failed: %r",
                    name,
                    attempt + 1,
                    CAMERA_INIT_ATTEMPTS,
                    exc,
                )
        destroy = getattr(cam, "destroy", None)
        if destroy is not None:
            try:
                destroy()
            except Exception:
                LOGGER.exception("camera %s: destroy() after failed initialize", name)
        raise CameraInitError(
            f"camera {name}: initialize() failed after {CAMERA_INIT_ATTEMPTS} attempts "
            f"with {CAMERA_INIT_RENDER_TICKS} render ticks between them"
        ) from last_error

    def _require_prim(self, prim_path: str) -> None:
        """Raise a helpful error if prim_path doesn't exist in the stage."""
        get_prim = getattr(self._isaac, "get_prim_at_path", None)
        if get_prim is None:
            return
        prim_path = prim_path.strip()  # a pasted path with stray whitespace is never valid
        prim = get_prim(prim_path)
        if prim is None or not prim.IsValid():
            parent_path = prim_path.rsplit("/", 1)[0] or "/"
            hint = ""
            parent = get_prim(parent_path)
            if parent is not None and parent.IsValid():
                children = [c.GetName() for c in parent.GetChildren()]
                hint = f"; children of {parent_path}: {children}"
            raise PrimNotFoundError(f"prim not found: {prim_path}{hint}")

    def create_base(self, name: str, attrs: dict[str, Any]) -> "BaseHandle":
        self._require_booted()
        if self.mock:

            def factory():
                return MockBaseHandle(name, attrs)
        else:

            def factory():
                return self.run(lambda: self._create_base_isaac(name, attrs), timeout=120.0)

        return self._cached_handle("base", name, attrs, factory)

    def _create_base_isaac(self, name: str, attrs: dict[str, Any]) -> "IsaacBaseHandle":
        from .spatial import to_vec3

        usd, meta = self._resolve_usd(attrs)
        prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"
        wheel_joints = attrs.get("wheel_joints") or meta.get("wheel_joints")
        if not wheel_joints or len(wheel_joints) != 2:
            raise ValueError(
                "base needs wheel_joints: [left_joint_name, right_joint_name] "
                "(known assets like 'jetbot' provide defaults)"
            )
        wheel_radius = float(attrs.get("wheel_radius", meta.get("wheel_radius", 0.05)))
        wheel_base = float(attrs.get("wheel_base", meta.get("wheel_base", 0.3)))
        position = to_vec3(attrs.get("position"))

        base_kwargs: dict[str, Any] = dict(
            prim_path=prim_path,
            name=name,
            wheel_dof_names=list(wheel_joints),
            create_robot=usd is not None,
            usd_path=usd,
            position=list(position),
        )
        if attrs.get("orientation_wxyz") is not None:
            base_kwargs["orientation"] = [float(v) for v in attrs["orientation_wxyz"]]
        robot = self._isaac.WheeledRobot(**base_kwargs)
        self.world.scene.add(robot)
        self._reset_world()

        controller = self._isaac.DifferentialController(
            name=f"{name}_controller",
            wheel_radius=wheel_radius,
            wheel_base=wheel_base,
        )
        handle = IsaacBaseHandle(self, robot, controller, wheel_radius, wheel_base)
        self.world.add_physics_callback(f"{name}_drive", handle._on_physics_step)
        return handle

    # ------------------------------------------------------------------
    # gripper (FINDINGS ARM-2..ARM-8, W12-W15; DEC-2, DEC-12)
    # ------------------------------------------------------------------

    def create_gripper(self, name: str, attrs: dict[str, Any]) -> "GripperHandle":
        """Attach a gripper to an arm that is already in the sim.

        attrs: world, arm (Viam name of the arm it rides - validate_config
        lists it as a dependency so viam-server builds the arm first), asset
        (default "robotiq_2f_85"), parent_prim (default <arm prim>/wrist_3_link),
        local_position / local_orientation_rpy_deg (mount pose of the gripper's
        base_link on parent_prim, default identity - NOT the frame, see
        models/gripper.py), open_deg, closed_deg, holding_tolerance_deg,
        mock_object_width_m."""
        self._require_booted()
        arm_name = str(attrs.get("arm", ""))
        arm_entry = self._handles.get(arm_name)
        if arm_entry is None:
            raise ValueError(
                f"gripper {name!r}: arm {arm_name!r} is not attached to the sim "
                '(set "arm" to the name of the isaac-sim arm component it rides)'
            )
        arm_attrs, arm_handle = arm_entry
        if not isinstance(arm_handle, ArmHandle):
            raise ValueError(f"gripper {name!r}: {arm_name!r} is not an arm")
        if self.mock:

            def factory():
                return MockGripperHandle(name, attrs, arm_handle)
        else:

            def factory():
                handle = self.run(
                    lambda: self._create_gripper_isaac(name, attrs, arm_attrs, arm_handle),
                    timeout=120.0,
                )
                # R-5 / GPU checklist item 6: re-command the last commanded
                # jaw target after a reset mid-pick, so it doesn't drop
                # whatever it was holding.
                self.register_post_reset(lambda: handle.post_reset(), owner=name)
                return handle

        return self._cached_handle("gripper", name, attrs, factory)

    def _create_gripper_isaac(
        self, name: str, attrs: dict[str, Any], arm_attrs: dict[str, Any], arm: "ArmHandle"
    ) -> "IsaacGripperHandle":
        """Sim thread. Reference the asset under f"{arm prim}/Gripper" with
        articulationEnabled=False on the gripper root so its joints join the
        ARM's articulation; author a PhysicsFixedJoint parent_prim (wrist_3_link)
        <-> gripper base_link at the local mount pose; assert the pad prims
        exist and carry PhysicsCollisionAPI (R-4 / OQ-4) and raise a clear
        error otherwise; ONE reset via _reset_world(); then arm.refresh_dofs()
        and log the full dof_names - refusing if the six UR joint names are no
        longer resolvable (R-3)."""
        from pxr import Gf, Sdf, Usd, UsdPhysics

        if not isinstance(arm, IsaacArmHandle):
            raise ValueError(
                f"gripper {name!r}: arm handle for {attrs.get('arm')!r} is not an Isaac arm handle"
            )

        gripper_attrs = dict(attrs)
        gripper_attrs.setdefault("asset", "robotiq_2f_85")
        usd, meta = self._resolve_usd(gripper_attrs)
        if usd is None:
            raise ValueError(f"gripper {name!r}: no usd_path or known asset resolved")

        arm_prim = arm._prim_path
        gripper_prim = f"{arm_prim}/Gripper"
        self._isaac.add_reference_to_stage(usd_path=usd, prim_path=gripper_prim)

        stage = self._isaac.get_prim_at_path(gripper_prim).GetStage()
        gripper_root = stage.GetPrimAtPath(gripper_prim)

        # OQ-4 diagnostics: what actually composed under the reference. A layer
        # without a defaultPrim composes NOTHING through AddReference(path), so
        # re-reference its first root prim explicitly before giving up.
        composed = _describe_composition(Usd, Sdf, gripper_root, usd)
        LOGGER.info("gripper %r composed under %s: %s", name, gripper_prim, composed)
        if not composed["children"] and composed["layer_root_prims"]:
            if not composed["layer_default_prim"]:
                root_prim_path = composed["layer_root_prims"][0]
                references = gripper_root.GetReferences()
                references.ClearReferences()
                references.AddReference(Sdf.Reference(usd, Sdf.Path(root_prim_path)))
                composed = _describe_composition(Usd, Sdf, gripper_root, usd)
                LOGGER.info(
                    "gripper %r: layer has no defaultPrim; re-referenced %s explicitly: %s",
                    name,
                    root_prim_path,
                    composed,
                )
        if not composed["children"]:
            raise ValueError(
                f"gripper {name!r}: {usd} composed no prims under {gripper_prim} "
                f"(layer defaultPrim={composed['layer_default_prim']!r}, root prims="
                f"{composed['layer_root_prims']}). The asset did not load - check the module "
                "log for omni.client/USD resolver errors and the asset path"
            )

        rewrite = self._rewrite_unresolvable_references(Usd, Sdf, gripper_root)
        LOGGER.info(
            "gripper %r: %s reference rewrite - applied %d, missing in bucket %d, "
            "de-instanced %d: %s",
            name,
            UNRESOLVABLE_ASSET_HOST,
            len(rewrite["applied"]),
            len(rewrite["missing"]),
            len(rewrite["de_instanced"]),
            rewrite,
        )

        # The gripper's own articulation root must GO, not merely be disabled:
        # it now sits inside the arm root's subtree, and PhysX drops an
        # articulation that contains a nested root (seen on the GPU as the
        # arm's view losing its backend). Its joints then join the arm's
        # articulation through the fixed joint below. The API is on the
        # asset's default prim (Gripper/Robotiq_2F_85), not on our Gripper prim.
        removed_roots = _remove_articulation_roots(
            Usd, UsdPhysics, self._isaac.PhysxSchema, gripper_root
        )
        if removed_roots:
            LOGGER.info("gripper %r: removed articulation root API from %s", name, removed_roots)
        else:
            LOGGER.warning("gripper %r: no ArticulationRootAPI found under %s", name, gripper_prim)

        base_link_prim = None
        for prim in Usd.PrimRange(gripper_root):
            if prim.GetName() == "base_link":
                base_link_prim = prim
                break
        if base_link_prim is None:
            raise ValueError(f"gripper {name!r}: base_link prim not found under {gripper_prim}")

        parent_prim = attrs.get("parent_prim") or f"{arm_prim}/wrist_3_link"
        local_position = to_vec3(attrs.get("local_position"), default=(0.0, 0.0, 0.0))
        r, p, y = to_vec3(attrs.get("local_orientation_rpy_deg"), default=(0.0, 0.0, 0.0))
        local_quat = quat_from_euler_deg(r, p, y)

        joint = UsdPhysics.FixedJoint.Define(stage, f"{gripper_prim}/WristFixedJoint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_prim)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(str(base_link_prim.GetPath()))])
        px, py, pz = (float(v) for v in local_position)
        joint.CreateLocalPos0Attr(Gf.Vec3f(px, py, pz))
        quat_w, quat_x, quat_y, quat_z = (float(v) for v in local_quat)
        joint.CreateLocalRot0Attr(Gf.Quatf(quat_w, Gf.Vec3f(quat_x, quat_y, quat_z)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        LOGGER.info(
            "authored gripper wrist fixed joint %s/WristFixedJoint: body0=%s body1=%s",
            gripper_prim,
            parent_prim,
            base_link_prim.GetPath(),
        )

        # R-4 / OQ-4: the 2F-85 pad prims must carry PhysicsCollisionAPI -
        # verify this BEFORE resetting so a missing-asset failure is clear.
        # A pad counts as collidable when it OR any descendant (the asset
        # nests collision meshes under the link) carries the API. Everything
        # observed is logged before the refusal, so the GPU checklist can
        # record it from the module log alone.
        pad_status = _pad_collision_status(Usd, UsdPhysics, gripper_root)
        asset_refs = _reference_asset_paths(Usd, gripper_root)
        unresolved_refs = [path for path in asset_refs if UNRESOLVABLE_ASSET_HOST in path]
        LOGGER.info(
            "gripper %r pad prims (path, collision on self, collision in subtree): %s",
            name,
            pad_status,
        )
        LOGGER.info(
            "gripper %r asset references under %s: %d total, %d on %s: %s",
            name,
            gripper_prim,
            len(asset_refs),
            len(unresolved_refs),
            UNRESOLVABLE_ASSET_HOST,
            unresolved_refs,
        )
        if not any(in_subtree for _path, _on_self, in_subtree in pad_status):
            raise ValueError(
                f"gripper {name!r}: 2F-85 pad prims missing PhysicsCollisionAPI (R-4/OQ-4). "
                f"pads seen: {pad_status}; references on {UNRESOLVABLE_ASSET_HOST}: "
                f"{unresolved_refs}; rewrite: {rewrite}; composed: {composed}. "
                "Fall back per R-2 (rewrite "
                "references to the assets bucket's parts/*.usd, or a module-authored "
                "parallel-jaw USD)"
            )

        dof_count_before = len(arm.all_dof_names())
        # The arm's SingleArticulation was initialized for the arm alone; its
        # default joint state is sized for that DOF count and the scene's
        # post_reset would push it into the new, larger articulation and fail.
        # Register a fresh wrapper BEFORE the reset so reset() initializes it
        # against the new topology (the prim, and the arm's name, stay).
        arm_object_name = arm._art.name
        self.world.scene.remove_object(arm_object_name, registry_only=True)
        fresh_articulation = self._isaac.SingleArticulation(
            prim_path=arm._prim_path, name=arm_object_name
        )
        self.world.scene.add(fresh_articulation)
        arm.replace_articulation(fresh_articulation)
        self._reset_world()
        arm.refresh_dofs()
        all_names = arm.all_dof_names()
        LOGGER.info(
            "arm %r articulation dof_names after gripper attach (%d): %s",
            arm_prim,
            len(all_names),
            all_names,
        )

        added_dof_count = len(all_names) - dof_count_before
        expected_dof_count = caps().gripper_dof_count
        if added_dof_count != expected_dof_count:
            LOGGER.warning(
                "gripper %r added %d dofs to the arm articulation; expected %d for this isaac "
                "release (OQ-5)",
                name,
                added_dof_count,
                expected_dof_count,
            )

        drive_joint = meta.get("drive_joint", "finger_joint")
        # Defaults for open/closed come from the drive joint's authored limits
        # (GPU run 12: this asset rests OPEN at its lower limit of ~7.8 deg, not
        # at 0 as W13 assumed, and closes at 47); explicit attrs still win, and
        # the paper values remain the fallback when limits cannot be read.
        lower_deg, upper_deg = self._drive_joint_limits_deg(arm, drive_joint)
        open_deg_default = lower_deg if lower_deg is not None else caps().gripper_open_deg
        closed_deg_default = upper_deg if upper_deg is not None else caps().gripper_closed_deg
        open_rad = math.radians(attrs.get("open_deg", open_deg_default))
        closed_rad = math.radians(attrs.get("closed_deg", closed_deg_default))
        LOGGER.info(
            "gripper %r drive joint %r: open %.2f deg, closed %.2f deg (authored limits %s..%s)",
            name,
            drive_joint,
            math.degrees(open_rad),
            math.degrees(closed_rad),
            lower_deg,
            upper_deg,
        )
        holding_tolerance_rad = math.radians(
            attrs.get("holding_tolerance_deg", DEFAULT_HOLDING_TOLERANCE_DEG)
        )
        effort_min = attrs.get("holding_effort_min_nm")
        holding_effort_min = None if effort_min is None else float(effort_min)
        handle = IsaacGripperHandle(
            self,
            arm._art,
            drive_joint,
            open_rad,
            closed_rad,
            holding_tolerance_rad,
            gripper_prim,
            holding_effort_min=holding_effort_min,
        )
        handle.parent_prim_path = parent_prim
        # the gripper handle just released the passive drives; the arm's
        # post-reset hook re-applies its gains snapshot, so retake it now
        try:
            arm._gains = arm._art.get_articulation_controller().get_gains()
        except Exception:
            LOGGER.exception("could not re-snapshot the arm gains after the gripper attach")
        return handle

    @staticmethod
    def _drive_joint_limits_deg(
        arm: "IsaacArmHandle", drive_joint: str
    ) -> tuple[float | None, float | None]:
        """(lower, upper) authored limits of the gripper's drive joint in
        degrees, read off the arm articulation after attach; (None, None)
        when the joint is absent or the API is unavailable. Sim thread."""
        names = arm.all_dof_names()
        if drive_joint not in names:
            return None, None
        index = names.index(drive_joint)
        art = arm._art
        try:
            # SingleArticulation (5.0) exposes the view; older wrappers had
            # get_dof_limits / dof_properties directly
            view = getattr(art, "_articulation_view", None)
            if view is not None and hasattr(view, "get_dof_limits"):
                limits = view.get_dof_limits()
                row = limits[0][index] if len(getattr(limits, "shape", ())) == 3 else limits[index]
                return math.degrees(float(row[0])), math.degrees(float(row[1]))
            if hasattr(art, "get_dof_limits"):
                row = art.get_dof_limits()[index]
                return math.degrees(float(row[0])), math.degrees(float(row[1]))
            properties = getattr(art, "dof_properties", None)
            if properties is not None:
                return (
                    math.degrees(float(properties["lower"][index])),
                    math.degrees(float(properties["upper"][index])),
                )
        except Exception:
            LOGGER.exception("could not read the drive joint limits for %r", drive_joint)
        return None, None

    def _rewrite_unresolvable_references(
        self, usd: Any, sdf: Any, root_prim: Any
    ) -> dict[str, Any]:
        """R-4 / OQ-4 first fallback (confirmed on the GPU, 2026-08-28): the
        5.0 2F-85 part meshes - the ONLY visual and collision geometry the
        asset has - are referenced from omniverse://isaac-dev..., which is
        NXDOMAIN outside NVIDIA, so the gripper composes with no geometry at
        all. The same files should live under the public assets root at the
        same /Isaac/... path, so re-point each such reference at the bucket,
        as an override in this stage's root layer (the remote asset layer is
        read-only). Instanceable prims are de-instanced first: a reference
        that lives on an instance proxy cannot be overridden. A part the
        bucket lacks is left alone, and the bucket directory is listed once so
        the report shows what IS there. Returns the report (applied pairs,
        missing candidates, de-instanced prims, bucket listing). Sim thread."""
        report: dict[str, Any] = {
            "applied": [],
            "missing": [],
            "de_instanced": [],
            "bucket_listing": {},
        }
        assets_root = self._isaac.get_assets_root_path()
        if not assets_root:
            LOGGER.warning("cannot re-point %s references: no assets root", UNRESOLVABLE_ASSET_HOST)
            report["error"] = "no assets root"
            return report
        # Two passes by PATH, never by held prim object: de-instancing an
        # instance expires every proxy prim under it, and touching an expired
        # proxy from a previously materialized traversal raises.
        stage = root_prim.GetStage()
        instance_paths = [
            str(prim.GetPath())
            for prim in _prim_range(usd, root_prim)
            if not prim.IsInstanceProxy() and prim.IsInstanceable()
        ]
        for path in instance_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsInstanceable():
                prim.SetInstanceable(False)
                report["de_instanced"].append(path)
        rewrite_paths = [
            str(prim.GetPath())
            for prim in _prim_range(usd, root_prim)
            if not prim.IsInstanceProxy()
        ]
        for path in rewrite_paths:
            prim = stage.GetPrimAtPath(path)
            try:
                list_op = prim.GetMetadata("references")
                items = list(list_op.GetAddedOrExplicitItems()) if list_op is not None else []
            except Exception:
                continue
            new_items, pairs, missing = _rewritten_references(
                sdf, items, assets_root, self._usd_exists
            )
            report["missing"].extend(missing)
            if pairs:
                prim.GetReferences().SetReferences(new_items)
                report["applied"].extend(pairs)
        for candidate in report["missing"]:
            directory = candidate.rsplit("/", 1)[0] + "/"
            if directory not in report["bucket_listing"]:
                report["bucket_listing"][directory] = self._list_assets_dir(directory)
        return report

    def _list_assets_dir(self, url: str) -> list[str] | None:
        """Names under a bucket directory via omni.client.list; None when the
        listing is unavailable (no client, or the directory does not exist)."""
        client = getattr(self._isaac, "client", None)
        if client is None:
            return None
        try:
            result, entries = client.list(url)
            if result != client.Result.OK:
                return None
            return sorted(str(entry.relative_path) for entry in entries)
        except Exception:
            return None


# FINDINGS R-4 / OQ-4: the host the 5.0 2F-85 sub-references point at; it is
# NXDOMAIN outside NVIDIA, so anything referencing it composes empty.
UNRESOLVABLE_ASSET_HOST = "isaac-dev"
MM_PER_M = 1000.0  # debug DoCommands report positions in mm (models/world.py has the same)
ASSETS_PATH_MARKER = "/Isaac/"  # every asset path is rooted here under both hosts

# R-4: the 2F-85 asset has no `*_pad` links; the fingertip geometry that must
# collide is the `..._fingertipsstep_..` mesh under `left/right_inner_finger`.
PAD_PRIM_NAME_FRAGMENTS = ("pad", "fingertip", "inner_finger")


def _rewritten_references(
    sdf: Any,
    items: Sequence[Any],
    assets_root: str,
    exists: Callable[[str], bool | None],
) -> tuple[list[Any], list[tuple[str, str]], list[str]]:
    """Map every reference on the unresolvable host onto ``assets_root`` +
    its ``/Isaac/...`` path, keeping the reference's primPath/layerOffset.
    Returns (the full new item list, the (old, new) pairs changed, the bucket
    candidates that provably do not exist). An item whose bucket file does
    not exist (``exists`` returned False) is kept as-is - None means "could
    not check", so it is tried."""
    new_items: list[Any] = []
    pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    for item in items:
        asset_path = str(getattr(item, "assetPath", ""))
        marker_at = asset_path.find(ASSETS_PATH_MARKER)
        if UNRESOLVABLE_ASSET_HOST not in asset_path or marker_at < 0:
            new_items.append(item)
            continue
        candidate = assets_root.rstrip("/") + asset_path[marker_at:]
        if exists(candidate) is False:
            LOGGER.warning("assets root has no %s; keeping %s", candidate, asset_path)
            new_items.append(item)
            missing.append(candidate)
            continue
        new_items.append(sdf.Reference(candidate, item.primPath, item.layerOffset))
        pairs.append((asset_path, candidate))
    return new_items, pairs, missing


def _remove_articulation_roots(
    usd: Any, usd_physics: Any, physx_schema: Any, root_prim: Any
) -> list[str]:
    """Remove ``UsdPhysics.ArticulationRootAPI`` (and PhysX's companion
    ``PhysxArticulationAPI``) from every prim under ``root_prim`` that carries
    it, as a root-layer override (the asset layer is read-only). Returns the
    prim paths changed. A gripper attached under an arm must not bring its
    own articulation root: PhysX rejects a nested root and drops the whole
    articulation."""
    removed: list[str] = []
    for prim in _prim_range(usd, root_prim):
        if prim.IsInstanceProxy() or not prim.HasAPI(usd_physics.ArticulationRootAPI):
            continue
        prim.RemoveAPI(usd_physics.ArticulationRootAPI)
        if physx_schema is not None and prim.HasAPI(physx_schema.PhysxArticulationAPI):
            prim.RemoveAPI(physx_schema.PhysxArticulationAPI)
        removed.append(str(prim.GetPath()))
    return removed


def _pad_collision_status(
    usd: Any, usd_physics: Any, root_prim: Any
) -> list[tuple[str, bool, bool]]:
    """(path, has CollisionAPI itself, has it anywhere in its subtree) for
    every prim under ``root_prim`` whose name contains "pad" (R-4). Pure over
    the pxr modules passed in, so it is testable with fakes."""
    status: list[tuple[str, bool, bool]] = []
    for prim in _prim_range(usd, root_prim):
        prim_name = prim.GetName().lower()
        if not any(fragment in prim_name for fragment in PAD_PRIM_NAME_FRAGMENTS):
            continue
        on_self = bool(prim.HasAPI(usd_physics.CollisionAPI))
        in_subtree = on_self or any(
            bool(child.HasAPI(usd_physics.CollisionAPI)) for child in _prim_range(usd, prim)
        )
        status.append((str(prim.GetPath()), on_self, in_subtree))
    return status


def _prim_range(usd: Any, root_prim: Any) -> Any:
    """Traverse ``root_prim``'s subtree INCLUDING instance proxies: Isaac's
    robot assets mark link meshes instanceable, and a default PrimRange
    stops at an instance, hiding the collision meshes under it."""
    traverse_instances = getattr(usd, "TraverseInstanceProxies", None)
    if traverse_instances is None:
        return usd.PrimRange(root_prim)
    return usd.PrimRange(root_prim, traverse_instances())


SAMPLE_PRIM_PATHS = 60


def _describe_composition(usd: Any, sdf: Any, root_prim: Any, usd_path: str) -> dict[str, Any]:
    """What composed under ``root_prim`` after referencing ``usd_path``, plus
    what the referenced layer itself declares - enough to tell "the asset
    did not load" from "it loaded with names we did not expect" (OQ-4).
    Best-effort: every field degrades to an empty value rather than raising."""
    out: dict[str, Any] = {
        "children": [],
        "prim_count": 0,
        "instanceable_count": 0,
        "sample_paths": [],
        "layer_default_prim": "",
        "layer_root_prims": [],
    }
    try:
        out["children"] = [child.GetName() for child in root_prim.GetChildren()]
        prims = list(_prim_range(usd, root_prim))
        out["prim_count"] = len(prims)
        out["instanceable_count"] = sum(1 for prim in prims if prim.IsInstanceable())
        root_path = str(root_prim.GetPath())
        out["sample_paths"] = [
            str(prim.GetPath())[len(root_path) :] for prim in prims[1 : SAMPLE_PRIM_PATHS + 1]
        ]
    except Exception:
        LOGGER.exception("could not describe the prims under %s", root_prim)
    try:
        layer = sdf.Layer.FindOrOpen(usd_path)
        if layer is not None:
            out["layer_default_prim"] = str(layer.defaultPrim)
            out["layer_root_prims"] = [str(spec.path) for spec in layer.rootPrims]
    except Exception:
        LOGGER.exception("could not open the referenced layer %s", usd_path)
    return out


def _reference_asset_paths(usd: Any, root_prim: Any) -> list[str]:
    """Every reference/payload asset path authored on prims under
    ``root_prim`` (OQ-4: which hosts the composed asset actually pulls from).
    Best-effort - metadata shapes vary across USD versions, so failures
    yield an empty list rather than breaking the attach."""
    paths: list[str] = []
    for prim in _prim_range(usd, root_prim):
        for key in ("references", "payload"):
            try:
                list_op = prim.GetMetadata(key)
                items = list_op.GetAddedOrExplicitItems() if list_op is not None else []
            except Exception:
                items = []
            for item in items:
                asset_path = getattr(item, "assetPath", "")
                if asset_path:
                    paths.append(str(asset_path))
    return paths


def _version_string(version: tuple[int, int, int] | None) -> str | None:
    return None if version is None else ".".join(str(part) for part in version)


def _local_ip() -> str:
    """Best-effort primary local IP (no traffic is actually sent)."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return ""


def _prim_name(name: str) -> str:
    """Component names may contain characters USD prim names can't."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


# ======================================================================
# Handles - the interface component models talk to. All public methods are
# safe to call from any thread.
# ======================================================================


def resolve_joint_indices(
    dof_names: Sequence[str], joint_names: Sequence[str] | None
) -> list[int] | None:
    """Map an asset's declared arm joint names onto the articulation's PhysX
    dof order, by name rather than position (FINDINGS ARM-1; R-3: attaching a
    gripper later can add/reorder dofs, so a positional slice would silently
    pick up the wrong joints). Returns None when the asset declares no joint
    names, meaning "all dofs, in PhysX order"."""
    if joint_names is None:
        return None
    indices = []
    missing = []
    for name in joint_names:
        try:
            indices.append(dof_names.index(name))
        except ValueError:
            missing.append(name)
    if missing:
        raise ValueError(
            f"joint(s) not found in articulation: {missing}; actual dof_names: {list(dof_names)}"
        )
    return indices


# ARM-12 (FINDINGS R-7): the two "arrived" notions - velocity-based IsMoving
# and position-based move completion - share these constants and one
# predicate on every backend.
# |joint velocity| at or below this counts as still. A 12-DOF PhysX articulation
# at 120 Hz idles with ~1e-3 rad/s of residual jitter (GPU run 11), so 1e-3
# never settled; 1e-2 rad/s (0.6 deg/s) is still far below any real motion.
VEL_EPS_RAD_S = 1e-2
SETTLE_TOL_RAD = math.radians(0.5)  # commanded-vs-measured gap that counts as arrived
# Joint-space moves are executed as a time-synchronized straight line: every
# joint's target advances so that all joints arrive together, the longest
# travel at this speed unless the move caps it lower. Handing PhysX the final
# targets directly let each joint run at its own speed, and the Cartesian
# path between two planner waypoints became an arc: on the pick cell the
# fingertips swept through the block during the approach (2026-09-04).
# 120 deg/s let the fingertips graze a block 12 mm away during a linear descent
# (block nudged 2 mm and a few degrees; grasp closed on nothing, 2026-09-04);
# 45 deg/s keeps a 335 mm transit under 4 s and the pads clear.
SYNC_JOINT_VEL_RAD_S = math.radians(45.0)
# While a path is in flight the commanded target never runs ahead of the
# measured joints by more than this: the path advances only as fast as the
# drives track it, so the tool stays on the planner's line (3 deg of lead at
# this reach was ~30 mm of deviation and nudged the block before a grasp,
# 2026-09-04). A paused path whose arm makes no progress toward the target
# for STALL_NO_PROGRESS_STEPS is a stall (contact).
PATH_LAG_TOL_RAD = SETTLE_TOL_RAD
SETTLE_WINDOW_STEPS = 5  # consecutive physics steps the predicate must hold
# A blocked arm under contact vibrates above VEL_EPS_RAD_S and never reads
# still (GPU run 15: 30 s of pushing into a block), so a stall is also
# declared when the worst joint error stops improving for this many steps
# (1 s at physics_dt 1/120) by at least STALL_PROGRESS_EPS_RAD.
STALL_NO_PROGRESS_STEPS = 120
STALL_PROGRESS_EPS_RAD = math.radians(0.1)


class SettleOutcome(str, Enum):
    """Result of ArmHandle.wait_for_settle (ARM-12 / ARM-13)."""

    REACHED = "reached"  # every named joint within tolerance for SETTLE_WINDOW_STEPS steps
    STALLED = "stalled"  # still (|v| <= VEL_EPS_RAD_S) for the window with a joint off target
    TIMED_OUT = "timed_out"  # the sim-time deadline passed with joints still converging


# ----------------------------------------------------------------------
# world handle - the seam the world component drives the scene through
# (FINDINGS SCN-16; phase 4). IsaacWorldHandle wraps the SimManager the
# real sim runs behind; MockWorldHandle keeps a plain-python registry so
# every scene behaviour is unit-testable without Isaac Sim (SCN-8).
#
# Units at this seam: metres and (w,x,y,z) quaternions, world frame. The
# Viam edge (models/world.py) owns mm and orientation-vector conversion.
# ----------------------------------------------------------------------

# randomize_props separation default: FINDINGS W26 block spacing rule
DEFAULT_MIN_SEPARATION_M = 0.15
# sized props (dynamic-blocks phase 1): pairwise face gap beyond the two
# footprints, and how far above the support face a placed prop rests
PROP_EDGE_CLEARANCE_M = 0.01
PROP_REST_EPSILON_M = 0.0005
RANDOMIZE_MAX_ATTEMPTS = 100  # per prop, within one layout attempt
# a dense-but-feasible request can strand the LAST prop no matter how many
# single-prop draws it gets (GPU run 8, seed 6) - redraw the whole layout
RANDOMIZE_LAYOUT_RESTARTS = 50


class PropGeometry(NamedTuple):
    """One prop's oriented box at its CURRENT world pose (SCN-5).

    ``box_dims_m`` is the full edge length per axis: exact for cube props
    (size x scale); a usd prop reports its optional ``box_dims`` config
    attr, else (0, 0, 0) = unknown.
    """

    name: str
    box_dims_m: tuple[float, float, float]
    position_m: Vec3
    orientation_wxyz: Quat
    color: tuple[float, float, float] | None
    fixed: bool


class RandomizeResult(NamedTuple):
    """What randomize_props changed: each named prop's new centre position
    and its full box dims after the call (freshly drawn where a size range
    covered it, its current dims otherwise)."""

    positions_m: dict[str, Vec3]
    dims_m: dict[str, tuple[float, float, float]]


def prop_spawn_orientation(prop: dict[str, Any]) -> Quat:
    """The (w,x,y,z) quaternion a prop spawns with (SCN-2).

    ``orientation_wxyz`` wins over ``orientation_rpy_deg`` (extrinsic
    x-y-z euler, degrees); identity when neither is set.
    """
    if prop.get("orientation_wxyz") is not None:
        return _as_quat(prop["orientation_wxyz"])
    rpy = prop.get("orientation_rpy_deg")
    if rpy is not None:
        roll, pitch, yaw = (float(v) for v in rpy)
        return quat_from_euler_deg(roll, pitch, yaw)
    return (1.0, 0.0, 0.0, 0.0)


def prop_box_dims(prop: dict[str, Any]) -> tuple[float, float, float]:
    """Full edge lengths (m) of a prop's box: cube = size x scale
    (defaults 0.05 and [1, 1, 1]); usd = its ``box_dims`` attr else zeros
    (= unknown; the README obstacle recipe tells users to set it)."""
    if str(prop.get("type", "cube")) == "usd":
        dims = prop.get("box_dims")
        if dims is None:
            return (0.0, 0.0, 0.0)
        return (float(dims[0]), float(dims[1]), float(dims[2]))
    size = float(prop.get("size", 0.05))
    scale = prop.get("scale") or (1.0, 1.0, 1.0)
    return (size * float(scale[0]), size * float(scale[1]), size * float(scale[2]))


def _prop_footprint_m(dims: tuple[float, float, float]) -> float:
    """A prop's placement footprint (SCN-16 sized props): the larger of
    its x/y edges, used for edge-aware separation."""
    return max(dims[0], dims[1])


def _place_props(
    dims_by_name: dict[str, tuple[float, float, float]],
    region: tuple[Vec3, Vec3],
    rng: random.Random,
    min_separation_m: float,
) -> dict[str, Vec3]:
    """The draw loop behind ``sample_prop_positions``, taking an
    already-seeded ``rng`` so a caller can consume size draws from the
    same stream first (sized props, dynamic-blocks phase 1)."""
    (x0, y0, z0), (x1, y1, z1) = region
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    face_z = (float(z0) + float(z1)) / 2.0
    for name, dims in dims_by_name.items():
        half_x, half_y = dims[0] / 2.0, dims[1] / 2.0
        if lo_x + half_x > hi_x - half_x or lo_y + half_y > hi_y - half_y:
            raise ValueError(f"randomize_props: region cannot hold {name!r}'s footprint")
    for _restart in range(RANDOMIZE_LAYOUT_RESTARTS):
        placed: dict[str, Vec3] = {}
        footprints: dict[str, float] = {}
        for name, dims in dims_by_name.items():
            half_x, half_y = dims[0] / 2.0, dims[1] / 2.0
            footprint = _prop_footprint_m(dims)
            for _ in range(RANDOMIZE_MAX_ATTEMPTS):
                x = rng.uniform(lo_x + half_x, hi_x - half_x)
                y = rng.uniform(lo_y + half_y, hi_y - half_y)
                if all(
                    math.hypot(x - px, y - py)
                    >= max(
                        min_separation_m,
                        (footprint + footprints[pname]) / 2.0 + PROP_EDGE_CLEARANCE_M,
                    )
                    for pname, (px, py, _pz) in placed.items()
                ):
                    placed[name] = (x, y, face_z + dims[2] / 2.0 + PROP_REST_EPSILON_M)
                    footprints[name] = footprint
                    break
            else:
                break  # this layout stranded ``name``: redraw everything
        else:
            return placed
    raise ValueError(
        f"randomize_props: no layout for {sorted(dims_by_name)} after "
        f"{RANDOMIZE_LAYOUT_RESTARTS} layout attempts; widen the region, "
        "drop props, or lower min_separation"
    )


def sample_prop_positions(
    dims_by_name: dict[str, tuple[float, float, float]],
    region: tuple[Vec3, Vec3],
    seed: int,
    min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
) -> dict[str, Vec3]:
    """Deterministic prop placement on a table's top face (SCN-7).

    ``region`` is ((x0, y0, z), (x1, y1, z)) in metres, world frame: the
    rectangle of the top face the props' footprints must stay inside, at
    the face's height z (the two z values are averaged). Each prop lands
    with its footprint (centre +/- dims/2 in x and y) inside the
    rectangle, its centre at least max(``min_separation_m``, the two
    props' edge-aware gap) from every other placed centre in the x/y
    plane (FINDINGS W26; sized props: edge-aware = (footprint_a +
    footprint_b) / 2 + PROP_EDGE_CLEARANCE_M, footprint = max x/y dim),
    and its centre z at face z + dims_z / 2 + PROP_REST_EPSILON_M so it
    rests just above the face.

    Same inputs -> the same placements on every call: draws come from one
    ``random.Random(seed)`` stream and props place in ``dims_by_name``
    insertion order. A layout that strands a prop (no clear spot within
    RANDOMIZE_MAX_ATTEMPTS draws) is redrawn wholesale, up to
    RANDOMIZE_LAYOUT_RESTARTS times, so a dense-but-feasible request still
    converges. Raises ValueError when the region cannot hold a footprint
    or no layout fits.
    """
    return _place_props(dims_by_name, region, random.Random(seed), min_separation_m)


def _validate_size_range_names(
    names: list[str], size_range_m: dict[str, tuple[float, float]] | None
) -> None:
    if not size_range_m:
        return
    unknown = set(size_range_m) - set(names)
    if unknown:
        raise ValueError(f"randomize_props: size_range_m names not in names: {sorted(unknown)}")


def _require_cube_prop(name: str, spec: dict[str, Any]) -> None:
    if str(spec.get("type", "cube")) != "cube":
        raise ValueError(f"randomize_props: size_range_m on non-cube prop {name!r}")


def _draw_sizes_and_positions(
    names: list[str],
    dims_by_name: dict[str, tuple[float, float, float]],
    region: tuple[Vec3, Vec3],
    seed: int,
    min_separation_m: float,
    size_range_m: dict[str, tuple[float, float]] | None,
) -> tuple[dict[str, Vec3], dict[str, tuple[float, float, float]]]:
    """One ``random.Random(seed)`` stream: sizes first (``names`` order,
    only props with a range), then positions (WorldHandle.randomize_props
    contract). Returns the placements and every named prop's post-draw
    dims."""
    rng = random.Random(seed)
    drawn_dims = dict(dims_by_name)
    if size_range_m:
        for name in names:
            size_range = size_range_m.get(name)
            if size_range is None:
                continue
            lo, hi = size_range
            drawn_edge = rng.uniform(lo, hi)
            drawn_dims[name] = (drawn_edge, drawn_edge, drawn_edge)
    placed = _place_props({name: drawn_dims[name] for name in names}, region, rng, min_separation_m)
    return placed, drawn_dims


class WorldHandle:
    """The seam every world-component verb drives the sim through
    (SCN-16). models/world.py talks to this interface only; SimManager
    stays an implementation detail behind it.

    Poses cross this seam in metres and (w,x,y,z) world-frame
    quaternions. Prims added via ``add_usd`` are stage furniture, not
    registered props: they never appear in ``prop_geometries`` and no
    pose verb can move them (spawn a ``type: "usd"`` prop for that).
    """

    def status(self) -> dict[str, Any]:
        raise NotImplementedError

    def play(self) -> None:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def reset(self, soft: bool = False) -> None:
        """Full reset (soft=False): the phase-1 chokepoint - the sim
        world resets (every prop returns to its configured spawn pose)
        and every post-reset hook replays (gain snapshots, camera
        re-inits). Soft reset (soft=True): pose-only - every registered
        prop returns to its spawn pose with velocities zeroed;
        articulations, hooks and gains are left alone (SCN-7)."""
        raise NotImplementedError

    def add_usd(
        self,
        usd_path: str,
        prim_path: str,
        position_m: Vec3,
        orientation_wxyz: Quat | None = None,
    ) -> None:
        """Reference a USD file into the stage at ``prim_path`` (SCN-2:
        orientation applies here too)."""
        raise NotImplementedError

    def prop_geometries(self) -> list[PropGeometry]:
        """One entry per registered prop (configured or runtime-spawned),
        at its current world pose (SCN-5)."""
        raise NotImplementedError

    def set_prop_pose(
        self, name: str, position_m: Vec3, orientation_wxyz: Quat | None = None
    ) -> None:
        """Teleport prop ``name`` (orientation kept when None) and zero
        its velocities. Never rewrites the prop's spawn/default state: a
        later reset() still restores the configured pose (mock gate).
        Unknown name -> ValueError."""
        raise NotImplementedError

    def randomize_props(
        self,
        names: list[str],
        region: tuple[Vec3, Vec3],
        seed: int,
        min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
        size_range_m: dict[str, tuple[float, float]] | None = None,
    ) -> RandomizeResult:
        """Place ``names`` (in list order) deterministically and teleport
        each one there with set_prop_pose semantics, optionally redrawing
        sizes first.

        ``size_range_m`` maps a prop name to (lo, hi) full edge length in
        metres: that prop's new size draws as one uniform(lo, hi) scalar
        applied to all three axes. Keys must name cube props in ``names``
        (ValueError otherwise). Sizes and positions come from one
        ``random.Random(seed)`` stream - sizes first, in ``names`` order,
        then positions - so a seed reproduces both, and a call with no
        ranges consumes no size draws (existing seeded layouts are
        unchanged). Rescaling is absolute against the spawn size
        (scale = drawn / spawn edge), so repeated draws never accumulate,
        and the new dims persist through reset (reset restores poses,
        never sizes). prop_geometries serves the post-draw dims afterward.

        A sized call (any range given) replays spawn_prop's full-reset
        pattern before the teleports: rescaling a live rigid body
        invalidates PhysX's tensor views, so the world resets (every prop
        snaps to its spawn pose, post-reset hooks fire) and then the named
        props teleport to their sampled positions. A call with no ranges
        never resets.

        Placement uses the post-draw dims: the centres of props a and b
        stay at least max(min_separation_m, (footprint_a + footprint_b)/2
        + PROP_EDGE_CLEARANCE_M) apart in x/y, where a prop's footprint is
        the max of its x/y dims, and each prop rests at
        face z + dims_z / 2 + PROP_REST_EPSILON_M.
        Unknown name -> ValueError."""
        raise NotImplementedError

    def spawn_prop(self, prop: dict[str, Any]) -> None:
        """Add a prop at runtime (SCN-11): same schema as the world's
        ``props`` config attr (validated at the Viam edge; name
        uniqueness re-checked here -> ValueError). On the real sim this
        replays the component-spawn path - spawn on the sim thread, then
        a full world reset (hooks fire; earlier teleports snap back to
        spawn poses) - so spawn before randomizing. The prop registers
        like a configured one: it appears in prop_geometries and
        survives reset."""
        raise NotImplementedError


class MockWorldHandle(WorldHandle):
    """Plain-python scene registry (SCN-8): every scene behaviour above
    is testable without Isaac Sim. Registry entries keep the spawn attrs
    and both spawn and current poses."""

    def __init__(self, sim: "SimManager", props: Sequence[dict[str, Any]]) -> None:
        self._sim = sim
        self._registry: dict[str, dict[str, Any]] = {}
        for prop in props:
            self._register(prop)

    def _register(self, prop: dict[str, Any]) -> None:
        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        name = _prim_name(str(prop["name"]))
        if name in self._registry:
            raise ValueError(f"prop {name!r} already exists")
        position = to_vec3(prop.get("position"))
        orientation = prop_spawn_orientation(prop)
        self._registry[name] = {
            "spawn": dict(prop),
            "spawn_position": position,
            "spawn_orientation": orientation,
            "position": position,
            "orientation": orientation,
        }

    def registry(self) -> dict[str, dict[str, Any]]:
        """The live registry, keyed by prim name (SCN-8; tests read it)."""
        return self._registry

    def _entry(self, name: str) -> dict[str, Any]:
        entry = self._registry.get(_prim_name(name))
        if entry is None:
            raise ValueError(f"unknown prop {name!r}; have {sorted(self._registry)}")
        return entry

    def status(self) -> dict[str, Any]:
        return self._sim.status()

    def play(self) -> None:
        self._sim.play()

    def pause(self) -> None:
        self._sim.pause()

    def reset(self, soft: bool = False) -> None:
        for entry in self._registry.values():
            entry["position"] = entry["spawn_position"]
            entry["orientation"] = entry["spawn_orientation"]
        if not soft:
            self._sim.reset()

    def add_usd(
        self,
        usd_path: str,
        prim_path: str,
        position_m: Vec3,
        orientation_wxyz: Quat | None = None,
    ) -> None:
        self._sim.add_usd_reference(usd_path, prim_path, position_m)

    def prop_geometries(self) -> list[PropGeometry]:
        out: list[PropGeometry] = []
        for name, entry in self._registry.items():
            spawn = entry["spawn"]
            color = spawn.get("color")
            out.append(
                PropGeometry(
                    name=name,
                    box_dims_m=prop_box_dims(spawn),
                    position_m=entry["position"],
                    orientation_wxyz=entry["orientation"],
                    color=(float(color[0]), float(color[1]), float(color[2]))
                    if color is not None
                    else None,
                    fixed=bool(spawn.get("fixed", False)),
                )
            )
        return out

    def set_prop_pose(
        self, name: str, position_m: Vec3, orientation_wxyz: Quat | None = None
    ) -> None:
        entry = self._entry(name)
        entry["position"] = to_vec3(position_m)
        if orientation_wxyz is not None:
            entry["orientation"] = _as_quat(orientation_wxyz)

    def randomize_props(
        self,
        names: list[str],
        region: tuple[Vec3, Vec3],
        seed: int,
        min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
        size_range_m: dict[str, tuple[float, float]] | None = None,
    ) -> RandomizeResult:
        _validate_size_range_names(names, size_range_m)
        entries = {name: self._entry(name) for name in names}
        if size_range_m:
            for name in size_range_m:
                _require_cube_prop(name, entries[name]["spawn"])
        dims = {name: prop_box_dims(entries[name]["spawn"]) for name in names}
        placed, drawn_dims = _draw_sizes_and_positions(
            names, dims, region, seed, min_separation_m, size_range_m
        )
        for name in size_range_m or ():
            drawn_edge = drawn_dims[name][0]
            entries[name]["spawn"] = {
                **entries[name]["spawn"],
                "size": drawn_edge,
                "scale": (1.0, 1.0, 1.0),
            }
        if size_range_m:
            # parity with IsaacWorldHandle: a sized randomize replays
            # spawn_prop's full-reset pattern (all props snap to spawn poses,
            # hooks fire) before the named props teleport to their draws
            for entry in self._registry.values():
                entry["position"] = entry["spawn_position"]
                entry["orientation"] = entry["spawn_orientation"]
            self._sim._reset_world()
        for name in names:
            self.set_prop_pose(name, placed[name])
        dims_m = {name: prop_box_dims(entries[name]["spawn"]) for name in names}
        return RandomizeResult(positions_m=placed, dims_m=dims_m)

    def spawn_prop(self, prop: dict[str, Any]) -> None:
        self._register(prop)


def _zero_prop_velocity(sim: "SimManager", prim_path: str) -> None:
    """Best-effort velocity zeroing after a teleport (SCN-7). Fixed props
    have no rigid-body API to zero (and never move on their own), and the
    exact RigidPrim import path is unverified against real Kit (isaacsim
    vs omni.isaac naming, per compat.py's pattern) - both are swallowed so
    a missing/incompatible API can't block the teleport itself."""
    try:
        try:
            from isaacsim.core.prims import SingleRigidPrim as RigidPrim
        except ImportError:
            from omni.isaac.core.prims import RigidPrim
        rigid = RigidPrim(prim_path)
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))
    except Exception:
        pass


class IsaacWorldHandle(WorldHandle):
    """Drives the real sim by wrapping SimManager (SCN-16). Scene
    mutations run on the sim thread via SimManager.run."""

    def __init__(self, sim: "SimManager") -> None:
        self._sim = sim

    def status(self) -> dict[str, Any]:
        return self._sim.status()

    def play(self) -> None:
        self._sim.play()

    def pause(self) -> None:
        self._sim.pause()

    def reset(self, soft: bool = False) -> None:
        if not soft:
            self._sim.reset()
            return

        def _restore() -> None:
            for name, spec in self._sim._prop_specs.items():
                self._teleport(name, spec["position"], spec.get("spawn_orientation"))

        self._sim.run(_restore)

    def add_usd(
        self,
        usd_path: str,
        prim_path: str,
        position_m: Vec3,
        orientation_wxyz: Quat | None = None,
    ) -> None:
        self._sim.add_usd_reference(usd_path, prim_path, position_m, orientation_wxyz)

    def prop_geometries(self) -> list[PropGeometry]:
        def _read() -> list[PropGeometry]:
            out: list[PropGeometry] = []
            for name, spec in self._sim._prop_specs.items():
                pos, quat = self._sim._isaac.SingleXFormPrim(f"/World/{name}").get_world_pose()
                color = spec.get("color")
                out.append(
                    PropGeometry(
                        name=name,
                        box_dims_m=prop_box_dims(spec),
                        position_m=(float(pos[0]), float(pos[1]), float(pos[2])),
                        orientation_wxyz=_as_quat(quat),
                        color=(float(color[0]), float(color[1]), float(color[2]))
                        if color is not None
                        else None,
                        fixed=bool(spec.get("fixed", False)),
                    )
                )
            return out

        return self._sim.run(_read)

    def _prop_spec(self, name: str) -> dict[str, Any]:
        spec = self._sim._prop_specs.get(_prim_name(name))
        if spec is None:
            raise ValueError(f"unknown prop {name!r}; have {sorted(self._sim._prop_specs)}")
        return spec

    def _teleport(self, name: str, position_m: Vec3, orientation_wxyz: Quat | None) -> None:
        """Runs on the sim thread: teleport + velocity zeroing. Never touches
        ``_prop_specs`` (mock gate: a later reset must still restore the
        configured spawn pose). Cube props teleport through their scene
        object - the API PhysX tracks; a raw-XForm teleport of a live rigid
        body desyncs it and the prop can tumble (GPU run 3). usd props are
        not scene-registered, so they keep the raw-XForm fallback."""
        prim_path = f"/World/{name}"
        kwargs: dict[str, Any] = {"position": list(position_m)}
        if orientation_wxyz is not None:
            kwargs["orientation"] = list(orientation_wxyz)
        scene_object = self._sim.world.scene.get_object(name)
        if scene_object is not None:
            scene_object.set_world_pose(**kwargs)
            for setter_name in ("set_linear_velocity", "set_angular_velocity"):
                setter = getattr(scene_object, setter_name, None)
                if setter is None:
                    continue
                try:
                    setter(np.zeros(3))
                except Exception:  # fixed props have no rigid-body velocity
                    pass
            return
        self._sim._isaac.SingleXFormPrim(prim_path).set_world_pose(**kwargs)
        _zero_prop_velocity(self._sim, prim_path)

    def set_prop_pose(
        self, name: str, position_m: Vec3, orientation_wxyz: Quat | None = None
    ) -> None:
        self._prop_spec(name)  # ValueError on unknown name
        prim_name = _prim_name(name)
        self._sim.run(lambda: self._teleport(prim_name, position_m, orientation_wxyz))

    def randomize_props(
        self,
        names: list[str],
        region: tuple[Vec3, Vec3],
        seed: int,
        min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
        size_range_m: dict[str, tuple[float, float]] | None = None,
    ) -> RandomizeResult:
        _validate_size_range_names(names, size_range_m)
        specs = {name: self._prop_spec(name) for name in names}
        if size_range_m:
            for name in size_range_m:
                _require_cube_prop(name, specs[name])
        dims = {name: prop_box_dims(specs[name]) for name in names}
        placed, drawn_dims = _draw_sizes_and_positions(
            names, dims, region, seed, min_separation_m, size_range_m
        )
        if size_range_m:

            def _rescale_and_rebuild() -> None:
                # stop BEFORE writing the scale: the stop inside reset restores
                # the stage's pre-play state, so a scale authored mid-play is
                # reverted (GPU: drawn 44.7 mm, block stayed 60 mm) - and a
                # live-play write also invalidates PhysX's tensor view (GPU:
                # "Failed to get rigid body transforms from backend")
                self._sim.world.stop()
                self._rescale_props(specs, drawn_dims, size_range_m)
                # spawn_prop's pattern: full reset re-cooks the colliders at
                # the new scale and refires the post-reset hooks (arm gains,
                # camera re-inits) before any teleport touches a pose
                self._sim._reset_world()

            self._sim.run(_rescale_and_rebuild)
        for name, position in placed.items():
            self.set_prop_pose(name, position)
        dims_m = {name: prop_box_dims(specs[name]) for name in names}
        return RandomizeResult(positions_m=placed, dims_m=dims_m)

    def _rescale_props(
        self,
        specs: dict[str, dict[str, Any]],
        drawn_dims: dict[str, tuple[float, float, float]],
        size_range_m: dict[str, tuple[float, float]],
    ) -> None:
        """Runs on the sim thread: absolute rescale against the spawn
        size (never the previous draw), so repeated randomize calls don't
        compound (dynamic-blocks phase 1)."""
        for name in size_range_m:
            spec = specs[name]
            spawn_size = float(spec.get("size", 0.05))
            scale_factor = drawn_dims[name][0] / spawn_size
            scale = (scale_factor, scale_factor, scale_factor)
            scene_object = self._sim.world.scene.get_object(name)
            if scene_object is not None:
                set_local_scale = getattr(scene_object, "set_local_scale", None)
                if set_local_scale is not None:
                    set_local_scale(np.array(scale))
            spec["scale"] = scale

    def spawn_prop(self, prop: dict[str, Any]) -> None:
        if not prop.get("name"):
            raise ValueError(f"every prop needs a name: {prop}")
        name = _prim_name(str(prop["name"]))
        if name in self._sim._prop_specs:
            raise ValueError(f"prop {name!r} already exists")

        def _spawn() -> None:
            self._sim._spawn_prop(prop)
            self._sim._reset_world()

        self._sim.run(_spawn)


class ArmHandle:
    def dof_names(self) -> list[str]:
        """Names of the arm's named joints, in the asset's declared order
        (all DOFs, in PhysX order, when the asset declares none)."""
        raise NotImplementedError

    def all_dof_names(self) -> list[str]:
        """Every DOF of the articulation in PhysX order - the arm's joints
        plus anything attached under it (a gripper); the GPU checklist's
        `len == 12` (OQ-5). The mock includes its padding dofs."""
        raise NotImplementedError

    def joint_state(self) -> list[dict[str, Any]]:
        """Per DOF of the whole articulation, in PhysX order: ``name``,
        ``position`` and ``velocity`` (rad, rad/s), the drive ``target`` the
        physics is actually holding (rad, None when unreadable) and ``named``
        (True for the arm's own joints). The diagnostic that separates "wrong
        target" from "physics fought the target" (GPU run 16)."""
        raise NotImplementedError

    def get_joint_positions(self) -> list[float]:  # radians
        """Positions of the arm's named joints, in the asset's declared
        order (all DOFs when the asset declares none)."""
        raise NotImplementedError

    def set_joint_targets(self, positions: list[float], max_vel_rad_s: float | None = None) -> None:
        """Targets for the arm's named joints, in the asset's declared
        order (all DOFs when the asset declares none). ``max_vel_rad_s`` caps
        every named joint's speed for this move (ARM-13: MoveOptions
        max_vel_degs_per_sec, converted by the model); None = the drive's
        own limit."""
        raise NotImplementedError

    def follow_joint_path(
        self, waypoints: list[list[float]], max_vel_rad_s: float | None = None
    ) -> None:
        """Execute a planned trajectory as ONE continuous piecewise-linear
        joint path through ``waypoints`` (the motion service's plan), all
        joints synchronized within each segment, without settling at
        intermediate waypoints. The last waypoint becomes the settle target.
        Settling at every waypoint cost 28 s for a 335 mm linear move whose
        plan itself took 0.7 s (2026-09-04). Default: the last waypoint only."""
        if waypoints:
            self.set_joint_targets(waypoints[-1], max_vel_rad_s)

    def path_progress(self) -> tuple[int, int] | None:
        """(segments completed, segments total) of the path in flight, or None."""
        return None

    def path_trace(self) -> list[dict[str, Any]]:
        """Debug: one entry per physics step of the last path: commanded and
        measured joint angles (deg), their worst gap, and the end effector's
        measured world position (mm). The mock returns []."""
        return []

    def is_moving(self) -> bool:
        """True while any named joint's |velocity| > VEL_EPS_RAD_S OR any
        |target - measured| > SETTLE_TOL_RAD (ARM-12). A stalled arm that
        never reached its target therefore keeps reporting True."""
        raise NotImplementedError

    def wait_for_settle(
        self, timeout_s: float, tolerance_rad: float = SETTLE_TOL_RAD
    ) -> SettleOutcome:
        """Block the calling (non-sim) thread until the last commanded targets
        are REACHED (within ``tolerance_rad`` for SETTLE_WINDOW_STEPS
        consecutive steps), the arm STALLED (still for the window, outside
        tolerance), or ``timeout_s`` of SIM time has elapsed (TIMED_OUT).
        Backends: Isaac via a physics-step callback + threading.Event, the
        mock via its interpolation clock. The model never polls wall clock."""
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def post_reset(self) -> None:
        """XC-5 hook: re-apply anything a world.reset() undoes (solver
        iteration count, controller gains) and re-command the last targets,
        so a reset mid-move holds position instead of teleporting to the
        default pose (ARM-15/ARM-16). No-op by default (the mock has no such
        state)."""
        return None

    def release(self) -> None:
        """Called by SimManager.release_handle when the owning component
        closes (XC-4). Isaac backends drop the world.scene registry entry
        (registry_only - the prim stays) and any physics callbacks, so a
        later create_arm for the same name can re-attach."""
        return None

    def get_end_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """((x,y,z) meters, (w,x,y,z) quaternion) of the end effector, in
        Viam's arm frame - the Isaac root un-rotated by the asset's
        base_frame_correction, if any (FINDINGS ARM-10)."""
        raise NotImplementedError

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        """((x,y,z) meters, (w,x,y,z) quaternion) world pose of an arbitrary
        prim on the stage (FINDINGS XC-1 GPU acceptance)."""
        raise NotImplementedError


class IsaacArmHandle(ArmHandle):
    def __init__(
        self,
        sim: SimManager,
        articulation,
        ee_prim,
        joint_names: Sequence[str] | None = None,
        base_correction: Quat = (1.0, 0.0, 0.0, 0.0),
        prim_path: str = "",
    ) -> None:
        self._sim = sim
        self._art = articulation
        self._ee = ee_prim
        self._joint_names = joint_names
        self._base_correction: Quat = base_correction
        self._prim_path = prim_path
        self._dof_names: list[str] = list(articulation.dof_names)
        LOGGER.info(
            "arm %r articulation dof_names: %s",
            getattr(articulation, "name", ""),
            self._dof_names,
        )
        self._joint_indices: list[int] | None = resolve_joint_indices(self._dof_names, joint_names)
        # last commanded targets for the named joints (ARM-12's second term)
        self._targets: list[float] | None = None
        # ARM-13: the previous per-joint velocity cap, restored when a move
        # arrives with max_vel_rad_s=None after one that set it.
        self._saved_max_joint_velocities: Any | None = None
        # ARM-15/ARM-16: snapshotted by create_arm's factory, re-applied by
        # post_reset after a world.reset() undoes them.
        self._solver_iterations: int | None = None
        self._gains: Any | None = None
        # the in-flight wait_for_settle's event/outcome, so stop() can
        # short-circuit it (None when no wait is in flight).
        self._active_settle: dict[str, Any] | None = None
        self._active_settle_lock = threading.Lock()
        # in-flight synchronized interpolation (sim thread only)
        self._interp: dict[str, Any] | None = None
        # debug: per-physics-step trace of the last path (path_trace())
        self._trace: list[dict[str, Any]] = []

    def refresh_dofs(self) -> None:
        """Re-read dof_names and re-resolve the named-joint indices after the
        articulation's topology changed - a gripper was attached under it
        (ARM-2; R-3). Raises ValueError if the named joints are no longer all
        present."""
        previous_count = len(self._dof_names)
        if not getattr(self._art, "handles_initialized", True):
            # a wrapper registered after the world's first reset may not have
            # been initialized by the scene yet
            self._art.initialize()
        names_now = list(self._art.dof_names)
        initialize = getattr(self._art, "initialize", None)
        if len(names_now) != previous_count and initialize is not None:
            # the wrapper captured its default joint state before the topology
            # changed (GPU run 19: "setDofActuationForces expected 12, received
            # 6" at every later reset); re-initialize so it is re-captured
            initialize()
            names_now = list(self._art.dof_names)
        self._dof_names = names_now
        LOGGER.info("arm %r articulation dof_names now: %s", self._art.name, self._dof_names)
        self._joint_indices = resolve_joint_indices(self._dof_names, self._joint_names)
        if self._gains is None:
            # a fresh wrapper (replace_articulation) has no snapshot yet: take
            # it from the new topology and re-apply the solver count (ARM-15/16)
            if self._solver_iterations is not None:
                set_iterations = getattr(self._art, "set_solver_position_iteration_count", None)
                if set_iterations is not None:
                    set_iterations(self._solver_iterations)
            self._gains = self._art.get_articulation_controller().get_gains()

    def replace_articulation(self, articulation: Any) -> None:
        """Swap in a fresh SingleArticulation wrapper before a reset that
        changes the articulation's topology (a gripper joining it). The old
        wrapper's default joint state and gains snapshot are sized for the old
        DOF count, and the scene's post_reset would push them into the new
        articulation and fail ("Failed to set DOF actuation forces"). The
        gains snapshot is dropped here and retaken by refresh_dofs()."""
        self._art = articulation
        self._gains = None

    def all_dof_names(self) -> list[str]:
        return list(self._dof_names)

    def joint_state(self) -> list[dict[str, Any]]:
        def _state() -> list[dict[str, Any]]:
            positions = [float(v) for v in self._art.get_joint_positions()]
            velocities = [float(v) for v in self._art.get_joint_velocities()]
            targets: list[float | None] = [None] * len(positions)
            view = getattr(self._art, "_articulation_view", None)
            try:
                applied = view.get_applied_actions() if view is not None else None
                if applied is not None and applied.joint_positions is not None:
                    row = applied.joint_positions
                    row = row[0] if len(getattr(row, "shape", ())) == 2 else row
                    targets = [float(v) for v in row]
            except Exception:
                LOGGER.exception("could not read the applied drive targets")
            named = set(self._joint_indices or range(len(positions)))
            return [
                {
                    "name": name,
                    "position": positions[i],
                    "velocity": velocities[i],
                    "target": targets[i],
                    "named": i in named,
                }
                for i, name in enumerate(self._dof_names)
            ]

        return self._sim.run(_state)

    def dof_names(self) -> list[str]:
        if self._joint_indices is None:
            return list(self._dof_names)
        return [self._dof_names[i] for i in self._joint_indices]

    def get_joint_positions(self) -> list[float]:
        def _get():
            positions = self._art.get_joint_positions(joint_indices=self._joint_indices)
            return [float(v) for v in positions]

        return self._sim.run(_get)

    def set_joint_targets(self, positions: list[float], max_vel_rad_s: float | None = None) -> None:
        self.follow_joint_path([list(positions)], max_vel_rad_s)

    def follow_joint_path(
        self, waypoints: list[list[float]], max_vel_rad_s: float | None = None
    ) -> None:
        if not waypoints:
            return

        def _apply():
            self._apply_velocity_cap(max_vel_rad_s)
            speed = max_vel_rad_s if max_vel_rad_s else self._sync_speed_rad_s()
            current = [
                float(p) for p in self._art.get_joint_positions(joint_indices=self._joint_indices)
            ]
            segments = []
            start = current
            for wp in waypoints:
                goal = [float(v) for v in wp]
                travel = max((abs(g - s) for g, s in zip(goal, start, strict=True)), default=0.0)
                # keep EVERY waypoint, however small its travel: a linear plan's
                # waypoints are closer together than the settle tolerance, and
                # dropping them collapsed the whole path into one direct jump to
                # the last waypoint (an unsynchronized arc that swung the tool
                # ~15 mm off a straight-up retreat and dragged the released
                # block, 2026-09-05)
                if travel > 0.0 and speed and speed > 0.0:
                    segments.append({"start": start, "goal": goal, "duration": travel / speed})
                start = goal
            self._targets = start
            if not segments:
                self._interp = None
                self._apply_joint_positions(start)
                return
            self._interp = {
                "segments": segments,
                "index": 0,
                "elapsed": 0.0,
                "commanded": current,
                "lag_steps": 0,
                "stalled": False,
            }
            self._trace = []
            self._ensure_interp_callback()
            self._apply_joint_positions(current)

        self._sim.run(_apply)

    def path_progress(self) -> tuple[int, int] | None:
        interp = self._interp
        if interp is None:
            return None
        return int(interp["index"]), len(interp["segments"])

    def path_trace(self) -> list[dict[str, Any]]:
        return self._sim.run(lambda: list(self._trace))

    def _apply_joint_positions(self, positions: list[float]) -> None:
        import numpy as np

        self._art.apply_action(
            self._sim._isaac.ArticulationAction(
                joint_positions=np.array(positions, dtype=float),
                joint_indices=self._joint_indices,
            )
        )

    def _sync_speed_rad_s(self) -> float:
        """The synchronized speed for the longest-travelling joint: the
        drive's smallest max joint velocity when it reports one, never more
        than SYNC_JOINT_VEL_RAD_S."""
        get_max = getattr(self._art, "get_max_joint_velocities", None)
        if get_max is not None:
            try:
                values = [float(v) for v in get_max(joint_indices=self._joint_indices)]
                lowest = min(values) if values else 0.0
                if 0.0 < lowest < SYNC_JOINT_VEL_RAD_S:
                    return lowest
            except Exception:
                LOGGER.exception("could not read the arm's max joint velocities")
        return SYNC_JOINT_VEL_RAD_S

    def _interp_callback_name(self) -> str:
        return f"{getattr(self._art, 'name', '')}_interp"

    def _ensure_interp_callback(self) -> None:
        name = self._interp_callback_name()
        if not self._sim.world.physics_callback_exists(name):
            self._sim.world.add_physics_callback(name, self._on_interp_step)

    def _on_interp_step(self, step_size: float) -> None:
        """Sim thread, every physics step: advance the in-flight path's
        target along its current straight joint-space segment so all joints
        arrive together; pause while the arm lags the target by more than
        PATH_LAG_TOL_RAD, and flag a stall when the pause outlasts
        STALL_NO_PROGRESS_STEPS."""
        interp = self._interp
        if interp is None:
            return
        measured = self._art.get_joint_positions(joint_indices=self._joint_indices)
        lag = max(
            (abs(float(m) - c) for m, c in zip(measured, interp["commanded"], strict=True)),
            default=0.0,
        )
        if len(self._trace) < 4000:
            ee = None
            if self._ee is not None:
                try:
                    pos, _ = self._ee.get_world_pose()
                    ee = [float(pos[0]) * MM_PER_M, float(pos[1]) * MM_PER_M, float(pos[2]) * MM_PER_M]
                except Exception:
                    ee = None
            self._trace.append({
                "segment": interp["index"],
                "commanded_deg": [math.degrees(c) for c in interp["commanded"]],
                "measured_deg": [math.degrees(float(m)) for m in measured],
                "lag_deg": math.degrees(lag),
                "ee_mm": ee,
            })
        if lag > PATH_LAG_TOL_RAD:
            # hold the commanded target until the arm catches up; a lag that
            # stops shrinking is a stall (the settle rule's progress test)
            if lag < interp.get("best_lag", math.inf) - STALL_PROGRESS_EPS_RAD:
                interp["best_lag"] = lag
                interp["lag_steps"] = 0
            else:
                interp["lag_steps"] += 1
                if interp["lag_steps"] >= STALL_NO_PROGRESS_STEPS:
                    interp["stalled"] = True
            return
        interp["lag_steps"] = 0
        interp["best_lag"] = math.inf
        segment = interp["segments"][interp["index"]]
        interp["elapsed"] += float(step_size)
        fraction = min(1.0, interp["elapsed"] / segment["duration"])
        positions = [
            s + (g - s) * fraction
            for s, g in zip(segment["start"], segment["goal"], strict=True)
        ]
        interp["commanded"] = positions
        self._apply_joint_positions(positions)
        if fraction >= 1.0:
            interp["index"] += 1
            interp["elapsed"] = 0.0
            if interp["index"] >= len(interp["segments"]):
                self._interp = None

    def _apply_velocity_cap(self, max_vel_rad_s: float | None) -> None:
        """ARM-13: cap the named joints' max velocity for this move via
        set_max_joint_velocities (a 5.0+ API - guarded by getattr, not a
        version check), restoring the pre-cap values read via
        get_max_joint_velocities when a later move passes None."""
        set_max = getattr(self._art, "set_max_joint_velocities", None)
        if set_max is None:
            return
        if max_vel_rad_s is not None:
            if self._saved_max_joint_velocities is None:
                get_max = getattr(self._art, "get_max_joint_velocities", None)
                if get_max is not None:
                    self._saved_max_joint_velocities = get_max(joint_indices=self._joint_indices)
            count = (
                len(self._joint_indices)
                if self._joint_indices is not None
                else len(self._dof_names)
            )
            set_max(np.array([max_vel_rad_s] * count), joint_indices=self._joint_indices)
        elif self._saved_max_joint_velocities is not None:
            set_max(self._saved_max_joint_velocities, joint_indices=self._joint_indices)
            self._saved_max_joint_velocities = None

    def is_moving(self) -> bool:
        def _check():
            vels = self._art.get_joint_velocities(joint_indices=self._joint_indices)
            is_moving = vels is not None and bool(max(abs(float(v)) for v in vels) > VEL_EPS_RAD_S)
            if is_moving or self._targets is None:
                return is_moving
            positions = self._art.get_joint_positions(joint_indices=self._joint_indices)
            return any(
                abs(float(p) - t) > SETTLE_TOL_RAD
                for p, t in zip(positions, self._targets, strict=True)
            )

        return self._sim.run(_check)

    def _settle_callback_name(self) -> str:
        return f"{getattr(self._art, 'name', '')}_settle"

    def wait_for_settle(
        self, timeout_s: float, tolerance_rad: float = SETTLE_TOL_RAD
    ) -> SettleOutcome:
        # ARM-12: a physics-step callback evaluates the settle predicate on
        # the sim thread every step and signals a threading.Event; this
        # (caller) thread only waits on it, so no wall-clock polling.
        if self._targets is None:
            return SettleOutcome.REACHED

        targets = list(self._targets)
        joint_indices = self._joint_indices
        event = threading.Event()
        outcome: list[SettleOutcome] = []
        counters: dict[str, float] = {
            "within": 0,
            "still_off_target": 0,
            "sim_time": 0.0,
            "max_speed": 0.0,
            "best_error": math.inf,
            "no_progress": 0,
        }
        settle_state: dict[str, Any] = {"event": event, "outcome": outcome}

        def _on_step(step_size: float) -> None:
            # sim thread only: touch the counters/Event, never self._sim.run.
            if event.is_set():
                return
            velocities = self._art.get_joint_velocities(joint_indices=joint_indices)
            max_speed = max((abs(float(v)) for v in velocities), default=0.0)
            counters["max_speed"] = max(counters["max_speed"], max_speed)
            is_still = max_speed <= VEL_EPS_RAD_S
            positions = self._art.get_joint_positions(joint_indices=joint_indices)
            errors = [abs(float(p) - t) for p, t in zip(positions, targets, strict=True)]
            worst_error = max(errors, default=0.0)
            is_within = worst_error <= tolerance_rad
            # REACHED is a position criterion held over the window - holding
            # within tolerance for SETTLE_WINDOW_STEPS steps IS "settled".
            # Velocity only decides stalls: PhysX never reads exactly still.
            counters["within"] = counters["within"] + 1 if is_within else 0
            counters["still_off_target"] = (
                counters["still_off_target"] + 1 if (is_still and not is_within) else 0
            )
            # no-progress stall: the worst error has not improved for a while
            if worst_error < counters["best_error"] - STALL_PROGRESS_EPS_RAD:
                counters["best_error"] = worst_error
                counters["no_progress"] = 0
            else:
                counters["no_progress"] += 1

            if counters["within"] >= SETTLE_WINDOW_STEPS:
                outcome.append(SettleOutcome.REACHED)
                event.set()
                return
            path = self._interp
            if path is not None:
                # a path in flight: the error to the FINAL target need not
                # shrink monotonically, so only the path's own lag flag stalls
                counters["no_progress"] = 0
                if path.get("stalled"):
                    outcome.append(SettleOutcome.STALLED)
                    event.set()
                    return
            elif counters["still_off_target"] >= SETTLE_WINDOW_STEPS or (
                not is_within and counters["no_progress"] >= STALL_NO_PROGRESS_STEPS
            ):
                outcome.append(SettleOutcome.STALLED)
                event.set()
                return

            counters["sim_time"] += step_size
            if counters["sim_time"] >= timeout_s:
                LOGGER.warning(
                    "arm %r settle timed out after %.2fs sim time: within tolerance=%s, "
                    "still=%s, max |v| seen=%.4f rad/s (VEL_EPS %.4f)",
                    getattr(self._art, "name", ""),
                    counters["sim_time"],
                    is_within,
                    is_still,
                    counters["max_speed"],
                    VEL_EPS_RAD_S,
                )
                outcome.append(SettleOutcome.TIMED_OUT)
                event.set()

        callback_name = self._settle_callback_name()

        def _register() -> None:
            self._sim.world.add_physics_callback(callback_name, _on_step)

        def _remove() -> None:
            if self._sim.world.physics_callback_exists(callback_name):
                self._sim.world.remove_physics_callback(callback_name)

        with self._active_settle_lock:
            self._active_settle = settle_state
        self._sim.run(_register)
        try:
            # a generous wall-clock guard so a paused sim can't hang forever.
            wall_clock_guard_s = timeout_s * 4 + 5
            event_was_set = event.wait(timeout=wall_clock_guard_s)
            if not event_was_set:
                return SettleOutcome.TIMED_OUT
            return outcome[0]
        finally:
            with self._active_settle_lock:
                if self._active_settle is settle_state:
                    self._active_settle = None
            self._sim.run(_remove)

    def stop(self) -> None:
        # hold the current position (and drop any in-flight interpolation)
        self._sim.run(lambda: setattr(self, "_interp", None))
        current = self.get_joint_positions()
        self.set_joint_targets(current)
        # the new target IS the current position, so any in-flight
        # wait_for_settle should read as having reached it, not stalled.
        with self._active_settle_lock:
            active = self._active_settle
            if active is not None and not active["event"].is_set():
                active["outcome"].append(SettleOutcome.REACHED)
                active["event"].set()

    def get_end_pose(self):
        if self._ee is None:
            raise NotImplementedError(
                "set end_effector_prim in the arm config to report end position"
            )

        def _pose():
            root_pos, root_quat = self._art.get_world_pose()
            pos, quat = self._ee.get_world_pose()
            root_pos_t = (float(root_pos[0]), float(root_pos[1]), float(root_pos[2]))
            root_quat_t = (
                float(root_quat[0]),
                float(root_quat[1]),
                float(root_quat[2]),
                float(root_quat[3]),
            )
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            quat_t = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            base_pos, base_quat = viam_base_frame(root_pos_t, root_quat_t, self._base_correction)
            return pose_in_frame(base_pos, base_quat, pos_t, quat_t)

        return self._sim.run(_pose)

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        def _pose() -> tuple[Vec3, Quat]:
            self._sim._require_prim(prim_path)
            pos, quat = self._sim._isaac.SingleXFormPrim(prim_path).get_world_pose()
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            quat_t = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            return pos_t, quat_t

        return self._sim.run(_pose)

    def post_reset(self) -> None:
        """ARM-15/ARM-16: a world.reset() resets the solver iteration count
        and controller gains to the prim's authored defaults, and can
        teleport the articulation to its default pose - re-apply the
        snapshotted solver count/gains and hold the pose the reset put the
        arm in. Re-commanding the pre-reset targets instead drove the arm
        from the default pose back to wherever it had been, through props
        the same reset had just teleported to their spawn poses: with the
        fingertips pressing on the block at reset time the block was knocked
        20-60 mm out of place (2026-09-04). A reset returns the whole cell to
        its start, arm included; the caller moves the arm from there."""

        def _redo() -> None:
            set_iterations = getattr(self._art, "set_solver_position_iteration_count", None)
            if set_iterations is not None and self._solver_iterations is not None:
                set_iterations(self._solver_iterations)
            if self._gains is not None:
                kps, kds = self._gains
                self._art.get_articulation_controller().set_gains(kps=kps, kds=kds)
            self._interp = None
            current = [
                float(p) for p in self._art.get_joint_positions(joint_indices=self._joint_indices)
            ]
            self._targets = current
            self._apply_joint_positions(current)

        self._sim.run(_redo)

    def release(self) -> None:
        """XC-4: remove the settle callback if one is registered and drop the
        scene-registry entry (registry_only - the prim stays), so a later
        create_arm for this name can re-attach."""

        def _release() -> None:
            world = self._sim.world
            callback_name = self._settle_callback_name()
            if world.physics_callback_exists(callback_name):
                world.remove_physics_callback(callback_name)
            name = getattr(self._art, "name", "")
            if world.scene.get_object(name) is not None:
                world.scene.remove_object(name, registry_only=True)

        self._sim.run(_release)


class MockArmHandle(ArmHandle):
    """Joints move linearly toward their targets at a fixed speed. Total dof
    count is mock_dof (default: the number of declared joint names, else 6);
    the arm's named joints are selected by index the same way the Isaac
    handle does (FINDINGS ARM-1; R-3), and any remaining dofs are padding
    that never moves."""

    SPEED = 1.0  # rad/s per joint
    STEP_S = 1.0 / 120.0  # the mock's "physics step" for wait_for_settle polling

    # the mock's end effector, fixed in Viam's arm frame (public,
    # deterministic value; unchanged by spawn pose or base_frame_correction).
    FIXED_LOCAL_EE: tuple[Vec3, Quat] = ((0.3, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0))

    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        from .spatial import to_vec3

        self.name = name
        # test knob (ARM-13 "stalled vs timed out"): when set, every move stops
        # after this fraction of its travel, like an arm blocked by an obstacle
        stall = attrs.get("mock_stall_fraction")
        self.mock_stall_fraction: float | None = None if stall is None else float(stall)
        self._speed = self.SPEED
        meta = KNOWN_ASSETS.get(str(attrs.get("asset", "")), {})
        joint_names: Sequence[str] | None = meta.get("joint_names")
        default_dof = len(joint_names) if joint_names else 6
        dof = int(attrs.get("mock_dof", default_dof))
        if joint_names:
            names = list(joint_names) + [f"mock_extra_{i}" for i in range(dof - len(joint_names))]
            self._joint_indices: list[int] | None = list(range(len(joint_names)))
        else:
            names = [f"mock_joint_{i}" for i in range(dof)]
            self._joint_indices = None
        self._dof_names = names
        self._lock = threading.Lock()
        self._start = [0.0] * dof
        self._target = [0.0] * dof
        self._t0 = time.monotonic()
        self.spawn_position = to_vec3(attrs.get("position"))
        self.spawn_orientation = spawn_orientation(attrs, meta)
        correction = meta.get("base_frame_correction")
        self._base_correction: Quat = (
            _as_quat(correction)
            if correction is not None
            else (
                1.0,
                0.0,
                0.0,
                0.0,
            )
        )
        self._prim_path = attrs.get("prim_path") or f"/World/{_prim_name(name)}"

    def dof_names(self) -> list[str]:
        return list(self._dof_names)

    def all_dof_names(self) -> list[str]:
        return list(self._dof_names)

    def _selected(self) -> list[int]:
        if self._joint_indices is None:
            return list(range(len(self._dof_names)))
        return self._joint_indices

    def _travel_limit(self, delta: float) -> float:
        """How far a joint may travel toward its target this move: all the
        way, or mock_stall_fraction of it when the mock is told to stall."""
        if self.mock_stall_fraction is None:
            return abs(delta)
        return abs(delta) * self.mock_stall_fraction

    def _positions_at(self, now: float) -> list[float]:
        out = []
        dt = max(0.0, now - self._t0)
        for s, t in zip(self._start, self._target, strict=True):
            delta = t - s
            travel = min(self._speed * dt, self._travel_limit(delta))
            if travel >= abs(delta):
                out.append(t)
            else:
                out.append(s + math.copysign(travel, delta))
        return out

    def _velocities_at(self, now: float) -> list[float]:
        dt = max(0.0, now - self._t0)
        return [
            0.0 if self._speed * dt >= self._travel_limit(t - s) else self._speed
            for s, t in zip(self._start, self._target, strict=True)
        ]

    def get_all_joint_positions(self) -> list[float]:
        """Test-only accessor for the full (unselected) dof array."""
        with self._lock:
            return self._positions_at(time.monotonic())

    def joint_state(self) -> list[dict[str, Any]]:
        with self._lock:
            now = time.monotonic()
            positions = self._positions_at(now)
            velocities = self._velocities_at(now)
            targets = list(self._target)
        named = set(self._selected())
        return [
            {
                "name": name,
                "position": positions[i],
                "velocity": velocities[i],
                "target": targets[i],
                "named": i in named,
            }
            for i, name in enumerate(self._dof_names)
        ]

    def get_joint_positions(self) -> list[float]:
        with self._lock:
            all_pos = self._positions_at(time.monotonic())
        return [all_pos[i] for i in self._selected()]

    def set_joint_targets(self, positions: list[float], max_vel_rad_s: float | None = None) -> None:
        with self._lock:
            now = time.monotonic()
            all_pos = self._positions_at(now)
            selected = self._selected()
            if len(positions) != len(selected):
                raise ValueError(f"expected {len(selected)} joint positions, got {len(positions)}")
            self._start = all_pos
            self._target = list(all_pos)
            for i, p in zip(selected, positions, strict=True):
                self._target[i] = p
            self._t0 = now
            self._speed = self.SPEED if max_vel_rad_s is None else min(self.SPEED, max_vel_rad_s)

    def is_moving(self) -> bool:
        with self._lock:
            now = time.monotonic()
            pos = self._positions_at(now)
            vel = self._velocities_at(now)
        selected = self._selected()
        return any(
            abs(vel[i]) > VEL_EPS_RAD_S or abs(pos[i] - self._target[i]) > SETTLE_TOL_RAD
            for i in selected
        )

    def wait_for_settle(
        self, timeout_s: float, tolerance_rad: float = SETTLE_TOL_RAD
    ) -> SettleOutcome:
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                now = time.monotonic()
                pos = self._positions_at(now)
                vel = self._velocities_at(now)
                target = list(self._target)
            selected = self._selected()
            is_within_tolerance = all(abs(pos[i] - target[i]) <= tolerance_rad for i in selected)
            is_still = all(abs(vel[i]) <= VEL_EPS_RAD_S for i in selected)
            # REACHED means settled, not merely close: is_moving() must read
            # False the instant this returns (ARM-12 / R-7 unification).
            if is_within_tolerance and is_still:
                return SettleOutcome.REACHED
            if is_still:
                return SettleOutcome.STALLED
            if now >= deadline:
                return SettleOutcome.TIMED_OUT
            time.sleep(self.STEP_S)

    def stop(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._start = self._positions_at(now)
            self._target = list(self._start)
            self._t0 = now

    def _ee_world_pose(self) -> tuple[Vec3, Quat]:
        """The mock's simulated Isaac root is (spawn_position,
        spawn_orientation) - already composed with base_frame_correction -
        so its end effector's world pose is FIXED_LOCAL_EE expressed in
        Viam's arm frame, then re-composed onto that rotated root."""
        base_pos, base_quat = viam_base_frame(
            self.spawn_position, self.spawn_orientation, self._base_correction
        )
        local_pos, local_quat = self.FIXED_LOCAL_EE
        return compose_pose(base_pos, base_quat, local_pos, local_quat)

    def get_end_pose(self):
        # a fixed, deterministic pose for testing, defined in Viam's arm
        # frame (FINDINGS ARM-10) - it must not change with spawn_position/
        # spawn_orientation/base_frame_correction.
        base_pos, base_quat = viam_base_frame(
            self.spawn_position, self.spawn_orientation, self._base_correction
        )
        ee_pos, ee_quat = self._ee_world_pose()
        return pose_in_frame(base_pos, base_quat, ee_pos, ee_quat)

    def get_prim_world_pose(self, prim_path: str) -> tuple[Vec3, Quat]:
        ee_prim_path = f"{self._prim_path}/wrist_3_link"
        if prim_path != ee_prim_path:
            raise PrimNotFoundError(f"prim not found: {prim_path}")
        return self._ee_world_pose()


# ----------------------------------------------------------------------
# gripper handles (FINDINGS ARM-3, ARM-4, ARM-8; W13)
# ----------------------------------------------------------------------

GRIPPER_HOLDING_STEPS = 5  # consecutive still steps outside tolerance before is_holding() flips
# The 2F-85's five passive joints follow finger_joint through PhysxMimicJointAPI
# in the standalone asset. Attached under the arm those mimics fail to create
# (parsed before the joints join the articulation - GPU run 20) and the
# passive drives then hold the linkage open against the finger drive, so the
# handle commands them itself: sign x finger angle, signs read off the
# linkage at rest.
GRIPPER_COUPLED_JOINT_SIGNS: dict[str, float] = {
    "right_outer_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": 1.0,
    "left_inner_finger_knuckle_joint": -1.0,
    "right_inner_finger_knuckle_joint": -1.0,
}
# The real 2F-85 has ONE motor; the linkage (loop-closing fixed joints in the
# asset) moves the other joints. Stiff drives on all six over-constrain the
# loops and the jaw buzzes in place (GPU run 21: ±85 deg/s at 1.7 deg), so the
# passive joints' drives are released to a little damping and finger_joint
# alone is driven.
PASSIVE_JOINT_DAMPING = 0.1
# A jaw pressed onto an object vibrates (GPU run 23: +/-90 deg/s at the contact
# angle), so stillness can never gate the stall/holding predicates. Like the
# arm's settle rule: a stall is "the gap to the target stopped improving by
# this much over GRIPPER_HOLDING_STEPS consecutive checks".
JAW_PROGRESS_EPS_RAD = math.radians(0.5)
GRIPPER_OPEN_WIDTH_M = 0.085  # 2F-85 jaw opening at the open angle (W13); linear to 0 at closed
DEFAULT_HOLDING_TOLERANCE_DEG = 2.0  # holding_tolerance_deg attrs default


class GripperHandle:
    """A parallel-jaw gripper riding an arm. Angles are radians on the drive
    joint (finger_joint on the 2F-85), increasing from open toward closed;
    the Viam edge (models/gripper.py) owns the [0,1]-normalised inputs and
    degrees (DEC-12). All methods are safe from any thread."""

    def jaw_limits(self) -> tuple[float, float]:
        """(open_rad, closed_rad) of the drive joint - the ends of the [0,1]
        input range."""
        raise NotImplementedError

    def get_jaw(self) -> float:
        """Measured drive-joint angle, radians."""
        raise NotImplementedError

    def set_jaw(self, rad: float) -> None:
        """Command the drive joint (clamped to jaw_limits). Returns at once."""
        raise NotImplementedError

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Freeze the jaw. While an object is held the grasp is kept (the
        commanded target stays, so the squeeze does not relax); otherwise
        the jaw holds its current angle. viam-server calls Stop on every
        actuator a session commanded once that session lapses, so a stop
        that relaxed the drive dropped the object two seconds after any
        one-shot client (the CLI's ``part run``) exited."""
        raise NotImplementedError

    def finger_effort(self) -> float | None:
        """Measured drive-joint effort (N m on a revolute drive), or None when
        the backend cannot read one (the mock; an Isaac build without
        get_measured_joint_efforts)."""
        return None

    def is_moving(self) -> bool:
        """True while the jaw is travelling; False once it has settled, whether
        at its target or stalled on an object."""
        raise NotImplementedError

    def is_holding(self) -> bool:
        """ARM-4 stall predicate, version-neutral (no Isaac contact query):
        the jaw is still AND |commanded - measured| > holding tolerance for
        GRIPPER_HOLDING_STEPS consecutive steps - i.e. it closed onto
        something short of its target. False while moving or when the jaw
        reached its target."""
        raise NotImplementedError

    def dof_names(self) -> list[str]:
        """The gripper's own DOF names as PhysX reports them after attach
        (finger_joint first); the mock returns its drive joint only."""
        raise NotImplementedError

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        """World poses ((x,y,z) m, (w,x,y,z)) of the mount link the gripper is
        bolted to ("parent") and its two fingertip links ("left_inner_finger",
        "right_inner_finger") - the GPU checklist's TCP measurement (item 4 /
        OQ-7). The mock returns a synthetic set consistent with tcp_offset_m."""
        raise NotImplementedError

    def fingertip_world_bounds(self) -> dict[str, tuple[Vec3, Vec3]]:
        """World-space axis-aligned bounds (min, max) in metres of the two
        fingertip PAD meshes, keyed "left"/"right". The 2F-85 asset authors
        every link frame at the base, so link origins say nothing about where
        the pads are - the mesh bounds do (item 4 / OQ-7; W15's jaw box)."""
        raise NotImplementedError

    def post_reset(self) -> None:
        """R-5: re-command the last commanded jaw target after a world
        reset, so a reset mid-pick doesn't drop the object. No-op by
        default (the mock has no such state)."""
        return None

    def release(self) -> None:
        """XC-4: drop callbacks; the prim stays attached to the arm."""
        return None

    def contacts(self) -> list[dict[str, Any]]:
        """Debug: the contact pairs involving the gripper's links in the
        latest physics step (PhysX contact reports): both actor paths, which
        side is the gripper, and the contact points (mm, world), normals,
        impulses and separations. The mock has no physics and returns []."""
        return []

    def collision_shapes(self, prim_path: str | None = None) -> list[dict[str, Any]]:
        """Debug: every collider (and, marked, every visual mesh) under
        ``prim_path`` (default: the gripper prim): path, rigid link,
        approximation, enabled, contact/rest offsets, and its world AABB in
        mm from the physics-aware link pose. The mock returns []."""
        return []


class IsaacGripperHandle(GripperHandle):
    """Drives the finger_joint DOF of the ARM's articulation: the gripper is
    referenced with articulationEnabled=False, so its joints join the arm's
    DOF list and are addressed by name, never by position (R-3)."""

    def __init__(
        self,
        sim: SimManager,
        articulation: Any,
        drive_joint: str,
        open_rad: float,
        closed_rad: float,
        holding_tolerance_rad: float,
        prim_path: str,
        holding_effort_min: float | None = None,
    ) -> None:
        self._sim = sim
        self._art = articulation
        self._drive_joint = drive_joint
        self._open_rad = open_rad
        self._closed_rad = closed_rad
        self._holding_tolerance_rad = holding_tolerance_rad
        # holding_effort_min_nm: when set and the articulation reports measured
        # joint efforts, is_holding() reads the drive's effort instead of the
        # stall window (a contact measurement rather than a position guess)
        self._holding_effort_min = holding_effort_min
        self._effort_warned = False
        self._prim_path = prim_path
        # debug contact reports (contacts()): subscribed on first use
        self._contact_sub: Any = None
        self._last_contacts: list[dict[str, Any]] = []
        # set by _create_gripper_isaac: the link base_link is bolted to
        self.parent_prim_path: str | None = None
        dof_names = list(articulation.dof_names)
        try:
            self._idx = dof_names.index(drive_joint)
        except ValueError as exc:
            raise ValueError(
                f"gripper drive joint {drive_joint!r} not found in articulation "
                f"dof_names: {dof_names}"
            ) from exc
        # last commanded target, plus the progress window for the stall-based
        # is_moving/is_holding predicates (R-4/ARM-4; GPU run 23). The latch
        # carries a detected hold across slow polls (GPU run 25): once the jaw
        # stalls outside tolerance it stays "holding" until it reaches its
        # target (nothing left between the jaws) or a new set_jaw.
        self._target: float | None = None
        self._best_gap_rad: float | None = None
        self._no_progress_count = 0
        self._held_latch = False
        self._coupled: list[tuple[int, float]] = [
            (dof_names.index(name), sign)
            for name, sign in GRIPPER_COUPLED_JOINT_SIGNS.items()
            if name in dof_names
        ]
        LOGGER.info(
            "gripper drive %r at dof %d; %d passive linkage joints: %s",
            drive_joint,
            self._idx,
            len(self._coupled),
            [dof_names[i] for i, _sign in self._coupled],
        )
        self._release_passive_drives()

    def _release_passive_drives(self) -> None:
        """Zero the passive linkage joints' drive stiffness (small damping) so
        the loop closures, not competing drives, couple them to finger_joint.
        Logs the authored gains so the asset's tuning stays on record."""
        try:
            controller = self._art.get_articulation_controller()
            kps, kds = controller.get_gains()
            kps = np.array(kps, dtype=float).copy()
            kds = np.array(kds, dtype=float).copy()
            LOGGER.info(
                "gripper passive joint gains before release (kp, kd): %s",
                [(float(kps[i]), float(kds[i])) for i, _sign in self._coupled],
            )
            for index, _sign in self._coupled:
                kps[index] = 0.0
                kds[index] = PASSIVE_JOINT_DAMPING
            controller.set_gains(kps=kps, kds=kds)
        except Exception:
            LOGGER.exception("could not release the gripper's passive joint drives")

    def jaw_limits(self) -> tuple[float, float]:
        return (self._open_rad, self._closed_rad)

    def get_jaw(self) -> float:
        def _get() -> float:
            return float(self._art.get_joint_positions(joint_indices=[self._idx])[0])

        return self._sim.run(_get)

    def _apply_target(self, rad: float) -> None:
        """Command the drive joint and reset the stall window. Sim thread only."""
        # finger_joint only: the linkage carries the passive joints
        action = self._sim._isaac.ArticulationAction(
            joint_positions=np.array([rad], dtype=float),
            joint_indices=[self._idx],
        )
        self._art.apply_action(action)
        self._target = rad
        self._best_gap_rad = None
        self._no_progress_count = 0
        self._held_latch = False

    def set_jaw(self, rad: float) -> None:
        rad = min(max(rad, self._open_rad), self._closed_rad)
        self._sim.run(lambda: self._apply_target(rad))

    def open(self) -> None:
        self.set_jaw(self._open_rad)

    def close(self) -> None:
        self.set_jaw(self._closed_rad)

    def stop(self) -> None:
        def _stop() -> None:
            if self._is_holding_locked():
                return  # keep the grasp: the commanded target stays
            measured = float(self._art.get_joint_positions(joint_indices=[self._idx])[0])
            self._apply_target(measured)

        self._sim.run(_stop)

    def _gap_and_stall(self) -> tuple[float, bool]:
        """(|target - measured|, stalled): the jaw is stalled when the gap has
        not improved by JAW_PROGRESS_EPS_RAD for GRIPPER_HOLDING_STEPS
        consecutive checks. Velocity plays no part: a jaw pressed onto an
        object vibrates and never reads still. The window counts CALLS, so
        grab()'s 120 Hz poll detects the stall but a client sampling at 1 Hz
        never re-accumulates it once the jaw creeps (GPU run 25) - a stall
        outside tolerance therefore latches _held_latch, cleared only when
        the jaw reaches its target or on a new set_jaw. Sim thread only."""
        if self._target is None:
            return 0.0, False
        measured = float(self._art.get_joint_positions(joint_indices=[self._idx])[0])
        gap = abs(self._target - measured)
        if self._best_gap_rad is None or gap < self._best_gap_rad - JAW_PROGRESS_EPS_RAD:
            self._best_gap_rad = gap
            self._no_progress_count = 0
        else:
            self._no_progress_count += 1
        stalled = self._no_progress_count >= GRIPPER_HOLDING_STEPS
        if gap <= self._holding_tolerance_rad:
            self._held_latch = False  # the jaw reached its target: nothing is held
        elif stalled:
            self._held_latch = True
        return gap, stalled

    def is_moving(self) -> bool:
        def _check() -> bool:
            gap, stalled = self._gap_and_stall()
            return gap > self._holding_tolerance_rad and not stalled and not self._held_latch

        return self._sim.run(_check)

    def _finger_effort_locked(self) -> float | None:
        """|measured drive-joint effort|, or None when unreadable. Sim thread only."""
        read = getattr(self._art, "get_measured_joint_efforts", None)
        if read is None:
            return None
        try:
            return abs(float(read(joint_indices=[self._idx])[0]))
        except Exception:
            if not self._effort_warned:
                self._effort_warned = True
                LOGGER.exception("could not read the gripper drive joint's measured effort")
            return None

    def _is_holding_locked(self) -> bool:
        """Sim thread only. With holding_effort_min_nm configured and the
        drive effort readable: the drive is pushing at least that hard while
        the jaw sits short of fully closed - a contact measurement that does
        not depend on the commanded target, so it survives stop(). Otherwise
        the ARM-4 stall predicate."""
        gap, stalled = self._gap_and_stall()
        if self._holding_effort_min is not None:
            effort = self._finger_effort_locked()
            if effort is not None:
                measured = float(self._art.get_joint_positions(joint_indices=[self._idx])[0])
                short_of_closed = self._closed_rad - measured > self._holding_tolerance_rad
                return short_of_closed and effort >= self._holding_effort_min
        return (stalled or self._held_latch) and gap > self._holding_tolerance_rad

    def is_holding(self) -> bool:
        return self._sim.run(self._is_holding_locked)

    def finger_effort(self) -> float | None:
        return self._sim.run(self._finger_effort_locked)

    def dof_names(self) -> list[str]:
        def _names() -> list[str]:
            return [n for n in self._art.dof_names if n not in UR_JOINT_NAMES]

        return self._sim.run(_names)

    def post_reset(self) -> None:
        """R-5: re-command the last commanded jaw target - a world.reset()
        can otherwise let a held object drop."""

        def _redo() -> None:
            if self._target is None:
                return
            action = self._sim._isaac.ArticulationAction(
                joint_positions=np.array([self._target], dtype=float),
                joint_indices=[self._idx],
            )
            self._art.apply_action(action)

        self._sim.run(_redo)

    def release(self) -> None:
        """XC-4: the gripper drives a DOF of the arm's articulation and owns
        no scene-registry entry or physics callback of its own (post_reset is
        an XC-5 hook, not a physics callback); only the debug contact-report
        subscription, if contacts() was ever called."""
        self._contact_sub = None
        return None

    # ---- debug: what is the gripper touching, and with what shapes? ----

    def _ensure_contact_reports(self) -> None:
        """Sim thread. Apply PhysxContactReportAPI to every rigid link under
        the gripper (PhysX reports a pair when either actor carries the API)
        and subscribe to the simulation's contact report events, once."""
        if self._contact_sub is not None:
            return
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, Usd, UsdPhysics

        root = self._sim._isaac.get_prim_at_path(self._prim_path)
        applied = 0
        for prim in _prim_range(Usd, root):
            if prim.IsInstanceProxy() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(0.0)
            applied += 1
        self._contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact_report
        )
        LOGGER.info("gripper contact reports enabled on %d links under %s", applied, self._prim_path)

    def _on_contact_report(self, contact_headers: Any, contact_data: Any) -> None:
        from pxr import PhysicsSchemaTools

        def path_of(handle: Any) -> str:
            return str(PhysicsSchemaTools.intToSdfPath(handle))

        out: list[dict[str, Any]] = []
        for header in contact_headers:
            kind = str(getattr(header, "type", "")).rsplit(".", 1)[-1]
            if "LOST" in kind:
                continue
            actor0, actor1 = path_of(header.actor0), path_of(header.actor1)
            gripper_is_0 = actor0.startswith(self._prim_path)
            start = int(header.contact_data_offset)
            points = []
            for i in range(start, start + int(header.num_contact_data)):
                d = contact_data[i]
                points.append(
                    {
                        "position_mm": [float(v) * MM_PER_M for v in d.position],
                        "normal": [float(v) for v in d.normal],
                        "impulse": [float(v) for v in d.impulse],
                        "separation_mm": float(d.separation) * MM_PER_M,
                    }
                )
            out.append(
                {
                    "type": kind,
                    "gripper_link": actor0 if gripper_is_0 else actor1,
                    "gripper_collider": path_of(header.collider0 if gripper_is_0 else header.collider1),
                    "other": actor1 if gripper_is_0 else actor0,
                    "other_collider": path_of(header.collider1 if gripper_is_0 else header.collider0),
                    "points": points,
                }
            )
        self._last_contacts = out

    def contacts(self) -> list[dict[str, Any]]:
        def _read() -> list[dict[str, Any]]:
            try:
                self._ensure_contact_reports()
            except Exception as exc:
                LOGGER.exception("gripper contact reports unavailable")
                return [{"error": f"contact reports unavailable: {exc}"}]
            return list(self._last_contacts)

        return self._sim.run(_read)

    def collision_shapes(self, prim_path: str | None = None) -> list[dict[str, Any]]:
        """Every Boundable prim under ``prim_path`` (default: the gripper) that
        is a collider, plus, for comparison, non-collider meshes marked
        ``visual: False``; world AABBs from the physics-aware link pose."""
        def _shapes() -> list[dict[str, Any]]:
            from pxr import Gf, Usd, UsdGeom, UsdPhysics

            try:
                from pxr import PhysxSchema
            except Exception:
                PhysxSchema = None  # noqa: N806
            time = Usd.TimeCode.Default()
            root = self._sim._isaac.get_prim_at_path(prim_path or self._prim_path)
            out: list[dict[str, Any]] = []
            for prim in _prim_range(Usd, root):
                is_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
                if not is_collider and not prim.IsA(UsdGeom.Gprim):
                    continue
                link = prim
                while link.IsValid() and not link.HasAPI(UsdPhysics.RigidBodyAPI):
                    link = link.GetParent()
                entry: dict[str, Any] = {
                    "path": str(prim.GetPath()),
                    "type": str(prim.GetTypeName()),
                    "link": str(link.GetPath()) if link.IsValid() else None,
                    "collider": is_collider,
                    "enabled": (
                        bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
                        if is_collider
                        else False
                    ),
                    "approximation": (
                        str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
                        if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                        else None
                    ),
                    "purpose": (
                        str(UsdGeom.Imageable(prim).GetPurposeAttr().Get())
                        if prim.IsA(UsdGeom.Imageable)
                        else None
                    ),
                }
                if PhysxSchema is not None and prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    physx = PhysxSchema.PhysxCollisionAPI(prim)
                    entry["contact_offset_mm"] = float(physx.GetContactOffsetAttr().Get() or 0.0) * MM_PER_M
                    entry["rest_offset_mm"] = float(physx.GetRestOffsetAttr().Get() or 0.0) * MM_PER_M
                extent = (
                    UsdGeom.Boundable(prim).GetExtentAttr().Get(time)
                    if prim.IsA(UsdGeom.Boundable)
                    else None
                )
                if extent is not None and len(extent) == 2:
                    pose_prim = link if link.IsValid() else prim
                    mesh_in_link = (
                        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
                        * UsdGeom.Xformable(pose_prim).ComputeLocalToWorldTransform(time).GetInverse()
                    )
                    pos, quat = self._sim._isaac.SingleXFormPrim(str(pose_prim.GetPath())).get_world_pose()
                    rotate = Gf.Matrix4d().SetRotate(
                        Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3])))
                    )
                    translate = Gf.Matrix4d().SetTranslate(
                        Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
                    )
                    world = mesh_in_link * rotate * translate
                    corners = [
                        world.Transform(Gf.Vec3d(x, y, z))
                        for x in (extent[0][0], extent[1][0])
                        for y in (extent[0][1], extent[1][1])
                        for z in (extent[0][2], extent[1][2])
                    ]
                    entry["world_min_mm"] = [min(float(c[i]) for c in corners) * MM_PER_M for i in range(3)]
                    entry["world_max_mm"] = [max(float(c[i]) for c in corners) * MM_PER_M for i in range(3)]
                out.append(entry)
            return out

        return self._sim.run(_shapes)

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        def _poses() -> dict[str, tuple[Vec3, Quat]]:
            from pxr import Usd, UsdPhysics

            # every rigid-body link under the gripper, keyed by link name (the
            # GPU checklist reads the whole chain, not only the fingertips)
            paths: dict[str, str] = {}
            root = self._sim._isaac.get_prim_at_path(self._prim_path)
            for prim in _prim_range(Usd, root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() not in paths:
                    paths[prim.GetName()] = str(prim.GetPath())
            if self.parent_prim_path:
                paths["parent"] = self.parent_prim_path
            out: dict[str, tuple[Vec3, Quat]] = {}
            for key, path in paths.items():
                pos, quat = self._sim._isaac.SingleXFormPrim(path).get_world_pose()
                out[key] = (
                    (float(pos[0]), float(pos[1]), float(pos[2])),
                    (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                )
            return out

        return self._sim.run(_poses)

    def fingertip_world_bounds(self) -> dict[str, tuple[Vec3, Vec3]]:
        def _bounds() -> dict[str, tuple[Vec3, Vec3]]:
            from pxr import Gf, Usd, UsdGeom, UsdPhysics

            time = Usd.TimeCode.Default()
            root = self._sim._isaac.get_prim_at_path(self._prim_path)
            out: dict[str, tuple[Vec3, Vec3]] = {}
            for mesh in _prim_range(Usd, root):
                if "fingertip" not in mesh.GetName().lower():
                    continue
                link = mesh.GetParent()
                while link.IsValid() and not link.HasAPI(UsdPhysics.RigidBodyAPI):
                    link = link.GetParent()
                if not link.IsValid():
                    continue
                # mesh-in-link from USD (static), link-in-world from the
                # physics-aware pose: robust whether PhysX writes to USD or Fabric
                mesh_in_link = (
                    UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(time)
                    * UsdGeom.Xformable(link).ComputeLocalToWorldTransform(time).GetInverse()
                )
                pos, quat = self._sim._isaac.SingleXFormPrim(str(link.GetPath())).get_world_pose()
                rotate = Gf.Matrix4d().SetRotate(
                    Gf.Quatd(
                        float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))
                    )
                )
                translate = Gf.Matrix4d().SetTranslate(
                    Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
                )
                mesh_world = mesh_in_link * rotate * translate
                extent = UsdGeom.Boundable(mesh).GetExtentAttr().Get(time)
                if extent is None or len(extent) != 2:
                    continue
                corners = [
                    mesh_world.Transform(Gf.Vec3d(x, y, z))
                    for x in (extent[0][0], extent[1][0])
                    for y in (extent[0][1], extent[1][1])
                    for z in (extent[0][2], extent[1][2])
                ]
                low = tuple(min(float(c[i]) for c in corners) for i in range(3))
                high = tuple(max(float(c[i]) for c in corners) for i in range(3))
                side = "left" if "left" in str(mesh.GetPath()).lower() else "right"
                out[side] = ((low[0], low[1], low[2]), (high[0], high[1], high[2]))
            return out

        return self._sim.run(_bounds)


class MockGripperHandle(GripperHandle):
    """The jaw interpolates at MockArmHandle.SPEED toward its target. With
    attrs["mock_object_width_m"] set, the jaw stalls at the angle where the
    jaws would touch that object (GRIPPER_OPEN_WIDTH_M at open_rad, linear
    to 0 at closed_rad) and is_holding() flips true after
    GRIPPER_HOLDING_STEPS; unset = nothing between the jaws, so close()
    reaches closed_rad and is_holding() stays False (ARM-8)."""

    def __init__(self, name: str, attrs: dict[str, Any], arm: ArmHandle) -> None:
        self.name = name
        self._arm = arm
        default_meta = KNOWN_ASSETS["robotiq_2f_85"]
        meta = KNOWN_ASSETS.get(str(attrs.get("asset", "robotiq_2f_85")), default_meta)
        self._drive_joint = meta.get("drive_joint", "finger_joint")
        self.open_rad = math.radians(attrs.get("open_deg", meta.get("open_deg", 0.0)))
        self.closed_rad = math.radians(attrs.get("closed_deg", caps().gripper_closed_deg))
        self.holding_tolerance_rad = math.radians(
            attrs.get("holding_tolerance_deg", DEFAULT_HOLDING_TOLERANCE_DEG)
        )
        self.mock_object_width_m: float | None = attrs.get("mock_object_width_m")
        self.tcp_offset_m = float(attrs.get("tcp_offset_m", meta.get("tcp_offset_m", 0.134)))
        self._speed = MockArmHandle.SPEED
        self._lock = threading.Lock()
        now = time.monotonic()
        self._start = self.open_rad
        self._target = self.open_rad
        self._t0 = now

    def _contact_angle(self) -> float:
        """The drive-joint angle at which the jaws would touch
        mock_object_width_m - GRIPPER_OPEN_WIDTH_M at open_rad, linearly to
        closed_rad at width 0 (also the value used when nothing is set, since
        there is then nothing to stop the jaw short of closed_rad)."""
        if self.mock_object_width_m is None:
            return self.closed_rad
        width = min(max(self.mock_object_width_m, 0.0), GRIPPER_OPEN_WIDTH_M)
        fraction_closed = 1.0 - width / GRIPPER_OPEN_WIDTH_M
        return self.open_rad + (self.closed_rad - self.open_rad) * fraction_closed

    def _effective_target(self) -> float:
        """The commanded target, clamped short of an object in the way."""
        return min(self._target, self._contact_angle())

    def _arrival_time(self) -> float:
        """The monotonic time the jaw reaches _effective_target, given the
        move that started at (_start, _t0)."""
        delta = self._effective_target() - self._start
        return self._t0 + abs(delta) / self._speed

    def _jaw_at(self, now: float) -> float:
        start = self._start
        target = self._effective_target()
        delta = target - start
        travel = min(self._speed * max(0.0, now - self._t0), abs(delta))
        if travel >= abs(delta):
            return target
        return start + math.copysign(travel, delta)

    def jaw_limits(self) -> tuple[float, float]:
        return (self.open_rad, self.closed_rad)

    def get_jaw(self) -> float:
        with self._lock:
            return self._jaw_at(time.monotonic())

    def set_jaw(self, rad: float) -> None:
        rad = min(max(rad, self.open_rad), self.closed_rad)
        with self._lock:
            now = time.monotonic()
            self._start = self._jaw_at(now)
            self._target = rad
            self._t0 = now

    def open(self) -> None:
        self.set_jaw(self.open_rad)

    def close(self) -> None:
        self.set_jaw(self.closed_rad)

    def stop(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._grasp_locked(now):
                return  # keep the grasp: the commanded target stays
            current = self._jaw_at(now)
            self._start = current
            self._target = current
            self._t0 = now

    def _grasp_locked(self, now: float) -> bool:
        """The jaw has arrived on an object short of its target (lock held)."""
        if now < self._arrival_time():
            return False
        return abs(self._target - self._effective_target()) > self.holding_tolerance_rad

    def is_moving(self) -> bool:
        with self._lock:
            return time.monotonic() < self._arrival_time()

    def is_holding(self) -> bool:
        with self._lock:
            now = time.monotonic()
            arrival = self._arrival_time()
            if now < arrival:
                return False
            measured = self._effective_target()
            if abs(self._target - measured) <= self.holding_tolerance_rad:
                return False
            still_duration = now - arrival
            return still_duration >= GRIPPER_HOLDING_STEPS * MockArmHandle.STEP_S

    def dof_names(self) -> list[str]:
        return [self._drive_joint]

    def link_world_poses(self) -> dict[str, tuple[Vec3, Quat]]:
        """Synthetic: the mount link at the origin, fingertips straddling the
        TCP at tcp_offset_m along +Z, GRIPPER_OPEN_WIDTH_M apart."""
        identity: Quat = (1.0, 0.0, 0.0, 0.0)
        half_width = GRIPPER_OPEN_WIDTH_M / 2.0
        return {
            "parent": ((0.0, 0.0, 0.0), identity),
            "base_link": ((0.0, 0.0, 0.0), identity),
            "left_inner_finger": ((half_width, 0.0, self.tcp_offset_m), identity),
            "right_inner_finger": ((-half_width, 0.0, self.tcp_offset_m), identity),
        }

    FINGERTIP_PAD_HALF_EXTENT_M: tuple[float, float, float] = (0.005, 0.011, 0.019)

    def fingertip_world_bounds(self) -> dict[str, tuple[Vec3, Vec3]]:
        """Synthetic pads centred at tcp_offset_m along +Z, straddling the jaw."""
        half_width = GRIPPER_OPEN_WIDTH_M / 2.0
        hx, hy, hz = self.FINGERTIP_PAD_HALF_EXTENT_M
        out: dict[str, tuple[Vec3, Vec3]] = {}
        for side, cx in (("left", half_width), ("right", -half_width)):
            cz = self.tcp_offset_m
            out[side] = ((cx - hx, -hy, cz - hz), (cx + hx, hy, cz + hz))
        return out


# CAM-2: bounded retry on the caller's thread while the renderer warms up
# after create/reset, sleeping between attempts so the sim thread gets to run.
WARMUP_RETRIES = 30
WARMUP_SLEEP_S = 1.0 / 60.0
WARMUP_MESSAGE = "no frame available yet - is the simulation playing?"
# The renderer's first frames after a boot lag the simulation by seconds (RTX
# shader warm-up): the first one or two wrist-camera reads showed the scene
# from before the arm had moved, and a block was located a metre off
# (2026-09-04). A frame whose rendering time trails the simulation clock by
# more than this is served to nobody; get_frame's retry loop waits for a
# fresh one.
STALE_FRAME_S = 0.25
STALE_MESSAGE = "the latest rendered frame is older than the simulation clock - renderer warming up"


class IsaacCameraHandle(CameraHandle):
    def __init__(
        self,
        sim: SimManager,
        cam: Any,
        *,
        depth_enabled: bool,
        image_format: str,
        frequency: float | None,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._sim = sim
        self._cam = cam
        self.depth_enabled = depth_enabled
        self.image_format = image_format
        self.frequency = frequency
        self._now = now or (lambda: float(sim.world.current_time))
        self._sleep = sleep
        self._cached_frame: Frame | None = None

    def _grab(self) -> Frame:
        # runs on the sim thread (via sim.run): one rgb(+depth) read per sim
        # step, cached by sim_time so GetImages + GetPointCloud in the same
        # tick share one grab (CAM-9).
        sim_time = self._now()
        cached = self._cached_frame
        if cached is not None and cached.sim_time == sim_time:
            return cached

        rendering_time = self._rendering_time()
        if rendering_time is not None and sim_time - rendering_time > STALE_FRAME_S:
            raise NoFrameYetError(STALE_MESSAGE)
        rgba = self._cam.get_rgba()
        if rgba is None or rgba.size == 0:
            raise NoFrameYetError(WARMUP_MESSAGE)
        rgb = rgba[:, :, :3].copy()

        depth = None
        if self.depth_enabled:
            raw_depth = self._cam.get_depth()
            if raw_depth is None:
                raise NoFrameYetError(WARMUP_MESSAGE)
            depth = np.asarray(raw_depth)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            depth = depth.astype(np.float32)

        frame = Frame(rgb=rgb, depth=depth, sim_time=sim_time)
        self._cached_frame = frame
        return frame

    def _rendering_time(self) -> float | None:
        """Simulation time of the frame the camera would return now, from
        Isaac's get_current_frame() ("rendering_time"); None when the API or
        the key is unavailable, in which case no staleness check is made."""
        read = getattr(self._cam, "get_current_frame", None)
        if read is None:
            return None
        try:
            info = read()
            value = info.get("rendering_time") if isinstance(info, dict) else None
            return None if value is None else float(value)
        except Exception:
            return None

    def get_frame(self) -> Frame:
        last_error: NoFrameYetError | None = None
        for _ in range(WARMUP_RETRIES):
            try:
                return self._sim.run(self._grab)
            except NoFrameYetError as exc:
                last_error = exc
                self._sleep(WARMUP_SLEEP_S)
        raise NoFrameYetError(WARMUP_MESSAGE) from last_error

    def get_intrinsics(self) -> Intrinsics:
        def _read() -> Intrinsics:
            focal_length = self._cam.get_focal_length()
            horizontal_aperture = self._cam.get_horizontal_aperture()
            vertical_aperture = self._cam.get_vertical_aperture()
            width, height = self._cam.get_resolution()
            if not focal_length or not horizontal_aperture or not vertical_aperture:
                raise RuntimeError(
                    "camera intrinsics unavailable: focal length or aperture is 0 "
                    "(has the camera been initialized?)"
                )
            return Intrinsics(
                fx=width * focal_length / horizontal_aperture,
                fy=height * focal_length / vertical_aperture,
                cx=width / 2,
                cy=height / 2,
                width=width,
                height=height,
            )

        return self._sim.run(_read)

    def post_reset(self) -> None:
        def _reset() -> None:
            self._cached_frame = None
            try:
                post_reset = getattr(self._cam, "post_reset", None)
                if post_reset is not None:
                    post_reset()
                else:
                    self._cam.initialize()
            except Exception:
                LOGGER.exception("camera post-reset failed")

        self._sim.run(_reset)

    def release(self) -> None:
        """XC-4: Camera.destroy() only exists on isaac 5.0 (XC-4/W28) -
        guarded by getattr, never a version check; older releases have
        nothing to release."""

        def _release() -> None:
            destroy = getattr(self._cam, "destroy", None)
            if destroy is not None:
                destroy()

        self._sim.run(_release)


class BaseHandle:
    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def is_moving(self) -> bool:
        raise NotImplementedError

    def release(self) -> None:
        """XC-4: Isaac backends remove their `<name>_drive` physics callback
        and the scene-registry entry (registry_only); the prim stays."""
        return None


class IsaacBaseHandle(BaseHandle):
    def __init__(
        self, sim: SimManager, robot, controller, wheel_radius: float, wheel_base: float
    ) -> None:
        self._sim = sim
        self._robot = robot
        self._controller = controller
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self._cmd = (0.0, 0.0)
        self._lock = threading.Lock()

    def _on_physics_step(self, step_size: float) -> None:
        # runs on the sim thread every physics step
        with self._lock:
            lin, ang = self._cmd
        try:
            self._robot.apply_wheel_actions(self._controller.forward(command=[lin, ang]))
        except Exception:
            LOGGER.exception("error driving base")

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        with self._lock:
            self._cmd = (float(linear_mps), float(angular_rps))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def is_moving(self) -> bool:
        with self._lock:
            return self._cmd != (0.0, 0.0)

    def release(self) -> None:
        """XC-4: remove the `<name>_drive` physics callback and drop the
        scene-registry entry (registry_only - the prim stays), so a later
        create_base for this name can re-attach."""

        def _release() -> None:
            world = self._sim.world
            name = getattr(self._robot, "name", "")
            callback_name = f"{name}_drive"
            if world.physics_callback_exists(callback_name):
                world.remove_physics_callback(callback_name)
            if world.scene.get_object(name) is not None:
                world.scene.remove_object(name, registry_only=True)

        self._sim.run(_release)


class MockBaseHandle(BaseHandle):
    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self.name = name
        self.wheel_radius = float(attrs.get("wheel_radius", 0.05))
        self.wheel_base = float(attrs.get("wheel_base", 0.3))
        self._cmd = (0.0, 0.0)
        self._lock = threading.Lock()

    def set_velocity(self, linear_mps: float, angular_rps: float) -> None:
        with self._lock:
            self._cmd = (float(linear_mps), float(angular_rps))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def is_moving(self) -> bool:
        with self._lock:
            return self._cmd != (0.0, 0.0)
