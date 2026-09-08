"""Isaac Sim compatibility layer (FINDINGS XC-6).

Everything the module needs from Isaac Sim is imported HERE, as data on a
namespace object, so the 4.5 (``omni.isaac.*``) vs 5.0 (``isaacsim.*``) module
renames live in one place. Version- and capability-dependent behaviour is keyed
on :func:`isaac_version` / :func:`caps`, never on scattered ``try/except
ImportError`` in feature code.

Mock mode never imports Isaac: :func:`isaac_version` returns ``None`` and
:func:`caps` returns the entry for the newest known release.
"""

from __future__ import annotations

import importlib.metadata
import re
from typing import Any, NamedTuple

from viam.logging import getLogger

IsaacVersion = tuple[int, int, int]

_LOGGER = getLogger(__name__)
ASSET_RETRY_MAX_MS = 2000  # omni.client retry budget per URL; the default is 120,000

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def import_isaac() -> Any:
    """Import everything we need from isaac sim, tolerating the module
    renames across releases (isaacsim.* in >=4.5, omni.isaac.* before)."""

    class NS:
        pass

    ns: Any = NS()

    try:
        from isaacsim.core.api import World
    except ImportError:
        from omni.isaac.core import World
    ns.World = World

    try:
        from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage, open_stage
    ns.add_reference_to_stage = add_reference_to_stage
    ns.open_stage = open_stage

    try:
        from isaacsim.storage.native import get_assets_root_path
    except ImportError:
        from omni.isaac.core.utils.nucleus import get_assets_root_path
    ns.get_assets_root_path = get_assets_root_path

    try:
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
    except ImportError:
        from omni.isaac.core.articulations import Articulation as SingleArticulation
        from omni.isaac.core.prims import XFormPrim as SingleXFormPrim
    ns.SingleArticulation = SingleArticulation
    ns.SingleXFormPrim = SingleXFormPrim

    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction
    ns.ArticulationAction = ArticulationAction

    try:
        import omni.client

        ns.client = omni.client
        # The 2F-85 asset references part meshes on omniverse://isaac-dev...,
        # unreachable outside NVIDIA. omni.client retries a failing URL for
        # max_ms=120,000 by default (measured 2026-09-04: every gripper attach
        # spent exactly 120 s in that budget, failed viam-server's construction
        # deadline, and succeeded on the retry). Cap the budget so unresolvable
        # references fail fast; _rewrite_unresolvable_references re-points them
        # at the public bucket afterwards.
        try:
            previous = omni.client.set_retries(ASSET_RETRY_MAX_MS, 100, 100)
            _LOGGER.info("omni.client retries capped at %d ms (was %s)", ASSET_RETRY_MAX_MS, previous)
        except Exception:
            _LOGGER.exception("could not cap omni.client retries")
    except ImportError:
        ns.client = None

    try:
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    except ImportError:
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
    ns.DynamicCuboid = DynamicCuboid
    ns.FixedCuboid = FixedCuboid

    try:
        from isaacsim.core.utils.prims import get_prim_at_path
    except ImportError:
        try:
            from omni.isaac.core.utils.prims import get_prim_at_path
        except ImportError:
            get_prim_at_path = None
    ns.get_prim_at_path = get_prim_at_path

    try:
        from isaacsim.sensors.camera import Camera
    except ImportError:
        from omni.isaac.sensor import Camera
    ns.Camera = Camera

    try:
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot
    except ImportError:
        from omni.isaac.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from omni.isaac.wheeled_robots.robots import WheeledRobot
    ns.WheeledRobot = WheeledRobot
    ns.DifferentialController = DifferentialController

    # phase 3 (SCN-6 / ARM-2): physics materials and the USD physics schemas.
    # Exposed on the namespace (None when absent) so physics.py can be driven
    # with fakes on a machine without Kit.
    try:
        from isaacsim.core.api.materials import PhysicsMaterial

        ns.PhysicsMaterial = PhysicsMaterial
    except ImportError:
        try:
            from omni.isaac.core.materials import PhysicsMaterial as _PhysicsMaterial

            ns.PhysicsMaterial = _PhysicsMaterial
        except ImportError:
            ns.PhysicsMaterial = None
    try:
        from pxr import PhysxSchema, UsdPhysics

        ns.PhysxSchema = PhysxSchema
        ns.UsdPhysics = UsdPhysics
    except ImportError:
        ns.PhysxSchema = None
        ns.UsdPhysics = None

    return ns


def _parse_version(value: Any) -> IsaacVersion | None:
    """Tolerantly extract (major, minor, patch) from an unknown-shape value.

    ``get_version()``'s return shape is undocumented (FINDINGS OQ-14): a
    ``str`` is regexed directly; a sequence is first regexed after joining
    its stringified elements, then falls back to its first three int-like
    elements.
    """
    if isinstance(value, str):
        match = _VERSION_RE.search(value)
        return (int(match[1]), int(match[2]), int(match[3])) if match else None

    if isinstance(value, (list, tuple)):
        joined = " ".join(str(item) for item in value)
        match = _VERSION_RE.search(joined)
        if match:
            return (int(match[1]), int(match[2]), int(match[3]))

        parts: list[int] = []
        for item in value:
            try:
                parts.append(int(item))
            except (TypeError, ValueError):
                continue
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2])
        return None

    return None


def _probe_isaacsim_core_version() -> Any:
    from isaacsim.core.version import get_version

    return get_version()


def _probe_omni_isaac_version() -> Any:
    from omni.isaac.version import get_version

    return get_version()


def _probe_importlib_metadata() -> Any:
    return importlib.metadata.version("isaacsim")


_PROBES: tuple[Any, ...] = (
    _probe_isaacsim_core_version,
    _probe_omni_isaac_version,
    _probe_importlib_metadata,
)


def isaac_version() -> IsaacVersion | None:
    """Best-effort (major, minor, patch) of the installed Isaac Sim.

    ``None`` when Isaac is not importable (mock mode, unit tests) or when no
    known probe answers. The probe API is FINDINGS OQ-14 / RQ-23: try the
    candidates in order and stop at the first that works; never raise.
    """
    for probe in _PROBES:
        try:
            raw = probe()
        except Exception:  # any probe failure just tries the next
            continue
        parsed = _parse_version(raw)
        if parsed is not None:
            _LOGGER.info(
                "isaac_version: %s answered raw=%r parsed=%r",
                probe.__name__,
                raw,
                parsed,
            )
            return parsed
    return None


class Caps(NamedTuple):
    """Capability flags the feature code branches on (one row per release).

    Gripper fields are added in phase 3; keep the field order append-only.
    """

    has_depth_sensor: bool  # SingleViewDepthSensor exists (5.0+)
    pointcloud_is_world_frame: bool  # Camera.get_pointcloud() returns world-frame points
    camera_reads_cached_frame: bool  # get_depth() reads _current_frame (4.5), not the annotator
    # Robotiq 2F-85 asset differences per release (FINDINGS R-9, W13, OQ-18;
    # phase 3). Kept here so models/gripper.py never branches on the version.
    gripper_closed_deg: float  # finger_joint angle at full closure (open is 0)
    gripper_max_force: float  # authored finger drive maxForce - a tuning knob, not applied yet
    gripper_dof_count: int  # DOFs the gripper adds to the arm articulation (OQ-5: 4.5 unverified)
    # finger_joint angle at the OPEN limit: 0 on both releases (a 7.76 deg rest
    # seen on the GPU before the articulation fixes was an artifact; run 19
    # reaches 0.4 deg at an open target of 0)
    gripper_open_deg: float
    # CAM-12: Camera(annotator_device=...)/get_*(device=...) GPU-resident data
    # path exists only on 5.0 (CHANGELOG 0.4.0); 4.5 always lands in host numpy
    camera_supports_annotator_device: bool


CAPS_BY_RELEASE: dict[tuple[int, int], Caps] = {
    (4, 5): Caps(
        has_depth_sensor=False,
        pointcloud_is_world_frame=True,
        camera_reads_cached_frame=True,
        gripper_closed_deg=45.0,
        gripper_max_force=16.5,
        gripper_dof_count=10,
        gripper_open_deg=0.0,
        camera_supports_annotator_device=False,
    ),
    (5, 0): Caps(
        has_depth_sensor=True,
        pointcloud_is_world_frame=False,
        camera_reads_cached_frame=False,
        gripper_closed_deg=47.0,
        gripper_max_force=26.0,
        gripper_dof_count=6,
        gripper_open_deg=0.0,
        camera_supports_annotator_device=True,
    ),
}


def caps(version: IsaacVersion | None = None) -> Caps:
    """Flags for ``version`` (default: :func:`isaac_version`).

    Unknown/None → the newest known release's row; a version between two known
    rows → the nearest lower row. The flag table must carry both 4.5 and 5.0.
    """
    if version is None:
        version = isaac_version()

    releases = sorted(CAPS_BY_RELEASE)

    if version is None:
        return CAPS_BY_RELEASE[releases[-1]]

    major, minor = version[0], version[1]
    release = (major, minor)

    lower_or_equal = [row for row in releases if row <= release]
    if lower_or_equal:
        return CAPS_BY_RELEASE[lower_or_equal[-1]]

    return CAPS_BY_RELEASE[releases[0]]
