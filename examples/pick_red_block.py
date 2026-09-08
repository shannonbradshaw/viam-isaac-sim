"""Pick-red-block client for the mock/real Isaac Sim pick-and-place cell.

Orchestration lives OUTSIDE the module (DEC-4): this script does the sequencing
a real pick needs - detect, open, approach, grasp, lift, release - talking to
either a live Viam machine or, with ``--mock``, an in-process mock boot of the
module so the sequencing logic can be exercised without a GPU. ``MoveToPosition``
is never used (DEC-13); every move drives joint positions (mock) or the motion
service's ``move`` (real). Depends only on the stdlib, viam-sdk and numpy;
``isaac_module`` is imported lazily, only inside the ``--mock`` code path.

Usage (real machine, W36/XC-8)::

    python examples/pick_red_block.py --address <machine-address> \\
        --api-key <key> --api-key-id <key-id> \\
        --camera wrist-cam --arm pick-arm --gripper pick-grip \\
        --vision block-segmenter --motion builtin \\
        --block pick_cube --block-size-mm 60

Usage (in-process mock, no GPU, no running machine)::

    PYTHONPATH=src python examples/pick_red_block.py --mock

Real mode assumes the machine config carries the vision pipeline (color
detector -> detections-to-segments) and motion service from
``fragments/pick-and-place.json``, plus a gripper riding the arm's flange
(DEC-20 target block name is ``pick_cube``, not upstream's ``block_red``)::

    {
      "name": "pick-grip",
      "namespace": "rdk",
      "type": "gripper",
      "model": "viam:isaac-sim-devin:gripper",
      "frame": {"parent": "pick-arm", "translation": {"x": 0, "y": 0, "z": 115}},
      "attributes": {"world": "sim-world", "arm": "pick-arm"}
    },
    {
      "name": "builtin",
      "api": "rdk:service:motion",
      "model": "rdk:builtin:builtin",
      "attributes": {}
    },
    {
      "name": "red-detector",
      "api": "rdk:service:vision",
      "model": "color_detector",
      "attributes": {
        "detect_color": "#EA8D8D",
        "hue_tolerance_pct": 0.1,
        "segment_size_px": 100,
        "value_cutoff_pct": 0.15,
        "camera_name": "wrist-cam"
      }
    },
    {
      "name": "block-segmenter",
      "api": "rdk:service:vision",
      "model": "viam:vision:detections-to-segments",
      "attributes": {
        "detector_name": "red-detector",
        "camera_name": "wrist-cam",
        "mean_k": 5,
        "sigma": 1.25,
        "confidence_threshold_pct": 0.5,
        "infer_minimum_depth": true
      }
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import threading
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from google.protobuf.json_format import MessageToDict
from viam.components.arm import Arm, JointPositions
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Transform,
    Vector3,
    WorldState,
)
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

# ----------------------------------------------------------------------
# pure helpers - unit-testable without a robot (see tests/test_pick_red_block.py)
# ----------------------------------------------------------------------

MM_PER_M = 1000.0
PRE_GRASP_STANDOFF_MM = 100.0
POINTING_DOWN_O_Z = -1.0

TABLE_DIMS_MM: tuple[float, float, float] = (1200.0, 800.0, 740.0)
TABLE_CENTER_MM: tuple[float, float, float] = (600.0, 0.0, 370.0)

HELD_BLOCK_TRANSFORM_MARKER = "HELD_BLOCK_TRANSFORM_JSON="
DETECTED_BLOCK_POSE_MARKER = "DETECTED_BLOCK_POSE_JSON="
GRAB_DIAGNOSTICS_MARKER = "GRAB_DIAGNOSTICS_JSON="
MOVE_DIAGNOSTICS_MARKER = "MOVE_DIAGNOSTICS_JSON="
HOLD_SAMPLES_MARKER = "HOLD_SAMPLES_JSON="
RESET_MID_HOLD_MARKER = "RESET_MID_HOLD_JSON="
MEASURED_BLOCK_MARKER = "MEASURED_BLOCK_JSON="

# checklist item 5 wants the block held 100 mm up for 5 s; item 6 resets the
# world mid-hold and expects the post-reset hooks (ARM-15/XC-5) to keep it held
DEFAULT_HOLD_S = 5.0
HOLD_SAMPLE_S = 1.0
RESET_SETTLE_S = 2.0


def _pointing_down(x: float, y: float, z: float, theta_deg: float = 0.0) -> Pose:
    return Pose(x=x, y=y, z=z, o_x=0.0, o_y=0.0, o_z=POINTING_DOWN_O_Z, theta=theta_deg)


# default scan spot: inside UR5e reach with the shipped cell's blocks in the
# 90 deg field of view; the height is ABOVE THE SUPPORT, because an absolute
# scan z is a floor-cell assumption (GPU run 13: the P5 table cell sent the
# camera 400 mm below the table top)
DEFAULT_LOOK_XY_MM = (500.0, 150.0)
SCAN_HEIGHT_ABOVE_SUPPORT_MM = 350.0


def default_scan_pose(support_z_mm: float) -> Pose:
    x, y = DEFAULT_LOOK_XY_MM
    return _pointing_down(x, y, support_z_mm + SCAN_HEIGHT_ABOVE_SUPPORT_MM)


DEPTH_PROBE_RADIUS_MM = 20.0


def centre_depth_mm(
    points_xyz_m: np.ndarray, radius_mm: float = DEPTH_PROBE_RADIUS_MM
) -> float | None:
    """Median camera-frame depth (z, in mm) of the points within ``radius_mm``
    of the optical axis. Looking straight down from a known height, this is
    the camera's own height - so the ratio to the expected value is the depth
    scale error (GPU run 15: detections landed ~10% too far along the ray)."""
    if points_xyz_m.size == 0:
        return None
    xy_mm = points_xyz_m[:, :2] * 1000.0
    near_axis = np.hypot(xy_mm[:, 0], xy_mm[:, 1]) <= radius_mm
    if not near_axis.any():
        return None
    return float(np.median(points_xyz_m[near_axis, 2]) * 1000.0)


def look_pose_from(xyz_mm: str) -> Pose:
    """The wrist-camera pose to detect from: at ``x,y,z`` (mm, world) with the
    optical axis pointing straight down (o_z = -1), so the block region is in
    view. At the UR5e zero pose the camera looks along world -Y, away from it."""
    x, y, z = (float(v) for v in xyz_mm.split(","))
    return Pose(x=x, y=y, z=z, o_x=0.0, o_y=0.0, o_z=POINTING_DOWN_O_Z, theta=0.0)


def _look_pose_from_args(args: argparse.Namespace) -> Pose | None:
    if args.no_look:
        return None
    if args.look_at:
        return look_pose_from(args.look_at)
    return default_scan_pose(args.support_z_mm)


def pre_grasp_pose(block_pose: Pose, standoff_mm: float = PRE_GRASP_STANDOFF_MM) -> Pose:
    """The stationary pose the arm detects from and lifts back to: directly
    above the block by ``standoff_mm``, gripper pointing straight down."""
    return _pointing_down(block_pose.x, block_pose.y, block_pose.z + standoff_mm)


# Measured on the GPU (phase 3, item 4): the 2F-85 pads reach 153 mm along the
# tool axis with the TCP at their centre, 134 mm, so the fingertips extend 19 mm
# past the TCP. A grasp lower than support + overhang drives them into the table.
FINGERTIP_OVERHANG_MM = 19.0
# GPU run 19: the arm model and Isaac disagree by ~10-15 mm in z at the grasp
# configuration, and the pads (38 mm tall) still cover 2/3 of a 60 mm block
# when the TCP sits 39 mm up - so leave real room above the support.
FINGERTIP_CLEARANCE_MM = 20.0


def grasp_height_mm(
    detected_z_mm: float,
    block_size_mm: float,
    support_z_mm: float,
    fingertip_overhang_mm: float = FINGERTIP_OVERHANG_MM,
    clearance_mm: float = FINGERTIP_CLEARANCE_MM,
) -> float:
    """The TCP height to grasp at. A block resting on its support cannot have
    its centre below support + size/2 (a depth centroid seen from above lands
    low), and the TCP cannot go below support + overhang + clearance without
    the fingertips hitting the support."""
    centre_floor = support_z_mm + block_size_mm / 2.0
    fingertip_floor = support_z_mm + fingertip_overhang_mm + clearance_mm
    return max(detected_z_mm, centre_floor, fingertip_floor)


# 2F-85 opens 85 mm (W13/W15); 10 mm covers the closed fingers' own
# clearance, so a measured block wider than this cannot be grasped.
JAW_MAX_BLOCK_MM = 75.0


# The frame system and Isaac disagree by ~17-19 mm in z at the grasp
# configuration while agreeing at the look pose (GPU runs 19-22, ARM-10
# follow-up). Measured at the pre-grasp pose (believed TCP vs the physical pad
# centre) and applied to the grasp/lift targets; capped so a bad reading
# cannot command a wild pose.
TCP_CORRECTION_CAP_MM = 40.0


def corrected_pose(
    pose: Pose, delta_mm: tuple[float, float, float], cap_mm: float = TCP_CORRECTION_CAP_MM
) -> Pose:
    """``pose`` shifted by the measured believed-minus-physical TCP offset, so
    the physical pads land where the plan intended. Each axis is clamped to
    +/- cap_mm."""
    dx, dy, dz = (max(-cap_mm, min(cap_mm, v)) for v in delta_mm)
    return Pose(
        x=pose.x + dx,
        y=pose.y + dy,
        z=pose.z + dz,
        o_x=pose.o_x,
        o_y=pose.o_y,
        o_z=pose.o_z,
        theta=pose.theta,
    )


def with_z(pose: Pose, z_mm: float) -> Pose:
    return Pose(
        x=pose.x, y=pose.y, z=z_mm, o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta
    )


def grasp_pose(block_pose: Pose) -> Pose:
    """The block's own pose, gripper pointing straight down - no standoff."""
    return _pointing_down(block_pose.x, block_pose.y, block_pose.z)


def table_obstacle() -> Geometry:
    """W4/README "Table recipe": 1200 x 800 x 740 mm box centred at
    (600, 0, 370) mm in the world frame, the motion-planner obstacle for the
    table (10 mm below the real 750 mm surface so the arm can rest on it)."""
    x, y, z = TABLE_CENTER_MM
    dim_x, dim_y, dim_z = TABLE_DIMS_MM
    return Geometry(
        center=Pose(x=x, y=y, z=z, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        box=RectangularPrism(dims_mm=Vector3(x=dim_x, y=dim_y, z=dim_z)),
        label="table",
    )


SUPPORT_OBSTACLE_SIDE_MM = 3000.0
# thick: a 20 mm slab let a discrete collision check step a link straight
# through the floor mid-swing (GPU run 7)
SUPPORT_OBSTACLE_THICKNESS_MM = 200.0


def support_obstacle(support_z_mm: float) -> Geometry:
    """The surface the block rests on, as a thin slab whose top is at
    support_z_mm. Without it the planner happily swings the fingertips through
    the floor on a joint-space descent (GPU run 18)."""
    return Geometry(
        center=Pose(
            x=0.0,
            y=0.0,
            z=support_z_mm - SUPPORT_OBSTACLE_THICKNESS_MM / 2.0,
            o_x=0.0,
            o_y=0.0,
            o_z=1.0,
            theta=0.0,
        ),
        box=RectangularPrism(
            dims_mm=Vector3(
                x=SUPPORT_OBSTACLE_SIDE_MM,
                y=SUPPORT_OBSTACLE_SIDE_MM,
                z=SUPPORT_OBSTACLE_THICKNESS_MM,
            )
        ),
        label="support",
    )


def obstacles_from_prop_geometries(
    geometries: Sequence[Mapping[str, Any]], exclude: set[str]
) -> list[Geometry]:
    """Sim obstacle boxes from the world's ``prop_geometries`` DoCommand
    result: one box per entry, skipping ``exclude`` (the block about to be
    grasped) and any entry whose dims are all zero (an unknown-size usd prop
    the module could not infer a box for)."""
    obstacles: list[Geometry] = []
    for geometry in geometries:
        name = geometry["name"]
        if name in exclude:
            continue
        dim_x, dim_y, dim_z = geometry["box_dims_mm"]
        if dim_x == 0.0 and dim_y == 0.0 and dim_z == 0.0:
            continue
        pose_mm = geometry["pose_in_world_mm"]
        obstacles.append(
            Geometry(
                center=Pose(
                    x=pose_mm["x"],
                    y=pose_mm["y"],
                    z=pose_mm["z"],
                    o_x=pose_mm["o_x"],
                    o_y=pose_mm["o_y"],
                    o_z=pose_mm["o_z"],
                    theta=pose_mm["theta"],
                ),
                box=RectangularPrism(dims_mm=Vector3(x=dim_x, y=dim_y, z=dim_z)),
                label=name,
            )
        )
    return obstacles


def table_recipe_unless_served(
    table: Geometry | None, sim_obstacles: Sequence[Geometry]
) -> Geometry | None:
    """The --table recipe box, or None when the live scene already serves a
    geometry with the same label - the motion service rejects two WorldState
    geometries sharing a name, and the P5 cell's table arrives live via
    ``prop_geometries``."""
    if table is None:
        return None
    if any(obstacle.label == table.label for obstacle in sim_obstacles):
        return None
    return table


RANDOMIZE_REGION_MARGIN_MM = 50.0

# placing: the held block hovers this gap above the pad top at release and
# drops it - the pad stays a planner obstacle, so the gripper never touches it
PLACE_CLEARANCE_MM = 15.0
PLACE_XY_TOLERANCE_MM = 100.0  # verdict: block centre inside the 200 mm pad footprint
PLACE_Z_TOLERANCE_MM = 10.0
PLACE_SETTLE_S = 1.0  # wall-clock pause before reading the placed pose back
# extra size on the held block's planning box: absorbs the ~4 mm believed-vs-
# physical TCP gap (ARM-10) plus tracking error, so a grazing plan cannot
# become a touch
HELD_BLOCK_PADDING_MM = 20.0

# the pick-area keep-out (GPU run 12): a no-fly box over the scatter region
# lets the carry plan FREELY (fast joint-space motion) instead of crawling
# along a constrained linear line - the planner simply may not enter the
# airspace where blocks live. Height above the support = block tops (60) +
# held-cube hang + margin; the region's own z is the support it stands on.
KEEPOUT_HEIGHT_MM = 130.0
KEEPOUT_MARGIN_MM = 50.0
# TCP height above the support whose held padded cube bottom (TCP - 9 offset
# - 40 half-box) clears the keep-out ceiling with ~20 mm to spare
CARRY_CLEAR_ABOVE_SUPPORT_MM = 200.0


def pick_area_keepout(
    region_mm: tuple[Sequence[float], Sequence[float]],
    height_mm: float = KEEPOUT_HEIGHT_MM,
) -> Geometry:
    """The carry-phase no-fly box over the scatter region: from the region's
    own z (the support surface) up ``height_mm`` (KEEPOUT_HEIGHT_MM by
    default; phase 4 passes the measured-tallest-derived height), grown by
    KEEPOUT_MARGIN_MM sideways."""
    (x0, y0, z0), (x1, y1, _z1) = region_mm
    lo_x = min(x0, x1) - KEEPOUT_MARGIN_MM
    hi_x = max(x0, x1) + KEEPOUT_MARGIN_MM
    lo_y = min(y0, y1) - KEEPOUT_MARGIN_MM
    hi_y = max(y0, y1) + KEEPOUT_MARGIN_MM
    return Geometry(
        center=Pose(
            x=(lo_x + hi_x) / 2.0,
            y=(lo_y + hi_y) / 2.0,
            z=z0 + height_mm / 2.0,
            o_x=0.0,
            o_y=0.0,
            o_z=1.0,
            theta=0.0,
        ),
        box=RectangularPrism(
            dims_mm=Vector3(x=hi_x - lo_x, y=hi_y - lo_y, z=height_mm)
        ),
        label="pick_area_keepout",
    )
PLACED_BLOCK_MARKER = "PLACED_BLOCK_JSON="
MEASURED_TALLEST_MARKER = "MEASURED_TALLEST_JSON="
# refine the look when the detected block sits farther than this from the scan
# centre; closer than this, a second measurement gains nothing
FOCUS_LOOK_OFFSET_MM = 30.0
# a resting block's centre must sit at support + size/2; a reading farther off
# than this is not the block (GPU run 10: a gripper-shadowed cube read z 115)
DETECT_Z_TOLERANCE_MM = 15.0
# retry ladder for a gripper-shadowed detection: the fingers hang ~150 mm
# below the wrist camera, ~80 mm off its axis, so a quarter wrist turn sweeps
# the shadow to a new direction while the camera stays put; stepping the
# camera sideways is the last resort. Entries are (x offset, y offset, theta).
SCAN_ATTEMPTS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 90.0),
    (0.0, 0.0, 180.0),
    (0.0, 0.0, 270.0),
    (150.0, 150.0, 0.0),
    (-150.0, -150.0, 0.0),
)

# tallest-estimator trust thresholds (seam: phase-4-tallest-carry.md, "Client
# measurement API")
# the fragment segmenter's segment_size_px: 100, the cell's smallest-credible-
# object constant - fewer in-region above-support points is not a block
MIN_TALLEST_REGION_POINTS = 100
# points at/below the support plus this are sensor noise, not object height
TALLEST_SUPPORT_EPSILON_MM = 1.0
# a 30 mm face at 900 mm range through 848 px / 90.5 deg intrinsics yields
# hundreds of points, so requiring 5 within this band of the max z is
# conservative - a lone stray point is not a block top
TALLEST_TOP_BAND_MM = 10.0
MIN_TALLEST_TOP_POINTS = 5


@dataclass
class TallestEstimate:
    tallest_mm: float
    points: int  # in-region points above the support plane
    trusted: bool
    reasons: list[str]  # empty when trusted


def tallest_in_region_mm(
    xyz_world_mm: np.ndarray,
    region_mm: tuple[Sequence[float], Sequence[float]] | None,
    support_z_mm: float,
    size_range_mm: tuple[float, float],
) -> TallestEstimate:
    """Tallest object height above ``support_z_mm`` in ``xyz_world_mm``
    (world frame, mm): clipped to ``region_mm``'s x/y footprint when given
    (None skips the clip and the quadrant-coverage check below), points at or
    below the support dropped before taking the max. Four independent trust
    checks each append a distinct reason on failure; ``trusted`` is true only
    when none do."""
    if region_mm is not None:
        (x0, y0, _z0), (x1, y1, _z1) = region_mm
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        in_region = (
            (xyz_world_mm[:, 0] >= lo_x)
            & (xyz_world_mm[:, 0] <= hi_x)
            & (xyz_world_mm[:, 1] >= lo_y)
            & (xyz_world_mm[:, 1] <= hi_y)
        )
        region_points = xyz_world_mm[in_region]
    else:
        region_points = xyz_world_mm

    reasons: list[str] = []

    if region_mm is not None:
        mid_x = (lo_x + hi_x) / 2.0
        mid_y = (lo_y + hi_y) / 2.0
        # counted BEFORE the support drop: a quadrant shadowed by a near
        # block has no support returns either - the side view's real failure
        quadrant_counts = [
            int(
                (
                    (region_points[:, 0] < mid_x if left else region_points[:, 0] >= mid_x)
                    & (region_points[:, 1] < mid_y if bottom else region_points[:, 1] >= mid_y)
                ).sum()
            )
            for left in (True, False)
            for bottom in (True, False)
        ]
        if any(count < 1 for count in quadrant_counts):
            reasons.append("region-quadrant coverage: a footprint quadrant has no in-region points")

    above_support = region_points[:, 2] > support_z_mm + TALLEST_SUPPORT_EPSILON_MM
    above_points = region_points[above_support]
    points = int(len(above_points))
    if points < MIN_TALLEST_REGION_POINTS:
        reasons.append(
            f"point floor: {points} in-region above-support points < {MIN_TALLEST_REGION_POINTS}"
        )

    tallest_mm = float(above_points[:, 2].max()) - support_z_mm if points > 0 else 0.0

    lo_size, hi_size = size_range_mm
    widened_lo = lo_size - DETECT_Z_TOLERANCE_MM
    widened_hi = hi_size + DETECT_Z_TOLERANCE_MM
    if not (widened_lo <= tallest_mm <= widened_hi):
        reasons.append(
            f"size window: tallest {tallest_mm:.1f} mm outside [{widened_lo:.1f}, {widened_hi:.1f}]"
        )

    near_top = (
        int((above_points[:, 2] >= float(above_points[:, 2].max()) - TALLEST_TOP_BAND_MM).sum())
        if points > 0
        else 0
    )
    if near_top < MIN_TALLEST_TOP_POINTS:
        reasons.append(
            f"lone-point top: {near_top} points within {TALLEST_TOP_BAND_MM} mm of the max z "
            f"< {MIN_TALLEST_TOP_POINTS}"
        )

    return TallestEstimate(tallest_mm=tallest_mm, points=points, trusted=not reasons, reasons=reasons)


# keep-out/carry derivation (seam): tallest + held-cube hang + margin. The
# hang fraction reproduces today's GPU-validated 60 mm-block numbers
# (keepout_height_mm(60, 60) == 130, carry_clear_above_support_mm(60, 60) ==
# 200); a held cube of a different size re-validates on GPU (phase 4
# checklist item 2).
KEEPOUT_HELD_HANG_FRACTION = 1.0 / 3.0
# the held padded cube's bottom clears the keep-out ceiling by this much once
# carried (GPU run 12: "~20 mm to spare" for the 60 mm case)
CARRY_KEEPOUT_CLEARANCE_MM = 21.0
# believed-vs-physical TCP gap already named at CARRY_CLEAR_ABOVE_SUPPORT_MM's
# definition (ARM-10)
CARRY_TCP_TO_CUBE_BOTTOM_OFFSET_MM = 9.0


def keepout_height_mm(tallest_mm: float, held_size_mm: float) -> float:
    """Pick-area keep-out ceiling height above the support: the tallest
    scattered object, plus room for the held cube's hang, plus
    KEEPOUT_MARGIN_MM reused as vertical margin."""
    return tallest_mm + held_size_mm * KEEPOUT_HELD_HANG_FRACTION + KEEPOUT_MARGIN_MM


def carry_clear_above_support_mm(tallest_mm: float, held_size_mm: float) -> float:
    """TCP height for the free-carry hop: the keep-out ceiling top, plus
    CARRY_KEEPOUT_CLEARANCE_MM, plus the held padded cube's own half-height
    and TCP offset so its bottom face clears the ceiling."""
    half_padded_held_mm = (held_size_mm + HELD_BLOCK_PADDING_MM) / 2.0
    return (
        keepout_height_mm(tallest_mm, held_size_mm)
        + CARRY_KEEPOUT_CLEARANCE_MM
        + CARRY_TCP_TO_CUBE_BOTTOM_OFFSET_MM
        + half_padded_held_mm
    )


TALLEST_SWEEP_CORNER_INSET_MM = 50.0


def tallest_sweep_attempts(
    region_mm: tuple[Sequence[float], Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    """The wrist-sweep fallback ladder for tallest measurement: 4 region-corner
    vantages (offsets from the region centre, inset
    TALLEST_SWEEP_CORNER_INSET_MM from each corner, theta 0) FIRST, then
    SCAN_ATTEMPTS. Corners lead because a region-centre vantage hangs the
    camera's own gripper inside the region footprint, where it reads as a
    ~274 mm object (GPU phase-4 run 1: all four centre vantages discarded);
    a corner vantage keeps the arm outside the footprint."""
    (x0, y0, _z0), (x1, y1, _z1) = region_mm
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    centre_x = (lo_x + hi_x) / 2.0
    centre_y = (lo_y + hi_y) / 2.0
    inset = TALLEST_SWEEP_CORNER_INSET_MM
    corners = (
        (lo_x + inset - centre_x, lo_y + inset - centre_y, 0.0),
        (hi_x - inset - centre_x, lo_y + inset - centre_y, 0.0),
        (lo_x + inset - centre_x, hi_y - inset - centre_y, 0.0),
        (hi_x - inset - centre_x, hi_y - inset - centre_y, 0.0),
    )
    return corners + SCAN_ATTEMPTS


def randomize_region_mm(
    margin_mm: float = RANDOMIZE_REGION_MARGIN_MM,
    face_z_mm: float = 0.0,
) -> tuple[list[float], list[float]]:
    """The table-footprint rectangle (table_obstacle's own x/y constants),
    inset by ``margin_mm`` so a randomized block cannot land hanging off the
    edge, at ``face_z_mm``: the surface the blocks rest on - the floor in the
    current fragment (``support_z_mm``), the table top in the P5 cell. The
    region ``randomize_props`` scatters the movable blocks within (checklist
    item 1: two consecutive picks with re-randomised blocks)."""
    center_x, center_y, _center_z = TABLE_CENTER_MM
    dim_x, dim_y, _dim_z = TABLE_DIMS_MM
    half_x = dim_x / 2.0 - margin_mm
    half_y = dim_y / 2.0 - margin_mm
    return (
        [center_x - half_x, center_y - half_y, face_z_mm],
        [center_x + half_x, center_y + half_y, face_z_mm],
    )


def pad_top_centre_mm(
    geometries: Sequence[Mapping[str, Any]], pad_name: str
) -> tuple[float, float, float] | None:
    """The place pad's top-face centre (x, y, top z) in mm from a
    ``prop_geometries`` result, or None when the scene has no such prop."""
    for geometry in geometries:
        if geometry["name"] == pad_name:
            pose = geometry["pose_in_world_mm"]
            return (pose["x"], pose["y"], pose["z"] + geometry["box_dims_mm"][2] / 2.0)
    return None


# the arm base sits at the world origin in this fragment; a UR5e reaches
# ~850 mm and phase 3 verified a pick at 743 mm radius (block at 700, 250)
REACHABLE_REGION_X_MM = (450.0, 700.0)
REACHABLE_REGION_Y_MM = (-250.0, 250.0)


def reachable_region_mm(face_z_mm: float = 0.0) -> tuple[list[float], list[float]]:
    """The scatter rectangle a randomized block stays pickable in: inside the
    arm's reach envelope, resting on ``face_z_mm``. The table-footprint recipe
    (randomize_region_mm) suits the P5 arm-on-table cell instead - on this
    floor cell its far corners are beyond UR5e reach (GPU run 4: a block at
    (846, 242) = 880 mm radius could not be looked at or picked)."""
    (x0, x1), (y0, y1) = REACHABLE_REGION_X_MM, REACHABLE_REGION_Y_MM
    return ([x0, y0, face_z_mm], [x1, y1, face_z_mm])


def world_state(
    table: Geometry | None,
    other_blocks: Sequence[Geometry] = (),
    support: Geometry | None = None,
    transforms: Sequence[Transform] = (),
) -> WorldState:
    """A WorldState whose obstacles are the support surface, the table (when
    the scene has one) and any distractor blocks (SCN-5), all in the world
    frame. ``transforms`` carries the held block while grasping (DEC-14), so
    the planner treats it as geometry attached to the gripper."""
    return WorldState(
        obstacles=[
            GeometriesInFrame(
                reference_frame="world",
                geometries=[g for g in (support, table, *other_blocks) if g is not None],
            )
        ],
        transforms=list(transforms),
    )


def held_block_transform(
    block_name: str, block_size_mm: float, gripper_name: str, centre_below_tcp_mm: float = 0.0
) -> Transform:
    """DEC-14(a): once grasped, the block's pose is reported to the motion
    service as a Transform parented to the gripper (the gripper's own
    GetGeometries cannot carry it - see DEC-14's rationale).
    ``centre_below_tcp_mm`` hangs the box below the gripper frame by the
    grasp offset, so the planner carries it at its real height (GPU run 9:
    an unmodelled held cube clipped a distractor the arm itself cleared)."""
    origin = Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    held_centre = Pose(
        x=0.0, y=0.0, z=-centre_below_tcp_mm, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0
    )
    return Transform(
        reference_frame=block_name,
        pose_in_observer_frame=PoseInFrame(reference_frame=gripper_name, pose=held_centre),
        physical_object=Geometry(
            center=origin,
            box=RectangularPrism(
                dims_mm=Vector3(x=block_size_mm, y=block_size_mm, z=block_size_mm)
            ),
            label=block_name,
        ),
    )


def is_red_point(rgb: tuple[int, int, int], threshold: float = 0.5) -> bool:
    """Same rule as gpu_checklist_camera.is_red_pixel: r is at least 100 and
    both g and b are at most `threshold` fractions of r."""
    r, g, b = rgb
    return r >= 100 and g <= r * threshold and b <= r * threshold


def red_centroid_m(
    points_xyz: np.ndarray, colors_rgb: np.ndarray, threshold: float = 0.5
) -> tuple[float, float, float]:
    """Mean xyz (metres) of the points whose colour is "red" (see
    is_red_point). Raises ValueError when no point matches."""
    red_mask = np.array(
        [is_red_point((int(r), int(g), int(b)), threshold) for r, g, b in colors_rgb]
    )
    if not red_mask.any():
        raise ValueError("no red points found in the point cloud")
    centroid = points_xyz[red_mask].mean(axis=0)
    return (float(centroid[0]), float(centroid[1]), float(centroid[2]))


# points within this much of the nearest depth are "the top face"; 10 mm let the
# near side face's top edge in and pulled x 30 mm toward the camera (GPU run 18)
TOP_FACE_BAND_M = 0.002
MIN_BLOCK_DEPTH_M = 0.15  # nearer than this is the gripper itself, not the scene
MIN_RED_BAND_FRACTION = 0.3  # red points in the band are trusted only when there are enough
LINEAR_LINE_TOLERANCE_MM = 10.0
LINEAR_ORIENTATION_TOLERANCE_DEG = 10.0


def top_face_centre_m(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    band_m: float = TOP_FACE_BAND_M,
    red_threshold: float = 0.5,
) -> tuple[float, float, float] | None:
    """Camera-frame centre (m) of the block's TOP FACE from a segment point
    cloud: keep red points (a detections-to-segments box also contains floor
    around the block, which drags a plain centroid away from the camera and
    down - GPU runs 13-16), then the nearest-depth band, i.e. the face seen
    straight down from the look pose. None when nothing red is left."""
    far_enough = xyz[:, 2] >= MIN_BLOCK_DEPTH_M  # the fingers sit ~90 mm in front of the camera
    if not far_enough.any():
        return None
    nearest = float(xyz[far_enough, 2].min())
    in_band = far_enough & (xyz[:, 2] <= nearest + band_m)
    chosen = in_band
    if rgb is not None and len(rgb) == len(xyz):
        # the lit top face can wash out past the red test (GPU run 21: only a
        # vertical face passed it), so red is a tie-breaker inside the band,
        # never the way to find the band
        r = rgb[:, 0].astype(float)
        red = (r >= 100) & (rgb[:, 1] <= r * red_threshold) & (rgb[:, 2] <= r * red_threshold)
        red_in_band = in_band & red
        if red_in_band.sum() >= MIN_RED_BAND_FRACTION * in_band.sum():
            chosen = red_in_band
    centre = xyz[chosen].mean(axis=0)
    return (float(centre[0]), float(centre[1]), float(centre[2]))


FOOTPRINT_TRIM_PCT_LO = 2.0
FOOTPRINT_TRIM_PCT_HI = 98.0
# a cube's three independent size readings (footprint x, footprint y,
# height) should agree; a bigger spread means a shadowed or edge-on view,
# not a block to grasp on (seam decision, phase 3)
MEASURED_SIZE_DEGENERATE_FRACTION = 0.25


def footprint_extents_mm(
    xyz: np.ndarray, band_m: float = TOP_FACE_BAND_M
) -> tuple[float, float] | None:
    """Top-face x/y footprint (mm) of the focused segment: the segment is
    already the detected block (the segmenter cuts it out of the detector's
    box), so measure the nearest-depth band - the same points
    top_face_centre_m trusts - each axis trimmed to the 2nd-98th percentile
    so a stray point cannot blow out the extent. Never select by redness:
    the lit top face washes out past any red test (GPU run 21; the phase-3
    checklist run saw red: 0 on every top-down scan). None when no point
    clears MIN_BLOCK_DEPTH_M."""
    far_enough = xyz[:, 2] >= MIN_BLOCK_DEPTH_M
    if not far_enough.any():
        return None
    nearest = float(xyz[far_enough, 2].min())
    band_xy = xyz[far_enough & (xyz[:, 2] <= nearest + band_m), :2]
    lo = np.percentile(band_xy, FOOTPRINT_TRIM_PCT_LO, axis=0)
    hi = np.percentile(band_xy, FOOTPRINT_TRIM_PCT_HI, axis=0)
    # the trim shaves 4% of a uniformly sampled extent (GPU: 57.3 mm measured
    # on a true 60 mm face) - rescale so a clean face measures true size
    trim_fraction = (FOOTPRINT_TRIM_PCT_HI - FOOTPRINT_TRIM_PCT_LO) / 100.0
    extent_mm = (hi - lo) * MM_PER_M / trim_fraction
    return (float(extent_mm[0]), float(extent_mm[1]))


def measured_block_size_mm(estimates: Sequence[float]) -> tuple[float, list[float]] | None:
    """Cube-prior size estimate: the median of independent size readings (a
    real detection uses footprint x, footprint y and height). None (a
    degenerate view) when any estimate strays more than
    MEASURED_SIZE_DEGENERATE_FRACTION of the median from it - advance the
    scan ladder instead of grasping on a bad number."""
    values = [float(v) for v in estimates]
    if any(v <= 0 for v in values):
        return None
    size_mm = float(np.median(values))
    if any(abs(v - size_mm) > MEASURED_SIZE_DEGENERATE_FRACTION * size_mm for v in values):
        return None
    return size_mm, values


def segment_stats(
    xyz: np.ndarray, rgb: np.ndarray | None, band_m: float = TOP_FACE_BAND_M
) -> dict[str, Any]:
    """What a segment is made of, camera frame: point counts (all / red /
    nearest-depth band) and the red points' extents in mm. Printed at detect
    so a biased block pose can be read off the segment's shape."""
    stats: dict[str, Any] = {"points": int(len(xyz)), "red": 0, "band": 0}
    if xyz.size == 0:
        return stats
    far_enough = xyz[:, 2] >= MIN_BLOCK_DEPTH_M
    scene = xyz[far_enough] if far_enough.any() else xyz
    nearest = float(scene[:, 2].min())
    in_band = scene[:, 2] <= nearest + band_m
    stats["band"] = int(in_band.sum())
    stats["band_min_mm"] = [round(float(v) * 1000.0, 1) for v in scene[in_band].min(axis=0)]
    stats["band_max_mm"] = [round(float(v) * 1000.0, 1) for v in scene[in_band].max(axis=0)]
    if rgb is not None and len(rgb) == len(xyz):
        r = rgb[:, 0].astype(float)
        red = (r >= 100) & (rgb[:, 1] <= r * 0.5) & (rgb[:, 2] <= r * 0.5)
        stats["red"] = int(red.sum())
        if red.any():
            stats["red_min_mm"] = [round(float(v) * 1000.0, 1) for v in xyz[red].min(axis=0)]
            stats["red_max_mm"] = [round(float(v) * 1000.0, 1) for v in xyz[red].max(axis=0)]
    return stats


def parse_pcd(data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse binary `pointcloud/pcd` bytes written by
    isaac_module.encoding.xyz_rgb_to_pcd back into (xyz metres, rgb uint8 or
    None). Mirrors that function's header/body layout exactly."""
    data_marker = b"DATA binary\n"
    header_end = data.index(data_marker) + len(data_marker)
    header_fields: dict[str, list[str]] = {}
    for line in data[:header_end].decode("ascii").splitlines():
        key, _, rest = line.partition(" ")
        header_fields[key] = rest.split()

    field_names = header_fields["FIELDS"]
    num_points = int(header_fields["POINTS"][0])
    coloured = "rgb" in field_names

    if coloured:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    else:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

    points = np.frombuffer(data, dtype=dtype, count=num_points, offset=header_end)
    xyz = np.stack([points["x"], points["y"], points["z"]], axis=-1).astype(np.float32)
    if not coloured:
        return xyz, None

    packed = points["rgb"].astype(np.uint32)
    rgb = np.stack([(packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF], axis=-1).astype(
        np.uint8
    )
    return xyz, rgb


def _pose_to_dict(pose: Pose) -> dict[str, float]:
    return {
        "x": pose.x,
        "y": pose.y,
        "z": pose.z,
        "o_x": pose.o_x,
        "o_y": pose.o_y,
        "o_z": pose.o_z,
        "theta": pose.theta,
    }


# ----------------------------------------------------------------------
# collaborators - the pipeline only depends on these two seams plus the
# gripper's own Viam API (open/grab/is_holding_something)
# ----------------------------------------------------------------------


class Detector(Protocol):
    async def block_pose_world(self) -> Pose:
        """The target block's pose (mm), in the world frame, detected from a
        stationary pre-grasp pose (motion blur)."""
        ...


class Mover(Protocol):
    async def look_from(self, pose: Pose, world_state: WorldState) -> None:
        """Move the wrist CAMERA frame to ``pose`` (world, mm) so the block is
        in view before detection; the arm's zero pose looks away from it."""
        ...

    async def move_to(self, pose: Pose, world_state: WorldState, linear: bool = False) -> None:
        """Move the gripper's TCP frame to `pose` (mm, world frame). ``linear``
        asks for a straight-line approach (the grasp descent and the lift)."""
        ...


class GripperApi(Protocol):
    async def open(self) -> None: ...

    async def grab(self) -> bool: ...

    async def is_holding_something(self) -> Any: ...


class WorldApi(Protocol):
    async def do_command(self, command: Mapping[str, Any]) -> Mapping[str, Any]:
        """The world component's DoCommand (a Generic client in real mode,
        the world model itself in mock) - used here for ``prop_geometries``
        and ``randomize_props``."""
        ...


class TallestScanner(Protocol):
    async def scan_world_mm(self) -> np.ndarray:
        """World-frame mm points (N x 3) of the scanner's current view, fed
        to ``tallest_in_region_mm``."""
        ...


async def camera_world_transform_mm(robot: Any, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(R 3x3, t mm) camera-to-world affine from 4 ``transform_pose`` probes:
    the camera-frame origin and +100 mm x/y/z offsets. Applied as
    ``world_mm = xyz_cam_m * 1000 @ R.T + t`` - no orientation-vector math,
    just the measured effect of the frame transform on known offsets."""

    async def probe(x_mm: float, y_mm: float, z_mm: float) -> np.ndarray:
        camera_pif = PoseInFrame(
            reference_frame=camera_name,
            pose=Pose(x=x_mm, y=y_mm, z=z_mm, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        )
        pose = (await robot.transform_pose(camera_pif, "world")).pose
        return np.array([pose.x, pose.y, pose.z])

    origin = await probe(0.0, 0.0, 0.0)
    x_probe = await probe(100.0, 0.0, 0.0)
    y_probe = await probe(0.0, 100.0, 0.0)
    z_probe = await probe(0.0, 0.0, 100.0)
    rotation = np.column_stack(
        [(x_probe - origin) / 100.0, (y_probe - origin) / 100.0, (z_probe - origin) / 100.0]
    )
    return rotation, origin


@dataclass
class PickPipeline:
    """W36's sequence: look -> detect -> open -> pre-grasp -> grasp -> grab -> lift ->
    held-block Transform -> place on the pad (when the scene has one) -> open.
    Orchestration only - no IK, no planning."""

    detector: Detector
    mover: Mover
    gripper: GripperApi
    block_name: str
    # None = measure from the focused segment's point cloud (the default);
    # a float overrides the measurement end to end (today's fixed-size path)
    block_size_mm: float | None
    gripper_name: str
    table: Geometry | None = None  # W4's table exists only in the P5 cell: opt in with --table
    other_blocks: Sequence[Geometry] = ()
    # the live sim is the default source of obstacle blocks (prop_geometries);
    # None keeps the pipeline to table/support/other_blocks only (mock's world
    # has no configured props, so this is a no-op there without --randomize-seed)
    world: WorldApi | None = None
    target_prop_name: str | None = None  # excluded from sim obstacles and from randomize_props
    # blocks randomised together with the target; empty = derive every
    # non-fixed prop with known dims from the live scene (distractors too)
    movable_prop_names: Sequence[str] = ()
    randomize_seed: int | None = None  # checklist item 1: re-randomise before this pick
    randomize_region_mm: tuple[Sequence[float], Sequence[float]] | None = None
    # adds size_range_mm to the randomize_props payload for every movable
    # name (None = today's payload, byte-identical)
    randomize_size_range_mm: tuple[float, float] | None = None
    # Measured over seeds 0-99, six 60 mm cubes in [450, 700] x [-250, 250] mm:
    # 200 mm succeeds 0/100, 140 mm succeeds 100/100. W26's gripper-clearance
    # intent still holds at 140 (worst-case face gap 80 mm vs 12.5 mm jaw
    # overhang; the 2F-85 opens 85 mm).
    randomize_min_separation_mm: float = 140.0
    # centre of the scatter region a randomize run used; the camera scans the
    # workspace from above it - the DETECTOR finds the block, never sim truth
    scan_centre_mm: tuple[float, float] | None = None
    # the scatter region itself; when known, the carry gets a keep-out box
    # over it instead of the slow constrained linear traverse
    pick_region_mm: tuple[Sequence[float], Sequence[float]] | None = None
    # the mock detector reports a camera-frame z (mock mode has no frame
    # system), so the resting-height gate only applies to real detections
    verify_detection_height: bool = True
    # the fixed prop to set the block down on; None = release at the lift pose
    place_prop_name: str | None = None
    place_clearance_mm: float = PLACE_CLEARANCE_MM
    # pad top-face centre (x, y, top z) in mm; set by _sim_obstacles when found
    place_pad_top_mm: tuple[float, float, float] | None = None
    standoff_mm: float = PRE_GRASP_STANDOFF_MM
    look_pose: Pose | None = None  # None = detect from wherever the arm is
    support_z_mm: float = 0.0  # the surface the block rests on (floor in the current fragment)
    fingertip_overhang_mm: float = FINGERTIP_OVERHANG_MM
    # optional: gathers jaw angle, pad poses and the block's actual pose after a
    # failed grab, so the failure explains itself (missed vs closed-but-not-holding)
    diagnose: Callable[[], Awaitable[dict[str, Any]]] | None = None
    # optional: measures (believed - physical) TCP in mm at the current pose;
    # the result is added to the grasp/lift targets (None = no correction)
    tcp_correction: Callable[[], Awaitable[tuple[float, float, float]]] | None = None
    # checklist item 5: hold at the lift pose this many seconds, sampling
    # is_holding_something once per HOLD_SAMPLE_S (0 = release immediately)
    hold_s: float = 0.0
    # checklist item 6: called mid-hold to reset the world; must report
    # holding_before_reset/holding_after_reset (None = no reset probe)
    mid_hold_reset: Callable[[], Awaitable[dict[str, Any]]] | None = None
    # set by _detect_block once a detection is accepted: block_size_mm when
    # explicit, else the measured size - every downstream consumer reads
    # this, never block_size_mm directly
    resolved_block_size_mm: float | None = None
    # phase 4: primary/fallback tallest-object scanners, run only when
    # randomize_size_range_mm is set (dynamic keep-out/carry heights)
    side_scanner: TallestScanner | None = None
    wrist_scanner: TallestScanner | None = None
    # set by _sim_obstacles from the randomize response's sizes_mm (log-only
    # ground truth - the control path never consumes it): max drawn z-dim,
    # None without a sizes-bearing response
    drawn_tallest_mm: float | None = None
    # set by _measure_tallest: the trusted-or-fallback tallest estimate, its
    # source, and the wrist-sweep vantages tried - fed into the derived
    # keep-out/carry heights once the held size is known (_run_steps, after
    # the jaw check) and printed in MEASURED_TALLEST_JSON
    tallest_estimate: TallestEstimate | None = None
    tallest_source: str | None = None
    tallest_scan_poses_mm: list[dict[str, float]] = field(default_factory=list)
    # set alongside the marker: the derived heights that replace
    # KEEPOUT_HEIGHT_MM / CARRY_CLEAR_ABOVE_SUPPORT_MM for this pick
    measured_keepout_height_mm: float | None = None
    measured_carry_clear_above_support_mm: float | None = None

    async def _move_or_diagnose(
        self, pose: Pose, move_world_state: WorldState, linear: bool = False
    ) -> None:
        """A failed move prints the arm's joint state (measured vs drive
        target), the gripper pads and the block's pose before re-raising."""
        try:
            await self.mover.move_to(pose, move_world_state, linear)
        except Exception:
            if self.diagnose is not None:
                report = await self.diagnose()
                print(f"{MOVE_DIAGNOSTICS_MARKER}{json.dumps(report, default=str, sort_keys=True)}")
            raise

    async def _sim_obstacles(self) -> list[Geometry]:
        if self.world is None:
            return []
        if self.randomize_seed is not None:
            names = list(self.movable_prop_names)
            if not names:
                pre = await self.world.do_command({"command": "prop_geometries"})
                names = [
                    g["name"]
                    for g in pre.get("geometries", [])
                    if not g["fixed"] and any(d > 0 for d in g["box_dims_mm"])
                ]
            if not names:
                print("step: randomize skipped - no movable props in the scene")
            else:
                region = self.randomize_region_mm or reachable_region_mm(
                    face_z_mm=self.support_z_mm
                )
                print(
                    f"step: randomize props (checklist item 1, seed {self.randomize_seed}, "
                    f"names {names}, region {region})"
                )
                randomize_command: dict[str, Any] = {
                    "command": "randomize_props",
                    "names": names,
                    "region": [list(region[0]), list(region[1])],
                    "seed": self.randomize_seed,
                    "min_separation": self.randomize_min_separation_mm,
                }
                if self.randomize_size_range_mm is not None:
                    randomize_command["size_range_mm"] = list(self.randomize_size_range_mm)
                response = await self.world.do_command(randomize_command)
                (x0, y0, _z0), (x1, y1, _z1) = region
                self.scan_centre_mm = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                self.pick_region_mm = region
                print(f"  scattered: {sorted(response.get('positions') or {})}")
                # log-only ground truth: the pipeline never consumes these
                # dims (the camera measures the target), but the log showing
                # them is what proves a size range was actually applied
                sizes_mm = response.get("sizes_mm") or {}
                if sizes_mm:
                    rounded = {
                        name: [round(float(v), 1) for v in dims]
                        for name, dims in sorted(sizes_mm.items())
                    }
                    print(f"  sizes (mm, module-reported): {rounded}")
                    self.drawn_tallest_mm = max(float(dims[2]) for dims in sizes_mm.values())
        response = await self.world.do_command({"command": "prop_geometries"})
        geometries = response.get("geometries", [])
        if self.place_prop_name is not None:
            self.place_pad_top_mm = pad_top_centre_mm(geometries, self.place_prop_name)
        exclude = {self.target_prop_name} if self.target_prop_name else set()
        return obstacles_from_prop_geometries(geometries, exclude)

    async def _measure_tallest(self, move_world_state: WorldState) -> None:
        """Tallest-object measurement (phase 4): side scan first (occlusion-
        proof except for a nearer silhouette hiding a farther block), then
        the wrist-sweep ladder when the side scan is untrusted, then the
        size-range max as a conservative last resort. ``verify_detection_height``
        doubles as the mock/real switch (the existing convention): mock skips
        the region clip and reads the support at 0 (seam's mock mapping)."""
        assert self.randomize_size_range_mm is not None
        mock = not self.verify_detection_height
        region_mm = None if mock else self.pick_region_mm
        support_z_mm = 0.0 if mock else self.support_z_mm

        if self.side_scanner is not None:
            points = await self.side_scanner.scan_world_mm()
            estimate = tallest_in_region_mm(
                points, region_mm, support_z_mm, self.randomize_size_range_mm
            )
            print(f"  tallest side scan: {estimate}")
            if estimate.trusted:
                self.tallest_estimate = estimate
                self.tallest_source = "side"
                return

        if (
            self.wrist_scanner is not None
            and self.look_pose is not None
            and self.scan_centre_mm is not None
            and self.pick_region_mm is not None
        ):
            for offset_x, offset_y, wrist_theta_deg in tallest_sweep_attempts(self.pick_region_mm):
                vantage = _pointing_down(
                    self.scan_centre_mm[0] + offset_x,
                    self.scan_centre_mm[1] + offset_y,
                    self.look_pose.z,
                    theta_deg=wrist_theta_deg,
                )
                print(f"step: tallest wrist sweep vantage: {_pose_to_dict(vantage)}")
                await self.mover.look_from(vantage, move_world_state)
                self.tallest_scan_poses_mm.append(_pose_to_dict(vantage))
                points = await self.wrist_scanner.scan_world_mm()
                estimate = tallest_in_region_mm(
                    points, region_mm, support_z_mm, self.randomize_size_range_mm
                )
                print(f"  tallest wrist sweep attempt: {estimate}")
                if estimate.trusted:
                    self.tallest_estimate = estimate
                    self.tallest_source = "wrist_sweep"
                    return

        lo_mm, hi_mm = self.randomize_size_range_mm
        print(
            "  WARNING: tallest measurement untrusted from every vantage - falling back to "
            f"the size-range max {hi_mm:.1f} mm as a conservative keep-out ceiling"
        )
        self.tallest_estimate = TallestEstimate(
            tallest_mm=hi_mm, points=0, trusted=False, reasons=["no trusted scan"]
        )
        self.tallest_source = "fallback"

    async def _place(
        self,
        grasp: Pose,
        lift: Pose,
        held_world_state: WorldState,
        carry_world_state: WorldState | None,
        free_world_state: WorldState,
    ) -> bool:
        """Carry the held block over the pad, descend as far as the planner
        allows, release, retreat, then report whether the block rests on the
        pad. Returns True once the block has been released here (the caller
        then skips its release)."""
        assert self.place_pad_top_mm is not None
        pad_x, pad_y, pad_top_z = self.place_pad_top_mm
        # reproduce the grasp configuration over the pad: same TCP-to-block
        # offset, the support is now the pad top, plus the drop gap
        place_z = grasp.z + (pad_top_z - self.support_z_mm) + self.place_clearance_mm
        pre_place = _pointing_down(pad_x, pad_y, max(lift.z, place_z + PRE_GRASP_STANDOFF_MM))
        if carry_world_state is not None:
            # keep-out carry (GPU run 12): hop above the no-fly box, then let
            # the planner move freely - it cannot enter the block airspace
            clear_above_support_mm = (
                self.measured_carry_clear_above_support_mm
                if self.measured_carry_clear_above_support_mm is not None
                else CARRY_CLEAR_ABOVE_SUPPORT_MM
            )
            clear = _pointing_down(lift.x, lift.y, self.support_z_mm + clear_above_support_mm)
            print(f"step: raise above the pick-area keep-out: {_pose_to_dict(clear)}")
            await self._move_or_diagnose(clear, held_world_state, linear=True)
            print(f"step: carry to pre-place (free, keep-out boxed): {_pose_to_dict(pre_place)}")
            try:
                await self.mover.move_to(pre_place, carry_world_state, False)
            except Exception as error:
                print(f"  keep-out carry failed ({error}); falling back to the linear carry")
                await self._move_or_diagnose(pre_place, held_world_state, linear=True)
        else:
            # no known scatter region to box off: carry along a straight line
            # at constant height so the held cube cannot dip (GPU run 11)
            carry_start = _pointing_down(lift.x, lift.y, pre_place.z)
            print(f"step: raise to carry height: {_pose_to_dict(carry_start)}")
            await self._move_or_diagnose(carry_start, held_world_state, linear=True)
            print(f"step: carry to pre-place: {_pose_to_dict(pre_place)}")
            try:
                await self.mover.move_to(pre_place, held_world_state, True)
            except Exception as error:
                print(f"  linear carry failed ({error}); replanning the carry freely")
                await self._move_or_diagnose(pre_place, held_world_state)

        # the current fragment's pad sits 783 mm from the arm base - near the
        # UR5e's reach boundary, where a constraint-locked straight descent can
        # have no continuous IK solution (GPU run 6). Try increasingly
        # permissive descents; failing all, drop from the hover pose onto the
        # pad (restitution 0 keeps the bounce small; the verdict reports it).
        half_z = (place_z + pre_place.z) / 2.0
        descents = [
            ("linear", _pointing_down(pad_x, pad_y, place_z), True),
            ("planned", _pointing_down(pad_x, pad_y, place_z), False),
            ("half-height linear", _pointing_down(pad_x, pad_y, half_z), True),
        ]
        stage = "hover_drop"
        release_pose = pre_place
        for label, pose, linear in descents:
            print(f"step: move to place ({label}): {_pose_to_dict(pose)}")
            try:
                await self.mover.move_to(pose, held_world_state, linear)
            except Exception as error:
                print(f"  place descent ({label}) failed: {error}")
                continue
            stage, release_pose = label, pose
            break
        if stage == "hover_drop":
            print("  every descent failed - releasing from the hover pose")

        print(f"step: open (place, {stage})")
        await self.gripper.open()
        if release_pose is not pre_place:
            print("step: retreat after place")
            try:
                await self.mover.move_to(pre_place, free_world_state, True)
            except Exception:
                await self.mover.move_to(pre_place, free_world_state, False)
        await self._report_placement(pad_x, pad_y, pad_top_z, stage)
        return True

    async def _report_placement(
        self, pad_x: float, pad_y: float, pad_top_z: float, stage: str
    ) -> None:
        if self.world is None:
            return
        await asyncio.sleep(PLACE_SETTLE_S)
        response = await self.world.do_command({"command": "prop_geometries"})
        block_name = self.target_prop_name or self.block_name
        block = next(
            (g for g in response.get("geometries", []) if g["name"] == block_name), None
        )
        if block is None:
            print(f"  placement: prop {block_name!r} not found in prop_geometries")
            return
        pose = block["pose_in_world_mm"]
        assert self.resolved_block_size_mm is not None
        expected_z = pad_top_z + self.resolved_block_size_mm / 2.0
        placed_on_pad = (
            abs(pose["x"] - pad_x) <= PLACE_XY_TOLERANCE_MM
            and abs(pose["y"] - pad_y) <= PLACE_XY_TOLERANCE_MM
            and abs(pose["z"] - expected_z) <= PLACE_Z_TOLERANCE_MM
        )
        report = {
            "block_pose_mm": pose,
            "expected_z_mm": expected_z,
            "placed_on_pad": placed_on_pad,
            "place_stage": stage,
        }
        print(f"{PLACED_BLOCK_MARKER}{json.dumps(report, default=str, sort_keys=True)}")

    async def _set_ignored(self, names: Sequence[str]) -> None:
        """DEC-21 route (c): with sim-world's live GetGeometries in the frame
        system, the target block must not obstruct its own pick - ignore it
        for the run, restore afterwards."""
        if self.world is None or self.target_prop_name is None:
            return
        await self.world.do_command({"command": "ignore_props", "names": list(names)})
        print(f"step: ignore_props {list(names)}")

    async def run(self) -> Transform:
        await self._set_ignored([self.target_prop_name] if self.target_prop_name else [])
        try:
            return await self._run_steps()
        finally:
            await self._set_ignored([])

    def _expected_block_z_mm(self, block_size_mm: float) -> float:
        return self.support_z_mm + block_size_mm / 2.0

    def _is_resting_height(self, pose: Pose, block_size_mm: float) -> bool:
        if not self.verify_detection_height:
            return True
        return abs(pose.z - self._expected_block_z_mm(block_size_mm)) <= DETECT_Z_TOLERANCE_MM

    def _current_measurement(self) -> dict[str, Any] | None:
        """The most recent measurement from the detector's last
        ``block_pose_world`` call, or None when block_size_mm is explicit
        (no measurement happens) or the detector offers none."""
        if self.block_size_mm is not None:
            return None
        last_measurement = getattr(self.detector, "last_measurement", None)
        return last_measurement() if last_measurement is not None else None

    def _resolve_block_size_mm(self, measurement: dict[str, Any] | None) -> float | None:
        """The size in effect for this detection: block_size_mm when
        explicit, else the measured size, or None when the view was
        degenerate (no measurement to grasp on)."""
        if self.block_size_mm is not None:
            return self.block_size_mm
        return measurement["size_mm"] if measurement is not None else None

    async def _detect_block(self, move_world_state: WorldState) -> Pose:
        """Scan, detect, focus, and sanity-check the red block's pose. A
        detection whose height cannot be a resting block means the gripper's
        shadow swallowed it (GPU run 10: z 115 for a 60 mm cube), so the scan
        walks SCAN_ATTEMPTS instead of grasping at a phantom. When
        block_size_mm is None, a degenerate size measurement (footprint and
        height disagreeing) walks the same ladder instead of grasping on a
        bad number (phase 3 seam decision)."""
        for offset_x, offset_y, wrist_theta_deg in SCAN_ATTEMPTS:
            look_pose = self.look_pose
            if look_pose is not None and self.scan_centre_mm is not None:
                look_pose = _pointing_down(
                    self.scan_centre_mm[0] + offset_x,
                    self.scan_centre_mm[1] + offset_y,
                    look_pose.z,
                    theta_deg=wrist_theta_deg,
                )
            if look_pose is not None:
                print(f"step: look (scan the workspace from {_pose_to_dict(look_pose)})")
                await self.mover.look_from(look_pose, move_world_state)

            print("step: detect (from the stationary look pose)")
            block_pose = await self.detector.block_pose_world()
            print(f"  block_pose_world (mm): {_pose_to_dict(block_pose)}")
            detect_pose = look_pose
            measurement = self._current_measurement()
            block_size_mm = self._resolve_block_size_mm(measurement)
            if block_size_mm is None:
                print(
                    "  degenerate size measurement (footprint/height disagree by more "
                    f"than {MEASURED_SIZE_DEGENERATE_FRACTION:.0%} of the median) - "
                    "re-scanning"
                )
                if self.scan_centre_mm is None:
                    break
                continue
            if (
                look_pose is not None
                and self.scan_centre_mm is not None
                and self._is_resting_height(block_pose, block_size_mm)
                and math.hypot(block_pose.x - look_pose.x, block_pose.y - look_pose.y)
                > FOCUS_LOOK_OFFSET_MM
            ):
                # the top-face estimate degrades off-centre (GPU run 5: ~14 mm);
                # re-aim above the DETECTED position and measure again
                focus = _pointing_down(
                    block_pose.x, block_pose.y, look_pose.z, theta_deg=wrist_theta_deg
                )
                print(f"step: focus above the detected block: {_pose_to_dict(focus)}")
                await self.mover.look_from(focus, move_world_state)
                print("step: detect (focused)")
                block_pose = await self.detector.block_pose_world()
                print(f"  block_pose_world (mm): {_pose_to_dict(block_pose)}")
                detect_pose = focus
                measurement = self._current_measurement()
                block_size_mm = self._resolve_block_size_mm(measurement)
                if block_size_mm is None:
                    print(
                        "  degenerate size measurement (footprint/height disagree by "
                        f"more than {MEASURED_SIZE_DEGENERATE_FRACTION:.0%} of the "
                        "median) - re-scanning"
                    )
                    if self.scan_centre_mm is None:
                        break
                    continue
            if self._is_resting_height(block_pose, block_size_mm):
                print(f"{DETECTED_BLOCK_POSE_MARKER}{json.dumps(_pose_to_dict(block_pose))}")
                self.resolved_block_size_mm = block_size_mm
                if measurement is not None:
                    report = dict(measurement)
                    report["scan_pose_mm"] = _pose_to_dict(detect_pose) if detect_pose else None
                    print(f"{MEASURED_BLOCK_MARKER}{json.dumps(report, default=str, sort_keys=True)}")
                return block_pose
            print(
                f"  implausible detection: z {block_pose.z:.1f} vs expected "
                f"{self._expected_block_z_mm(block_size_mm):.1f} mm for a resting block "
                "- the gripper likely shadows it; re-scanning from an offset pose"
            )
            if self.scan_centre_mm is None:
                break
        raise RuntimeError(
            "no plausible red-block detection from any scan pose - is the block "
            "visible and resting on its support?"
        )

    async def _run_steps(self) -> Transform:
        sim_obstacles = await self._sim_obstacles()
        table = table_recipe_unless_served(self.table, sim_obstacles)
        if table is None and self.table is not None:
            print(f"  --table dropped: the live scene already serves a {self.table.label!r} box")
        self.table = table
        move_world_state = world_state(
            self.table, (*self.other_blocks, *sim_obstacles), support_obstacle(self.support_z_mm)
        )
        if self.randomize_size_range_mm is not None:
            await self._measure_tallest(move_world_state)
        block_pose = await self._detect_block(move_world_state)
        assert self.resolved_block_size_mm is not None
        block_size_mm = self.resolved_block_size_mm
        if self.randomize_size_range_mm is not None:
            lo_mm, hi_mm = self.randomize_size_range_mm
            if not (lo_mm - DETECT_Z_TOLERANCE_MM <= block_size_mm <= hi_mm + DETECT_Z_TOLERANCE_MM):
                print(
                    f"  WARNING: measured size {block_size_mm:.1f} mm falls outside the "
                    f"--randomize-size-mm range [{lo_mm:.1f}, {hi_mm:.1f}] mm "
                    f"(+/- {DETECT_Z_TOLERANCE_MM:.0f} mm tolerance)"
                )
        print(f"step: jaw check ({block_size_mm:.1f} mm measured vs {JAW_MAX_BLOCK_MM:.0f} mm jaw)")
        if block_size_mm > JAW_MAX_BLOCK_MM:
            raise RuntimeError(
                f"target block measures {block_size_mm:.1f} mm, wider than the gripper's "
                f"{JAW_MAX_BLOCK_MM:.0f} mm jaw limit (2F-85 85 mm open - 10 mm finger "
                "clearance) - refusing the grasp, arm left parked"
            )

        if self.tallest_estimate is not None:
            self.measured_keepout_height_mm = keepout_height_mm(
                self.tallest_estimate.tallest_mm, block_size_mm
            )
            self.measured_carry_clear_above_support_mm = carry_clear_above_support_mm(
                self.tallest_estimate.tallest_mm, block_size_mm
            )
            drawn_delta_mm = (
                self.tallest_estimate.tallest_mm - self.drawn_tallest_mm
                if self.drawn_tallest_mm is not None
                else None
            )
            marker = {
                "tallest_mm": self.tallest_estimate.tallest_mm,
                "source": self.tallest_source,
                "trusted": self.tallest_estimate.trusted,
                "reasons": self.tallest_estimate.reasons,
                "points": self.tallest_estimate.points,
                "scan_poses_mm": self.tallest_scan_poses_mm,
                "keepout_height_mm": self.measured_keepout_height_mm,
                "carry_clear_above_support_mm": self.measured_carry_clear_above_support_mm,
                "drawn_tallest_mm": self.drawn_tallest_mm,
                "drawn_delta_mm": drawn_delta_mm,
            }
            print(f"{MEASURED_TALLEST_MARKER}{json.dumps(marker, default=str, sort_keys=True)}")

        grasp_z = grasp_height_mm(
            block_pose.z, block_size_mm, self.support_z_mm, self.fingertip_overhang_mm
        )
        held_centre_below_tcp_mm = grasp_z - block_pose.z
        if grasp_z != block_pose.z:
            print(
                f"  grasp height raised from {block_pose.z:.1f} to {grasp_z:.1f} mm "
                f"(block size {block_size_mm:.0f}, support z {self.support_z_mm:.0f}, "
                f"fingertip overhang {self.fingertip_overhang_mm:.0f})"
            )
            block_pose = with_z(block_pose, grasp_z)

        print("step: open")
        await self.gripper.open()

        pre_grasp = pre_grasp_pose(block_pose, self.standoff_mm)
        print(f"step: move to pre-grasp: {_pose_to_dict(pre_grasp)}")
        await self._move_or_diagnose(pre_grasp, move_world_state)

        grasp = grasp_pose(block_pose)
        lift = pre_grasp
        if self.tcp_correction is not None:
            delta = await self.tcp_correction()
            print(
                f"  tcp correction at pre-grasp (believed - physical, mm): "
                f"({delta[0]:.1f}, {delta[1]:.1f}, {delta[2]:.1f})"
            )
            grasp = corrected_pose(grasp, delta)
            lift = corrected_pose(pre_grasp, delta)
        print(f"step: move to grasp: {_pose_to_dict(grasp)}")
        await self._move_or_diagnose(grasp, move_world_state, linear=True)

        print("step: grab")
        grabbed = await self.gripper.grab()
        print(f"  grab: {grabbed}")
        if not grabbed:
            if self.diagnose is not None:
                report = await self.diagnose()
                print(f"{GRAB_DIAGNOSTICS_MARKER}{json.dumps(report, default=str, sort_keys=True)}")
            raise RuntimeError(
                f"grab failed: gripper {self.gripper_name!r} reported no hold at "
                f"{_pose_to_dict(grasp)}"
            )

        transform = held_block_transform(
            self.block_name,
            block_size_mm,
            self.gripper_name,
            centre_below_tcp_mm=held_centre_below_tcp_mm,
        )
        padded_transform = held_block_transform(
            self.block_name,
            block_size_mm + HELD_BLOCK_PADDING_MM,
            self.gripper_name,
            centre_below_tcp_mm=held_centre_below_tcp_mm,
        )
        held_world_state = world_state(
            self.table,
            (*self.other_blocks, *sim_obstacles),
            support_obstacle(self.support_z_mm),
            transforms=(padded_transform,),
        )
        carry_world_state = None
        if self.pick_region_mm is not None:
            keepout_kwargs = (
                {"height_mm": self.measured_keepout_height_mm}
                if self.measured_keepout_height_mm is not None
                else {}
            )
            carry_world_state = world_state(
                self.table,
                (
                    *self.other_blocks,
                    *sim_obstacles,
                    pick_area_keepout(self.pick_region_mm, **keepout_kwargs),
                ),
                support_obstacle(self.support_z_mm),
                transforms=(padded_transform,),
            )
        print(f"step: move to lift: {_pose_to_dict(lift)}")
        await self._move_or_diagnose(lift, held_world_state, linear=True)

        if self.hold_s > 0:
            print(f"step: hold at the lift pose for {self.hold_s:.0f} s (checklist item 5)")
            samples: list[dict[str, Any]] = []
            for _ in range(max(1, round(self.hold_s / HOLD_SAMPLE_S))):
                await asyncio.sleep(HOLD_SAMPLE_S)
                status = await self.gripper.is_holding_something()
                meta = dict(getattr(status, "meta", None) or {})
                samples.append(
                    {"holding": bool(status.is_holding_something), "jaw_deg": meta.get("jaw_deg")}
                )
            print(f"{HOLD_SAMPLES_MARKER}{json.dumps(samples, default=str)}")
            if not all(sample["holding"] for sample in samples):
                raise RuntimeError(
                    f"hold failed: is_holding_something sampled {samples} at the lift pose"
                )

        if self.mid_hold_reset is not None:
            print("step: world reset mid-hold (checklist item 6)")
            reset_report = await self.mid_hold_reset()
            payload = json.dumps(reset_report, default=str, sort_keys=True)
            print(f"{RESET_MID_HOLD_MARKER}{payload}")
            if reset_report.get("holding_after_reset"):
                print("  reset mid-hold: the grip survived the reset")
            else:
                # GPU run 26: isaac's world.reset() returns every scene-registered
                # prim (props included) to its spawn state, so "not holding" after
                # a reset is the designed outcome, not a dropped grip. The JSON's
                # block_prim_pose says where the block went - a human judges it.
                print(
                    "  reset mid-hold: not holding after the reset (isaac's world.reset() "
                    "returns props to their spawn state; judge RESET_MID_HOLD_JSON)"
                )

        transform_json = json.dumps(
            MessageToDict(transform, preserving_proto_field_name=True), sort_keys=True
        )
        print("held-block transform (DEC-14):")
        print(f"{HELD_BLOCK_TRANSFORM_MARKER}{transform_json}")

        placed = False
        if self.place_pad_top_mm is not None and self.mid_hold_reset is None:
            placed = await self._place(
                grasp, lift, held_world_state, carry_world_state, move_world_state
            )
        elif self.place_prop_name is not None and self.place_pad_top_mm is None:
            print(f"step: place skipped - no prop {self.place_prop_name!r} in the scene")
        if not placed:
            print("step: open (release)")
            await self.gripper.open()
        print("done")
        return transform


# ----------------------------------------------------------------------
# real-mode collaborators
# ----------------------------------------------------------------------


class RealDetector:
    """Block pose from the vision service's largest segment: the red points'
    top face gives x/y and the top height; the block centre is size/2 below
    it. The segmenter's own centre is printed for comparison.

    ``block_size_mm`` None measures the size from the same segment
    (footprint x/y extents plus top-face-minus-support height, cube-prior
    median) instead of trusting a caller-supplied value; the measurement is
    cached for ``last_measurement`` (None = explicit size, no measurement)."""

    def __init__(
        self,
        robot: RobotClient,
        vision: VisionClient,
        camera_name: str,
        block_size_mm: float | None,
        support_z_mm: float = 0.0,
    ) -> None:
        self._robot = robot
        self._vision = vision
        self._camera_name = camera_name
        self._block_size_mm = block_size_mm
        self._support_z_mm = support_z_mm
        self._last_measurement: dict[str, Any] | None = None

    def last_measurement(self) -> dict[str, Any] | None:
        return self._last_measurement

    async def block_pose_world(self) -> Pose:
        objects = await self._vision.get_object_point_clouds(self._camera_name)
        if not objects:
            raise RuntimeError(
                f"vision service returned no segments for camera {self._camera_name!r}"
            )
        largest = max(objects, key=lambda obj: len(obj.point_cloud))
        segment_centre = largest.geometries.geometries[0].center
        xyz, rgb = parse_pcd(largest.point_cloud)
        print(f"  segment (camera frame): {segment_stats(xyz, rgb)}")
        top = top_face_centre_m(xyz, rgb)
        if top is None:
            raise RuntimeError("segment has no points to take a top face from")
        top_world = await self._to_world(top[0] * 1000.0, top[1] * 1000.0, top[2] * 1000.0)
        segment_world = await self._to_world(segment_centre.x, segment_centre.y, segment_centre.z)
        print(
            f"  segmenter centre (world, mm): {_pose_to_dict(segment_world)}; "
            f"top face centre: {_pose_to_dict(top_world)} from {len(xyz)} points"
        )
        if self._block_size_mm is not None:
            self._last_measurement = None
            size_mm = self._block_size_mm
        else:
            footprint_mm = footprint_extents_mm(xyz)
            height_mm = top_world.z - self._support_z_mm
            measured = (
                measured_block_size_mm([footprint_mm[0], footprint_mm[1], height_mm])
                if footprint_mm is not None
                else None
            )
            if measured is None:
                if footprint_mm is None:
                    print("  size estimates: no top-face band points")
                else:
                    print(
                        f"  size estimates (mm): footprint [{footprint_mm[0]:.1f}, "
                        f"{footprint_mm[1]:.1f}], height {height_mm:.1f} (top z - "
                        f"support z {self._support_z_mm:.0f})"
                    )
                self._last_measurement = None
                size_mm = 0.0  # degenerate: the caller re-scans and discards this pose
            else:
                size_mm, estimates = measured
                self._last_measurement = {
                    "footprint_mm": [footprint_mm[0], footprint_mm[1]],
                    "height_mm": estimates[2],
                    "size_mm": size_mm,
                }
        return with_z(top_world, top_world.z - size_mm / 2.0)

    async def _to_world(self, x_mm: float, y_mm: float, z_mm: float) -> Pose:
        camera_pif = PoseInFrame(
            reference_frame=self._camera_name,
            pose=Pose(x=x_mm, y=y_mm, z=z_mm, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
        )
        return (await self._robot.transform_pose(camera_pif, "world")).pose


class RealMover:
    """Drives the motion service. At viam-sdk 0.80 ``MotionClient.move`` takes
    the component NAME as a plain string (``MoveRequest.component_name`` is a
    string; the ResourceName form is ``component_name_deprecated``)."""

    def __init__(self, motion: MotionClient, gripper_name: str, camera_name: str) -> None:
        self._motion = motion
        self._gripper_name = gripper_name
        self._camera_name = camera_name

    async def look_from(self, pose: Pose, world_state: WorldState) -> None:
        await self._move_frame(self._camera_name, pose, world_state)

    async def move_to(self, pose: Pose, world_state: WorldState, linear: bool = False) -> None:
        await self._move_frame(self._gripper_name, pose, world_state, linear)

    async def _move_frame(
        self, component_name: str, pose: Pose, world_state: WorldState, linear: bool = False
    ) -> None:
        destination = PoseInFrame(reference_frame="world", pose=pose)
        constraints = (
            Constraints(
                linear_constraint=[
                    LinearConstraint(
                        line_tolerance_mm=LINEAR_LINE_TOLERANCE_MM,
                        orientation_tolerance_degs=LINEAR_ORIENTATION_TOLERANCE_DEG,
                    )
                ]
            )
            if linear
            else None
        )
        success = await self._motion.move(
            component_name=component_name,
            destination=destination,
            world_state=world_state,
            constraints=constraints,
        )
        if not success:
            raise RuntimeError(
                f"motion move of {component_name!r} to {_pose_to_dict(pose)} reported failure"
            )


async def _grab_diagnostics(arm: Arm, gripper: Gripper, block_name: str) -> dict[str, Any]:
    """After a failed grab: jaw angle + pad poses (gripper `tcp_pose`), the
    holding predicate's meta, and where the block actually is."""
    report: dict[str, Any] = {}
    try:
        report["arm_joint_state"] = dict(await arm.do_command({"command": "joint_state"}))
    except Exception as exc:  # noqa: BLE001 - diagnostics never mask the failure
        report["arm_joint_state_error"] = repr(exc)
    try:
        report["tcp_pose"] = dict(await gripper.do_command({"command": "tcp_pose"}))
    except Exception as exc:  # noqa: BLE001 - diagnostics never mask the grab failure
        report["tcp_pose_error"] = repr(exc)
    try:
        status = await gripper.is_holding_something()
        report["holding"] = {
            "is_holding_something": status.is_holding_something,
            "meta": status.meta,
        }
    except Exception as exc:  # noqa: BLE001
        report["holding_error"] = repr(exc)
    try:
        prim_path = f"/World/{block_name.replace('-', '_')}"
        report["block_prim_pose"] = dict(
            await arm.do_command({"command": "prim_world_pose", "prim_path": prim_path})
        )
    except Exception as exc:  # noqa: BLE001
        report["block_prim_pose_error"] = repr(exc)
    return report


async def _reset_mid_hold_report(
    world: Any,
    gripper: GripperApi,
    diagnose: Callable[[], Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Checklist item 6 (ARM-15/XC-5): reset the world while holding, give the
    post-reset hooks time to re-apply gains and re-command the jaw, and report
    whether the grip survived. ``world`` is anything with the world model's
    ``do_command`` - a Generic client in real mode, the model itself in mock."""
    before = await gripper.is_holding_something()
    await world.do_command({"command": "reset"})
    await asyncio.sleep(RESET_SETTLE_S)
    after = await gripper.is_holding_something()
    report: dict[str, Any] = {
        "holding_before_reset": bool(before.is_holding_something),
        "holding_after_reset": bool(after.is_holding_something),
        "meta_after": getattr(after, "meta", None),
    }
    if diagnose is not None:
        report["diagnostics"] = await diagnose()
    return report


async def _camera_client(robot: RobotClient, camera_name: str) -> Any:
    """The resource list is a snapshot taken at connect; a module that was
    still rebuilding the camera then (a redeploy) is missing from it until a
    refresh. On a real miss, name the cameras the machine does report."""
    from viam.components.camera import Camera
    from viam.errors import ResourceNotFoundError

    await robot.refresh()
    try:
        return Camera.from_robot(robot, camera_name)
    except ResourceNotFoundError:
        cameras = sorted(
            name.name
            for name in robot.resource_names
            if name.subtype == Camera.API.resource_subtype
        )
        raise RuntimeError(
            f"camera {camera_name!r} is not in the machine's resource list (cameras present: "
            f"{cameras}) - check the component's status on the machine page"
        ) from None


class FixedCameraScanner:
    """A ``TallestScanner`` over a real camera: grabs its point cloud, then
    transforms the camera-frame points to world mm through the measured
    camera->world affine (``camera_world_transform_mm``). Used both for the
    fixed side camera (primary) and, aimed at a wrist-sweep vantage, the
    wrist camera (fallback)."""

    def __init__(self, robot: RobotClient, camera_name: str) -> None:
        self._robot = robot
        self._camera_name = camera_name

    async def scan_world_mm(self) -> np.ndarray:
        camera = await _camera_client(self._robot, self._camera_name)
        pcd_bytes, _mime = await camera.get_point_cloud()
        xyz_m, _rgb = parse_pcd(pcd_bytes)
        rotation, translation = await camera_world_transform_mm(self._robot, self._camera_name)
        world_mm = xyz_m * MM_PER_M @ rotation.T + translation
        valid = ~np.isnan(world_mm).any(axis=1)
        return world_mm[valid]


async def _probe_depth(robot: RobotClient, args: argparse.Namespace, look_pose: Pose) -> None:
    """Move to the look pose, then compare the depth straight below the camera
    with the camera's commanded height above the support."""

    motion = MotionClient.from_robot(robot, args.motion)
    mover = RealMover(motion, args.gripper, args.camera)
    await mover.look_from(look_pose, world_state(None))
    camera = await _camera_client(robot, args.camera)
    pcd_bytes, _mime = await camera.get_point_cloud()
    xyz, _rgb = parse_pcd(pcd_bytes)
    measured = centre_depth_mm(xyz)
    expected = look_pose.z - args.support_z_mm
    if measured is None:
        print("depth probe: no points within the probe radius of the optical axis")
        return
    print(
        f"depth probe: centre depth {measured:.1f} mm, expected {expected:.1f} mm, "
        f"ratio {measured / expected:.3f} ({len(xyz)} points)"
    )


async def _tcp_correction(
    robot: RobotClient, gripper: Gripper, gripper_name: str
) -> tuple[float, float, float]:
    """(believed - physical) TCP position in mm: where the frame system says
    the gripper frame is, minus where the pads actually are (module
    `tcp_pose`). Positive z = the physical gripper hangs lower than believed."""
    identity = Pose(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0)
    believed = (
        await robot.transform_pose(
            PoseInFrame(reference_frame=gripper_name, pose=identity), "world"
        )
    ).pose
    physical = (await gripper.do_command({"command": "tcp_pose"}))["pad_center_midpoint_mm"]
    return (
        believed.x - float(physical[0]),
        believed.y - float(physical[1]),
        believed.z - float(physical[2]),
    )


async def _run_real(args: argparse.Namespace) -> Transform:
    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()
    robot = await RobotClient.at_address(args.address, opts)
    try:
        if args.probe_depth:
            probe_pose = (
                look_pose_from(args.look_at)
                if args.look_at
                else default_scan_pose(args.support_z_mm)
            )
            await _probe_depth(robot, args, probe_pose)
            probe_size_mm = args.block_size_mm if args.block_size_mm is not None else 0.0
            return held_block_transform(args.block, probe_size_mm, args.gripper)
        await robot.refresh()  # the resource list is a snapshot from connect time
        vision = VisionClient.from_robot(robot, args.vision)
        motion = MotionClient.from_robot(robot, args.motion)
        gripper = Gripper.from_robot(robot, args.gripper)
        arm = Arm.from_robot(robot, args.arm)
        world = Generic.from_robot(robot, args.world)

        # the wrist sweep must survive a disabled/missing side camera (GPU
        # item 3: --tallest-camera "" went straight to the ceiling fallback)
        side_scanner = (
            FixedCameraScanner(robot, args.tallest_camera) if args.tallest_camera else None
        )
        wrist_scanner = FixedCameraScanner(robot, args.camera)

        pipeline = PickPipeline(
            detector=RealDetector(
                robot, vision, args.camera, args.block_size_mm, args.support_z_mm
            ),
            mover=RealMover(motion, args.gripper, args.camera),
            gripper=gripper,
            block_name=args.block,
            block_size_mm=args.block_size_mm,
            gripper_name=args.gripper,
            look_pose=_look_pose_from_args(args),
            table=table_obstacle() if args.table else None,
            support_z_mm=args.support_z_mm,
            fingertip_overhang_mm=args.fingertip_overhang_mm,
            world=world,
            target_prop_name=args.block,
            randomize_seed=args.randomize_seed,
            randomize_size_range_mm=args.randomize_size_mm,
            side_scanner=side_scanner,
            wrist_scanner=wrist_scanner,
            place_prop_name=None if args.no_place else args.place_pad,
            diagnose=lambda: _grab_diagnostics(arm, gripper, args.block),
            tcp_correction=(
                None
                if args.no_tcp_correction
                else (lambda: _tcp_correction(robot, gripper, args.gripper))
            ),
            hold_s=args.hold_s,
            mid_hold_reset=(
                (
                    lambda: _reset_mid_hold_report(
                        world, gripper, lambda: _grab_diagnostics(arm, gripper, args.block)
                    )
                )
                if args.reset_mid_hold
                else None
            ),
        )
        return await pipeline.run()
    finally:
        await robot.close()


# ----------------------------------------------------------------------
# mock-mode collaborators (ARM-8, CAM-14) - lazy isaac_module import
# ----------------------------------------------------------------------

# Four distinct canned joint sets (degrees, 6 DOF) so look/pre-grasp/grasp/lift
# are visibly different moves even though the mock arm ignores IK entirely.
_MOCK_LOOK_JOINTS_DEG = [-90.0, -100.0, -80.0, -90.0, 90.0, 0.0]
_MOCK_PRE_GRASP_JOINTS_DEG = [-90.0, -90.0, -90.0, -90.0, 90.0, 0.0]
_MOCK_GRASP_JOINTS_DEG = [-90.0, -70.0, -100.0, -100.0, 90.0, 0.0]
_MOCK_LIFT_JOINTS_DEG = [-90.0, -95.0, -85.0, -90.0, 90.0, 0.0]
_MOCK_JOINT_SETS_DEG = [
    _MOCK_LOOK_JOINTS_DEG,
    _MOCK_PRE_GRASP_JOINTS_DEG,
    _MOCK_GRASP_JOINTS_DEG,
    _MOCK_LIFT_JOINTS_DEG,
]


class MockDetector:
    """``block_size_mm`` None measures the size from the mock camera's own
    red pixels (footprint x/y only - the mock scene is a flat depth plane
    with no independent height axis to cross-check, unlike RealDetector)."""

    def __init__(self, camera: Any, block_size_mm: float | None = None) -> None:
        self._camera = camera
        self._block_size_mm = block_size_mm
        self._last_measurement: dict[str, Any] | None = None

    def last_measurement(self) -> dict[str, Any] | None:
        return self._last_measurement

    async def block_pose_world(self) -> Pose:
        pcd_bytes, _ = await self._camera.get_point_cloud()
        xyz_m, rgb = parse_pcd(pcd_bytes)
        if rgb is None:
            raise RuntimeError("mock point cloud carries no colour channel")
        x_m, y_m, z_m = red_centroid_m(xyz_m, rgb)
        pose = Pose(
            x=x_m * MM_PER_M,
            y=y_m * MM_PER_M,
            z=z_m * MM_PER_M,
            o_x=0.0,
            o_y=0.0,
            o_z=1.0,
            theta=0.0,
        )
        print(
            "  mock detector: no frame system in mock mode - the camera-frame "
            f"centroid is treated as world frame: {_pose_to_dict(pose)}"
        )
        if self._block_size_mm is not None:
            self._last_measurement = None
        else:
            footprint_mm = footprint_extents_mm(xyz_m)
            measured = (
                measured_block_size_mm([footprint_mm[0], footprint_mm[1]])
                if footprint_mm is not None
                else None
            )
            self._last_measurement = (
                {"footprint_mm": [footprint_mm[0], footprint_mm[1]], "height_mm": None, "size_mm": measured[0]}
                if measured is not None
                else None
            )
        return pose


class MockSideScanner:
    """A ``TallestScanner`` over the mock side camera (seam's mock world
    mapping, no frame system in mock mode):
    ``xyz_world_mm = (x_cam, z_cam, -y_cam) * 1000``, NaN rows (no hit)
    dropped."""

    def __init__(self, camera: Any) -> None:
        self._camera = camera

    async def scan_world_mm(self) -> np.ndarray:
        pcd_bytes, _mime = await self._camera.get_point_cloud()
        xyz_cam_m, _rgb = parse_pcd(pcd_bytes)
        x_cam, y_cam, z_cam = xyz_cam_m[:, 0], xyz_cam_m[:, 1], xyz_cam_m[:, 2]
        world_mm = np.column_stack([x_cam, z_cam, -y_cam]) * MM_PER_M
        valid = ~np.isnan(world_mm).any(axis=1)
        return world_mm[valid]


class MockMover:
    def __init__(self, arm: Any, joint_sets_deg: Sequence[Sequence[float]]) -> None:
        self._arm = arm
        self._joint_sets_deg = list(joint_sets_deg)
        self._call_count = 0

    async def look_from(self, pose: Pose, world_state: WorldState) -> None:
        await self.move_to(pose, world_state)

    async def move_to(self, pose: Pose, world_state: WorldState, linear: bool = False) -> None:
        if self._call_count >= len(self._joint_sets_deg):
            raise RuntimeError("mock mover: no more canned joint sets for this pick sequence")
        joints = self._joint_sets_deg[self._call_count]
        self._call_count += 1
        print(f"  mock mover: requested pose {_pose_to_dict(pose)} -> canned joints {joints}")
        await self._arm.move_to_joint_positions(JointPositions(values=list(joints)))


def _ensure_mock_sim_booted() -> Any:
    from isaac_module.sim_manager import SimConfig, SimManager

    manager = SimManager.get()
    if not manager._booted.is_set():
        sim_thread = threading.Thread(target=manager.main_loop, daemon=True)
        sim_thread.start()
        manager.ensure_booted(SimConfig(mock=True))
    return manager


async def _run_mock(args: argparse.Namespace) -> Transform:
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from isaac_module.models.arm import IsaacArm
    from isaac_module.models.camera import IsaacCamera
    from isaac_module.models.gripper import IsaacGripper
    from isaac_module.models.world import IsaacWorld

    def config(name: str, attrs: dict[str, Any]) -> ComponentConfig:
        return ComponentConfig(name=name, attributes=dict_to_struct(attrs))

    _ensure_mock_sim_booted()

    world_name = "mock-pick-world"
    arm_name = "mock-pick-arm"
    camera_name = "mock-wrist-cam"
    side_camera_name = "mock-side-cam"
    gripper_name = "mock-pick-grip"

    world = IsaacWorld.new(config(world_name, {"mock": True}), {})
    arm = IsaacArm.new(config(arm_name, {"world": world_name, "asset": "ur5e"}), {})
    camera_attrs: dict[str, Any] = {"world": world_name, "depth": True}
    if args.mock_block_size_mm is not None:
        camera_attrs["block_size_mm"] = args.mock_block_size_mm
    camera = IsaacCamera.new(config(camera_name, camera_attrs), {})
    # three distractors at distinct heights (45/90/60 mm), distinct columns
    # and staggered depths so nothing overlaps in the fabricated side view
    side_blocks = [
        {"rgb": [200, 60, 60], "size_mm": 60.0, "height_mm": 45.0, "column_offset_px": -220, "depth_m": 0.80},
        {"rgb": [60, 200, 60], "size_mm": 60.0, "height_mm": 90.0, "column_offset_px": 0, "depth_m": 0.90},
        {"rgb": [60, 60, 200], "size_mm": 60.0, "height_mm": 60.0, "column_offset_px": 220, "depth_m": 1.00},
    ]
    side_camera = IsaacCamera.new(
        config(
            side_camera_name,
            {"world": world_name, "depth": True, "view": "side", "blocks": side_blocks},
        ),
        {},
    )
    gripper = IsaacGripper.new(
        config(gripper_name, {"world": world_name, "arm": arm_name, "mock_object_width_m": 0.05}),
        {},
    )

    pipeline = PickPipeline(
        detector=MockDetector(camera, args.block_size_mm),
        mover=MockMover(arm, _MOCK_JOINT_SETS_DEG),
        gripper=gripper,
        verify_detection_height=False,
        block_name=args.block,
        block_size_mm=args.block_size_mm,
        gripper_name=gripper_name,
        look_pose=_look_pose_from_args(args),
        world=world,
        target_prop_name=args.block,
        randomize_seed=args.randomize_seed,
        randomize_size_range_mm=args.randomize_size_mm,
        side_scanner=MockSideScanner(side_camera),
        place_prop_name=None if args.no_place else args.place_pad,
        hold_s=args.hold_s,
        mid_hold_reset=(
            (lambda: _reset_mid_hold_report(world, gripper)) if args.reset_mid_hold else None
        ),
    )
    return await pipeline.run()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_randomize_size_mm(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--randomize-size-mm wants lo,hi, got {value!r}")
    try:
        lo, hi = (float(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--randomize-size-mm wants two numbers, got {value!r}")
    if not (lo > 0 and hi > 0 and lo <= hi):
        raise argparse.ArgumentTypeError(
            f"--randomize-size-mm wants 0 < lo <= hi, got {value!r}"
        )
    return (lo, hi)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock", action="store_true", help="run in-process against the module's mock backend"
    )
    parser.add_argument("--address", help="machine address (required unless --mock)")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--camera", default="wrist-cam")
    parser.add_argument("--arm", default="pick-arm")
    parser.add_argument("--gripper", default="pick-grip")
    parser.add_argument("--vision", default="block-segmenter")
    parser.add_argument("--motion", default="builtin")
    parser.add_argument("--block", default="pick_cube")
    parser.add_argument(
        "--block-size-mm",
        type=float,
        default=None,
        help="override the target block's size (mm) instead of measuring it from the "
        "focused detection's point cloud; omit to measure (the default)",
    )
    parser.add_argument(
        "--randomize-size-mm",
        type=_parse_randomize_size_mm,
        default=None,
        metavar="LO,HI",
        help="lo,hi (mm) size range added to the --randomize-seed randomize_props call as "
        "size_range_mm, for every movable name; warns (never fails) if the measured size "
        "falls outside it (default off, byte-identical randomize_props payload)",
    )
    parser.add_argument(
        "--mock-block-size-mm",
        type=float,
        default=None,
        help="test-only: fabricate the --mock wrist camera's red block at this metric "
        "size (mm) instead of the default fixed pixel rectangle",
    )
    parser.add_argument(
        "--place-pad", default="place_pad", help="fixed prop to set the block down on"
    )
    parser.add_argument(
        "--no-place", action="store_true", help="release at the lift pose instead of placing"
    )
    parser.add_argument("--world", default="sim-world", help="the isaac-sim world component name")
    parser.add_argument(
        "--hold-s",
        type=float,
        default=DEFAULT_HOLD_S,
        help="seconds to hold at the lift pose sampling IsHoldingSomething at 1 Hz "
        "(checklist item 5; 0 = release immediately)",
    )
    parser.add_argument(
        "--reset-mid-hold",
        action="store_true",
        help='send the world {"command": "reset"} while holding and require the grip to '
        "survive the post-reset re-tune (checklist item 6, ARM-15/XC-5)",
    )
    parser.add_argument(
        "--look-at",
        default=None,
        help="x,y,z (mm, world) the wrist camera is moved to, pointing down, before detecting "
        f"(default {DEFAULT_LOOK_XY_MM[0]:.0f},{DEFAULT_LOOK_XY_MM[1]:.0f},"
        f"<--support-z-mm + {SCAN_HEIGHT_ABOVE_SUPPORT_MM:.0f}>: within UR5e reach, with the "
        "fragment's blocks inside the 90 deg field of view)",
    )
    parser.add_argument(
        "--no-look", action="store_true", help="detect from wherever the arm already is"
    )
    parser.add_argument(
        "--support-z-mm",
        type=float,
        default=0.0,
        help="height of the surface the block rests on (0 = floor, the current fragment)",
    )
    parser.add_argument(
        "--fingertip-overhang-mm",
        type=float,
        default=FINGERTIP_OVERHANG_MM,
        help="how far the fingertips extend past the TCP (measured 19 mm on the 2F-85)",
    )
    parser.add_argument(
        "--probe-depth",
        action="store_true",
        help="only move to the look pose and report the wrist camera's depth straight down "
        "against its commanded height (a depth-scale check); no pick",
    )
    parser.add_argument(
        "--no-tcp-correction",
        action="store_true",
        help="skip measuring the believed-vs-physical TCP offset at pre-grasp",
    )
    parser.add_argument(
        "--randomize-seed",
        type=int,
        default=None,
        help="re-randomise the movable blocks' positions (world DoCommand randomize_props) "
        "before this pick, within the table-top region (checklist item 1: two consecutive "
        "picks with re-randomised blocks); default off",
    )
    parser.add_argument(
        "--tallest-camera",
        default="side-cam",
        help="fixed side camera measuring the tallest scattered object (phase 4), primary "
        "source for the dynamic keep-out/carry heights; empty string disables it, falling "
        "back to the wrist sweep then the --randomize-size-mm range max",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="add the W4 table box (README recipe) as a motion obstacle - only for a scene "
        "whose table is NOT served live; the shipped fragment serves its table via "
        "prop_geometries, and the flag is dropped automatically when the live box is present",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.mock and not args.address:
        print("FAILED: --address is required unless --mock is set")
        return 1

    try:
        if args.mock:
            asyncio.run(_run_mock(args))
        else:
            asyncio.run(_run_real(args))
    except Exception as exc:  # noqa: BLE001 - surface any failure as a clean exit code
        print(f"FAILED: {exc!r}")
        print(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
