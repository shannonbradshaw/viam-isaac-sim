"""Unit tests for the pure helpers in examples/gpu_checklist_arm.py.

The module lives under examples/ (not a package under src/) and must not be
imported by anything in src/isaac_module (it runs on a laptop against a
remote machine), so it is loaded here via importlib rather than adding
examples/ to pyproject's pythonpath.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "gpu_checklist_arm.py"
_spec = importlib.util.spec_from_file_location("gpu_checklist_arm", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
gpu_checklist_arm = importlib.util.module_from_spec(_spec)
sys.modules["gpu_checklist_arm"] = gpu_checklist_arm
_spec.loader.exec_module(gpu_checklist_arm)

pose_delta_mm_deg = gpu_checklist_arm.pose_delta_mm_deg
axis_from_quaternion = gpu_checklist_arm.axis_from_quaternion
angle_between_deg = gpu_checklist_arm.angle_between_deg
verdict = gpu_checklist_arm.verdict


def test_pose_delta_identical_poses_is_zero():
    pose = (100.0, 200.0, 300.0, 0.0, 0.0, 1.0, 0.0)
    translation_mm, rotation_deg = pose_delta_mm_deg(pose, pose)
    assert translation_mm == pytest.approx(0.0, abs=1e-9)
    assert rotation_deg == pytest.approx(0.0, abs=1e-9)


def test_pose_delta_one_mm_apart():
    pose_a = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    pose_b = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    translation_mm, rotation_deg = pose_delta_mm_deg(pose_a, pose_b)
    assert translation_mm == pytest.approx(1.0, abs=1e-9)
    assert rotation_deg == pytest.approx(0.0, abs=1e-6)


def test_pose_delta_rotated_by_tenth_degree():
    pose_a = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    pose_b = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1)
    translation_mm, rotation_deg = pose_delta_mm_deg(pose_a, pose_b)
    assert translation_mm == pytest.approx(0.0, abs=1e-9)
    assert rotation_deg == pytest.approx(0.1, abs=1e-6)


def test_axis_from_quaternion_rz180_leaves_z_flips_x():
    # (w, x, y, z) for Rz(180deg)
    q_rz180 = (0.0, 0.0, 0.0, 1.0)
    z_axis = axis_from_quaternion(q_rz180, axis=(0.0, 0.0, 1.0))
    x_axis = axis_from_quaternion(q_rz180, axis=(1.0, 0.0, 0.0))
    assert z_axis == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert x_axis == pytest.approx((-1.0, 0.0, 0.0), abs=1e-9)


def test_axis_from_quaternion_defaults_to_plus_z():
    identity = (1.0, 0.0, 0.0, 0.0)
    assert axis_from_quaternion(identity) == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)


def test_angle_between_deg_z_and_negative_y_is_90():
    angle = angle_between_deg((0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    assert angle == pytest.approx(90.0, abs=1e-9)


def test_angle_between_deg_identical_vectors_is_zero():
    angle = angle_between_deg((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
    assert angle == pytest.approx(0.0, abs=1e-9)


def test_verdict_formats_pass_and_fail():
    assert verdict("thing", True, "all good") == "[PASS] thing: all good"
    assert verdict("thing", False, "off by 5mm") == "[FAIL] thing: off by 5mm"


def test_pose_delta_is_symmetric_for_translation():
    pose_a = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    pose_b = (3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    translation_mm, _ = pose_delta_mm_deg(pose_a, pose_b)
    assert translation_mm == pytest.approx(5.0, abs=1e-9)


def test_angle_between_deg_orthogonal_axes_matches_math_expectation():
    # sanity check the helper against a hand computation, not just itself
    expected = math.degrees(math.acos(0.0))
    assert angle_between_deg((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(expected)
