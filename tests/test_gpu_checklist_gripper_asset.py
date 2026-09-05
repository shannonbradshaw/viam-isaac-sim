"""Unit tests for the pure helpers in examples/gpu_checklist_gripper_asset.py (item 1:
R-4 / OQ-4 pad-collision probe). Loaded via importlib like the phase-1/2
checklist tests - examples/ is not on pythonpath and the runner half only
works inside Isaac's python."""

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "gpu_checklist_gripper_asset.py"
)
_spec = importlib.util.spec_from_file_location("gpu_checklist_gripper_asset", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
sys.modules["gpu_checklist_gripper_asset"] = probe
_spec.loader.exec_module(probe)

PrimRecord = probe.PrimRecord

ROOT = "/Robotiq_2F_85"
LEFT_PAD = f"{ROOT}/left_inner_finger_pad"
RIGHT_PAD = f"{ROOT}/right_inner_finger_pad"
ISAAC_DEV_REF = "omniverse://isaac-dev.ov.nvidia.com/Projects/robotiq/parts/pad.usd"
BUCKET_REF = "https://omniverse-content-production.s3.amazonaws.com/Assets/Isaac/5.0/parts/pad.usd"


def _records(*, collision_on_pad: bool, collision_on_mesh: bool, refs: tuple[str, ...] = ()):
    return [
        PrimRecord(ROOT, False, refs),
        PrimRecord(f"{ROOT}/base_link", False),
        PrimRecord(f"{ROOT}/base_link/collisions", True),  # not a pad; must not count
        PrimRecord(LEFT_PAD, collision_on_pad),
        PrimRecord(f"{LEFT_PAD}/collisions/mesh", collision_on_mesh),
        PrimRecord(RIGHT_PAD, collision_on_pad),
        PrimRecord(f"{RIGHT_PAD}/collisions/mesh", collision_on_mesh),
        PrimRecord(f"{ROOT}/left_outer_finger", False),  # an outer finger is not a pad
    ]


def test_is_pad_matches_last_segment_only():
    assert probe.is_pad(LEFT_PAD)
    assert probe.is_pad(f"{ROOT}/left_inner_finger")  # the 5.0 asset: fingertip mesh lives here
    assert probe.is_pad(
        f"{ROOT}/left_inner_finger/visuals/Defeatured_2F_85_PAD_OPEN_fingertipsstep_01"
    )
    assert not probe.is_pad(f"{LEFT_PAD}/collisions/mesh")  # child of a pad is not itself a pad
    assert not probe.is_pad(f"{ROOT}/left_outer_finger")


def test_pad_reports_count_collision_in_subtree_not_only_on_self():
    reports = probe.pad_reports(_records(collision_on_pad=False, collision_on_mesh=True))
    assert [r.path for r in reports] == [LEFT_PAD, RIGHT_PAD]
    assert all(r.has_collision_on_self is False for r in reports)
    assert all(r.has_collision_in_subtree is True for r in reports)


def test_pad_reports_ignore_collision_outside_the_pad_subtree():
    reports = probe.pad_reports(_records(collision_on_pad=False, collision_on_mesh=False))
    # base_link/collisions carries the API but is not under a pad
    assert all(r.has_collision_in_subtree is False for r in reports)


def test_unresolvable_references_flags_isaac_dev_only():
    records = _records(
        collision_on_pad=False, collision_on_mesh=False, refs=(ISAAC_DEV_REF, BUCKET_REF)
    )
    assert probe.unresolvable_references(records) == [ISAAC_DEV_REF]


def test_item1_pass_when_any_pad_has_collision():
    records = _records(collision_on_pad=False, collision_on_mesh=True, refs=(ISAAC_DEV_REF,))
    verdict, detail = probe.item1_verdict(
        probe.pad_reports(records), probe.unresolvable_references(records)
    )
    assert verdict == "PASS"
    assert "unresolved refs: 1" in detail


def test_item1_fail_attributes_to_isaac_dev_when_present():
    records = _records(collision_on_pad=False, collision_on_mesh=False, refs=(ISAAC_DEV_REF,))
    verdict, detail = probe.item1_verdict(
        probe.pad_reports(records), probe.unresolvable_references(records)
    )
    assert verdict == "FAIL"
    assert "R-4 confirmed" in detail


def test_item1_fail_distinguishes_no_pads_from_pads_without_collision():
    no_pads = probe.item1_verdict([], [])
    assert no_pads[0] == "FAIL" and "no pad prims" in no_pads[1]
    records = _records(collision_on_pad=False, collision_on_mesh=False)
    verdict, detail = probe.item1_verdict(probe.pad_reports(records), [])
    assert verdict == "FAIL" and "authors no collision" in detail
