"""Unit tests for the pure helpers in examples/gpu_checklist_camera.py.

The module lives under examples/ (not a package under src/) and must not be
imported by anything in src/isaac_module (it runs on a laptop against a
remote machine), so it is loaded here via importlib rather than adding
examples/ to pyproject's pythonpath.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "gpu_checklist_camera.py"
_spec = importlib.util.spec_from_file_location("gpu_checklist_camera", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
gpu_checklist_camera = importlib.util.module_from_spec(_spec)
sys.modules["gpu_checklist_camera"] = gpu_checklist_camera
_spec.loader.exec_module(gpu_checklist_camera)

parse_xyz = gpu_checklist_camera.parse_xyz
red_bbox = gpu_checklist_camera.red_bbox
raised_bbox = gpu_checklist_camera.raised_bbox
depth_stats = gpu_checklist_camera.depth_stats
region_center = gpu_checklist_camera.region_center
pose_delta_mm = gpu_checklist_camera.pose_delta_mm
is_reddish = gpu_checklist_camera.is_reddish
verdict = gpu_checklist_camera.verdict
skip = gpu_checklist_camera.skip


def test_parse_xyz():
    assert parse_xyz("0.6,0.1,0.7755") == pytest.approx((0.6, 0.1, 0.7755))


def _synthetic_red_rectangle_pixels(
    width: int, height: int, cols: range, rows: range
) -> list[tuple[int, int, int]]:
    """A row-major width*height pixel list, red inside [cols x rows], gray elsewhere."""
    pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if x in cols and y in rows:
                pixels.append((200, 10, 10))
            else:
                pixels.append((50, 50, 50))
    return pixels


def test_red_bbox_finds_rectangle():
    # 20x10 image, red rectangle spanning cols 5-9 across all 10 rows = 50 red px,
    # at the MIN_RED_PIXELS threshold; bbox is (5, 0, 10, 10).
    pixels = _synthetic_red_rectangle_pixels(20, 10, range(5, 10), range(0, 10))
    bbox = red_bbox(pixels, 20, 10, threshold=0.5)
    assert bbox == (5, 0, 10, 10)


def test_red_bbox_none_when_rectangle_is_smaller_than_50_px():
    # cols 5-9 / rows 2-5 = 5 x 4 = 20 red px, below MIN_RED_PIXELS (50).
    pixels = _synthetic_red_rectangle_pixels(20, 10, range(5, 10), range(2, 6))
    assert red_bbox(pixels, 20, 10, threshold=0.5) is None


def test_depth_stats_mixed():
    assert depth_stats([[0, 400], [500, 0]]) == (400, 500, 0.5)


def test_depth_stats_all_zero():
    assert depth_stats([[0, 0], [0, 0]]) == (0, 0, 1.0)


def test_region_center():
    assert region_center((5, 2, 10, 6)) == (7, 4)


def test_pose_delta_mm():
    assert pose_delta_mm((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)


def test_is_reddish_true_for_red():
    assert is_reddish([224, 32, 32]) is True


def test_is_reddish_false_for_gray():
    assert is_reddish([120, 120, 120]) is False


def test_verdict_and_skip_shapes():
    assert verdict("thing", True, "all good") == "[PASS] thing: all good"
    assert verdict("thing", False, "off by 5mm") == "[FAIL] thing: off by 5mm"
    assert skip("thing", "needs manual step") == "[SKIP] thing: needs manual step"


def _depth_scene(width: int = 20, height: int = 10, floor_mm: int = 430, block_mm: int = 370):
    rows = [[floor_mm] * width for _ in range(height)]
    for y in range(2, 7):  # 5 rows x 10 cols = 50 px, the minimum
        for x in range(5, 15):
            rows[y][x] = block_mm
    return rows


def test_raised_bbox_finds_the_block_from_depth():
    assert raised_bbox(_depth_scene()) == (5, 2, 15, 7)


def test_raised_bbox_ignores_zero_and_flat_floor():
    rows = _depth_scene(block_mm=430)  # nothing raised
    rows[0][0] = 0  # an invalid pixel must not be mistaken for a rise
    assert raised_bbox(rows) is None
    assert raised_bbox([[0, 0], [0, 0]]) is None


def test_raised_bbox_respects_min_rise():
    rows = _depth_scene(block_mm=415)  # only 15 mm above the floor
    assert raised_bbox(rows) is None
    assert raised_bbox(rows, min_rise_mm=10) == (5, 2, 15, 7)
