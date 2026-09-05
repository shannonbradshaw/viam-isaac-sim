"""Tests for the mock camera scene (FINDINGS CAM-14, slice 2b)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from isaac_module.encoding import depth_to_xyz, intrinsics_from_fov
from isaac_module.mock_camera import (
    FLOOR_FAR_M,
    FLOOR_NEAR_M,
    MOCK_RED_BLOCK_CENTER_M,
    RED_BLOCK_DEPTH_M,
    RED_BLOCK_RGB,
    MockCameraHandle,
)


def make_handle(**attrs: object) -> MockCameraHandle:
    return MockCameraHandle("cam1", attrs)


def test_rgb_and_depth_shapes_and_dtypes() -> None:
    handle = make_handle(depth=True)
    rgb = handle.get_rgb()
    depth = handle.get_depth()
    assert rgb.shape == (480, 848, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (480, 848)
    assert depth.dtype == np.float32


def test_get_depth_raises_when_depth_disabled() -> None:
    handle = make_handle()
    with pytest.raises(RuntimeError):
        handle.get_depth()


def test_red_block_mask_size_and_depth() -> None:
    handle = make_handle(depth=True)
    rgb = handle.get_rgb()
    depth = handle.get_depth()
    mask = (rgb == np.array(RED_BLOCK_RGB)).all(-1)
    assert mask.sum() == 100 * 80
    assert np.all(depth[mask] == RED_BLOCK_DEPTH_M)


def test_nan_band_is_exactly_the_top_rows() -> None:
    handle = make_handle(depth=True)
    depth = handle.get_depth()
    nan_band_rows = 480 // 10  # 48
    assert np.all(np.isnan(depth[:nan_band_rows, :]))
    assert not np.any(np.isnan(depth[nan_band_rows:, :]))


def test_red_cluster_centroid_matches_analytic_center() -> None:
    handle = make_handle(depth=True)
    rgb = handle.get_rgb()
    depth = handle.get_depth()
    k = handle.get_intrinsics()

    xyz, mask = depth_to_xyz(depth, k)
    masked_rgb = rgb[mask]
    red_selector = (masked_rgb == np.array(RED_BLOCK_RGB)).all(-1)
    red_points = xyz[red_selector]

    expected = np.array(handle.red_block_center_m, dtype=np.float32)
    # A centred block would have produced (0, 0, 0.40); the block is
    # deliberately off-centre so this comparison is non-degenerate.
    assert not np.allclose(expected[:2], [0.0, 0.0])
    # float32 mean-of-8000-identical-values accumulation error dominates over
    # the 1e-5 m analytic tolerance; use float64 accumulation to honor it.
    centroid_f64 = red_points.astype(np.float64).mean(axis=0)
    expected_f64 = np.array(handle.red_block_center_m, dtype=np.float64)
    np.testing.assert_allclose(centroid_f64, expected_f64, atol=1e-5)
    np.testing.assert_allclose(
        centroid_f64, np.array(MOCK_RED_BLOCK_CENTER_M, dtype=np.float64), atol=1e-5
    )


def test_default_center_is_off_centre_both_ways() -> None:
    assert MOCK_RED_BLOCK_CENTER_M[0] > 0.05
    assert MOCK_RED_BLOCK_CENTER_M[1] > 0
    assert MOCK_RED_BLOCK_CENTER_M[2] == RED_BLOCK_DEPTH_M


def test_lower_resolution_handle_has_different_x_centre() -> None:
    # The block is defined by fixed pixel offsets from the principal point,
    # not scaled with resolution, while fx scales with width. So the same
    # pixel offset maps to a different metric x at a different resolution.
    default_handle = make_handle(depth=True)
    small_handle = make_handle(width=424, height=240, depth=True)
    assert small_handle.red_block_center_m[0] != default_handle.red_block_center_m[0]


def test_get_intrinsics_matches_shared_helper() -> None:
    handle = make_handle()
    assert handle.get_intrinsics() == intrinsics_from_fov(848, 480, 90.5)


def test_post_reset_increments_reset_count() -> None:
    handle = make_handle()
    assert handle.reset_count == 0
    handle.post_reset()
    handle.post_reset()
    assert handle.reset_count == 2


def test_block_size_mm_unset_matches_default_block_size_mm_attr() -> None:
    handle = make_handle()
    assert handle.block_size_mm is None


@pytest.mark.parametrize("size_mm", [45, 75])
def test_block_size_mm_back_projects_to_configured_size(size_mm: float) -> None:
    handle = make_handle(depth=True, block_size_mm=size_mm)
    assert handle.block_size_mm == size_mm

    rgb = handle.get_rgb()
    mask = (rgb == np.array(RED_BLOCK_RGB)).all(-1)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    height_px = rows.max() - rows.min() + 1
    width_px = cols.max() - cols.min() + 1

    k = handle.get_intrinsics()
    width_m = width_px * RED_BLOCK_DEPTH_M / k.fx
    height_m = height_px * RED_BLOCK_DEPTH_M / k.fy
    one_px_u_mm = 1000 * RED_BLOCK_DEPTH_M / k.fx
    one_px_v_mm = 1000 * RED_BLOCK_DEPTH_M / k.fy

    assert abs(width_m * 1000 - size_mm) <= one_px_u_mm
    assert abs(height_m * 1000 - size_mm) <= one_px_v_mm


def test_block_size_mm_center_matches_sized_bounds_and_is_off_centre() -> None:
    handle = make_handle(depth=True, block_size_mm=45)
    rgb = handle.get_rgb()
    depth = handle.get_depth()
    k = handle.get_intrinsics()

    xyz, mask = depth_to_xyz(depth, k)
    masked_rgb = rgb[mask]
    red_selector = (masked_rgb == np.array(RED_BLOCK_RGB)).all(-1)
    red_points = xyz[red_selector]

    centroid_f64 = red_points.astype(np.float64).mean(axis=0)
    expected_f64 = np.array(handle.red_block_center_m, dtype=np.float64)
    np.testing.assert_allclose(centroid_f64, expected_f64, atol=1e-5)

    assert handle.red_block_center_m[0] != 0.0
    assert handle.red_block_center_m[1] != 0.0


def _side_block_bounds(
    handle: MockCameraHandle,
    column_offset_px: int,
    size_mm: float,
    height_mm: float,
    depth_m: float,
) -> tuple[int, int, int, int]:
    k = handle.get_intrinsics()
    cx_i, cy_i = int(k.cx), int(k.cy)
    span_u = round(size_mm / 1000 * k.fx / depth_m)
    span_v = round(height_mm / 1000 * k.fy / depth_m)
    u0 = cx_i + column_offset_px
    v1 = cy_i
    v0 = v1 - span_v
    return u0, u0 + span_u, v0, v1


def test_side_view_two_blocks_geometry() -> None:
    blocks = [
        {
            "rgb": [200, 30, 30],
            "size_mm": 40.0,
            "height_mm": 60.0,
            "column_offset_px": -80,
            "depth_m": 0.5,
        },
        {
            "rgb": [30, 200, 30],
            "size_mm": 30.0,
            "height_mm": 90.0,
            "column_offset_px": 80,
            "depth_m": 0.6,
        },
    ]
    handle = make_handle(view="side", blocks=blocks, depth=True)
    rgb = handle.get_rgb()
    depth = handle.get_depth()

    for block in blocks:
        u0, u1, v0, v1 = _side_block_bounds(
            handle,
            block["column_offset_px"],
            block["size_mm"],
            block["height_mm"],
            block["depth_m"],
        )
        assert np.all(rgb[v0:v1, u0:u1] == np.array(block["rgb"]))
        assert np.all(depth[v0:v1, u0:u1] == np.float32(block["depth_m"]))


def test_side_view_backdrop_is_nan_above_support_line_off_block() -> None:
    blocks = [
        {
            "rgb": [200, 30, 30],
            "size_mm": 40.0,
            "height_mm": 60.0,
            "column_offset_px": -80,
            "depth_m": 0.5,
        }
    ]
    handle = make_handle(view="side", blocks=blocks, depth=True)
    depth = handle.get_depth()
    k = handle.get_intrinsics()
    cy_i = int(k.cy)

    # A column far from the block's span, above the support line, should be NaN.
    far_column = int(k.cx) + 300
    assert np.all(np.isnan(depth[:cy_i, far_column]))


def test_side_view_floor_rows_at_and_below_support_line_are_ramped() -> None:
    handle = make_handle(view="side", blocks=[], depth=True)
    depth = handle.get_depth()
    k = handle.get_intrinsics()
    cy_i = int(k.cy)

    expected_row_depth = np.linspace(FLOOR_FAR_M, FLOOR_NEAR_M, handle.get_intrinsics().height)
    assert np.allclose(depth[cy_i:, 0], expected_row_depth[cy_i:], atol=1e-6)
    assert not np.any(np.isnan(depth[cy_i:, :]))


def test_side_view_occlusion_nearer_block_wins_in_overlap() -> None:
    blocks = [
        {
            "rgb": [200, 30, 30],
            "size_mm": 80.0,
            "height_mm": 100.0,
            "column_offset_px": 0,
            "depth_m": 0.8,
        },
        {
            "rgb": [30, 200, 30],
            "size_mm": 80.0,
            "height_mm": 60.0,
            "column_offset_px": 0,
            "depth_m": 0.4,
        },
    ]
    handle = make_handle(view="side", blocks=blocks, depth=True)
    rgb = handle.get_rgb()
    depth = handle.get_depth()
    k = handle.get_intrinsics()
    cx_i, cy_i = int(k.cx), int(k.cy)

    # Both blocks share the same column span; near the support line both
    # rectangles overlap, and the nearer (depth 0.4) block should win there.
    overlap_row = cy_i - 2
    assert np.all(rgb[overlap_row, cx_i] == np.array([30, 200, 30]))
    assert depth[overlap_row, cx_i] == np.float32(0.4)


def test_tallest_block_height_mm_is_max_of_blocks() -> None:
    blocks = [
        {
            "rgb": [1, 1, 1],
            "size_mm": 40.0,
            "height_mm": 60.0,
            "column_offset_px": -80,
            "depth_m": 0.5,
        },
        {
            "rgb": [2, 2, 2],
            "size_mm": 30.0,
            "height_mm": 90.0,
            "column_offset_px": 80,
            "depth_m": 0.6,
        },
    ]
    handle = make_handle(view="side", blocks=blocks)
    assert handle.tallest_block_height_mm == 90.0


def test_tallest_block_height_mm_is_none_for_top_view_and_no_blocks() -> None:
    assert make_handle().tallest_block_height_mm is None
    assert make_handle(view="side", blocks=[]).tallest_block_height_mm is None


def test_side_view_float_coerced_attrs_build_same_scene_as_ints() -> None:
    int_blocks = [
        {
            "rgb": [10, 20, 30],
            "size_mm": 40,
            "height_mm": 60,
            "column_offset_px": -50,
            "depth_m": 0.5,
        }
    ]
    float_blocks = [
        {
            "rgb": [10.0, 20.0, 30.0],
            "size_mm": 40.0,
            "height_mm": 60.0,
            "column_offset_px": -50.0,
            "depth_m": 0.5,
        }
    ]
    int_handle = make_handle(view="side", blocks=int_blocks, depth=True)
    float_handle = make_handle(view="side", blocks=float_blocks, depth=True)

    assert np.array_equal(int_handle.get_rgb(), float_handle.get_rgb())
    depth_a, depth_b = int_handle.get_depth(), float_handle.get_depth()
    assert np.array_equal(depth_a, depth_b, equal_nan=True)


def test_get_frame_shares_sim_time_within_one_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = make_handle()
    fake_time = [1000.001]
    monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

    frame1 = handle.get_frame()
    fake_time[0] += 1.0 / 120  # still within the same 1/60s tick
    frame2 = handle.get_frame()

    assert frame1.sim_time == frame2.sim_time
