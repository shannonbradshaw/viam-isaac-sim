"""IsaacCameraHandle + the camera-creation helpers, driven through duck-typed
fake camera objects (no Isaac here). CAM-1/2/3/4/9/17, the create_camera attrs
contract, and the CAM-10 local-pose sim side."""

import math

import numpy as np
import pytest

from isaac_module.camera_base import NoFrameYetError
from isaac_module.encoding import intrinsics_from_fov
from isaac_module.sim_manager import (
    IsaacCameraHandle,
    _configure_camera_optics,
    _place_camera,
)


class FakeCam:
    """Records every setter call; returns are pre-configured per test."""

    def __init__(
        self,
        *,
        resolution=(848, 480),
        horizontal_aperture=20.0,
        vertical_aperture=15.0,
        focal_length=10.0,
        name="fake-cam",
    ):
        self.name = name
        self._resolution = resolution
        self._horizontal_aperture = horizontal_aperture
        self._vertical_aperture = vertical_aperture
        self._focal_length = focal_length
        self._clipping_range = (1.0, 1000000.0)
        self.calls: list[tuple[str, tuple]] = []
        self._rgba_sequence: list = []
        self._depth_sequence: list = []

    # -- setters (recorded) -------------------------------------------------
    def set_focal_length(self, value):
        self._focal_length = value
        self.calls.append(("set_focal_length", (value,)))

    def set_vertical_aperture(self, value):
        self._vertical_aperture = value
        self.calls.append(("set_vertical_aperture", (value,)))

    def set_clipping_range(self, near, far):
        self._clipping_range = (near, far)
        self.calls.append(("set_clipping_range", (near, far)))

    def add_distance_to_image_plane_to_frame(self):
        self.calls.append(("add_distance_to_image_plane_to_frame", ()))

    def set_frequency(self, value):
        self.calls.append(("set_frequency", (value,)))

    def set_local_pose(self, position, orientation, camera_axes):
        self.calls.append(("set_local_pose", (position, orientation, camera_axes)))

    def set_world_pose(self, position, orientation, camera_axes):
        self.calls.append(("set_world_pose", (position, orientation, camera_axes)))

    def initialize(self):
        self.calls.append(("initialize", ()))

    def post_reset(self):
        self.calls.append(("post_reset", ()))

    # -- getters --------------------------------------------------------
    def get_resolution(self):
        return self._resolution

    def get_horizontal_aperture(self):
        return self._horizontal_aperture

    def get_vertical_aperture(self):
        return self._vertical_aperture

    def get_focal_length(self):
        return self._focal_length

    def get_clipping_range(self):
        return self._clipping_range

    # -- frame capture (configurable) ------------------------------------
    def set_rgba_sequence(self, seq):
        self._rgba_sequence = list(seq)

    def set_depth_sequence(self, seq):
        self._depth_sequence = list(seq)

    def get_rgba(self):
        return self._rgba_sequence.pop(0)

    def get_depth(self):
        return self._depth_sequence.pop(0)

    def call_count(self, name):
        return sum(1 for n, _ in self.calls if n == name)


class FakeWorld:
    def __init__(self, current_time=0.0):
        self.current_time = current_time


class FakeSim:
    def __init__(self):
        self.world = FakeWorld()

    def run(self, fn, timeout=30.0):
        return fn()


def _rgb_frame(width=848, height=480):
    return np.zeros((height, width, 4), dtype=np.uint8)


def _make_handle(cam, *, depth_enabled=False, sleep=None):
    sim = FakeSim()
    recorded_sleeps: list[float] = []

    def _sleep(seconds):
        recorded_sleeps.append(seconds)

    handle = IsaacCameraHandle(
        sim,
        cam,
        depth_enabled=depth_enabled,
        image_format="png",
        frequency=None,
        sleep=sleep or _sleep,
    )
    return handle, sim, recorded_sleeps


# ---------------------------------------------------------------------
# per-step cache (CAM-9)
# ---------------------------------------------------------------------


def test_get_frame_caches_by_sim_time():
    cam = FakeCam()
    cam.set_rgba_sequence([_rgb_frame(), _rgb_frame()])
    handle, sim, _ = _make_handle(cam)

    handle.get_frame()
    handle.get_frame()
    assert len(cam._rgba_sequence) == 1  # only one grab so far

    sim.world.current_time = 1.0
    handle.get_frame()
    assert len(cam._rgba_sequence) == 0  # advancing time triggers a new grab


# ---------------------------------------------------------------------
# warm-up retry (CAM-2)
# ---------------------------------------------------------------------


def test_get_frame_retries_through_warmup_then_succeeds():
    cam = FakeCam()
    cam.set_rgba_sequence([None, None, None, _rgb_frame()])
    handle, _, sleeps = _make_handle(cam)

    frame = handle.get_frame()
    assert frame.rgb.shape == (480, 848, 3)
    assert len(sleeps) == 3


def test_get_frame_raises_after_warmup_retries_exhausted():
    from isaac_module.sim_manager import WARMUP_RETRIES

    cam = FakeCam()
    cam.set_rgba_sequence([None] * (WARMUP_RETRIES + 5))
    handle, _, sleeps = _make_handle(cam, sleep=lambda s: None)

    with pytest.raises(NoFrameYetError):
        handle.get_frame()

    # exactly WARMUP_RETRIES grab attempts were made
    assert (WARMUP_RETRIES + 5) - len(cam._rgba_sequence) == WARMUP_RETRIES


# ---------------------------------------------------------------------
# depth (CAM-1)
# ---------------------------------------------------------------------


def test_get_depth_squeezes_and_casts_to_float32():
    cam = FakeCam()
    cam.set_rgba_sequence([_rgb_frame()])
    cam.set_depth_sequence([np.ones((480, 848, 1), dtype=np.float64) * 2.5])
    handle, _, _ = _make_handle(cam, depth_enabled=True)

    depth = handle.get_depth()
    assert depth.shape == (480, 848)
    assert depth.dtype == np.float32


def test_get_depth_disabled_raises_and_never_calls_get_depth():
    cam = FakeCam()
    cam.set_rgba_sequence([_rgb_frame()])
    handle, _, _ = _make_handle(cam, depth_enabled=False)

    with pytest.raises(RuntimeError):
        handle.get_depth()
    assert cam._depth_sequence == []  # get_depth was never called/popped


# ---------------------------------------------------------------------
# intrinsics
# ---------------------------------------------------------------------


def test_get_intrinsics_matches_intrinsics_from_fov_for_square_pixels():
    fov_deg = 90.5
    horizontal_aperture = 20.0
    focal_length = horizontal_aperture / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    cam = FakeCam(
        resolution=(848, 480),
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=horizontal_aperture * 480 / 848,  # square pixels
        focal_length=focal_length,
    )
    handle, _, _ = _make_handle(cam)

    expected = intrinsics_from_fov(848, 480, fov_deg)
    actual = handle.get_intrinsics()
    assert actual.fx == pytest.approx(expected.fx, abs=1e-6)
    assert actual.fy == pytest.approx(expected.fy, abs=1e-6)
    assert actual.cx == expected.cx
    assert actual.cy == expected.cy


def test_get_intrinsics_reads_nonsquare_vertical_aperture():
    horizontal_aperture = 20.0
    vertical_aperture = 12.0  # deliberately non-square
    focal_length = 10.0
    cam = FakeCam(
        resolution=(848, 480),
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        focal_length=focal_length,
    )
    handle, _, _ = _make_handle(cam)

    intrinsics = handle.get_intrinsics()
    assert intrinsics.fx != intrinsics.fy


def test_get_intrinsics_zero_focal_length_raises():
    cam = FakeCam(focal_length=0.0)
    handle, _, _ = _make_handle(cam)

    with pytest.raises(RuntimeError):
        handle.get_intrinsics()


# ---------------------------------------------------------------------
# _configure_camera_optics (CAM-3, CAM-4, CAM-1, frequency)
# ---------------------------------------------------------------------


def test_configure_camera_optics_sets_clip_focal_aperture_depth_frequency():
    horizontal_aperture = 20.0
    cam = FakeCam(resolution=(848, 480), horizontal_aperture=horizontal_aperture)
    attrs = {"depth": True, "frequency": 30, "fov_deg": 90.5}

    _configure_camera_optics(cam, attrs)

    assert ("set_clipping_range", (0.05, 10.0)) in cam.calls
    expected_focal = horizontal_aperture / (2.0 * math.tan(math.radians(90.5 / 2.0)))
    focal_calls = [args[0] for name, args in cam.calls if name == "set_focal_length"]
    assert focal_calls[-1] == pytest.approx(expected_focal, abs=1e-9)
    assert ("set_vertical_aperture", (horizontal_aperture * 480 / 848,)) in cam.calls
    assert cam.call_count("add_distance_to_image_plane_to_frame") == 1
    assert ("set_frequency", (30.0,)) in cam.calls


def test_configure_camera_optics_no_depth_skips_annotator():
    cam = FakeCam(resolution=(848, 480))
    _configure_camera_optics(cam, {"fov_deg": 90.5})
    assert cam.call_count("add_distance_to_image_plane_to_frame") == 0


def test_configure_camera_optics_custom_clip_range():
    cam = FakeCam(resolution=(848, 480))
    _configure_camera_optics(cam, {"clip_near": 0.1, "clip_far": 3.0})
    assert ("set_clipping_range", (0.1, 3.0)) in cam.calls


# ---------------------------------------------------------------------
# _place_camera (CAM-10)
# ---------------------------------------------------------------------


def test_place_camera_parent_prim_with_frame_orientation_uses_ros_axes():
    cam = FakeCam()
    attrs = {
        "parent_prim": "/World/arm/wrist_3_link",
        "local_position": [0.0, 0.0, 0.06],
        "local_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    _place_camera(cam, attrs)
    assert cam.calls[-1] == (
        "set_local_pose",
        ([0.0, 0.0, 0.06], [1.0, 0.0, 0.0, 0.0], "ros"),
    )


def test_place_camera_parent_prim_legacy_rpy_uses_usd_axes():
    cam = FakeCam()
    attrs = {
        "parent_prim": "/World/arm/wrist_3_link",
        "local_orientation_rpy_deg": [180.0, 0.0, 0.0],
    }
    _place_camera(cam, attrs)
    assert len(cam.calls) == 1
    name, args = cam.calls[-1]
    assert name == "set_local_pose"
    assert args[2] == "usd"


# ---------------------------------------------------------------------
# post_reset (CAM-17)
# ---------------------------------------------------------------------


def test_post_reset_drops_cache_and_calls_cam_post_reset():
    cam = FakeCam()
    cam.set_rgba_sequence([_rgb_frame(), _rgb_frame()])
    handle, sim, _ = _make_handle(cam)

    handle.get_frame()
    handle.post_reset()
    assert cam.call_count("post_reset") == 1

    # same sim_time, but the cache was dropped so this must grab again
    handle.get_frame()
    assert len(cam._rgba_sequence) == 0


class FakeCamNoPostReset(FakeCam):
    post_reset = None  # not present as a callable


def test_post_reset_falls_back_to_initialize_when_absent():
    cam = FakeCamNoPostReset()
    cam.set_rgba_sequence([_rgb_frame()])
    handle, _, _ = _make_handle(cam)

    handle.post_reset()
    assert cam.call_count("initialize") == 1


# ---------------------------------------------------------------------
# create_camera wires post_reset into the sim's registry (CAM-17, mock mode)
# ---------------------------------------------------------------------


def test_create_camera_registers_post_reset_hook(sim):
    handle = sim.create_camera("hook-cam", {"world": "sim-world"})

    fired: list[None] = []
    handle.post_reset = lambda: fired.append(None)

    sim.reset()
    assert len(fired) == 1


def test_create_camera_called_twice_registers_hook_only_once(sim):
    """viam-server re-runs reconfigure -> create_camera on every config
    change; _cached_handle returns the same handle both times, so the
    post-reset hook must only be registered on first construction."""
    sim.create_camera("dup-cam", {"world": "sim-world"})
    handle = sim.create_camera("dup-cam", {"world": "sim-world"})

    call_count = 0

    def _count_post_reset():
        nonlocal call_count
        call_count += 1

    handle.post_reset = _count_post_reset

    sim.reset()
    assert call_count == 1



class FakeCamWithFrameInfo(FakeCam):
    def __init__(self, rendering_times, **kw):
        super().__init__(**kw)
        self._rendering_times = list(rendering_times)

    def get_current_frame(self):
        return {"rendering_time": self._rendering_times.pop(0)}


def test_stale_frame_is_not_served_until_the_renderer_catches_up():
    """After a boot the renderer lags the simulation by seconds and the first
    reads show the scene from before the arm moved (2026-09-04)."""
    import numpy as np

    cam = FakeCamWithFrameInfo(rendering_times=[0.0, 0.0, 9.9])
    cam.set_rgba_sequence([np.zeros((4, 4, 4), dtype=np.uint8)])
    handle, sim, sleeps = _make_handle(cam)
    sim.world.current_time = 10.0
    frame = handle.get_frame()
    assert frame.sim_time == 10.0
    assert len(sleeps) == 2  # two stale reads, then a frame within STALE_FRAME_S
