"""WorldHandle seam contract (SCN-16): the scene mutations the world
component's DoCommand verbs drive through PropGeometry / prop_spawn_orientation
/ prop_box_dims / sample_prop_positions / MockWorldHandle / IsaacWorldHandle."""

import math
import threading

import numpy as np
import pytest

from isaac_module.sim_manager import (
    DEFAULT_MIN_SEPARATION_M,
    PROP_REST_EPSILON_M,
    IsaacWorldHandle,
    MockWorldHandle,
    PropGeometry,
    RandomizeResult,
    SimConfig,
    SimManager,
    prop_box_dims,
    sample_prop_positions,
)
from isaac_module.spatial import quat_from_euler_deg

SIM_THREAD_JOIN_TIMEOUT_S = 5


def _cube(name: str, **extra) -> dict:
    return {"type": "cube", "name": name, "size": 0.05, **extra}


def _mock_handle(props: list[dict]) -> MockWorldHandle:
    # production invariant: MockWorldHandle is only built by the mock boot
    # path, so the manager it wraps always has mock=True (a sized randomize
    # relies on it - _reset_world skips the isaac world only in mock mode)
    manager = SimManager()
    manager.mock = True
    return MockWorldHandle(manager, props)


# ----------------------------------------------------------------------
# registry / prop_geometries
# ----------------------------------------------------------------------


def test_registered_props_appear_in_registry_with_spawn_attrs():
    handle = _mock_handle([_cube("block", position=[0.1, 0.2, 0.03])])
    registry = handle.registry()
    assert set(registry) == {"block"}
    entry = registry["block"]
    assert entry["spawn_position"] == (0.1, 0.2, 0.03)
    assert entry["spawn_orientation"] == (1.0, 0.0, 0.0, 0.0)
    assert entry["position"] == entry["spawn_position"]


def test_prop_geometries_one_entry_per_prop_matching_config():
    handle = _mock_handle(
        [
            _cube("a", position=[0.0, 0.0, 0.03], color=[1.0, 0.0, 0.0]),
            _cube("b", position=[0.2, 0.0, 0.03], size=0.1, fixed=True),
        ]
    )
    geoms = {g.name: g for g in handle.prop_geometries()}
    assert set(geoms) == {"a", "b"}

    a = geoms["a"]
    assert isinstance(a, PropGeometry)
    assert a.box_dims_m == (0.05, 0.05, 0.05)
    assert a.position_m == (0.0, 0.0, 0.03)
    assert a.orientation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert a.color == (1.0, 0.0, 0.0)
    assert a.fixed is False

    b = geoms["b"]
    assert b.box_dims_m == (0.1, 0.1, 0.1)
    assert b.fixed is True
    assert b.color is None


def test_prop_with_rpy_orientation_reports_rotated_pose():
    handle = _mock_handle([_cube("tilted", orientation_rpy_deg=[0.0, 0.0, 90.0])])
    expected = quat_from_euler_deg(0.0, 0.0, 90.0)
    (geom,) = handle.prop_geometries()
    assert geom.orientation_wxyz == expected


# ----------------------------------------------------------------------
# set_prop_pose / reset
# ----------------------------------------------------------------------


def test_set_prop_pose_then_reset_restores_configured_pose():
    handle = _mock_handle([_cube("block", position=[0.0, 0.0, 0.03])])
    handle.set_prop_pose("block", (1.0, 1.0, 1.0), (0.0, 1.0, 0.0, 0.0))

    moved = handle.registry()["block"]
    assert moved["position"] == (1.0, 1.0, 1.0)
    assert moved["orientation"] == (0.0, 1.0, 0.0, 0.0)

    handle.reset(soft=True)

    restored = handle.registry()["block"]
    assert restored["position"] == (0.0, 0.0, 0.03)
    assert restored["orientation"] == (1.0, 0.0, 0.0, 0.0)


def test_set_prop_pose_keeps_orientation_when_none_passed():
    handle = _mock_handle([_cube("block", orientation_rpy_deg=[0.0, 0.0, 45.0])])
    expected_orientation = quat_from_euler_deg(0.0, 0.0, 45.0)

    handle.set_prop_pose("block", (0.5, 0.5, 0.03))

    entry = handle.registry()["block"]
    assert entry["position"] == (0.5, 0.5, 0.03)
    assert entry["orientation"] == expected_orientation


def test_set_prop_pose_unknown_name_raises_value_error():
    handle = _mock_handle([_cube("block")])
    with pytest.raises(ValueError):
        handle.set_prop_pose("no-such-prop", (0.0, 0.0, 0.0))


def test_soft_reset_restores_poses_without_full_sim_reset():
    handle = _mock_handle([_cube("block", position=[0.0, 0.0, 0.03])])
    handle.set_prop_pose("block", (2.0, 2.0, 2.0))
    handle.reset(soft=True)
    assert handle.registry()["block"]["position"] == (0.0, 0.0, 0.03)


# ----------------------------------------------------------------------
# spawn_prop
# ----------------------------------------------------------------------


def test_spawn_prop_adds_prop_that_appears_in_geometries_and_survives_reset():
    handle = _mock_handle([_cube("existing")])
    handle.spawn_prop(_cube("new-block", position=[0.4, 0.4, 0.03]))

    names = {g.name for g in handle.prop_geometries()}
    assert names == {"existing", "new_block"}

    handle.set_prop_pose("new_block", (9.0, 9.0, 9.0))
    handle.reset(soft=True)

    restored = {g.name: g for g in handle.prop_geometries()}["new_block"]
    assert restored.position_m == (0.4, 0.4, 0.03)


def test_spawn_prop_duplicate_name_raises_value_error():
    handle = _mock_handle([_cube("dup")])
    with pytest.raises(ValueError):
        handle.spawn_prop(_cube("dup"))


# ----------------------------------------------------------------------
# randomize_props
# ----------------------------------------------------------------------


REGION = ((0.0, 0.0, 0.5), (1.0, 1.0, 0.5))


def test_randomize_props_is_deterministic_for_a_given_seed():
    handle = _mock_handle([_cube("a"), _cube("b")])
    first = handle.randomize_props(["a", "b"], REGION, seed=1)

    handle2 = _mock_handle([_cube("a"), _cube("b")])
    second = handle2.randomize_props(["a", "b"], REGION, seed=1)

    assert first == second
    assert isinstance(first, RandomizeResult)


def test_randomize_props_respects_region_and_separation_bounds():
    handle = _mock_handle([_cube("a"), _cube("b"), _cube("c")])
    dims = prop_box_dims(_cube("a"))
    result = handle.randomize_props(["a", "b", "c"], REGION, seed=1)

    (lo_x, lo_y, z0), (hi_x, hi_y, z1) = REGION
    face_z = (z0 + z1) / 2.0
    half = dims[0] / 2.0

    positions = list(result.positions_m.values())
    for x, y, z in positions:
        assert lo_x + half <= x <= hi_x - half
        assert lo_y + half <= y <= hi_y - half
        assert z == pytest.approx(face_z + dims[2] / 2.0 + PROP_REST_EPSILON_M)

    for i, (x0, y0, _z0) in enumerate(positions):
        for x1, y1, _z1 in positions[i + 1 :]:
            assert math.hypot(x0 - x1, y0 - y1) >= DEFAULT_MIN_SEPARATION_M


def test_randomize_props_unknown_name_raises_value_error():
    handle = _mock_handle([_cube("a")])
    with pytest.raises(ValueError):
        handle.randomize_props(["a", "ghost"], REGION, seed=1)


def test_randomize_props_no_range_matches_sample_prop_positions_baseline():
    dims = {"a": prop_box_dims(_cube("a")), "b": prop_box_dims(_cube("b"))}
    baseline = sample_prop_positions(dims, REGION, seed=5)
    handle = _mock_handle([_cube("a"), _cube("b")])
    result = handle.randomize_props(["a", "b"], REGION, seed=5)
    assert result.positions_m == baseline


def test_randomize_props_size_range_reproduces_sizes_and_positions_for_a_seed():
    handle = _mock_handle([_cube("a"), _cube("b")])
    size_range = {"a": (0.03, 0.09), "b": (0.03, 0.09)}
    first = handle.randomize_props(["a", "b"], REGION, seed=2, size_range_m=size_range)

    handle2 = _mock_handle([_cube("a"), _cube("b")])
    second = handle2.randomize_props(["a", "b"], REGION, seed=2, size_range_m=size_range)

    assert first == second
    assert first.dims_m["a"] != prop_box_dims(_cube("a"))  # actually drew a new size


def test_randomize_props_drawn_dims_flow_into_prop_geometries_mock():
    handle = _mock_handle([_cube("a")])
    result = handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.09, 0.09)})
    (geom,) = handle.prop_geometries()
    assert geom.box_dims_m == pytest.approx(result.dims_m["a"])
    assert geom.box_dims_m == pytest.approx((0.09, 0.09, 0.09))


def test_randomize_props_edge_aware_separation_at_the_default_min_separation():
    handle = _mock_handle([_cube("a"), _cube("b")])
    result = handle.randomize_props(
        ["a", "b"],
        REGION,
        seed=1,
        min_separation_m=0.1,
        size_range_m={"a": (0.09, 0.09), "b": (0.09, 0.09)},
    )
    (ax, ay, _az), (bx, by, _bz) = result.positions_m.values()
    assert math.hypot(ax - bx, ay - by) >= 0.1


def test_randomize_props_edge_aware_separation_exceeds_min_separation():
    handle = _mock_handle([_cube("a"), _cube("b")])
    result = handle.randomize_props(
        ["a", "b"],
        REGION,
        seed=1,
        min_separation_m=0.02,
        size_range_m={"a": (0.12, 0.12), "b": (0.12, 0.12)},
    )
    (ax, ay, _az), (bx, by, _bz) = result.positions_m.values()
    edge_bound = 0.12 + 0.01  # (0.12 + 0.12) / 2 + PROP_EDGE_CLEARANCE_M
    assert math.hypot(ax - bx, ay - by) >= edge_bound
    assert (
        edge_bound > 0.02
    )  # the old plain-min_separation rule would have allowed this pair closer


def test_randomize_props_rescale_is_absolute_not_compounding_mock():
    handle = _mock_handle([_cube("a", size=0.05)])
    first = handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.09, 0.09)})
    assert first.dims_m["a"] == pytest.approx((0.09, 0.09, 0.09))

    second = handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.03, 0.03)})
    assert second.dims_m["a"] == pytest.approx((0.03, 0.03, 0.03))

    third = handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.09, 0.09)})
    assert third.dims_m["a"] == pytest.approx((0.09, 0.09, 0.09))


def test_randomize_props_size_range_non_cube_prop_raises_value_error():
    handle = _mock_handle(
        [{"type": "usd", "name": "u", "usd_path": "x.usd", "box_dims": [0.1, 0.1, 0.1]}]
    )
    with pytest.raises(ValueError):
        handle.randomize_props(["u"], REGION, seed=1, size_range_m={"u": (0.03, 0.09)})


def test_randomize_props_size_range_key_not_in_names_raises_value_error():
    handle = _mock_handle([_cube("a"), _cube("b")])
    with pytest.raises(ValueError):
        handle.randomize_props(["a"], REGION, seed=1, size_range_m={"b": (0.03, 0.09)})


# ----------------------------------------------------------------------
# sample_prop_positions (unit-level)
# ----------------------------------------------------------------------


def test_sample_prop_positions_is_deterministic():
    dims = {"a": (0.05, 0.05, 0.05), "b": (0.05, 0.05, 0.05)}
    first = sample_prop_positions(dims, REGION, seed=7)
    second = sample_prop_positions(dims, REGION, seed=7)
    assert first == second


def test_sample_prop_positions_raises_when_region_too_small_for_footprint():
    dims = {"huge": (10.0, 10.0, 0.05)}
    with pytest.raises(ValueError):
        sample_prop_positions(dims, REGION, seed=1)


def test_sample_prop_positions_raises_when_crowded_region_has_no_room():
    tiny_region = ((0.0, 0.0, 0.5), (0.2, 0.2, 0.5))
    dims = {f"p{i}": (0.05, 0.05, 0.05) for i in range(20)}
    with pytest.raises(ValueError):
        sample_prop_positions(dims, tiny_region, seed=1, min_separation_m=0.15)


def test_sample_prop_positions_placements_respect_both_bounds():
    dims = {"a": (0.05, 0.05, 0.05), "b": (0.05, 0.05, 0.05), "c": (0.05, 0.05, 0.05)}
    placed = sample_prop_positions(dims, REGION, seed=3)
    (lo_x, lo_y, z0), (hi_x, hi_y, z1) = REGION
    half = 0.025
    positions = list(placed.values())
    for x, y, _z in positions:
        assert lo_x + half <= x <= hi_x - half
        assert lo_y + half <= y <= hi_y - half
    for i, (x0, y0, _z0) in enumerate(positions):
        for x1, y1, _z1 in positions[i + 1 :]:
            assert math.hypot(x0 - x1, y0 - y1) >= DEFAULT_MIN_SEPARATION_M


# ----------------------------------------------------------------------
# boot-recording: a fresh SimManager boots mock with props (own thread)
# ----------------------------------------------------------------------


def test_fresh_sim_manager_boot_registers_configured_props():
    manager = SimManager()
    sim_thread = threading.Thread(target=manager.main_loop, daemon=True)
    sim_thread.start()
    try:
        manager.ensure_booted(
            SimConfig(mock=True, props=[_cube("boot-block", position=[0.1, 0.1, 0.03])])
        )
        registry = manager.world_handle().registry()
        assert "boot_block" in registry
        assert registry["boot_block"]["spawn_position"] == (0.1, 0.1, 0.03)
    finally:
        manager.request_stop()
        sim_thread.join(timeout=SIM_THREAD_JOIN_TIMEOUT_S)


# ----------------------------------------------------------------------
# IsaacWorldHandle, driven with a fake isaac namespace (cheap coverage only:
# spawn-orientation plumbing, prop_geometries, set_prop_pose, spawn_prop,
# reset(soft=True) - no fake PhysX).
# ----------------------------------------------------------------------


class _FakeXForm:
    """Shared pose store keyed by prim_path; stands in for
    isaacsim's SingleXFormPrim AND the Dynamic/FixedCuboid constructors
    (both end up recording a pose the same way here)."""

    _STORE: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __init__(self, prim_path: str, name: str = "", position=None, orientation=None, **_ignored):
        self.prim_path = prim_path
        self.name = name
        if position is not None or orientation is not None:
            self.set_world_pose(position=position, orientation=orientation)
        elif prim_path not in self._STORE:
            self._STORE[prim_path] = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    def set_world_pose(self, position=None, orientation=None) -> None:
        pos, quat = self._STORE.get(self.prim_path, (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
        if position is not None:
            pos = np.array([float(v) for v in position])
        if orientation is not None:
            quat = np.array([float(v) for v in orientation])
        self._STORE[self.prim_path] = (pos, quat)

    def get_world_pose(self):
        return self._STORE[self.prim_path]


# ordered trace of the sized-randomize sim calls: the GPU defect was purely
# an ordering one (a scale authored mid-play is reverted by the stop inside
# reset), so the assertion has to see the sequence, not just the counts
FAKE_SIM_EVENTS: list[str] = []


class _FakeCubeObject(_FakeXForm):
    """Stands in for the scene-registered Dynamic/FixedCuboid object
    (SCN-16 sized props): adds set_local_scale, the API IsaacWorldHandle
    rescales through."""

    _SCALE_STORE: dict[str, np.ndarray] = {}

    def set_local_scale(self, scale) -> None:
        FAKE_SIM_EVENTS.append("scale")
        self._SCALE_STORE[self.prim_path] = np.array([float(v) for v in scale])

    def get_local_scale(self):
        return self._SCALE_STORE.get(self.prim_path, np.array([1.0, 1.0, 1.0]))


class _FakeScene:
    def __init__(self) -> None:
        self._objects: dict[str, object] = {}

    def add(self, obj) -> None:
        name = getattr(obj, "name", None)
        if name:
            self._objects[name] = obj

    def get_object(self, name):
        return self._objects.get(name)


class _FakeWorld:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.reset_calls = 0
        self.stop_calls = 0

    def reset(self) -> None:
        FAKE_SIM_EVENTS.append("reset")
        self.reset_calls += 1

    def stop(self) -> None:
        FAKE_SIM_EVENTS.append("stop")
        self.stop_calls += 1


class _FakeIsaacNamespace:
    SingleXFormPrim = _FakeXForm
    DynamicCuboid = _FakeCubeObject
    FixedCuboid = _FakeCubeObject
    PhysicsMaterial = None
    PhysxSchema = None

    @staticmethod
    def add_reference_to_stage(usd_path: str, prim_path: str) -> None:
        pass


@pytest.fixture
def isaac_handle():
    _FakeXForm._STORE.clear()
    _FakeCubeObject._SCALE_STORE.clear()
    FAKE_SIM_EVENTS.clear()
    manager = SimManager()
    manager.mock = False
    manager.world = _FakeWorld()
    manager._isaac = _FakeIsaacNamespace()
    manager._booted.set()
    manager._sim_thread_id = threading.get_ident()
    for prop in [_cube("a", position=[0.0, 0.0, 0.03])]:
        manager._spawn_prop(prop)
    return IsaacWorldHandle(manager)


def test_isaac_handle_spawn_orientation_is_recorded_on_the_prim(isaac_handle):
    isaac_handle._sim._spawn_prop(_cube("tilted", orientation_rpy_deg=[0.0, 0.0, 90.0]))
    (geom,) = [g for g in isaac_handle.prop_geometries() if g.name == "tilted"]
    expected = quat_from_euler_deg(0.0, 0.0, 90.0)
    assert geom.orientation_wxyz == pytest.approx(expected)


def test_isaac_handle_prop_geometries_matches_config(isaac_handle):
    (geom,) = isaac_handle.prop_geometries()
    assert geom.name == "a"
    assert geom.box_dims_m == (0.05, 0.05, 0.05)
    assert geom.position_m == pytest.approx((0.0, 0.0, 0.03))


def test_isaac_handle_set_prop_pose_then_reset_restores_spawn_pose(isaac_handle):
    isaac_handle.set_prop_pose("a", (5.0, 5.0, 5.0))
    moved = isaac_handle.prop_geometries()[0]
    assert moved.position_m == pytest.approx((5.0, 5.0, 5.0))

    isaac_handle.reset(soft=True)

    restored = isaac_handle.prop_geometries()[0]
    assert restored.position_m == pytest.approx((0.0, 0.0, 0.03))


def test_isaac_handle_set_prop_pose_unknown_name_raises_value_error(isaac_handle):
    with pytest.raises(ValueError):
        isaac_handle.set_prop_pose("ghost", (0.0, 0.0, 0.0))


def test_isaac_handle_spawn_prop_appears_and_survives_reset(isaac_handle):
    isaac_handle.spawn_prop(_cube("new-block", position=[0.2, 0.2, 0.03]))
    names = {g.name for g in isaac_handle.prop_geometries()}
    assert "new_block" in names
    assert isaac_handle._sim.world.reset_calls == 1


def test_isaac_handle_spawn_prop_duplicate_name_raises_value_error(isaac_handle):
    with pytest.raises(ValueError):
        isaac_handle.spawn_prop(_cube("a"))


def test_isaac_handle_randomize_props_rescales_and_updates_prop_geometries(isaac_handle):
    result = isaac_handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.09, 0.09)})
    assert result.dims_m["a"] == pytest.approx((0.09, 0.09, 0.09))

    (geom,) = isaac_handle.prop_geometries()
    assert geom.box_dims_m == pytest.approx((0.09, 0.09, 0.09))

    scene_object = isaac_handle._sim.world.scene.get_object("a")
    assert scene_object.get_local_scale() == pytest.approx(np.array([1.8, 1.8, 1.8]))


def test_isaac_handle_randomize_props_rescale_is_absolute_not_compounding(isaac_handle):
    first = isaac_handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.09, 0.09)})
    assert first.dims_m["a"] == pytest.approx((0.09, 0.09, 0.09))

    second = isaac_handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.03, 0.03)})
    assert second.dims_m["a"] == pytest.approx((0.03, 0.03, 0.03))

    third = isaac_handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.09, 0.09)})
    assert third.dims_m["a"] == pytest.approx((0.09, 0.09, 0.09))


def test_isaac_handle_randomize_props_size_range_non_cube_prop_raises_value_error(isaac_handle):
    isaac_handle._sim._spawn_prop(
        {"type": "usd", "name": "u", "usd_path": "x.usd", "box_dims": [0.1, 0.1, 0.1]}
    )
    with pytest.raises(ValueError):
        isaac_handle.randomize_props(["u"], REGION, seed=1, size_range_m={"u": (0.03, 0.09)})


def test_isaac_handle_sized_randomize_stops_scales_then_resets_in_order(isaac_handle):
    """GPU failures (phase-3 checklist): a scale written mid-play invalidates
    PhysX's tensor view ('Failed to get rigid body transforms from backend'),
    and a scale written before the reset is REVERTED by the stop inside it
    (drawn 44.7 mm, block stayed 60 mm). The only working order is stop ->
    scale -> reset; an unsized randomize must do none of it."""
    hook_runs: list[int] = []
    isaac_handle._sim.register_post_reset(lambda: hook_runs.append(1), owner="test")

    isaac_handle.randomize_props(["a"], REGION, seed=1)
    assert FAKE_SIM_EVENTS == []
    assert hook_runs == []

    isaac_handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.09, 0.09)})
    assert FAKE_SIM_EVENTS == ["stop", "scale", "reset"]
    assert hook_runs == [1]


def test_mock_sized_randomize_snaps_unnamed_props_to_spawn_like_isaac():
    """API parity with the isaac handle's full-reset pattern: a sized
    randomize returns every prop to its spawn pose before the named ones
    teleport; an unsized randomize leaves other props where they are."""
    handle = _mock_handle([_cube("a"), _cube("bystander", position=[0.1, 0.2, 0.03])])
    handle.set_prop_pose("bystander", (0.5, 0.5, 0.03))

    handle.randomize_props(["a"], REGION, seed=1)
    (bystander,) = [g for g in handle.prop_geometries() if g.name == "bystander"]
    assert bystander.position_m == pytest.approx((0.5, 0.5, 0.03))

    handle.randomize_props(["a"], REGION, seed=1, size_range_m={"a": (0.04, 0.04)})
    (bystander,) = [g for g in handle.prop_geometries() if g.name == "bystander"]
    assert bystander.position_m == pytest.approx((0.1, 0.2, 0.03))


def test_sample_prop_positions_restarts_stranded_layouts():
    """GPU run 8: with per-prop-only retries, seed 6 strands the third cube in
    the demo cell's exact region. A stranded layout must be redrawn whole."""
    dims = {
        name: (0.06, 0.06, 0.06) for name in ("pick_cube", "ignore_cube_green", "ignore_cube_blue")
    }
    region = ((0.45, -0.25, 0.0), (0.7, 0.25, 0.0))
    for seed in range(50):
        placed = sample_prop_positions(dims, region, seed=seed, min_separation_m=0.2)
        assert set(placed) == set(dims)
        positions = list(placed.values())
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                gap = math.hypot(
                    positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]
                )
                assert gap >= 0.2


def test_six_block_cell_scatter_succeeds_at_the_measured_separation():
    """Six-block fragment packing envelope (phase-2 seam decision): 60 mm
    cubes in the scatter region at 140 mm separation place all six on every
    seed, matching the measured 100/100 success rate this default relies on."""
    dims = {
        name: (0.06, 0.06, 0.06)
        for name in (
            "pick_cube",
            "ignore_cube_green",
            "ignore_cube_blue",
            "ignore_cube_yellow",
            "ignore_cube_purple",
            "ignore_cube_orange",
        )
    }
    region = ((0.45, -0.25, 0.75), (0.70, 0.25, 0.75))
    successes = 0
    for seed in range(100):
        placed = sample_prop_positions(dims, region, seed=seed, min_separation_m=0.14)
        assert set(placed) == set(dims)
        successes += 1
    assert successes == 100
