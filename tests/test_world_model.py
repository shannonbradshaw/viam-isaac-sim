"""SCN-16: every DoCommand verb on the world component routes through
WorldHandle, never SimManager scene methods directly; plus the new
scene verbs and live get_geometries."""

import asyncio
import itertools
import logging

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.world import IsaacWorld
from isaac_module.sim_manager import DEFAULT_MIN_SEPARATION_M, RandomizeResult, SimManager

MIN_SEPARATION_MM = DEFAULT_MIN_SEPARATION_M * 1000.0


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


class _RecordingWorldHandle:
    """Duck-types WorldHandle and records every call it receives, so the
    every-verb test can assert do_command reaches it and nothing else."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def status(self) -> dict:
        self.calls.append(("status",))
        return {"recorded": True}

    def play(self) -> None:
        self.calls.append(("play",))

    def pause(self) -> None:
        self.calls.append(("pause",))

    def reset(self, soft: bool = False) -> None:
        self.calls.append(("reset", soft))

    def add_usd(self, usd_path, prim_path, position_m, orientation_wxyz=None) -> None:
        self.calls.append(("add_usd", usd_path, prim_path, position_m, orientation_wxyz))

    def prop_geometries(self) -> list:
        self.calls.append(("prop_geometries",))
        return []

    def spawn_prop(self, prop) -> None:
        self.calls.append(("spawn_prop", prop))

    def set_prop_pose(self, name, position_m, orientation_wxyz=None) -> None:
        self.calls.append(("set_prop_pose", name, position_m, orientation_wxyz))

    def randomize_props(
        self,
        names,
        region,
        seed,
        min_separation_m=DEFAULT_MIN_SEPARATION_M,
        size_range_m=None,
    ):
        self.calls.append(("randomize_props", names, region, seed, min_separation_m, size_range_m))
        return RandomizeResult(
            positions_m={name: (0.0, 0.0, 0.0) for name in names},
            dims_m={name: (0.05, 0.05, 0.05) for name in names},
        )


def test_every_verb_routes_through_handle(world, monkeypatch):
    fake = _RecordingWorldHandle()
    monkeypatch.setattr(SimManager, "world_handle", lambda self: fake)

    def _boom(*args, **kwargs):
        raise AssertionError("do_command must not call SimManager scene methods directly")

    monkeypatch.setattr(SimManager, "play", _boom)
    monkeypatch.setattr(SimManager, "pause", _boom)
    monkeypatch.setattr(SimManager, "reset", _boom)
    monkeypatch.setattr(SimManager, "status", _boom)
    monkeypatch.setattr(SimManager, "add_usd_reference", _boom)

    async def scenario():
        await world.do_command({"command": "status"})
        await world.do_command({"command": "play"})
        await world.do_command({"command": "pause"})
        await world.do_command({"command": "reset"})
        await world.do_command({"command": "reset", "soft": True})
        await world.do_command(
            {
                "command": "add_usd",
                "usd_path": "a.usd",
                "prim_path": "/World/a",
                "position": [1.0, 2.0, 3.0],
            }
        )
        await world.do_command({"command": "prop_geometries"})
        await world.do_command({"command": "spawn_prop", "prop": {"name": "verb_test_prop"}})
        await world.do_command(
            {"command": "set_prop_pose", "name": "verb_test_prop", "position": [10.0, 20.0, 30.0]}
        )
        await world.do_command(
            {
                "command": "randomize_props",
                "names": ["verb_test_prop"],
                "region": [[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]],
                "seed": 1,
            }
        )

    asyncio.run(scenario())

    verbs = [call[0] for call in fake.calls]
    assert verbs == [
        "status",
        "play",
        "pause",
        "reset",
        "reset",
        "add_usd",
        "prop_geometries",
        "spawn_prop",
        "set_prop_pose",
        "randomize_props",
    ]
    assert fake.calls[3] == ("reset", False)
    assert fake.calls[4] == ("reset", True)


def test_spawn_prop_and_prop_geometries_round_trip(world):
    prop = {
        "name": "rt_prop",
        "type": "cube",
        "position": [0.1, 0.2, 0.3],
        "size": 0.05,
        "scale": [1.0, 2.0, 3.0],
    }

    async def scenario():
        await world.do_command({"command": "spawn_prop", "prop": prop})
        return await world.do_command({"command": "prop_geometries"})

    result = asyncio.run(scenario())
    entry = next(g for g in result["geometries"] if g["name"] == "rt_prop")
    pose = entry["pose_in_world_mm"]
    assert pose["x"] == pytest.approx(100.0)
    assert pose["y"] == pytest.approx(200.0)
    assert pose["z"] == pytest.approx(300.0)
    assert pose["theta"] == pytest.approx(0.0, abs=1e-6)
    assert entry["box_dims_mm"][0] == pytest.approx(50.0)
    assert entry["box_dims_mm"][1] == pytest.approx(100.0)
    assert entry["box_dims_mm"][2] == pytest.approx(150.0)


def test_spawn_prop_with_orientation_reports_rotated_pose(world):
    async def scenario():
        await world.do_command(
            {"command": "spawn_prop", "prop": {"name": "rot_base", "position": [0.0, 0.0, 0.0]}}
        )
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {
                    "name": "rot_test",
                    "position": [0.0, 0.0, 0.0],
                    "orientation_rpy_deg": [0.0, 90.0, 0.0],
                },
            }
        )
        return await world.do_command({"command": "prop_geometries"})

    result = asyncio.run(scenario())
    geoms = {g["name"]: g["pose_in_world_mm"] for g in result["geometries"]}
    base, rotated = geoms["rot_base"], geoms["rot_test"]
    changed = (
        rotated["o_x"] != pytest.approx(base["o_x"])
        or rotated["o_y"] != pytest.approx(base["o_y"])
        or rotated["theta"] != pytest.approx(base["theta"])
    )
    assert changed


def test_set_prop_pose_then_reset_restores_configured_pose(world):
    async def scenario():
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {"name": "reset_prop", "position": [0.1, 0.1, 0.1]},
            }
        )
        await world.do_command(
            {
                "command": "set_prop_pose",
                "name": "reset_prop",
                "position": [500.0, 500.0, 500.0],
            }
        )
        moved = await world.do_command({"command": "prop_geometries"})
        await world.do_command({"command": "reset"})
        after = await world.do_command({"command": "prop_geometries"})
        return moved, after

    moved, after = asyncio.run(scenario())
    moved_entry = next(g for g in moved["geometries"] if g["name"] == "reset_prop")
    after_entry = next(g for g in after["geometries"] if g["name"] == "reset_prop")
    assert moved_entry["pose_in_world_mm"]["x"] == pytest.approx(500.0)
    assert after_entry["pose_in_world_mm"]["x"] == pytest.approx(100.0)


def test_randomize_props_deterministic_and_within_region(world):
    names = ["rand_a", "rand_b", "rand_c"]
    region = [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]]

    async def scenario():
        for name in names:
            await world.do_command(
                {"command": "spawn_prop", "prop": {"name": name, "position": [0.0, 0.0, 0.0]}}
            )
        first = await world.do_command(
            {"command": "randomize_props", "names": names, "region": region, "seed": 1}
        )
        second = await world.do_command(
            {"command": "randomize_props", "names": names, "region": region, "seed": 1}
        )
        return first, second

    first, second = asyncio.run(scenario())
    assert first == second
    positions = first["positions"]
    for name in names:
        x, y, _z = positions[name]
        assert 0.0 <= x <= 1000.0
        assert 0.0 <= y <= 1000.0
    for name_a, name_b in itertools.combinations(names, 2):
        ax, ay, _ = positions[name_a]
        bx, by, _ = positions[name_b]
        distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        assert distance >= MIN_SEPARATION_MM - 1e-6

    sizes = first["sizes_mm"]
    for name in names:
        assert sizes[name] == pytest.approx([50.0, 50.0, 50.0])  # unranged: default cube size


def test_randomize_props_size_range_mm_list_form_reaches_the_handle_in_metres(world):
    names = ["sz_a", "sz_b"]
    region = [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]]

    async def scenario():
        for name in names:
            await world.do_command({"command": "spawn_prop", "prop": {"name": name}})
        return await world.do_command(
            {
                "command": "randomize_props",
                "names": names,
                "region": region,
                "seed": 1,
                "size_range_mm": [30.0, 90.0],
            }
        )

    result = asyncio.run(scenario())
    for name in names:
        x, y, z = result["sizes_mm"][name]
        assert 30.0 <= x <= 90.0
        assert x == y == z


def test_randomize_props_size_range_mm_map_form_reaches_the_handle_in_metres(world):
    names = ["sz_c", "sz_d"]
    region = [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]]

    async def scenario():
        for name in names:
            await world.do_command({"command": "spawn_prop", "prop": {"name": name}})
        return await world.do_command(
            {
                "command": "randomize_props",
                "names": names,
                "region": region,
                "seed": 1,
                "size_range_mm": {"sz_c": [30.0, 90.0]},
            }
        )

    result = asyncio.run(scenario())
    x, y, z = result["sizes_mm"]["sz_c"]
    assert 30.0 <= x <= 90.0
    assert x == y == z
    assert result["sizes_mm"]["sz_d"] == pytest.approx([50.0, 50.0, 50.0])


def test_randomize_props_size_range_mm_validation_errors(world):
    async def randomize(size_range_mm):
        return await world.do_command(
            {
                "command": "randomize_props",
                "names": ["sz_bad"],
                "region": [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]],
                "seed": 1,
                "size_range_mm": size_range_mm,
            }
        )

    asyncio.run(world.do_command({"command": "spawn_prop", "prop": {"name": "sz_bad"}}))

    with pytest.raises(ValueError):
        asyncio.run(randomize([0.0, 90.0]))  # lo not > 0
    with pytest.raises(ValueError):
        asyncio.run(randomize([90.0, 30.0]))  # lo > hi
    with pytest.raises(ValueError):
        asyncio.run(randomize([30.0]))  # wrong arity
    with pytest.raises(ValueError):
        asyncio.run(randomize(["a", 90.0]))  # non-number entry
    with pytest.raises(ValueError):
        asyncio.run(randomize({"not_sz_bad": [30.0, 90.0]}))  # key not in names


def test_ignore_props_and_get_geometries(world):
    async def scenario():
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {"name": "geo_a", "position": [0.1, 0.2, 0.3], "size": 0.05},
            }
        )
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {
                    "name": "geo_b",
                    "type": "usd",
                    "usd_path": "x.usd",
                    "position": [0.0, 0.0, 0.0],
                },
            }
        )
        await world.do_command({"command": "ignore_props", "names": ["geo_a"]})
        while_ignored = await world.get_geometries()
        await world.do_command({"command": "ignore_props", "names": []})
        after_clear = await world.get_geometries()
        return while_ignored, after_clear

    while_ignored, after_clear = asyncio.run(scenario())

    labels_while_ignored = {g.label for g in while_ignored}
    assert "geo_a" not in labels_while_ignored
    assert "geo_b" not in labels_while_ignored  # zero (unknown) dims stays excluded

    labels_after_clear = {g.label for g in after_clear}
    assert "geo_a" in labels_after_clear
    assert "geo_b" not in labels_after_clear

    entry = next(g for g in after_clear if g.label == "geo_a")
    assert entry.center.x == pytest.approx(100.0)
    assert entry.box.dims_mm.x == pytest.approx(50.0)


def test_spawn_prop_validation_errors(world):
    async def spawn(prop):
        return await world.do_command({"command": "spawn_prop", "prop": prop})

    with pytest.raises(ValueError, match="orientation_rpy_deg"):
        asyncio.run(spawn({"name": "bad_1", "orientation_rpy_deg": [1.0, 2.0]}))

    with pytest.raises(ValueError, match="only one of"):
        asyncio.run(
            spawn(
                {
                    "name": "bad_2",
                    "orientation_rpy_deg": [1.0, 2.0, 3.0],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            )
        )

    with pytest.raises(ValueError, match="box_dims"):
        asyncio.run(spawn({"name": "bad_3", "box_dims": [1.0, -1.0, 1.0]}))

    with pytest.raises(ValueError, match="prop"):
        asyncio.run(world.do_command({"command": "spawn_prop"}))


def test_usd_stage_without_lighting_warns(caplog):
    config = _config("sim-world-warn", {"mock": True, "usd_stage": "foo.usd"})
    with caplog.at_level(logging.WARNING):
        IsaacWorld.new(config, {})
    assert any("stage must provide floor and lights" in record.message for record in caplog.records)


def test_oracle_commands_false_hides_ground_truth_but_keeps_operation(sim):
    world = IsaacWorld.new(_config("sim-world-gated", {"mock": True, "oracle_commands": False}), {})
    for cmd in ("prop_geometries", "set_prop_pose", "randomize_props", "spawn_prop"):
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(world.do_command({"command": cmd}))
        assert "oracle_commands" in str(exc_info.value)
        assert "cameras" in str(exc_info.value)
    # Operating the world is unaffected.
    assert asyncio.run(world.do_command({"command": "status"}))
    assert asyncio.run(world.do_command({"command": "reset"})) == {"ok": True}
    assert asyncio.run(world.do_command({"command": "ignore_props", "names": []})) == {"ignored": []}


def test_oracle_commands_default_on_and_validated(world):
    assert asyncio.run(world.do_command({"command": "prop_geometries"}))
    with pytest.raises(ValueError):
        IsaacWorld.validate_config(_config("bad", {"mock": True, "oracle_commands": "no"}))


def test_unknown_command_lists_verbs(world):
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(world.do_command({"command": "bogus"}))
    message = str(exc_info.value)
    for verb in (
        "status",
        "play",
        "pause",
        "reset",
        "add_usd",
        "prop_geometries",
        "spawn_prop",
        "set_prop_pose",
        "randomize_props",
        "ignore_props",
    ):
        assert verb in message


def test_get_geometries_serves_the_floor_when_the_module_owns_the_stage(world):
    async def scenario():
        await world.do_command({"command": "ignore_props", "names": []})
        return await world.get_geometries()

    geometries = asyncio.run(scenario())
    floor = next(g for g in geometries if g.label == "floor")
    assert floor.box.dims_mm.z == 200.0
    assert floor.center.z == -100.0  # top face exactly at z = 0

    async def hide_floor():
        await world.do_command({"command": "ignore_props", "names": ["floor"]})
        try:
            return await world.get_geometries()
        finally:
            await world.do_command({"command": "ignore_props", "names": []})

    assert all(g.label != "floor" for g in asyncio.run(hide_floor()))


def test_get_geometries_serves_no_floor_over_a_user_stage():
    from isaac_module.models.world import IsaacWorld

    stage_world = IsaacWorld.new(
        _config("stage-world", {"mock": True, "usd_stage": "user_stage.usd"}), {}
    )

    async def scenario():
        return await stage_world.get_geometries()

    assert all(g.label != "floor" for g in asyncio.run(scenario()))
