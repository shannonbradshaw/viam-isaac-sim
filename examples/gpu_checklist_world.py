"""Phase-4 GPU checklist probes for the running world (R-27, OQ-9, OQ-10, OQ-15,
SCN-7): scene-randomization determinism, sim-time/wall-time step rate, runtime
spawn/teleport/reset without a Kit restart, and the two items phase 4 marks not
applicable.

Runs against either a live Viam machine or, with ``--mock``, an in-process mock
boot of the module - the item runners below only need something that duck-types
the world component's ``do_command`` (a ``Generic`` client in real mode, the
``IsaacWorld`` model itself in mock). Depends only on the stdlib and viam-sdk;
``isaac_module`` is imported lazily, only inside the ``--mock`` code path, same
as ``examples/pick_red_block.py``.

Usage (real machine)::

    python examples/gpu_checklist_world.py --address <machine-address> \\
        --api-key <key> --api-key-id <key-id> --world sim-world

Usage (in-process mock, no GPU, no running machine)::

    PYTHONPATH=src python examples/gpu_checklist_world.py --mock

Prints one report line per item plus the raw observations to paste into
``.claude/plans/pick-place-mvp/phase-4-*.md`` Notes. The pure/async helpers at
the top take a duck-typed world so they are unit-tested on a laptop against the
mock world (see tests/test_gpu_checklist_world.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

MM_PER_M = 1000.0

DEFAULT_WINDOW_S = 10.0
DEFAULT_PROP_NAME = "gpu_checklist_world_cube"
DEFAULT_CUBE_SIZE_M = 0.05
DEFAULT_SPAWN_POSITION_MM = (500.0, 0.0, 100.0)
DEFAULT_TELEPORT_POSITION_MM = (700.0, 200.0, 150.0)
# x/y scatter rectangle for item 1; the z here is only the no-props fallback -
# item1_scene_defaults replaces it with the live face height the blocks rest on
DEFAULT_RANDOMIZE_REGION_MM = ((300.0, -300.0, 50.0), (900.0, 300.0, 50.0))
DEFAULT_SEED_A = 1
DEFAULT_SEED_B = 2
PICK_COMMAND_TEMPLATE = (
    "python examples/pick_red_block.py --address <machine-address> "
    "--api-key <key> --api-key-id <key-id> --randomize-seed {seed}"
)

ITEM4_NOTE = "not applicable (DEC-5 cube table)"
ITEM5_NOTE = "not applicable (XC-9 deferred)"

# prim poses round-trip through float32; run 3 showed ~0.012 mm of noise
POSE_TOLERANCE_MM = 0.1
XY_TOLERANCE_MM = 0.5
TUMBLE_AXIS_TOLERANCE = 0.02
TUMBLE_THETA_TOLERANCE_DEG = 2.0


def resting_at_commanded(
    pose_mm: Mapping[str, float] | None, commanded_mm: Sequence[float]
) -> bool:
    """True when a prop sits where a pose command put it, allowing for the
    free fall a playing sim adds between the command and the read (GPU runs
    3-4): x/y at the commanded spot, z no higher than commanded, and no
    tumble (orientation still identity)."""
    if pose_mm is None:
        return False
    if abs(pose_mm["x"] - commanded_mm[0]) > XY_TOLERANCE_MM:
        return False
    if abs(pose_mm["y"] - commanded_mm[1]) > XY_TOLERANCE_MM:
        return False
    if pose_mm["z"] > commanded_mm[2] + XY_TOLERANCE_MM:
        return False
    if abs(pose_mm.get("o_x", 0.0)) > TUMBLE_AXIS_TOLERANCE:
        return False
    if abs(pose_mm.get("o_y", 0.0)) > TUMBLE_AXIS_TOLERANCE:
        return False
    return abs(pose_mm.get("theta", 0.0)) <= TUMBLE_THETA_TOLERANCE_DEG


class WorldApi(Protocol):
    async def do_command(self, command: Mapping[str, Any]) -> Mapping[str, Any]: ...


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------


def prop_pose_mm(
    geometries: Sequence[Mapping[str, Any]], name: str
) -> Mapping[str, float] | None:
    """The ``pose_in_world_mm`` entry for ``name`` out of a ``prop_geometries``
    response's ``geometries`` list, or None when it is not (yet) registered."""
    for geometry in geometries:
        if geometry.get("name") == name:
            return geometry["pose_in_world_mm"]
    return None


def pose_close(
    a: Mapping[str, float], b: Mapping[str, float], tolerance_mm: float = POSE_TOLERANCE_MM
) -> bool:
    return all(abs(a[axis] - b[axis]) <= tolerance_mm for axis in ("x", "y", "z"))


# ----------------------------------------------------------------------
# item 1 (R-27 half, drives examples/pick_red_block.py --randomize-seed)
# ----------------------------------------------------------------------


def item1_scene_defaults(
    geometries: Sequence[Mapping[str, Any]],
) -> tuple[list[str], tuple[list[float], list[float]]]:
    """Item 1 inputs derived from the live scene: every movable prop with a
    known box, scattered over DEFAULT_RANDOMIZE_REGION_MM's x/y rectangle at
    the height the movable props actually rest on (min over their box
    bottoms). The current fragment has no table, so that face is the floor."""
    movable = [g for g in geometries if not g["fixed"] and any(d > 0 for d in g["box_dims_mm"])]
    names = [g["name"] for g in movable]
    (x0, y0, z0), (x1, y1, z1) = DEFAULT_RANDOMIZE_REGION_MM
    if not names:
        return [], ([x0, y0, z0], [x1, y1, z1])
    face_z = min(g["pose_in_world_mm"]["z"] - g["box_dims_mm"][2] / 2.0 for g in movable)
    return names, ([x0, y0, face_z], [x1, y1, face_z])


async def run_item1(
    world: WorldApi,
    names: Sequence[str],
    region_mm: tuple[Sequence[float], Sequence[float]] = DEFAULT_RANDOMIZE_REGION_MM,
    seed_a: int = DEFAULT_SEED_A,
    seed_b: int = DEFAULT_SEED_B,
) -> dict[str, Any]:
    """Item 1's scene half: randomize twice with ``seed_a`` and assert the
    positions are identical, then once more with ``seed_b`` to report a
    different layout. ``prop_geometries`` is captured before and after so a
    human can see the props actually moved. The pick half - two consecutive
    picks against the re-randomized scene - is driven separately by
    ``examples/pick_red_block.py --randomize-seed``."""
    before = (await world.do_command({"command": "prop_geometries"}))["geometries"]

    def _randomize(seed: int) -> Mapping[str, Any]:
        return {
            "command": "randomize_props",
            "names": list(names),
            "region": [list(region_mm[0]), list(region_mm[1])],
            "seed": seed,
        }

    first = await world.do_command(_randomize(seed_a))
    second = await world.do_command(_randomize(seed_a))
    deterministic_same_seed = first["positions"] == second["positions"]

    different = await world.do_command(_randomize(seed_b))

    after = (await world.do_command({"command": "prop_geometries"}))["geometries"]

    return {
        "deterministic_same_seed": deterministic_same_seed,
        "positions_seed_a": first["positions"],
        "positions_seed_b": different["positions"],
        "prop_geometries_before": before,
        "prop_geometries_after": after,
        "pick_command": PICK_COMMAND_TEMPLATE.format(seed=seed_b),
    }


# ----------------------------------------------------------------------
# item 2 (OQ-10: step-rate measurement)
# ----------------------------------------------------------------------


async def _sample_step_rate(
    world: WorldApi, window_s: float, activity: Callable[[], Awaitable[None]] | None = None
) -> dict[str, Any]:
    """sim-time/wall-time ratio over one window. Runs ``activity`` (or just
    sleeps) while the window elapses so camera-grab cost shows up in the
    ratio. When the status has no ``sim_time`` (mock, or a booting real world)
    the wall time is still reported but the ratio is None with a note."""
    status_before = await world.do_command({"command": "status"})
    started_at = time.monotonic()
    if activity is not None:
        # keep the load on for the whole window, not one grab
        while time.monotonic() - started_at < window_s:
            await activity()
    else:
        await asyncio.sleep(window_s)
    wall_s = time.monotonic() - started_at
    status_after = await world.do_command({"command": "status"})

    if "sim_time" not in status_before or "sim_time" not in status_after:
        return {"wall_s": wall_s, "sim_time_ratio": None, "note": "mock: no sim_time"}
    sim_delta_s = float(status_after["sim_time"]) - float(status_before["sim_time"])
    ratio = sim_delta_s / wall_s if wall_s > 0 else None
    return {"wall_s": wall_s, "sim_delta_s": sim_delta_s, "sim_time_ratio": ratio}


async def run_item2(
    world: WorldApi,
    window_s: float = DEFAULT_WINDOW_S,
    camera_activity: Mapping[str, Callable[[], Awaitable[None]]] | None = None,
) -> dict[str, Any]:
    """OQ-10: baseline sim-time/wall-time ratio over ``window_s``, then the
    same measurement again once per entry in ``camera_activity`` (e.g. "rgb"
    grabbing frames from both cameras, "rgb_depth" grabbing depth too) so
    depth-on vs depth-off cost is visible. Reports numbers; decides nothing."""
    report: dict[str, Any] = {"baseline": await _sample_step_rate(world, window_s)}
    for label, activity in (camera_activity or {}).items():
        report[label] = await _sample_step_rate(world, window_s, activity)
    return report


# ----------------------------------------------------------------------
# item 3 (R-27: spawn_prop + set_prop_pose at runtime, no Kit restart)
# ----------------------------------------------------------------------


async def run_item3(
    world: WorldApi,
    prop_name: str = DEFAULT_PROP_NAME,
    spawn_position_mm: Sequence[float] = DEFAULT_SPAWN_POSITION_MM,
    teleport_position_mm: Sequence[float] = DEFAULT_TELEPORT_POSITION_MM,
    cube_size_m: float = DEFAULT_CUBE_SIZE_M,
) -> dict[str, Any]:
    """Spawn a uniquely-named cube via DoCommand, verify it appears in
    ``prop_geometries``, teleport it with ``set_prop_pose`` and verify the
    pose moved, then ``reset`` and verify it returned to its spawn pose -
    all without a Kit restart."""
    spawn_position_m = [value / MM_PER_M for value in spawn_position_mm]
    try:
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {
                    "name": prop_name,
                    "type": "cube",
                    "position": spawn_position_m,
                    "size": cube_size_m,
                },
            }
        )
        reused_existing = False
    except Exception as error:
        # a second checklist run against the same Kit session finds the prop
        # already in the scene; teleport/reset checks still mean the same
        if "already exists" not in str(error):
            raise
        reused_existing = True
    after_spawn = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    spawn_pose_mm = prop_pose_mm(after_spawn, prop_name)

    await world.do_command(
        {"command": "set_prop_pose", "name": prop_name, "position": list(teleport_position_mm)}
    )
    after_teleport = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    teleported_pose_mm = prop_pose_mm(after_teleport, prop_name)
    moved_after_teleport = resting_at_commanded(teleported_pose_mm, teleport_position_mm)

    await world.do_command({"command": "reset"})
    after_reset = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    post_reset_pose_mm = prop_pose_mm(after_reset, prop_name)
    restored_after_reset = resting_at_commanded(post_reset_pose_mm, spawn_position_mm)

    return {
        "reused_existing": reused_existing,
        "appeared_after_spawn": spawn_pose_mm is not None,
        "spawn_pose_mm": spawn_pose_mm,
        "teleported_pose_mm": teleported_pose_mm,
        "moved_after_teleport": moved_after_teleport,
        "post_reset_pose_mm": post_reset_pose_mm,
        "restored_after_reset": restored_after_reset,
    }


# ----------------------------------------------------------------------
# soft-reset demo (SCN-7 on real hardware)
# ----------------------------------------------------------------------


async def run_soft_reset_demo(
    world: WorldApi,
    prop_name: str = DEFAULT_PROP_NAME,
    teleport_position_mm: Sequence[float] = DEFAULT_TELEPORT_POSITION_MM,
    spawn_position_mm: Sequence[float] = DEFAULT_SPAWN_POSITION_MM,
) -> dict[str, Any]:
    """``set_prop_pose`` then ``reset {"soft": true}``. SCN-7: a soft reset is
    pose-only and restores the SPAWN pose - checked with resting_at_commanded,
    since a playing sim drops the cube between each command and its read. A
    paused sim is no alternative: pose writes are not visible to reads until
    a step happens (GPU run 4)."""

    async def _current_pose() -> Mapping[str, float] | None:
        geometries = (await world.do_command({"command": "prop_geometries"}))["geometries"]
        return prop_pose_mm(geometries, prop_name)

    pre_teleport_pose_mm = await _current_pose()
    await world.do_command(
        {"command": "set_prop_pose", "name": prop_name, "position": list(teleport_position_mm)}
    )
    teleported_pose_mm = await _current_pose()
    await world.do_command({"command": "reset", "soft": True})
    restored_pose_mm = await _current_pose()

    spawn_pose_mm = {
        "x": float(spawn_position_mm[0]),
        "y": float(spawn_position_mm[1]),
        "z": float(spawn_position_mm[2]),
    }
    return {
        "pre_teleport_pose_mm": pre_teleport_pose_mm,
        "teleported_at_commanded": resting_at_commanded(teleported_pose_mm, teleport_position_mm),
        "teleported_pose_mm": teleported_pose_mm,
        "restored_pose_mm": restored_pose_mm,
        "spawn_pose_mm": spawn_pose_mm,
        "restored_matches_spawn": resting_at_commanded(restored_pose_mm, spawn_position_mm),
    }


# ----------------------------------------------------------------------
# item 4 / item 5 (not applicable this phase)
# ----------------------------------------------------------------------


def run_item4() -> str:
    """OQ-9 mesh-table check: not applicable under DEC-5 (cube table)."""
    return ITEM4_NOTE


def run_item5() -> str:
    """OQ-15 XC-9 executor check: XC-9 was deferred this phase."""
    return ITEM5_NOTE


# ----------------------------------------------------------------------
# real/mock connection + CLI
# ----------------------------------------------------------------------


async def _camera_frame_activity(camera: Any, *, depth: bool) -> None:
    # get_images is the API the module serves (and phase 2 verified on the GPU);
    # CameraClient has no get_image in viam-sdk 0.80
    if depth:
        await camera.get_images()
    else:
        await camera.get_images(filter_source_names=["color"])


def _ensure_mock_sim_booted() -> Any:
    import threading

    from isaac_module.sim_manager import SimConfig, SimManager

    manager = SimManager.get()
    if not manager._booted.is_set():
        sim_thread = threading.Thread(target=manager.main_loop, daemon=True)
        sim_thread.start()
        manager.ensure_booted(SimConfig(mock=True))
    return manager


async def _run(world: WorldApi, camera_activity: Mapping[str, Callable[[], Awaitable[None]]]) -> None:
    print("item 1 (R-27 scene half - determinism, prop_geometries before/after):")
    scene = (await world.do_command({"command": "prop_geometries"}))["geometries"]
    item1_names, item1_region = item1_scene_defaults(scene)
    if item1_names:
        item1 = await run_item1(world, names=item1_names, region_mm=item1_region)
        print(f"  {json.dumps(item1, default=str, sort_keys=True)}")
    else:
        print("  skipped: no movable props in the scene (configure at least one non-fixed prop)")

    print(f"item 2 (OQ-10 step-rate over {DEFAULT_WINDOW_S:.0f} s windows):")
    item2 = await run_item2(world, camera_activity=camera_activity)
    print(f"  {json.dumps(item2, default=str, sort_keys=True)}")

    print("item 3 (R-27 spawn_prop/set_prop_pose/reset at runtime):")
    item3 = await run_item3(world)
    print(f"  {json.dumps(item3, default=str, sort_keys=True)}")

    print(f"item 4 (OQ-9 mesh table): {run_item4()}")
    print(f"item 5 (OQ-15 XC-9 executor): {run_item5()}")

    print("soft-reset demo (SCN-7):")
    soft_reset = await run_soft_reset_demo(world)
    print(f"  {json.dumps(soft_reset, default=str, sort_keys=True)}")


async def _run_real(args: argparse.Namespace) -> None:
    from viam.components.camera import Camera
    from viam.components.generic import Generic
    from viam.robot.client import RobotClient

    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()
    robot = await RobotClient.at_address(args.address, opts)
    try:
        world = Generic.from_robot(robot, args.world)
        camera_activity: dict[str, Callable[[], Awaitable[None]]] = {}
        if args.camera:
            camera = Camera.from_robot(robot, args.camera)
            camera_activity["rgb"] = lambda: _camera_frame_activity(camera, depth=False)
            camera_activity["rgb_depth"] = lambda: _camera_frame_activity(camera, depth=True)
        await _run(world, camera_activity)
    finally:
        await robot.close()


async def _run_mock(args: argparse.Namespace) -> None:
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from isaac_module.models.world import IsaacWorld

    def config(name: str, attrs: dict[str, Any]) -> ComponentConfig:
        return ComponentConfig(name=name, attributes=dict_to_struct(attrs))

    _ensure_mock_sim_booted()
    world = IsaacWorld.new(config("gpu-checklist-phase4-world", {"mock": True}), {})
    # the mock world boots without props; give item 1 two movable blocks
    for name, y_m in (("mock_block_a", 0.1), ("mock_block_b", -0.1)):
        await world.do_command(
            {
                "command": "spawn_prop",
                "prop": {"name": name, "position": [0.6, y_m, 0.025], "size": 0.05},
            }
        )
    from isaac_module.models.camera import IsaacCamera

    camera = IsaacCamera.new(
        config(
            "gpu-checklist-phase4-cam",
            {"world": "gpu-checklist-phase4-world", "width": 320, "height": 240, "depth": True},
        ),
        {},
    )
    await _run(
        world,
        camera_activity={
            "rgb": lambda: _camera_frame_activity(camera, depth=False),
            "rgb_depth": lambda: _camera_frame_activity(camera, depth=True),
        },
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mock", action="store_true", help="run in-process against the module's mock backend"
    )
    parser.add_argument("--address", help="machine address (required unless --mock)")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--world", default="sim-world", help="the isaac-sim world component name")
    parser.add_argument(
        "--camera",
        default="wrist-cam",
        help="the camera item 2 grabs RGB/depth load from - the fragment's wrist-cam"
        " is the depth-enabled one (scene-cam is RGB-only); pass another name for"
        " other cells, or an empty string to skip camera load",
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
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
