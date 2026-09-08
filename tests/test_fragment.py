"""The shipped fragment must validate against the current models — it is the
config a fresh machine is built from, so a validator change that breaks it
has to fail here, not on the machine."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module import physics
from isaac_module.models.arm import IsaacArm
from isaac_module.models.base import IsaacBase
from isaac_module.models.camera import IsaacCamera
from isaac_module.models.gripper import IsaacGripper
from isaac_module.models.world import IsaacWorld

FRAGMENT_PATH = Path(__file__).resolve().parent.parent / "fragments" / "pick-and-place.json"
MODELS = {
    "world": IsaacWorld,
    "arm": IsaacArm,
    "camera": IsaacCamera,
    "base": IsaacBase,
    "gripper": IsaacGripper,
}
API_PATTERN = re.compile(r"^rdk:component:[a-z_]+$")

# Seam — P5 canonical cell: the nine `$variable`s the fragment ships, keyed by
# name, with the default_value a fresh machine that sets nothing must boot.
EXPECTED_VARIABLE_DEFAULTS: dict[str, Any] = {
    "table-height-m": 0.75,
    "pick-block-color": [0.9, 0.1, 0.1],
    "distractor-color-green": [0.05, 0.65, 0.1],
    "distractor-color-blue": [0.05, 0.1, 0.9],
    "distractor-color-yellow": [0.9, 0.75, 0.05],
    "distractor-color-purple": [0.55, 0.1, 0.75],
    "distractor-color-orange": [1.0, 0.55, 0.05],
    "detect-color": "#EA8D8D",
    "hue-tolerance-pct": 0.05,
}


def _fragment() -> dict:
    return json.loads(FRAGMENT_PATH.read_text())


def _resolve_variables(node: Any) -> Any:
    """Mimic the app-side `$variable` substitution
    (`fragment_variable_substitution.go`): replace every `{"$variable":
    {"name", "default_value"}}` object with its `default_value`, recursing
    into arrays so a variable inside a `color`/`scale` array resolves too."""
    if isinstance(node, dict):
        variable = node.get("$variable")
        if variable is not None and set(node.keys()) == {"$variable"}:
            return _resolve_variables(variable["default_value"])
        return {key: _resolve_variables(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_variables(item) for item in node]
    return node


def _resolved_fragment() -> dict:
    return _resolve_variables(_fragment())


def _collect_variables(node: Any, found: dict[str, Any]) -> None:
    if isinstance(node, dict):
        variable = node.get("$variable")
        if variable is not None and set(node.keys()) == {"$variable"}:
            found[variable["name"]] = variable.get("default_value")
            return
        for value in node.values():
            _collect_variables(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_variables(item, found)


def _component_config(component: dict) -> ComponentConfig:
    """Build the proto viam-server hands the module from the fragment's JSON
    (only the frame shapes the fragment actually uses are translated)."""
    config = ComponentConfig(
        name=component["name"], attributes=dict_to_struct(component["attributes"])
    )
    frame = component.get("frame")
    if frame is None:
        return config
    config.frame.parent = frame.get("parent", "world")
    translation = frame.get("translation", {})
    config.frame.translation.x = translation.get("x", 0)
    config.frame.translation.y = translation.get("y", 0)
    config.frame.translation.z = translation.get("z", 0)
    orientation = frame.get("orientation")
    if orientation is not None:
        if orientation["type"] != "ov_degrees":
            raise AssertionError(
                f"extend _component_config for orientation type {orientation['type']}"
            )
        value = orientation["value"]
        vector = config.frame.orientation.vector_degrees
        vector.x, vector.y, vector.z, vector.theta = value["x"], value["y"], value["z"], value["th"]
    return config


def test_fragment_is_valid_json_with_the_expected_components():
    names = [c["name"] for c in _fragment()["components"]]
    assert names == ["sim-world", "pick-arm", "pick-grip", "scene-cam", "side-cam", "wrist-cam"]


def test_every_component_uses_the_api_form_not_the_legacy_namespace_type_pair():
    for component in _fragment()["components"]:
        assert "namespace" not in component
        assert "type" not in component
        assert API_PATTERN.match(component["api"])


def test_every_non_world_component_names_sim_world_in_its_attributes():
    for component in _fragment()["components"]:
        if component["name"] == "sim-world":
            continue
        assert component["attributes"]["world"] == "sim-world"


@pytest.mark.parametrize("component", _resolved_fragment()["components"], ids=lambda c: c["name"])
def test_every_fragment_component_validates_against_its_model(component):
    model = MODELS[component["model"].rsplit(":", 1)[1]]
    dependencies, _ = model.validate_config(_component_config(component))
    # components riding the arm (the gripper, the wrist camera's parent_prim)
    # must depend on it so viam-server builds the arm's prim first
    if component["name"] in ("pick-grip", "wrist-cam"):
        assert list(dependencies) == ["sim-world", "pick-arm"]
    elif component["name"] != "sim-world":
        assert list(dependencies) == ["sim-world"]


def test_gripper_frame_matches_its_tcp_offset():
    gripper = next(c for c in _fragment()["components"] if c["name"] == "pick-grip")
    assert gripper["frame"]["parent"] == "pick-arm"
    z = gripper["frame"]["translation"]["z"]
    assert z == 134
    default_tcp_offset_m = 0.134
    assert default_tcp_offset_m * 1000 == z


def test_arm_frame_matches_w6():
    arm = next(c for c in _fragment()["components"] if c["name"] == "pick-arm")
    assert arm["frame"] == {
        "parent": "world",
        "translation": {"x": 150, "y": -250, "z": 750},
    }


def test_pick_cube_physics_matches_the_named_pick_cell_constant():
    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    pick_cube = next(p for p in world["attributes"]["props"] if p["name"] == "pick_cube")
    physics_keys = {k: pick_cube[k] for k in physics.PICK_CELL_BLOCK_PHYSICS}
    assert physics_keys == physics.PICK_CELL_BLOCK_PHYSICS


def test_world_step_rates_match_the_pick_cell_constants():
    world = next(c for c in _fragment()["components"] if c["name"] == "sim-world")
    attrs = world["attributes"]
    assert attrs["physics_dt"] == pytest.approx(physics.PICK_CELL_PHYSICS_DT)
    assert attrs["rendering_dt"] == pytest.approx(physics.PICK_CELL_RENDERING_DT)


def test_wrist_camera_matches_the_phase_2_contract():
    wrist = next(c for c in _fragment()["components"] if c["name"] == "wrist-cam")
    assert wrist["attributes"]["depth"] is True
    assert wrist["frame"]["parent"] == "pick-arm"
    assert wrist["attributes"]["parent_prim"] == "/World/pick_arm/wrist_3_link"
    assert "local_position" not in wrist["attributes"]  # the frame is the mount's source of truth


def test_wrist_and_side_cameras_carry_a_collision_geometry_for_the_planner():
    """Both cameras ride within reach of the planner's swept volume, so each
    carries its RealSense body (90x25x25 mm, long axis along the frame's x,
    centred on the frame origin) as frame geometry with no translation.
    A box, not the tools/generate_realsense_mesh.py mesh: the app's fragment
    validation rejected mesh geometries on 2026-09-02 (phase-4 §Deferred)."""
    for name in ("wrist-cam", "side-cam"):
        component = next(c for c in _fragment()["components"] if c["name"] == name)
        geometry = component["frame"]["geometry"]
        assert geometry["type"] == "box"
        assert geometry["x"] == 90
        assert geometry["y"] == 25
        assert geometry["z"] == 25
        assert "translation" not in geometry


def test_side_cam_body_is_a_fixed_prop_facing_away_from_its_own_camera():
    """The rendered RealSense housing must be fixed (it's bolted, not
    scattered) and its front face must sit behind the side-cam viewpoint
    (y 0.65 m) so the camera never sees the inside of its own body."""
    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    body = next(p for p in world["attributes"]["props"] if p["name"] == "side_cam_body")
    assert body["fixed"] is True
    y_position = body["position"][1]
    y_half_extent = body["size"] * body["scale"][1] / 2
    front_face_y = y_position - y_half_extent
    assert front_face_y >= 0.65


def test_side_camera_sits_outside_the_scatter_region_and_aims_at_its_centre():
    """Phase 4 seam: `side-cam` is planted just past the scatter region's +y
    edge (y 250 mm) at lens height ~150 mm above the table top, below
    1000 mm so it stays tabletop-adjacent. It aims via a FRAME orientation,
    not `target`: the frame is what transform_pose reports, so prim aim and
    frame claim must be one quaternion (GPU phase-4 run 1: `target` aimed the
    prim while the frame claimed identity, and side scans measured the
    backdrop at 7994 mm). The orientation vector must point from the lens to
    the scatter-region centre on the table top."""
    side = next(c for c in _fragment()["components"] if c["name"] == "side-cam")
    assert side["frame"]["parent"] == "world"
    translation = side["frame"]["translation"]
    assert translation["y"] > 250
    assert translation["y"] < 1000
    assert side["attributes"]["depth"] is True
    assert "target" not in side["attributes"]

    orientation = side["frame"]["orientation"]
    assert orientation["type"] == "ov_degrees"
    vector = orientation["value"]
    region_centre_mm = (575.0, 0.0, 750.0)
    aim = [
        region_centre_mm[i] - (translation["x"], translation["y"], translation["z"])[i]
        for i in range(3)
    ]
    ov = [vector["x"], vector["y"], vector["z"]]
    cross = [
        aim[1] * ov[2] - aim[2] * ov[1],
        aim[2] * ov[0] - aim[0] * ov[2],
        aim[0] * ov[1] - aim[1] * ov[0],
    ]
    assert all(abs(c) < 1e-9 for c in cross)  # parallel to the lens->centre ray
    assert sum(a * o for a, o in zip(aim, ov, strict=True)) > 0  # and not flipped away from it


def test_sim_world_is_in_the_frame_system():
    """DEC-21 route (c): the motion service only pulls sim-world's live
    GetGeometries (props + floor) into planning when the component has a
    frame (GPU run 7: without it, app-side moves plan with no obstacles)."""
    world = next(c for c in _fragment()["components"] if c["name"] == "sim-world")
    assert world.get("frame", {}).get("parent") == "world"


def test_sim_world_frame_has_no_translation():
    """Live GetGeometries are expressed in sim-world's own frame (DEC-21 route
    (c)); a frame translation would offset every served geometry."""
    world = next(c for c in _fragment()["components"] if c["name"] == "sim-world")
    assert "translation" not in world["frame"]


def test_sim_world_geometry_matches_the_table_prop_minus_ten_millimetres():
    """W4: the planner box rides inside sim-world's frame.geometry, sized to
    the table prop's x/y footprint with its top 10 mm below the real
    surface (R-24), centred so the box top sits at that height."""
    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    table = next(p for p in world["attributes"]["props"] if p["name"] == "table")
    table_x_mm = table["size"] * table["scale"][0] * 1000
    table_y_mm = table["size"] * table["scale"][1] * 1000
    table_top_mm = (table["position"][2] + table["size"] * table["scale"][2] / 2) * 1000

    geometry = world["frame"]["geometry"]
    assert geometry["type"] == "box"
    assert geometry["x"] == pytest.approx(table_x_mm)
    assert geometry["y"] == pytest.approx(table_y_mm)
    assert geometry["z"] == pytest.approx(table_top_mm - 10)

    translation = geometry["translation"]
    assert translation["z"] + geometry["z"] / 2 == pytest.approx(table_top_mm - 10)


def test_six_blocks_follow_the_layout_rules():
    """W23-W26 via the DEC-20 naming: one red target plus five colour-distinct
    distractors, all movable, spawned >= 0.20 m apart (W26), inside the
    verified 743 mm pick radius measured from the arm base (150, -250 mm),
    and clear of the place_pad footprint."""
    import itertools
    import math

    world = next(c for c in _resolved_fragment()["components"] if c["name"] == "sim-world")
    props = {p["name"]: p for p in world["attributes"]["props"]}
    blocks = [
        "pick_cube",
        "ignore_cube_green",
        "ignore_cube_blue",
        "ignore_cube_yellow",
        "ignore_cube_purple",
        "ignore_cube_orange",
    ]
    assert all(name in props for name in blocks)

    dominant_channel_blocks = ["pick_cube", "ignore_cube_green", "ignore_cube_blue"]
    dominant_channels = [
        max(range(3), key=lambda i: props[b]["color"][i]) for b in dominant_channel_blocks
    ]
    assert dominant_channels == [0, 1, 2]  # red target, green and blue distractors

    arm_base_x, arm_base_y = 0.150, -0.250
    block_half_size = 0.03
    place_pad_x_range = (0.2, 0.4)
    place_pad_y_range = (0.15, 0.35)
    for name in blocks:
        assert not props[name].get("fixed", False)
        x, y, _z = props[name]["position"]
        assert math.hypot(x - arm_base_x, y - arm_base_y) <= 0.743
        clear_of_pad_x = (
            x + block_half_size < place_pad_x_range[0] or x - block_half_size > place_pad_x_range[1]
        )
        clear_of_pad_y = (
            y + block_half_size < place_pad_y_range[0] or y - block_half_size > place_pad_y_range[1]
        )
        assert clear_of_pad_x or clear_of_pad_y

    for a, b in itertools.combinations(blocks, 2):
        ax, ay, _az = props[a]["position"]
        bx, by, _bz = props[b]["position"]
        assert math.hypot(ax - bx, ay - by) >= 0.20


def test_the_pick_cell_roster_is_present():
    fragment = _fragment()
    world = next(c for c in fragment["components"] if c["name"] == "sim-world")
    props = {p["name"] for p in world["attributes"]["props"]}
    assert props == {
        "table",
        "pick_cube",
        "ignore_cube_green",
        "ignore_cube_blue",
        "ignore_cube_yellow",
        "ignore_cube_purple",
        "ignore_cube_orange",
        "place_pad",
        "side_cam_body",
    }

    component_names = {c["name"] for c in fragment["components"]}
    assert component_names == {
        "sim-world",
        "pick-arm",
        "pick-grip",
        "wrist-cam",
        "scene-cam",
        "side-cam",
    }

    service_names = {(s["name"], s["api"]) for s in fragment["services"]}
    # RDK serves the builtin motion service implicitly, so the fragment may
    # carry the entry or omit it - either way the pick client's "builtin"
    # motion resource resolves.
    service_names.discard(("builtin", "rdk:service:motion"))
    assert service_names == {
        ("red-detector", "rdk:service:vision"),
        ("block-segmenter", "rdk:service:vision"),
    }


def test_the_nine_variables_ship_with_the_seam_default_values():
    found: dict[str, Any] = {}
    _collect_variables(_fragment(), found)
    assert found == EXPECTED_VARIABLE_DEFAULTS


def _hue_degrees(color: list[float]) -> float:
    import colorsys

    r, g, b = color
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360


def _hex_to_hue_degrees(hex_color: str) -> float:
    import colorsys

    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360


def test_the_red_detector_band_admits_only_the_pick_cube_hue():
    """The detector must fire on the red target and stay clear of every
    distractor by margin, with each distractor still inside its intended
    colour family so the margin cannot be gamed by drifting a hue."""
    fragment = _resolved_fragment()
    world = next(c for c in fragment["components"] if c["name"] == "sim-world")
    props = {p["name"]: p for p in world["attributes"]["props"]}
    detector = next(s for s in fragment["services"] if s["name"] == "red-detector")

    detect_hue = _hex_to_hue_degrees(detector["attributes"]["detect_color"])
    hue_tolerance_pct = detector["attributes"]["hue_tolerance_pct"]
    band_half_width = hue_tolerance_pct * 360

    def hue_distance(hue: float) -> float:
        diff = abs(hue - detect_hue) % 360
        return min(diff, 360 - diff)

    pick_cube_hue = _hue_degrees(props["pick_cube"]["color"])
    assert hue_distance(pick_cube_hue) <= band_half_width

    distractor_families = {
        "ignore_cube_green": (90, 150),
        "ignore_cube_blue": (200, 260),
        "ignore_cube_yellow": (40, 70),
        "ignore_cube_purple": (240, 320),
        "ignore_cube_orange": (20, 40),
    }
    margin_deg = 10
    for name, (low, high) in distractor_families.items():
        hue = _hue_degrees(props[name]["color"])
        assert low <= hue <= high
        assert hue_distance(hue) >= band_half_width + margin_deg
