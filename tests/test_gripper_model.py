"""Unit tests for IsaacGripper's Viam-facing contract in mock mode."""

import asyncio
import math
import json

import pytest
from grpclib import Status
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.models.gripper import IsaacGripper
from isaac_module.sim_manager import SimManager

_ABSTRACT_METHODS = {
    "open",
    "stop",
    "grab",
    "is_moving",
    "is_holding_something",
    "get_kinematics",
    "get_current_inputs",
    "go_to_inputs",
}


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _make_arm(world, name: str) -> None:
    from isaac_module.models.arm import IsaacArm

    IsaacArm.new(_config(name, {"world": "sim-world", "asset": "ur5e", "mock_dof": 6}), {})


def _make_gripper(world, arm_name: str, name: str, extra: dict | None = None) -> IsaacGripper:
    attrs = {"world": "sim-world", "arm": arm_name}
    if extra:
        attrs.update(extra)
    return IsaacGripper.new(_config(name, attrs), {})


def test_instantiates_with_exactly_the_eight_abstract_methods():
    assert Gripper.__abstractmethods__ == frozenset(_ABSTRACT_METHODS)
    IsaacGripper("gripper-instantiate-only")


def test_validate_config_requires_arm():
    with pytest.raises(ValueError, match="arm"):
        IsaacGripper.validate_config(_config("gripper-no-arm", {"world": "sim-world"}))


def test_validate_config_frame_parent_must_be_arm():
    config = ComponentConfig(
        name="gripper-bad-frame",
        attributes=dict_to_struct({"world": "sim-world", "arm": "my-arm"}),
    )
    config.frame.parent = "not-my-arm"
    with pytest.raises(ValueError, match="frame.parent"):
        IsaacGripper.validate_config(config)


def test_validate_config_valid_returns_deps():
    config = ComponentConfig(
        name="gripper-valid",
        attributes=dict_to_struct({"world": "sim-world", "arm": "my-arm"}),
    )
    config.frame.parent = "my-arm"
    deps, implicit = IsaacGripper.validate_config(config)
    assert list(deps) == ["sim-world", "my-arm"]
    assert list(implicit) == []


def test_grab_and_release_with_object(world):
    _make_arm(world, "grab-arm-a")
    gripper = _make_gripper(world, "grab-arm-a", "grab-gripper-a", {"mock_object_width_m": 0.05})

    async def scenario():
        await gripper.open()
        assert await gripper.grab() is True

        status = await gripper.is_holding_something()
        assert status.is_holding_something is True
        assert status.meta["open_deg"] < status.meta["jaw_deg"] < status.meta["closed_deg"]

        await gripper.open()
        status = await gripper.is_holding_something()
        assert status.is_holding_something is False

    asyncio.run(scenario())


def test_grab_with_no_object_returns_false(world):
    _make_arm(world, "grab-arm-b")
    gripper = _make_gripper(world, "grab-arm-b", "grab-gripper-b")

    async def scenario():
        assert await gripper.grab() is False
        status = await gripper.is_holding_something()
        assert status.meta["jaw_deg"] == pytest.approx(status.meta["closed_deg"], abs=0.5)

    asyncio.run(scenario())


def test_go_to_inputs_and_get_current_inputs_round_trip(world):
    _make_arm(world, "inputs-arm")
    gripper = _make_gripper(world, "inputs-arm", "inputs-gripper")

    async def scenario():
        await gripper.go_to_inputs([0.5])
        inputs = await gripper.get_current_inputs()
        assert inputs == pytest.approx([0.5], abs=1e-3)

    asyncio.run(scenario())


def test_go_to_inputs_out_of_range_raises(world):
    _make_arm(world, "inputs-arm-oob")
    gripper = _make_gripper(world, "inputs-arm-oob", "inputs-gripper-oob")

    async def scenario():
        with pytest.raises(Exception) as excinfo:
            await gripper.go_to_inputs([1.5])
        assert excinfo.value.grpc_code == Status.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_go_to_inputs_wrong_length_raises(world):
    _make_arm(world, "inputs-arm-len")
    gripper = _make_gripper(world, "inputs-arm-len", "inputs-gripper-len")

    async def scenario():
        with pytest.raises(Exception) as excinfo:
            await gripper.go_to_inputs([0.2, 0.3])
        assert excinfo.value.grpc_code == Status.INVALID_ARGUMENT

    asyncio.run(scenario())


def test_get_kinematics_is_one_link_zero_joints(world):
    _make_arm(world, "kinematics-arm")
    gripper = _make_gripper(world, "kinematics-arm", "kinematics-gripper")

    async def scenario():
        fmt, data = await gripper.get_kinematics()
        from viam.proto.common import KinematicsFileFormat

        assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
        sva = json.loads(data)
        assert len(sva["links"]) == 1
        assert sva["joints"] == []
        link = sva["links"][0]
        assert link["parent"] == "world"
        geometry = link["geometry"]
        assert (geometry["x"], geometry["y"], geometry["z"]) == (36, 146, 153)
        # flange -> fingertips: centre 57.5 mm behind the TCP, so the box never
        # extends below the pads (a floor-level grasp would read as a collision)
        assert geometry["translation"]["z"] == pytest.approx(-57.5)

    asyncio.run(scenario())


def test_get_geometries_is_one_box(world):
    _make_arm(world, "geometries-arm")
    gripper = _make_gripper(world, "geometries-arm", "geometries-gripper")

    async def scenario():
        geometries = await gripper.get_geometries()
        assert len(geometries) == 1
        geometry = geometries[0]
        box = geometry.box
        assert (box.dims_mm.x, box.dims_mm.y, box.dims_mm.z) == (36, 146, 153)
        assert geometries[0].center.z == pytest.approx(-57.5)
        assert geometry.center.o_z == 1

    asyncio.run(scenario())


def test_is_moving_true_during_open_then_false(world):
    _make_arm(world, "moving-arm")
    gripper = _make_gripper(world, "moving-arm", "moving-gripper")

    async def scenario():
        await gripper.grab()  # closes with nothing to grab -> jaw at closed_rad
        await gripper.open()
        assert await gripper.is_moving() is True
        for _ in range(500):
            if not await gripper.is_moving():
                break
            await asyncio.sleep(0.01)
        assert await gripper.is_moving() is False

    asyncio.run(scenario())


def test_close_releases_the_handle(world):
    _make_arm(world, "close-arm")
    gripper = _make_gripper(world, "close-arm", "close-gripper")

    async def scenario():
        await gripper.close()
        assert "close-gripper" not in SimManager.get()._handles

    asyncio.run(scenario())


def test_tcp_pose_do_command_measures_the_configured_offset_in_mock(world):
    """GPU checklist item 4: the fingertip midpoint sits tcp_offset_m along the
    mount link's +Z, so measured == configured and delta is 0 in the mock."""
    arm = IsaacArm.new(_config("tcp-arm", {"world": "sim-world", "asset": "ur5e"}), {})
    gripper = IsaacGripper.new(
        _config("tcp-grip", {"world": "sim-world", "arm": arm.name, "tcp_offset_m": 0.115}), {}
    )
    out = asyncio.run(gripper.do_command({"command": "tcp_pose"}))
    assert out["configured_tcp_offset_mm"] == pytest.approx(115.0)
    assert out["measured_tcp_offset_mm"] == pytest.approx(115.0)
    assert out["delta_mm"] == pytest.approx(0.0)
    assert out["pad_center_midpoint_mm"] == pytest.approx([0.0, 0.0, 115.0])
    assert out["jaw_gap_mm"] == pytest.approx(85.0)
    assert out["fingertip_reach_mm"] == pytest.approx(115.0 + 19.0)
    assert set(out) >= {"parent", "left_inner_finger", "right_inner_finger", "fingertips"}


def test_debug_commands_are_empty_in_the_mock(world):
    _make_arm(world, "debug-arm")
    gripper = _make_gripper(world, "debug-arm", "debug-gripper")

    async def scenario():
        assert await gripper.do_command({"command": "contacts"}) == {"contacts": []}
        assert await gripper.do_command({"command": "collision_shapes"}) == {"collision_shapes": []}

    asyncio.run(scenario())



class _JammingHandle:
    """Closes onto something without holding the first time (a jammed
    linkage), holds after a regrip. Records what grab() commanded."""

    def __init__(self, jams: int = 1) -> None:
        self.jams = jams
        self.closes = 0
        self.commands: list[str] = []
        self._jaw = 0.0
        self._holding = False

    def jaw_limits(self):
        return (0.0, 0.8203)

    def get_jaw(self):
        return self._jaw

    def close(self):
        self.closes += 1
        self.commands.append("close")
        self._jaw = 0.235  # ~13.5 deg: stalled on the block
        self._holding = self.closes > self.jams

    def set_jaw(self, rad):
        self.commands.append(f"set_jaw({math.degrees(rad):.0f})")
        self._jaw = rad
        if getattr(self, "nudge_holds", False) and rad > 0.235:
            self._holding = True  # a nudge past the jam bites

    def open(self):
        self.commands.append("open")
        self._jaw = 0.0

    def is_moving(self):
        return False

    def is_holding(self):
        return self._holding

    def finger_effort(self):
        return 2.4 if self._holding else 0.017


def test_grab_regrips_when_the_jaw_jams_short_of_closed_without_holding():
    import math as _math

    gripper = IsaacGripper("gripper-regrip")
    handle = _JammingHandle(jams=1)
    gripper._handle = handle
    gripper._grab_timeout = 5.0
    assert asyncio.run(gripper.grab()) is True
    # first regrip is a nudge past the jam; it does not bite in this fake, so the
    # second regrip backs off and closes again, which does
    assert handle.closes == 2
    assert handle.commands == [
        "close",
        f"set_jaw({_math.degrees(0.235 + _math.radians(5.0)):.0f})",
        f"set_jaw({_math.degrees(0.235 + _math.radians(5.0) - _math.radians(8.0)):.0f})",  # backs off from the nudged jaw
        "close",
    ]


def test_grab_nudge_past_the_jam_bites():
    gripper = IsaacGripper("gripper-nudge")
    handle = _JammingHandle(jams=10)
    handle.nudge_holds = True
    gripper._handle = handle
    gripper._grab_timeout = 5.0
    assert asyncio.run(gripper.grab()) is True
    assert handle.closes == 1 and handle.commands[-1].startswith("set_jaw(18")


def test_grab_gives_up_after_the_regrip_budget():
    gripper = IsaacGripper("gripper-regrip-fail")
    handle = _JammingHandle(jams=10)
    gripper._handle = handle
    gripper._grab_timeout = 5.0
    assert asyncio.run(gripper.grab()) is False
    assert handle.closes == 2  # close, nudge, back off + close, nudge
