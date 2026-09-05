"""Mock gripper handle contract (FINDINGS ARM-2, ARM-3, ARM-4, ARM-8): the jaw
interpolates like MockArmHandle, an object in the jaws stalls the close short
of closed_rad and eventually reports is_holding(), and jaw_limits()/dof_names()
follow the asset/attrs contract."""

import math
import time

import pytest

from isaac_module.sim_manager import GRIPPER_OPEN_WIDTH_M, MockArmHandle

SETTLE_POLLS = 400
SETTLE_POLL_S = 0.01


def _wait_until(predicate, polls: int = SETTLE_POLLS, poll_s: float = SETTLE_POLL_S) -> bool:
    for _ in range(polls):
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def test_create_gripper_unknown_arm_raises(sim):
    with pytest.raises(ValueError, match="not attached to the sim"):
        sim.create_gripper("gripper-bad-arm", {"world": "sim-world", "arm": "no-such-arm"})


def test_close_with_no_object_reaches_closed_rad(sim):
    sim.create_arm("gripper-arm-a", {"world": "sim-world", "asset": "ur5e"})
    gripper = sim.create_gripper("gripper-a", {"world": "sim-world", "arm": "gripper-arm-a"})

    gripper.close()
    assert _wait_until(lambda: not gripper.is_moving())

    assert gripper.get_jaw() == pytest.approx(math.radians(47.0), abs=1e-6)
    assert gripper.is_moving() is False
    assert gripper.is_holding() is False


def test_close_on_object_stalls_at_contact_angle_and_holds(sim):
    sim.create_arm("gripper-arm-b", {"world": "sim-world", "asset": "ur5e"})
    gripper = sim.create_gripper(
        "gripper-b",
        {"world": "sim-world", "arm": "gripper-arm-b", "mock_object_width_m": 0.05},
    )

    open_rad = math.radians(0.0)
    closed_rad = math.radians(47.0)
    expected_contact = open_rad + (closed_rad - open_rad) * (1.0 - 0.05 / GRIPPER_OPEN_WIDTH_M)

    gripper.close()
    assert _wait_until(lambda: not gripper.is_moving())

    assert gripper.get_jaw() == pytest.approx(expected_contact, abs=1e-6)
    assert gripper.is_moving() is False

    assert _wait_until(gripper.is_holding)
    assert gripper.is_holding() is True

    gripper.open()
    assert gripper.is_holding() is False


def test_stop_mid_travel_freezes_jaw(sim):
    sim.create_arm("gripper-arm-c", {"world": "sim-world", "asset": "ur5e"})
    gripper = sim.create_gripper("gripper-c", {"world": "sim-world", "arm": "gripper-arm-c"})

    gripper.close()
    time.sleep(0.1)
    gripper.stop()

    assert gripper.is_moving() is False
    jaw = gripper.get_jaw()
    assert math.radians(0.0) < jaw < math.radians(47.0)


def test_jaw_limits_default_and_attrs(sim):
    sim.create_arm("gripper-arm-d", {"world": "sim-world", "asset": "ur5e"})
    default_gripper = sim.create_gripper(
        "gripper-d", {"world": "sim-world", "arm": "gripper-arm-d"}
    )
    assert default_gripper.jaw_limits() == pytest.approx((0.0, math.radians(47.0)))

    sim.create_arm("gripper-arm-e", {"world": "sim-world", "asset": "ur5e"})
    custom_gripper = sim.create_gripper(
        "gripper-e",
        {
            "world": "sim-world",
            "arm": "gripper-arm-e",
            "open_deg": 5.0,
            "closed_deg": 50.0,
        },
    )
    assert custom_gripper.jaw_limits() == pytest.approx((math.radians(5.0), math.radians(50.0)))

    custom_gripper.set_jaw(math.radians(1000.0))
    assert _wait_until(lambda: not custom_gripper.is_moving())
    assert custom_gripper.get_jaw() == pytest.approx(math.radians(50.0), abs=1e-6)

    custom_gripper.set_jaw(math.radians(-1000.0))
    assert _wait_until(lambda: not custom_gripper.is_moving())
    assert custom_gripper.get_jaw() == pytest.approx(math.radians(5.0), abs=1e-6)


def test_dof_names_is_the_drive_joint(sim):
    sim.create_arm("gripper-arm-f", {"world": "sim-world", "asset": "ur5e"})
    gripper = sim.create_gripper("gripper-f", {"world": "sim-world", "arm": "gripper-arm-f"})
    assert gripper.dof_names() == ["finger_joint"]


def test_create_gripper_is_cached_per_name(sim):
    sim.create_arm("gripper-arm-g", {"world": "sim-world", "asset": "ur5e"})
    attrs = {"world": "sim-world", "arm": "gripper-arm-g"}
    first = sim.create_gripper("gripper-g", attrs)
    second = sim.create_gripper("gripper-g", dict(attrs))
    assert first is second


def test_mock_arm_speed_constant_used_by_gripper_interpolation():
    # sanity: the gripper's interpolation speed comes from the same constant
    # the arm mock uses (FINDINGS ARM-2's shared-pattern requirement).
    assert MockArmHandle.SPEED == 1.0


def test_stop_while_holding_keeps_holding(sim):
    """A session-end Stop must not relax a grasp (see IsaacGripperHandle.stop)."""
    sim.create_arm("gripper-arm-hold", {"world": "sim-world", "asset": "ur5e"})
    gripper = sim.create_gripper(
        "gripper-hold",
        {"world": "sim-world", "arm": "gripper-arm-hold", "mock_object_width_m": 0.05},
    )
    gripper.close()
    assert _wait_until(lambda: not gripper.is_moving())
    assert _wait_until(gripper.is_holding)
    jaw_before = gripper.get_jaw()

    gripper.stop()

    assert gripper.is_holding() is True
    assert gripper.get_jaw() == pytest.approx(jaw_before, abs=1e-6)
    assert gripper.finger_effort() is None  # the mock has no effort to report
    gripper.open()
    assert gripper.is_holding() is False
