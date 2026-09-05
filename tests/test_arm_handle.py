"""DOF-index-aware joint I/O on the arm handles (FINDINGS ARM-1; R-3):
ArmHandle.dof_names() exposes the full articulation, while
get_joint_positions/set_joint_targets/is_moving/stop operate on exactly the
arm's named joints, selected by index rather than position - so attaching a
gripper later cannot shift or truncate arm joints.

Also covers the Isaac-side ARM-12/ARM-15/ARM-16/XC-4 physics-callback settle,
post-reset re-tune, and handle release, against a fake articulation/world
harness (no real Isaac Sim needed)."""

import asyncio
import threading
import time

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.models.arm import IsaacArm
from isaac_module.sim_manager import (
    SETTLE_WINDOW_STEPS,
    STALL_NO_PROGRESS_STEPS,
    UR_JOINT_NAMES,
    IsaacArmHandle,
    SettleOutcome,
    resolve_joint_indices,
)

SETTLE_POLLS = 200
SETTLE_POLL_S = 0.01


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def test_mock_ur_arm_dof_names_and_padding(world):
    arm = IsaacArm.new(
        _config("padded-arm", {"world": "sim-world", "asset": "ur5e", "mock_dof": 12}), {}
    )
    handle = arm._handle
    names = handle.dof_names()
    assert len(names) == 12
    assert tuple(names[:6]) == UR_JOINT_NAMES

    async def scenario():
        start = await arm.get_joint_positions()
        assert len(start.values) == 6

    asyncio.run(scenario())


def test_mock_ur_arm_moves_selected_joints_only_padding_stays_zero(world):
    arm = IsaacArm.new(
        _config("padded-move-arm", {"world": "sim-world", "asset": "ur5e", "mock_dof": 12}),
        {},
    )
    handle = arm._handle

    async def scenario():
        await asyncio.to_thread(handle.set_joint_targets, [0.1] * 6)
        for _ in range(SETTLE_POLLS):
            if not await asyncio.to_thread(handle.is_moving):
                break
            await asyncio.sleep(SETTLE_POLL_S)
        assert not await asyncio.to_thread(handle.is_moving)

        end = await asyncio.to_thread(handle.get_joint_positions)
        assert end == pytest.approx([0.1] * 6, abs=1e-6)

        all_positions = handle.get_all_joint_positions()
        assert all_positions[:6] == pytest.approx([0.1] * 6, abs=1e-6)
        assert all_positions[6:] == pytest.approx([0.0] * 6, abs=1e-9)

    asyncio.run(scenario())


def test_mock_franka_arm_has_no_declared_joint_names(world):
    arm = IsaacArm.new(
        _config("franka-arm", {"world": "sim-world", "asset": "franka", "mock_dof": 7}), {}
    )
    handle = arm._handle
    assert len(handle.dof_names()) == 7

    async def scenario():
        positions = await arm.get_joint_positions()
        assert len(positions.values) == 7

    asyncio.run(scenario())


def test_mock_ur_arm_rejects_length_mismatch(world):
    arm = IsaacArm.new(
        _config("mismatch-arm", {"world": "sim-world", "asset": "ur5e", "mock_dof": 12}), {}
    )
    handle = arm._handle
    with pytest.raises(ValueError, match="6") as excinfo:
        handle.set_joint_targets([0.1] * 5)
    assert "5" in str(excinfo.value)


def test_resolve_joint_indices_maps_by_name_not_position():
    dof_names = [
        "wrist_2_joint",
        "shoulder_pan_joint",
        "wrist_3_joint",
        "shoulder_lift_joint",
        "wrist_1_joint",
        "elbow_joint",
    ]
    indices = resolve_joint_indices(dof_names, UR_JOINT_NAMES)
    assert indices == [1, 3, 5, 4, 0, 2]


def test_resolve_joint_indices_none_when_no_names_declared():
    assert resolve_joint_indices(["a", "b"], None) is None


def test_resolve_joint_indices_raises_naming_missing_joint():
    dof_names = ["shoulder_pan_joint", "shoulder_lift_joint"]
    with pytest.raises(ValueError, match="elbow_joint") as excinfo:
        resolve_joint_indices(dof_names, UR_JOINT_NAMES)
    assert "shoulder_pan_joint" in str(excinfo.value)


# ----------------------------------------------------------------------
# ARM-12/ARM-15/ARM-16/XC-4: IsaacArmHandle against a fake articulation/world
# (no real Isaac Sim). FakeSim.run(fn) calls fn() inline - the tests drive
# the registered physics callback directly from a helper thread to simulate
# the sim thread stepping while wait_for_settle blocks the caller thread.
# ----------------------------------------------------------------------

CALLBACK_WAIT_POLLS = 200
CALLBACK_WAIT_POLL_S = 0.005
FAKE_PHYSICS_STEP_S = 0.01  # step_size passed to the settle callback each fake step


class FakeArticulationAction:
    def __init__(self, joint_positions=None, joint_indices=None) -> None:
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class FakeArticulationController:
    def __init__(self, kps, kds) -> None:
        self._gains = (kps, kds)
        self.set_gains_calls: list[tuple] = []

    def get_gains(self):
        return self._gains

    def set_gains(self, kps, kds) -> None:
        self._gains = (kps, kds)
        self.set_gains_calls.append((kps, kds))


class FakeArticulation:
    def __init__(self, name: str, dof_names: list[str]) -> None:
        self.name = name
        self.dof_names = list(dof_names)
        dof_count = len(dof_names)
        self.positions = [0.0] * dof_count
        self.velocities = [0.0] * dof_count
        self._controller = FakeArticulationController([1.0] * dof_count, [1.0] * dof_count)
        self.solver_iteration_calls: list[int] = []
        self.applied_actions: list[FakeArticulationAction] = []

    def get_articulation_controller(self):
        return self._controller

    def set_solver_position_iteration_count(self, count: int) -> None:
        self.solver_iteration_calls.append(count)

    def get_joint_positions(self, joint_indices=None):
        indices = range(len(self.positions)) if joint_indices is None else joint_indices
        return [self.positions[i] for i in indices]

    def get_joint_velocities(self, joint_indices=None):
        indices = range(len(self.velocities)) if joint_indices is None else joint_indices
        return [self.velocities[i] for i in indices]

    def apply_action(self, action: FakeArticulationAction) -> None:
        self.applied_actions.append(action)
        indices = (
            range(len(self.positions)) if action.joint_indices is None else action.joint_indices
        )
        for i, p in zip(indices, action.joint_positions, strict=True):
            self.positions[i] = p


class FakeScene:
    def __init__(self) -> None:
        self.objects: dict[str, object] = {}
        self.removed: list[tuple[str, bool]] = []

    def get_object(self, name: str):
        return self.objects.get(name)

    def remove_object(self, name: str, registry_only: bool = False) -> None:
        self.removed.append((name, registry_only))
        self.objects.pop(name, None)


class FakeWorld:
    def __init__(self) -> None:
        self._callbacks: dict[str, object] = {}
        self.scene = FakeScene()

    def add_physics_callback(self, name: str, fn) -> None:
        self._callbacks[name] = fn

    def remove_physics_callback(self, name: str) -> None:
        self._callbacks.pop(name, None)

    def physics_callback_exists(self, name: str) -> bool:
        return name in self._callbacks


class FakeIsaacNamespace:
    ArticulationAction = FakeArticulationAction


class FakeSim:
    def __init__(self) -> None:
        self.world = FakeWorld()
        self._isaac = FakeIsaacNamespace()

    def run(self, fn, timeout: float = 30.0):
        return fn()


def _make_handle(dof_names=("j0", "j1")) -> tuple[IsaacArmHandle, FakeArticulation, FakeSim]:
    art = FakeArticulation("fake-arm", list(dof_names))
    sim = FakeSim()
    sim.world.scene.objects[art.name] = art
    handle = IsaacArmHandle(sim, art, None, joint_names=None)
    return handle, art, sim


def _wait_until(predicate, polls: int = CALLBACK_WAIT_POLLS, poll_s: float = CALLBACK_WAIT_POLL_S):
    for _ in range(polls):
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def test_isaac_settle_reaches_after_window_not_before():
    handle, art, sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.0, 0.0]
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=5.0))
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]

    # a broken (no-window) implementation would report REACHED on step 1
    for _ in range(SETTLE_WINDOW_STEPS - 1):
        callback(FAKE_PHYSICS_STEP_S)
        assert "outcome" not in result
    callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.REACHED
    assert not sim.world.physics_callback_exists(callback_name)


def test_isaac_settle_stalls_when_still_but_off_target():
    handle, art, sim = _make_handle()
    handle._targets = [1.0, 1.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.0, 0.0]  # still, but never converges
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=5.0))
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]

    for _ in range(SETTLE_WINDOW_STEPS):
        callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.STALLED
    assert not sim.world.physics_callback_exists(callback_name)


def test_isaac_settle_times_out_while_still_converging():
    handle, art, sim = _make_handle()
    handle._targets = [1.0, 1.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.5, 0.5]  # still moving - never "still", never stalled
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=0.05))
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]

    # step_size accumulation crosses timeout_s (0.05s) well before the
    # window's worth of steps would (it never reaches the window anyway).
    for _ in range(10):
        callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.TIMED_OUT
    assert not sim.world.physics_callback_exists(callback_name)


def test_isaac_arm_post_reset_reapplies_solver_gains_and_holds_the_reset_pose():
    handle, art, sim = _make_handle()
    handle._solver_iterations = 64
    handle._gains = ([2.0, 2.0], [3.0, 3.0])
    handle._targets = [0.4, 0.5]  # pre-reset targets: NOT re-commanded (see post_reset)
    art.positions = [0.1, 0.2]  # where the reset put the joints

    handle.post_reset()

    assert art.solver_iteration_calls == [64]
    assert art._controller.set_gains_calls == [([2.0, 2.0], [3.0, 3.0])]
    assert art.applied_actions
    last = art.applied_actions[-1]
    assert list(last.joint_positions) == pytest.approx([0.1, 0.2])
    assert handle._targets == pytest.approx([0.1, 0.2])


def test_isaac_arm_release_removes_settle_callback_and_registry_entry():
    handle, art, sim = _make_handle()
    callback_name = handle._settle_callback_name()
    sim.world.add_physics_callback(callback_name, lambda step_size: None)

    handle.release()

    assert not sim.world.physics_callback_exists(callback_name)
    assert sim.world.scene.removed == [(art.name, True)]
    assert art.name not in sim.world.scene.objects


def test_mock_is_moving_true_while_stalled_and_false_after_stop():
    """ARM-12 / R-7: IsMoving unifies the two "arrived" notions. A stalled arm
    (velocity ~0 but commanded != measured) still reports moving; stop()
    re-targets the current position so both terms agree and it reports still."""
    from isaac_module.sim_manager import MockArmHandle, SettleOutcome

    handle = MockArmHandle("stalled-is-moving-arm", {"asset": "ur5e", "mock_stall_fraction": 0.5})
    handle.set_joint_targets([0.2] * 6)
    assert handle.wait_for_settle(5.0) is SettleOutcome.STALLED
    assert handle.is_moving() is True  # position term: 0.1 rad short of every target
    handle.stop()
    assert handle.is_moving() is False


def test_replace_articulation_retakes_gains_and_dofs_from_the_new_topology():
    """A gripper joining the arm changes the DOF count; the fresh wrapper's
    gains snapshot comes from the new topology, and the named joints resolve
    against the new dof list (ARM-2 / ARM-15)."""
    handle, _old_art, _sim = _make_handle(("j0", "j1"))
    handle._solver_iterations = 64
    handle._gains = ("stale", "stale")
    new_art = FakeArticulation("fake-arm", ["j0", "j1", "finger_joint"])
    new_art.get_articulation_controller().set_gains([2.0] * 3, [3.0] * 3)

    handle.replace_articulation(new_art)
    assert handle._gains is None

    handle.refresh_dofs()
    assert handle.all_dof_names() == ["j0", "j1", "finger_joint"]
    assert handle._gains == ([2.0] * 3, [3.0] * 3)  # retaken from the new articulation


def test_isaac_settle_reaches_despite_residual_velocity_jitter():
    """GPU run 11: a 12-DOF PhysX articulation idles with ~1e-3 rad/s of jitter
    and never reads exactly still. Holding within tolerance for the window IS
    settled - a velocity-gated REACHED would time out forever."""
    handle, art, sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.004, -0.003]  # above the old 1e-3 gate, below any real motion
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=5.0))
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]
    for _ in range(SETTLE_WINDOW_STEPS):
        callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.REACHED


def test_isaac_is_moving_ignores_jitter_but_not_motion():
    handle, art, _sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.0, 0.0]
    art.velocities = [0.004, -0.003]
    assert handle.is_moving() is False
    art.velocities = [0.05, 0.0]  # 2.9 deg/s: real motion
    assert handle.is_moving() is True


def test_isaac_settle_stalls_on_no_progress_even_while_vibrating():
    """GPU run 15: pinned against a block the joints vibrate above VEL_EPS and
    never read still, so the old stall rule let the arm push for the whole
    deadline. A worst-error that stops improving is a stall too."""
    handle, art, sim = _make_handle()
    handle._targets = [0.0, 0.0]
    art.positions = [0.4, 0.0]  # pinned 0.4 rad short of target
    art.velocities = [0.05, -0.05]  # vibrating, well above VEL_EPS
    callback_name = handle._settle_callback_name()

    result: dict[str, SettleOutcome] = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__("outcome", handle.wait_for_settle(timeout_s=30.0))
    )
    thread.start()
    assert _wait_until(lambda: sim.world.physics_callback_exists(callback_name))
    callback = sim.world._callbacks[callback_name]
    # step 1 records the first best error; the window counts steps AFTER it
    for _ in range(STALL_NO_PROGRESS_STEPS):
        callback(FAKE_PHYSICS_STEP_S)
        assert "outcome" not in result  # progress window not yet exhausted
    callback(FAKE_PHYSICS_STEP_S)
    thread.join(timeout=2.0)

    assert result["outcome"] == SettleOutcome.STALLED



# ---- time-synchronized joint interpolation (set_joint_targets) ----------------


def _interp_step(sim, art):
    return sim.world._callbacks[f"{art.name}_interp"]


def test_set_joint_targets_moves_all_joints_along_a_straight_line_together():
    """A planner validates the straight joint-space segment between two
    waypoints. Handing PhysX the final targets let each joint run at its own
    speed, so the executed path was an arc that swept the fingertips through
    the block (2026-09-04). Every joint must reach its goal on the same step."""
    handle, art, sim = _make_handle()
    handle.set_joint_targets([1.0, 0.1], max_vel_rad_s=1.0)  # 1 s at 1 rad/s for the long joint
    assert art.applied_actions[-1].joint_positions.tolist() == [0.0, 0.0]  # starts where it is
    step = _interp_step(sim, art)
    for k in range(1, 5):
        step(0.25)
        assert art.applied_actions[-1].joint_positions.tolist() == pytest.approx([0.25 * k, 0.025 * k])
    assert handle._interp is None  # arrived: no further targets
    n = len(art.applied_actions)
    step(0.25)
    assert len(art.applied_actions) == n


def test_set_joint_targets_uses_sync_speed_when_the_move_sets_no_cap():
    import math

    from isaac_module.sim_manager import SYNC_JOINT_VEL_RAD_S

    handle, art, sim = _make_handle()
    handle.set_joint_targets([math.radians(60.0), 0.0])
    assert handle._interp["segments"][0]["duration"] == pytest.approx(
        math.radians(60.0) / SYNC_JOINT_VEL_RAD_S
    )


def test_tiny_moves_are_still_interpolated_and_finish_in_one_step():
    """Even a 0.06 deg move is a segment (dropping small travels collapsed
    dense linear plans into a jump); it finishes on the first step."""
    handle, art, sim = _make_handle()
    handle.set_joint_targets([0.001, 0.0], max_vel_rad_s=1.0)
    assert handle._interp is not None and len(handle._interp["segments"]) == 1
    _interp_step(sim, art)(0.01)
    assert handle._interp is None
    assert art.applied_actions[-1].joint_positions.tolist() == pytest.approx([0.001, 0.0])


def test_stop_drops_the_in_flight_interpolation_and_holds():
    handle, art, sim = _make_handle()
    handle.set_joint_targets([1.0, 0.0], max_vel_rad_s=1.0)
    step = _interp_step(sim, art)
    step(0.5)
    assert art.positions[0] == pytest.approx(0.5)
    handle.stop()
    assert handle._interp is None
    assert handle._targets == pytest.approx([0.5, 0.0])
    n = len(art.applied_actions)
    step(0.5)
    assert len(art.applied_actions) == n  # nothing keeps driving toward the old goal



def test_post_reset_holds_the_reset_pose_not_the_old_targets():
    """A world.reset() teleports the arm to its default pose and props to
    their spawn poses. Re-commanding the pre-reset targets drove the arm back
    through the freshly placed block (2026-09-04); it must hold where it is."""
    handle, art, sim = _make_handle()
    handle.set_joint_targets([1.0, 0.5], max_vel_rad_s=100.0)
    art.positions = [0.0, 0.0]  # the reset put the joints back at their defaults
    handle.post_reset()
    assert handle._targets == pytest.approx([0.0, 0.0])
    assert art.applied_actions[-1].joint_positions.tolist() == pytest.approx([0.0, 0.0])
    assert handle._interp is None



def test_follow_joint_path_runs_through_every_waypoint_without_settling():
    """A plan executes as one continuous path: each segment is a synchronized
    straight line, the next starts the step the previous ends, and only the
    final waypoint is the settle target."""
    handle, art, sim = _make_handle()
    handle.follow_joint_path([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], max_vel_rad_s=1.0)
    assert handle._targets == pytest.approx([0.0, 1.0])
    assert handle.path_progress() == (0, 3)
    step = _interp_step(sim, art)
    seen = []
    for _ in range(12):
        step(0.25)
        seen.append(tuple(round(v, 3) for v in art.applied_actions[-1].joint_positions.tolist()))
    assert (1.0, 0.0) in seen and (1.0, 1.0) in seen and (0.0, 1.0) in seen
    assert seen.index((1.0, 0.0)) < seen.index((1.0, 1.0)) < seen.index((0.0, 1.0))
    assert handle._interp is None and handle.path_progress() is None


def test_path_pauses_while_the_arm_lags_and_flags_a_stall():
    """Contact: the drives cannot follow. The commanded target must not run
    ahead, and a lag that outlasts STALL_NO_PROGRESS_STEPS is a stall."""
    from isaac_module.sim_manager import STALL_NO_PROGRESS_STEPS

    handle, art, sim = _make_handle()
    art.apply_action = lambda action: art.applied_actions.append(action)  # joints never move
    handle.follow_joint_path([[1.0, 0.0]], max_vel_rad_s=1.0)
    step = _interp_step(sim, art)
    for _ in range(5):
        step(0.1)
    commanded = art.applied_actions[-1].joint_positions.tolist()
    assert commanded[0] <= 0.1 + 1e-9  # one step at most: the target stopped once the arm fell behind
    # once the arm catches up to within the tolerance, the path advances again
    art.positions[0] = commanded[0] - 0.005
    step(0.1)
    assert art.applied_actions[-1].joint_positions[0] > commanded[0]
    assert handle._interp["stalled"] is False and handle._interp["lag_steps"] == 0
    art.positions[0] = 0.0
    commanded = art.applied_actions[-1].joint_positions.tolist()
    for _ in range(STALL_NO_PROGRESS_STEPS + 1):  # the first lagging step records the best lag
        step(0.1)
    assert handle._interp["stalled"] is True
    assert art.applied_actions[-1].joint_positions.tolist() == pytest.approx(commanded)



def test_path_lead_never_exceeds_the_settle_tolerance():
    """The commanded target may run ahead of the measured joints by at most
    SETTLE_TOL_RAD, so the tool stays on the planner's line."""
    from isaac_module.sim_manager import SETTLE_TOL_RAD

    handle, art, sim = _make_handle()
    art.apply_action = lambda action: art.applied_actions.append(action)  # drives, not teleports
    handle.follow_joint_path([[1.0, 0.0]], max_vel_rad_s=10.0)
    step = _interp_step(sim, art)
    worst = 0.0
    for _ in range(400):
        # a drive: every physics step the joint closes 30% of the gap to the last target
        target = art.applied_actions[-1].joint_positions[0]
        art.positions[0] += (float(target) - art.positions[0]) * 0.3
        step(0.01)
        worst = max(worst, art.applied_actions[-1].joint_positions[0] - art.positions[0])
    assert worst <= SETTLE_TOL_RAD + 10.0 * 0.01 + 1e-9  # one step of advance past the check
    assert handle._interp is None, "the path should complete once the drive keeps up"
    assert art.positions[0] == pytest.approx(1.0, abs=0.02)



def test_dense_waypoints_are_all_kept_and_followed_in_order():
    """A linear plan's waypoints are ~0.3 deg apart, under the settle
    tolerance; every one must become a segment so the tool follows the line
    instead of jumping to the last waypoint (2026-09-05)."""
    import math

    handle, art, sim = _make_handle()
    step_rad = math.radians(0.3)
    waypoints = [[step_rad * (i + 1), -step_rad * (i + 1)] for i in range(30)]
    handle.follow_joint_path(waypoints, max_vel_rad_s=1.0)
    assert handle.path_progress() == (0, 30)
    step = _interp_step(sim, art)
    seen = []
    for _ in range(400):
        step(0.001)
        seen.append(round(art.applied_actions[-1].joint_positions[0], 6))
        if handle._interp is None:
            break
    assert handle._interp is None
    assert seen == sorted(seen)  # monotonic through the waypoints, never a jump
    assert seen[-1] == pytest.approx(step_rad * 30, rel=1e-4)
