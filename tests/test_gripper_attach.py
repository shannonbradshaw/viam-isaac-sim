"""Pure helpers behind the Isaac gripper attach (FINDINGS ARM-2, R-4), driven
with fake prims: articulation-root removal and the fingertip collision check."""

import pytest

from isaac_module.sim_manager import (
    PAD_PRIM_NAME_FRAGMENTS,
    _pad_collision_status,
    _remove_articulation_roots,
)


class FakeApi:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeUsdPhysics:
    ArticulationRootAPI = FakeApi("ArticulationRootAPI")
    CollisionAPI = FakeApi("CollisionAPI")


class FakePhysxSchema:
    PhysxArticulationAPI = FakeApi("PhysxArticulationAPI")


class FakePrim:
    def __init__(self, path: str, apis: set[str] | None = None, children=(), proxy: bool = False):
        self._path = path
        self.apis = set(apis or ())
        self.children = list(children)
        self._proxy = proxy

    def GetPath(self):
        return self._path

    def GetName(self):
        return self._path.rsplit("/", 1)[-1]

    def HasAPI(self, api: FakeApi) -> bool:
        return api.name in self.apis

    def RemoveAPI(self, api: FakeApi) -> None:
        self.apis.discard(api.name)

    def IsInstanceProxy(self) -> bool:
        return self._proxy

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


class FakeUsd:
    @staticmethod
    def PrimRange(prim: FakePrim):
        return list(prim.walk())


def _gripper_tree() -> tuple[FakePrim, FakePrim, FakePrim]:
    fingertip = FakePrim(
        "/World/pick_arm/Gripper/Robotiq_2F_85/left_inner_finger/visuals/PAD_OPEN_fingertipsstep_01",
        apis={"CollisionAPI"},
    )
    inner_finger = FakePrim(
        "/World/pick_arm/Gripper/Robotiq_2F_85/left_inner_finger",
        children=[
            FakePrim(
                "/World/pick_arm/Gripper/Robotiq_2F_85/left_inner_finger/visuals",
                children=[fingertip],
            )
        ],
    )
    asset_root = FakePrim(
        "/World/pick_arm/Gripper/Robotiq_2F_85",
        apis={"ArticulationRootAPI", "PhysxArticulationAPI"},
        children=[inner_finger],
    )
    gripper_prim = FakePrim("/World/pick_arm/Gripper", children=[asset_root])
    return gripper_prim, asset_root, fingertip


def test_articulation_root_is_removed_from_the_asset_prim_not_our_gripper_prim():
    gripper_prim, asset_root, _fingertip = _gripper_tree()
    removed = _remove_articulation_roots(FakeUsd, FakeUsdPhysics, FakePhysxSchema, gripper_prim)
    assert removed == ["/World/pick_arm/Gripper/Robotiq_2F_85"]
    assert asset_root.apis == set()  # both APIs gone
    assert gripper_prim.apis == set()  # nothing was ever applied to our prim


def test_articulation_root_removal_skips_instance_proxies_and_reports_none_when_absent():
    proxy = FakePrim("/g/proxy", apis={"ArticulationRootAPI"}, proxy=True)
    root = FakePrim("/g", children=[proxy])
    assert _remove_articulation_roots(FakeUsd, FakeUsdPhysics, FakePhysxSchema, root) == []
    assert "ArticulationRootAPI" in proxy.apis


def test_fingertip_collision_counts_for_the_inner_finger_link_subtree():
    gripper_prim, _asset_root, fingertip = _gripper_tree()
    status = dict(
        (path, (on_self, in_subtree))
        for path, on_self, in_subtree in _pad_collision_status(
            FakeUsd, FakeUsdPhysics, gripper_prim
        )
    )
    assert status["/World/pick_arm/Gripper/Robotiq_2F_85/left_inner_finger"] == (False, True)
    assert status[fingertip.GetPath()] == (True, True)
    assert "inner_finger" in PAD_PRIM_NAME_FRAGMENTS and "fingertip" in PAD_PRIM_NAME_FRAGMENTS


def test_no_collision_anywhere_reports_all_false():
    gripper_prim, _asset_root, fingertip = _gripper_tree()
    fingertip.apis.clear()
    status = _pad_collision_status(FakeUsd, FakeUsdPhysics, gripper_prim)
    assert status and all(not in_subtree for _path, _on_self, in_subtree in status)


# ---------------------------------------------------------------------------
# de-instance + rewrite must not touch expired instance proxies (GPU run 6)
# ---------------------------------------------------------------------------

ISAAC_DEV_PART = "omniverse://isaac-dev.ov.nvidia.com/Isaac/Robots/Robotiq/2F-85/parts/part.usd"
ASSETS_ROOT = "https://bucket.example/Assets/Isaac/5.0"


class ExpiringPrim(FakePrim):
    """A prim whose proxies expire (raise on any access) once their instance is
    de-instanced, like pxr instance proxies do."""

    def __init__(self, path, *, instanceable=False, proxy=False, refs=(), children=()):
        super().__init__(path, children=children, proxy=proxy)
        self._instanceable = instanceable
        self.expired = False
        self.refs = [FakeRef(r) for r in refs]
        self.set_refs: list | None = None

    def _check(self):
        if self.expired:
            raise RuntimeError(f"Accessed invalid expired instance proxy prim <{self._path}>")

    def IsInstanceProxy(self):
        self._check()
        return self._proxy

    def IsInstanceable(self):
        self._check()
        return self._instanceable

    def IsValid(self):
        return not self.expired

    def SetInstanceable(self, value):
        self._check()
        self._instanceable = value
        if not value:
            self._replace_proxies_with_fresh_prims(self)

    @staticmethod
    def _replace_proxies_with_fresh_prims(parent):
        """Like pxr: the old proxy objects expire, and a fresh traversal of
        the stage yields new, real prims at the same paths."""
        for index, child in enumerate(list(parent.children)):
            if child._proxy:
                child.expired = True
                fresh = ExpiringPrim(child.GetPath(), refs=[r.assetPath for r in child.refs])
                fresh.children = child.children
                parent.children[index] = fresh
                child = fresh
            ExpiringPrim._replace_proxies_with_fresh_prims(child)

    def GetMetadata(self, key):
        self._check()
        return FakeListOp(self.refs) if key == "references" else None

    def GetReferences(self):
        self._check()
        return self

    def SetReferences(self, items):
        self.set_refs = list(items)

    def GetStage(self):
        return FakeStage(self)


class FakeRef:
    def __init__(self, asset_path):
        self.assetPath = asset_path
        self.primPath = ""
        self.layerOffset = None


class FakeListOp:
    def __init__(self, items):
        self._items = items

    def GetAddedOrExplicitItems(self):
        return self._items


class FakeStage:
    """Resolves a path against the CURRENT tree, like a real stage."""

    def __init__(self, root):
        self._root = root

    def GetPrimAtPath(self, path):
        return next(prim for prim in self._root.walk() if prim.GetPath() == path)


class FakeSdf:
    @staticmethod
    def Reference(asset_path, prim_path, layer_offset):
        return FakeRef(asset_path)


class FakeIsaac:
    client = None  # _usd_exists -> None -> "could not check", so the rewrite is tried

    @staticmethod
    def get_assets_root_path():
        return ASSETS_ROOT


def test_rewrite_de_instances_by_path_and_never_touches_expired_proxies():
    from isaac_module.sim_manager import SimManager

    mesh = ExpiringPrim("/g/visuals/mesh", proxy=True, refs=[ISAAC_DEV_PART])
    visuals = ExpiringPrim("/g/visuals", instanceable=True, children=[mesh])
    root = ExpiringPrim("/g", children=[visuals])
    stage = FakeStage(root)
    root.GetStage = lambda: stage  # one shared stage for the whole tree

    manager = SimManager()
    manager._isaac = FakeIsaac()
    report = manager._rewrite_unresolvable_references(FakeUsd, FakeSdf, root)

    assert report["de_instanced"] == ["/g/visuals"]
    assert report["applied"] == [
        (ISAAC_DEV_PART, ASSETS_ROOT + "/Isaac/Robots/Robotiq/2F-85/parts/part.usd")
    ]
    assert report["missing"] == []
    fresh_mesh = stage.GetPrimAtPath("/g/visuals/mesh")
    assert fresh_mesh.set_refs is not None and fresh_mesh.set_refs[0].assetPath.startswith(
        ASSETS_ROOT
    )


# ---------------------------------------------------------------------------
# passive linkage drives are released; only finger_joint is driven (GPU run 21)
# ---------------------------------------------------------------------------


class FakeController:
    def __init__(self, count: int) -> None:
        self.kps = [1000.0] * count
        self.kds = [100.0] * count
        self.set_calls: list[tuple[list[float], list[float]]] = []

    def get_gains(self):
        return list(self.kps), list(self.kds)

    def set_gains(self, kps, kds) -> None:
        self.kps, self.kds = [float(v) for v in kps], [float(v) for v in kds]
        self.set_calls.append((list(self.kps), list(self.kds)))


class FakeAction:
    def __init__(self, joint_positions=None, joint_indices=None) -> None:
        self.joint_positions = list(joint_positions)
        self.joint_indices = list(joint_indices)


class FakeIsaacNs:
    ArticulationAction = FakeAction


class FakeGripperSim:
    _isaac = FakeIsaacNs()

    @staticmethod
    def run(fn, timeout=30.0):
        return fn()


class FakeArmArticulation:
    def __init__(self, dof_names: list[str]) -> None:
        self.dof_names = list(dof_names)
        self.controller = FakeController(len(dof_names))
        self.actions: list[FakeAction] = []
        self.positions = [0.0] * len(dof_names)

    def get_articulation_controller(self):
        return self.controller

    def apply_action(self, action: FakeAction) -> None:
        self.actions.append(action)

    def get_joint_positions(self, joint_indices=None):
        indices = joint_indices if joint_indices is not None else range(len(self.positions))
        return [self.positions[i] for i in indices]


ARM_AND_GRIPPER_DOFS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]


def test_gripper_handle_releases_passive_drives_and_drives_finger_joint_only():
    import math

    from isaac_module.sim_manager import PASSIVE_JOINT_DAMPING, IsaacGripperHandle

    art = FakeArmArticulation(ARM_AND_GRIPPER_DOFS)
    handle = IsaacGripperHandle(
        FakeGripperSim(), art, "finger_joint", 0.0, math.radians(47), math.radians(2), "/g"
    )
    kps, kds = art.controller.get_gains()
    assert kps[:7] == [1000.0] * 7  # arm joints and finger_joint untouched
    assert kps[7:] == [0.0] * 5 and kds[7:] == [PASSIVE_JOINT_DAMPING] * 5

    handle.set_jaw(0.5)
    assert art.actions[-1].joint_indices == [6]
    assert art.actions[-1].joint_positions == pytest.approx([0.5])


def _jaw_handle():
    import math

    from isaac_module.sim_manager import IsaacGripperHandle

    art = FakeArmArticulation(ARM_AND_GRIPPER_DOFS)
    handle = IsaacGripperHandle(
        FakeGripperSim(), art, "finger_joint", 0.0, math.radians(47), math.radians(2), "/g"
    )
    return handle, art


def test_jaw_vibrating_on_a_block_stalls_and_holds():
    """GPU run 23: at the contact angle the jaw oscillates +/-1 deg at 90 deg/s;
    velocity-gated predicates never fire. The progress window must."""
    import math

    from isaac_module.sim_manager import GRIPPER_HOLDING_STEPS

    handle, art = _jaw_handle()
    handle.set_jaw(math.radians(47))
    contact = math.radians(14.6)
    for step in range(GRIPPER_HOLDING_STEPS + 1):
        art.positions[6] = contact + math.radians(1.0 if step % 2 else -1.0)
        handle.is_moving()  # each poll advances the progress window
    assert handle.is_moving() is False
    assert handle.is_holding() is True


def test_jaw_closing_freely_reports_moving_then_not_holding():
    import math

    from isaac_module.sim_manager import GRIPPER_HOLDING_STEPS

    handle, art = _jaw_handle()
    handle.set_jaw(math.radians(47))
    for angle_deg in (5.0, 15.0, 25.0, 35.0):
        art.positions[6] = math.radians(angle_deg)
        assert handle.is_moving() is True  # gap keeps improving
    art.positions[6] = math.radians(47.0)
    for _ in range(GRIPPER_HOLDING_STEPS + 1):
        assert handle.is_moving() is False  # within tolerance of the target
    assert handle.is_holding() is False  # fully closed = nothing between the jaws


def _latched_hold():
    """A handle driven to a latched hold: closing onto a block at 14.6 deg,
    polled fast (grab()'s loop) until the stall fires."""
    import math

    from isaac_module.sim_manager import GRIPPER_HOLDING_STEPS

    handle, art = _jaw_handle()
    handle.set_jaw(math.radians(47))
    art.positions[6] = math.radians(14.6)
    for _ in range(GRIPPER_HOLDING_STEPS):
        handle.is_moving()  # fast poll: no progress, the stall fires and latches
    assert handle.is_holding() is True
    return handle, art


def test_jaw_hold_latches_through_slow_creep_samples():
    """GPU run 25: grab()'s 120 Hz poll saw the stall, but the jaw kept
    creeping ~0.6 deg/s squeezing the lifted block, so a 1 Hz client sample
    reset the progress window every read and never re-accumulated it. The
    latch must carry the hold across such samples."""
    import math

    handle, art = _latched_hold()
    for sample in range(1, 6):  # slow samples, the jaw creeping 0.6 deg per read
        art.positions[6] = math.radians(14.6 + 0.6 * sample)
        assert handle.is_holding() is True  # progress resets the window, not the latch
        assert handle.is_moving() is False


def test_jaw_hold_latch_clears_when_the_jaw_reaches_its_target():
    import math

    handle, art = _latched_hold()
    art.positions[6] = math.radians(47.0)  # the block escaped: full close
    assert handle.is_holding() is False
    assert handle.is_moving() is False


def test_set_jaw_clears_the_hold_latch():
    import math

    handle, art = _latched_hold()
    handle.set_jaw(0.0)  # open: a new command starts a fresh verdict
    art.positions[6] = math.radians(14.6)
    assert handle.is_holding() is False


def test_stop_while_holding_keeps_the_grasp():
    """isaac-try M0: viam-server calls Stop on the gripper when a session
    lapses (every CLI `part run` exit). A stop that re-targeted the jaw to
    its measured angle zeroed the squeeze and cleared the hold."""
    handle, art = _latched_hold()
    before = len(art.actions)
    handle.stop()
    assert len(art.actions) == before  # no new drive target: the grasp stays
    assert handle.is_holding() is True
    assert handle.is_moving() is False


def test_stop_mid_travel_freezes_at_the_measured_angle():
    import math

    handle, art = _jaw_handle()
    handle.set_jaw(math.radians(47))
    art.positions[6] = math.radians(20.0)
    handle.stop()
    assert art.actions[-1].joint_positions == pytest.approx([math.radians(20.0)])
    assert handle.is_holding() is False


class FakeEffortArticulation(FakeArmArticulation):
    def __init__(self, dof_names: list[str]) -> None:
        super().__init__(dof_names)
        self.efforts = [0.0] * len(dof_names)

    def get_measured_joint_efforts(self, joint_indices=None):
        indices = joint_indices if joint_indices is not None else range(len(self.efforts))
        return [self.efforts[i] for i in indices]


def _effort_handle(effort_min: float = 0.5):
    import math

    from isaac_module.sim_manager import IsaacGripperHandle

    art = FakeEffortArticulation(ARM_AND_GRIPPER_DOFS)
    handle = IsaacGripperHandle(
        FakeGripperSim(),
        art,
        "finger_joint",
        0.0,
        math.radians(47),
        math.radians(2),
        "/g",
        holding_effort_min=effort_min,
    )
    return handle, art


def test_effort_predicate_reports_holding_from_contact_not_the_stall_window():
    import math

    handle, art = _effort_handle()
    handle.set_jaw(math.radians(47))
    art.positions[6] = math.radians(14.6)
    art.efforts[6] = -1.2  # sign is direction; magnitude counts
    assert handle.is_holding() is True  # first poll, no stall window needed
    assert handle.finger_effort() == pytest.approx(1.2)
    art.efforts[6] = 0.0  # the block slipped out
    assert handle.is_holding() is False


def test_effort_predicate_survives_stop_and_a_cleared_target():
    import math

    handle, art = _effort_handle()
    handle.set_jaw(math.radians(47))
    art.positions[6] = math.radians(14.6)
    art.efforts[6] = 0.9
    handle.stop()
    assert handle.is_holding() is True
    handle._target = None  # even with no commanded target on record
    assert handle.is_holding() is True


def test_effort_predicate_ignores_fingers_pressing_on_each_other_when_closed():
    import math

    handle, art = _effort_handle()
    handle.set_jaw(math.radians(47))
    art.positions[6] = math.radians(46.5)
    art.efforts[6] = 3.0  # fully closed on nothing, fingertips touching
    assert handle.is_holding() is False


def test_effort_predicate_falls_back_to_the_stall_window_when_unreadable():
    """A FakeArmArticulation has no get_measured_joint_efforts: the effort
    threshold is configured but the stall predicate still decides."""
    import math

    from isaac_module.sim_manager import GRIPPER_HOLDING_STEPS, IsaacGripperHandle

    art = FakeArmArticulation(ARM_AND_GRIPPER_DOFS)
    handle = IsaacGripperHandle(
        FakeGripperSim(), art, "finger_joint", 0.0, math.radians(47), math.radians(2), "/g",
        holding_effort_min=0.5,
    )
    assert handle.finger_effort() is None
    handle.set_jaw(math.radians(47))
    art.positions[6] = math.radians(14.6)
    for _ in range(GRIPPER_HOLDING_STEPS):
        handle.is_moving()
    assert handle.is_holding() is True
