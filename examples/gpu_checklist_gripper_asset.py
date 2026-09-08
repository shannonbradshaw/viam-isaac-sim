"""Phase-3 GPU checklist probes for the Robotiq 2F-85 gripper asset.

Item 1 (FINDINGS R-4 / OQ-4): does ``Robotiq_2F_85_edit.usd`` compose with pad
prims that carry ``PhysicsCollisionAPI`` on this Isaac install, and which
asset hosts do its references pull from? The module refuses to attach the
gripper when the pads have no collision, so this probe answers the question
without viam-server in the loop.

Runs INSIDE Isaac Sim's python (it boots a headless ``SimulationApp`` to get
the asset resolver and ``omni.client``), e.g. on the GPU machine::

    ~/isaacsim/python.sh examples/gpu_checklist_gripper_asset.py            # item 1, default asset
    ~/isaacsim/python.sh examples/gpu_checklist_gripper_asset.py --usd omniverse://.../Robotiq_2F_85_edit.usd

Prints one PASS/FAIL line per item plus the raw observations to paste into
``.claude/plans/pick-place-mvp/phase-3-grasp-and-lift.md`` Notes.

The pure helpers at the top take plain records so they are unit-tested on a
laptop (see tests/test_gpu_checklist_gripper_asset.py).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEFAULT_ASSET_REL = "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
UNRESOLVABLE_ASSET_HOST = "isaac-dev"
# the 5.0 asset has no `*_pad` links; its fingertip geometry is the
# `..._fingertipsstep_..` mesh under `left/right_inner_finger`
PAD_NAME_FRAGMENTS = ("pad", "fingertip", "inner_finger")
SAMPLE_PRIM_PATHS = 60


@dataclass(frozen=True)
class PrimRecord:
    """What the probe reads off one prim: its path, whether it carries
    ``PhysicsCollisionAPI`` itself, and the reference/payload asset paths
    authored on it."""

    path: str
    has_collision_api: bool
    asset_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class PadReport:
    path: str
    has_collision_on_self: bool
    has_collision_in_subtree: bool


def is_pad(path: str) -> bool:
    """A prim is a pad when its LAST path segment names pad or fingertip
    geometry, or an inner-finger link (the 5.0 asset has no ``*_pad`` links)."""
    name = path.rsplit("/", 1)[-1].lower()
    return any(fragment in name for fragment in PAD_NAME_FRAGMENTS)


def is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def pad_reports(records: Iterable[PrimRecord]) -> list[PadReport]:
    """One report per pad prim; a pad has collision "in subtree" when it or
    any prim under it carries the API (the asset nests collision meshes
    under the link prim)."""
    records = list(records)
    reports: list[PadReport] = []
    for record in records:
        if not is_pad(record.path):
            continue
        in_subtree = any(
            other.has_collision_api for other in records if is_under(other.path, record.path)
        )
        reports.append(PadReport(record.path, record.has_collision_api, in_subtree))
    return reports


def unresolvable_references(records: Iterable[PrimRecord]) -> list[str]:
    """Reference/payload asset paths that point at the host that is NXDOMAIN
    outside NVIDIA (R-4)."""
    return [
        asset_path
        for record in records
        for asset_path in record.asset_paths
        if UNRESOLVABLE_ASSET_HOST in asset_path
    ]


def item1_verdict(reports: Sequence[PadReport], unresolved: Sequence[str]) -> tuple[str, str]:
    """PASS when at least one pad has collision in its subtree; FAIL with the
    most likely cause otherwise."""
    if any(report.has_collision_in_subtree for report in reports):
        return (
            "PASS",
            f"{len(reports)} pad prim(s), collision present; unresolved refs: {len(unresolved)}",
        )
    if not reports:
        return (
            "FAIL",
            "no pad prims composed at all - the asset (or its sub-references) did not resolve",
        )
    if unresolved:
        return "FAIL", (
            f"{len(reports)} pad prim(s) without collision; {len(unresolved)} reference(s) on "
            f"{UNRESOLVABLE_ASSET_HOST} - R-4 confirmed, fall back per R-2"
        )
    return "FAIL", (
        f"{len(reports)} pad prim(s) without collision and no {UNRESOLVABLE_ASSET_HOST} "
        "references - the asset authors no collision on the pads on this release"
    )


# ----------------------------------------------------------------------
# Isaac-side collection (only runs inside Isaac's python)
# ----------------------------------------------------------------------


def _prim_range(usd, root_prim):
    """Traverse including instance proxies: Isaac assets mark link meshes
    instanceable, and a default PrimRange stops at each instance."""
    return usd.PrimRange(root_prim, usd.TraverseInstanceProxies())


def _collect_records(stage, usd, usd_physics) -> list[PrimRecord]:
    records: list[PrimRecord] = []
    for prim in _prim_range(usd, stage.GetPseudoRoot()):
        asset_paths: list[str] = []
        for key in ("references", "payload"):
            try:
                list_op = prim.GetMetadata(key)
                items = list_op.GetAddedOrExplicitItems() if list_op is not None else []
            except Exception:
                items = []
            asset_paths.extend(
                str(item.assetPath) for item in items if getattr(item, "assetPath", "")
            )
        records.append(
            PrimRecord(
                path=str(prim.GetPath()),
                has_collision_api=bool(prim.HasAPI(usd_physics.CollisionAPI)),
                asset_paths=tuple(asset_paths),
            )
        )
    return records


def run_item1(usd_path: str | None) -> int:
    from isaacsim import SimulationApp  # only importable inside Isaac Sim's python

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdPhysics

        if usd_path is None:
            try:
                from isaacsim.storage.native import get_assets_root_path
            except ImportError:
                from omni.isaac.core.utils.nucleus import get_assets_root_path
            root = get_assets_root_path()
            if root is None:
                print("item 1: FAIL - could not reach the isaac assets server")
                return 2
            usd_path = root + DEFAULT_ASSET_REL
        print(f"asset: {usd_path}")
        stage = Usd.Stage.Open(usd_path)
        if stage is None:
            print("item 1: FAIL - Usd.Stage.Open returned None (asset path did not resolve)")
            return 2
        layer = stage.GetRootLayer()
        root_prims = [str(spec.path) for spec in layer.rootPrims]
        print(f"  layer defaultPrim={layer.defaultPrim!r} root prims={root_prims}")
        records = _collect_records(stage, Usd, UsdPhysics)
        plain_count = sum(1 for _ in Usd.PrimRange(stage.GetPseudoRoot()))
        print(
            f"  {len(records)} prim(s) incl. instance proxies ({plain_count} without); first "
            f"{SAMPLE_PRIM_PATHS}:"
        )
        for record in records[1 : SAMPLE_PRIM_PATHS + 1]:
            print(f"    {record.path}{'  [collision]' if record.has_collision_api else ''}")
        reports = pad_reports(records)
        unresolved = unresolvable_references(records)
        verdict, detail = item1_verdict(reports, unresolved)
        print(f"item 1 (R-4/OQ-4 pad collision): {verdict} - {detail}")
        for report in reports:
            print(
                f"  pad {report.path}: collision on self={report.has_collision_on_self} "
                f"in subtree={report.has_collision_in_subtree}"
            )
        all_refs = sorted({p for r in records for p in r.asset_paths})
        print(f"  {len(all_refs)} distinct reference/payload asset path(s):")
        for path in all_refs:
            flag = "  <-- unresolvable host" if UNRESOLVABLE_ASSET_HOST in path else ""
            print(f"    {path}{flag}")
        return 0 if verdict == "PASS" else 1
    finally:
        app.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--usd", default=None, help="asset to probe (default: the 2F-85 under the assets root)"
    )
    args = parser.parse_args(argv)
    return run_item1(args.usd)


if __name__ == "__main__":
    sys.exit(main())
