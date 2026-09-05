"""Phase-2 GPU acceptance checklist for the Isaac Sim wrist camera.

Connects to a running Viam machine (the module running on the Isaac GPU
box) and walks the six phase-2 GPU checklist items from
`.claude/plans/pick-place-mvp/phase-2-see-red-block.md`, printing PASS/FAIL/SKIP
and raw numbers for each so the results can be pasted back into that plan's
Notes.

Depends only on the stdlib, viam-sdk and Pillow: it runs on a laptop against a
remote machine, not inside the module process.

Usage::

    python examples/gpu_checklist_camera.py --address <machine-address> \\
        --api-key <key> --api-key-id <key-id>
"""

from __future__ import annotations

import argparse
import asyncio
import math
from dataclasses import dataclass
from io import BytesIO

from PIL import Image
from viam.components.camera import Camera
from viam.components.generic import Generic
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

# ----------------------------------------------------------------------
# pure helpers - unit-testable without a robot (see tests/test_gpu_checklist_camera.py)
# ----------------------------------------------------------------------

RgbPixel = tuple[int, int, int]
Bbox = tuple[int, int, int, int]  # (x0, y0, x1, y1), exclusive upper bounds
Vec3 = tuple[float, float, float]

MIN_RED_PIXELS = 50
MM_PER_M = 1000.0


def parse_xyz(text: str) -> Vec3:
    """Parse "x,y,z" into a 3-tuple of floats."""
    x, y, z = (float(v) for v in text.split(","))
    return (x, y, z)


def is_red_pixel(pixel: RgbPixel, threshold: float) -> bool:
    """A pixel is "red" when r is at least 100 and both g and b are at most
    `threshold` fractions of r."""
    r, g, b = pixel
    return r >= 100 and g <= r * threshold and b <= r * threshold


def red_bbox(
    pixels: list[RgbPixel] | tuple[RgbPixel, ...], width: int, height: int, threshold: float
) -> Bbox | None:
    """Bounding box (x0, y0, x1, y1), exclusive upper bounds, of red pixels in
    a row-major `pixels` sequence. None when fewer than MIN_RED_PIXELS match."""
    min_x, min_y = width, height
    max_x, max_y = -1, -1
    count = 0
    for index, pixel in enumerate(pixels):
        if not is_red_pixel(pixel, threshold):
            continue
        count += 1
        x, y = index % width, index // width
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x), max(max_y, y)
    if count < MIN_RED_PIXELS:
        return None
    return (min_x, min_y, max_x + 1, max_y + 1)


MIN_RISE_MM = 30
"""A pixel at least this much closer than the floor counts as "on the block"."""


def raised_bbox(
    rows: list[list[int]] | tuple[tuple[int, ...], ...], min_rise_mm: int = MIN_RISE_MM
) -> Bbox | None:
    """Bounding box (x0, y0, x1, y1), exclusive upper bounds, of the pixels that
    sit at least `min_rise_mm` closer to the camera than the floor (taken as the
    deepest valid reading). None when fewer than MIN_RED_PIXELS qualify or there
    is no valid depth. Locates the block from geometry, independent of colour."""
    valid = [d for row in rows for d in row if d > 0]
    if not valid:
        return None
    floor_mm = max(valid)
    min_x, min_y = len(rows[0]), len(rows)
    max_x, max_y = -1, -1
    count = 0
    for y, row in enumerate(rows):
        for x, depth_mm in enumerate(row):
            if depth_mm <= 0 or depth_mm > floor_mm - min_rise_mm:
                continue
            count += 1
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
    if count < MIN_RED_PIXELS:
        return None
    return (min_x, min_y, max_x + 1, max_y + 1)


def depth_stats(rows: list[list[int]] | tuple[tuple[int, ...], ...]) -> tuple[int, int, float]:
    """(min non-zero mm, max mm, fraction of zero pixels). (0, 0, 1.0) when
    every pixel is zero (or there are no pixels)."""
    total = 0
    zero_count = 0
    non_zero_values: list[int] = []
    max_value = 0
    for row in rows:
        for value in row:
            total += 1
            if value == 0:
                zero_count += 1
            else:
                non_zero_values.append(value)
            max_value = max(max_value, value)
    if total == 0 or not non_zero_values:
        return (0, 0, 1.0)
    return (min(non_zero_values), max_value, zero_count / total)


def region_center(bbox: Bbox) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def pose_delta_mm(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def is_reddish(mean_rgb: list[int] | tuple[int, int, int]) -> bool:
    r, g, b = mean_rgb
    return r > g and r > b and r >= 100


def verdict(name: str, ok: bool, detail: str) -> str:
    """Format one checklist line: "[PASS|FAIL] name: detail"."""
    status = "PASS" if ok else "FAIL"
    return f"[{status}] {name}: {detail}"


def skip(name: str, detail: str) -> str:
    """Format one checklist line: "[SKIP] name: detail"."""
    return f"[SKIP] {name}: {detail}"


# ----------------------------------------------------------------------
# checklist items
# ----------------------------------------------------------------------

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 2000
POSE_TOLERANCE_MM = 10.0


@dataclass
class Args:
    address: str
    api_key: str | None
    api_key_id: str | None
    camera: str
    world: str
    segmenter: str | None
    block_xyz_m: Vec3
    sample_region: Bbox | None
    red_threshold: float


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-id")
    parser.add_argument("--camera", default="wrist-cam")
    parser.add_argument("--world", default="sim-world")
    parser.add_argument("--segmenter")
    parser.add_argument("--block-xyz-m", default="0.60,0.10,0.7755")
    parser.add_argument("--sample-region")
    parser.add_argument("--red-threshold", type=float, default=0.5)
    ns = parser.parse_args()
    sample_region: Bbox | None = None
    if ns.sample_region:
        x0, y0, x1, y1 = (int(v) for v in ns.sample_region.split(","))
        sample_region = (x0, y0, x1, y1)
    return Args(
        address=ns.address,
        api_key=ns.api_key,
        api_key_id=ns.api_key_id,
        camera=ns.camera,
        world=ns.world,
        segmenter=ns.segmenter,
        block_xyz_m=parse_xyz(ns.block_xyz_m),
        sample_region=sample_region,
        red_threshold=ns.red_threshold,
    )


async def _connect(args: Args) -> RobotClient:
    if args.api_key and args.api_key_id:
        opts = RobotClient.Options.with_api_key(api_key=args.api_key, api_key_id=args.api_key_id)
    else:
        opts = RobotClient.Options()
    return await RobotClient.at_address(args.address, opts)


async def _fetch_depth_rows(cam: Camera) -> list[list[int]]:
    """Decoded depth rows (mm), or [] when the camera serves no depth image."""
    try:
        imgs, _ = await cam.get_images(filter_source_names=["depth"])
    except Exception:  # noqa: BLE001 - item 1 reports depth problems itself
        return []
    if len(imgs) != 1:
        return []
    return ViamImage(imgs[0].data, CameraMimeType.VIAM_RAW_DEPTH).bytes_to_depth_array()


async def _check_sample_color(
    cam: Camera, args: Args, depth_rows: list[list[int]]
) -> tuple[str, bool, Bbox | None]:
    imgs, _ = await cam.get_images(filter_source_names=["color"])
    if len(imgs) != 1:
        line = verdict(
            "3. sample_color on the block", False, f"expected 1 color image, got {len(imgs)}"
        )
        print(line)
        return line, False, None

    image = Image.open(BytesIO(imgs[0].data)).convert("RGB")
    width, height = image.size
    bbox: Bbox | None
    if args.sample_region is not None:
        bbox, located_by = args.sample_region, "--sample-region"
    elif (depth_bbox := raised_bbox(depth_rows)) is not None:
        bbox, located_by = depth_bbox, f"depth (pixels >= {MIN_RISE_MM} mm above the floor)"
    else:
        bbox = red_bbox(list(image.getdata()), width, height, args.red_threshold)
        located_by = "colour (red threshold)"
    if bbox is None:
        line = verdict(
            "3. sample_color on the block",
            False,
            "block not found: no raised region in depth and no red region in colour",
        )
        print(line)
        return line, False, None
    print(f"  block located by: {located_by}")

    result = await cam.do_command({"command": "sample_color", "region": list(bbox)})
    srgb_hex = result["srgb_hex"]
    mean_rgb = [int(v) for v in result["mean_rgb"]]  # type: ignore[union-attr]
    ok = is_reddish(mean_rgb)
    print(f"  region (bbox): {bbox}")
    print(f"  srgb_hex: {srgb_hex}  mean_rgb: {mean_rgb}")
    print(f"  detect_color = {srgb_hex}")
    if not ok:
        print(
            "  FINDING (OQ-8): the block region does not read as red under the shipping lighting; "
            "record this hex and revisit the lighting/material before trusting color_detector"
        )
    line = verdict(
        "3. sample_color on the block (OQ-8, sets detect_color/W31)", ok, f"mean_rgb={mean_rgb}"
    )
    print(line)
    return line, ok, bbox


async def _check_depth(cam: Camera, bbox: Bbox | None) -> tuple[str, bool]:
    props = await cam.get_properties()
    intr = props.intrinsic_parameters
    print(
        f"  supports_pcd={props.supports_pcd} fx={intr.focal_x_px} fy={intr.focal_y_px} "
        f"cx={intr.center_x_px} cy={intr.center_y_px} w={intr.width_px} h={intr.height_px}"
    )
    print(f"  mime_types: {list(props.mime_types)}")

    imgs, _ = await cam.get_images(filter_source_names=["depth"])
    got_one_depth_image = len(imgs) == 1
    rows: list[list[int]] = []
    if got_one_depth_image:
        rows = ViamImage(imgs[0].data, CameraMimeType.VIAM_RAW_DEPTH).bytes_to_depth_array()
    min_mm, max_mm, zero_fraction = depth_stats(rows)
    print(
        f"  depth stats: min_non_zero_mm={min_mm} max_mm={max_mm} zero_fraction={zero_fraction:.4f}"
    )

    check_mm = min_mm
    if bbox is not None and rows:
        cx, cy = region_center(bbox)
        cy = min(cy, len(rows) - 1)
        cx = min(cx, len(rows[0]) - 1)
        check_mm = rows[cy][cx]
        print(f"  depth at block-region centre ({cx},{cy}): {check_mm} mm")

    ok = (
        props.supports_pcd
        and intr.focal_x_px > 0
        and got_one_depth_image
        and DEPTH_MIN_MM <= check_mm <= DEPTH_MAX_MM
    )
    print(
        "  reminder: check the module logs for "
        f"'camera {cam.name} clipping range (0.05, 10.0)' (OQ-11)"
    )
    line = verdict(
        "1. depth image sees the block (OQ-11)",
        ok,
        f"supports_pcd={props.supports_pcd} fx={intr.focal_x_px} depth_mm={check_mm}",
    )
    print(line)
    return line, ok


async def _check_get_images_filter(cam: Camera) -> tuple[str, bool]:
    no_filter_imgs, _ = await cam.get_images()
    no_filter_names = [img.name for img in no_filter_imgs]
    no_filter_ok = no_filter_names == ["color", "depth"]
    print(f"  no filter -> {no_filter_names}")

    depth_only_imgs, _ = await cam.get_images(filter_source_names=["depth"])
    depth_only_ok = [img.name for img in depth_only_imgs] == ["depth"]
    print(f"  filter=['depth'] -> {[img.name for img in depth_only_imgs]}")

    color_only_imgs, _ = await cam.get_images(filter_source_names=["color"])
    color_only_ok = [img.name for img in color_only_imgs] == ["color"]
    print(f"  filter=['color'] -> {[img.name for img in color_only_imgs]}")

    bogus_imgs, _ = await cam.get_images(filter_source_names=["bogus"])
    bogus_ok = len(bogus_imgs) == 0
    print(f"  filter=['bogus'] -> {[img.name for img in bogus_imgs]}")

    ok = no_filter_ok and depth_only_ok and color_only_ok and bogus_ok
    print(
        f"  reminder: check the module logs for 'camera {cam.name} get_images "
        "filter_source_names=' (OQ-16)"
    )
    line = verdict("4. GetImages honours filter_source_names (OQ-16)", ok, f"details ok={ok}")
    print(line)
    return line, ok


async def _check_segmenter(machine: RobotClient, args: Args) -> tuple[str, bool]:
    name = "2. detections-to-segments block pose (W32)"
    if not args.segmenter:
        line = skip(
            name,
            "no --segmenter given; add a vision service config like W32 "
            "(detections-to-segments over the wrist camera) and re-run with --segmenter",
        )
        print(line)
        return line, True

    vision = VisionClient.from_robot(machine, args.segmenter)
    objs = await vision.get_object_point_clouds(args.camera)
    if not objs:
        line = verdict(name, False, "get_object_point_clouds returned no objects")
        print(line)
        return line, False

    center = objs[0].geometries.geometries[0].center
    camera_pose = (center.x, center.y, center.z)
    pif = PoseInFrame(
        reference_frame=args.camera,
        pose=Pose(x=center.x, y=center.y, z=center.z, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0),
    )
    world_pif = await machine.transform_pose(pif, "world")
    world_pose = (world_pif.pose.x, world_pif.pose.y, world_pif.pose.z)
    bx, by, bz = args.block_xyz_m
    expected_mm: Vec3 = (bx * MM_PER_M, by * MM_PER_M, bz * MM_PER_M)
    delta_mm = pose_delta_mm(world_pose, expected_mm)

    print(f"  camera-frame centre (mm): {camera_pose}")
    print(f"  world-frame centre (mm): {world_pose}")
    print(f"  expected W23 (mm): {expected_mm}")
    ok = delta_mm <= POSE_TOLERANCE_MM
    line = verdict(name, ok, f"delta {delta_mm:.3f} mm")
    print(line)
    return line, ok


async def _check_isaac_version(machine: RobotClient, args: Args) -> tuple[str, bool]:
    world = Generic.from_robot(machine, args.world)
    status = await world.do_command({"command": "status"})
    isaac_version = status.get("isaac_version")
    print(f"  isaac_version: {isaac_version!r}")
    print(f"  lighting: {status.get('lighting')!r}")
    ok = isinstance(isaac_version, str) and len(isaac_version) > 0
    line = verdict(
        "6. compat.isaac_version() on the real install (OQ-14)",
        ok,
        f"isaac_version={isaac_version!r}",
    )
    print(line)
    return line, ok


def _check_isaac_4_5_manual() -> tuple[str, bool]:
    line = skip(
        "5. depth fresh after reset + vFOV matches (CAM-4, CAM-17)",
        "needs an Isaac 4.5 box; manually check depth after `reset` is fresh "
        "and that focal_y_px == focal_x_px",
    )
    print(line)
    return line, True


async def main() -> None:
    args = _parse_args()
    machine = await _connect(args)
    try:
        cam = Camera.from_robot(machine, args.camera)

        results: list[tuple[str, bool]] = []
        bbox: Bbox | None = None

        print("\n-- item 6: compat.isaac_version() --")
        try:
            results.append(await _check_isaac_version(machine, args))
        except Exception as exc:  # noqa: BLE001 - never crash the checklist run
            line = verdict(
                "6. compat.isaac_version() on the real install (OQ-14)",
                False,
                f"exception: {exc!r}",
            )
            print(line)
            results.append((line, False))

        print("\n-- item 3: sample_color on the block --")
        try:
            line, ok, bbox = await _check_sample_color(cam, args, await _fetch_depth_rows(cam))
            results.append((line, ok))
        except Exception as exc:  # noqa: BLE001 - never crash the checklist run
            line = verdict("3. sample_color on the block", False, f"exception: {exc!r}")
            print(line)
            results.append((line, False))

        print("\n-- item 1: depth image sees the block --")
        try:
            results.append(await _check_depth(cam, bbox))
        except Exception as exc:  # noqa: BLE001 - never crash the checklist run
            line = verdict("1. depth image sees the block (OQ-11)", False, f"exception: {exc!r}")
            print(line)
            results.append((line, False))

        print("\n-- item 4: GetImages filter_source_names --")
        try:
            results.append(await _check_get_images_filter(cam))
        except Exception as exc:  # noqa: BLE001 - never crash the checklist run
            line = verdict(
                "4. GetImages honours filter_source_names (OQ-16)", False, f"exception: {exc!r}"
            )
            print(line)
            results.append((line, False))

        print("\n-- item 2: detections-to-segments block pose --")
        try:
            results.append(await _check_segmenter(machine, args))
        except Exception as exc:  # noqa: BLE001 - never crash the checklist run
            line = verdict(
                "2. detections-to-segments block pose (W32)", False, f"exception: {exc!r}"
            )
            print(line)
            results.append((line, False))

        print("\n-- item 5: Isaac 4.5 manual checks --")
        results.append(_check_isaac_4_5_manual())

        print("\n== summary ==")
        for line, _ in results:
            print(line)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
