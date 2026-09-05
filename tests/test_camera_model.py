"""Contract tests for IsaacCamera against the mock sim backend (FINDINGS
CAM-5, CAM-6, CAM-8, CAM-15, CAM-16, CAM-18)."""

import asyncio

import numpy as np
import pytest
from grpclib import Status
from viam.errors import MethodNotImplementedError, ViamGRPCError
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from isaac_module.camera_base import NoFrameYetError
from isaac_module.encoding import intrinsics_from_fov
from isaac_module.errors import SimTimeoutError
from isaac_module.mock_camera import MOCK_RED_BLOCK_CENTER_M, RED_BLOCK_RGB
from isaac_module.models.camera import IsaacCamera


def _config(name: str, attrs: dict) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _make_camera(world, name: str, attrs: dict) -> IsaacCamera:
    full_attrs = {"world": "sim-world", **attrs}
    return IsaacCamera.new(_config(name, full_attrs), {})


def test_properties_with_depth(world):
    cam = _make_camera(world, "cam-props-depth", {"depth": True})

    async def scenario():
        props = await cam.get_properties()
        assert props.supports_pcd is True

        k = intrinsics_from_fov(848, 480, 90.5)
        ip = props.intrinsic_parameters
        assert ip.focal_x_px == pytest.approx(k.fx)
        assert ip.focal_y_px == pytest.approx(k.fy)
        assert ip.focal_x_px == pytest.approx(ip.focal_y_px)
        assert ip.center_x_px == pytest.approx(424)
        assert ip.center_y_px == pytest.approx(240)
        assert ip.width_px == 848
        assert ip.height_px == 480

        assert "image/png" in props.mime_types
        assert "image/vnd.viam.dep" in props.mime_types
        assert "pointcloud/pcd" in props.mime_types

    asyncio.run(scenario())


def test_properties_without_depth(world):
    cam = _make_camera(world, "cam-props-nodepth", {})

    async def scenario():
        props = await cam.get_properties()
        assert props.supports_pcd is False
        assert "image/vnd.viam.dep" not in props.mime_types
        assert "pointcloud/pcd" not in props.mime_types

        with pytest.raises(MethodNotImplementedError):
            await cam.get_point_cloud()

    asyncio.run(scenario())


def test_get_images_names_and_order(world):
    cam = _make_camera(world, "cam-images-order", {"depth": True})

    async def scenario():
        images, _meta = await cam.get_images()
        assert [img.name for img in images] == ["color", "depth"]
        assert images[0].mime_type == CameraMimeType.PNG
        assert images[1].mime_type == CameraMimeType.VIAM_RAW_DEPTH

        from io import BytesIO

        from PIL import Image

        decoded = Image.open(BytesIO(images[0].data))
        assert decoded.size == (848, 480)

    asyncio.run(scenario())


def test_get_images_filter_source_names(world):
    cam = _make_camera(world, "cam-images-filter", {"depth": True})

    async def scenario():
        images, _ = await cam.get_images(filter_source_names=["depth"])
        assert [img.name for img in images] == ["depth"]

        images, _ = await cam.get_images(filter_source_names=["color"])
        assert [img.name for img in images] == ["color"]

        images, _ = await cam.get_images(filter_source_names=[])
        assert [img.name for img in images] == ["color", "depth"]

        images, _ = await cam.get_images(filter_source_names=None)
        assert [img.name for img in images] == ["color", "depth"]

        images, _ = await cam.get_images(filter_source_names=["bogus"])
        assert images == []

    asyncio.run(scenario())


def test_depth_image_round_trip(world):
    cam = _make_camera(world, "cam-depth-roundtrip", {"depth": True})

    async def scenario():
        images, _ = await cam.get_images()
        depth_named = images[1]
        vi = ViamImage(depth_named.data, CameraMimeType.VIAM_RAW_DEPTH)
        arr = vi.bytes_to_depth_array()
        assert len(arr) == 480
        assert len(arr[0]) == 848

        # block centre pixel: derive from the handle's intrinsics + mock's
        # known offsets (BLOCK_LEFT/RIGHT/TOP/BOTTOM_OFFSET_PX in mock_camera)
        # rather than hardcoding indices twice; use the frame's rgb mask.
        frame = cam._h().get_frame()
        mask = np.all(frame.rgb == np.array(RED_BLOCK_RGB), axis=-1)
        ys, xs = np.nonzero(mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        assert arr[cy][cx] == 400

        # NaN-band pixel (top row) reads 0
        assert arr[0][0] == 0

    asyncio.run(scenario())


def test_point_cloud_red_cluster_centroid(world):
    cam = _make_camera(world, "cam-pcd", {"depth": True})

    async def scenario():
        data, mime = await cam.get_point_cloud()
        assert mime == "pointcloud/pcd"

        header_end = data.index(b"DATA binary\n") + len(b"DATA binary\n")
        header = data[:header_end].decode("ascii")
        lines = {ln.split(" ")[0]: ln for ln in header.splitlines()}
        points_line = lines["POINTS"]
        n_points = int(points_line.split(" ")[1])
        assert n_points == 848 * 480 - 48 * 848

        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        payload = np.frombuffer(data[header_end:], dtype=dtype)
        assert len(payload) == n_points

        r, g, b = RED_BLOCK_RGB
        packed = (r << 16) | (g << 8) | b
        red_points = payload[payload["rgb"] == packed]
        assert len(red_points) == 8000

        # a centred block would read (0, 0, 0.40); this is deliberately
        # off-centre so a sign/axis bug in back-projection is caught.
        centroid = (
            float(np.mean(red_points["x"].astype(np.float64))),
            float(np.mean(red_points["y"].astype(np.float64))),
            float(np.mean(red_points["z"].astype(np.float64))),
        )
        for actual, expected in zip(centroid, MOCK_RED_BLOCK_CENTER_M, strict=True):
            assert actual == pytest.approx(expected, abs=1e-3)

    asyncio.run(scenario())


def test_jpeg_image_format(world):
    cam = _make_camera(world, "cam-jpeg", {"image_format": "jpeg"})

    async def scenario():
        images, _ = await cam.get_images()
        assert images[0].mime_type == CameraMimeType.JPEG

    asyncio.run(scenario())


def test_do_command_sample_color(world):
    cam = _make_camera(world, "cam-sample-color", {})

    async def scenario():
        frame = cam._h().get_frame()
        mask = np.all(frame.rgb == np.array(RED_BLOCK_RGB), axis=-1)
        ys, xs = np.nonzero(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        result = await cam.do_command({"command": "sample_color", "region": [x0, y0, x1, y1]})
        assert result["srgb_hex"] == "#E02020"
        assert result["mean_rgb"] == [224, 32, 32]

        with pytest.raises(ValueError):
            await cam.do_command({"command": "sample_color", "region": [0, 0, 10000, 10000]})

        with pytest.raises(ValueError):
            await cam.do_command({"command": "bogus"})

    asyncio.run(scenario())


def test_error_mapping(world, monkeypatch):
    cam = _make_camera(world, "cam-error-mapping", {})
    handle = cam._h()

    def raise_no_frame():
        raise NoFrameYetError("no frame yet")

    monkeypatch.setattr(handle, "get_frame", raise_no_frame)
    with pytest.raises(ViamGRPCError) as excinfo:
        asyncio.run(cam.get_images())
    assert excinfo.value.grpc_code == Status.FAILED_PRECONDITION

    def raise_timeout():
        raise SimTimeoutError("timed out")

    monkeypatch.setattr(handle, "get_frame", raise_timeout)
    with pytest.raises(ViamGRPCError) as excinfo:
        asyncio.run(cam.get_images())
    assert excinfo.value.grpc_code == Status.DEADLINE_EXCEEDED


def test_validate_config_rejects_bad_attrs():
    with pytest.raises(ValueError, match="width"):
        IsaacCamera.validate_config(_config("c", {"world": "sim-world", "width": 0}))
    with pytest.raises(ValueError, match="clip_near"):
        IsaacCamera.validate_config(
            _config("c", {"world": "sim-world", "clip_near": 5, "clip_far": 1})
        )
    with pytest.raises(ValueError, match="image_format"):
        IsaacCamera.validate_config(_config("c", {"world": "sim-world", "image_format": "bmp"}))
    with pytest.raises(ValueError, match="fov_deg"):
        IsaacCamera.validate_config(_config("c", {"world": "sim-world", "fov_deg": 200}))


def test_validate_config_accepts_integral_float_dimensions():
    # protobuf Struct decodes every number as float (dict_to_struct/struct_to_dict
    # round-trip, as _config below does), so 848 arrives as 848.0 - must be accepted.
    deps, _ = IsaacCamera.validate_config(
        _config("c", {"world": "sim-world", "width": 848, "height": 480})
    )
    assert list(deps) == ["sim-world"]


def test_validate_config_rejects_non_integral_or_bool_dimensions():
    with pytest.raises(ValueError, match="width"):
        IsaacCamera.validate_config(_config("c", {"world": "sim-world", "width": 848.5}))
    with pytest.raises(ValueError, match="width"):
        IsaacCamera.validate_config(_config("c", {"world": "sim-world", "width": True}))


def test_validate_config_accepts_wrist_cam_attrs():
    deps, _ = IsaacCamera.validate_config(
        _config(
            "wrist-cam",
            {"world": "sim-world", "depth": True, "image_format": "png", "frequency": 30},
        )
    )
    assert list(deps) == ["sim-world"]


def test_frame_derived_orientation_is_marked_ros_axes(world):
    """A quat folded in from the Viam frame is ROS-optical (+Z forward); the
    spawn must not read it as world axes (+X forward). GPU phase-4 run 1:
    that mismatch aimed the side camera at the backdrop (7994 mm read)."""
    config = _config("cam-frame-ov", {"world": "sim-world", "depth": True})
    config.frame.parent = "world"
    config.frame.translation.x = 575
    config.frame.translation.y = 650
    config.frame.translation.z = 900
    vector = config.frame.orientation.vector_degrees
    vector.x, vector.y, vector.z, vector.theta = 0.0, -650.0, -150.0, 0.0
    cam = IsaacCamera.new(config, {})
    assert cam._attrs["orientation_axes"] == "ros"
    assert cam._attrs.get("orientation_wxyz") is not None


def test_parent_prim_frame_orientation_is_not_marked(world):
    config = _config(
        "cam-riding", {"world": "sim-world", "depth": True, "parent_prim": "/World/arm/wrist"}
    )
    config.frame.parent = "pick-arm"
    vector = config.frame.orientation.vector_degrees
    vector.x, vector.y, vector.z, vector.theta = 0.0, 0.0, 1.0, 180.0
    cam = IsaacCamera.new(config, {})
    assert "orientation_axes" not in cam._attrs


def test_place_camera_passes_orientation_axes_through():
    from isaac_module.sim_manager import _place_camera

    class FakeCam:
        def __init__(self):
            self.calls = []

        def set_world_pose(self, position, quat, camera_axes):
            self.calls.append((tuple(position), tuple(quat), camera_axes))

    ros_cam = FakeCam()
    _place_camera(
        ros_cam,
        {
            "position": [0.575, 0.65, 0.9],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "orientation_axes": "ros",
        },
    )
    legacy_cam = FakeCam()
    _place_camera(
        legacy_cam,
        {"position": [0.575, 0.65, 0.9], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
    )
    assert ros_cam.calls[0][2] == "ros"
    assert legacy_cam.calls[0][2] == "world"
