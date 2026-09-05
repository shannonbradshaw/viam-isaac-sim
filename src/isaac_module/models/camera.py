"""viam:isaac-sim-devin:camera - a simulated RGB(-D) camera.

Attributes:
  world (string, required)        - name of the viam:isaac-sim-devin:world component
  prim_path (string)              - existing camera prim to attach to, or
                                    where to create one (default /World/<name>)
  width / height (int)            - resolution, default 848x480
  position ([x,y,z] meters)       - where to place a newly created camera
  target ([x,y,z] meters)         - aim the camera at this point (easiest way
                                    to make a scene-monitor camera)
  orientation_rpy_deg ([r,p,y])   - explicit orientation instead of target
  fov_deg (float)                 - horizontal field of view, default 90.5
  depth (bool)                    - attach the depth annotator and advertise
                                    GetPointCloud / the "depth" GetImages
                                    source, default false
  clip_near / clip_far (m)        - render clipping planes, default 0.05/10.0
  image_format ("png"|"jpeg")     - colour encoding for GetImages, default "png"
  frequency (float)               - capture rate; unset = every rendered frame
  parent_prim (string)            - create the camera as a child of this prim
                                    so it moves with it, e.g. an arm's wrist
                                    link for an end-effector camera. A
                                    parent_prim camera MUST carry a "frame"
                                    whose "parent" is the arm component; the
                                    frame's translation/orientation become the
                                    camera's local mount pose (see
                                    models/utils.apply_frame_to_attrs).
  local_position ([x,y,z] m)      - offset from parent_prim, default [0,0,0.05]
  local_orientation_rpy_deg       - orientation relative to parent_prim;
                                    default [180,0,0] = look out the +Z
                                    (tool) axis
  annotator_device (string)       - GPU-resident annotator data path
                                    (CAM-12), e.g. "cuda"; 5.0 only - ignored
                                    with a log line on 4.5

DoCommand:
  {"command": "sample_color", "region": [x0, y0, x1, y1]} - mean RGB over the
  given pixel region of the most recent frame -> {"srgb_hex", "mean_rgb"}.

close() releases the handle and its post-reset hook (XC-4); the prim stays
in the stage. A reconfigure that changes a spawn attribute (prim_path,
position, parent_prim, or the frame it derives from) after the camera is
already attached raises ValueError - restart the module to apply it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, TypeVar

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore[import-untyped]
from grpclib import Status
from typing_extensions import Self
from viam.components.camera import Camera
from viam.errors import MethodNotImplementedError, ViamGRPCError
from viam.logging import getLogger
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.proto.component.camera import GetPropertiesResponse, IntrinsicParameters
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..camera_base import NoFrameYetError
from ..encoding import (
    DEPTH_MIME,
    PCD_MIME,
    depth_m_to_viam_dep,
    depth_to_xyz,
    rgb_to_jpeg,
    rgb_to_png,
    xyz_rgb_to_pcd,
)
from ..errors import SimTimeoutError
from ..sim_manager import (
    DEFAULT_CAMERA_FOV_DEG,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CLIP_FAR_M,
    DEFAULT_CLIP_NEAR_M,
    CameraHandle,
    SimConfig,
    SimManager,
)
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component

_SAMPLE_COLOR_REGION_LEN = 4
_SUPPORTED_COMMANDS = ("sample_color",)


LOGGER = getLogger(__name__)


class IsaacCamera(Camera, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "camera")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: CameraHandle | None = None
        self._attrs: dict[str, Any] = {}

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        cam = cls(config.name)
        cam.reconfigure(config, dependencies)
        return cam

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        # cameras can always be created fresh, no asset/usd required
        deps, opt_deps = validate_sim_component(config, needs_source=False)
        attrs = get_attrs(config)
        _validate_camera_attrs(config.name, attrs)
        return deps, opt_deps

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        # a frame-derived quat is ROS-optical (+Z forward, the frame system's
        # convention); a legacy orientation_wxyz attr stays world axes
        if (
            config.HasField("frame")
            and config.frame.HasField("orientation")
            and not attrs.get("parent_prim")
        ):
            attrs["orientation_axes"] = "ros"
        self._attrs = attrs
        self._handle = SimManager.get().create_camera(self.name, attrs)

    async def close(self) -> None:
        """XC-4: release the handle (hooks, callbacks); the prim stays attached."""
        SimManager.get().release_handle(self.name)
        self._handle = None

    def _h(self) -> CameraHandle:
        if self._handle is None:
            raise RuntimeError(f"camera {self.name} is not attached to the sim")
        return self._handle

    async def get_images(
        self,
        *,
        filter_source_names: Sequence[str] | None = None,
        extra: dict[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> tuple[list[NamedImage], ResponseMetadata]:
        handle = self._h()
        # OQ-16 / GPU checklist item 4: how viam-server delivers the filter
        LOGGER.info("camera %s get_images filter_source_names=%r", self.name, filter_source_names)
        frame = await asyncio.to_thread(_call, handle.get_frame)

        color_mime = CameraMimeType.JPEG if handle.image_format == "jpeg" else CameraMimeType.PNG
        color_data = (
            rgb_to_jpeg(frame.rgb) if handle.image_format == "jpeg" else rgb_to_png(frame.rgb)
        )
        images: list[NamedImage] = [NamedImage("color", color_data, color_mime)]
        if handle.depth_enabled and frame.depth is not None:
            images.append(
                NamedImage("depth", depth_m_to_viam_dep(frame.depth), CameraMimeType.VIAM_RAW_DEPTH)
            )

        if filter_source_names:
            wanted = set(filter_source_names)
            images = [img for img in images if img.name in wanted]

        timestamp = Timestamp()
        timestamp.GetCurrentTime()
        return images, ResponseMetadata(captured_at=timestamp)

    async def get_point_cloud(self, **kwargs: Any) -> tuple[bytes, str]:
        handle = self._h()
        if not handle.depth_enabled:
            raise MethodNotImplementedError('get_point_cloud (set "depth": true on this camera)')

        def _grab() -> tuple[np.ndarray, np.ndarray]:
            frame = handle.get_frame()
            if frame.depth is None:
                raise RuntimeError("depth is not enabled on this camera (set depth: true)")
            k = handle.get_intrinsics()
            xyz, mask = depth_to_xyz(frame.depth, k)
            return xyz, frame.rgb[mask]

        xyz, rgb = await asyncio.to_thread(_call, _grab)
        pcd = xyz_rgb_to_pcd(xyz, rgb)
        return pcd, PCD_MIME

    async def get_geometries(self, **kwargs: Any) -> list[Any]:
        # cameras occupy no space; needed so the motion service can build a
        # world state when this camera has a frame
        return []

    async def get_properties(self, **kwargs: Any) -> GetPropertiesResponse:
        handle = self._h()
        k = await asyncio.to_thread(_call, handle.get_intrinsics)

        color_mime = str(
            CameraMimeType.JPEG if handle.image_format == "jpeg" else CameraMimeType.PNG
        )
        mime_types = [color_mime]
        if handle.depth_enabled:
            mime_types += [DEPTH_MIME, PCD_MIME]

        cfg = SimManager.get().cfg
        rendering_dt = cfg.rendering_dt if cfg is not None else SimConfig.rendering_dt
        frame_rate = handle.frequency if handle.frequency else 1.0 / rendering_dt

        return GetPropertiesResponse(
            supports_pcd=handle.depth_enabled,
            intrinsic_parameters=IntrinsicParameters(
                width_px=k.width,
                height_px=k.height,
                focal_x_px=k.fx,
                focal_y_px=k.fy,
                center_x_px=k.cx,
                center_y_px=k.cy,
            ),
            mime_types=mime_types,
            frame_rate=frame_rate,
        )

    async def do_command(self, command: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        name = command.get("command")
        if name == "sample_color":
            return await asyncio.to_thread(self._sample_color, command.get("region"))
        raise ValueError(
            f"unknown command {name!r}; supported commands: {', '.join(_SUPPORTED_COMMANDS)}"
        )

    def _sample_color(self, region: Any) -> dict[str, Any]:
        handle = self._h()
        frame = _call(handle.get_frame)
        height, width, _ = frame.rgb.shape

        if (
            not isinstance(region, Sequence)
            or isinstance(region, (str, bytes))
            or len(region) != _SAMPLE_COLOR_REGION_LEN
        ):
            raise ValueError("sample_color requires a region [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (int(v) for v in region)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"sample_color region {list(region)} is out of bounds for a "
                f"{width}x{height} image (need 0 <= x0 < x1 <= width, 0 <= y0 < y1 <= height)"
            )

        patch = frame.rgb[y0:y1, x0:x1]
        mean_rgb = [int(round(v)) for v in patch.reshape(-1, 3).mean(axis=0)]
        srgb_hex = "#" + "".join(f"{v:02X}" for v in mean_rgb)
        return {"srgb_hex": srgb_hex, "mean_rgb": mean_rgb}


_T = TypeVar("_T")


def _call(fn: Callable[[], _T]) -> _T:
    """Run a handle call, mapping sim-layer errors to their gRPC status (CAM-18)."""
    try:
        return fn()
    except NoFrameYetError as e:
        raise ViamGRPCError(str(e), Status.FAILED_PRECONDITION) from e
    except SimTimeoutError as e:
        raise ViamGRPCError(str(e), Status.DEADLINE_EXCEEDED) from e
    except NotImplementedError as e:
        raise MethodNotImplementedError(str(e)) from e


def _positive_int(name: str, attr: str, value: Any) -> None:
    """protobuf Struct decodes every number as float, so accept any non-bool,
    integral, positive number (848.0 is fine; 848.5 and True are not)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}: "{attr}" must be a positive int, got {value!r}')
    if not float(value).is_integer() or value <= 0:
        raise ValueError(f'{name}: "{attr}" must be a positive int, got {value!r}')


def _validate_camera_attrs(name: str, attrs: dict[str, Any]) -> None:
    width = attrs.get("width", DEFAULT_CAMERA_WIDTH)
    height = attrs.get("height", DEFAULT_CAMERA_HEIGHT)
    _positive_int(name, "width", width)
    _positive_int(name, "height", height)

    fov_deg = attrs.get("fov_deg", DEFAULT_CAMERA_FOV_DEG)
    if not (0 < fov_deg < 180):
        raise ValueError(f'{name}: "fov_deg" must be in (0, 180), got {fov_deg!r}')

    clip_near = attrs.get("clip_near", DEFAULT_CLIP_NEAR_M)
    clip_far = attrs.get("clip_far", DEFAULT_CLIP_FAR_M)
    if not (isinstance(clip_near, (int, float)) and clip_near > 0):
        raise ValueError(f'{name}: "clip_near" must be > 0, got {clip_near!r}')
    if not (isinstance(clip_far, (int, float)) and clip_far > 0):
        raise ValueError(f'{name}: "clip_far" must be > 0, got {clip_far!r}')
    if not clip_near < clip_far:
        raise ValueError(
            f'{name}: "clip_near" ({clip_near!r}) must be less than "clip_far" ({clip_far!r})'
        )

    image_format = attrs.get("image_format", "png")
    if image_format not in ("png", "jpeg"):
        raise ValueError(f'{name}: "image_format" must be "png" or "jpeg", got {image_format!r}')

    frequency = attrs.get("frequency")
    if frequency is not None and not (isinstance(frequency, (int, float)) and frequency > 0):
        raise ValueError(f'{name}: "frequency" must be > 0 when set, got {frequency!r}')

    depth = attrs.get("depth", False)
    if not isinstance(depth, bool):
        raise ValueError(f'{name}: "depth" must be a bool, got {depth!r}')

    annotator_device = attrs.get("annotator_device")
    if annotator_device is not None and not (
        isinstance(annotator_device, str) and annotator_device
    ):
        raise ValueError(
            f'{name}: "annotator_device" must be a non-empty string when set, '
            f"got {annotator_device!r}"
        )
