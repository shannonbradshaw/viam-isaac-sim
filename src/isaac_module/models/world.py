"""viam:isaac-sim-devin:world - the generic component that owns the simulator.

Configure exactly one of these per machine. All other isaac-sim components
name it in their "world" attribute; their validate_config returns it as an
implicit dependency so viam-server boots the world first.

Attributes:
  mock (bool, default false)        - run without isaac sim (for dev/testing)
  headless (bool, default true)     - run kit without a local GUI window
  livestream (bool, default true)   - enable WebRTC livestreaming (view with
                                      the Isaac Sim WebRTC Streaming Client)
  livestream_public_ip (string)     - IP advertised to streaming clients;
                                      auto-detected if unset
  usd_stage (string)                - USD file/omniverse URL to open; if unset
                                      an empty stage with a ground plane is used
  physics_dt / rendering_dt (float) - sim step sizes, default 1/60
  boot_timeout_sec (float)          - how long to wait for kit to boot
  kit_log_level (string)            - kit console verbosity, default "warning"
  props (list)                      - objects spawned into the scene at boot:
                                      {"name": non-empty str, unique after
                                        sanitizing to a USD prim name,
                                       "type": "cube"|"usd" (default "cube"),
                                       "position": [x,y,z] meters (3 numbers),
                                       "size" (m, > 0), "scale" [sx,sy,sz]
                                        (3 numbers),
                                       "color" [r,g,b] each in [0, 1],
                                       "fixed" (bool),
                                       "usd_path" (non-empty str, required
                                        when type is "usd"),
                                       "orientation_rpy_deg" [r,p,y] degrees,
                                       "orientation_wxyz" [w,x,y,z] (not all
                                        zero); at most one of the two,
                                       "box_dims" [x,y,z] meters, each > 0
                                        (used by "usd" props whose geometry
                                        this module can't infer),
                                       "mass" (kg, > 0), "friction" (unitless,
                                        static = dynamic, >= 0), "restitution"
                                        (unitless, in [0, 1]), "contact_offset"
                                        (m, >= 0), "rest_offset" (m, >= 0,
                                        <= contact_offset when both are set)}
  lighting (object)                 - scene lights to configure at boot:
                                      {"dome": {"intensity": 1000,
                                                 "color": [1, 1, 1]},
                                       "sphere_intensity": 30000}. Both keys
                                      optional; unset means leave the stage's
                                      lights alone.
  render (object)                   - render-cost levers applied at boot,
                                      best-effort (CAM-12): {"motion_bvh":
                                      bool, "disable_viewport_updates": bool}.
                                      Both keys optional; unset means leave
                                      the renderer's defaults alone.
                                      disable_viewport_updates: true requires
                                      livestream: false (the livestream needs
                                      viewport updates).

DoCommand:
  {"command": "status"} | {"command": "play"} | {"command": "pause"} |
  {"command": "reset", "soft"?: bool (default false)} |
  {"command": "add_usd", "usd_path": "...", "prim_path": "/World/thing",
   "position": [x, y, z] meters, "orientation_rpy_deg"?: [r, p, y] degrees} |
  {"command": "prop_geometries"} ->
    {"geometries": [{"name", "box_dims_mm": [x,y,z],
                      "pose_in_world_mm": {"x","y","z","o_x","o_y","o_z",
                                            "theta"} (theta in degrees),
                      "color": [r,g,b] or None, "fixed": bool}]} |
  {"command": "spawn_prop", "prop": {...same schema as the props config
   attr...}} |
  {"command": "set_prop_pose", "name": "...", "position": [x,y,z] mm,
   "orientation_rpy_deg"?: [r,p,y] degrees} |
  {"command": "randomize_props", "names": [...],
   "region": [[x0,y0,z],[x1,y1,z]] mm, "seed": int,
   "min_separation"?: mm (default 150),
   "size_range_mm"?: [lo, hi] (applies to every named prop) or
     {name: [lo, hi]} (keys must be a subset of "names"); cube props
     only, 0 < lo <= hi. Redraws that prop's size (one uniform(lo, hi)
     scalar applied to all three axes) before placing it, from the same
     seeded stream as the positions, so sizes and positions both
     reproduce for a given seed} ->
    {"positions": {name: [x,y,z] mm}, "sizes_mm": {name: [x,y,z] mm}}
    ("sizes_mm" is always present: the drawn dims for a ranged prop, its
     current dims otherwise) |
  {"command": "ignore_props", "names": [...]} -> {"ignored": [...]}
    (empty list clears; excludes named props from get_geometries - DEC-21:
     excludes the pick target while grasping)
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from typing_extensions import Self
from viam.components.generic import Generic
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, RectangularPrism, ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .. import FAMILY, NAMESPACE
from ..physics import PROP_PHYSICS_KEYS
from ..sim_manager import SimConfig, SimManager, WorldHandle, _prim_name
from ..spatial import Quat, quat_from_euler_deg, quat_to_ov, to_vec3

LOGGER = getLogger(__name__)

MM_PER_M = 1000.0
DEFAULT_MIN_SEPARATION_MM = 150.0

# the ground plane the module adds when no usd_stage is configured, served by
# get_geometries so motion plans keep the arm out of the floor (GPU run 7);
# thick, so discrete collision checks cannot step through it
FLOOR_LABEL = "floor"
FLOOR_SIDE_MM = 10000.0
FLOOR_THICKNESS_MM = 200.0

_SUPPORTED_COMMANDS = (
    "status",
    "play",
    "pause",
    "reset",
    "add_usd",
    "prop_geometries",
    "spawn_prop",
    "set_prop_pose",
    "randomize_props",
    "ignore_props",
)

# Commands that reveal or rewrite the scene's ground truth. A world configured
# with "oracle_commands": false refuses them, so a caller holding the
# machine's key (an agent under test) has to find objects by looking; the
# grading oracle then measures placement through the cameras too, or through a
# separate world that keeps these on.
_ORACLE_COMMANDS = frozenset({"prop_geometries", "spawn_prop", "set_prop_pose", "randomize_props"})


def _prop_label(prop: object, index: int) -> str:
    if isinstance(prop, Mapping) and prop.get("name"):
        return str(prop["name"])
    return f"props[{index}]"


def _validate_number_triple(prop_label: str, key: str, value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
        raise ValueError(f"prop {prop_label}: {key!r} must be a list of 3 numbers")
    for v in value:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"prop {prop_label}: {key!r} must be a list of 3 numbers")


def _validate_orientation(label: str, prop: Mapping[str, object]) -> None:
    has_rpy = "orientation_rpy_deg" in prop
    has_wxyz = "orientation_wxyz" in prop
    if has_rpy and has_wxyz:
        raise ValueError(
            f"prop {label}: only one of 'orientation_rpy_deg' or 'orientation_wxyz' may be set"
        )
    if has_rpy:
        _validate_number_triple(label, "orientation_rpy_deg", prop["orientation_rpy_deg"])
    if has_wxyz:
        value = prop["orientation_wxyz"]
        if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 4:
            raise ValueError(f"prop {label}: 'orientation_wxyz' must be a list of 4 numbers")
        for v in value:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"prop {label}: 'orientation_wxyz' must be a list of 4 numbers")
        if all(float(v) == 0.0 for v in value):
            raise ValueError(f"prop {label}: 'orientation_wxyz' must not be all zero")


def _validate_size_range_pair(range_value: object) -> tuple[float, float]:
    if (
        not isinstance(range_value, Sequence)
        or isinstance(range_value, str)
        or len(range_value) != 2
    ):
        raise ValueError("randomize_props: size_range_mm entries must be [lo, hi]")
    lo, hi = range_value
    if not isinstance(lo, (int, float)) or isinstance(lo, bool):
        raise ValueError("randomize_props: size_range_mm entries must be [lo, hi] numbers")
    if not isinstance(hi, (int, float)) or isinstance(hi, bool):
        raise ValueError("randomize_props: size_range_mm entries must be [lo, hi] numbers")
    lo_f, hi_f = float(lo), float(hi)
    if not (0.0 < lo_f <= hi_f):
        raise ValueError(
            f"randomize_props: size_range_mm [{lo_f}, {hi_f}] must satisfy 0 < lo <= hi"
        )
    return lo_f, hi_f


def _validate_size_range_mm(names: list[str], value: object) -> dict[str, tuple[float, float]]:
    if isinstance(value, Mapping):
        unknown = set(value) - set(names)
        if unknown:
            raise ValueError(
                f"randomize_props: size_range_mm names not in names: {sorted(unknown)}"
            )
        return {
            str(name): _validate_size_range_pair(range_value) for name, range_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        pair = _validate_size_range_pair(value)
        return dict.fromkeys(names, pair)
    raise ValueError("randomize_props: size_range_mm must be [lo, hi] or {name: [lo, hi]}")


def _validate_box_dims(label: str, prop: Mapping[str, object]) -> None:
    if "box_dims" not in prop:
        return
    dims = cast("Sequence[float]", prop["box_dims"])
    _validate_number_triple(label, "box_dims", dims)
    for v in dims:
        is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
        if is_number and v <= 0:
            raise ValueError(f"prop {label}: 'box_dims' values must be positive")


def _validate_props(props: object) -> None:
    if not isinstance(props, Sequence) or isinstance(props, str):
        raise ValueError("props must be a list")
    seen_prim_names: set[str] = set()
    for index, prop in enumerate(props):
        label = _prop_label(prop, index)
        if not isinstance(prop, Mapping):
            raise ValueError(f"prop {label}: must be an object")
        name = prop.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"prop {label}: 'name' must be a non-empty string")
        prim_name = _prim_name(name)
        if prim_name in seen_prim_names:
            raise ValueError(
                f"prop {label}: 'name' collides with another prop after sanitizing "
                f"to a USD prim name ({prim_name!r})"
            )
        seen_prim_names.add(prim_name)

        kind = prop.get("type", "cube")
        if kind not in ("cube", "usd"):
            raise ValueError(f'prop {label}: \'type\' must be "cube" or "usd" (got {kind!r})')
        if kind == "usd" and not prop.get("usd_path"):
            raise ValueError(f"prop {label}: 'usd_path' is required when 'type' is \"usd\"")

        if "position" in prop:
            _validate_number_triple(label, "position", prop["position"])
        if "scale" in prop:
            _validate_number_triple(label, "scale", prop["scale"])
        if "color" in prop:
            color = prop["color"]
            _validate_number_triple(label, "color", color)
            if isinstance(color, Sequence) and not isinstance(color, str):
                for v in color:
                    is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
                    if is_number and not (0 <= v <= 1):
                        raise ValueError(f"prop {label}: 'color' values must be in [0, 1]")
        if "size" in prop:
            size = prop["size"]
            if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"prop {label}: 'size' must be a positive number")
        if "fixed" in prop and not isinstance(prop["fixed"], bool):
            raise ValueError(f"prop {label}: 'fixed' must be a bool")

        _validate_orientation(label, prop)
        _validate_box_dims(label, prop)
        _validate_prop_physics(label, prop)


def _validate_number(label: str, key: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"prop {label}: {key!r} must be a number")
    return float(value)


def _validate_prop_physics(label: str, prop: Mapping[str, object]) -> None:
    values: dict[str, float] = {}
    for key in PROP_PHYSICS_KEYS:
        if key in prop:
            values[key] = _validate_number(label, key, prop[key])

    if "mass" in values and values["mass"] <= 0:
        raise ValueError(f"prop {label}: 'mass' must be positive")
    if "friction" in values and values["friction"] < 0:
        raise ValueError(f"prop {label}: 'friction' must be >= 0")
    if "restitution" in values and not (0 <= values["restitution"] <= 1):
        raise ValueError(f"prop {label}: 'restitution' must be in [0, 1]")
    if "contact_offset" in values and values["contact_offset"] < 0:
        raise ValueError(f"prop {label}: 'contact_offset' must be >= 0")
    if "rest_offset" in values and values["rest_offset"] < 0:
        raise ValueError(f"prop {label}: 'rest_offset' must be >= 0")
    if "rest_offset" in values and "contact_offset" in values:
        if values["rest_offset"] > values["contact_offset"]:
            raise ValueError(
                f"prop {label}: 'rest_offset' must not be greater than 'contact_offset'"
            )


_LIGHTING_KEYS = {"dome", "sphere_intensity"}


def _validate_lighting(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("lighting must be an object")
    for key in value:
        if key not in _LIGHTING_KEYS:
            raise ValueError(f"lighting: unknown key {key!r}")

    dome = value.get("dome")
    if dome is not None:
        if not isinstance(dome, Mapping):
            raise ValueError("lighting.dome must be an object")
        if "intensity" in dome:
            intensity = dome["intensity"]
            is_number = isinstance(intensity, (int, float)) and not isinstance(intensity, bool)
            if not is_number or intensity <= 0:
                raise ValueError("lighting.dome.intensity must be a positive number")
        if "color" in dome:
            _validate_number_triple("lighting.dome", "color", dome["color"])
            for v in dome["color"]:
                is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
                if is_number and not (0 <= v <= 1):
                    raise ValueError("lighting.dome.color values must be in [0, 1]")

    sphere_intensity = value.get("sphere_intensity")
    if sphere_intensity is not None:
        is_number = isinstance(sphere_intensity, (int, float)) and not isinstance(
            sphere_intensity, bool
        )
        if not is_number or sphere_intensity < 0:
            raise ValueError("lighting.sphere_intensity must be a non-negative number")


_RENDER_KEYS = {"motion_bvh", "disable_viewport_updates"}


def _validate_render(value: object, livestream: bool) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("render must be an object")
    for key in value:
        if key not in _RENDER_KEYS:
            raise ValueError(f"render: unknown key {key!r}")
        if not isinstance(value[key], bool):
            raise ValueError(f"render.{key} must be a bool")

    if value.get("disable_viewport_updates") and livestream:
        raise ValueError(
            "render.disable_viewport_updates cannot be true while livestream is true "
            "(the livestream needs viewport updates)"
        )


def _orientation_wxyz_from_rpy_deg(rpy: Sequence[float] | None) -> Quat | None:
    if rpy is None:
        return None
    roll, pitch, yaw = (float(v) for v in rpy)
    return quat_from_euler_deg(roll, pitch, yaw)


def _pose_mm_from_m(position_m: Sequence[float], orientation_wxyz: Quat) -> dict[str, float]:
    x, y, z = position_m
    ox, oy, oz, theta = quat_to_ov(orientation_wxyz)
    return {
        "x": x * MM_PER_M,
        "y": y * MM_PER_M,
        "z": z * MM_PER_M,
        "o_x": ox,
        "o_y": oy,
        "o_z": oz,
        "theta": math.degrees(theta),
    }


class IsaacWorld(Generic, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "world")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        w = cls(config.name)
        w.reconfigure(config, dependencies)
        return w

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        for key in ("physics_dt", "rendering_dt", "boot_timeout_sec"):
            if key in attrs and float(attrs[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if "props" in attrs:
            _validate_props(attrs["props"])
        if "lighting" in attrs:
            _validate_lighting(attrs["lighting"])
        if "render" in attrs:
            _validate_render(attrs["render"], bool(attrs.get("livestream", True)))
        if "oracle_commands" in attrs and not isinstance(attrs["oracle_commands"], bool):
            raise ValueError('"oracle_commands" must be true or false')
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: dict[str, Any] = dict(struct_to_dict(config.attributes))
        self._oracle_commands = bool(attrs.get("oracle_commands", True))
        if attrs.get("usd_stage") and "lighting" not in attrs:
            LOGGER.warning("usd_stage is set without lighting: stage must provide floor and lights")
        cfg = SimConfig(
            mock=bool(attrs.get("mock", False)),
            headless=bool(attrs.get("headless", True)),
            livestream=bool(attrs.get("livestream", True)),
            usd_stage=attrs.get("usd_stage") or None,
            physics_dt=float(attrs.get("physics_dt", 1.0 / 60.0)),
            rendering_dt=float(attrs.get("rendering_dt", 1.0 / 60.0)),
            boot_timeout=float(attrs.get("boot_timeout_sec", 300.0)),
            kit_log_level=str(attrs.get("kit_log_level", "warning")),
            livestream_public_ip=str(attrs.get("livestream_public_ip", "")),
            props=[dict(p) for p in attrs.get("props", [])],
            lighting=dict(attrs["lighting"]) if attrs.get("lighting") is not None else None,
            render=dict(attrs["render"]) if attrs.get("render") is not None else None,
        )
        SimManager.get().ensure_booted(cfg)
        # the module adds a ground plane only when it owns the stage
        self._serves_floor = not attrs.get("usd_stage")
        if not hasattr(self, "_ignored_props"):
            self._ignored_props: set[str] = set()

    def _handle(self) -> WorldHandle:
        return SimManager.get().world_handle()

    async def get_geometries(self, **kwargs: Any) -> list[Geometry]:
        ignored: set[str] = getattr(self, "_ignored_props", set())
        geometries: list[Geometry] = []
        if getattr(self, "_serves_floor", False) and FLOOR_LABEL not in ignored:
            geometries.append(
                Geometry(
                    center=Pose(
                        x=0.0,
                        y=0.0,
                        z=-FLOOR_THICKNESS_MM / 2.0,
                        o_x=0.0,
                        o_y=0.0,
                        o_z=1.0,
                        theta=0.0,
                    ),
                    box=RectangularPrism(
                        dims_mm=Vector3(x=FLOOR_SIDE_MM, y=FLOOR_SIDE_MM, z=FLOOR_THICKNESS_MM)
                    ),
                    label=FLOOR_LABEL,
                )
            )
        for prop in self._handle().prop_geometries():
            if prop.name in ignored:
                continue
            if prop.box_dims_m == (0.0, 0.0, 0.0):
                continue
            dims_mm = tuple(d * MM_PER_M for d in prop.box_dims_m)
            pose_mm = _pose_mm_from_m(prop.position_m, prop.orientation_wxyz)
            geometries.append(
                Geometry(
                    center=Pose(**pose_mm),
                    box=RectangularPrism(dims_mm=Vector3(x=dims_mm[0], y=dims_mm[1], z=dims_mm[2])),
                    label=prop.name,
                )
            )
        return geometries

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        handle = self._handle()
        cmd = str(command.get("command", ""))
        if cmd in _ORACLE_COMMANDS and not getattr(self, "_oracle_commands", True):
            raise ValueError(
                f"{cmd!r} is disabled on this world (oracle_commands is false): scene ground truth "
                "is not exposed; observe the scene through the cameras"
            )
        if cmd == "status":
            return handle.status()
        if cmd == "play":
            handle.play()
            return {"ok": True}
        if cmd == "pause":
            handle.pause()
            return {"ok": True}
        if cmd == "reset":
            handle.reset(soft=bool(command.get("soft", False)))
            return {"ok": True}
        if cmd == "add_usd":
            usd_path = str(command.get("usd_path", ""))
            prim_path = str(command.get("prim_path", ""))
            if not usd_path or not prim_path:
                raise ValueError("add_usd requires usd_path and prim_path")
            position = cast("Sequence[float]", command.get("position") or [0.0, 0.0, 0.0])
            orientation_wxyz = _orientation_wxyz_from_rpy_deg(
                cast("Sequence[float] | None", command.get("orientation_rpy_deg"))
            )
            handle.add_usd(
                usd_path, prim_path, to_vec3(position), orientation_wxyz=orientation_wxyz
            )
            return {"ok": True}
        if cmd == "prop_geometries":
            geometries: list[ValueTypes] = []
            for prop in handle.prop_geometries():
                pose_mm = _pose_mm_from_m(prop.position_m, prop.orientation_wxyz)
                geometries.append(
                    {
                        "name": prop.name,
                        "box_dims_mm": [d * MM_PER_M for d in prop.box_dims_m],
                        "pose_in_world_mm": pose_mm,
                        "color": list(prop.color) if prop.color is not None else None,
                        "fixed": prop.fixed,
                    }
                )
            return {"geometries": geometries}
        if cmd == "spawn_prop":
            prop_config = command.get("prop")
            if not isinstance(prop_config, Mapping):
                raise ValueError("spawn_prop requires a 'prop' object")
            prop_attrs = dict(prop_config)
            _validate_props([prop_attrs])
            handle.spawn_prop(prop_attrs)
            return {"ok": True}
        if cmd == "set_prop_pose":
            name = str(command.get("name", ""))
            if not name:
                raise ValueError("set_prop_pose requires 'name'")
            position_mm = cast("Sequence[float]", command.get("position"))
            if position_mm is None:
                raise ValueError("set_prop_pose requires 'position'")
            position_m = tuple(float(v) / MM_PER_M for v in position_mm)
            orientation_wxyz = _orientation_wxyz_from_rpy_deg(
                cast("Sequence[float] | None", command.get("orientation_rpy_deg"))
            )
            handle.set_prop_pose(name, cast("Any", position_m), orientation_wxyz=orientation_wxyz)
            return {"ok": True}
        if cmd == "randomize_props":
            names = [str(n) for n in cast("Sequence[str]", command.get("names") or [])]
            region_mm = cast("Sequence[Sequence[float]]", command.get("region"))
            if not names or region_mm is None:
                raise ValueError("randomize_props requires 'names' and 'region'")
            (x0, y0, z0), (x1, y1, z1) = region_mm
            region_m = (
                (x0 / MM_PER_M, y0 / MM_PER_M, z0 / MM_PER_M),
                (x1 / MM_PER_M, y1 / MM_PER_M, z1 / MM_PER_M),
            )
            seed = int(cast("Any", command.get("seed", 0)))
            min_separation_mm = float(
                cast("Any", command.get("min_separation", DEFAULT_MIN_SEPARATION_MM))
            )
            size_range_mm = command.get("size_range_mm")
            size_range_m: dict[str, tuple[float, float]] | None = None
            if size_range_mm is not None:
                ranges_mm = _validate_size_range_mm(names, size_range_mm)
                size_range_m = {
                    name: (lo / MM_PER_M, hi / MM_PER_M) for name, (lo, hi) in ranges_mm.items()
                }
            result = handle.randomize_props(
                names,
                cast("Any", region_m),
                seed,
                min_separation_m=min_separation_mm / MM_PER_M,
                size_range_m=size_range_m,
            )
            positions_mm: dict[str, ValueTypes] = {
                name: [v * MM_PER_M for v in position]
                for name, position in result.positions_m.items()
            }
            sizes_mm: dict[str, ValueTypes] = {
                name: [v * MM_PER_M for v in dims] for name, dims in result.dims_m.items()
            }
            return {"positions": positions_mm, "sizes_mm": sizes_mm}
        if cmd == "ignore_props":
            names = [str(n) for n in cast("Sequence[str]", command.get("names") or [])]
            self._ignored_props = set(names)
            ignored_names = cast("list[ValueTypes]", sorted(self._ignored_props))
            return {"ignored": ignored_names}
        raise ValueError(f"unknown command {cmd!r}; supported: {', '.join(_SUPPORTED_COMMANDS)}")
