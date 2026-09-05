# viam-isaac-sim

A [Viam](https://www.viam.com) module for controlling and simulating robots in
[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim).

What it is/does
====
* Lets you use Viam to control robots in NVIDIA Isaac Sim
* Models
  * Core simulator model (`world`) that configures the world and runs Isaac Sim
  * A model for each arm, camera, base or other component you want to control/simulate from Viam
* User does:
  * creates the core sim/world component in Viam
  * adds their components, e.g. an arm with `"asset": "ur20"`
  * the world model starts the sim and the component models spawn the right prims
  * views the simulator via the built-in WebRTC livestream or through Viam camera components
  * controls robots and sees cameras through the normal Viam APIs

## Models

| Model | Viam API | What it does |
|---|---|---|
| `viam:isaac-sim-devin:world` | `generic` | Boots Isaac Sim, opens the USD stage, runs the sim loop. Configure exactly one. |
| `viam:isaac-sim-devin:arm` | `arm` | Spawns (or attaches to) an articulation - UR arms, Franka, or any USD - and exposes joint control. |
| `viam:isaac-sim-devin:camera` | `camera` | Creates (or attaches to) a camera prim and serves RGB, depth (`image/vnd.viam.dep`) and point clouds (`pointcloud/pcd`). |
| `viam:isaac-sim-devin:base` | `base` | Spawns a differential-drive robot (e.g. jetbot) and drives it. |
| `viam:isaac-sim-devin:gripper` | `gripper` | Bolts a parallel-jaw gripper (e.g. Robotiq 2F-85) onto an arm's link and drives it open/closed. |

Known assets (usable via the `asset` attribute): `ur3e`, `ur5e`, `ur10`,
`ur10e`, `ur16e`, `ur20`, `franka`, `jetbot`. Anything else can be loaded with
`usd_path`, or attach to prims already in your stage with `prim_path`.

## Example machine config

```json
{
  "components": [
    {
      "name": "sim-world",
      "api": "rdk:component:generic",
      "model": "viam:isaac-sim-devin:world",
      "attributes": {
        "headless": true,
        "livestream": true
      }
    },
    {
      "name": "my-ur20",
      "api": "rdk:component:arm",
      "model": "viam:isaac-sim-devin:arm",
      "frame": { "parent": "world" },
      "attributes": {
        "world": "sim-world",
        "asset": "ur20"
      }
    },
    {
      "name": "overhead-cam",
      "api": "rdk:component:camera",
      "model": "viam:isaac-sim-devin:camera",
      "frame": {
        "parent": "world",
        "translation": { "x": 2000, "y": 2000, "z": 2000 }
      },
      "attributes": {
        "world": "sim-world",
        "target": [0, 0, 0.5],
        "width": 1280,
        "height": 720
      }
    },
    {
      "name": "my-jetbot",
      "api": "rdk:component:base",
      "model": "viam:isaac-sim-devin:base",
      "frame": {
        "parent": "world",
        "translation": { "x": 1000, "y": 0, "z": 100 }
      },
      "attributes": {
        "world": "sim-world",
        "asset": "jetbot"
      }
    }
  ]
}
```

Every non-world component must set `"world"` to the world component's name.
That attribute is also returned as an implicit dependency from each model's
validate, so viam-server starts the world first - no `depends_on` needed.

Components are **placed with the standard frame config** (translations in mm,
any orientation representation) - the spawn pose in Isaac and viam's frame
system then agree, so things like the motion service see components where
they actually are. The `position` (meters) / `orientation_rpy_deg` attributes
still work as a fallback when no frame is set. A camera `target` attribute
overrides orientation to aim at a point.

**Frames:** a spawned component's `frame.parent` must be `"world"` - the
spawn path does not resolve an arbitrary frame chain, only the world's own
origin. The one exception is a component that also sets `parent_prim` (e.g.
a wrist camera riding an arm link): it must instead name, as its
`frame.parent`, the component that owns that prim (e.g. `parent_prim
"/World/pick_arm/wrist_3_link"` requires `frame.parent` to be `"pick-arm"`
or a sub-frame like `"pick-arm:ee_link"`). `frame.parent "world"`, or no
`frame` at all, is rejected with a validation error - for a mounted
component the frame is the single source of truth for the mount, and its
translation/orientation become the camera's local pose on the link.

### world attributes

| attribute | default | notes |
|---|---|---|
| `mock` | `false` | run without Isaac Sim installed (development/testing) |
| `headless` | `true` | no local GUI window |
| `livestream` | `true` | WebRTC viewer at `http://<host>:8211/streaming/webrtc-client` |
| `livestream_public_ip` | _unset_ | IP advertised to streaming clients when the sim machine has multiple interfaces |
| `usd_stage` | _empty stage + ground plane_ | USD file or omniverse:// URL to open. When set without `lighting`, the module logs a warning that the stage must provide its own floor and lights (it adds neither to a user stage) |
| `physics_dt` / `rendering_dt` | `1/60` | step sizes in seconds. The pick cell uses `1/120` for `physics_dt` (>= 80 steps/s is the floor for a 2F-85 grasp), rendering stays `1/60` |
| `boot_timeout_sec` | `300` | Isaac Sim can take a while on first boot |
| `kit_log_level` | `"warning"` | kit console verbosity |
| `props` | `[]` | objects spawned into the scene at boot, see below |
| `oracle_commands` | `true` | serve the ground-truth `DoCommand`s (`prop_geometries`, `spawn_prop`, `set_prop_pose`, `randomize_props`); `false` refuses them, for a cell whose callers must perceive the scene through the cameras |
| `lighting` | _unset_ | `{"dome": {"intensity": 1000, "color": [1, 1, 1]}, "sphere_intensity": 30000}` - both keys optional, unset leaves the stage's lights alone. The default stage has a single 100 000-intensity sphere light, so a dome light is useful to even out colour for detection |
| `render` | _unset_ | render-cost levers applied at boot, best-effort: `{"motion_bvh": bool, "disable_viewport_updates": bool}` - both keys optional, unset leaves the renderer's defaults alone. `disable_viewport_updates: true` requires `livestream: false` (the livestream needs viewport updates) and is refused otherwise |

Each entry in `props` is an object: `name` (string, snake_cased for the prim
path), `type` (`"cube"` or `"usd"`), `position` ([x,y,z] meters, the prop's
**centre**), `size` (meters, the cube's base edge length, > 0), `scale`
([sx,sy,sz], multiplies `size` per axis), `color` ([r,g,b] each in `[0, 1]`),
`fixed` (bool - static vs. dynamic/physics-driven), and `usd_path` (required
when `type` is `"usd"`). `orientation_rpy_deg` ([r,p,y] degrees) or
`orientation_wxyz` ([w,x,y,z], not all zero) sets the prop's initial
orientation - at most one of the two may be set. `box_dims` ([x,y,z] meters,
each > 0) gives the obstacle box for a `"usd"` prop whose geometry this
module can't infer from the asset. Without it, that prop's `get_geometries` /
`prop_geometries` box is all zero and it is skipped as an obstacle (see
"Props and obstacles" below). Physics keys apply only when set - otherwise
Isaac's authored defaults are left alone: `mass` (kg, > 0), `friction`
(unitless, static = dynamic, >= 0, combine mode `max`), `restitution`
(unitless, in `[0, 1]`), `contact_offset` (m, >= 0), `rest_offset` (m, >= 0,
<= `contact_offset` when both are set). The shipped pick cell sets `mass:
0.05, friction: 0.7, restitution: 0, contact_offset: 0.005` on the block and
`friction: 0.7, restitution: 0` on the (fixed, so massless) place pad.

`props` validation rules (`ValueError` on config, surfaced as
`INVALID_ARGUMENT`):
* names must be unique once snake_cased (the same normalisation used for prim
  paths)
* `type` must be `"cube"` or `"usd"`
* `usd_path` is required when `type` is `"usd"`
* `position`, `scale`, and `color` must each be 3-number sequences
* `color` values must be in `[0, 1]`
* `size` must be a positive number
* at most one of `orientation_rpy_deg` / `orientation_wxyz` may be set, and
  `orientation_wxyz` must not be all zero
* `box_dims` values must be positive

The world also supports `DoCommand`:

* `{"command": "status" | "play" | "pause"}`
* `{"command": "reset", "soft"?: bool (default false)}`
* `{"command": "add_usd", "usd_path": "...", "prim_path": "/World/thing",
  "position": [x, y, z] meters, "orientation_rpy_deg"?: [r, p, y] degrees}` -
  drop an extra USD reference into the scene
* `{"command": "prop_geometries"}` -> `{"geometries": [{"name",
  "box_dims_mm": [x,y,z], "pose_in_world_mm": {"x", "y", "z", "o_x", "o_y",
  "o_z", "theta"} (theta in degrees), "color": [r,g,b] or `None`, "fixed":
  bool}]}` - every prop's current box and pose, in millimetres, for a client
  that builds its own `WorldState` (see "Props and obstacles" below)
* `{"command": "spawn_prop", "prop": {...same schema as the `props` config
  attribute...}}` - spawn a prop at runtime
* `{"command": "set_prop_pose", "name": "...", "position": [x,y,z] mm,
  "orientation_rpy_deg"?: [r,p,y] degrees}` - move an existing prop
* `{"command": "randomize_props", "names": [...], "region": [[x0,y0,z],
  [x1,y1,z]] mm, "seed": int, "min_separation"?: mm (default 150),
  "size_range_mm"?: [lo, hi] (applies to every named prop) or
  {name: [lo, hi]} (keys must be a subset of `names`); cube props only,
  0 < lo <= hi}` -> `{"positions": {name: [x,y,z] mm}, "sizes_mm":
  {name: [x,y,z] mm}}` - scatter the named props inside the region,
  optionally redrawing each ranged prop's size first (see the worked
  example below)
* `{"command": "ignore_props", "names": [...]}` -> `{"ignored": [...]}` - an
  empty list clears the exclusion. Excludes the named props from
  `GetGeometries` (e.g. the block currently being grasped)

### Units and conventions

Viam's frame system (component `frame` config, `GetEndPosition`, camera
`target`, etc.) uses **millimetres and degrees**, per the standard Viam
convention. The world's `props` attribute, by contrast, is Isaac-native:
**metres**, and `position` is always the prop's **centre**, not a corner.
For a cube prop the rendered extent along each axis is `size × scale[axis]`
(so `size` is a base edge length and `scale` stretches it per axis). Isaac
Sim is Z-up, matching Viam's frame convention.

Worked example - a table as a `fixed` cube prop:

```json
{"type": "cube", "fixed": true, "size": 1.0, "scale": [1.2, 0.8, 0.75],
 "position": [0.60, 0.00, 0.375]}
```

The table top's height above the world origin is:

```
z_top = position.z + size * scale.z / 2
      = 0.375 + 1.0 * 0.75 / 2
      = 0.75 m
```

and its top face spans x ∈ [0.00, 1.20], y ∈ [-0.40, +0.40] (the cube is
centred at `position`, so each face sits `size * scale[axis] / 2` from the
centre along that axis). Anything you place *on* the table - a block, a
mount frame - belongs at `z_top + <that thing's own half-height>`, e.g. a
block of `size` 0.05 sits with its centre at `z_top + 0.025`.

### arm attributes

| attribute | default | notes |
|---|---|---|
| `world` | _required_ | name of the world component |
| `asset` | _one of asset/usd_path/prim_path required_ | known asset, e.g. `"ur20"` |
| `usd_path` | _one of asset/usd_path/prim_path required_ | arbitrary USD file or omniverse:// URL |
| `prim_path` | _one of asset/usd_path/prim_path required_ | attach to an articulation already in the stage |
| `position` | _unset_ | `[x,y,z]` meters, fallback when no `frame` is set |
| `orientation_wxyz` | _unset (identity)_ | `[w,x,y,z]`, fallback orientation when no `frame` is set - the frame config is converted into this same field |
| `end_effector_prim` | `<arm prim>/wrist_3_link` for UR assets, else unset | prim path whose pose is reported by `GetEndPosition`, converted to Viam's orientation-vector convention |
| `move_timeout_sec` | `30` | deadline for a move to converge |
| `kinematics_url` | _unset_ | SVA `.json` or `.urdf` (http(s):// or file://) for assets `GetKinematics` doesn't fetch automatically |

**`GetEndPosition` reports the end effector's pose in the arm base frame**
(a breaking change this release - it previously reported world frame). This
matches how a real arm driver reports its end position, and lets Viam's
frame system (via the component's `frame` config) compose it into world
frame itself.

`MoveToJointPositions` / `GetJointPositions` work today. IK and motion
planning are deliberately left to Viam (the motion service), not Isaac - the
module's job is just to expose the simulated arm. `MoveToPosition` is
**unimplemented by decision** and returns `UNIMPLEMENTED` - use the motion
service instead.

`GetGeometries` returns `[]` by decision: rdk derives arm link geometry from
`GetKinematics` (the SVA already carries the link capsules) and never calls
`Geometries` for an arm that serves kinematics.

`GetKinematics` works: for `ur3e`/`ur5e`/`ur20` the official viam SVA
kinematics files are fetched automatically (and cached in the module data
dir). For anything else set `kinematics_url` to an SVA `.json` or `.urdf`
(http(s):// or file://). With kinematics served, the motion service can plan
for the simulated arm.

UR assets (`ur3e`/`ur5e`/`ur10`/`ur10e`/`ur16e`/`ur20`) get a built-in
**base-frame correction** applied at spawn so the arm's frame in Isaac lines
up with the kinematics Viam's motion service uses - without it the sim and
Viam's idea of the arm's pose would silently disagree.

**`IsMoving`** is true while any named joint's `|velocity| > VEL_EPS_RAD_S`
OR any `|commanded - measured| > SETTLE_TOL_RAD` - a stalled arm that never
reached its target therefore keeps reporting `True`.

**Move errors**: a target outside the SVA's declared joint limits, or a
joint count that doesn't match the arm's DOF count, raises `INVALID_ARGUMENT`.
The arm stalling (settling short of its target, e.g. blocked by an obstacle)
raises `ABORTED`. The move deadline (`move_timeout_sec`, capped by the SDK's
`timeout=`) passing while still converging raises `DEADLINE_EXCEEDED`. For a
multi-waypoint trajectory, an intermediate waypoint that times out only
warns and continues (it uses a loose tolerance and short deadline so the arm
flows through it). An intermediate waypoint that stalls still raises
`ABORTED` (an obstacle blocking the path won't clear itself). The final
waypoint settles tight against the full move deadline.

**`MoveOptions`**: `max_vel_degs_per_sec_joints` (per-joint velocity limits,
the min across joints) wins over the scalar `max_vel_degs_per_sec` when set.
The acceleration fields and `max_tcp_speed` are logged once and not honoured.

**DoCommand** `{"command": "all_dof_names"}` returns every DOF of the
articulation (arm joints plus anything attached under it, e.g. a gripper).
`{"command": "dof_names"}` stays just the arm's named joints.

### gripper attributes

`world` (required), `arm` (required, name of the `viam:isaac-sim-devin:arm`
component this gripper is bolted to).

| attribute | default | notes |
|---|---|---|
| `world` | _required_ | name of the world component |
| `arm` | _required_ | name of the arm it is bolted to |
| `asset` | `"robotiq_2f_85"` | known gripper asset |
| `parent_prim` | `<arm prim>/wrist_3_link` | link it is bolted to |
| `local_position` | _unset (identity)_ | `[x,y,z]` meters, mount pose of the gripper's `base_link` on `parent_prim` |
| `local_orientation_rpy_deg` | _unset (identity)_ | the 2F-85 base sits flush on the flange |
| `tcp_offset_m` | `0.134` | flange -> tool centre point along tool +Z: the fingertip pad centre as measured in Isaac (the pads span 115-153 mm) |
| `open_deg` | `0` | drive-joint angle when fully open |
| `closed_deg` | `47` on Isaac 5.0, `45` on 4.5 | drive-joint angle when fully closed (the Isaac-release value from `compat.caps()`) |
| `grab_timeout_sec` | `5` | how long `grab()` waits for a stall or full closure |
| `holding_tolerance_deg` | `2` | commanded-vs-measured gap that counts as holding |
| `holding_effort_min_nm` | _unset_ | Isaac only: measured drive-joint effort (N m) at which the jaw counts as holding; unset = the stall check below |
| `mock_object_width_m` | _unset_ | mock only: width of the object between the jaws (unset = nothing to grab, so `grab()` returns `false`) |

**Frame** - unlike a mounted camera, the gripper's frame does not place its
prim: `base_link` bolts to `parent_prim` at `local_position` /
`local_orientation_rpy_deg`, and the frame's translation is the TCP the
motion service plans against (not the flange). `frame.parent` must be the
arm, and the translation is the TCP offset along the arm's tool axis, e.g.:

```json
{"frame": {"parent": "pick-arm", "translation": {"x": 0, "y": 0, "z": 134}}}
```

**Tool axis (confirmed on the GPU):** the arm's tool/approach axis is the
link frame's **+Z**, so gripper and wrist-camera `frame.translation` offsets
off an arm link both go along +Z.

**API mapping** (viam-sdk `Gripper`, all eight abstract methods): `open` /
`is_moving` drive the handle directly. `stop` freezes the jaw, and while an
object is held it keeps the grasp (the commanded target stays) rather than
relaxing the drive: viam-server calls `Stop` on every actuator a session
commanded once that session lapses, so a one-shot client such as the CLI's
`part run` would otherwise drop the object two seconds after exiting.
`grab()` closes the jaw, waits up to `grab_timeout_sec` for a stall or full
closure, and returns `is_holding_something()`. By default both are
stall-short-of-closure checks, not a force sensor; with
`holding_effort_min_nm` set (and an Isaac build that reports measured joint
efforts) holding means the drive is pushing at least that hard with the jaw
short of fully closed, and `is_holding_something()` reports the effort as
`finger_effort_nm` in its metadata. Measured on Isaac 5.0 with the 2F-85 and
the pick cell's 60 mm, 50 g cube (2026-09-04): open and idle 0.04-0.06 N m,
closed on nothing 0.004-0.008, closed on the cube resting on the table 0.25,
the same grasp lifted 4.0, unchanged after `stop`. The pick cell sets
`holding_effort_min_nm` to `0.15`. `get_current_inputs()` / `go_to_inputs([v])` use a single
value in `[0, 1]`: `0` = open, `1` = closed. `GetKinematics` returns a
1-link/0-joint SVA whose link is the 36 × 146 × 153 mm gripper box (flange to fingertips, centre 57.5 mm behind the
TCP. `GetGeometries` returns that same single box.

### Lifecycle (close / reconfigure)

Closing a component (`close()`) releases its handle and post-reset hooks.
The underlying prim stays in the stage (Kit cannot un-spawn it), so a later
`create_*` for the same name re-attaches to it.

Changing a **spawn** attribute on an already-attached component - `asset`,
`usd_path`, `prim_path`, `position`, `orientation`, the frame pose, `parent_prim`,
camera optics, etc. - is rejected with a `ValueError` that says to restart
the module to apply it. The component is not left silently running a stale
prim.

**Runtime attributes** re-apply live without a restart: `world`,
`move_timeout_sec`, `max_linear_mps`, `max_angular_rps`, `arm`,
`tcp_offset_m`, `open_deg`, `closed_deg`, `grab_timeout_sec`,
`holding_tolerance_deg`, `holding_effort_min_nm`, `mock_object_width_m`.

### camera attributes

`world` (required), and either `prim_path` of an existing camera in your stage
or `position` plus `target` (aim-at point) or `orientation_rpy_deg` to create
one.

| attribute | default | notes |
|---|---|---|
| `world` | _required_ | name of the world component |
| `prim_path` | _unset_ | attach to an existing camera prim instead of spawning one |
| `position` | _unset_ | `[x,y,z]` meters, fallback when no `frame` is set |
| `target` | _unset_ | aim-at point, overrides orientation to point at it |
| `orientation_rpy_deg` | _unset_ | fallback orientation when no `frame` is set |
| `orientation_wxyz` | _unset (identity)_ | `[w,x,y,z]`, alternate fallback orientation when no `frame` is set and no `target` - the frame config is converted into this same field |
| `local_position` | `[0, 0, 0.05]` | mount position on `parent_prim` when no `frame` is set (legacy) |
| `local_orientation_rpy_deg` | `[180, 0, 0]` | mount orientation on `parent_prim` when no `frame` is set (legacy) |
| `width` | `848` | image width in pixels |
| `height` | `480` | image height in pixels |
| `fov_deg` | `90.5` | horizontal field of view |
| `depth` | `false` | attach the depth annotator (wrist cam: `true`) |
| `clip_near` | `0.05` | metres, near clip plane |
| `clip_far` | `10.0` | metres, far clip plane |
| `image_format` | `"png"` | colour encoding for `GetImages` (`"png"` or `"jpeg"`) |
| `block_size_mm` | _unset_ | mock backend only: metric edge of the fabricated red block; unset keeps the fixed pixel-offset block |
| `view` | `"top"` | mock backend only: `"side"` fabricates a profile scene of the `blocks` list rising from the support line at the principal row |
| `blocks` | _unset_ | mock backend only, side view: list of `{rgb, size_mm, height_mm, column_offset_px, depth_m}` fabricated blocks |
| `frequency` | _unset_ | capture rate, unset = every rendered frame |
| `parent_prim` | _unset_ | ride a link (e.g. a wrist camera) instead of spawning free-standing, requires a `frame` whose `parent` is the component that owns that prim (see "Frames" above) - its translation/orientation become the camera's local pose, applied in ROS-optical axes (camera +Z = frame +Z). The legacy `local_position`/`local_orientation_rpy_deg` attributes only apply when no `frame` is set |
| `annotator_device` | _unset_ | GPU-resident annotator data path, e.g. `"cuda"`, applied on Isaac Sim 5.0+, logged and ignored on 4.5 |
| `orientation_axes` | `"world"` | convention of `orientation_wxyz` for a free-standing camera: `"world"` (+X forward) for the legacy attr, `"ros"` (+Z forward) set automatically when the quat comes from a Viam `frame` orientation - don't set it by hand |

**What the camera serves:** `GetImages` returns `[NamedImage("color", png|jpeg),
NamedImage("depth", image/vnd.viam.dep)]`, colour first, honouring
`filter_source_names` (unknown names are dropped, so fewer images come back).
`GetPointCloud` (depth cameras only) returns binary `pointcloud/pcd`, in
metres, in the camera-optical frame (+X right, +Y down, +Z forward). Invalid
points are dropped. `GetProperties` reports `supports_pcd` (true iff `depth`
is set), real pinhole intrinsics (`fx = fy = W/(2·tan(hfov/2))`, e.g. 420.3 px
at 848x480 @ 90.5°), `mime_types`, and `frame_rate`. `DoCommand
{"command": "sample_color", "region": [x0, y0, x1, y1]}` returns
`{"srgb_hex": "#RRGGBB", "mean_rgb": [r, g, b]}`, useful for picking
`detect_color` for Viam's `color_detector`. While the renderer has not
produced a frame yet (just after create/reset), calls fail with gRPC
`FAILED_PRECONDITION` - retry.

A wrist camera riding an arm link:

```json
{
  "name": "wrist-cam",
  "api": "rdk:component:camera",
  "model": "viam:isaac-sim-devin:camera",
  "frame": {
    "parent": "pick-arm",
    "translation": { "x": 0, "y": 0, "z": 60 },
    "orientation": { "type": "ov_degrees", "value": { "x": 0, "y": 0, "z": 1, "th": 0 } }
  },
  "attributes": {
    "world": "sim-world",
    "parent_prim": "/World/pick_arm/wrist_3_link",
    "depth": true
  }
}
```

### base attributes

`world` (required), `asset` (e.g. `jetbot`, which brings wheel defaults) or
`usd_path`/`prim_path` plus `wheel_joints: [left, right]`, `wheel_radius`,
`wheel_base`. `max_linear_mps` / `max_angular_rps` scale `SetPower`.

## Pick-and-place fragment

The `isaac-sim-pick-and-place` fragment (source in
`fragments/pick-and-place.json`) is a ready-made work cell: a UR5e
(`pick-arm`) mounted at the corner of a 0.75 m table, a `pick-grip`
gripper, a red 6 cm cube (`pick_cube`) plus five distractor cubes
(`ignore_cube_green`, `ignore_cube_blue`, `ignore_cube_yellow`,
`ignore_cube_purple`, `ignore_cube_orange`), a `place_pad` to set the block
down on, a `wrist-cam` riding the arm's flange, a `scene-cam` watching the
whole workspace, a fixed `side-cam` looking across the table for tallest-block
measurement, the `builtin` motion service, and the `red-detector` /
`block-segmenter` vision services that find the block. Add the fragment to
any machine that meets the requirements above and the world spawns
everything at boot.

Props are configured on the world with the `props` attribute (cubes or USD
references, fixed or dynamic) - see the fragment for the shape of it.

The fragment ships nine `$variable`s, each with a `default_value` equal to
the numbers below, so a machine that sets nothing boots the exact cell:

| variable | binds | default |
|---|---|---|
| `table-height-m` | table prop `scale[2]` | `0.75` |
| `pick-block-color` | `pick_cube.color` | `[0.9, 0.1, 0.1]` |
| `distractor-color-green` | `ignore_cube_green.color` | `[0.05, 0.65, 0.1]` |
| `distractor-color-blue` | `ignore_cube_blue.color` | `[0.05, 0.1, 0.9]` |
| `distractor-color-yellow` | `ignore_cube_yellow.color` | `[0.9, 0.75, 0.05]` |
| `distractor-color-purple` | `ignore_cube_purple.color` | `[0.55, 0.1, 0.75]` |
| `distractor-color-orange` | `ignore_cube_orange.color` | `[1.0, 0.55, 0.05]` |
| `detect-color` | `red-detector`'s `detect_color` | `"#EA8D8D"` |
| `hue-tolerance-pct` | `red-detector`'s `hue_tolerance_pct` | `0.05` |

`table-height-m` only substitutes the table prop's `scale[2]`: it has no
arithmetic, so overriding it desyncs every other number derived from the
table height - the table's own `position[2]` (`h / 2`), `sim-world`'s
`frame.geometry` z (`h - 0.01` m, box centred at `(h - 0.01) / 2`), the
six blocks' and `place_pad`'s z, and `pick-arm`'s frame z (`h` in mm).
Override the table height only via `fragment_mods` `$set` overrides on
those other fields too, kept in sync by hand.

The `isaac-sim-pick-and-place` fragment in the registry is the original
upstream public one. This fork's fragment ships with P5, until then use
`viam module reload-local` (local module) with the JSON in
`fragments/pick-and-place.json`.

`examples/pick_red_block.py` drives this cell end to end: it detects the
red block with `red-detector`/`block-segmenter` on `wrist-cam`, picks it,
and places it on `place_pad`. Run it against the fragment with:

```sh
PYTHONPATH=src python examples/pick_red_block.py \
  --address <machine-address> --api-key <key> --api-key-id <key-id> \
  --support-z-mm 750
```

`--support-z-mm 750` tells the script the block rests on the 0.75 m table,
not the floor. The table is already a planner obstacle - the world serves
every prop geometry live - so `--table` (the recipe box for scenes that
serve none) is unnecessary and is dropped automatically when a live
`table` box is present. Add `--randomize-seed <n>` to scatter the six
blocks deterministically first (see "Randomizing props" below).

`--block-size-mm` is optional: omit it (the default) and the script measures
the target's size itself, from the focused detection's point cloud -
footprint (x/y extents) and height (top face minus the support), cross-checked
against each other, printed as `MEASURED_BLOCK_JSON={"footprint_mm": [x, y],
"height_mm": h, "size_mm": s, "scan_pose_mm": {...}}`. A view where the three
readings disagree is treated as degenerate and re-scanned rather than grasped
on. A measured size wider than the gripper's jaw (75 mm - the 2F-85's 85 mm
opening minus finger clearance) refuses the grasp cleanly instead of
attempting a doomed pick. Pass `--block-size-mm <mm>` to skip measurement
and use a fixed size end to end, as before.

Combine `--randomize-seed <n>` with `--randomize-size-mm <lo>,<hi>` to also
redraw each scattered block's size (see "Randomizing props" below); the
script warns, without failing, if the measured size falls outside that range.

### Measuring the tallest block (dynamic carry heights)

With sizes randomized, a fixed constant no longer bounds how high the arm
must lift the held block to clear the scattered ones, so the script measures
the tallest object in the scatter region before picking. The primary sensor
is `side-cam`, a fixed camera looking across the table: frame-oriented (no
arm motion needed) and occlusion-proof for the max height, except when a
nearer block's silhouette covers a farther, taller one. Each candidate
measurement runs four trust checks - enough in-region points above the
support plane, a size-range-plausible result, enough points near the
measured top (no lone stray point), and all four footprint quadrants
covered - and only a measurement that passes all four is used. When the side
scan fails trust (or `--tallest-camera` disables it), the wrist camera sweeps
region-corner vantages as a fallback; if that also fails, the
`--randomize-size-mm` range's upper bound is used as a conservative ceiling,
with a logged warning. GPU runs validated the side scan within 0.8 mm of the
drawn (ground-truth) size and the wrist sweep within 0.1 mm.

When `--randomize-size-mm` is on, the measured tallest height replaces the
legacy fixed 130 mm keep-out ceiling and 200 mm carry-hop height (both
60 mm-block constants) with heights derived from the measurement; without
`--randomize-size-mm`, the script keeps using those fixed constants.
`--tallest-camera` (default `side-cam`) names the camera component to use as
the primary sensor; pass an empty string to disable it and go straight to
the wrist sweep.

The script prints `MEASURED_TALLEST_JSON={"tallest_mm": 81.06, "source":
"side", "trusted": true, "reasons": [], "points": 1842, "scan_poses_mm": [],
"keepout_height_mm": 148.3, "carry_clear_above_support_mm": 231.6,
"drawn_tallest_mm": 81.06, "drawn_delta_mm": 0.0}`. `source` is `"side"`,
`"wrist_sweep"`, or `"fallback"`. `drawn_tallest_mm`/`drawn_delta_mm` are
log-only evidence from the randomize response's ground-truth sizes (null
without a sizes-bearing randomize response) - the pipeline never reads sim
ground truth to make a decision, only the client-side measurement does.

### Props and obstacles

A table is just a `fixed` cube prop on the world (see "Units and
conventions" above for how `size`/`scale`/`position` work):

```json
{
  "props": [
    {"name": "table", "type": "cube", "fixed": true, "size": 1.0,
     "scale": [1.2, 0.8, 0.75], "position": [0.60, 0.00, 0.375]}
  ]
}
```

`props` are visual/physical geometry in the sim. Whether the motion service
plans around them depends on which of three routes you take:

1. **Nothing extra - `sim-world`'s own `GetGeometries`.** The world
   component serves every prop's current box live via `GetGeometries` (world
   frame, millimetres, `label` = the prop's `name`), skipping any prop whose
   box is all zero (an unknown-size `"usd"` prop with no `box_dims` set - see
   "world attributes" above). A motion service that plans with the frame
   system picks these up with no client code - active only when `sim-world`
   itself has a `frame` (`{"parent": "world"}`, as the fragment configures).
   When the module added its own ground plane (no `usd_stage`), the served
   geometries also include a 10 x 10 m `floor` box whose top face is at
   z = 0, so plans keep the arm out of the floor. Use `{"command":
   "ignore_props", "names": [...]}` on the world to exclude the prop being
   grasped, e.g. so the block doesn't obstruct its own pick.
2. **A client-built `WorldState`.** A client that assembles its own
   `WorldState` fetches `{"command": "prop_geometries"}` (see the `DoCommand`
   list above) and converts the returned boxes into `Geometry` obstacles
   itself. `obstacles_from_prop_geometries` in `examples/pick_red_block.py`
   is the recipe: one `RectangularPrism` per entry, skipping the excluded
   name(s) and any all-zero box. `table_obstacle` in the same file is the
   worked example for the fragment's 1.2 x 0.8 m table top, centred at (600,
   0, 370) mm.
3. **A static `frame.geometry` on `sim-world`.** For fixed furniture that
   never moves, add its box as `frame.geometry` on the world component
   itself, in millimetres and centred on the frame origin. This route stays
   valid alongside the other two. Here the table box is 1200 x 800 x 740 mm
   at translation (600, 0, 370), 10 mm below the real 750 mm surface so the
   arm isn't blocked from resting on it:

```json
{
  "name": "sim-world",
  "api": "rdk:component:generic",
  "model": "viam:isaac-sim-devin:world",
  "frame": {
    "parent": "world",
    "geometry": {
      "type": "box",
      "x": 1200,
      "y": 800,
      "z": 740,
      "translation": { "x": 600, "y": 0, "z": 370 }
    }
  },
  "attributes": { "props": [ /* ... */ ] }
}
```

### Randomizing props

`{"command": "randomize_props", ...}` scatters named props to random
positions inside a region, deterministically: the same `seed` always
produces the same layout. Positions are kept at least `min_separation` mm
apart, or the two props' edge-to-edge gap if that is larger for props with
known sizes (default `min_separation` 150 mm). A worked example scattering
three of the fragment's six blocks across the table top (1200 x 800 mm, centred at
(600, 0, 370) mm, so the top face is at z = 740 mm):

```json
{
  "command": "randomize_props",
  "names": ["pick_cube", "ignore_cube_green", "ignore_cube_blue"],
  "region": [[0, -400, 740], [1200, 400, 740]],
  "seed": 42
}
```

-> `{"positions": {"pick_cube": [x, y, 770.5], "ignore_cube_green": [x, y, 770.5],
"ignore_cube_blue": [x, y, 770.5]}, "sizes_mm": {"pick_cube": [x, y, z], ...}}`
(centre z = face z + half the 60 mm block + a 0.5 mm rest gap) with the same
`x`/`y` values every time `seed: 42` is passed for this region
and these names. `sizes_mm` is always present: each named prop's current
box dims, in millimetres.

Adding `"size_range_mm": [30, 90]` (or `{"pick_cube": [30, 90]}` to target
specific props) redraws a fresh size for each ranged cube prop before
placing it - one uniform draw per prop, in millimetres, applied to all
three axes - from the same seeded stream as the positions, so `seed: 42`
reproduces both the sizes and the positions. The prop's `size` config
attribute stays the spawn baseline: rescaling is always relative to it, so
repeated `randomize_props` calls with different ranges never compound.

A sized call also resets the world before the teleports, the same way
`spawn_prop` does: rescaling a live rigid body invalidates PhysX's cooked
state, so every prop snaps to its spawn pose and the post-reset hooks fire,
and then the named props teleport to their sampled positions. A call
without `size_range_mm` never resets.

### Arm mount recipe

Mount an arm on the table by frame-placing it at the table's top height,
inset from the edge:

```json
{
  "name": "pick-arm",
  "api": "rdk:component:arm",
  "model": "viam:isaac-sim-devin:arm",
  "frame": { "parent": "world", "translation": { "x": 150, "y": -250, "z": 750 } },
  "attributes": { "world": "sim-world", "asset": "ur5e" }
}
```

The Isaac articulation root is a fixed joint to the world - no mount joint
needs authoring. Keep the base at least 70 mm inside the table's edge so its
collider clears the table. UR assets carry a built-in base-frame correction
so this frame placement and Viam's kinematics agree with the simulated pose
(see "arm attributes" above).

## Viewing the simulator

* **Through Viam (recommended)**: add an `viam:isaac-sim-devin:camera` component with
  `position` + `target` (see the example config) and watch it in the Viam app
  like any other camera - control tab, data capture, SDKs, everything works.
* **Full interactive viewport**: install NVIDIA's
  [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)
  and connect it to the sim machine's IP (plain IP, no port - TCP 49100 and
  UDP 47998 are hardcoded and must be reachable). If the machine has multiple
  interfaces, set `livestream_public_ip` on the world.
* **Local GUI**: set `"headless": false` on the world (needs a display on the
  sim machine).

## How it works

Isaac Sim's Python API only runs inside Isaac Sim's own interpreter, and
Omniverse Kit wants to own the thread it runs on. So:

* `run.sh` launches the module with Isaac Sim's bundled python (found via
  `$ISAAC_SIM_PATH/python.sh` or `$ISAAC_PYTHON`), installing `viam-sdk` into
  it on first run.
* The **main thread** runs the simulation loop (`SimulationApp` boot, stepping,
  and a task queue). The **Viam module server** runs on a side thread, and all
  component calls are marshalled onto the sim thread.
* All models live in one module process and share the sim through a singleton,
  so arms/cameras/bases just name their world component and get attached.

## Machine requirements & automatic setup

On a standard Ubuntu 22.04/24.04 x86_64 machine, the module sets itself up:
when first installed, viam-server runs `first_run.sh`, which installs the
system libraries kit needs (vulkan/GL), the right python (via deadsnakes on
24.04), an NVIDIA driver if none is present (the validated 580 branch - newer
is not better here, see below), and Isaac Sim itself
(pip-installed into a venv under the module's data directory - 4.5.0 on
22.04, 5.0.0 on 24.04). `run.sh` finds that install automatically. The EULA
is accepted via environment variable.

Notes on the automatic setup:

* The Isaac Sim download is 10GB+. If it exceeds viam-server's default
  first-run timeout, set `"first_run_timeout": "2h0m0s"` on the module entry
  in your machine config.
* If the script had to install the NVIDIA driver, **reboot** before
  configuring the world component.
* Already have Isaac Sim? Set `ISAAC_SIM_PATH` (dir containing `python.sh`)
  or `ISAAC_PYTHON` in the module's environment variables and the script
  skips everything.

What the machine must already be/have (the script can't do these for you):

* Ubuntu 22.04 or 24.04 on x86_64 with an RTX-capable NVIDIA GPU (8GB+ VRAM
  minimum, RTX 4080+/L40 recommended), 32GB+ RAM, ~60GB free disk. See
  NVIDIA's [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
* An NVIDIA driver from a branch Isaac Sim validates against - **use the 580
  branch**. Newer branches (590/595+) are known to crash Isaac's RTX renderer
  on startup (`librtx.scenedb.plugin.so`) and break CUDA init
  (`cuDeviceGetUuid` Warp errors), see
  [isaac-sim/IsaacSim#537](https://github.com/isaac-sim/IsaacSim/issues/537).
  If you're on 595+: `sudo apt-get install -y nvidia-driver-580 && sudo reboot`.
* viam-server installed, machine online in the Viam app, running as root (or
  a user with passwordless sudo) for the apt/driver steps.
* Network access to pypi.nvidia.com, pypi.org, and NVIDIA's asset servers.
* Open ports for the livestream viewer if you want it: TCP 49100 (signaling)
  plus UDP 47998 (media) - both hardcoded in NVIDIA's streaming client.

No GPU/Isaac at all? `"mock": true` on the world runs the module anywhere for
development.

## Development without Isaac Sim (mock mode)

Set `"mock": true` on the world component and the module runs anywhere python
does - arms integrate joint targets over time, cameras produce synthetic
frames, bases accept velocity commands, and a gripper's `grab()` succeeds iff
its `mock_object_width_m` attribute is set (it is the width of the object
between the jaws, unset means nothing to grab). This is what the test suite
uses. Try it end to end with `PYTHONPATH=src python examples/pick_red_block.py
--mock`.

Set up a dev venv and run the checks with `make`:

```sh
uv venv --python 3.11 .venv && uv pip install -r requirements-dev.txt
# or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make fmt-check lint typecheck test
```

Tests run under the `pyproject.toml` config (`pythonpath = ["src"]`). The
`gpu` marker is skipped by default, so `make test` only runs mock-mode tests
and needs no Isaac Sim install. CI runs the same suite on both Python 3.10
and 3.11.

## Status / roadmap

- [x] world boot, stage loading, livestream, play/pause/reset, add_usd
- [x] arm joint control (UR family, Franka, arbitrary USD articulations)
- [x] RGB cameras
- [x] differential-drive bases
- [x] cloud builds / registry publishing (tag a release)
- [x] serve kinematics files (`GetKinematics`) so Viam's motion service can do
      IK and planning for simulated arms (all motion stays in Viam, not Isaac)
- [x] depth / point clouds from cameras
- [x] gripper support
