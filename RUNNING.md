# Running FoundationPose on Tracy

How to run the live 6D pose pipeline on Tracy's Orbbec camera, and what was
changed to make it work on this machine.

Companion document: `PROGRESS.md` is the dated work log with the reasoning
behind each decision. This file is the operational summary.

---

## 1. Quick start

Three terminals.

### Terminal 1 — camera driver

The depth stream **must be registered to colour**. Launch the Orbbec driver with:

```bash
ros2 launch orbbec_camera <your_camera>.launch.py depth_registration:=true
```

Verify before going further — all four of these must hold:

```bash
ros2 topic echo --once --field width  /camera/depth/image_raw   # 1920
ros2 topic echo --once --field height /camera/depth/image_raw   # 1080
ros2 topic echo --once --field encoding /camera/depth/image_raw # 16UC1
ros2 topic echo --once --field header /camera/depth/image_raw   # camera_color_optical_frame
```

If depth comes back **640x576** in `camera_depth_optical_frame`, registration is
off. Nothing downstream will be correct: the masks are cut from the 1920x1080
colour image, and unregistered depth has both a different resolution *and* a
different optical frame, so the mask would address the wrong pixels.

### Terminal 2 — the pipeline

```bash
cd /home/tracy/robokudo_foundation_pose
source env_fp_tracy.sh
ros2 run robokudo_ros main _ae=demo_tracy_cubes_live _ros_pkg=robokudo_foundation_pose
```

Note it is `robokudo_ros`, **not** `robokudo` — the launcher lives in the
`robokudo_ros` ament package. `ros2 run robokudo main` fails with
"Package 'robokudo' not found".

### Terminal 3 — asking for poses

With `QUERY_DRIVEN = True` (the default) the pipeline sits idle until it gets a
`robokudo_msgs/action/Query` goal on `/robokudo/query`, then processes **one
fresh frame** and replies. Ask it with:

```bash
cd /home/tracy/robokudo_foundation_pose
source env_fp_tracy.sh
python query_cube_poses.py               # one query
python query_cube_poses.py --repeat 10   # repeatability over 10 queries
```

To watch the overlay continuously while arranging the scene, set
`QUERY_DRIVEN = False` — the pipeline then runs frame after frame, but has no
query interface.

---

## 2. What the pipeline does

```
Orbbec camera
  └─ CollectionReaderAnnotator     colour + registered depth + camera_info → CAS
      └─ ColorBlobDetector         HSV threshold + metric size check → masks
          └─ FoundationPoseAnnotator   mask → 6D pose, then tracks it
```

The two cube assemblies are **both red and blue**, so colour cannot tell them
apart. The detector separates them by their *metric* size, measured from the
depth image and matched against the CAD extents:

| object id | mesh | extents (m) |
|---|---|---|
| 0 | `child_cube_0.ply` | 0.050 x 0.203 x 0.050 |
| 2 | `child_cube_2.ply` | 0.152 x 0.101 x 0.101 |

`operation_mode = 1`: estimate a pose for any object that has none, then refine
it on every later frame. The five modes are:

| mode | estimation | tracking | behaviour |
|---|---|---|---|
| 0 | no | no | Do nothing. The model is not even loaded, and no poses are produced — the detector still draws masks, so the pipeline looks alive while FoundationPose is inert |
| 1 | yes | yes | Estimate a pose for objects that have none, refine the previous pose otherwise |
| 2 | yes | no | Only estimate, and only for objects with no pose yet |
| 3 | yes | no | Only estimate, treating every object as new on every frame |
| 4 | no | yes | Only track; needs a pose from elsewhere, never creates one |

**Tracking does not actually engage in this pipeline.** `ColorBlobDetector` runs
with `clear_own_hypotheses`, so every frame it replaces its `ObjectHypothesis`
objects and last frame's `PoseAnnotation` goes with them. Every hypothesis
therefore arrives without a pose and takes the expensive `register()` path
(`est_refine_iter=5`, batch 64); `track_one()` (`track_refine_iter=2`) never
runs, which makes mode 1 behave like mode 3.

That is fine for checking accuracy — each frame is independent, so nothing
drifts — but it is the slow path, and it is the reason for any disappointing
framerate. Real tracking needs an `ObjectAssociator` to carry hypotheses across
frames (setting `clear_own_hypotheses = False` also works, but then hypotheses
pile up in the CAS).

### Which frame the poses are in

**`camera_color_optical_frame`** — and that is still true with
`POSES_IN_MAP_FRAME = True`.

The flag makes the CollectionReader look the viewpoint up and store
`cas.cam_to_world_transform`, but **nothing in this pipeline reads that value**.
The only consumer that affects pose output is `Pose2ODConverter`
(`robokudo/utils/annotation_conversion.py:251`), which runs inside
`GenerateQueryResult` — not currently part of the pipeline. The
`PoseAnnotation` values stay in the camera optical frame because
`cam_r_w2c` / `cam_t_w2c` are deliberately unset.

So the flag is a **prerequisite for the query handoff, not a switch for it**.
Set it once Tracy is on (it is valid then, and a broken TF will surface as
lookup errors), but expect the reported frame to change only when
`QueryAnnotator()` and `GenerateQueryResult()` are added to the pipeline.

**Do not** set `cam_r_w2c` / `cam_t_w2c` on the FoundationPose annotator to
compensate for a missing transform. Two independent camera-to-world transforms
exist and they compound:

1. `FoundationPoseAnnotator` applies `twc` built from those parameters.
2. `Pose2ODConverter` applies `cas.cam_to_world_transform` from live TF.

Setting both applies the transform twice. The parameters are deliberately left
unset so the TF path is the single source of truth.

---

## 3. What FoundationPose returns, and checking it

**A full 6D pose per object:** position (x, y, z in metres) plus orientation (a
quaternion, x y z w). One pose per object (`est_num_pose_hypothesis = 1`).

It is the pose of the **mesh's own frame**. Both cube meshes were recentred on
their bounding box, so:

- **position = the geometric centre of the object's bounding box** — not a
  corner, not the bottom face;
- **orientation = the mesh axes.** For `child_cube_0` the 20.3 cm long side is
  mesh **y**; for `child_cube_2` the 15.2 cm side is mesh **x**.

Inside the pipeline it is stored in `camera_color_optical_frame`.
`GenerateQueryResult` transforms it into **`map`** (when
`POSES_IN_MAP_FRAME = True`) and sends each object as an `ObjectDesignator`:

| field | content |
|---|---|
| `type` | classname, `child_cube_0` / `child_cube_2` |
| `pose[0]` | `geometry_msgs/PoseStamped`, `frame_id: map` |
| `pose_source[0]` | `FoundationPose` |

The planning team reads `result.res[i].pose[0]`. Two details to pass on:
the stamp carries **whole seconds only** (`Pose2ODConverter` drops the
nanoseconds), and **every detected object is returned** — the goal's contents
are not used as a filter.

### The reference: the `table` frame

The robot's TF root is `table`; `map -> table` is a pure +0.880 m lift with no
rotation, so the `table` frame's origin lies on the tabletop and **a `table`
z coordinate is a height above the table**. This was cross-checked: the camera
sits 0.894 m above `table` with its optical axis cos 0.921 from vertical,
predicting 0.971 m to the table along the axis; the depth sensor measured
0.970 m.

### Checking a pose

`query_cube_poses.py` prints, per object, exactly what the planners receive and
then:

- position in the `table` frame;
- **expected height** of the centre if the object rests on the table — worked
  out from the mesh and the *estimated* orientation — and the **vertical
  error**. A cube lying flat should come out at half its thickness: 25 mm for
  the bar, 50.5 mm for the assembly on a 10.1 cm face, 76 mm standing on its
  15.2 cm side;
- which mesh axis points up and its tilt — should be near 0 deg for a cube
  lying on a face;
- the yaw of the longest horizontal side, folded modulo 180 deg (90 deg for a
  square footprint), for comparison with a protractor or tape measure.

The procedure: place a cube flat on the table, measure its centre's x/y from
the `table` origin, run the script, compare. Then `--repeat 10` for spread.

## 4. Tuning

All in `demo_tracy_cubes_live.py`:

| constant | meaning |
|---|---|
| `MESH_DIR` | where the `.ply` files live |
| `OBJECTS` | id → (mesh, extents, classname); ids must match the detector's `class_id` |
| `POSES_IN_MAP_FRAME` | `False` until Tracy publishes TF; on its own it does not change the reported frame (see above) |
| `COLOR2DEPTH_RATIO` | `(1.0, 1.0)` with registered depth at colour resolution |
| `QUERY_DRIVEN` | `True`: answer `Query` goals, one frame each; `False`: run continuously |
| `MAX_OBJECT_DISTANCE` | `1.2` m — measured scene sits at ~0.96 m, background beyond ~2.4 m |

Detector parameters worth touching if detections are poor: `hsv_ranges`,
`min_pixel_area`, `extent_match_tolerance` (default 0.05 m mean error).

---

## 5. Environment

`source env_fp_tracy.sh` sets up everything. It:

1. puts `/usr/bin` first so Python 3.12 is not shadowed,
2. sources ROS 2 Jazzy and `/home/tracy/ros2_ws/install/setup.bash`,
3. points `CUDA_HOME` at the pip CUDA toolkit,
4. sets `TORCH_CUDA_ARCH_LIST=8.6` for the RTX 3080,
5. activates `~/.virtualenvs/robokudo` **last**, so `ros2 run`'s
   `#!/usr/bin/env python3` shebang picks up the venv's Python.

Use `source`, not `./env_fp_tracy.sh` — a subshell would discard the variables.

Expected output:

```
python : /home/tracy/.virtualenvs/robokudo/bin/python (Python 3.12.3)
ros    : jazzy
nvcc   : Cuda compilation tools, release 13.0, V13.0.88
```

If it prints `nvcc : MISSING`, run `bash setup_cuda_home.sh` once.

### The CUDA toolkit

There is **no system CUDA toolkit** and no root access was used. `nvidia-smi`
reporting "CUDA Version: 13.0" is the *driver's* ceiling, not an installed
compiler — `dpkg -l | grep -c cuda` is 0 on this machine.

Instead `cuda-toolkit[nvcc]` provides a complete toolkit inside site-packages,
and `setup_cuda_home.sh` assembles it into a conventional `CUDA_HOME` at
`.venv_fp/cuda`, fixing two things: the wheel calls the library directory `lib`
where torch expects `lib64`, and it ships only versioned sonames
(`libcudart.so.13`) so plain `-lcudart` cannot link.

**All CUDA components must be the same version (13.0.88).** A mismatch produces:

```
ptxas fatal : Unsupported .version 9.3; current version is '9.0'
```

which means `nvidia-nvvm` / `nvidia-cuda-crt` drifted ahead of `nvidia-cuda-nvcc`.

> **`.venv_fp/` is not the runtime environment.** It is kept only as the
> container for the pip CUDA toolkit. The pipeline runs in
> `~/.virtualenvs/robokudo`.

### Installing anything else into the venv

**Always pin numpy:**

```bash
pip install <package> "numpy==1.26.4"
```

The robokudo venv sits on numpy 1.26.4, matching the apt-installed packages that
`--system-site-packages` exposes. An unpinned install can silently upgrade numpy
and break every C extension compiled against 1.x, with
`numpy.core.multiarray failed to import` or
`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`.
This happened once during setup (`pip install transformations` pulled numpy
2.5.3 and broke trimesh and open3d) and was repaired by reinstalling
`numpy==1.26.4`.

---

## 6. Changes made to the repository

| file | status | what |
|---|---|---|
| `robokudo_foundation_pose/transforms.py` | **new** | Vendored `so3_exp_map`, `rotation_6d_to_matrix`, `hat` — removes the pytorch3d dependency |
| `learning/training/predict_pose_refine.py` | modified | line 20 now imports those from `robokudo_foundation_pose.transforms` instead of `pytorch3d.transforms` |
| `descriptors/analysis_engines/demo_tracy_cubes_live.py` | **new** | The live AE for Tracy's camera and the two cubes, with `QueryAnnotator` / `GenerateQueryResult` |
| `query_cube_poses.py` | **new** | Query client that checks poses against the table |
| `env_fp_tracy.sh` | **new** | Environment for this machine, replacing `env_fp.sh` |
| `setup_cuda_home.sh` | **new** | Builds the pip `CUDA_HOME` tree; idempotent |
| `PROGRESS.md` | **new** | Dated work log |
| `RUNNING.md` | **new** | This file |

`env_fp.sh` is left untouched but is **not usable here** — it points at a
`/home/student` user and a CUDA 11.8 system toolkit, neither of which exist.

### Why pytorch3d was dropped

It never built: its `pulsar` point renderer fails to link under CUDA 13
(`undefined reference to pulsar::Renderer::render<true>`), roughly 15 minutes per
attempt. `FORCE_CUDA=0` does not avoid it — pytorch3d compiles CUDA whenever
`CUDA_HOME` is set.

The repo used exactly two names from it, both pure tensor math with no compiled
kernel. They are reproduced faithfully in `transforms.py` and checked against
`scipy.spatial.transform.Rotation`: agreement is **9.99e-16** (machine
precision) for rotation angles above 0.01 rad. Below that, deviation reaches
6.2e-08 — this is upstream pytorch3d's own `eps=1e-4` clamp on the squared
angle, reproduced deliberately, and amounts to ~3.6e-6 degrees. Rotations stay
orthonormal with `det == 1` to 6.3e-10 throughout.

Only `so3_exp_map` is actually reached, since the refiner config sets
`rot_rep: axis_angle`.

### Import order in the AE

`robokudo.descriptors` must be imported **before**
`robokudo.annotators.collection_reader`, and the camera config is obtained via
`CrDescriptorFactory.create_descriptor("orbbec", ...)` rather than by importing
`robokudo.descriptors.camera_configs.config_orbbec` directly. Reaching the
camera configs the other way round trips a circular import:

```
ImportError: cannot import name 'CollectionReaderAnnotator' from partially
initialized module 'robokudo.annotators.collection_reader'
```

The `demo_colored_cubes.py` and `demo_tetris_noctis.py` demos still use the old
direct-import style and will hit this against the current robokudo.

---

## 7. Workspace / build

Both packages are symlinked into the ROS workspace:

```
/home/tracy/ros2_ws/src/robokudo_foundation_pose -> <repo>/robokudo_foundation_pose
/home/tracy/ros2_ws/src/robokudo_cad_data        -> <repo>/robokudo_cad_data
```

A colcon build is **required for the `ros2 run` path** — that is how robokudo
resolves `_ros_pkg` through the ament index:

```bash
cd /home/tracy/ros2_ws
colcon build --merge-install --symlink-install --packages-select robokudo_foundation_pose
```

`--merge-install` is required because this workspace's `install/` already uses
the merged layout; without it colcon refuses to build. `--symlink-install` means
Python edits take effect without rebuilding — but **adding a new file still
needs a rebuild**.

Nothing in this package compiles C/C++, so colcon is only about ament
registration. The compiled dependency is nvdiffrast, which JIT-builds its CUDA
kernels on first use (a one-off delay, cached in `~/.cache/`).

For standalone scripts that bypass ROS (e.g. `run_cubes_offline.py`), no build
is needed, but the inner package must be on the path:

```bash
export PYTHONPATH=/home/tracy/robokudo_foundation_pose/robokudo_foundation_pose:$PYTHONPATH
```

Without it, `import robokudo_foundation_pose` resolves to the outer *ament*
folder as a namespace package, which has no `annotator` submodule.

---

## 8. Troubleshooting

| symptom | cause |
|---|---|
| `Package 'robokudo' not found` | use `ros2 run robokudo_ros main` |
| `nvcc : MISSING` from the env script | run `bash setup_cuda_home.sh` |
| `ptxas fatal : Unsupported .version 9.3` | CUDA wheels drifted; pin all to 13.0.88 |
| `numpy.core.multiarray failed to import` | numpy was upgraded past 1.x; reinstall `numpy==1.26.4` |
| `cannot import name 'CollectionReaderAnnotator'` | circular import; import `robokudo.descriptors` first |
| Poses stamped `camera_color_optical_frame`, not `map` | Expected until `GenerateQueryResult()` is in the pipeline. Also needs Tracy on and `POSES_IN_MAP_FRAME = True` |
| Pipeline runs, masks drawn, but no poses ever appear | `operation_mode` is `0` |
| `query failed: no answer within N s` | the action server exists (robokudo always creates it) but nothing consumes goals — `QUERY_DRIVEN` is `False`, or a pipeline started before the change is still running |
| Overlay only updates when a query is sent | expected with `QUERY_DRIVEN = True`; set it to `False` to watch continuously |
| Poses wildly wrong / nonsense depth | depth registration off; relaunch driver with `depth_registration:=true` |
| No detections | check `MAX_OBJECT_DISTANCE`, `hsv_ranges`, `min_pixel_area` against the actual scene |

---

## 9. Not done yet

- **NOCTIS is not integrated.** `NoctisDetectionReader` replays pre-computed
  BOP-style JSON from disk and cannot run live; `demo_tetris_noctis.py` is
  therefore not a live AE. See `PROGRESS.md` for the integration options — the
  recommended one is to run NOCTIS as a separate ROS 2 node behind
  `robokudo_msgs/action/GenericImgProcAnnotator` and write a client annotator,
  which keeps its dependencies out of this venv.
- **The query handoff is wired but not yet validated on real cubes.** Poses are
  served over `robokudo_msgs/action/Query` on `/robokudo/query`;
  `query_cube_poses.py` is the checking tool, and
  `robokudo/descriptors/analysis_engines/robokudo_cram_integration.py` is a
  reference client for the planning team.
- **Accuracy has not been validated** against ground truth, and the cubes are
  symmetric — FoundationPose may return an orientation that is correct up to a
  symmetry. Confirm the planners' grasping tolerates that, or supply
  `symmetry_tfs`.
