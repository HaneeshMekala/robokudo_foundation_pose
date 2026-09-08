# Project Tracker — RoboKudo FoundationPose

Running log of what has been done, what the environment looks like, and what is
still blocking the pipeline. Newest entries at the top of the Changelog.

---

## Changelog

### 2026-09-08 — Q&A: ColorBlobDetector explained, NOCTIS-live sketch (no code changes)

Answered questions about the current pipeline; nothing in the repo was touched.

#### `demo_tracy_cubes_live.py` is the main entry point

It's the `AnalysisEngine` RoboKudo loads for
`ros2 run robokudo_ros main _ae=demo_tracy_cubes_live _ros_pkg=robokudo_foundation_pose`.
It wires camera reader -> detector -> pose estimator into one pipeline in
`implementation()`; everything else is a building block it assembles.

#### `ColorBlobDetector` is standing in for NOCTIS

Both annotators exist only to produce the same output — an `ObjectHypothesis`
carrying an `ImageROI` (rect + mask) and a `Classification` (class_id,
classname, confidence). `FoundationPoseAnnotator` reads `ObjectHypothesis` off
the CAS and does not care which annotator produced it, so the two are
interchangeable segmentation front-ends:

- `NoctisDetectionReader` — replays **precomputed** NOCTIS detections from JSON
  on disk, one frame per iteration. Cannot run live.
- `ColorBlobDetector` — segments **live**, colour + geometry instead of a
  learned model. What makes `demo_tracy_cubes_live.py` live-capable today.

#### How `ColorBlobDetector.update()` works, step by step

1. **`segment()`** — HSV-threshold the colour image (OR across configured
   colour bands; red needs two bands since it wraps hue 0/180), morphological
   close then open to bridge specular highlights and drop speckle, then
   `cv2.connectedComponents` to label blobs.
2. **`measure()`** — for each blob past `min_pixel_area`, resize the mask to
   depth resolution, back-project the masked depth pixels into 3D camera space
   (`p = z * K^-1 * (u,v,1)`), drop halo pixels near the silhouette
   (`depth_outlier_band`), then take the spread along the patch's principal
   axes via SVD/PCA — rotation-invariant, so it works regardless of how the
   object is oriented toward the camera.
3. **`identify()`** — compare only the two *largest* measured extents (the
   third/depth-facing axis is unreliable, foreshortened by self-occlusion)
   against each configured CAD model's extents; best match under
   `extent_match_tolerance` (0.05 m default) wins. **This is exactly what
   separates the two cubes here** — both are red+blue so colour cannot tell
   them apart, but 0.203 m vs 0.152 m longest-side is decisive.
4. **`make_hypothesis()`** — crop the mask to its bounding rectangle exactly
   (FoundationPose pastes it back with `full_mask[y:y+h, x:x+w] = crop`, so
   rect and mask must agree pixel-for-pixel), confidence scaled linearly from
   match error (perfect -> 1.0, at-tolerance -> ~0).
5. Old hypotheses from this annotator are cleared (`clear_own_hypotheses`),
   new ones appended to `cas.annotations` for `FoundationPoseAnnotator` to
   consume next.

#### Sketch: what a live NOCTIS annotator would look like

Same shape as `ColorBlobDetector` — a `core.BaseAnnotator` subclass whose
`update()` reads `CASViews.COLOR_IMAGE`/`DEPTH_IMAGE` and writes
`ObjectHypothesis` objects to `cas.annotations`. Only `segment()`'s role
changes:

```
ColorBlobDetector.segment()  ->  cv2 threshold, in-process, milliseconds
NOCTIS annotator's core step ->  send rgb+depth to NOCTIS, get masks back
```

Two structural options (from the earlier integration plan, Phase 3):

- **In-process** — import NOCTIS's model directly inside `update()`, like
  `FoundationPoseAnnotator` does with its own networks. Simplest, but risks a
  torch/CUDA dependency clash with FoundationPose in the same venv — a real
  risk given the trouble already had installing FoundationPose alone.
- **Out-of-process (recommended)** — NOCTIS runs as a separate ROS 2
  node/service, naturally behind `robokudo_msgs/action/GenericImgProcAnnotator`
  (rgb+depth in; bounding boxes + masks + class ids out — the right shape
  already). The new annotator is a thin client that calls it and converts the
  response. Keeps NOCTIS's dependencies out of this venv entirely.

Either way, the mask-handling logic should be **copied from
`NoctisDetectionReader.build_hypothesis`**, not reinvented: it decodes the COCO
RLE segmentation NOCTIS produces, and critically **recomputes the rectangle
from the nonzero pixels of the decoded mask** rather than trusting NOCTIS's own
`bbox` field, which is measurably looser than the mask and would violate the
exact-crop-matches-rectangle contract FoundationPose depends on. The
`category_id -> obj_id` mapping (NOCTIS categories come from its rendered
template folders, not `mesh_obj_ids`) carries over unchanged too.

Concretely: a new `annotator/noctis_live_detector.py`, structurally parallel to
`color_blob_detector.py`'s `update()`/`make_hypothesis()` shape, but internally
reusing `NoctisDetectionReader.decode_mask()`/`build_hypothesis()`'s
mask-and-rectangle logic against live frames instead of a JSON file. Not yet
written — this is a plan, still Phase 3 of the integration plan above.

---

### 2026-09-08 — Pipeline runs live; venv pivot; `RUNNING.md` written

Depth registration enabled on the driver — depth is now **1920x1080, `16UC1`, in
`camera_color_optical_frame`**, matching colour exactly, so
`COLOR2DEPTH_RATIO = (1.0, 1.0)` is correct. Quality: 51.6% valid globally
(background holes are normal for registration), **97.1% valid in the centre
200x200 at 0.970 m**. Blocker A closed.

Package built and running:

```
cd /home/tracy/ros2_ws
colcon build --merge-install --symlink-install --packages-select robokudo_foundation_pose
```

`--merge-install` is mandatory here: this workspace's `install/` already uses the
merged layout and colcon refuses otherwise.

#### Three things that had to be fixed to launch

1. **The launcher is `robokudo_ros`, not `robokudo`.** `ros2 run robokudo main`
   fails with "Package 'robokudo' not found"; the ament package providing the
   `main` executable is `robokudo_ros`.
2. **Circular import in robokudo.** Importing
   `robokudo.descriptors.camera_configs.config_orbbec` directly trips
   "cannot import name 'CollectionReaderAnnotator' from partially initialized
   module". The AE now imports `robokudo.descriptors` first and builds the reader
   through `CrDescriptorFactory.create_descriptor("orbbec", ...)`, the pattern
   `stacking_robokudo.py` uses. **The existing `demo_colored_cubes.py` and
   `demo_tetris_noctis.py` still use the old style and will hit this.**
3. **Undeclared runtime dependencies of `semantic_digital_twin`:** `mujoco`,
   `giskardpy_bullet_bindings`, `giskardpy`, and `transformations` for this repo.

#### Abandoned the dedicated `.venv_fp` (reversing the earlier choice)

The isolated venv could not be made to work. `semantic_digital_twin`'s modern
tree (jax, scipy 1.18, ortools, rerun, plyfile) requires **numpy >= 2.0** while
numba requires **<= 2.2**, forcing numpy 2.2.6 — but `--system-site-packages`
exposes apt packages compiled against **numpy 1.x**, each of which then dies with
`numpy.core.multiarray failed to import`. matplotlib was the first of an
open-ended set, and pip will not even shadow it without `-I` because the system
copy satisfies the requirement.

`~/.virtualenvs/robokudo` sits on numpy 1.26.4, consistent with the system, and
already had the entire robokudo / semantic_digital_twin / giskardpy / krrood /
mujoco chain working. It needed only nvdiffrast, kornia, warp-lang, h5py,
imageio, pycocotools and transformations. **Dropping pytorch3d removed the main
reason a separate env looked attractive.**

`.venv_fp/` is retained **only** as the container for the pip CUDA toolkit;
`CUDA_HOME` still points into it. `env_fp_tracy.sh` now activates the robokudo
venv.

#### Incident: numpy upgrade broke the shared venv

`pip install transformations` (unpinned) upgraded numpy to 2.5.3 in the robokudo
venv, breaking `trimesh` and `open3d`. Repaired with `pip install numpy==1.26.4`;
verified numpy 1.26.4, scipy 1.11.4, trimesh 4.12.2, open3d 0.19.0, torch
2.12.1+cu130, nvdiffrast, robokudo and semantic_digital_twin all import again.

**Rule for this venv: every install must carry `"numpy==1.26.4"` explicitly.**
Recorded in `RUNNING.md`.

#### New: `RUNNING.md`

Operational guide — quick start, the pipeline's shape, tuning constants,
environment and CUDA notes, the full list of repository changes, workspace/build
instructions, a troubleshooting table, and what is still outstanding.

---

### 2026-09-08 — Phase 0 COMPLETE: FoundationPose loads and initialises on the RTX 3080

`FoundationPoseAnnotator.setup()` runs clean with both cube meshes, and the two
networks are on the GPU with their trained weights:

```
FoundationPose ready in 0.5s
meshes loaded: [0, 2]
  obj 0: extents [0.05   0.05   0.2031]
  obj 2: extents [0.101 0.101 0.152]
  scorer  :  15.77M params | device cuda:0 | mean|w| 0.07072
  refiner :  16.83M params | device cuda:0 | mean|w| 0.07307
```

#### pytorch3d dropped in favour of a vendored module

pytorch3d never built: its `pulsar` point renderer fails to link under CUDA 13
(`undefined reference to pulsar::Renderer::render<true>`), ~15 minutes per
attempt. `FORCE_CUDA=0` does not help — pytorch3d compiles CUDA whenever
`CUDA_HOME` is set and ignores that variable.

The repo used exactly two names from it, both pure tensor math:
`rotation_6d_to_matrix` and `so3_exp_map`. New **`robokudo_foundation_pose/transforms.py`**
reproduces them faithfully (with `hat`), and
`learning/training/predict_pose_refine.py:20` now imports from there. **pytorch3d
is no longer a dependency of this project.**

Verified against `scipy.spatial.transform.Rotation` as an independent reference:

| check | result |
|---|---|
| `so3_exp_map(0) == I` | exact (0.00e+00) |
| vs `scipy.from_rotvec`, angle >= 0.01 rad | 9.99e-16 — machine precision |
| vs scipy inside the `eps` clamp (< 0.01 rad) | <= 6.2e-08, shrinking with angle |
| orthonormality / `det == 1` everywhere | <= 6.3e-10 |
| `hat(a) @ b == cross(a, b)` | exact |
| `rotation_6d_to_matrix` round-trip from a known R | 4.4e-16 |
| batch shapes `(7,5,6) -> (7,5,3,3)`, cuda float32 | OK |

The sub-0.01 rad deviation is **upstream pytorch3d's own behaviour**, not an
approximation introduced here: it clamps the squared angle at `eps=1e-4` to keep
`1/t` finite. Worst case ~6e-8 rad ≈ 3.6e-6 degrees — irrelevant for pose
estimation. Only `so3_exp_map` is actually reached, since the refiner config sets
`rot_rep: axis_angle`.

#### `.venv_fp` dependency resolution

`robokudo` pulls `semantic_digital_twin`, which drags a large tree (jax, mlflow,
PySide6, ortools, rerun, mujoco...). Two traps:

- A fresh resolve took **numpy 2.5.3**, which breaks `numba` ("Numba needs NumPy
  2.2 or less"). But `semantic_digital_twin`'s tree (jax, scipy 1.18, ortools,
  rerun, plyfile) requires numpy >= 2.0, so 1.26.4 is not an option either.
  **`numpy==2.2.6` is the window that satisfies both** and is now pinned.
- `mujoco` is an undeclared runtime import of `semantic_digital_twin.exceptions`;
  installed explicitly.

Also removed `/home/tracy/robokudo_foundation_pose/pytorch3d/` — a 100 MB source
clone pip left behind when a build was interrupted. It was a namespace package
shadowing any real `pytorch3d` import (`pytorch3d.__file__` was `None`), which
made the module look installed when it was not.

#### Working environment

```
source env_fp_tracy.sh
```
gives Python 3.12.3, ROS 2 Jazzy, `nvcc` 13.0.88, torch 2.12.1+cu130 with CUDA
available, plus nvdiffrast 0.4.0, kornia 0.8.3, warp 1.17.0 (sees the 3080 as
sm_86), open3d 0.19.0, trimesh 5.1.0, numpy 2.2.6, mujoco 3.12.0, robokudo 1.0.0
and semantic_digital_twin 0.0.6.

Note the offline/standalone entry points still need
`PYTHONPATH=$REPO/robokudo_foundation_pose` (the namespace-package issue from the
first entry); the `ros2 run` path will not, once the package is colcon-built.

#### Only remaining blocker before live poses

`depth_registration:=false` on the Orbbec driver. Relaunch with
`depth_registration:=true`, re-check the depth width/height and frame_id, set
`COLOR2DEPTH_RATIO` in `demo_tracy_cubes_live.py` accordingly, then run it.

---

### 2026-09-08 — CUDA toolchain solved without root; `.venv_fp` built

**`nvidia-smi` does not indicate an installed toolkit.** Its "CUDA Version: 13.0"
is the *driver's* maximum supported runtime, printed whether or not a toolkit
exists. Hard evidence on this machine: `command -v nvcc` → not found,
`/usr/local/` has no `cuda*`, and `dpkg -l | grep -c cuda` → **0**. The only
NVIDIA packages installed are driver libraries (`libnvidia-*`). The compiler
ships in the toolkit, which is a separate install.

**Resolution: a pip-only, no-sudo CUDA toolkit.** `cuda-toolkit[nvcc]` ships a
complete toolkit — `bin/`, `include/`, `lib/`, `nvvm/` — inside site-packages at
`.venv_fp/lib/python3.12/site-packages/nvidia/cu13`. Two fixups are needed, both
automated in the new **`setup_cuda_home.sh`**, which builds `.venv_fp/cuda`:

1. The wheel names the library directory `lib`; torch's `cpp_extension` and
   nvdiffrast's JIT build expect `lib64`.
2. The wheels ship only versioned sonames (`libcudart.so.13`), so a plain
   `-lcudart` fails to link. Unversioned aliases are symlinked in.

**Version skew trap (cost a debugging cycle):** pip resolved
`nvidia-nvvm` and `nvidia-cuda-crt` to **13.3.73** while `nvidia-cuda-nvcc` and
`ptxas` were **13.0.88**. The newer frontend emits PTX `.version 9.3`, which the
older ptxas rejects:

```
ptxas fatal : Unsupported .version 9.3; current version is '9.0'
```

Fix: pin them to the nvcc version — `nvidia-nvvm==13.0.88`,
`nvidia-cuda-crt==13.0.88`. **Any CUDA component added later must match 13.0.88.**

**Verified end to end:** a hand-written `__global__` kernel compiled with
`nvcc -arch=sm_86` and ran on the RTX 3080, returning `0.0 2.0 4.0 6.0`. So the
toolchain is genuinely working, not merely present.

**`.venv_fp` created** (`--system-site-packages`, Python 3.12.3) with
torch 2.12.1+cu130 / torchvision 0.27.1+cu130, `torch.cuda.is_available()` True
on the RTX 3080. `setuptools` pinned `<80` (79.0.1) because colcon-core requires
it and the venv would otherwise shadow the system copy.

**New files:**
- `setup_cuda_home.sh` — builds the `CUDA_HOME` tree; idempotent.
- `env_fp_tracy.sh` — replaces `env_fp.sh` for this machine: `/home/tracy/ros2_ws`,
  the pip `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST=8.6` for the 3080, and no g++-11
  pinning (gcc 13 is fine for CUDA 13).

Usage:

```
bash setup_cuda_home.sh      # once, or after changing CUDA wheels
source env_fp_tracy.sh
```

---

### 2026-09-08 — Live cube AE written; assets verified; CUDA is the last blocker

**Correction to the first entry: the weights are present.** Both
`model_best.pth` files are in place (66 MB and 182 MB) under
`weights/2023-10-28-18-33-37/` and `weights/2024-01-11-20-02-45/`. The earlier
"weights absent" finding is superseded.

**Cube meshes are already FoundationPose-ready.** `child_cube_0.ply` and
`child_cube_2.ply` sit at the repository root and need no preparation —
`prepare_cube_meshes.py` does not have to be run:

| mesh | verts/faces | extents (m) | bbox centre | watertight |
|---|---|---|---|---|
| `child_cube_0.ply` | 24 / 40 | 0.050 x 0.203 x 0.050 | (0,0,0) | yes |
| `child_cube_2.ply` | 97 / 186 | 0.152 x 0.101 x 0.101 | (0,0,0) | yes |

Already in metres and recentred on the bounding box, so
`default_mesh_scale_factor = 1.0`. Extents match `demo_colored_cubes.py`.

**Both assemblies are red *and* blue**, not one colour each: vertex colours are
RGBA (30,60,220) and (220,30,30), i.e. OpenCV-HSV ~(115,220,220) and
~(0,220,220). Those fall inside `ColorBlobDetector`'s default `hsv_ranges`
bands, so the defaults need no tuning. Colour therefore cannot separate the two
objects — **the metric size check is what distinguishes them**, and their
extents are far enough apart (0.203 vs 0.152 longest side, and very different
cross-sections) that `extent_match_tolerance = 0.05` should hold.

**Live scene depth sampled** from `/camera/depth/image_raw`: 62.3% valid pixels,
median 0.955 m, centre-patch median 0.965 m, p95 2.41 m. So the working surface
is at ~0.96 m and `max_distance = 1.2` cleanly separates it from background.

**TF blocker resolved in principle:** Tracy is simply switched off. Turning it on
publishes the rest of the tree, and both robot and camera are fixed (arm-only
manipulation), so a single static camera→base calibration will hold. Until then,
`POSES_IN_MAP_FRAME = False` and poses come out in
`camera_color_optical_frame`, which is sufficient for checking the estimator.

**New file:** `descriptors/analysis_engines/demo_tracy_cubes_live.py` — Orbbec
config + `ColorBlobDetector` + `FoundationPoseAnnotator`, `operation_mode = 1`,
`cam_r_w2c`/`cam_t_w2c` deliberately unset. Compiles clean. Run with:

```
ros2 run robokudo main _ae=demo_tracy_cubes_live _ros_pkg=robokudo_foundation_pose
```

**Remaining blockers, in order:**
1. **CUDA toolkit + `nvdiffrast`, `pytorch3d`, `warp-lang`, `kornia`** — still
   entirely absent; `nvcc` is nowhere and no `/usr/local/cuda*` exists. torch
   ships only CUDA *runtime* wheels (`nvidia-cuda-runtime` 13.0.96,
   `nvidia-cuda-nvrtc` 13.0.88), no compiler. A no-sudo route exists:
   `nvidia-cuda-nvcc==13.0.88` + `nvidia-cuda-crt` + `nvidia-cuda-cccl` are on
   PyPI at versions matching torch's 13.0.88, and can be assembled into a
   `CUDA_HOME`. The apt route needs NVIDIA's repo added and sudo.
   Note `nvdiffrast` JIT-compiles at first use, so a clean `pip install` proves
   nothing — `dr.RasterizeCudaContext()` is the real test. `pytorch3d` compiles
   at install time and will fail loudly without nvcc.
2. **`depth_registration:=false`** on the running driver — relaunch required
   before any pose is trustworthy.

---

### 2026-09-08 — Live camera verified on Tracy: two blockers found

Orbbec driver is up (`/camera/camera`, node container `/camera/camera_container`,
rviz running). Topic names match `OrbbecCameraConfig` defaults **exactly** —
`/camera/color/image_raw/compressed`, `/camera/depth/image_raw` (`depth_hints="raw"`),
`/camera/color/camera_info`. No config change needed for topics.

**Verified stream properties**

| | colour | depth |
|---|---|---|
| resolution | 1920 x 1080 | 640 x 576 |
| frame_id | `camera_color_optical_frame` | `camera_depth_optical_frame` |
| encoding | compressed (BGR) | `16UC1` |
| aspect | 1.778 | 1.111 |

Intrinsics are real and populated (fx 1117.02, fy 1116.93, cx 950.57, cy 527.61,
non-zero distortion) — the *intrinsic* calibration is genuinely done.
Depth `16UC1` is millimetres, which matches the annotator's `* 1e-3`.

#### BLOCKER A — depth is not registered to colour

`ros2 param get /camera/camera depth_registration` → **False**
(`align_mode: SW`, `align_target_stream: COLOR` are already correct; only the
switch is off). There is no `/camera/aligned_depth_to_color/...` topic.

Why `COLOR2DEPTH_RATIO` cannot paper over this: it is a single `(sx, sy)` scale
that assumes depth is the *same view* at a lower resolution. Here the two streams
have different aspect ratios (1.778 vs 1.111) **and** sit in different optical
frames, so the relationship is a different FOV plus a rigid transform — not a
scale. Masks cut from the colour image would land on the wrong depth pixels, and
`FoundationPose.register()` would get garbage geometry.

Fix: relaunch the driver with `depth_registration:=true`, then **re-measure**
the depth resolution and frame_id and set `color2depth_ratio` from the new
numbers (it becomes `(1.0, 1.0)` only if depth comes back at 1920x1080).
Not changed live — the camera is in use and this param needs a driver restart.

#### BLOCKER B — there is no `map` frame; the TF tree is camera-only

`ros2 run tf2_tools view_frames` returns the complete tree:

```
camera_link
 └─ camera_depth_frame
     ├─ camera_color_frame ── camera_color_optical_frame
     └─ camera_depth_optical_frame
```

That is all of it. Root is `camera_link`. No `map`, no robot, no table.
`/tf` carries nothing dynamic; `/robot_description` is published but nothing is
broadcasting the robot's transforms. `tf2_echo map camera_color_optical_frame`
times out.

Consequence for the handoff: `lookup_viewpoint=True` (the Orbbec default) will
fail to resolve `camera_color_optical_frame → map`, so
`cas.cam_to_world_transform` stays `None`, and `Pose2ODConverter` falls back to
stamping poses in `camera_color_optical_frame`
(`annotation_conversion.py:251-266`). The planning team would receive poses in
the camera frame while the field says so — correct, but useless for manipulation.

So "the camera is calibrated" holds **intrinsically** but not **extrinsically**:
nothing relates the camera to the robot or the world.

Two ways out, to settle with the planning team:
1. Bring up the robot's TF (robot_state_publisher + whatever localises `map`) and
   extrinsically calibrate the camera into that tree. Correct for a moving camera.
2. If the camera is static, publish one measured/calibrated
   `static_transform_publisher` from `map` (or the robot base) to `camera_link`.
   Much faster, valid only while the camera does not move.

Explicitly rejected: setting `cam_r_w2c`/`cam_t_w2c` on the annotator to fake it.
That reintroduces the double-transform hazard and hides a missing calibration.

#### Revised phase status

- Phase 1 (camera bring-up): **partially passed.** Topics/encodings/intrinsics
  confirmed good; registration and TF are the two open items above.
- Phase 0 (CUDA, nvdiffrast/pytorch3d, weights) is untouched and still blocking.
- Phase 4's frame_id assertion is now known to fail today — Blocker B is its cause.

---

### 2026-09-08 — Integration plan: NOCTIS + FoundationPose live on Tracy → planning team

Goal: run NOCTIS (segmentation) + FoundationPose (6D pose) live on Tracy's
camera and hand the resulting object poses to the planning/manipulation team.

#### Architecture decided

```
Orbbec camera (Tracy)
   └─ CollectionReaderAnnotator      colour + depth + camera_info + TF → CAS
       └─ QueryAnnotator             planning team's Query goal arrives here
           └─ NOCTIS detector        → ObjectHypothesis + ImageROI mask
               └─ FoundationPoseAnnotator   → PoseAnnotation (camera frame)
                   └─ GenerateQueryResult   → ObjectDesignator[] w/ PoseStamped in 'map'
                                              ↑ planning team consumes this
```

The handoff contract is **`robokudo_msgs/action/Query`** on `/robokudo/query`.
The planning team sends a `Query.Goal` (`obj.type`, `obj.color`, ...) and gets
back `Query.Result.res`, an `ObjectDesignator[]` whose `pose` field is a
`geometry_msgs/PoseStamped[]`. `robokudo/descriptors/analysis_engines/robokudo_cram_integration.py`
is a working client for exactly this and is the reference for their side.

#### Key findings that shape the plan

- **Tracy's camera is an Orbbec, not a RealSense.** `stacking_robokudo.py`
  ("for the Tracy robot") uses `CrDescriptorFactory.create_descriptor("orbbec")`.
  `config_orbbec.py`: colour `/camera/color/image_raw/compressed`, depth
  `/camera/depth/image_raw` (`depth_hints="raw"`), info
  `/camera/color/camera_info`, `tf_from="camera_color_optical_frame"`,
  `tf_to="map"`. The demo AEs in this repo subclass `RealsenseCameraConfig` and
  must be repointed.
- **Do NOT set `cam_r_w2c` / `cam_t_w2c` for the live pipeline.** Two separate
  cam→world transforms exist and they will compound:
  1. `FoundationPoseAnnotator` applies `twc` from those static params
     (`foundation_pose_annotator.py:211-229`, applied at `poses_two = twc @ poses_tco`).
  2. `Pose2ODConverter` applies `cas.cam_to_world_transform` from live TF and
     stamps the frame as `map` (`annotation_conversion.py:251-266`).
  Leave the annotator params `None` (identity → poses stay in the camera optical
  frame) and let RoboKudo's TF path do the single, correct transform. This is
  also what makes the result frame honest for the planning team.
- **`lookup_viewpoint` defaults to `True`** (`components.py:TfComponent`) and
  Orbbec does not override it, so `cas.cam_to_world_transform` will be populated
  as long as TF `camera_color_optical_frame → map` is being published. If it is
  not, poses silently come back in the camera frame instead — must be asserted.
- **`only_stable_viewpoints=True`, `max_viewpoint_distance=0.01`** on the Orbbec
  config: frames are dropped while the camera moves. Fine for a static head,
  a trap if Tracy pans mid-query.
- **`NoctisDetectionReader` cannot run live.** It replays BOP-style JSON off
  disk, indexed by `image_id`, advancing one frame per pipeline iteration. It is
  a replay harness, not a detector. Live NOCTIS needs new code (see Phase 3).
- **`robokudo_msgs/action/GenericImgProcAnnotator`** exists and is the natural
  contract for an out-of-process detector (`rgb`, `depth` in; `bounding_boxes`,
  `class_ids`, `class_confidences`, masks-via-`image` out). **RoboKudo core ships
  no client annotator for it** — grep over the robokudo source finds zero
  references — so the client side is ours to write.
- **Mask geometry is strict.** FoundationPose pastes the crop with
  `full_mask[y:y+h, x:x+w] = crop`, so the ROI rectangle must match the mask crop
  exactly. `NoctisDetectionReader.build_hypothesis` recomputes the rect from the
  decoded mask rather than trusting the file's `bbox`; any new detector
  annotator must do the same.
- **Depth is assumed uint16 millimetres** (`* 1e-3` at line 377) and registered
  to colour, modulo `COLOR2DEPTH_RATIO`.
- **Perf shape:** NOCTIS is heavy (template matching), FoundationPose `register()`
  is heavy, `track_one()` is cheap. So: segment + register **once** per object on
  query, then track. `operation_mode = 1` does exactly this.

#### Phases

**Phase 0 — Unblock the environment** (nothing runs until this is done)
- CUDA toolkit matching torch (2.12.1+cu130 → 13.x), `CUDA_HOME` exported.
- `pip install imageio h5py kornia warp-lang pycocotools` into the venv.
- `pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git`
  then the same for `pytorch3d@stable`.
- Drop both `model_best.pth` into `weights/2023-10-28-18-33-37/` and
  `weights/2024-01-11-20-02-45/`.
- Gate: `python -c "import nvdiffrast.torch, pytorch3d, warp, kornia"` clean.

**Phase 1 — Camera bring-up**
- Start Tracy's Orbbec driver; confirm with `ros2 topic list` and
  `ros2 topic echo --once` the exact topic names, encodings and depth dtype.
- Confirm depth is aligned to colour and in millimetres; set `color2depth_ratio`
  if the resolutions differ.
- Confirm `ros2 run tf2_ros tf2_echo map camera_color_optical_frame` resolves.
- Gate: a trivial AE (`pipeline_init` + `CollectionReaderAnnotator`) shows live
  frames in the RoboKudo visualiser.

**Phase 2 — FoundationPose live, no NOCTIS yet**
- Objects: pick the manipulation targets, get CAD meshes, convert to metres and
  recentre (`prepare_cube_meshes.py` is the pattern).
- New AE `demo_tracy_live.py`: Orbbec config + `ColorBlobDetector` +
  `FoundationPoseAnnotator`, `operation_mode=1`, `cam_r_w2c`/`cam_t_w2c` unset.
- Tune `ColorBlobDetector`: `object_extents`, `hsv_ranges`, `max_distance`,
  `min_pixel_area` for the real scene.
- Gate: overlay shows a stable, correctly-oriented axis on the real object.
  This isolates FoundationPose from NOCTIS — debug one thing at a time.

**Phase 3 — NOCTIS live**
Three options, in the order they should be attempted:

  - *3a (bring-up only)* Run NOCTIS offline over a short recording of Tracy's
    scene, replay through the existing `NoctisDetectionReader`. Proves the
    NOCTIS→FoundationPose mask contract with zero new code. Not a live path.
  - *3b (recommended)* Wrap NOCTIS as a standalone ROS 2 node serving
    `GenericImgProcAnnotator`, and write a `GenericDetectorClient` annotator in
    this repo that sends the CAS rgb/depth and turns the returned masks into
    `ObjectHypothesis` + `ImageROI` + `Classification`. Keeps NOCTIS's
    dependencies in their own venv — they will otherwise fight FoundationPose
    over torch/CUDA versions. Reuse `NoctisDetectionReader.build_hypothesis`'s
    rect-from-mask logic verbatim.
  - *3c (fallback)* Import NOCTIS in-process as a new annotator. Lowest latency,
    but only viable if its deps coexist with FoundationPose's in one venv.
- Template set: NOCTIS `category_id` indexes its rendered template folders and
  will not match `mesh_obj_ids` — keep the explicit `category_to_obj_id` map.
- Gate: live masks on the real objects at an acceptable rate.

**Phase 4 — Planning handoff**
- Add `QueryAnnotator()` after `pipeline_init()` and `GenerateQueryResult()` as
  the last child, per `stacking_robokudo.py`.
- Decide and document the query vocabulary with the planning team: what
  `obj.type` / `obj.color` values they send, and what `ObjectDesignator.type`
  they get back (it comes from `Classification.classname`, so classnames must be
  the names the planners use).
- Verify frame: `res[i].pose[0].header.frame_id` must read `map`, not
  `camera_color_optical_frame`. If it reads the latter, TF is missing — do not
  paper over it with `cam_r_w2c`.
- Give the planning team `robokudo_cram_integration.py` as their client
  reference.
- Gate: an out-of-process client sends a `Query` goal and gets back poses in
  `map` that match a tape-measure check on the real object.

**Phase 5 — Validation before handing over**
- Static accuracy: measure a known object position, compare to the reported pose.
- Repeatability: same scene, N queries, spread of returned poses.
- Symmetry check: cubes and boxes are symmetric — FoundationPose will return a
  pose consistent with the mesh but possibly rotated by a symmetry. Confirm the
  planners' grasping is symmetry-tolerant, or supply `symmetry_tfs`.
- Latency: time register vs. track; report both to the planners so they can size
  their timeouts (`wait_for_server` is 5 s in the reference client).
- Failure modes: what the query returns for an absent object (empty `res`) —
  planners must handle it.

#### Open questions for the planning team
1. Which frame do they want poses in — `map`, or a robot/base frame?
2. Which objects, and do we have CAD for all of them?
3. Do they want a one-shot pose per query, or a continuously tracked pose?
4. Grasp convention: is the mesh origin the grasp frame, or do they apply their
   own offset? (`use_cram_visual_axis` affects only the overlay, not the
   returned pose.)

#### Biggest risks
- **Phase 0 stays blocked** — `nvdiffrast`/`pytorch3d` builds are the single
  most likely place to lose a day.
- **NOCTIS/FoundationPose dependency conflict** in one venv — mitigated by 3b.
- **Missing TF** silently degrading poses to camera frame — caught by the
  Phase 4 frame_id assertion.
- **Object symmetry** producing "wrong" but valid orientations for grasping.

---

### 2026-09-08 — Live-camera requirements review (no code changes)

Scope changed: bag/offline replay and `env_fp.sh` are set aside; target is now a
**live camera** feed with the camera already calibrated. Reviewed the pipeline
for what live operation needs. No files modified.

**What the live switch actually changes:** only the *source* of the frames and
the *detector*. `FoundationPoseAnnotator` itself is source-agnostic — it reads
`COLOR_IMAGE`, `DEPTH_IMAGE`, `COLOR2DEPTH_RATIO` and `CAM_INTRINSIC` out of the
CAS and does not care where they came from.

**Findings:**

- **Intrinsics are free.** Calibration arrives through `/camera/color/camera_info`
  → `CASViews.CAM_INTRINSIC`. Nothing to configure.
- **Extrinsics are NOT read from TF.** `cam_r_w2c` / `cam_t_w2c` are static
  descriptor parameters (`foundation_pose_annotator.py:211-229`). Left at `None`
  they default to identity and poses come out **in the camera optical frame** —
  which is the right choice for a first live test. The `lookup_viewpoint` /
  `tf_from` / `tf_to` fields on the camera config belong to RoboKudo's reader,
  not to this annotator.
- **Depth must be uint16 millimetres.** `foundation_pose_annotator.py:377` does
  `depth * 1e-3` unconditionally. RealSense raw depth already is mm.
- **Depth must be aligned/registered to colour**, or `COLOR2DEPTH_RATIO` has to
  describe the mapping correctly.
- **A mask-producing detector is mandatory.** FoundationPose cannot `register()`
  without an `ObjectHypothesis` carrying an `ImageROI` mask.
  - `ColorBlobDetector` (in this repo) works live — HSV threshold + metric size
    check against CAD extents.
  - `NoctisDetectionReader` **cannot** be used live; it replays precomputed JSON.
- **`demo_tetris_noctis.py` is therefore not a live-capable AE**;
  `demo_colored_cubes.py` is, once `MESH_DIR` and the camera topics are fixed.
- **Machine check (this box):** `realsense2_camera` driver is **not** installed
  (only `realsense2_description` is in `ros2_ws/install`), no RealSense on USB,
  and no ROS graph is running — so no camera is publishing here right now. To be
  re-verified on the machine that actually has the camera.
- All blockers from the previous entry still stand: CUDA toolkit, `nvdiffrast` /
  `pytorch3d` / `warp-lang` / `kornia`, and the two `model_best.pth` weights.

**Live checklist (ordered):**

1. CUDA toolkit + `nvdiffrast`, `pytorch3d`, `warp-lang`, `kornia`, `imageio`, `h5py`.
2. Both `model_best.pth` into `weights/2023-10-28-18-33-37/` and `weights/2024-01-11-20-02-45/`.
3. Camera driver up with **aligned depth**, e.g.
   `ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true pointcloud.enable:=false`.
4. Confirm actual topic names/encodings with `ros2 topic list` / `ros2 topic echo --once`.
5. Camera-config subclass matching those topics (`topic_color`, `color_hints`,
   `topic_depth`, `depth_hints`, `topic_cam_info`, `color2depth_ratio`).
6. Object meshes in metres; `mesh_obj_ids` must equal the detector's `class_id`s.
7. Detector tuned to the real scene: `object_extents`, `hsv_ranges`,
   `max_distance`, `min_pixel_area`.
8. `operation_mode = 1` (estimate once, then track).
9. Symlink both packages into a workspace `src/` and
   `colcon build --symlink-install` — still required for the `ros2 run` path.
10. Run, and watch the annotator overlay for a first sanity check.

---

### 2026-09-08 — Environment verification on tracy's machine (no code changes)

Audited the checkout at `/home/tracy/robokudo_foundation_pose` against what the
pipeline actually needs. No files were modified; findings recorded below.

**Verdict: the pipeline cannot run yet.** Blockers are the missing CUDA toolkit,
four missing GPU Python packages, the missing model weights, and the missing
input data. `env_fp.sh` also points at a user (`/home/student`) that does not
exist on this machine.

---

## Environment status (as verified 2026-09-08)

### Present and working

| Item | Status |
|---|---|
| GPU | RTX 3080, 10 GB, driver 580.173.02 (CUDA 13.0 runtime) |
| ROS 2 | Jazzy at `/opt/ros/jazzy` |
| Python | 3.12.3 at `/usr/bin/python3` |
| Venv | `/home/tracy/.virtualenvs/robokudo` (system-site-packages) |
| `robokudo` | 1.0.0, editable from `/home/tracy/blue/cognitive_robot_abstract_machine/robokudo` |
| torch | 2.12.1+cu130, `torch.cuda.is_available() == True` |
| Also in venv | torchvision, numpy 1.26.4, scipy, opencv, trimesh 4.12.2, open3d 0.19.0, omegaconf, py_trees, sklearn, PIL |
| colcon | `/usr/bin/colcon` |
| Recording | `/home/tracy/blue/data/01/01_0.mcap` (the `01_0.mcap` the scripts expect) |
| CAD meshes | `robokudo_cad_data/meshes/` — cracker_box, mustard, sugar_box, tomato_soup_can, pot_silver |

### Missing — blockers

1. **CUDA toolkit / `nvcc`** — none installed anywhere on the machine
   (`/usr/local/cuda*` does not exist). Needed to compile `nvdiffrast` and
   `pytorch3d`. Driver supports CUDA 13.0 and torch is built against cu130, so
   install a 13.x toolkit (or a version matching whatever torch build is used).
   `g++-11`/`gcc-11` are also absent; only g++ 13.3 is present. Those are only
   required for the CUDA <= 11.8 workaround in the README, so with a 13.x
   toolkit they are probably not needed.

2. **Python packages missing from the `robokudo` venv:**
   - GPU/build-required: `nvdiffrast`, `pytorch3d`, `warp-lang`, `kornia`
   - plain pip: `imageio`, `h5py`, `transformers`
   - only for `extract_all.py`: `mcap`, `mcap_ros2`
   - only for `noctis_detection_reader.py`: `pycocotools`
   - referenced somewhere in the tree: `transformations`, `ruamel`

3. **Model weights are not in the repo.** `weights/2023-10-28-18-33-37/` and
   `weights/2024-01-11-20-02-45/` contain only `config.yml`. The code loads
   `model_best.pth` from each
   (`learning/training/predict_score.py:127`, `predict_pose_refine.py:132`).
   Both `.pth` files must be downloaded from the upstream FoundationPose release
   and dropped into those two directories.

4. **Input data directories do not exist** (all are gitignored, so they were
   never cloned): `extracted/`, `meshes_m/`, `noctis_acrambly/`, `runs/`.
   `run_cubes_offline.py` and `demo_tetris_noctis.py` need all of them.
   - `extracted/` is regenerated by `extract_all.py` from the mcap.
   - `meshes_m/` is regenerated by `prepare_cube_meshes.py`, but its inputs
     (`child_cube_0_colored.ply`, `child_cube_2_colored.ply`) are **not on this
     machine** — they must be copied over.
   - `noctis_acrambly/` NOCTIS detection JSONs are **not on this machine** —
     must be copied over. There is no script here that regenerates them.

5. **`env_fp.sh` is stale for this machine.** It sources
   `/home/student/ros_ws/install/setup.bash` and
   `/home/student/robokudo_foundation_pose/.venv_fp/bin/activate`; neither
   `/home/student` nor those paths exist. The equivalents here are
   `/home/tracy/ros2_ws/install/setup.bash` and
   `/home/tracy/.virtualenvs/robokudo/bin/activate`.

6. **Hardcoded `/home/student` paths in source** that need repointing:
   - `extract_all.py:27` — `MCAP_PATH`
   - `descriptors/analysis_engines/demo_colored_cubes.py:24` — `MESH_DIR`
   - `descriptors/analysis_engines/demo_tetris_noctis.py:23,25` — `MESH_DIR`, `NOCTIS_RESULTS`

---

## Does it need a colcon build?

**It depends on which of the two entry points is used.**

### ROS path — `ros2 run robokudo main _ae=... _ros_pkg=robokudo_foundation_pose`
**Yes, colcon build is required.** `robokudo` resolves `_ros_pkg` through the
ament index, and `robokudo_foundation_pose` is an `ament_python` package that is
currently in **no** workspace — `/home/tracy/ros2_ws/src/` does not contain it,
and `/opt/ros/jazzy` does not either. So it must be symlinked or copied into a
workspace `src/` and built:

```bash
ln -s /home/tracy/robokudo_foundation_pose/robokudo_foundation_pose /home/tracy/ros2_ws/src/
ln -s /home/tracy/robokudo_foundation_pose/robokudo_cad_data       /home/tracy/ros2_ws/src/
cd /home/tracy/ros2_ws && colcon build --symlink-install \
    --packages-select robokudo_foundation_pose robokudo_cad_data
```

`--symlink-install` matters here: without it, every Python edit needs a rebuild.
`robokudo_cad_data` is `ament_cmake` and only installs meshes, so it is only
needed for the YCB demos that reference the meshes by package share path.

### Offline path — `python run_cubes_offline.py`
**No colcon build needed**, but it will not import as-is. From the repo root,
`import robokudo_foundation_pose` resolves to the outer *ament package* folder
(a namespace package with no `annotator` submodule), so the import fails. Fix
with either:

```bash
export PYTHONPATH=/home/tracy/robokudo_foundation_pose/robokudo_foundation_pose:$PYTHONPATH
# or, once:
pip install -e /home/tracy/robokudo_foundation_pose/robokudo_foundation_pose
```

Nothing in this package builds a C/C++ extension of its own, so colcon is only
ever about ament registration, never compilation. The real compilation cost is
`nvdiffrast` and `pytorch3d`, which are compiled by pip at install time.

---

## Suggested order to get to a first run

1. Install a CUDA toolkit matching torch (13.x), set `CUDA_HOME`.
2. `pip install imageio h5py kornia warp-lang transformers pycocotools mcap mcap-ros2-support`
   into `/home/tracy/.virtualenvs/robokudo`.
3. `pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git`
   and the same for `pytorch3d@stable`.
4. Drop the two `model_best.pth` files into the `weights/*/` directories.
5. Rewrite `env_fp.sh` for tracy's paths.
6. Repoint the hardcoded `/home/student` paths (or make them relative to the repo root).
7. Copy over the cube source PLYs and the NOCTIS results; run `extract_all.py`
   and `prepare_cube_meshes.py` to populate `extracted/` and `meshes_m/`.
8. Offline smoke test: `python run_cubes_offline.py --frames 0`.
9. Only then wire up the workspace and colcon build for the ROS demos.
