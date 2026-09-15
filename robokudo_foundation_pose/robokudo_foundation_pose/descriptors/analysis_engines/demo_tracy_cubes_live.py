"""Live 6D pose estimation of the two cube assemblies on Tracy's Orbbec camera.

This is the bring-up pipeline: colour+size segmentation produces the masks and
FoundationPose turns each mask into a 6D pose, which it then tracks.  Unlike the
'.mcap' demos, the frames come straight off the camera.

    ros2 run robokudo main _ae=demo_tracy_cubes_live _ros_pkg=robokudo_foundation_pose

Two things must be true before this gives sensible numbers:

1. The Orbbec driver must run with depth registered to colour, otherwise the
   masks (cut from the 1920x1080 colour image) address the wrong pixels of the
   640x576 depth image, which sits in a different optical frame:

       ros2 launch orbbec_camera <your>.launch.py depth_registration:=true

   After enabling it, re-check 'ros2 topic echo --once --field width
   /camera/depth/image_raw' and set COLOR2DEPTH_RATIO below accordingly.

2. For poses in the 'map' frame, Tracy must be on and publishing TF.  While it
   is off, the camera's TF tree is rooted at 'camera_link' and there is no
   'map', so POSES_IN_MAP_FRAME has to stay False.

Note that POSES_IN_MAP_FRAME only fills 'cas.cam_to_world_transform'; nothing in
*this* pipeline reads it.  The 'PoseAnnotation' values stay in the camera optical
frame either way.  The transform is applied by 'Pose2ODConverter', which runs
inside 'GenerateQueryResult' - so the flag changes the reported frame only once
the query annotators are added to the pipeline.
"""

import os.path as osp

import robokudo.analysis_engine
import robokudo.idioms
import robokudo.pipeline
# 'robokudo.descriptors' must be imported before 'robokudo.annotators.collection_reader':
# the descriptors package imports the CollectionReader itself, and reaching the
# camera configs the other way round leaves that module half-initialised
# ("cannot import name 'CollectionReaderAnnotator' ... partially initialized").
from robokudo.descriptors import CrDescriptorFactory
from robokudo.annotators.collection_reader import CollectionReaderAnnotator

from robokudo_foundation_pose.annotator.color_blob_detector import ColorBlobDetector
from robokudo_foundation_pose.annotator.foundation_pose_annotator import FoundationPoseAnnotator


# The '.ply' files live at the repository root.  They are already in metres and
# recentred on their bounding box, so no scaling is applied.
MESH_DIR = "/home/tracy/robokudo_foundation_pose"

# object id -> (mesh file, CAD extents in metres, classname)
# Both assemblies are red-and-blue, so colour alone cannot tell them apart - the
# detector separates them by their metric size, which is very different.
OBJECTS = {
    0: ("child_cube_0.ply", (0.05, 0.20305, 0.05), "child_cube_0"),
    2: ("child_cube_2.ply", (0.15203, 0.10102, 0.10102), "child_cube_2"),
}

# Set to True once Tracy is on and 'map -> camera_color_optical_frame' resolves.
# It makes the CollectionReader look the viewpoint up and store the transform in
# the CAS, which is what lets 'GenerateQueryResult' report poses in 'map'.  On its
# own it does not change the poses this pipeline produces (see the module
# docstring) - it is a prerequisite for the query handoff, not a switch for it.
POSES_IN_MAP_FRAME = True

# (sx, sy) mapping colour pixels to depth pixels.  With depth registered to
# colour both streams share a resolution and this is (1.0, 1.0).  If the driver
# publishes registered depth at a lower resolution, use
# (depth_width / colour_width, depth_height / colour_height).
COLOR2DEPTH_RATIO = (1.0, 1.0)

# Measured live: the scene sits at ~0.96 m, the background beyond ~2.4 m.
MAX_OBJECT_DISTANCE = 1.2


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "demo_tracy_cubes_live"

    def implementation(self):
        # Tracy's Orbbec already matches the registered 'orbbec' config's topic
        # names, so only the scene-specific fields are overridden here.
        reader_config = CrDescriptorFactory.create_descriptor(
            "orbbec",
            color2depth_ratio=COLOR2DEPTH_RATIO,
            lookup_viewpoint=POSES_IN_MAP_FRAME,
            # the camera is fixed, so there is no viewpoint motion to reject;
            # leaving the check on only risks dropping frames while nothing moves
            only_stable_viewpoints=False,
        )

        # ---- detector: colour threshold + metric size check -> masks --------
        detector_desc = ColorBlobDetector.Descriptor()
        detector_desc.parameters.object_extents = {
            obj_id: extents for obj_id, (_, extents, _) in OBJECTS.items()}
        detector_desc.parameters.object_names = {
            obj_id: name for obj_id, (_, _, name) in OBJECTS.items()}
        detector_desc.parameters.max_distance = MAX_OBJECT_DISTANCE
        detector_desc.parameters.one_instance_per_class = True

        # ---- pose estimation and tracking -----------------------------------
        fp_desc = FoundationPoseAnnotator.Descriptor()
        fp_desc.parameters.mesh_files = [
            osp.join(MESH_DIR, mesh) for mesh, _, _ in OBJECTS.values()]
        fp_desc.parameters.mesh_obj_ids = list(OBJECTS.keys())
        fp_desc.parameters.default_mesh_scale_factor = 1.0   # meshes are already in metres
        fp_desc.parameters.name_to_obj_id = {
            name: obj_id for obj_id, (_, _, name) in OBJECTS.items()}

        # Operation modes:
        #   0 = do nothing (the model is not even loaded)
        #   1 = estimate a pose for objects that have none, then track it
        #   2 = only estimate, and only for objects that have no pose yet
        #   3 = only estimate, treating every object as new on every frame
        #   4 = only track, using a pose that something else produced
        #
        # Note that tracking does not actually engage here: 'ColorBlobDetector'
        # runs with 'clear_own_hypotheses', so each frame replaces its
        # 'ObjectHypothesis' objects and last frame's 'PoseAnnotation' goes with
        # them.  Every hypothesis therefore arrives without a pose and takes the
        # 'register()' path, which makes mode 1 behave like mode 3.  That is fine
        # for checking accuracy - every frame is independent, so nothing drifts -
        # but it is the slow path.  Real tracking needs an 'ObjectAssociator' to
        # carry hypotheses across frames.
        fp_desc.parameters.operation_mode = 1
        fp_desc.parameters.est_refine_iter = 5
        fp_desc.parameters.track_refine_iter = 2
        fp_desc.parameters.est_batch_size = 64
        fp_desc.parameters.update_old_pose_annotations = True

        # 'cam_r_w2c' / 'cam_t_w2c' are deliberately left unset.  RoboKudo already
        # transforms the pose from the camera to the world frame when it builds
        # the query result, using the TF viewpoint from the CAS; setting them
        # here as well would apply the transform twice.

        seq = robokudo.pipeline.Pipeline("TracyCubesLivePipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=reader_config),
                ColorBlobDetector(descriptor=detector_desc),
                FoundationPoseAnnotator(descriptor=fp_desc),
            ])
        return seq
