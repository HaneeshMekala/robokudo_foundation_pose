"""6D pose estimation of the tetris cube assemblies using NOCTIS segmentations.

Replays the NOCTIS detections computed offline for the '01_0.mcap' recording and
hands each mask to FoundationPose, which estimates a 6D pose and then refines it
on the following frames.

NOCTIS was run on every 50th frame, so the reader is configured with the same
stride: each pipeline iteration consumes the next frame it has detections for.
"""

import os.path as osp

import robokudo.analysis_engine
import robokudo.idioms
import robokudo.io.camera_interface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.camera_configs.config_realsense import CameraConfig as RealsenseCameraConfig

from robokudo_foundation_pose.annotator.noctis_detection_reader import NoctisDetectionReader
from robokudo_foundation_pose.annotator.foundation_pose_annotator import FoundationPoseAnnotator


MESH_DIR = "/home/student/robokudo_foundation_pose/meshes_m"

NOCTIS_RESULTS = ("/home/student/robokudo_foundation_pose/noctis_acrambly/"
                  "datasets/results/noctis")

# NOCTIS numbers its categories after the template folders it rendered
# (obj_000001, obj_000002), which are not the mesh ids FoundationPose uses.
#   category 1 = obj_000001 = the four-cube bar   -> mesh id 0
#   category 2 = obj_000002 = the L/T assembly    -> mesh id 2
CATEGORY_TO_OBJ_ID = {1: 0, 2: 2}

OBJECT_MESHES = {
    0: "child_cube_0.ply",
    2: "child_cube_2.ply",
}

# Full 'table -> camera_color_optical_frame' extrinsic, inverted to give the
# world->camera transform the annotator expects.  Derived from the 'table ->
# camera_link' transform in extracted/tf_static.json composed with the standard
# REP-103 camera_link->optical rotation, which the recording does not publish.
CAM_R_W2C = [0.020329, -0.999638, 0.017644,
              -0.951457, -0.024764, -0.306785,
              0.307111, -0.010551, -0.951615]
CAM_T_W2C = [-0.018652, 0.661975, 0.715832]

# NOCTIS was run on every 50th frame of the recording
NOCTIS_FRAME_IDS = list(range(0, 1951, 50))


class TetrisSceneCameraConfig(RealsenseCameraConfig):
    """Camera config for the '01_0.mcap' recording.

    The RealSense driver default assumes an aligned *compressed* depth topic; this
    recording publishes raw depth on '/camera/depth/image_raw', already registered
    to colour at the same 1920x1080 resolution.
    """

    topic_depth = "/camera/depth/image_raw"
    depth_hints = "raw"

    topic_color = "/camera/color/image_raw/compressed"
    color_hints = "compressed"

    topic_cam_info = "/camera/color/camera_info"

    color2depth_ratio = (1.0, 1.0)

    tf_from = "camera_color_optical_frame"
    tf_to = "table"
    lookup_viewpoint = False


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "demo_tetris_noctis"

    def implementation(self):
        camera_config = TetrisSceneCameraConfig()
        reader_config = CollectionReaderAnnotator.Descriptor(
            camera_config=camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(camera_config))

        # ---- masks: replay the offline NOCTIS segmentations ------------------
        noctis_desc = NoctisDetectionReader.Descriptor()
        noctis_desc.parameters.results_path = NOCTIS_RESULTS
        noctis_desc.parameters.results_glob = "result_tetris_every50_frame_*.json"
        noctis_desc.parameters.category_to_obj_id = CATEGORY_TO_OBJ_ID
        noctis_desc.parameters.category_names = {1: "child_cube_0", 2: "child_cube_2"}
        # observed scores run 0.60 to 0.78, so this only cuts clear outliers
        noctis_desc.parameters.score_threshold = 0.5
        noctis_desc.parameters.one_instance_per_class = True
        # consume one detected frame per iteration, wrapping when the bag loops
        noctis_desc.parameters.frame_ids = NOCTIS_FRAME_IDS

        # ---- pose estimation and tracking ------------------------------------
        fp_desc = FoundationPoseAnnotator.Descriptor()
        fp_desc.parameters.mesh_files = [
            osp.join(MESH_DIR, mesh) for mesh in OBJECT_MESHES.values()]
        fp_desc.parameters.mesh_obj_ids = list(OBJECT_MESHES.keys())
        fp_desc.parameters.default_mesh_scale_factor = 1.0   # meshes are already in metres
        fp_desc.parameters.name_to_obj_id = {
            osp.splitext(mesh)[0]: obj_id for obj_id, mesh in OBJECT_MESHES.items()}

        fp_desc.parameters.operation_mode = 1
        fp_desc.parameters.est_refine_iter = 5
        fp_desc.parameters.track_refine_iter = 2
        fp_desc.parameters.est_batch_size = 64
        fp_desc.parameters.update_old_pose_annotations = True

        # full table<-optical extrinsic, so poses come out in table coordinates
        fp_desc.parameters.cam_r_w2c = CAM_R_W2C
        fp_desc.parameters.cam_t_w2c = CAM_T_W2C

        seq = robokudo.pipeline.Pipeline("TetrisNoctisPipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=reader_config),
                NoctisDetectionReader(descriptor=noctis_desc),
                FoundationPoseAnnotator(descriptor=fp_desc),
            ])
        return seq
