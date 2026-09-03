"""6D pose estimation of the coloured cube assemblies.

Runs the full pipeline on the '01_0.mcap' recording: a colour-and-size detector
produces the segmentation masks, and FoundationPose turns each mask into a 6D
pose and then tracks it.

The CAD models come from 'prepare_cube_meshes.py', which converts the exported
PLYs to metres and recentres them on their bounding box.
"""

import os.path as osp

import robokudo.analysis_engine
import robokudo.idioms
import robokudo.io.camera_interface
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.camera_configs.config_realsense import CameraConfig as RealsenseCameraConfig

from robokudo_foundation_pose.annotator.color_blob_detector import ColorBlobDetector
from robokudo_foundation_pose.annotator.foundation_pose_annotator import FoundationPoseAnnotator


# absolute path to the meshes written by 'prepare_cube_meshes.py'
MESH_DIR = "/home/student/robokudo_foundation_pose/meshes_m"

# object id -> (mesh file, CAD extents in metres)
OBJECTS = {
    0: ("child_cube_0.ply", (0.05, 0.2031, 0.05)),
    2: ("child_cube_2.ply", (0.152, 0.101, 0.101)),
}

# Full 'table -> camera_color_optical_frame' extrinsic, inverted to give the
# world->camera transform the annotator expects.  Derived from the 'table ->
# camera_link' transform in extracted/tf_static.json composed with the standard
# REP-103 camera_link->optical rotation, which the recording does not publish.
CAM_R_W2C = [0.020329, -0.999638, 0.017644,
             -0.951457, -0.024764, -0.306785,
             0.307111, -0.010551, -0.951615]
CAM_T_W2C = [-0.018652, 0.661975, 0.715832]



class CubeSceneCameraConfig(RealsenseCameraConfig):
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
        return "demo_colored_cubes"

    def implementation(self):
        camera_config = CubeSceneCameraConfig()
        reader_config = CollectionReaderAnnotator.Descriptor(
            camera_config=camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(camera_config))

        # ---- detector: colour threshold + metric size check -> masks --------
        detector_desc = ColorBlobDetector.Descriptor()
        detector_desc.parameters.object_extents = {
            obj_id: extents for obj_id, (_, extents) in OBJECTS.items()}
        detector_desc.parameters.object_names = {
            obj_id: osp.splitext(mesh)[0] for obj_id, (mesh, _) in OBJECTS.items()}
        # the table top sits about 0.9 m from the camera, so anything past 1.2 m
        # is background and cannot be one of the cubes
        detector_desc.parameters.max_distance = 1.2
        detector_desc.parameters.one_instance_per_class = True

        # ---- pose estimation and tracking -----------------------------------
        fp_desc = FoundationPoseAnnotator.Descriptor()
        fp_desc.parameters.mesh_files = [
            osp.join(MESH_DIR, mesh) for mesh, _ in OBJECTS.values()]
        fp_desc.parameters.mesh_obj_ids = list(OBJECTS.keys())
        fp_desc.parameters.default_mesh_scale_factor = 1.0   # meshes are already in metres
        fp_desc.parameters.name_to_obj_id = {
            osp.splitext(mesh)[0]: obj_id for obj_id, (mesh, _) in OBJECTS.items()}

        # estimate a pose for objects that have none, refine it on every later frame
        fp_desc.parameters.operation_mode = 1
        fp_desc.parameters.est_refine_iter = 5
        fp_desc.parameters.track_refine_iter = 2
        fp_desc.parameters.est_batch_size = 64
        fp_desc.parameters.update_old_pose_annotations = True

        # 'table -> camera_link' from tf_static.json, so poses come out in table coordinates
        fp_desc.parameters.cam_r_w2c = CAM_R_W2C
        fp_desc.parameters.cam_t_w2c = CAM_T_W2C

        seq = robokudo.pipeline.Pipeline("ColoredCubesPipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=reader_config),
                ColorBlobDetector(descriptor=detector_desc),
                FoundationPoseAnnotator(descriptor=fp_desc),
            ])
        return seq
