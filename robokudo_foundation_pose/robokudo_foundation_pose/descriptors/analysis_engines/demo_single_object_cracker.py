import os
import os.path as osp

import robokudo.analysis_engine

from robokudo.annotators.collection_reader import CollectionReaderAnnotator

import robokudo.io.camera_interface
import robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform

from robokudo.annotators.lambda_function import LambdaFunctionAnnotator
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis

import robokudo.idioms

from robokudo_foundation_pose.masks import rle_to_mask
from robokudo_foundation_pose.annotator.foundation_pose_annotator import FoundationPoseAnnotator


import numpy as np


def get_pkg_share_dir(ros_pkg_name: str) -> str:
    """Returns the share dir of the package."""
    from ament_index_python.packages import get_package_share_directory

    if ros_pkg_name is None:
        raise ValueError("The value 'ros_pkg_name' cannot be 'None'.")

    return get_package_share_directory(ros_pkg_name)


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "demo_single_object_mustard"

    def implementation(self):
        """
        Create a basic pipeline that estimates the 6D pose estimation for cracker box object visible
        in the ROS2 "2025-01-09-11-12-19-YCB-drill-cheezit-mustard-masterchef-domino-tomatosoup.bag" file.
        """
        pkg_dir = get_pkg_share_dir("robokudo_cad_data")
        obj_path = osp.join(pkg_dir, "meshes", "cracker_box", "textured.obj")

        kinect_camera_config = robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform.CameraConfig()
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(kinect_camera_config))

        def cracker_hypothesis(context):
            object_hypothesis = ObjectHypothesis()
            (x, y, w, h) = [355, 310, 168, 183]
            object_hypothesis.roi.roi.pos.x = x
            object_hypothesis.roi.roi.pos.y = y
            object_hypothesis.roi.roi.width = w
            object_hypothesis.roi.roi.height = h

            full_mask = rle_to_mask({'counts': [341140, 9, 945, 20, 936, 29, 928, 36, 919, 46, 911, 52, 905, 63, 895,
                                                68, 892, 72, 887, 77, 883, 81, 878, 87, 873, 92, 868, 96, 864, 100, 860,
                                                105, 855, 109, 851, 114, 846, 118, 842, 123, 837, 127, 833, 131, 829,
                                                136, 824, 139, 821, 145, 815, 149, 811, 153, 807, 159, 801, 162, 798,
                                                167, 793, 171, 789, 175, 785, 178, 782, 179, 781, 179, 781, 180, 780,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 778, 181, 779, 182, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 180, 780, 180, 780, 180, 780, 180, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 778, 182, 778, 182, 778,
                                                182, 778, 182, 778, 182, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                                                181, 779, 181, 779, 181, 779, 181, 779, 180, 780, 180, 780, 180, 780,
                                                180, 780, 180, 780, 180, 780, 181, 779, 181, 779, 181, 780, 180, 783,
                                                177, 787, 173, 798, 162, 815, 145, 832, 128, 840, 120, 849, 111, 861,
                                                99, 869, 90, 879, 81, 896, 63, 910, 49, 922, 38, 727191],
                                     'size': [960, 1280]})   # 960 x 1280
            object_hypothesis.roi.mask = np.zeros((h, w), dtype=np.uint8)
            object_hypothesis.roi.mask[full_mask[y: y+h, x: x+w]] = 1
            object_hypothesis.annotations = []

            classification = Classification()
            classification.class_id = None   # use no id to see if the alternative classname to id matching works
            classification.classname = "cracker_box"
            classification.confidence = 1.0
            object_hypothesis.annotations.append(classification)

            context.get_cas().annotations = [object_hypothesis]

        lambda_descriptor = LambdaFunctionAnnotator.Descriptor()
        lambda_descriptor.parameters.func = cracker_hypothesis

        fp_desc = FoundationPoseAnnotator.Descriptor()
        fp_desc.parameters.mesh_files = [obj_path]
        fp_desc.parameters.mesh_obj_ids = [42]
        fp_desc.parameters.name_to_obj_id = {"cracker_box": 42}
        fp_desc.parameters.est_batch_size = 128
        fp_desc.parameters.debug = False
        fp_desc.parameters.debug_dir = None

        seq = robokudo.pipeline.Pipeline("StoragePipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                LambdaFunctionAnnotator(descriptor=lambda_descriptor),
                FoundationPoseAnnotator(descriptor=fp_desc),
            ])
        return seq
