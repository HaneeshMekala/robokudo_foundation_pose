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
        Create a basic pipeline that estimates the 6D pose estimation for mustard object visible
         in the ROS2 "2025-01-09-11-12-19-YCB-drill-cheezit-mustard-masterchef-domino-tomatosoup.bag" file.
        """
        pkg_dir = get_pkg_share_dir("robokudo_cad_data")
        obj_path = osp.join(pkg_dir, "meshes", "mustard", "textured_simple.obj")

        kinect_camera_config = robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform.CameraConfig()
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(kinect_camera_config))

        def mustard_hypothesis(context):
            object_hypothesis = ObjectHypothesis()
            (x, y, w, h) = [500, 489, 103, 174]
            object_hypothesis.roi.roi.pos.x = x
            object_hypothesis.roi.roi.pos.y = y
            object_hypothesis.roi.roi.width = w
            object_hypothesis.roi.roi.height = h

            full_mask = rle_to_mask({"counts": [480555, 9, 947, 19, 937, 27, 931, 31, 926, 36, 922, 41, 38, 20, 859, 47,
                                                27, 32, 853, 51, 18, 42, 847, 56, 10, 48, 844, 116, 843, 118, 840, 121,
                                                838, 123, 836, 125, 834, 126, 833, 128, 831, 129, 830, 131, 828, 132,
                                                827, 133, 826, 135, 825, 135, 824, 137, 822, 138, 821, 139, 819, 142,
                                                814, 146, 812, 148, 810, 151, 807, 153, 806, 154, 805, 156, 803, 157,
                                                802, 158, 801, 159, 800, 160, 798, 163, 795, 165, 793, 167, 791, 169,
                                                790, 170, 790, 170, 789, 172, 788, 172, 788, 172, 787, 173, 787, 173,
                                                787, 173, 787, 173, 787, 173, 787, 173, 787, 173, 787, 173, 787, 174,
                                                787, 173, 787, 173, 788, 172, 788, 172, 790, 170, 792, 168, 794, 166,
                                                796, 164, 797, 163, 797, 163, 798, 161, 800, 160, 801, 159, 802, 158,
                                                803, 157, 805, 155, 806, 154, 809, 151, 811, 149, 814, 146, 816, 144,
                                                816, 144, 817, 142, 819, 141, 819, 141, 820, 140, 821, 138, 822, 138,
                                                823, 137, 824, 135, 826, 134, 827, 133, 828, 132, 829, 130, 831, 129,
                                                832, 128, 833, 126, 835, 125, 836, 123, 838, 122, 840, 119, 842, 117,
                                                845, 115, 846, 44, 10, 59, 849, 39, 16, 54, 854, 33, 22, 50, 857, 27,
                                                30, 44, 866, 13, 43, 34, 933, 16, 650247],
                                     "size": [960, 1280]})   # 960 x 1280
            object_hypothesis.roi.mask = np.zeros((h, w), dtype=np.uint8)
            object_hypothesis.roi.mask[full_mask[y: y+h, x: x+w]] = 1
            object_hypothesis.annotations = []

            classification = Classification()
            classification.class_id = 42
            classification.classname = "mustard"
            classification.confidence = 1.0
            object_hypothesis.annotations.append(classification)

            context.get_cas().annotations = [object_hypothesis]

        lambda_descriptor = LambdaFunctionAnnotator.Descriptor()
        lambda_descriptor.parameters.func = mustard_hypothesis

        fp_desc = FoundationPoseAnnotator.Descriptor()
        fp_desc.parameters.mesh_files = [obj_path]
        fp_desc.parameters.mesh_obj_ids = [42]
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
