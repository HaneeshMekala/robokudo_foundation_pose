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


object_hypothesis_config_mustard = {
        "name": "mustard",
        "class_id": 0,
        "object_path": ["meshes", "mustard", "textured.obj"],
        "bbox": [500, 489, 103, 174],   # xywh
        "mask": {"counts": [480555, 9, 947, 19, 937, 27, 931, 31, 926, 36, 922, 41, 38, 20, 859, 47, 27, 32, 853, 51,
                            18, 42, 847, 56, 10, 48, 844, 116, 843, 118, 840, 121, 838, 123, 836, 125, 834, 126, 833,
                            128, 831, 129, 830, 131, 828, 132, 827, 133, 826, 135, 825, 135, 824, 137, 822, 138, 821,
                            139, 819, 142, 814, 146, 812, 148, 810, 151, 807, 153, 806, 154, 805, 156, 803, 157, 802,
                            158, 801, 159, 800, 160, 798, 163, 795, 165, 793, 167, 791, 169, 790, 170, 790, 170, 789,
                            172, 788, 172, 788, 172, 787, 173, 787, 173, 787, 173, 787, 173, 787, 173, 787, 173, 787,
                            173, 787, 173, 787, 174, 787, 173, 787, 173, 788, 172, 788, 172, 790, 170, 792, 168, 794,
                            166, 796, 164, 797, 163, 797, 163, 798, 161, 800, 160, 801, 159, 802, 158, 803, 157, 805,
                            155, 806, 154, 809, 151, 811, 149, 814, 146, 816, 144, 816, 144, 817, 142, 819, 141, 819,
                            141, 820, 140, 821, 138, 822, 138, 823, 137, 824, 135, 826, 134, 827, 133, 828, 132, 829,
                            130, 831, 129, 832, 128, 833, 126, 835, 125, 836, 123, 838, 122, 840, 119, 842, 117, 845,
                            115, 846, 44, 10, 59, 849, 39, 16, 54, 854, 33, 22, 50, 857, 27, 30, 44, 866, 13, 43, 34,
                            933, 16, 650247],
                 "size": [960, 1280]}
}
object_hypothesis_config_cracker_box = {
    "name": "cracker_box",
    "class_id": 3,
    "object_path": ["meshes", "cracker_box", "textured.obj"],
    "bbox": [355, 310, 168, 183],
    "mask": {"counts": [341140, 9, 945, 20, 936, 29, 928, 36, 919, 46, 911, 52, 905, 63, 895, 68, 892, 72, 887, 77,
                        883, 81, 878, 87, 873, 92, 868, 96, 864, 100, 860, 105, 855, 109, 851, 114, 846, 118, 842,
                        123, 837, 127, 833, 131, 829, 136, 824, 139, 821, 145, 815, 149, 811, 153, 807, 159, 801,
                        162, 798, 167, 793, 171, 789, 175, 785, 178, 782, 179, 781, 179, 781, 180, 780, 181, 779,
                        181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 181, 779, 181, 778, 181, 779, 182, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 180, 780, 180, 780, 180, 780, 180, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 778, 182, 778, 182, 778,
                        182, 778, 182, 778, 182, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779, 181, 779,
                        180, 780, 180, 780, 180, 780, 180, 780, 180, 780, 180, 780, 181, 779, 181, 779, 181, 780,
                        180, 783, 177, 787, 173, 798, 162, 815, 145, 832, 128, 840, 120, 849, 111, 861, 99, 869,
                        90, 879, 81, 896, 63, 910, 49, 922, 38, 727191],
             "size": [960, 1280]}
}
object_hypothesis_config_sugar_box = {
    "name": "sugar_box",
    "class_id": 4,
    "object_path": ["meshes", "sugar_box", "textured.obj"],
    "bbox": [815, 491, 129, 168],
    "mask": {"counts": [783028, 15, 939, 25, 931, 30, 927, 34, 923, 37, 920, 41, 914, 46, 905, 55, 902, 58, 897, 63,
                        887, 74, 875, 85, 868, 92, 864, 96, 855, 105, 850, 110, 845, 115, 838, 122, 831, 129, 827,
                        134, 819, 141, 813, 147, 808, 152, 805, 155, 803, 157, 801, 159, 800, 160, 800, 160, 800,
                        161, 799, 161, 799, 161, 798, 162, 798, 162, 799, 161, 799, 161, 799, 161, 799, 161, 799,
                        161, 799, 161, 799, 161, 799, 162, 798, 162, 798, 162, 798, 162, 798, 162, 798, 162, 798,
                        162, 799, 161, 799, 161, 799, 162, 798, 162, 798, 162, 798, 162, 798, 162, 798, 162, 798,
                        162, 798, 162, 798, 162, 798, 162, 798, 162, 798, 163, 797, 163, 797, 163, 798, 162, 798,
                        162, 798, 162, 798, 163, 797, 163, 797, 163, 797, 163, 797, 163, 797, 163, 797, 163, 798,
                        162, 798, 162, 798, 162, 798, 162, 798, 163, 797, 163, 797, 163, 797, 163, 797, 163, 797,
                        163, 797, 163, 797, 164, 796, 164, 796, 164, 797, 163, 797, 163, 797, 162, 798, 162, 798,
                        162, 798, 162, 798, 160, 800, 158, 802, 155, 805, 151, 810, 145, 815, 143, 817, 140, 820,
                        136, 824, 131, 829, 128, 832, 124, 836, 121, 839, 117, 843, 113, 847, 111, 849, 107, 853,
                        104, 857, 100, 860, 95, 865, 91, 869, 87, 873, 83, 877, 80, 880, 76, 884, 72, 888, 70,
                        891, 63, 897, 60, 900, 57, 903, 55, 906, 51, 910, 47, 914, 42, 920, 35, 929, 27, 939, 15,
                        322991],
             "size": [960, 1280]}
}
object_hypothesis_config_tomato_soup_can = {
    "name": "tomato_soup_can",
    "class_id": 5,
    "object_path": ["meshes", "tomato_soup_can", "textured.obj"],
    "bbox": [954, 497, 191, 114],
    "mask": {"counts": [916420, 8, 948, 15, 942, 20, 937, 24, 934, 27, 930, 31, 925, 37, 919, 42, 913, 48, 907, 53,
                        903, 58, 899, 62, 894, 67, 889, 71, 886, 75, 881, 79, 877, 84, 872, 88, 867, 94, 863, 97,
                        861, 99, 860, 101, 857, 103, 856, 105, 854, 106, 854, 106, 853, 107, 852, 109, 850, 110,
                        850, 110, 850, 110, 849, 112, 848, 112, 848, 112, 847, 113, 847, 113, 847, 113, 847, 113,
                        847, 114, 846, 114, 846, 114, 846, 114, 846, 114, 846, 113, 847, 113, 847, 113, 847, 113,
                        847, 113, 847, 113, 847, 113, 847, 112, 848, 112, 848, 112, 848, 112, 848, 111, 849, 111,
                        849, 110, 851, 109, 851, 108, 852, 107, 854, 105, 855, 104, 856, 103, 857, 101, 860, 96,
                        864, 93, 867, 89, 872, 84, 876, 82, 879, 77, 883, 73, 887, 69, 892, 65, 895, 63, 898, 60,
                        901, 56, 904, 54, 907, 51, 910, 48, 912, 45, 916, 42, 919, 39, 922, 36, 925, 33, 927, 31,
                        930, 27, 935, 23, 938, 20, 941, 17, 945, 12, 952, 1, 226037],
             "size": [960, 1280]}
}


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "demo_multiple_object"

    def implementation(self):
        """
        Create a basic pipeline that estimates the 6D pose estimation for multiple objects visible
        in the ROS2 "2025-01-09-11-12-19-YCB-drill-cheezit-mustard-masterchef-domino-tomatosoup.bag" file.
        """
        pkg_dir = get_pkg_share_dir("robokudo_cad_data")
        object_hypothesis_config_list = [
            object_hypothesis_config_mustard,
            object_hypothesis_config_cracker_box,
            object_hypothesis_config_sugar_box,
            object_hypothesis_config_tomato_soup_can
        ]

        kinect_camera_config = robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform.CameraConfig()
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(kinect_camera_config))

        def create_hypothesis(context):
            annotations = []

            for object_hypothesis_config in object_hypothesis_config_list:
                object_hypothesis = ObjectHypothesis()
                (x, y, w, h) = object_hypothesis_config["bbox"]
                object_hypothesis.roi.roi.pos.x = x
                object_hypothesis.roi.roi.pos.y = y
                object_hypothesis.roi.roi.width = w
                object_hypothesis.roi.roi.height = h

                full_mask = rle_to_mask(object_hypothesis_config["mask"])  # 960 x 1280
                object_hypothesis.roi.mask = np.zeros((h, w), dtype=np.uint8)
                object_hypothesis.roi.mask[full_mask[y: y+h, x: x+w]] = 1
                object_hypothesis.annotations = []

                classification = Classification()
                classification.class_id = object_hypothesis_config["class_id"]
                classification.classname = object_hypothesis_config["name"]
                classification.confidence = 1.0
                object_hypothesis.annotations.append(classification)

                annotations.append(object_hypothesis)

            context.get_cas().annotations = annotations

        lambda_descriptor = LambdaFunctionAnnotator.Descriptor()
        lambda_descriptor.parameters.func = create_hypothesis

        fp_desc = FoundationPoseAnnotator.Descriptor()
        fp_desc.parameters.mesh_files = [osp.join(pkg_dir, *ohc["object_path"]) for ohc in object_hypothesis_config_list]
        fp_desc.parameters.mesh_obj_ids = [ohc["class_id"] for ohc in object_hypothesis_config_list]
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
