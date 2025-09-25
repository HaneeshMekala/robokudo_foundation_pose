import robokudo.analysis_engine

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator

import robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform

import robokudo.io.camera_interface
import robokudo.idioms
from robokudo_yolo.annotators.YoloAnnotator import YoloAnnotator
from foundationpose import MultiObjectFoundationPoseAnnotator
from foundationpose import FoundationPose

from rospkg import RosPack
import os


rospack = RosPack()
pkg_path = rospack.get_path('robokudo_cad_data')



class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "demo"

    def implementation(self):
        """
        Create a basic pipeline that does tabletop segmentation
        """
        kinect_camera_config = robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform.CameraConfig()
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(kinect_camera_config))
        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.pretrained_model = "/home/robokudo/best.pt"
        yolo_descriptor.parameters.threshold = 0.2
        yolo_descriptor.parameters.precision_mode = True


        multi_fp_desc = MultiObjectFoundationPoseAnnotator.Descriptor()
        multi_fp_desc.parameters.mesh_files = [
            os.path.join(pkg_path, "meshes/004_sugar_box","textured.obj"),
            os.path.join(pkg_path, 'meshes/003_cracker_box',"textured.obj"),
            os.path.join(pkg_path, 'meshes/005_tomato_soup_can.obj',"textured.obj"),
            # Add more as needed
        ]
        multi_fp_desc.parameters.target_names = [
            "004_sugar_box",
            "003_cracker_box",
            "005_tomato_soup_can",
        ]

        multi_fp_annotator = MultiObjectFoundationPoseAnnotator(descriptor=multi_fp_desc)

        seq = robokudo.pipeline.Pipeline("RWPipeline")
        seq.add_children(
            [
                # PipelineTrigger(),
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=kinect_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                YoloAnnotator(descriptor=yolo_descriptor),
                multi_fp_annotator,
            ])
        return seq
