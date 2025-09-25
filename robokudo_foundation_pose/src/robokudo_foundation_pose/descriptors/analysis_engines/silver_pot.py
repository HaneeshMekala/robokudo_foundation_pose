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
#from robokudo_foundation_pose import MultiObjectFoundationPoseAnnotator
from foundationpose import FoundationPose
import rospy
from rospkg import RosPack
import os


rospack = RosPack()
pkg_path = rospack.get_path('robokudo_cad_data')
obj_path = os.path.join(pkg_path, 'meshes/pot_silver', 'Pot.obj')





class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "single_object_demo"

    def implementation(self):
        """
        Create a basic pipeline that does tabletop segmentation
        """
        kinect_camera_config = robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform.CameraConfig()
        kinect_config = CollectionReaderAnnotator.Descriptor(
            camera_config=kinect_camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(kinect_camera_config))
        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.pretrained_model = "/home/robokudo/Downloads/best.pt"
        yolo_descriptor.parameters.threshold = 0.2
        yolo_descriptor.parameters.precision_mode = True
        fp_desc = FoundationPose.Descriptor()
        fp_desc.parameters.mesh_file = obj_path
        fp_desc.parameters.target_name = "pot_silver"
        # fp_desc.parameters.debug = 1
        fp_annotator = FoundationPose(descriptor=fp_desc)

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

                fp_annotator,

            ])
        return seq
