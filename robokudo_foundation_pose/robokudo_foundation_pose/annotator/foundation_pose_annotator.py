import os
import os.path as osp
import cv2
import numpy as np
import open3d as o3d
import py_trees
import rclpy

from collections import defaultdict

import transformations as tr

import torch
import trimesh
import nvdiffrast.torch as dr

# Robokudo framework
from robokudo.cas import CAS
import robokudo.annotators.core as core
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.decorators import timer_decorator
from robokudo.utils import cv_helper, transform



from robokudo_foundation_pose.Utils import depth2xyzmap, toOpen3dCloud
from robokudo_foundation_pose.Utils import draw_posed_3d_box, draw_xyz_axis, glcam_in_cvcam
from robokudo_foundation_pose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

from typing_extensions import Optional, List, Union, Dict, Any, Tuple


class FoundationPoseAnnotator(core.ThreadedAnnotator):
    class Descriptor(core.ThreadedAnnotator.Descriptor):
        class Parameters:
            """
            This class contains all parameters that are necessary for the FoundationPoseAnnotator 6D pose estimator and tracker.

            Attributes:
                mesh_files:                  List of 3D/CAD model files to use.
                mesh_obj_ids:                List of the cad model id corresponding to each file in 'mesh_file'
                name_to_obj_id               Dictionary for mapping the classname to the corresponding cad model id.
                default_mesh_scale_factor:   Default mesh scaling factor to make the mesh in meter, if no 'units' are found.

                cam_r_w2c:                   Row-wise 3 x 3 rotation matrix, rotation part of the world to camera frame mapping.
                cam_t_w2c                    3 element translation vector, translation part of the world to camera frame mapping.

                operation_mode:              Define how the annotator operates:
                                                0: Do nothing;
                                                1: Both new pose estimation and tracking;
                                                2: Only new pose estimation;
                                                3: Only new pose estimation, while treating each object as new;
                                                4: Only tracking;
                est_refine_iter:             Number of iterations to refine the coarse pose for first detected objects only.
                est_num_pose_hypothesis:     Number of pose hypothesis to keep as 'PoseAnnotation'.
                est_batch_size:              Batch size used for processing the pose hypothesis.
                track_refine_iter:           Number of iterations to refine the pose from previous frame for tracked objects.

                update_old_pose_annotations  If 'True' updates old 'PoseAnnotation' with the refined pose for tracking,
                                             otherwise add new ones.

                debug:                       Should generated debug information, which are stored in the 'debug_dir' dir.
                debug_dir:                   Directory to store debug information.

                weights_ros_pkg_name:        Name of the ROS2 package containing the model weights.
                use_cuda:                    If 'True', the inference will use CUDA instead of the CPU.
            """

            def __init__(self):
                self.mesh_files: Optional[List[str]] = None
                self.mesh_obj_ids: List[int] = None
                self.default_mesh_scale_factor: float = 1.0
                self.name_to_obj_id: Dict[str, int] = {}

                self.cam_r_w2c: Optional[List[float]] = None
                self.cam_t_w2c: Optional[List[float]] = None

                self.operation_mode: int = 1
                self.est_refine_iter: int = 5
                self.est_num_pose_hypothesis: int = 1
                self.est_batch_size: Optional[int] = 64
                self.track_refine_iter: int = 2

                self.update_old_pose_annotations: bool = False

                self.debug: bool = True
                self.debug_dir: Optional[str] = None

                self.use_cuda: bool = True
                self.weights_ros_pkg_name: str = "robokudo_foundation_pose"
        parameters = Parameters()

    def __init__(self, name="FoundationPoseAnnotator", descriptor=Descriptor()):
        super().__init__(name, descriptor)

        self.device = "cuda" if descriptor.parameters.use_cuda and torch.cuda.is_available() else "cpu"

        self.with_estimation = False
        self.with_tracking = False
        self.always_new = False

        # mesh data
        self.obj_id_to_index = None
        self.to_origin = None
        self.extents = None
        self.mesh_setting = None

        # pose estimator
        self.est = None

        # world in camera and inverse
        self.cam_w2c = None
        self.cam_c2w = None

        # TODO remove after the 'double setup call' fix
        self.setup_guard = False

    def setup(self, timeout: float = None, node: Optional[rclpy.node.Node] = None, visitor=None):
        """
        Delayed initialisation.
        """
        if self.setup_guard:
            # ensure that setup is only called once
            return True

        self.rk_logger.debug("{}.setup()".format(self.__class__.__name__))

        # set operation mode
        operation_mode = self.descriptor.parameters.operation_mode
        self.with_estimation = operation_mode in [1, 2, 3]
        self.with_tracking = operation_mode in [1, 4]
        self.always_new = operation_mode in [3]

        if not (self.with_estimation or self.with_tracking):
            self.setup_guard = True
            return True

        #
        mesh_files = self.descriptor.parameters.mesh_files
        if mesh_files is None:
            mesh_files = []
        num_meshes = len(mesh_files)

        mesh_obj_ids = self.descriptor.parameters.mesh_obj_ids
        if mesh_obj_ids is None:
            mesh_obj_ids = list(range(num_meshes))

        self.obj_id_to_index = {obj_id: i for i, obj_id in enumerate(mesh_obj_ids)}

        assert num_meshes == len(mesh_obj_ids), "Number of mesh files does not match number of ids"

        #  create 'FoundationPose'6d pose estimation model
        self.rk_logger.debug("Initializing model")

        # use a simple box as temporary mesh for first initialisation
        mesh_tmp = trimesh.primitives.Box(extents=np.ones(3), transform=np.eye(4)).to_mesh()

        glctx = dr.RasterizeCudaContext()
        # glctx = dr.RasterizeGLContext()
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        self.est = FoundationPose(
            mesh=mesh_tmp,
            symmetry_tfs=None,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=self.descriptor.parameters.debug,
            debug_dir=self.descriptor.parameters.debug_dir,
        )

        assert self.device == "cuda", "FoundationPose needs to run on 'cuda'."

        # load all CAD models
        self.rk_logger.debug("Load and preprocess all CAD models")

        self.to_origin = [None] * len(mesh_obj_ids)
        self.extents = [None] * len(mesh_obj_ids)
        self.mesh_setting = [None] * len(mesh_obj_ids)

        for i, mesh_file in enumerate(mesh_files):
            # load mesh
            mesh = trimesh.load(mesh_file)

            if isinstance(mesh, trimesh.Scene):
                # concatenate a scene (multiple meshes) to one mesh
                mesh = mesh.dump(concatenate=True)

            # normalize mesh units to meters
            unit = getattr(mesh, "units", None)
            if unit in ("mm", "millimeter"):
                # milimeter
                mesh_scale_factor = 0.001
            elif unit in ("cm", "centimeter"):
                # centimeter
                mesh_scale_factor = 0.01
            elif unit in ["m", "meter"]:
                # meter
                mesh_scale_factor = 1.0
            else:
                # use default scaling factor
                mesh_scale_factor = self.descriptor.parameters.default_mesh_scale_factor

            mesh.apply_scale(mesh_scale_factor)

            self.to_origin[i], self.extents[i] = trimesh.bounds.oriented_bounds(mesh)

            self.mesh_setting[i] = self.est.get_object_settings(mesh=mesh, symmetry_tfs=None)

        # create optional camera to world frame mapping
        cam_r_w2c = self.descriptor.parameters.cam_r_w2c
        if cam_r_w2c:
            cam_r_w2c = np.array(cam_r_w2c).reshape(3, 3)
        else:
            cam_r_w2c = np.eye(3)
        cam_t_w2c = self.descriptor.parameters.cam_t_w2c
        if cam_t_w2c:
            cam_t_w2c = np.array(cam_t_w2c).reshape(3)
        else:
            cam_t_w2c = np.zeros(3)

        cam_w2c = np.eye(4, dtype=cam_r_w2c.dtype)
        cam_w2c[:3, :3] = cam_r_w2c
        cam_w2c[:3, 3] = cam_t_w2c
        self.cam_w2c = cam_w2c  # 4 x 4

        cam_c2w = np.eye(4, dtype=cam_r_w2c.dtype)
        cam_c2w[:3, :3] = cam_r_w2c.T
        cam_c2w[:3, 3] = -np.dot(cam_r_w2c.T, cam_t_w2c)
        self.cam_c2w = cam_c2w  # 4 x 4

        self.setup_guard = True

        return True

    def load_image(self):
        """Load and (if necessary) resize the color image from the CAS."""
        img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        if self.descriptor.parameters.global_with_depth:
            try:
                return cv_helper.get_scaled_color_image_for_depth_image(self.get_cas(), img)
            except RuntimeError as e:
                self.rk_logger.error(f"No color-depth ratio: {e}")
        return img

    def load_detections_from_cas(self, cas: 'CAS', color2depth_ratio: Tuple[float, float], img_height: int, img_width: int) \
            -> Dict[int, List[Dict[str, Union[np.ndarray, 'ObjectHypothesis', List['PoseAnnotation']]]]]:
        """Preprocess the mask from all 'ObjectHypothesis', who annotate a tracked object, and their 'PoseAnnotation'.

        :param 'CAS' cas:
        :param Tuple[float, float] color2depth_ratio:
        :param int img_height:
        :param int img_width:
        :return:
        :rtype: Dict[int, List[Dict[str, Union[np.ndarray, 'ObjectHypothesis', List['PoseAnnotation']]]]]
        """
        name_to_obj_id = self.descriptor.parameters.name_to_obj_id
        num_max_pose_hypothesis = self.descriptor.parameters.est_num_pose_hypothesis
        object_hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)

        obj_type_detections = {}
        for obj_hypo in object_hypotheses:
            if obj_hypo.roi is None or obj_hypo.roi.mask is None:
                # no ROI and/or mask available
                continue

            # search for highest scoring object id
            best_score = -1
            best_classification = None

            for obj_anno in obj_hypo.annotations:
                if isinstance(obj_anno, Classification):
                    obj_class_confidence = obj_anno.confidence

                    if obj_class_confidence and obj_class_confidence > best_score:
                        best_score = obj_class_confidence
                        best_classification = obj_anno

            if best_classification is None:
                # no classifications
                continue

            obj_id = best_classification.class_id or name_to_obj_id.get(best_classification.classname, None)

            if obj_id is None:
                # unknown class
                continue

            # try to map object id to CAD model index
            obj_index = self.obj_id_to_index.get(obj_id, None)

            if obj_index is None:
                # no CAD model available
                continue

            # search for 'PoseAnnotation'
            pose_annos = [obj_anno for obj_anno in obj_hypo.annotations if isinstance(obj_anno, PoseAnnotation)]
            pose_annos_empty = len(pose_annos) == 0

            if self.with_tracking and not pose_annos_empty:
                # tracked object, refine old pose for new frame
                # use only top-k for refinement
                pose_annos = pose_annos[:num_max_pose_hypothesis]

                poses = np.tile(np.eye(4, dtype=np.float32)[None, ...], (len(pose_annos), 1, 1))    # B x 4 x 4
                for i, pose_anno in enumerate(pose_annos):
                    poses[i, :3, :3] = tr.quaternion_matrix(pose_anno.rotation)[:3, :3]     # 3 x 3
                    poses[i, :3, 3] = np.array(pose_anno.translation)   # 3

                full_mask = None
            if self.with_estimation and (pose_annos_empty or (not self.with_tracking and self.always_new)):
                # 'new' object with unknown pose to estimate
                # rescale the color image mask to depth image resolution
                obj_roi = obj_hypo.roi
                x = int(obj_roi.roi.pos.x * color2depth_ratio[0])
                y = int(obj_roi.roi.pos.y * color2depth_ratio[1])
                w = int(obj_roi.roi.width * color2depth_ratio[0])
                h = int(obj_roi.roi.height * color2depth_ratio[1])

                crop_mask = obj_roi.mask
                crop_mask = cv2.resize(crop_mask, (int(crop_mask.shape[1] * color2depth_ratio[0]),
                                                   int(crop_mask.shape[0] * color2depth_ratio[1])),
                                       interpolation=cv2.INTER_NEAREST)

                # embed the mask crop into the full image
                full_mask = np.zeros((img_height, img_width), dtype=bool)   # H x W
                full_mask[y: y+h, x: x+w] = crop_mask   # H x W if this fails the data dims are defect

                poses = None

            # found a new object to pose estimate or existing one with pose to refine
            obj_type_detections.setdefault(obj_id, []).append(
                {"mask": full_mask, "poses": poses, "object_hypothesis": obj_hypo, "pose_annotations": pose_annos})

        return obj_type_detections

    @staticmethod
    def create_pose_annotations(object_hypothesis: 'ObjectHypothesis', poses: np.ndarray):
        """

        :param 'ObjectHypothesis' object_hypothesis:
        :param np.ndarray poses: [K, 4, 4]
        """
        for pose in poses:
            quaternion = tr.quaternion_from_matrix(pose)    # 4 xyzw
            translation_vector = pose[:3, 3]                # 3

            pose_anno = PoseAnnotation()
            pose_anno.rotation = quaternion.tolist()
            pose_anno.translation = translation_vector.tolist()
            pose_anno.source = "FoundationPose"

            object_hypothesis.annotations.append(pose_anno)

    @staticmethod
    def update_pose_annotations(pose_annotations: List['PoseAnnotation'], poses: np.ndarray):
        """

        :param List['PoseAnnotation'] pose_annotations:
        :param np.ndarray poses: [B, 4, 4]
        """
        for pose_anno, pose in zip(pose_annotations, poses):
            quaternion = tr.quaternion_from_matrix(pose)    # 4 xyzw
            translation_vector = pose[:3, 3]                # 3

            pose_anno.rotation = quaternion.tolist()
            pose_anno.translation = translation_vector.tolist()
            pose_anno.source = "FoundationPoseTracking"

    @staticmethod
    def make_3d_visualization(ob_in_cam: np.ndarray, extents: np.ndarray, size: float = 0.2,
                              obj_name: Optional[str] = None) -> None:
        geoms = []

        # bounding box
        obb = o3d.geometry.OrientedBoundingBox(
            center=ob_in_cam[:3, 3],
            R=ob_in_cam[:3, :3],
            extent=extents
        )
        obb.color = [0.0, 1.0, 0.0]
        geoms.append({"name": obj_name + "_3d_bbox", "geometry": obb})

        # coordinate axis
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        mesh.transform(ob_in_cam)
        geoms.append({"name": obj_name + "_axis", "geometry": mesh})

        return geoms

    #@timer_decorator
    def compute(self):
        if not (self.with_estimation or self.with_tracking):
            # nothing to do
            return py_trees.common.Status.SUCCESS

        # get image (BGR), depth, etc. from CAS
        cas = self.get_cas()

        img = self.load_image()     # H x W x 3 (BGR)
        depth = cas.get(CASViews.DEPTH_IMAGE).astype(np.float32) * 1e-3  # H' x W' (in meter)
        color2depth_ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)  # (sx, sy) mapping color -> depth

        if self.descriptor.parameters.global_with_visualization:
            vis2d = img.copy()  # H x W x 3 (BGR)
            geoms = []

        # scaled camera intrinsic
        cam_intrinsics = np.array(cas.get(CASViews.CAM_INTRINSIC).intrinsic_matrix)  # 3 x 3
        cam_intrinsics_scaled = cam_intrinsics.copy()
        cam_intrinsics_scaled[0, :3] *= color2depth_ratio[0]
        cam_intrinsics_scaled[1, :3] *= color2depth_ratio[1]

        # convert to RGB image
        color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # H x W x 3 (RGB)

        # analyse the current 'ObjectHypothesis' for new pose estimation or tracking detection
        depth_height, depth_width = depth.shape[:2]
        obj_type_detections = self.load_detections_from_cas(cas=cas, color2depth_ratio=color2depth_ratio,
                                                            img_height=depth_height, img_width=depth_width)

        # get all parameters from the annotator description
        est_refine_iter = self.descriptor.parameters.est_refine_iter
        est_num_pose_hypothesis = self.descriptor.parameters.est_num_pose_hypothesis
        est_batch_size = self.descriptor.parameters.est_batch_size

        track_refine_iter = self.descriptor.parameters.track_refine_iter
        update_old_pose_annotations = self.descriptor.parameters.update_old_pose_annotations

        # process all detections
        for obj_id, detections in obj_type_detections.items():
            if not len(detections):
                # nothing for that CAD model to do
                continue

            mesh_index = self.obj_id_to_index[obj_id]

            # set Foundation to the current CAD model
            self.est.reset_object_with_settings(self.mesh_setting[mesh_index])

            for detection in detections:
                # process the (multiple) detection of the CAD model
                if detection["poses"] is None:
                    # new object pose estimate
                    poses = self.est.register(rgb=color, depth=depth, cam_intrinsics=cam_intrinsics_scaled,
                                              obj_mask=detection["mask"], iteration=est_refine_iter,
                                              num_pose_hypothesis=est_num_pose_hypothesis,
                                              batch_size=est_batch_size)    # B x 4 x 4
                    update_old_pose_annos = False
                else:
                    # object tracking
                    old_poses = detection["poses"]  # B x 4 x 4

                    # transform poses from world to camera frame
                    old_poses = np.matmul(self.cam_w2c, old_poses)  # B x 4 x 4

                    old_poses = torch.from_numpy(old_poses).to(torch.float32).to(device="cuda")     # B x 4 x 4

                    poses = self.est.track_one(cam_intrinsics=cam_intrinsics_scaled, rgb=color, depth=depth,
                                               poses=old_poses, iteration=track_refine_iter)    # B x 4 x 4
                    update_old_pose_annos = update_old_pose_annotations

                poses = poses.detach().cpu().numpy()    # B x 4 x 4

                # transform poses from camera to world frame
                poses = np.matmul(self.cam_c2w, poses)

                if update_old_pose_annos:
                    # update 'PoseAnnotation'
                    self.update_pose_annotations(pose_annotations=detection["pose_annotations"], poses=poses)
                else:
                    # create new 'PoseAnnotation'
                    self.create_pose_annotations(object_hypothesis=detection["object_hypothesis"], poses=poses)

                if self.descriptor.parameters.global_with_visualization:
                    # first pose is the best one
                    pose = poses[0]  # 4 x 4

                    # draw the 2D bounding box and coordinate axis as 2D overlay
                    center_pose = np.matmul(pose, np.linalg.inv(self.to_origin[mesh_index]))

                    extents = self.extents[mesh_index]  # 3
                    bbox = np.stack([-extents / 2, extents / 2], axis=0)    # 2 x 3

                    vis2d = draw_posed_3d_box(K=cam_intrinsics_scaled, img=vis2d, ob_in_cam=center_pose,
                                              bbox=bbox, line_color=(0, 255, 0), linewidth=2)
                    vis2d = draw_xyz_axis(color=vis2d, ob_in_cam=center_pose, scale=0.1, K=cam_intrinsics_scaled,
                                          thickness=3, transparency=0, is_input_rgb=False)

                    # draw the 3D bounding box and coordinate axis
                    geoms.extend(self.make_3d_visualization(ob_in_cam=center_pose,
                                                            extents=extents,
                                                            obj_name="object_id_{}".format(obj_id)))

        if self.descriptor.parameters.global_with_visualization:
            if self.descriptor.parameters.global_with_depth:
                # create point cloud
                xyz_map = depth2xyzmap(depth, cam_intrinsics_scaled)
                valid = depth > 0
                geoms.append({"name": "point_cloud", "geometry": toOpen3dCloud(xyz_map[valid], color[valid])})

            # set visualizations objects
            out = self.get_annotator_output_struct()
            out.set_image(vis2d)
            out.set_geometries(geoms)

        return py_trees.common.Status.SUCCESS
