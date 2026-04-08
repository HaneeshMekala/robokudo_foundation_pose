import os
import os.path as osp
import cv2
import numpy as np
import open3d as o3d
import py_trees
import rclpy
import gc

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
from robokudo.utils import cv_helper

from robokudo_foundation_pose.Utils import depth2xyzmap, toOpen3dCloud
from robokudo_foundation_pose.Utils import draw_posed_3d_box, draw_xyz_axis, glcam_in_cvcam
from robokudo_foundation_pose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

from typing_extensions import Optional, List, Union, Dict, Any, Tuple

# changes orientation to be CRAM conform with x-axis is left, y-axis is backwards and z-axis is up.
cram_to_obj = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]], dtype=np.float32)    # 4 x 4


class FoundationPoseAnnotator(core.ThreadedAnnotator):
    class Descriptor(core.ThreadedAnnotator.Descriptor):
        class Parameters:
            """
            This class contains all parameters that are necessary for the FoundationPoseAnnotator 6D pose estimator and tracker.

            Attributes:
                mesh_files:                  List of 3D/CAD mesh model files to use.
                mesh_obj_ids:                List of the object ids corresponding to each file in 'mesh_file'
                mesh_scale_factors           List of scaling factors to make each mesh corresponding to file in meter.
                default_mesh_scale_factor:   Default scaling factor to make the mesh in meter.
                name_to_obj_id               Dictionary for mapping the classname to the corresponding cad model id.

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

                use_cuda:                    If 'True', the inference will use CUDA instead of the CPU.

                use_cram_visual_axis        If 'True', the axis visualisations will be aligned to the 3D mesh in CRAM
                                            orientation, otherwise they will be aligned to the orientated mesh bounding box.
                enforce_visual_axis_center  If 'True' the axis visualisations will be in the center of the orientated mesh
                                            bounding box but orientation is still dependent on 'use_cram_visual_axis',
                                            otherwise they dependent on mesh origin.
            """

            def __init__(self):
                self.mesh_files: Optional[List[str]] = None
                self.mesh_obj_ids: Optional[List[int]] = None
                self.mesh_scale_factors: Optional[List[float]] = None
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

                self.debug: bool = False
                self.debug_dir: Optional[str] = None

                self.use_cuda: bool = True

                self.use_cram_visual_axis: bool = True
                self.enforce_visual_axis_center: bool = False
        parameters = Parameters()

    def __init__(self, name="FoundationPoseAnnotator", descriptor=Descriptor()):
        super().__init__(name, descriptor)

        self.device = "cuda" if descriptor.parameters.use_cuda and torch.cuda.is_available() else "cpu"

        self.with_estimation = False
        self.with_tracking = False
        self.always_new = False

        # mesh data
        self.obj_id_to_index = None
        self.origin_in_obj = None
        self.extents = None
        self.mesh_setting = None

        # pose estimator
        self.est = None

        # world in camera and inverse
        self.tcw = None
        self.twc = None

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

        # check mesh files, their ids and scaling factors
        mesh_files = self.descriptor.parameters.mesh_files or []
        num_meshes = len(mesh_files)

        mesh_obj_ids = self.descriptor.parameters.mesh_obj_ids
        if mesh_obj_ids is None:
            mesh_obj_ids = list(range(num_meshes))

        assert all(obj_id >= 0 for obj_id in mesh_obj_ids), "Mesh object ids cannot be negative."
        assert len(set(mesh_obj_ids)) == len(mesh_obj_ids), "Mesh object ids are not unique."
        assert num_meshes == len(mesh_obj_ids), "Number of mesh files does not match number of ids"

        self.obj_id_to_index = {obj_id: i for i, obj_id in enumerate(mesh_obj_ids)}

        # use the given mesh scale factors or the default one, if missing
        mesh_scale_factors = defaultdict(lambda: self.descriptor.parameters.default_mesh_scale_factor,
                                         zip(list(range(num_meshes)),
                                             self.descriptor.parameters.mesh_scale_factors or []))

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

        self.origin_in_obj = [None] * len(mesh_obj_ids)
        self.extents = [None] * len(mesh_obj_ids)
        self.mesh_setting = [None] * len(mesh_obj_ids)

        for i, mesh_file in enumerate(mesh_files):
            # load mesh
            mesh = trimesh.load(mesh_file)

            if isinstance(mesh, trimesh.Scene):
                # concatenate a scene (multiple meshes) to one mesh
                mesh = mesh.dump(concatenate=True)

            # normalize mesh units to meters
            mesh.apply_scale(mesh_scale_factors[i])

            # determine the oriented bound box and a transformation from the object to this box frame
            obj_in_origin, self.extents[i] = trimesh.bounds.oriented_bounds(mesh)           # 4 x 4, 3
            origin_in_obj = np.eye(4, dtype=obj_in_origin.dtype)                         # 4 x 4
            origin_in_obj[:3, :3] = obj_in_origin[:3, :3].T                                 # 3 x 3
            origin_in_obj[:3, 3] = -np.dot(origin_in_obj[:3, :3], obj_in_origin[:3, 3])     # 3
            self.origin_in_obj[i] = origin_in_obj                                           # 4 x 4

            self.mesh_setting[i] = self.est.get_object_settings(mesh=mesh, symmetry_tfs=None)

        # create optional camera to world frame and inverse transformation matrices
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

        tcw = np.eye(4, dtype=cam_r_w2c.dtype)  # 4 x 4
        tcw[:3, :3] = cam_r_w2c     # 3 x 3
        tcw[:3, 3] = cam_t_w2c      # 3
        self.tcw = tcw  # 4 x 4

        twc = np.eye(4, dtype=cam_r_w2c.dtype)  # 4 x 4
        twc[:3, :3] = cam_r_w2c.T   # 3 x 3
        twc[:3, 3] = -np.dot(cam_r_w2c.T, cam_t_w2c)    # 3
        self.twc = twc  # 4 x 4

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

            obj_id = best_classification.class_id
            if obj_id is None:
                # try to map class name to object id
                obj_id = name_to_obj_id.get(best_classification.classname, None)

            if obj_id is None:
                # unknown class
                continue

            # test if object id to CAD model index exists
            if obj_id not in self.obj_id_to_index:
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
                    poses_tco = self.est.register(rgb=color, depth=depth, cam_intrinsics=cam_intrinsics_scaled,
                                                  obj_mask=detection["mask"], iteration=est_refine_iter,
                                                  num_pose_hypothesis=est_num_pose_hypothesis,
                                                  batch_size=est_batch_size)    # B x 4 x 4
                    update_old_pose_annos = False
                else:
                    # object tracking
                    # current/old poses as object to world frame
                    old_poses_two = detection["poses"]  # B x 4 x 4

                    # transform poses from world to camera frame
                    old_poses_tco = np.matmul(self.tcw, old_poses_two)  # B x 4 x 4

                    old_poses_tco = torch.from_numpy(old_poses_tco).to(torch.float32).to(device="cuda")     # B x 4 x 4

                    poses_tco = self.est.track_one(cam_intrinsics=cam_intrinsics_scaled, rgb=color, depth=depth,
                                                   poses=old_poses_tco, iteration=track_refine_iter)    # B x 4 x 4
                    update_old_pose_annos = update_old_pose_annotations

                    del old_poses_tco

                # new poses as object to camera frame
                poses_tco = poses_tco.detach().cpu().numpy()    # B x 4 x 4

                # clean up cache
                torch.cuda.empty_cache()
                gc.collect()

                # transform poses from camera to world frame
                poses_two = np.matmul(self.twc, poses_tco)      # B x 4 x 4

                if update_old_pose_annos:
                    # update 'PoseAnnotation'
                    self.update_pose_annotations(pose_annotations=detection["pose_annotations"], poses=poses_two)
                else:
                    # create new 'PoseAnnotation'
                    self.create_pose_annotations(object_hypothesis=detection["object_hypothesis"], poses=poses_two)

                if self.descriptor.parameters.global_with_visualization:
                    # first pose is the best one
                    obj_in_cam = poses_tco[0]  # 4 x 4

                    origin_in_obj = self.origin_in_obj[mesh_index]      # 4 x 4
                    center_pose = np.matmul(obj_in_cam, origin_in_obj)  # 4 x 4

                    if self.descriptor.parameters.use_cram_visual_axis:
                        # change axis orientation to be visual CRAM conform
                        if self.descriptor.parameters.enforce_visual_axis_center:
                            # enforce centered in oriented mesh bounding box
                            translation_from_orig = np.eye(4)                           # 4 x 4
                            translation_from_orig[:3, 3] = origin_in_obj[:3, 3]         # 3
                            obj_in_cam = np.matmul(obj_in_cam, translation_from_orig)   # 4 x 4

                        obj_axis_pose = np.matmul(obj_in_cam, cram_to_obj)      # 4 x 4
                    else:
                        # align 3D axis with the oriented mesh bounding box
                        obj_axis_pose = center_pose     # 4 x 4

                    # draw the 2D bounding box and coordinate axis as 2D overlay
                    extents = self.extents[mesh_index]  # 3
                    bbox = np.stack([-extents / 2, extents / 2], axis=0)    # 2 x 3

                    vis2d = draw_posed_3d_box(K=cam_intrinsics_scaled, img=vis2d, ob_in_cam=center_pose,
                                              bbox=bbox, line_color=(0, 255, 0), linewidth=2)
                    vis2d = draw_xyz_axis(color=vis2d, ob_in_cam=obj_axis_pose, scale=0.1, K=cam_intrinsics_scaled,
                                          thickness=3, transparency=0, is_input_rgb=False)

                    # draw the 3D bounding box and coordinate axis
                    obb = o3d.geometry.OrientedBoundingBox(
                        center=center_pose[:3, 3],
                        R=center_pose[:3, :3],
                        extent=extents
                    )
                    obb.color = (0.0, 1.0, 0.0)     # green
                    geoms.append({"name": "object_id_{}_3d_bbox".format(obj_id), "geometry": obb})

                    # coordinate axis
                    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                    mesh.transform(obj_axis_pose)
                    geoms.append({"name": "object_id_{}_axis".format(obj_id), "geometry": mesh})

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
