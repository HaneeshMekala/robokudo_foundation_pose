import cv2
import numpy as np
import open3d as o3d
import py_trees
import rospy
import torch
import trimesh
from robokudo_foundation_pose.Utils import *
from Utils import depth2xyzmap, toOpen3dCloud
from robokudo_foundation_pose.Utils import draw_posed_3d_box, draw_xyz_axis

from robokudo_foundation_pose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from robokudo.utils import cv_helper, transform

import robokudo
# Robokudo framework
import robokudo.annotators.core as core
from robokudo.cas import CASViews
from robokudo.utils import cv_helper
from robokudo.utils.decorators import timer_decorator
# Helper for Euler angles
import math
import tf.transformations as tr
def euler_xyz_from_R(R3: np.ndarray):
    """Return XYZ euler (sxyz) matching SOD."""
    M = np.eye(4); M[:3, :3] = R3
    rx, ry, rz = tr.euler_from_matrix(M, axes='sxyz')
    return rx, ry, rz


def rotation_matrix_to_euler_angles(R: np.ndarray):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else:
        roll = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw = 0.0
    return roll, pitch, yaw

class FoundationPoseAnnotator(core.ThreadedAnnotator):
    class Descriptor(core.ThreadedAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.mesh_file = None
                self.target_name = None
                self.est_refine_iter = 10
                self.track_refine_iter = 2
                self.debug = True
                self.debug_dir = "/home/robokudo/tmp/foundationpose_debug"
                self.global_with_depth = True
                self.global_with_visualization = True
        parameters = Parameters()

    def __init__(self, name="FoundationPoseAnnotator", descriptor=None):
        super().__init__(name, descriptor or FoundationPoseAnnotator.Descriptor())
        mesh = trimesh.load(self.descriptor.parameters.mesh_file)

        # Normalize mesh units to meters
        unit = getattr(mesh, 'units', None)
        if unit in ('mm', 'millimeter'):
            mesh.apply_scale(0.001)
        elif unit in ('cm', 'centimeter'):
            mesh.apply_scale(0.01)

        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(mesh)

        if self.descriptor.parameters.target_name is None:
            raise ValueError("target_name must be set")

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug=self.descriptor.parameters.debug,
            debug_dir=self.descriptor.parameters.debug_dir
        )
        self.frame_idx = 0


    def load_image(self):
        img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        if self.descriptor.parameters.global_with_depth:
            try:
                return cv_helper.get_scaled_color_image_for_depth_image(self.get_cas(), img)
            except RuntimeError as e:
                rospy.logerr(f"No color-depth ratio: {e}")
        return img

    def get_masks_by_name(self, cas, target_name, full_shape=None):
        """
        Resize color-space masks into depth resolution using color→depth ratio.
        """
        ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)  # (sx, sy) mapping color→depth
        masks = []
        for ann in cas.annotations:
            if not getattr(ann, 'annotations', None):
                continue
            if ann.annotations[0].classname != target_name:
                continue
            if not hasattr(ann, 'roi') or not hasattr(ann.roi, 'mask'):
                continue
            crop = ann.roi.mask  # in color resolution
            new_w = int(round(crop.shape[1] * ratio[0]))
            new_h = int(round(crop.shape[0] * ratio[1]))
            small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if full_shape is None:
                masks.append(small)
            else:
                # also scale ROI origin
                x = int(round(ann.roi.roi.pos.x * ratio[0]))
                y = int(round(ann.roi.roi.pos.y * ratio[1]))
                masks.append(cv_helper.full_mask_from_roi(small, x, y, full_shape))
        return masks

    def scale_intrinsics_to_depth_resolution(self,K, ratio):
        """
        Scale intrinsics from color-res to depth-res using color→depth ratio = (sx, sy).
        """
        K_mat = np.array(K.intrinsic_matrix if isinstance(K, o3d.camera.PinholeCameraIntrinsic) else K.K).reshape(3,
                                                                                                                  3).astype(
            np.float32)
        sx, sy = ratio
        K_mat[0, 0] *= sx;
        K_mat[0, 2] *= sx
        K_mat[1, 1] *= sy;
        K_mat[1, 2] *= sy
        K_mat[2, 2] = 1.0
        return K_mat

    @timer_decorator
    def compute(self):
        cas = self.get_cas()
        color = self.load_image()
        depth = cas.get(CASViews.DEPTH_IMAGE).astype(np.float32) * 1e-3  # mm→m if needed
        K = cas.get(CASViews.CAM_INTRINSIC)
        ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)
        K_scaled = self.scale_intrinsics_to_depth_resolution(K, ratio)

        masks = self.get_masks_by_name(cas, self.descriptor.parameters.target_name, full_shape=depth.shape[:2])
        if not masks:
            rospy.logwarn("No masks found; skipping frame")
            # Keep pipeline alive
            return py_trees.Status.SUCCESS

        ob_mask = masks[0].astype(bool)
        do_reg = (self.frame_idx == 0) or (self.est.pose_last is None)

        try:
            if do_reg:
                pose = self.est.register(K=K_scaled, rgb=color, depth=depth,
                                         ob_mask=ob_mask, iteration=self.descriptor.parameters.est_refine_iter)
            else:
                pose = self.est.track_one(rgb=color, depth=depth, K=K_scaled,
                                          iteration=self.descriptor.parameters.track_refine_iter)
        except RuntimeError as e:
            rospy.logwarn(f"Track failed ({e}), falling back to register")
            pose = self.est.register(K=K_scaled, rgb=color, depth=depth,
                                     ob_mask=ob_mask, iteration=self.descriptor.parameters.est_refine_iter)

        if self.descriptor.parameters.debug:
            try:
                mesh = self.est.mesh.copy()
                mesh.apply_transform(pose)
                os.makedirs(self.descriptor.parameters.debug_dir, exist_ok=True)
                mesh.export(f"{self.descriptor.parameters.debug_dir}/model_tf_{self.frame_idx}.obj")
            except Exception as e:
                rospy.logwarn(f"Debug export failed: {e}")

        self.frame_idx += 1


        # Proper 180° rotation about camera X-axis
        Rx180 = np.array([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.],
        ], dtype=np.float32)

        # Center pose and adjust for conventions
        center_pose = pose @ np.linalg.inv(self.to_origin)
        center_pose = center_pose @ Rx180
        R = center_pose[:3, :3]
        rx, ry, rz = euler_xyz_from_R(R)
        rospy.loginfo(f"Euler XYZ (rad): x={rx:.3f}, y={ry:.3f}, z={rz:.3f} (z=yaw)")

        # 2D overlay
        vis2d = draw_posed_3d_box(K_scaled, img=color.copy(), ob_in_cam=center_pose,
                                  bbox=np.stack([-self.extents / 2, self.extents / 2]))
        vis2d = draw_xyz_axis(vis2d, ob_in_cam=center_pose, scale=0.1, K=K_scaled, thickness=3)

        out = self.get_annotator_output_struct()
        out.set_image(vis2d)

        if self.descriptor.parameters.global_with_visualization:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame.transform(center_pose)
            obb = o3d.geometry.OrientedBoundingBox(center=center_pose[:3, 3], R=center_pose[:3, :3],
                                                   extent=self.extents)
            obb.color = [1, 0, 0]
            geoms = [frame, obb]
            if self.descriptor.parameters.global_with_depth:
                xyz = depth2xyzmap(depth, K_scaled)
                valid = depth > 0
                geoms.append(toOpen3dCloud(xyz[valid], color[valid]))
            out.set_geometries(geoms)

        return py_trees.Status.SUCCESS
