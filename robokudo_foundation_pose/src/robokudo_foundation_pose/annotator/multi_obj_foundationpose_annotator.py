import numpy as np
import open3d as o3d
import trimesh
import rospy
from typing import Union

import cv2
from robokudo_foundation_pose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from robokudo_foundation_pose.Utils import draw_posed_3d_box, draw_xyz_axis
import robokudo
import robokudo.annotators.core as core
from robokudo.cas import CASViews
from robokudo.utils.annotator_helper import resize_mask
from robokudo.utils.decorators import timer_decorator
from robokudo.utils import cv_helper
import os
from Utils import *

import os
from typing import Union, Tuple

import numpy as np
import torch
import trimesh
import open3d as o3d
import rospy
import py_trees

import cv2
import matplotlib.pyplot as plt

# FoundationPose libs
from robokudo_foundation_pose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from robokudo_foundation_pose.Utils import draw_posed_3d_box, draw_xyz_axis

# Robokudo framework
import robokudo
import robokudo.annotators.core as core
from robokudo.cas import CASViews
from robokudo.utils import cv_helper
from robokudo.utils.annotator_helper import resize_mask
from robokudo.utils.decorators import timer_decorator

# CUDA context (ensure dr is imported where RasterizeCudaContext exists)
#import dr

from Utils import depth2xyzmap, toOpen3dCloud


class MultiObjectFoundationPoseAnnotator(core.ThreadedAnnotator):
    class Descriptor(core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.mesh_files = []            # list of .obj paths
                self.target_names = []          # corresponding class names
                self.est_refine_iter = 5
                self.track_refine_iter = 2
                self.debug = 1
                self.debug_dir = "/tmp/foundationpose_debug"
                self.global_with_depth = True
                self.global_with_visualization = True
                self.axis_size = 0.2

        parameters = Parameters()

    def __init__(self, name="MultiObjectFoundationPoseAnnotator", descriptor=Descriptor()):
        super().__init__(name, descriptor)
        self.estimators = {}
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        # Ensure mesh_files and target_names align
        for mesh_file, target_name in zip(self.descriptor.parameters.mesh_files,
                                          self.descriptor.parameters.target_names):
            mesh = trimesh.load(mesh_file)
            if mesh.units in ('mm', 'millimeter'):
                mesh.apply_scale(0.001)
            elif mesh.units in ('cm', 'centimeter'):
                mesh.apply_scale(0.01)

            # oriented bounds: transform to center origin, and extents
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

            pose_est = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                glctx=glctx,
                debug=self.descriptor.parameters.debug,
                debug_dir=self.descriptor.parameters.debug_dir
            )

            key = target_name.lower()
            self.estimators[key] = {
                'pose_est': pose_est,
                'to_origin': to_origin,
                'extents': extents,
                'bbox': bbox,
                'frame_idx': 0
            }

    def load_image(self):
        img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        if self.descriptor.parameters.global_with_depth:
            try:
                return cv_helper.get_scaled_color_image_for_depth_image(self.get_cas(), img)
            except RuntimeError as e:
                rospy.logerr(f"No color-depth ratio: {e}")
        return img

    def get_masks(self, cas, full_shape: Tuple[int,int]) -> dict:
        masks_by_name = {}
        ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)
        if ratio is None:
            raise RuntimeError("Missing COLOR2DEPTH_RATIO")
        for name in self.descriptor.parameters.target_names:
            masks = []
            for ann in cas.annotations:
                if not getattr(ann, 'annotations', None):
                    continue
                if ann.annotations[0].classname.lower() != name.lower():
                    continue
                if not hasattr(ann, 'roi') or not hasattr(ann.roi, 'mask'):
                    continue
                crop = ann.roi.mask
                x1, y1 = ann.roi.roi.pos.x, ann.roi.roi.pos.y
                sx, sy = ratio
                fx, fy = int(round(1.0/sx)), int(round(1.0/sy))
                small = cv2.resize(crop, (crop.shape[1]//fx, crop.shape[0]//fy), interpolation=cv2.INTER_NEAREST)
                masks.append(cv_helper.full_mask_from_roi(small, x1//fx, y1//fy, full_shape))
            masks_by_name[name.lower()] = masks
        return masks_by_name

    def scale_intrinsics(self, K: Union[o3d.camera.PinholeCameraIntrinsic, np.ndarray],
                          ratio: Tuple[float,float]) -> np.ndarray:
        # Convert to 3x3 array
        if isinstance(K, o3d.camera.PinholeCameraIntrinsic):
            K = np.array(K.intrinsic_matrix).reshape(3,3)
        elif hasattr(K, 'K') and isinstance(K.K, (list, np.ndarray)):
            K = np.array(K.K).reshape(3,3)
        else:
            K = np.array(K).reshape(3,3)
        sx, sy = ratio
        K_scaled = K.copy()
        # scale focal lengths and principal point
        K_scaled[0,0] *= sx
        K_scaled[1,1] *= sy
        K_scaled[0,2] *= sx
        K_scaled[1,2] *= sy
        K_scaled[2,2] = 1.0
        return K_scaled

    def _create_frame(self, pose_in_cam: np.ndarray, size: float = 0.1):
        """Return a coordinate frame, compatible with both old (≤0.16) and new (≥0.17) Open3D."""
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        mesh.transform(pose_in_cam)

        if hasattr(mesh, "material"):  # 0.17 +
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"  # make vertex colours visible
            mesh.material = mat
            geom = mesh
        else:  # 0.16 / legacy
            # TriangleMesh colours *are* respected in legacy viewer, but
            # the Filament–based O3DVisualizer ignores them.  So fall back
            # to a coloured LineSet that is always rendered.
            geom = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

        # geom.name = "object_axes"  # nice to have in the scene tree
        return geom

    def _create_obb(self,
                    pose_in_cam: np.ndarray,
                    extent: np.ndarray
                    ) -> o3d.geometry.OrientedBoundingBox:
        """Return an OBB for these extents, transformed into camera frame."""
        center = pose_in_cam[:3, 3]
        R = pose_in_cam[:3, :3]
        obb = o3d.geometry.OrientedBoundingBox(
            center=center,
            R=R,
            extent=extent
        )
        obb.color = [1.0, 0.0, 0.0]
        return obb

    @timer_decorator
    def compute(self):
        cas = self.get_cas()
        color = self.load_image()
        depth = cas.get(CASViews.DEPTH_IMAGE).astype(np.float32) * 1e-3
        K = cas.get(CASViews.CAM_INTRINSIC)
        ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)
        K_scaled = self.scale_intrinsics(K, ratio)
        geoms = []

        masks_dict = self.get_masks(cas, depth.shape[:2])
        vis = color.copy()

        # per-object estimation and rendering
        for name, masks in masks_dict.items():
            if name not in self.estimators:
                rospy.logwarn(f"No estimator for class '{name}'")
                continue
            est_data = self.estimators[name]
            for mask in masks:
                do_reg = (est_data['frame_idx'] == 0 or
                          est_data['pose_est'].pose_last is None)
                try:
                    if do_reg:
                        pose = est_data['pose_est'].register(
                            K_scaled, color, depth,
                            mask.astype(bool),
                            iteration=self.descriptor.parameters.est_refine_iter)
                    else:
                        pose = est_data['pose_est'].track_one(
                            color, depth, K_scaled,
                            iteration=self.descriptor.parameters.track_refine_iter)
                except RuntimeError as e:
                    rospy.logwarn(f"Estimation failed for '{name}' ({e}), retrying register")
                    pose = est_data['pose_est'].register(
                        K_scaled, color, depth,
                        mask.astype(bool),
                        iteration=self.descriptor.parameters.est_refine_iter)

                est_data['frame_idx'] += 1
                # bring back to original mesh coordinates
                center_pose = pose @ np.linalg.inv(est_data['to_origin'])
                Rx180 = np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ], dtype=float)


                center_pose = center_pose @ Rx180

                # 2D overlay
                vis = draw_posed_3d_box(
                    K_scaled, vis, center_pose,
                    est_data["bbox"])
                vis = draw_xyz_axis(vis, center_pose,
                                    self.descriptor.parameters.axis_size,
                                    K_scaled, thickness=3,
                                    transparency=0,
                                    is_input_rgb=False)
                if self.descriptor.parameters.global_with_visualization:
                    geoms.append(self._create_frame(center_pose))
                    # pass the per-object extents here:
                    geoms.append(self._create_obb(center_pose, est_data['extents']))

        # output
        out = self.get_annotator_output_struct()
        out.set_image(vis)

        if self.descriptor.parameters.global_with_visualization:

            if self.descriptor.parameters.global_with_depth:
                xyz_map = depth2xyzmap(depth, K_scaled)
                valid = depth > 0
                geoms.append(toOpen3dCloud(xyz_map[valid], color[valid]))
            out.set_geometries(geoms)

        return py_trees.Status.SUCCESS