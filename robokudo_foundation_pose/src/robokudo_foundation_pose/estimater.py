# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import uuid
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import imageio
import matplotlib.pyplot as plt
from .Utils import *
from .Utils import (
    erode_depth,
    bilateral_filter_depth,
    depth2xyzmap,
    depth2xyzmap_batch,
    toOpen3dCloud,
    compute_mesh_diameter,
    make_mesh_tensors,
    sample_views_icosphere,
    euler_matrix,
    set_seed,
)
RX180 = torch.tensor([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]], dtype=torch.float32)

from robokudo_foundation_pose.learning.training.predict_score import ScorePredictor
from robokudo_foundation_pose.learning.training.predict_pose_refine import PoseRefinePredictor

# Visualization helper for object mask
def show_mask(mask, title="Object Mask"):
    """
    Display a binary or integer mask using matplotlib.
    Any non-zero pixel in mask will be shown white.
    """
    m = mask > 0
    plt.figure(figsize=(6, 6))
    plt.imshow(m, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.show()

class FoundationPose:
    def __init__(
        self,
        model_pts,
        model_normals,
        symmetry_tfs=None,
        mesh=None,
        scorer: ScorePredictor=None,
        refiner: PoseRefinePredictor=None,
        glctx=None,
        debug=0,
        debug_dir='/home/bowen/debug/novel_pose_debug/'
    ):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)

        self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
        self.make_rotation_grid(min_n_views=40, inplane_step=60)

        self.glctx = glctx
        self.scorer = scorer if scorer else ScorePredictor()
        self.refiner = refiner if refiner else PoseRefinePredictor()
        self.pose_last = None

    def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)

        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        self.vox_size = max(self.diameter / 20.0, 0.003)
        logging.info(f'self.diameter: {self.diameter}, vox_size: {self.vox_size}')

        pcd = toOpen3dCloud(mesh.vertices, normals=model_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)
        self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
        self.normals = F.normalize(
            torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'),
            dim=-1
        )

        self.mesh = mesh
        self.mesh_tensors = make_mesh_tensors(self.mesh)
        self.symmetry_tfs = (
            torch.eye(4).float().cuda()[None]
            if symmetry_tfs is None
            else torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)
        )
        logging.info('reset done')

    def get_tf_to_centered_mesh(self):
        tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
        tf_to_center[:3, 3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
        return tf_to_center

    def to_device(self, s='cuda:0'):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v) or isinstance(v, nn.Module):
                logging.info(f'Moving {k} to device {s}')
                self.__dict__[k] = v.to(s)
        for k, t in self.mesh_tensors.items():
            logging.info(f'Moving mesh tensor {k} to device {s}')
            self.mesh_tensors[k] = t.to(s)
        if self.refiner:
            self.refiner.model.to(s)
        if self.scorer:
            self.scorer.model.to(s)
        if self.glctx:
            self.glctx = dr.RasterizeCudaContext(s)

    def make_rotation_grid(self, min_n_views=40, inplane_step=60):
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        rot_grid = []
        for view in cam_in_obs:
            for angle in np.deg2rad(np.arange(0, 360, inplane_step)):
                R = euler_matrix(0, 0, angle)
                cam = view @ R
                rot_grid.append(np.linalg.inv(cam))
        rot_grid = np.asarray(rot_grid)
        if mycpp is not None:
            rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.cpu().numpy())
        else:
            logging.info("mycpp not found; using unclustered rotation grid")
        self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
        logging.info(f'self.rot_grid: {self.rot_grid.shape}')

    def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(center, device='cuda', dtype=torch.float)
        return ob_in_cams

    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros(3)
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = (mask.astype(bool)) & (depth >= 0.001)
        if not valid.any():
            return np.zeros(3)
        zc = np.median(depth[valid])
        pt = (np.linalg.inv(K) @ np.array([uc, vc, 1.0]).reshape(3, 1)) * zc
        return pt.flatten()

    def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
        """Compute pose from RGB-D and mask."""
        set_seed(0)
        logging.info('register() start')
        if self.glctx is None:
            self.glctx = glctx or dr.RasterizeCudaContext()

        # Preprocess depth
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')

        # Resize mask to match depth resolution
        # H, W = depth.shape[:2]
        # mask_uint8 = (ob_mask > 0).astype(np.uint8)
        # resized_masks = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
        # ob_mask = resized_masks.astype(bool)

        # Debug: log and display mask
        # if self.debug >= 1:
        #     logging.info(f'depth.shape: {depth.shape}, mask.shape: {ob_mask.shape}')
        #     try:
        #         show_mask(ob_mask, title='register(): ob_mask')
        #     except Exception as e:
        #         logging.warning(f'Mask display failed: {e}')

        # Early exit if too few valid pixels
        valid = (depth >= 0.001) & ob_mask
        if valid.sum() < 500:
            logging.info('valid too small, returning initial guess')
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth, ob_mask, K)
            return pose

        # Full pipeline
        xyz_map = depth2xyzmap(depth, K)
        poses = self.generate_random_pose_hypo(K, rgb, depth, ob_mask)
        poses_np = poses.cpu().numpy()
        center = self.guess_translation(depth, ob_mask, K)
        poses[:, :3, 3] = torch.tensor(center, device='cuda', dtype=torch.float)

        poses, vis_ref = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses_np,
            normal_map=None,
            xyz_map=xyz_map,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            iteration=iteration,
            get_vis=self.debug>=2
        )
        if vis_ref is not None:
            imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis_ref)

        scores, vis_score = self.scorer.predict(
            mesh=self.mesh,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses_np,
            normal_map=None,
            mesh_tensors=self.mesh_tensors,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            get_vis=self.debug>=2
        )
        if vis_score is not None:
            imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis_score)

        # Select best pose
        ids = torch.as_tensor(scores).argsort(descending=True)
        best_pose = poses[ids[0]] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[ids[0]]
        self.best_id = ids[0]
        self.poses = poses
        self.scores = torch.tensor(scores, device='cuda')

        return best_pose.cpu().numpy()

    def compute_add_err_to_gt_pose(self, poses):
        """Dummy: returns negative ones as placeholder error."""
        return -torch.ones(len(poses), device='cuda', dtype=torch.float)

    def track_one(self, rgb, depth, K, iteration, extra={}):
        if self.pose_last is None:
            logging.error("Call register() before track_one().")
            raise RuntimeError("Call register() before track_one()")
        depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        depth = erode_depth(depth_tensor, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        xyz_map = depth2xyzmap_batch(
            depth[None],
            torch.as_tensor(K, device='cuda')[None],
            zfar=np.inf
        )[0]

        pose, vis = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=self.pose_last[None].cpu().numpy(),
            normal_map=None,
            xyz_map=xyz_map,
            mesh_diameter=self.diameter,
            glctx=self.glctx,
            iteration=iteration,
            get_vis=self.debug>=2
        )
        if self.debug >= 2:
            extra['vis'] = vis
        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).cpu().numpy().reshape(4, 4)
