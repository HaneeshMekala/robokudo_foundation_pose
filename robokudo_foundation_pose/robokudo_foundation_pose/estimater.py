# Copyright (c) 2023, NVIDIA CORPORATION.    All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.    Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import imageio
import matplotlib.pyplot as plt
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
import nvdiffrast.torch as dr

from robokudo_foundation_pose.learning.training.predict_score import ScorePredictor
from robokudo_foundation_pose.learning.training.predict_pose_refine import PoseRefinePredictor

from typing_extensions import Optional, Dict, Any

# get robokudo logger
import logging
import robokudo.defs
rk_logger = logging.getLogger(robokudo.defs.PACKAGE_NAME) if False else logging.getLogger("dummy")


def cluster_poses_numpy(angle_diff: float, dist_diff: float, poses_in: np.ndarray, symmetry_tfs: Optional[np.ndarray],
                        degree: bool = True) -> np.ndarray:
    """Creates a sub set containing all unique poses from the given ones.

    :param float angle_diff: Minimum geodesic angle difference between orientation of poses to be different
    :param float dist_diff: Minimum euclidian distance between translation of poses to be different
    :param np.ndarray poses_in: Poses [B, 4, 4] (upper 3x4 matrix part is [R|t])
    :param Optional[np.ndarray] symmetry_tfs: Symmetric Poses [S, 4, 4] (upper 3x4 matrix part is [R|t])
    :param bool degree: If 'True' then 'angle_diff' is in degree, otherwise in radian.
    :return: Poses [P, 4, 4] (upper 3x4 matrix part is)
    :rtype: np.ndarray
    """
    if degree:
        angle_diff = (angle_diff * np.pi) / 180

    if symmetry_tfs is None:
        symmetry_tfs = np.eye(4, dtype=poses_in.dtype)[None]    # 1 x 4 x 4

    unique_poses = poses_in[None, 0]    # 1 x 4 x 4

    for i, pose in enumerate(poses_in[1:]):
        # test translation via euclidian distance
        norm_diff = np.linalg.norm(unique_poses[:, :3, 3] - pose[None, :3, 3], axis=-1) # P'
    
        if np.any(norm_diff < dist_diff):
            # rather similar translation, so test orientation via geodesic distance
            # under consideration of symmetries next
            sym_poses = np.matmul(pose[None], symmetry_tfs)     # S x 4 x 4

            rot_diffs = np.einsum("sij, bkj -> sbik",
                                    sym_poses[:, :3, :3], unique_poses[:, :3, :3])  # S x P' x 3 x 3
            traces = np.trace(rot_diffs, axis1=-2, axis2=-1)    # S x P'
            thetas = np.arccos(np.clip((traces - 1.0) / 2, -1.0, 1.0))  # S x P'

            if np.any(thetas < angle_diff):
                # already seen
                continue

        # new pose
        unique_poses = np.concatenate([unique_poses, pose[None]])   # (P'+1) x 4 x 4

    return unique_poses     # P x 4 x 4


class FoundationPose:
    def __init__(
        self,
        mesh,
        symmetry_tfs=None,
        scorer: ScorePredictor=None,
        refiner: PoseRefinePredictor=None,
        glctx=None,
        debug: bool = False,
        debug_dir: str = "/home/bowen/debug/novel_pose_debug/"
    ):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = debug_dir

        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

        self.rot_grid = None
        self.model_center = None
        self.diameter = None
        self.vox_size = None
        self.mesh = None
        self.mesh_tensors = None
        self.symmetry_tfs = None

        self.reset_object(mesh=mesh, symmetry_tfs=symmetry_tfs)
        self.make_rotation_grid(min_n_views=40, inplane_step=60)

        self.glctx = glctx
        self.scorer = scorer or ScorePredictor()
        self.refiner = refiner or PoseRefinePredictor()

        self.pose_last = None   # Used for tracking; per the centered mesh

    @torch.no_grad()
    def reset_object(self, mesh, symmetry_tfs=None):
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2

        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)

        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        rk_logger.info(f"self.diameter: {self.diameter}, vox_size: {max(self.diameter / 20.0, 0.003)}")

        self.mesh = mesh
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device="cuda", dtype=torch.float)

        rk_logger.info("reset done")

    @torch.no_grad()
    def get_object_settings(self, mesh, symmetry_tfs=None) -> Dict[str, Any]:
        settings = {}

        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        settings["model_center"] = (min_xyz + max_xyz) / 2

        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - settings["model_center"].reshape(1, 3)

        settings["diameter"] = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)

        settings["mesh"] = mesh
        settings["mesh_tensors"] = make_mesh_tensors(mesh)

        if symmetry_tfs is None:
            settings["symmetry_tfs"] = torch.eye(4).float().cuda()[None]
        else:
            settings["symmetry_tfs"] = torch.as_tensor(symmetry_tfs, device="cuda", dtype=torch.float)

        return settings

    def reset_object_with_settings(self, settings: Dict[str, Any]) -> None:
        self.model_center = settings["model_center"]

        self.diameter = settings["diameter"]

        rk_logger.info(f"self.diameter: {self.diameter}, vox_size: {max(self.diameter / 20.0, 0.003)}")

        self.mesh = settings["mesh"]
        self.mesh_tensors = settings["mesh_tensors"]

        self.symmetry_tfs = settings["symmetry_tfs"]

        rk_logger.info("reset done")

    @torch.no_grad()
    def get_tf_to_centered_mesh(self):
        tf_to_center = torch.eye(4, dtype=torch.float, device="cuda")
        tf_to_center[:3, 3] = -torch.as_tensor(self.model_center, device="cuda", dtype=torch.float)
        return tf_to_center

    def to_device(self, s="cuda:0"):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v) or isinstance(v, nn.Module):
                rk_logger.info(f"Moving {k} to device {s}")
                self.__dict__[k] = v.to(s)
        for k, t in self.mesh_tensors.items():
            rk_logger.info(f"Moving mesh tensor {k} to device {s}")
            self.mesh_tensors[k] = t.to(s)
        if self.refiner:
            self.refiner.model.to(s)
        if self.scorer:
            self.scorer.model.to(s)
        if self.glctx:
            self.glctx = dr.RasterizeCudaContext(s)

    @torch.no_grad()
    def make_rotation_grid(self, min_n_views=40, inplane_step=60):
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        rot_grid = []
        for cam_in_ob in cam_in_obs:
            for angle in np.deg2rad(np.arange(0, 360, inplane_step)):
                R_inplane = euler_matrix(0, 0, angle)
                rot_cam_in_ob = cam_in_ob @ R_inplane
                rot_grid.append(np.linalg.inv(rot_cam_in_ob))
        rot_grid = np.asarray(rot_grid)
        rot_grid = cluster_poses_numpy(30, 99999, rot_grid, self.symmetry_tfs.cpu().numpy())
        self.rot_grid = torch.as_tensor(rot_grid, device="cuda", dtype=torch.float)
        rk_logger.info(f"self.rot_grid: {self.rot_grid.shape}")

    @torch.no_grad()
    def generate_random_pose_hypo(self, K, depth, mask):
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(center, device="cuda", dtype=torch.float)
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
        center = (np.linalg.inv(K) @ np.array([uc, vc, 1.0])[:, None]) * zc

        return center.reshape(3)

    @torch.no_grad()
    def register(self, rgb: np.ndarray, depth: np.ndarray, cam_intrinsics: np.ndarray, obj_mask: np.ndarray,
                 iteration: int = 5, num_pose_hypothesis: int = 1, batch_size: Optional[int] = 64) -> torch.Tensor:
        """Compute pose from RGB-D and mask."""
        # set seeds for determinism
        set_seed(0)

        rk_logger.info("register() start")

        # preprocess depth
        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")
        depth_np = depth.detach().cpu().numpy()

        # early exit if too few valid pixels
        valid = (depth_np >= 0.001) & obj_mask
        if valid.sum() < 500:
            rk_logger.info("valid too small, returning initial guess")
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth_np, obj_mask, cam_intrinsics)
            return torch.from_numpy(pose)[None].to("cuda")  # 1 x 4 x 4

        # full pipeline
        #xyz_map = depth2xyzmap(depth, K)
        xyz_map = depth2xyzmap_batch(depth[None], torch.from_numpy(cam_intrinsics)[None].to("cuda"), zfar=np.inf)[0]
        poses = self.generate_random_pose_hypo(K=cam_intrinsics, depth=depth_np, mask=obj_mask)

        if batch_size is None:
            batch_size = poses.shape[0]

        num_batches = np.ceil(poses.shape[0] / batch_size).astype(int)
        refined_poses = [None] * num_batches
        scores = [None] * num_batches
        for i in range(num_batches):
            pose_batch = poses[(i*batch_size):((i+1)*batch_size)]

            batched_refined_poses, vis_ref = self.refiner.predict(
                rgb=rgb,
                depth=depth,
                K=cam_intrinsics,
                ob_in_cams=pose_batch,
                xyz_map=xyz_map,
                normal_map=None,
                get_vis=self.debug,
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                iteration=iteration
            )
            refined_poses[i] = batched_refined_poses

            # clean up cache
            torch.cuda.empty_cache()

            if vis_ref is not None:
                imageio.imwrite(f"{self.debug_dir}/vis_refiner.png", vis_ref)

            batched_scores, vis_score = self.scorer.predict(
                rgb=rgb,
                depth=depth,
                K=cam_intrinsics,
                ob_in_cams=batched_refined_poses,
                normal_map=None,
                get_vis=self.debug,
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                glctx=self.glctx,
                mesh_diameter=self.diameter
            )
            scores[i] = batched_scores

            if vis_score is not None:
                imageio.imwrite(f"{self.debug_dir}/vis_score.png", vis_score)

            # clean up cache
            torch.cuda.empty_cache()

        refined_poses = torch.cat(refined_poses, dim=0)     # B x 4 x 4
        scores = torch.cat(scores, dim=0)                   # B

        # select top-k poses based on score
        _, best_indices = torch.topk(scores, k=num_pose_hypothesis, dim=0, largest=True, sorted=False)  # B'
        best_poses = torch.matmul(refined_poses[best_indices], self.get_tf_to_centered_mesh())          # B' x 4 x 4

        return best_poses    # B' x 4 x 4

    @torch.no_grad()
    def track_one(self, rgb: np.ndarray, depth: np.ndarray, cam_intrinsics: np.ndarray, poses: torch.tensor,
                  iteration: int, extra={}, sort_by_score: bool = False) -> torch.Tensor:
        if poses is None:
            raise RuntimeError("Now poses to refine were given.")

        # set seeds for determinism
        set_seed(0)

        # remove 'get_tf_to_centered_mesh' correction
        tcmtf = self.get_tf_to_centered_mesh()  # 4 x 4
        ttfcm = tcmtf.clone()
        ttfcm[:3, 3] *= -1.0

        no_batch = poses.dim() < 3
        if no_batch:
            # add batch dimension
            poses = poses[None, ...]    # 1 x 4 x 4

        poses = torch.matmul(poses, ttfcm)      # B x 4 x 4

        # preprocess depth
        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")

        xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(cam_intrinsics, device="cuda")[None], zfar=np.inf)[0]

        refined_poses, vis_ref = self.refiner.predict(
            rgb=rgb,
            depth=depth,
            K=cam_intrinsics,
            ob_in_cams=poses,
            xyz_map=xyz_map,
            normal_map=None,
            get_vis=self.debug,
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            iteration=iteration
        )

        if self.debug:
            extra["vis_ref"] = vis_ref

        # clean up cache
        torch.cuda.empty_cache()

        if poses.shape[0] > 1 and sort_by_score:
            # sort refined poses by score
            scores, vis_score = self.scorer.predict(
                rgb=rgb,
                depth=depth,
                K=cam_intrinsics,
                ob_in_cams=refined_poses,
                normal_map=None,
                get_vis=self.debug,
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                glctx=self.glctx,
                mesh_diameter=self.diameter
            )

            if vis_score is not None:
                imageio.imwrite(f"{self.debug_dir}/vis_score.png", vis_score)

            # clean up cache
            torch.cuda.empty_cache()

            sort_indices = torch.argsort(scores, stable=False, dim=0, descending=True)
            refined_poses = refined_poses[sort_indices]

        refined_poses = torch.matmul(refined_poses, tcmtf)      # B x 4 x 4

        if no_batch:
            # remove batch dimension
            refined_poses = refined_poses[0]    # 4 x 4

        return refined_poses    # (B x) 4 x 4
