"""Live, in-process NOCTIS detector for RoboKudo.

Reads the current RoboKudo COLOR_IMAGE, runs NOCTIS directly in the same
Python process, and converts the resulting masks into ObjectHypothesis
annotations for FoundationPose.
"""

from __future__ import annotations

import os.path as osp
import sys

import cv2
import numpy as np
import py_trees
import torch
import torchvision.transforms as T

import robokudo.annotators.core as core
from robokudo.cas import CASViews
from robokudo.types.scene import ObjectHypothesis

from robokudo_foundation_pose.annotator.noctis_detection_reader import (
    NoctisDetectionReader,
)
from robokudo_foundation_pose.masks import mask_to_rle


class NoctisLiveDetector(NoctisDetectionReader):
    """Run NOCTIS directly on RoboKudo camera frames."""

    class Descriptor(NoctisDetectionReader.Descriptor):
        class Parameters(NoctisDetectionReader.Descriptor.Parameters):
            def __init__(self):
                super().__init__()

                self.noctis_root_dir: str = "/home/tracy/noctis_acrambly"
                self.config_name: str = "run_inference"
                self.dataset_name: str = "tetris"

                # NOCTIS template category -> FoundationPose mesh id
                self.category_to_obj_id = {
                    1: 0,
                    2: 2,
                }

                self.category_names = {
                    1: "child_cube_0",
                    2: "child_cube_2",
                }

                self.score_threshold: float = 0.5
                self.one_instance_per_class: bool = True
                self.clear_own_hypotheses: bool = True
                self.draw_visualization: bool = True

                # Additional Hydra overrides, if needed later.
                self.config_overrides = []

        parameters = Parameters()

    def __init__(
        self,
        name: str = "NoctisLiveDetector",
        descriptor: Descriptor = Descriptor(),
    ):
        super().__init__(name, descriptor)

        self.noctis_model = None
        self.rgb_transform = None
        self.iteration = 0

    def setup(self, timeout: float = None, node=None, visitor=None) -> bool:
        """Instantiate NOCTIS and its reference templates once."""
        self.rk_logger.debug(f"{self.__class__.__name__}.setup()")

        params = self.descriptor.parameters

        assert osp.isdir(params.noctis_root_dir), (
            f"NOCTIS root does not exist: {params.noctis_root_dir}"
        )

        # NOCTIS is a source-tree package whose imports assume the repository
        # root is on sys.path. The RoboKudo process may start elsewhere.
        if params.noctis_root_dir not in sys.path:
            sys.path.insert(0, params.noctis_root_dir)

        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        config_dir = osp.join(params.noctis_root_dir, "configs")

        with initialize_config_dir(
            version_base=None,
            config_dir=config_dir,
            job_name="robokudo_noctis_live",
        ):
            cfg = compose(
                config_name=params.config_name,
                overrides=[
                    f"dataset_name={params.dataset_name}",
                    f"user.local_root_dir={osp.join(params.noctis_root_dir, 'datasets')}",
                    f"model.segmentor_model.grounded_dino_config_dir="
                    f"{osp.join(params.noctis_root_dir, 'src/cfg/gdino')}",
                    *list(params.config_overrides),
                ],
            )

        if cfg.model.onboarding_config.rendering_type != "templates":
            raise RuntimeError(
                "NoctisLiveDetector currently expects "
                "onboarding_config.rendering_type='templates'."
            )

        # Reproduce the reference-dataset construction used by NOCTIS's
        # existing live-camera script.
        ref_cfg = cfg.data.reference_dataloader
        ref_cfg.template_dir = osp.join(
            str(ref_cfg.template_dir),
            str(cfg.template_dir),
            str(cfg.dataset_name),
        )

        ref_dataset = instantiate(ref_cfg)

        self.rk_logger.info(
            f"{self.name}: NOCTIS reference dataset "
            f"{ref_cfg.template_dir} with object ids {ref_dataset.obj_ids}"
        )

        # Return detections instead of writing BOP result files.
        cfg.model.return_detections = True

        model = instantiate(cfg.model)
        model.dataset_name = cfg.dataset_name
        model.ref_dataset = ref_dataset

        # Match NOCTIS's existing live-camera path.
        model.to("cuda")
        model.eval()

        self.noctis_model = model

        self.rgb_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self.iteration = 0

        self.rk_logger.info(
            f"{self.name}: NOCTIS initialized on CUDA for dataset "
            f"'{cfg.dataset_name}'"
        )

        return True

    @torch.inference_mode()
    def update(self) -> py_trees.common.Status:
        """Run NOCTIS on the current RoboKudo color frame."""
        cas = self.get_cas()
        params = self.descriptor.parameters

        color_image = cas.get_copy(CASViews.COLOR_IMAGE)

        if color_image is None:
            self.rk_logger.warning(f"{self.name}: no COLOR_IMAGE in CAS")
            return py_trees.common.Status.FAILURE

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # NOCTIS's RGB-only live path does not use the intrinsic values for
        # segmentation/matching, but test_step expects this field in its batch.
        cam_intrinsic = np.asarray(
            cas.get(CASViews.CAM_INTRINSIC).intrinsic_matrix,
            dtype=np.float32,
        )

        batch = {
            "image": self.rgb_transform(rgb_image)[None],
            "scene_id": ["0"],
            "frame_id": [str(self.iteration)],
            "cam_intrinsic": cam_intrinsic[None],
        }

        results = self.noctis_model.test_step(batch, self.iteration)
        detections = results["detections"]

        # Convert NOCTIS detections into the same dictionary representation
        # expected by NoctisDetectionReader.build_hypothesis().
        accepted = []

        for score, category_id, mask in zip(
            detections.scores,
            detections.object_ids,
            detections.masks,
        ):
            category_id = int(category_id)
            score = float(score)

            obj_id = params.category_to_obj_id.get(category_id)

            if obj_id is None:
                continue

            if score < params.score_threshold:
                continue

            detection = {
                "category_id": category_id,
                "score": score,
                "segmentation": mask_to_rle(
                    np.asarray(mask).astype(bool)
                ),
            }

            accepted.append((obj_id, detection))

        if params.one_instance_per_class:
            best = {}

            for obj_id, detection in accepted:
                if (
                    obj_id not in best
                    or detection["score"] > best[obj_id]["score"]
                ):
                    best[obj_id] = detection

            accepted = sorted(best.items())

        object_hypotheses = []

        for obj_id, detection in accepted:
            hypothesis = self.build_hypothesis(
                detection,
                obj_id,
                color_image.shape[:2],
            )

            if hypothesis is not None:
                object_hypotheses.append(hypothesis)

        if params.clear_own_hypotheses:
            cas.annotations = [
                annotation
                for annotation in cas.annotations
                if not (
                    isinstance(annotation, ObjectHypothesis)
                    and annotation.source == self.name
                )
            ]

        cas.annotations.extend(object_hypotheses)

        if params.draw_visualization:
            self.get_annotator_output_struct().set_image(
                self.draw(
                    color_image,
                    object_hypotheses,
                    self.iteration,
                )
            )

        self.rk_logger.debug(
            f"{self.name}: frame {self.iteration} -> "
            f"{len(object_hypotheses)} hypothesis"
        )

        self.iteration += 1

        return py_trees.common.Status.SUCCESS
