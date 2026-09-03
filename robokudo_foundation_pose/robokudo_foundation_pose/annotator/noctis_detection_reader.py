"""Feed pre-computed NOCTIS / CNOS segmentations into a RoboKudo pipeline.

NOCTIS is run offline over a recording and writes BOP-style detection files: a
JSON list per frame, each entry carrying ``scene_id``, ``image_id``,
``category_id``, ``bbox``, ``score`` and a COCO run-length encoded
``segmentation``.  This annotator replays those files as ``ObjectHypothesis``
objects so ``FoundationPoseAnnotator`` can estimate a 6D pose for each one.

Two details are worth knowing.  First, the ``category_id`` in the file indexes
the *template* folders NOCTIS rendered (``obj_000001``, ``obj_000002``, ...),
which need not match the mesh ids configured on FoundationPose - the mapping is
explicit in :attr:`Parameters.category_to_obj_id`.  Second, the ``bbox`` stored
in the file is a few pixels looser than the mask it ships with, so the region of
interest is recomputed from the decoded mask; FoundationPose pastes the mask
crop back with ``full_mask[y:y+h, x:x+w] = crop``, which silently breaks if the
crop and the rectangle disagree.

Frames without a detection produce no hypothesis.  That is the normal case when
NOCTIS was run on a subset of frames: FoundationPose only needs a mask to
initialise, and refines from the previous pose afterwards.
"""

from __future__ import annotations

import glob
import json
import os.path as osp

import cv2
import numpy as np
import py_trees

import robokudo.annotators.core as core
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis

from robokudo_foundation_pose.masks import rle_to_mask

from typing_extensions import Any, Dict, List, Optional, Sequence


class NoctisDetectionReader(core.BaseAnnotator):
    class Descriptor(core.BaseAnnotator.Descriptor):
        class Parameters:
            """Parameters of the :class:`NoctisDetectionReader`.

            Attributes:
                results_path:            A BOP result '.json' file, or a directory holding one file per frame.
                                         Directories are scanned with 'results_glob'.
                results_glob:            Pattern used when 'results_path' is a directory.

                category_to_obj_id:      Maps the NOCTIS 'category_id' to the object id used by
                                         'FoundationPoseAnnotator.mesh_obj_ids'. Categories that are absent
                                         from this map are skipped.
                category_names:          Maps the NOCTIS 'category_id' to the classname of the 'Classification'.

                score_threshold:         Detections scoring below this are discarded.
                one_instance_per_class:  If 'True', keep only the highest scoring detection per object id.

                frame_ids:               Explicit list of frame ids to replay, one per pipeline iteration. When
                                         'None' the reader uses 'start_frame' and 'frame_step' instead.
                start_frame:             Frame id used on the first iteration.
                frame_step:              Amount the frame id advances per iteration. Set it to the stride the
                                         recording is replayed at, not the stride NOCTIS was run at.

                hold_last_detections:    If 'True', re-emit the most recent detections on frames that have none.
                                         Leave it off unless something downstream needs a mask every frame -
                                         a stale mask is worse than no mask for pose estimation.

                clear_own_hypotheses:    If 'True', drop the 'ObjectHypothesis' this annotator wrote in earlier
                                         iterations. The CAS keeps its annotations between pipeline runs.

                draw_visualization:      If 'True', publish an overlay image to the annotator output.
            """

            def __init__(self):
                self.results_path: Optional[str] = None
                self.results_glob: str = "*.json"

                self.category_to_obj_id: Dict[int, int] = {}
                self.category_names: Dict[int, str] = {}

                self.score_threshold: float = 0.0
                self.one_instance_per_class: bool = True

                self.frame_ids: Optional[Sequence[int]] = None
                self.start_frame: int = 0
                self.frame_step: int = 1

                self.hold_last_detections: bool = False

                self.clear_own_hypotheses: bool = True

                self.draw_visualization: bool = True
        parameters = Parameters()

    def __init__(self, name: str = "NoctisDetectionReader", descriptor: Descriptor = Descriptor()):
        super().__init__(name, descriptor)

        # frame id -> list of raw NOCTIS detection dicts
        self.detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}

        self.iteration: int = 0
        self.last_hypotheses: List[ObjectHypothesis] = []

    def setup(self, timeout: float = None, node=None, visitor=None) -> bool:
        """Load every detection file once and index it by frame id."""
        self.rk_logger.debug("{}.setup()".format(self.__class__.__name__))

        params = self.descriptor.parameters

        assert params.results_path, "'results_path' is not set."
        assert params.category_to_obj_id, "'category_to_obj_id' is empty - every detection would be skipped."

        if osp.isdir(params.results_path):
            files = sorted(glob.glob(osp.join(params.results_path, params.results_glob)))
            assert files, f"No file matching '{params.results_glob}' in '{params.results_path}'."
        else:
            files = [params.results_path]

        for path in files:
            with open(path, "r") as file:
                detections = json.load(file)

            for detection in detections:
                # 'image_id' is the frame the detection belongs to; a file that carries
                # several frames is indexed just as well as one file per frame
                self.detections_by_frame.setdefault(int(detection["image_id"]), []).append(detection)

        self.iteration = 0
        self.last_hypotheses = []

        num_detections = sum(len(d) for d in self.detections_by_frame.values())
        frames = sorted(self.detections_by_frame)
        self.rk_logger.debug(
            f"{self.name}: {num_detections} detections over {len(frames)} frames "
            f"({frames[0]}..{frames[-1]})" if frames else f"{self.name}: no detections loaded")

        return True

    def current_frame_id(self) -> int:
        """Frame id this pipeline iteration should replay."""
        params = self.descriptor.parameters

        if params.frame_ids is not None:
            if not len(params.frame_ids):
                return -1
            # wrap around so a looping bag keeps producing detections
            return int(params.frame_ids[self.iteration % len(params.frame_ids)])

        return params.start_frame + self.iteration * params.frame_step

    def build_hypothesis(self, detection: Dict[str, Any], obj_id: int,
                         image_shape: Sequence[int]) -> Optional[ObjectHypothesis]:
        """Turn one NOCTIS detection into an 'ObjectHypothesis'.

        :param Dict[str, Any] detection: One entry of a BOP result file
        :param int obj_id: Object id this detection's category maps to
        :param Sequence[int] image_shape: (height, width) of the colour image
        :rtype: Optional[ObjectHypothesis]
        """
        params = self.descriptor.parameters

        mask = self.decode_mask(detection["segmentation"])   # H x W (bool)

        if mask.shape != tuple(image_shape):
            self.rk_logger.warning(
                f"{self.name}: mask is {mask.shape} but the colour image is {tuple(image_shape)}; "
                f"resizing - check that the detections belong to this recording.")
            mask = cv2.resize(mask.astype(np.uint8), (image_shape[1], image_shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        rows, cols = np.nonzero(mask)
        if not len(rows):
            return None

        # recompute the rectangle from the mask: the 'bbox' in the file is a few
        # pixels looser, and FoundationPose requires the crop to fill it exactly
        x, y = int(cols.min()), int(rows.min())
        w, h = int(cols.max() - x + 1), int(rows.max() - y + 1)

        roi = ImageROI()
        roi.roi.pos.x = x
        roi.roi.pos.y = y
        roi.roi.width = w
        roi.roi.height = h
        roi.mask = mask[y: y + h, x: x + w].astype(np.uint8)

        category_id = int(detection["category_id"])

        classification = Classification()
        classification.source = self.name
        classification.classname = params.category_names.get(category_id, str(obj_id))
        classification.class_id = obj_id
        classification.confidence = float(detection["score"])

        hypothesis = ObjectHypothesis()
        hypothesis.source = self.name
        hypothesis.id = str(obj_id)
        hypothesis.roi = roi
        hypothesis.annotations = [classification]

        return hypothesis

    @staticmethod
    def decode_mask(segmentation: Dict[str, Any]) -> np.ndarray:
        """Decode a COCO RLE segmentation to a boolean mask.

        Handles the uncompressed form NOCTIS writes ('counts' is a list of run
        lengths) and the compressed form pycocotools produces ('counts' is a
        string), which needs pycocotools to decode.

        :rtype: np.ndarray
        """
        counts = segmentation["counts"]

        if isinstance(counts, (str, bytes)):
            try:
                from pycocotools import mask as coco_mask
            except ImportError as error:
                raise ImportError(
                    "These detections use compressed RLE, which needs pycocotools "
                    "('pip install pycocotools').") from error

            rle = dict(segmentation)
            rle["counts"] = counts.encode("utf-8") if isinstance(counts, str) else counts
            return coco_mask.decode(rle).astype(bool)

        return rle_to_mask(segmentation)

    def update(self) -> py_trees.common.Status:
        cas = self.get_cas()
        params = self.descriptor.parameters

        color_image = cas.get_copy(CASViews.COLOR_IMAGE)    # H x W x 3 (BGR)

        frame_id = self.current_frame_id()
        detections = self.detections_by_frame.get(frame_id, [])

        # keep only categories we have a mesh for, and that score well enough
        accepted = []
        for detection in detections:
            category_id = int(detection["category_id"])
            obj_id = params.category_to_obj_id.get(category_id)

            if obj_id is None:
                continue
            if float(detection["score"]) < params.score_threshold:
                continue

            accepted.append((obj_id, detection))

        if params.one_instance_per_class:
            best: Dict[int, Dict[str, Any]] = {}
            for obj_id, detection in accepted:
                if obj_id not in best or detection["score"] > best[obj_id]["score"]:
                    best[obj_id] = detection
            accepted = sorted(best.items())

        object_hypotheses = []
        for obj_id, detection in accepted:
            hypothesis = self.build_hypothesis(detection, obj_id, color_image.shape[:2])
            if hypothesis is not None:
                object_hypotheses.append(hypothesis)

        if not object_hypotheses and params.hold_last_detections:
            object_hypotheses = self.last_hypotheses
        else:
            self.last_hypotheses = object_hypotheses

        if params.clear_own_hypotheses:
            cas.annotations = [annotation for annotation in cas.annotations
                               if not (isinstance(annotation, ObjectHypothesis)
                                       and annotation.source == self.name)]

        cas.annotations.extend(object_hypotheses)

        if params.draw_visualization:
            self.get_annotator_output_struct().set_image(
                self.draw(color_image, object_hypotheses, frame_id))

        self.rk_logger.debug(f"{self.name}: frame {frame_id} -> {len(object_hypotheses)} hypothesis")

        self.iteration += 1

        return py_trees.common.Status.SUCCESS

    @staticmethod
    def draw(color_image: np.ndarray, hypotheses: List[ObjectHypothesis], frame_id: int) -> np.ndarray:
        """Tint each mask and label it with its class and detection score."""
        vis = color_image.copy()

        for hypothesis in hypotheses:
            classification = hypothesis.annotations[0]
            rect = hypothesis.roi.roi
            x, y, w, h = rect.pos.x, rect.pos.y, rect.width, rect.height

            crop = vis[y: y + h, x: x + w]
            selected = hypothesis.roi.mask.astype(bool)
            crop[selected] = (0.45 * np.array((0, 255, 255)) + 0.55 * crop[selected]).astype(np.uint8)

            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(vis, f"{classification.classname} {classification.confidence:.2f}",
                        (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(vis, f"NOCTIS frame {frame_id}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return vis
