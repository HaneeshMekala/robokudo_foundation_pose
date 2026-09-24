"""A colour-and-size based object detector that feeds FoundationPose.

FoundationPose needs a segmentation mask per object before it can run
``register()``.  This annotator produces those masks for scenes where the
objects are strongly coloured and stand on a neutral surface: it thresholds the
colour image in HSV, splits the result into connected components, and then
identifies each component by measuring its *metric* size from the depth image
and matching that against the known CAD extents.

The size check is what makes the identification view independent - a bar seen
end-on looks compact in 2D but still measures 20 cm in 3D - and it doubles as a
rejection test, so blobs that are the right colour but the wrong size (a blue
sleeve, two objects merged by an occluding hand) are dropped instead of being
handed to the pose estimator as garbage.

Emits one ``ObjectHypothesis`` per accepted blob, carrying the ``ImageROI``
(rectangle plus cropped binary mask, in colour-image resolution) and a
``Classification`` whose ``class_id`` matches the ``mesh_obj_ids`` configured on
``FoundationPoseAnnotator``.  Place it before that annotator in the pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
import py_trees

import robokudo.annotators.core as core
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis

from typing_extensions import Dict, List, Optional, Sequence, Tuple


class ColorBlobDetector(core.BaseAnnotator):
    class Descriptor(core.BaseAnnotator.Descriptor):
        class Parameters:
            """Parameters of the :class:`ColorBlobDetector`.

            Attributes:
                hsv_ranges:              Named HSV bands that together select object pixels. Each entry maps a
                                         colour name to a list of (lower, upper) OpenCV-HSV triples; a colour
                                         needs more than one band when it wraps the hue circle, as red does.
                object_extents:          Maps an object id to the (x, y, z) extents of its CAD model in metres.
                                         Must use the same ids as 'FoundationPoseAnnotator.mesh_obj_ids'.
                object_names:            Maps an object id to the classname written into the 'Classification'.

                min_pixel_area:          Connected components smaller than this (in colour pixels) are ignored.
                morph_kernel_size:       Side length of the close/open kernel used to clean up the threshold.
                min_depth_pixels:        A blob needs at least this many valid depth pixels to be measurable.

                max_distance:            Blobs whose median depth exceeds this (metres) are dropped. Use it to
                                         cut away the background behind the working surface.
                depth_outlier_band:      Points further than this (metres) from the blob's median depth are
                                         discarded before measuring, which removes halo pixels at the silhouette.

                extent_match_tolerance:  Maximum mean error (metres) between the measured and the CAD extents
                                         for a blob to be accepted as that object.
                one_instance_per_class:  If 'True', keep only the best matching blob per object id. Use it when
                                         the scene contains at most one instance of each model; it removes the
                                         duplicate detections that occlusions tend to produce.

                clear_own_hypotheses:    If 'True', drop the 'ObjectHypothesis' this annotator wrote in earlier
                                         iterations before writing new ones. The CAS is not emptied between
                                         pipeline runs, so without this the hypotheses pile up. Set it to
                                         'False' only when something downstream (e.g. an 'ObjectAssociator')
                                         takes over the lifetime of the hypotheses.

                draw_visualization:      If 'True', publish an overlay image to the annotator output.
            """

            def __init__(self):
                self.hsv_ranges: Dict[str, List[Tuple[Sequence[int], Sequence[int]]]] = {
                    # red wraps around hue 0, so it needs a band at each end
                    "red": [((0, 110, 60), (10, 255, 255)),
                            ((170, 110, 60), (180, 255, 255))],
                    "blue": [((95, 110, 60), (130, 255, 255))],
                }
                self.object_extents: Dict[int, Sequence[float]] = {}
                self.object_names: Dict[int, str] = {}

                self.min_pixel_area: int = 1500
                self.morph_kernel_size: int = 7
                self.min_depth_pixels: int = 200

                self.max_distance: float = 1.2
                self.depth_outlier_band: float = 0.15

                self.extent_match_tolerance: float = 0.05
                self.one_instance_per_class: bool = True
                self.clear_own_hypotheses: bool = True

                self.draw_visualization: bool = True
        parameters = Parameters()

    def __init__(self, name: str = "ColorBlobDetector", descriptor: Descriptor = Descriptor()):
        super().__init__(name, descriptor)

        # object id -> CAD extents sorted large to small, prepared once
        self.sorted_extents: Dict[int, np.ndarray] = {}

    def setup(self, timeout: float = None, node=None, visitor=None) -> bool:
        """Validate the configuration and pre-sort the CAD extents."""
        self.rk_logger.debug("{}.setup()".format(self.__class__.__name__))

        params = self.descriptor.parameters

        assert params.object_extents, "'object_extents' is empty - nothing could ever be identified."

        for obj_id, extents in params.object_extents.items():
            extents = np.asarray(extents, dtype=np.float64)
            assert extents.shape == (3,), f"Extents of object {obj_id} must have 3 elements."
            assert np.all(extents > 0), f"Extents of object {obj_id} must be positive."
            # sorting makes the comparison independent of how the mesh happens to be axis-aligned
            self.sorted_extents[obj_id] = np.sort(extents)[::-1]

        return True

    def segment(self, color_image: np.ndarray) -> np.ndarray:
        """Label the object pixels of a BGR image.

        :param np.ndarray color_image: H x W x 3 (BGR)
        :return: H x W int32 label image, 0 is background
        :rtype: np.ndarray
        """
        params = self.descriptor.parameters

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)   # H x W x 3

        selected = np.zeros(hsv.shape[:2], dtype=np.uint8)   # H x W
        for bands in params.hsv_ranges.values():
            for lower, upper in bands:
                selected |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))

        # close first to bridge the specular highlights that split a face in two,
        # then open to drop the speckle the threshold picks up elsewhere
        kernel = np.ones((params.morph_kernel_size, params.morph_kernel_size), np.uint8)
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
        selected = cv2.morphologyEx(selected, cv2.MORPH_OPEN, kernel)

        _, labels = cv2.connectedComponents(selected, connectivity=8)

        return labels   # H x W

    def measure(self, mask_depth: np.ndarray, depth: np.ndarray, cam_intrinsics: np.ndarray) \
            -> Optional[Tuple[np.ndarray, float]]:
        """Measure the metric extents of the surface patch a mask covers.

        Back-projects the masked depth pixels into camera space and takes the spread
        along the principal axes of that point set. Only the object's visible side is
        measured, so the smallest of the three numbers is unreliable - the caller
        compares the two largest.

        :param np.ndarray mask_depth: H' x W' bool mask in depth resolution
        :param np.ndarray depth: H' x W' depth in metres
        :param np.ndarray cam_intrinsics: 3 x 3 intrinsics matching the depth resolution
        :return: Extents (3, sorted large to small) and the median distance, or 'None'
        :rtype: Optional[Tuple[np.ndarray, float]]
        """
        params = self.descriptor.parameters

        rows, cols = np.nonzero(mask_depth & (depth > 0.0))
        if len(cols) < params.min_depth_pixels:
            return None

        # back-project the pixels to camera space: p = z * K^-1 * (u, v, 1)
        z = depth[rows, cols].astype(np.float64)                        # N
        pixels = np.vstack([cols, rows, np.ones_like(cols)])            # 3 x N
        points = (np.linalg.inv(cam_intrinsics) @ pixels * z).T         # N x 3

        # the silhouette of a depth image carries a halo of interpolated pixels that
        # sit between the object and the background - cut them before measuring
        points = points[np.abs(points[:, 2] - np.median(points[:, 2])) < params.depth_outlier_band]
        if len(points) < params.min_depth_pixels:
            return None

        distance = float(np.median(points[:, 2]))
        if distance > params.max_distance:
            return None

        # principal axes of the patch, so the measurement does not depend on how the
        # object happens to be rotated with respect to the camera
        centered = points - points.mean(axis=0)                         # N x 3
        _, _, basis = np.linalg.svd(centered, full_matrices=False)      # 3 x 3
        projected = centered @ basis.T                                  # N x 3
        extents = projected.max(axis=0) - projected.min(axis=0)         # 3

        return np.sort(extents)[::-1], distance

    def identify(self, extents: np.ndarray) -> Optional[Tuple[int, float]]:
        """Match measured extents against the CAD models.

        :param np.ndarray extents: 3 measured extents, sorted large to small
        :return: The best object id and its match error in metres, or 'None' if nothing fits
        :rtype: Optional[Tuple[int, float]]
        """
        tolerance = self.descriptor.parameters.extent_match_tolerance

        best_id, best_error = None, np.inf
        for obj_id, cad_extents in self.sorted_extents.items():
            # only the two largest are compared: the third axis points away from the
            # camera and is truncated by self-occlusion
            error = float(np.abs(extents[:2] - cad_extents[:2]).mean())
            if error < best_error:
                best_id, best_error = obj_id, error

        if best_id is None or best_error > tolerance:
            return None

        return best_id, best_error

    def make_hypothesis(self, mask_color: np.ndarray, obj_id: int, confidence: float) -> ObjectHypothesis:
        """Wrap a colour-resolution mask into an 'ObjectHypothesis'.

        :param np.ndarray mask_color: H x W bool mask in colour resolution
        :param int obj_id: Identified object id
        :param float confidence: Match confidence in (0, 1]
        :rtype: ObjectHypothesis
        """
        rows, cols = np.nonzero(mask_color)
        x, y = int(cols.min()), int(rows.min())
        w, h = int(cols.max() - x + 1), int(rows.max() - y + 1)

        roi = ImageROI()
        roi.roi.pos.x = x
        roi.roi.pos.y = y
        roi.roi.width = w
        roi.roi.height = h
        # FoundationPose expects the mask cropped to the rectangle, not the full image
        roi.mask = mask_color[y: y + h, x: x + w].astype(np.uint8)

        classification = Classification()
        classification.source = self.name
        classification.classname = self.descriptor.parameters.object_names.get(obj_id, str(obj_id))
        classification.class_id = obj_id
        classification.confidence = confidence

        hypothesis = ObjectHypothesis()
        hypothesis.source = self.name
        hypothesis.id = str(obj_id)
        hypothesis.roi = roi
        hypothesis.annotations = [classification]

        return hypothesis

    def update(self) -> py_trees.common.Status:
        cas = self.get_cas()
        params = self.descriptor.parameters

        color_image = cas.get_copy(CASViews.COLOR_IMAGE)                        # H x W x 3 (BGR)
        depth = cas.get(CASViews.DEPTH_IMAGE).astype(np.float32) * 1e-3         # H' x W' (in meter)
        color2depth_ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)                 # (sx, sy)

        # the intrinsics describe the colour camera, so scale them to the depth
        # resolution before back-projecting depth pixels - same as FoundationPoseAnnotator
        cam_intrinsics = np.array(cas.get(CASViews.CAM_INTRINSIC).intrinsic_matrix)  # 3 x 3
        cam_intrinsics_scaled = cam_intrinsics.copy()
        cam_intrinsics_scaled[0, :3] *= color2depth_ratio[0]
        cam_intrinsics_scaled[1, :3] *= color2depth_ratio[1]

        depth_height, depth_width = depth.shape[:2]

        labels = self.segment(color_image)                                      # H x W

        # collect every blob that survives the size check, keyed by object id
        candidates: Dict[int, List[Tuple[float, np.ndarray]]] = {}

        for label in range(1, int(labels.max()) + 1):
            mask_color = labels == label                                        # H x W
            if int(mask_color.sum()) < params.min_pixel_area:
                continue

            # bring the mask into depth resolution to sample the depth image
            mask_depth = cv2.resize(mask_color.astype(np.uint8), (depth_width, depth_height),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)   # H' x W'

            measured = self.measure(mask_depth, depth, cam_intrinsics_scaled)
            if measured is None:
                continue

            identified = self.identify(measured[0])
            if identified is None:
                continue

            obj_id, error = identified
            candidates.setdefault(obj_id, []).append((error, mask_color))

        object_hypotheses = []
        for obj_id, blobs in candidates.items():
            if params.one_instance_per_class:
                blobs = [min(blobs, key=lambda blob: blob[0])]

            for error, mask_color in blobs:
                # a perfect match scores 1.0, a match at the tolerance scores ~0.0
                confidence = float(np.clip(1.0 - error / params.extent_match_tolerance, 1e-3, 1.0))
                object_hypotheses.append(self.make_hypothesis(mask_color, obj_id, confidence))

        if params.clear_own_hypotheses:
            # the CAS keeps its annotations across pipeline iterations, so remove the
            # ones this annotator contributed last time before adding the new ones
            cas.annotations = [annotation for annotation in cas.annotations
                               if not (isinstance(annotation, ObjectHypothesis)
                                       and annotation.source == self.name)]

        cas.annotations.extend(object_hypotheses)

        if params.draw_visualization:
            self.get_annotator_output_struct().set_image(
                self.draw(color_image, object_hypotheses))

        self.rk_logger.debug(f"{self.name}: {len(object_hypotheses)} hypothesis")

        return py_trees.common.Status.SUCCESS

    @staticmethod
    def draw(color_image: np.ndarray, hypotheses: List[ObjectHypothesis]) -> np.ndarray:
        """Tint each detected mask and label it with its class and confidence."""
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

        return vis
