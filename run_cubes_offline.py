"""Run FoundationPose over the extracted cube frames using the NOCTIS masks.

This is the offline path: no ROS, no bag playback, fully deterministic.  It walks
the frames NOCTIS produced detections for, estimates a 6D pose per object, and
writes the poses plus an overlay image for each frame.

    source env_fp.sh
    python run_cubes_offline.py                 # all NOCTIS frames
    python run_cubes_offline.py --frames 0 50   # just these
    python run_cubes_offline.py --track         # estimate once, then track

Outputs land in 'runs/<timestamp>/': 'poses.json' and one 'NNNNN.jpg' per frame.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time
import types
from datetime import datetime

import cv2
import numpy as np

from py_trees.common import Status
from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation

from robokudo_foundation_pose.annotator.noctis_detection_reader import NoctisDetectionReader
from robokudo_foundation_pose.annotator.foundation_pose_annotator import FoundationPoseAnnotator


ROOT = osp.dirname(osp.abspath(__file__))

RGB_DIR = osp.join(ROOT, "extracted", "rgb")
DEPTH_DIR = osp.join(ROOT, "extracted", "depth")
CAM_K = osp.join(ROOT, "extracted", "cam_K.txt")

NOCTIS_RESULTS = osp.join(ROOT, "noctis_acrambly", "datasets", "results", "noctis")
NOCTIS_GLOB = "result_tetris_every50_frame_*.json"

MESH_DIR = osp.join(ROOT, "meshes_m")
OBJECT_MESHES = {0: "child_cube_0.ply", 2: "child_cube_2.ply"}
CATEGORY_TO_OBJ_ID = {1: 0, 2: 2}
OBJECT_NAMES = {1: "child_cube_0", 2: "child_cube_2"}

# Full 'table -> camera_color_optical_frame' extrinsic, inverted to give the
# world->camera transform the annotator expects.  Derived from the 'table ->
# camera_link' transform in extracted/tf_static.json composed with the standard
# REP-103 camera_link->optical rotation, which the recording does not publish.
CAM_R_W2C = [0.020329, -0.999638, 0.017644,
              -0.951457, -0.024764, -0.306785,
              0.307111, -0.010551, -0.951615]
CAM_T_W2C = [-0.018652, 0.661975, 0.715832]


class OutputStruct:
    """Stands in for the RoboKudo annotator output when running without a tree."""

    def __init__(self):
        self.image = None
        self.geometries = []

    def set_image(self, image):
        self.image = image

    def set_geometries(self, geometries):
        self.geometries = geometries


class OfflineCAS:
    """A CAS holding one frame, enough for the annotators used here."""

    def __init__(self, cam_intrinsics):
        self.cam_intrinsics = cam_intrinsics
        self.annotations = []
        self._views = {}

    def set_frame(self, color, depth_mm):
        self._views = {
            CASViews.COLOR_IMAGE: color,
            CASViews.DEPTH_IMAGE: depth_mm,
            CASViews.COLOR2DEPTH_RATIO: (1.0, 1.0),
            CASViews.CAM_INTRINSIC: types.SimpleNamespace(intrinsic_matrix=self.cam_intrinsics),
        }

    def get(self, view):
        return self._views[view]

    def get_copy(self, view):
        return self._views[view].copy()

    def filter_annotations_by_type(self, annotation_type):
        return [a for a in self.annotations if isinstance(a, annotation_type)]


def attach(annotator, cas):
    """Wire an annotator to a CAS without a behaviour tree."""
    output = OutputStruct()
    annotator.get_cas = lambda: cas
    annotator.get_annotator_output_struct = lambda: output
    annotator._offline_output = output
    return annotator


def build_detector(frame_ids):
    descriptor = NoctisDetectionReader.Descriptor()
    descriptor.parameters.results_path = NOCTIS_RESULTS
    descriptor.parameters.results_glob = NOCTIS_GLOB
    descriptor.parameters.category_to_obj_id = CATEGORY_TO_OBJ_ID
    descriptor.parameters.category_names = OBJECT_NAMES
    descriptor.parameters.score_threshold = 0.5
    descriptor.parameters.one_instance_per_class = True
    descriptor.parameters.frame_ids = frame_ids
    descriptor.parameters.draw_visualization = False
    return NoctisDetectionReader(descriptor=descriptor)


def build_estimator(track):
    descriptor = FoundationPoseAnnotator.Descriptor()
    descriptor.parameters.mesh_files = [osp.join(MESH_DIR, m) for m in OBJECT_MESHES.values()]
    descriptor.parameters.mesh_obj_ids = list(OBJECT_MESHES)
    descriptor.parameters.default_mesh_scale_factor = 1.0    # meshes are already in metres
    descriptor.parameters.name_to_obj_id = {
        osp.splitext(mesh)[0]: obj_id for obj_id, mesh in OBJECT_MESHES.items()}

    # 1 = estimate a pose when there is none, refine it afterwards
    # 3 = always estimate from scratch, which is what we want frame by frame
    descriptor.parameters.operation_mode = 1 if track else 3
    descriptor.parameters.est_refine_iter = 5
    descriptor.parameters.track_refine_iter = 2
    descriptor.parameters.est_batch_size = 64
    descriptor.parameters.update_old_pose_annotations = True

    descriptor.parameters.cam_r_w2c = CAM_R_W2C
    descriptor.parameters.cam_t_w2c = CAM_T_W2C

    descriptor.parameters.global_with_depth = True
    descriptor.parameters.global_with_visualization = True

    return FoundationPoseAnnotator(descriptor=descriptor)


def available_frames():
    """Frame ids NOCTIS produced detections for, that we also have images for."""
    detector = build_detector([])
    detector.setup()
    frames = sorted(detector.detections_by_frame)
    return [f for f in frames if osp.exists(osp.join(RGB_DIR, f"{f:05d}.jpg"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="frame ids to process (default: every frame NOCTIS detected)")
    parser.add_argument("--track", action="store_true",
                        help="estimate once then refine from the previous pose instead of "
                             "re-estimating every frame")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()

    frames = args.frames if args.frames else available_frames()
    if not frames:
        raise SystemExit("No frames to process - is extracted/ populated?")

    out_dir = args.out or osp.join(ROOT, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    cam_intrinsics = np.loadtxt(CAM_K)
    cas = OfflineCAS(cam_intrinsics)

    detector = attach(build_detector(frames), cas)
    detector.setup()

    estimator = attach(build_estimator(args.track), cas)
    start = time.time()
    estimator.setup()
    print(f"FoundationPose ready in {time.time() - start:.1f}s "
          f"({len(OBJECT_MESHES)} meshes, {'tracking' if args.track else 'estimate every frame'})")

    results = {}
    for frame_id in frames:
        color = cv2.imread(osp.join(RGB_DIR, f"{frame_id:05d}.jpg"))
        depth = cv2.imread(osp.join(DEPTH_DIR, f"{frame_id:05d}.png"), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            print(f"frame {frame_id:5d}: images missing, skipped")
            continue

        cas.set_frame(color, depth)
        if not args.track:
            # drop last frame's hypotheses so every frame starts clean
            cas.annotations = []

        detector.update()
        began = time.time()
        status = estimator.compute()
        elapsed = time.time() - began

        frame_poses = []
        for hypothesis in cas.filter_annotations_by_type(type(cas.annotations[0])) if cas.annotations else []:
            classification = next((a for a in hypothesis.annotations if hasattr(a, "classname")), None)
            for pose in [a for a in hypothesis.annotations if isinstance(a, PoseAnnotation)]:
                frame_poses.append({
                    "object": classification.classname if classification else "?",
                    "class_id": classification.class_id if classification else None,
                    "translation": [round(v, 5) for v in pose.translation],
                    "rotation_xyzw": [round(v, 5) for v in pose.rotation],
                    "source": pose.source,
                })

        results[frame_id] = frame_poses
        summary = "  ".join(
            f"{p['object']}@[{p['translation'][0]:+.3f},{p['translation'][1]:+.3f},"
            f"{p['translation'][2]:+.3f}]" for p in frame_poses) or "no detections"
        print(f"frame {frame_id:5d}: {status.name:8s} {elapsed:5.1f}s  {summary}")

        overlay = estimator._offline_output.image
        if overlay is not None:
            cv2.imwrite(osp.join(out_dir, f"{frame_id:05d}.jpg"), overlay)

    with open(osp.join(out_dir, "poses.json"), "w") as handle:
        json.dump(results, handle, indent=2)

    posed = sum(1 for v in results.values() if v)
    print(f"\n{posed}/{len(results)} frames produced poses -> {out_dir}")


if __name__ == "__main__":
    main()
