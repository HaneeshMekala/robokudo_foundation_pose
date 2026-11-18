"""Live 6D pose estimation of the Battat 'Little Mechanic' toy plane parts.

One coloured mesh per part (from 'colorize_stl.py', under 'plane/colored/'),
segmented by colour and identified by metric size, then posed by FoundationPose:

    ros2 run robokudo_ros main _ae=demo_plane_parts _ros_pkg=robokudo_foundation_pose

The class names follow the parts list on the Battat instruction sheet
('plane_parts.jpeg'); the letter there is noted next to each part below.

What colour + size can and cannot tell apart (see also README_colored_meshes.md):

* green:  top_wing and bottom_wing are the same colour *and* the same size
          (0.254 x 0.104 m).  Both get a label thanks to 'assign_one_to_one',
          but which wing gets which label is arbitrary.  The two meshes are
          nearly identical, so the poses are still right - only the names can
          be swapped.
* red:    upper_body (0.239 m long) and under_body (0.187 m) differ enough.
* yellow: the rear wing's saturation swings from ~110 to ~210 with the lighting,
          which overlaps the pale propeller and chassis, so all three share one
          band.  rear_wing is told apart by size, but propeller (0.119 x 0.104)
          and chassis (0.120 x 0.103) are the same size and their labels are
          arbitrary - and unlike the wings, their shapes are not alike.
* black:  front_wheel (0.060), motor_grill (0.054) and rear_wheel (0.040) are
          only a few mm apart, so the tolerance is tight and a noisy depth image
          will mix them up.  rear_wheel_left/right are the same size; their
          labels are arbitrary.
* window: clear plastic has no colour to threshold, so it is not detected.

This works best with the parts lying apart.  On the assembled plane, touching
parts of the same colour merge into one blob that matches nothing and is dropped.
"""

import os
import os.path as osp

import robokudo.analysis_engine
import robokudo.idioms
import robokudo.pipeline
# 'robokudo.descriptors' must be imported before 'robokudo.annotators.collection_reader',
# see 'demo_tracy_cubes_live.py'
from robokudo.descriptors import CrDescriptorFactory
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.query import GenerateQueryResult, QueryAnnotator

from robokudo_foundation_pose.annotator.color_blob_detector import ColorBlobDetector
from robokudo_foundation_pose.annotator.foundation_pose_annotator import FoundationPoseAnnotator


# Folder holding '<Part>/<Part>.obj' for every part.  The meshes are already in
# metres, so no scaling is applied.  Override with the PLANE_MESH_DIR variable.
MESH_DIR = os.environ.get("PLANE_MESH_DIR", "/home/tracy/robokudo_foundation_pose/plane/colored")

# object id -> (classname, mesh folder, CAD extents in metres, colour group or None)
# The extents are the axis-aligned bounding box of each STL.
PARTS = {
    0: ("front_wheel",       "FrontWheel",     (0.0599, 0.0600, 0.0300), "black"),    # a
    1: ("top_wing",          "TopWing",        (0.1034, 0.2546, 0.0352), "green"),    # b
    2: ("bottom_wing",       "BottomWing",     (0.2543, 0.1041, 0.0291), "green"),    # c
    3: ("rear_wheel_left",   "RearWheelLeft",  (0.0260, 0.0400, 0.0400), "black"),    # d
    4: ("rear_wheel_right",  "RearWheelRight", (0.0290, 0.0400, 0.0400), "black"),    # d
    5: ("window",            "Window",         (0.0480, 0.0509, 0.0333), None),       # e
    6: ("propeller",         "Propeller",      (0.1039, 0.1188, 0.0160), "yellow"),   # f
    7: ("motor_grill",       "MotorGrill",     (0.0540, 0.0539, 0.0155), "black"),    # g
    8: ("upper_body",        "UpperBody",      (0.2389, 0.0772, 0.0922), "red"),      # h
    9: ("under_body",        "UnderBody",      (0.0772, 0.1869, 0.0433), "red"),      # i
    10: ("rear_wing",        "RearWing",       (0.1152, 0.0868, 0.0887), "yellow"),   # j
    11: ("chassis",          "Chassis",        (0.0300, 0.1200, 0.1030), "yellow"),   # k
}

# the plane has two front wheels, both using the same mesh
MAX_INSTANCES = {0: 2}
          
# OpenCV HSV (H 0-180, S and V 0-255), sampled from 'Plane.jpeg'.  Retune them
# on the real camera: the overlay of the detector shows what each band picks up.
HSV_RANGES = {
    # teal wings; specular glare drops the saturation to ~60
    "green": [((60, 50, 70), (95, 255, 255))],
    # orange-red body, wraps around hue 0
    "red": [((0, 110, 70), (10, 255, 255)),
            ((170, 110, 70), (180, 255, 255))],
    # saturated rear wing and pale propeller/chassis together, see the docstring
    "yellow": [((14, 80, 90), (32, 255, 255))],
    # tyres and grill; anything this dark, whatever the hue
    "black": [((0, 0, 0), (180, 255, 65))],
}

# black parts are only a few mm apart in size, so keep this tight
EXTENT_MATCH_TOLERANCE = 0.02

# a 40 mm rear wheel seen side-on at ~1 m covers roughly 2000 colour pixels
MIN_PIXEL_AREA = 800

MAX_OBJECT_DISTANCE = 1.2

# see 'demo_tracy_cubes_live.py'
COLOR2DEPTH_RATIO = (1.0, 1.0)
POSES_IN_MAP_FRAME = False

# False while tuning: run continuously and watch the detector overlay.
# True: answer 'robokudo_msgs/action/Query' goals on '/robokudo/query'.
QUERY_DRIVEN = False


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "demo_plane_parts"

    def implementation(self):
        detected = {obj_id: part for obj_id, part in PARTS.items() if part[3] is not None}

        reader_config = CrDescriptorFactory.create_descriptor(
            "orbbec",
            color2depth_ratio=COLOR2DEPTH_RATIO,
            lookup_viewpoint=POSES_IN_MAP_FRAME,
            only_stable_viewpoints=False,
        )

        # ---- detector: per-colour threshold + metric size check -> masks ----
        detector_desc = ColorBlobDetector.Descriptor()
        params = detector_desc.parameters
        params.hsv_ranges = HSV_RANGES
        params.object_extents = {obj_id: extents for obj_id, (_, _, extents, _) in detected.items()}
        params.object_names = {obj_id: name for obj_id, (name, _, _, _) in detected.items()}
        params.object_colors = {obj_id: colour for obj_id, (_, _, _, colour) in detected.items()}
        params.object_max_instances = MAX_INSTANCES
        params.one_instance_per_class = True
        params.assign_one_to_one = True
        params.extent_match_tolerance = EXTENT_MATCH_TOLERANCE
        params.min_pixel_area = MIN_PIXEL_AREA
        params.max_distance = MAX_OBJECT_DISTANCE

        # ---- pose estimation ------------------------------------------------
        # Only the meshes the detector can label are loaded; the window would
        # never receive a mask.
        fp_desc = FoundationPoseAnnotator.Descriptor()
        fp_desc.parameters.mesh_files = [
            osp.join(MESH_DIR, folder, folder + ".obj") for _, folder, _, _ in detected.values()]
        fp_desc.parameters.mesh_obj_ids = list(detected.keys())
        fp_desc.parameters.default_mesh_scale_factor = 1.0   # meshes are already in metres
        fp_desc.parameters.name_to_obj_id = {name: obj_id for obj_id, (name, _, _, _) in detected.items()}

        # mode 1 behaves like mode 3 here, see 'demo_tracy_cubes_live.py'
        fp_desc.parameters.operation_mode = 1
        fp_desc.parameters.est_refine_iter = 5
        fp_desc.parameters.track_refine_iter = 2
        fp_desc.parameters.est_batch_size = 64
        fp_desc.parameters.update_old_pose_annotations = True

        perception = [
            CollectionReaderAnnotator(descriptor=reader_config),
            ColorBlobDetector(descriptor=detector_desc),
            FoundationPoseAnnotator(descriptor=fp_desc),
        ]

        if QUERY_DRIVEN:
            perception = [QueryAnnotator(), *perception, GenerateQueryResult()]

        seq = robokudo.pipeline.Pipeline("PlanePartsPipeline")
        seq.add_children([robokudo.idioms.pipeline_init(), *perception])
        return seq
