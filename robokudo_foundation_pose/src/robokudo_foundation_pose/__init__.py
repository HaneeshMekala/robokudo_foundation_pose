"""
FoundationPose package initialisation.
"""

# ── compiled C++ extension ────────────────────────────────────────────────
from importlib import import_module
mycpp = import_module("robokudo_foundation_pose.mycpp")          # cluster_poses()
pytorch3d = import_module("robokudo_foundation_pose.pytorch3d")
# ── high‑level Python classes ─────────────────────────────────────────────
# from .annotator.foundation_pose_annotator import FoundationPoseAnnotator as FoundationPose
# from .annotator.multi_obj_foundationpose_annotator import MultiObjectFoundationPoseAnnotator


__all__ = ["mycpp","pytorch3d"]



