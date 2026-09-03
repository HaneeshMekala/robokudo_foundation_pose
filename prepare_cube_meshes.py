"""Convert the exported cube PLYs into FoundationPose-ready meshes.

The source meshes are in millimetres and sit at their authored table-frame
position, so their origin is far from the geometry.  FoundationPose reports the
pose of the *mesh origin*, which would make the returned translation an offset
from the table origin rather than the object's own location - so recentre each
mesh on its bounding-box centre and scale to metres (CRAM convention).

The applied offset is printed so the original frame can be recovered.
"""

import os

import numpy as np
import trimesh

SOURCES = ["child_cube_0_colored.ply", "child_cube_2_colored.ply"]
OUT_DIR = "meshes_m"
SCALE = 0.001   # millimetres -> metres


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for src in SOURCES:
        mesh = trimesh.load(src, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # keep per-face colours as vertex colours: FoundationPose renders
        # untextured meshes through mesh.visual.vertex_colors
        colors = np.asarray(mesh.visual.vertex_colors).copy()

        mesh.apply_scale(SCALE)
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        mesh.visual.vertex_colors = colors

        name = os.path.basename(src).replace("_colored", "").replace(".ply", "")
        dst = os.path.join(OUT_DIR, f"{name}.ply")
        mesh.export(dst)

        print(f"{dst}")
        print(f"    extents (m) : {np.round(mesh.extents, 4)}")
        print(f"    origin shift: {np.round(center, 4)} m  (add back to recover table frame)")
        print(f"    watertight  : {mesh.is_watertight}   vertex colours: "
              f"{len(np.unique(np.asarray(mesh.visual.vertex_colors), axis=0))} unique")


if __name__ == "__main__":
    main()
