# Battat plane: parts, face counts and collision geometry

Inspected on 2026-10-28. Scope: this directory and its `Imported Components` subdirectory.

## Findings

**There are 12 named part meshes and 191,879 triangular faces across their STL files.** More faces can make collision checking and geometry preparation more expensive. These counts identify opportunities to simplify the model, but do not establish that collision computation is currently too slow; no runtime benchmark was performed. Collision cost also depends on the backend, collision representation, number of objects and proximity of their surfaces. [MoveIt collision checking](https://moveit.picknik.ai/main/doc/examples/urdf_srdf/urdf_srdf_tutorial.html#collision-checking), [FCL interfaces](https://github.com/flexible-collision-library/fcl#interfaces).

The recommended solution is to **retain the detailed visual meshes and create separate, simpler collision geometry**, starting with `UnderBody`, `UpperBody` and the wings. Preserve functional openings and mating clearances throughout simplification.

## Current inventory

A face here means one mesh triangle, rather than a CAD surface. Parts are counted by distinct STL filenames. This is an asset inventory, not a verified assembly bill of materials or a count of connected mesh components.

| Part / STL filename | Triangular faces |
| --- | ---: |
| `UnderBody.stl` | 71,910 |
| `UpperBody.stl` | 34,103 |
| `TopWing.stl` | 26,069 |
| `BottomWing.stl` | 25,249 |
| `RearWing.stl` | 20,500 |
| `Propeller.stl` | 3,054 |
| `RearWheelLeft.stl` | 2,266 |
| `FrontWheel.stl` | 2,186 |
| `RearWheelRight.stl` | 2,146 |
| `MotorGrill.stl` | 1,962 |
| `Window.stl` | 1,570 |
| `Chassis.stl` | 864 |
| **Total: 12 parts** | **191,879** |

The five largest meshes contain **177,831 faces (92.7%)**. `UnderBody` alone contributes **37.5%**, making it the first candidate for collision simplification.

### Alternate exports and CAD sources

| Additional file | Triangular faces | Inventory treatment |
| --- | ---: | --- |
| `TopWing.dae` | 26,069 | Alternate export with the same part name and triangle count as its STL |
| `UpperBody.dae` | 32,955 | Alternate export with 1,148 fewer triangles than its STL |
| `Imported Components/TopWing/TopWing.ipt` | Not measured | Native CAD source; excluded from mesh-face totals |
| `Imported Components/BottomWing/BottomWing.ipt` | Not measured | Native CAD source; excluded from mesh-face totals |

There are **14 mesh files and 2 CAD source files**, representing 12 distinct part names. Adding all STL and DAE triangle counts gives **250,903**, but counts both exports of `TopWing` and `UpperBody`. Using the two DAE exports in place of their matching STLs gives **190,731** triangles. Equal face counts do not prove identical geometry.

Both DAE files declare millimetres (`meter="0.001"`). STL does not encode a standard unit declaration, so confirm scale when creating or loading collision assets.

## Why more faces can cost more

- **Geometry preparation:** loading detailed meshes and building acceleration structures or convex decompositions can add startup time and memory use. The repository's Bullet path caches decompositions, so preparation and repeated-query costs should be measured separately. See the local [Bullet mesh loading and cache implementation](../../../semantic_digital_twin/src/semantic_digital_twin/collision_checking/pybullet_collision_detector.py).
- **Triangle-mesh queries:** more surface detail can increase collision work. Bounding-volume hierarchies avoid checking every triangle against every other triangle, so multiplying face counts is not a realistic runtime estimate. [MoveIt collision checking](https://moveit.picknik.ai/main/doc/examples/urdf_srdf/urdf_srdf_tutorial.html#collision-checking), [FCL interfaces](https://github.com/flexible-collision-library/fcl#interfaces).
- **Convex collision queries:** when an engine converts a source mesh to convex geometry, the resulting hull count and hull complexity matter. Source triangle count alone does not describe that runtime geometry; generating too many convex pieces can also be expensive. [PickNik mesh optimization](https://docs.picknik.ai/8/how_to/configuration_tutorials/optimizing_robot_model_meshes/).

No universal triangle limit determines whether these assets are acceptable. Measure the actual collision or planning workload before setting a face budget.

## Relevant behavior in this repository

The generic [mesh parser](../../../semantic_digital_twin/src/semantic_digital_twin/adapters/mesh.py) assigns the same mesh to a body's visual and collision collections. Loading the original STL through that parser therefore starts with its full detail in both collections.

The [world configuration](../../../semantic_digital_twin/src/semantic_digital_twin/world.py) defaults to V-HACD for Bullet mesh decomposition. The [Bullet collision loader](../../../semantic_digital_twin/src/semantic_digital_twin/collision_checking/pybullet_collision_detector.py) decomposes non-convex meshes when a decomposer is configured and caches the result. Disabling decomposition causes non-convex meshes to be treated as convex hulls, as documented in the world configuration. That can change which assembly poses appear collision-free.

The [FCL collision detector](../../../semantic_digital_twin/src/semantic_digital_twin/collision_checking/trimesh_collision_detector.py) instead builds a bounding-volume hierarchy from each body's combined collision mesh. Reducing that collision mesh directly changes the triangle geometry supplied to FCL. These are available code paths; this inspection did not establish which backend currently loads the plane.

## Recommended solution

1. **Separate visual and collision assets.** Keep the existing meshes for appearance and reference. Supply dedicated collision shapes through each body's collision collection or separate visual/collision elements in an assembly description. The current generic mesh parser alone does not make this separation. [MoveIt collision checking](https://moveit.picknik.ai/main/doc/examples/urdf_srdf/urdf_srdf_tutorial.html#collision-checking).
2. **Simplify the five largest parts first.** For mesh-based checking, reduce triangles on broad surfaces and remove detail irrelevant to contact. For simple regions, consider boxes, cylinders or small compounds of primitives. Choose geometric error tolerances from actual task clearances; no numerical tolerance is proposed because those clearances were not provided. This is a recommendation for these assets, not a measured performance result.
3. **Preserve functional geometry.** Retain any bores, sockets, shell interiors, insertion channels and mating surfaces needed by the assembly. A single convex hull can fill an opening and report a false collision during insertion. Use several convex pieces or retain suitable mesh geometry where concavity matters, then validate the full insertion path. [PickNik mesh optimization](https://docs.picknik.ai/8/how_to/configuration_tutorials/optimizing_robot_model_meshes/).
4. **Match the representation to the backend.** For moving concave parts in Bullet/PyBullet, use a suitably small convex decomposition. Bullet's static BVH triangle-mesh shape and PyBullet's forced concave triangle-mesh flag are intended for fixed geometry; this restriction does not apply to every collision library. Reuse the repository's existing V-HACD support where appropriate. Inspect the resulting hulls for closed openings and excessive component counts. [Bullet triangle-mesh shape contract](https://github.com/bulletphysics/bullet3/blob/master/src/BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h), [PyBullet guide](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html).
5. **Compare accuracy and performance before adopting the assets.** Measure cold preparation time, cached load time, collision-query or simulation-step time, and memory under the same backend and settings. Compare clear, touching and penetrating poses, the assembled configuration, and insertion trajectories. Check small clearances with the actual scale and collision margins. Select the simplest representation that preserves the required collision decisions and meets the application's timing needs.

Decimation reduces triangles; convex decomposition changes the shape representation into several convex pieces. A backend may benefit from both, but either operation can remove assembly-critical detail. Neither automatically guarantees a speedup. [PickNik mesh optimization](https://docs.picknik.ai/8/how_to/configuration_tutorials/optimizing_robot_model_meshes/).

## Measurement method and limits

The STL counts were read from each binary STL header and independently checked by loading each file with the existing `trimesh` installation using `process=False`. All 12 counts agreed, and every STL file matched the binary layout size of 84 header/count bytes plus 50 bytes per triangle.

The DAE counts were read from their XML triangle elements. Each file contains one geometry and one geometry instance. The inventory used the project's `cram-env` Python interpreter and `trimesh` 4.12.2. CAD tessellation, mesh validity, physical part multiplicity and runtime collision performance were not measured. No mesh files were modified or dependencies added.
