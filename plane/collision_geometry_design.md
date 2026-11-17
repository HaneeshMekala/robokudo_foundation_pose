# Battat plane collision geometry

Status: plan confirmed on 2026-10-28. The user explicitly instructed that
implementation must not start. The requirements below were accepted during
the interview. Numerical tolerances, inferred assembly poses, and necessary
repairs will be presented as concrete proposals for review if implementation
is subsequently authorized.

## Objective

Reduce the number of triangles in the plane's collision geometry while
preserving assembly openings and clearances. The source directory is
`assembly_models/battat/plane`.

The repository's [glossary](../../../GLOSSARY.md) distinguishes visual geometry
from collision geometry. This design concerns collision geometry.

## Verified starting point

The 12 STL part files contain 191,879 triangles. The body halves and three wings
account for 177,831 triangles, or 92.7% of the total. See the existing
[mesh report](collision_mesh_report.md) for the complete inventory.

Lower-face variants already exist in [the convention directory](../convention):

| Part | Source STL triangles | Existing lower-face variant triangles |
| --- | ---: | ---: |
| UnderBody | 71,910 | 7,116 |
| UpperBody | 34,103 | 16,163 |
| TopWing | 26,069 | 2,443 |
| BottomWing | 25,249 | 7,757 |
| RearWing | 20,500 | 9,609 |

These are candidates for evaluation, not validated collision replacements.
Their coordinate frames differ from the source files; even some lower-face
and high-detail convention variants have different orientations.

Four source meshes are not watertight after loading and processing with the
existing trimesh installation: BottomWing, RearWing, TopWing, and UpperBody.
Mesh boundary defects must be distinguished from functional assembly openings.

STL files have no standard unit declaration. The nested asset repository's
history documents conversion to metres in commit `b7b6673` and a subsequent
wing-scale correction in `5689d0e`. Both DAE files declare millimetres. The
TopWing DAE dimensions agree with its STL after unit conversion; equivalence
of the UpperBody exports has not been established. The reduced convention
meshes have collision-related provenance in commit `1953270`, but no recorded
opening or clearance validation.

No complete readable assembly description, insertion paths, or numerical
clearance requirements were found in the inspected assets and workspace
references. Native Inventor assemblies exist, but their internal constraints
could not be read with installed tools. Their referenced UnderBody and
UpperBody native part files were not found. The DAE scenes provide no assembly
transforms, and the holder offsets supply only partial translations.

| Source part | Boundary edges | Face-connected components |
| --- | ---: | ---: |
| BottomWing | 343 | 2 |
| RearWing | 296 | 7 |
| TopWing | 1,267 | 2 |
| UpperBody | 545 | 2 |
| Each remaining part | 0 | 1 |

These counts follow normal vertex welding. They are not counts of assembly
openings: a closed solid with a through-bore can have no boundary edges.
TopWing's two components contain 13,627 and 12,442 triangles; retaining only
the largest component would discard substantial source geometry.

## Runtime constraints found in the code

The [world defaults](../../../semantic_digital_twin/src/semantic_digital_twin/world.py)
select Bullet and V-HACD decomposition. The
[Bullet loader](../../../semantic_digital_twin/src/semantic_digital_twin/collision_checking/pybullet_collision_detector.py)
can transform a nonconvex source mesh into convex collision pieces. Validation
must therefore examine the geometry actually loaded by the chosen backend.
The source triangle count alone does not describe those pieces.

The [FCL detector](../../../semantic_digital_twin/src/semantic_digital_twin/collision_checking/trimesh_collision_detector.py)
builds a triangle hierarchy from the body's collision mesh. No configured
consumer for these particular plane assets was identified.

Bullet's mesh-loading code does not explicitly set a mesh collision margin.
The box and sphere margins do not establish the mesh margin. The mesh margin
must be checked in the actual runtime before comparing it with small clearances.

The configured interpreter currently lacks the already-declared
`giskardpy_bullet_bindings==1.0.1` and `vhacdx` packages. Consequently, Bullet
validation cannot currently run. Installed trimesh, CoACD, FCL, SciPy, Embree,
rtree, and manifold3d support geometry analysis, but FCL evidence cannot
establish Bullet behavior. Installing new dependencies requires asking the user;
restoring existing declared dependencies does not add a project dependency.

Each precomputed convex piece can be assigned separately to a body's collision
collection to avoid another decomposition. Supplying one nonconvex combined
mesh under the default configuration would still trigger decomposition.

## Accepted requirements

1. Collision geometry must support assembly and disassembly, including parts
   moving through their assembly openings. An assembled-state check alone is
   insufficient.
2. Face count is flexible. When further simplification would change an assembly
   opening or mating clearance, preserve fit and retain the necessary geometry.
3. Protect potential assembly features by default, including openings, cavities,
   and mating surfaces. Removal requires establishing that a feature is
   nonfunctional.
4. Parts are rigid. Assembly does not rely on flexing or press fits.
5. The original meshes define the intended geometry. Physical measurements are
   not the reference for this work.
6. Target CRAM's default Bullet backend. The user answered "yes" to the backend
   question; this interprets that answer as accepting its CRAM/Bullet
   recommendation.
7. Allow intended contact between seated parts and reject penetration.
8. Produce separate collision assets for the 12 source parts and a validation
   report, preserving source coordinate frames and originals. A complete
   assembly loader is outside the accepted deliverable.
9. Infer mating poses and insertion paths, then present them for review before
   treating them as the assembly reference.
10. Propose necessary local mesh repairs for review. Show the changes and their
    effect on protected features before accepting repaired collision geometry.
11. Measure modeled clearances and candidate surface deviations, then propose
    explicit numerical tolerances for review. No arbitrary millimetre threshold
    or permitted loss of clearance has been accepted.

## Implementation plan

1. **Establish the baseline.** Use the 12 original STL parts in metres and keep
   their source coordinate frames. Inventory components and source defects,
   identify potential assembly interfaces, and evaluate alignment of the
   existing reduced variants.
2. **Prepare a review package.** Infer mating relationships, assembled poses,
   and insertion paths. Show those poses and paths, necessary local repairs,
   and candidate deviations. Propose numerical surface and query tolerances
   from the measured geometry and clearances. Present the package for review
   before treating those choices as accepted.
3. **Reduce collision complexity.** Evaluate reusable reduced variants and
   generate candidates with existing decomposition tools. Preserve openings,
   cavities, and mating surfaces. Favor the simplest candidate that satisfies
   the reviewed constraints; retain detail wherever reduction would break fit.
   Keep each convex piece separate for loading into one part's Bullet compound.
4. **Validate actual collision shapes.** Restore the missing declared Bullet
   dependencies in the configured environment before runtime checks. Examine
   the loaded pieces and effective margins. Compare source and candidate
   surfaces in both directions, probe protected openings, and check reviewed
   insertion paths, seated contact, nearby misaligned poses, and deliberately
   penetrating controls. Include the reverse paths for disassembly.
5. **Deliver assets and evidence.** Put collision assets under `collision/`,
   grouped by original part name, with separate convex-piece OBJ files. Include
   loading instructions and the reviewed reference transforms and paths. Report
   triangle counts, convex-piece counts, measured deviations and clearances,
   checks that passed or remain unresolved, and collision preparation/query
   timing comparisons where the runtime is available.

No fixed percentage reduction or runtime target is required. Fit takes priority;
the report will quantify the reduction and performance actually achieved.

The inferred reference, repaired geometry, and numerical tolerances remain
proposals until reviewed. Any original interference is reported separately;
it is not silently removed by shrinking parts. A part that cannot satisfy the
reviewed constraints is reported as unresolved rather than described as a
validated replacement.

Per-part geometry checks alone do not establish insertion success. Sampled
surface and path checks will state their coverage and resolution; they are not
a certificate covering every possible pose or every surface point.

## Design tree status

The interview's decisions about task scope, rigid-part behavior, geometry
reference, feature protection, backend, contact, face budget, and review policy
are settled, and the user confirmed shared understanding. Implementation has
not started and requires a subsequent instruction to begin. Determining viable
reduced variants, exact repairs, inferred poses, and numerical tolerances belongs
to the future reviewed implementation workflow rather than an assumed answer
to an open requirement.
