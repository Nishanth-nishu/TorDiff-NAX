# DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking
- **Citation:** G. Corso, H. Stärk, B. Jing, R. Barzilay, T. Jaakkola. ICLR 2023. arXiv:2210.01776
- **URL:** https://arxiv.org/abs/2210.01776 · PDF: `2022_corso_diffdock.pdf` · text: `fulltext/2022_corso_diffdock.txt`
- **Tags:** technique, product-manifold diffusion T(3) × SO(3) × SO(2)^m

## Key idea
Treats docking as generative modelling of the ligand pose over the **product space P = T(3) × SO(3) × SO(2)^m** (translation, rotation, torsions), with a map A: P × R^{3n} → R^{3n} (Sec. 4.2–4.3). The paper proves that torsion updates can be disentangled from rigid-body motion (by re-aligning after each torsion update), so independent diffusions on each factor define a valid diffusion on the pose manifold. One extrinsic SE(3)-equivariant network outputs the score for every factor. Ligand local structures come from RDKit, as in TD, with the same conformer-matching step.

## Relation to the three gaps
- Gap 1: **not addressed.** Ligand local structure stays frozen at RDKit's.
- Gap 2: **not addressed.** Uses the same conformer-matching workaround.
- Gap 3: **not addressed directly**, but it gives **the exact recipe for adding extra manifold factors** to TD's torus.

## Metrics
PDBBind docking only; no GEOM numbers.

## Reusable for FlexiTors
- Template for **T^m × (bond-angle relaxation space)**: the product-space construction, the proof pattern (disentangle each factor's action), independent heat kernels per factor, and one extrinsic network that outputs a score per factor.
- Separate σ schedules per factor.
