# Harmonic Self-Conditioned Flow Matching for joint Multi-Ligand Docking and Binding Site Design (HarmonicFlow / FlowSite)
- **Citation:** H. Stärk, B. Jing, R. Barzilay, T. Jaakkola. ICML 2024 (PMLR 235). arXiv:2310.05764
- **URL:** https://arxiv.org/abs/2310.05764 · PDF: `2023_stark_harmonicflow.pdf` · text: `fulltext/2023_stark_harmonicflow.txt`
- **Tags:** technique, harmonic prior, self-conditioned flow matching

## Key idea
HarmonicFlow generates ligand 3D coordinates in the pocket with flow matching from a **harmonic prior**: a Gaussian whose precision is the bond-graph Laplacian, so bonded atoms start close together. Self-conditioning feeds back the previous x̂_1 prediction. It replaces DiffDock's product-space (translation, rotation, torsion) diffusion with a simpler Cartesian flow that "improves upon the state-of-the-art generative process" in simplicity and average sample quality. FlowSite adds discrete residue-type generation.

## Relation to the three gaps
- Gap 1: **addresses (for docking).** No frozen RDKit local structure; bond lengths and angles are generated.
- Gap 2: **addresses.** No conformer matching needed.
- Gap 3: **addresses implicitly** (Cartesian).
- It is the direct ancestor of ET-Flow's harmonic prior for conformers.

## Metrics
PDBBind / Binding MOAD docking; no GEOM numbers.

## Reusable for FlexiTors
- Harmonic (Laplacian) prior, if FlexiTors adds a Cartesian local-refinement head.
- **Self-conditioning**, which is cheap and improves flow models.
- Evidence from the TD/DiffDock group that relaxing the rigid-L product space can be simpler and better.
