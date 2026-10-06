# Equivariant Blurring Diffusion for Hierarchical Molecular Conformer Generation (EBD)
- **Citation:** J. Park, Y. Shen (Texas A&M). NeurIPS 2024. arXiv:2410.20255
- **URL:** https://arxiv.org/abs/2410.20255 · PDF: `2024_park_equivariant_blurring_diffusion.pdf` · text: `fulltext/2024_park_equivariant_blurring_diffusion.txt`
- **Tags:** coarse-to-fine; an RDKit prior that the model is allowed to *correct*

## Key idea
Generation happens in two stages:
1. A coarse-grained **fragment-level** 3D structure, whose fragment coordinates come from RDKit.
2. A **blurring diffusion**: the forward process blurs atoms toward their fragment centroids; the reverse process, an equivariant network, restores atomic detail **while allowing the fragment structure to be adjusted at the same time**.

It uses T = 50 steps against GeoDiff's 5000. Sec. 5 / Fig. 5 shows the model corrects poor coarse-grained priors, and corrects them more when the prior is worse.

## Relation to the three gaps
- Gap 1: **addresses.** A cheminformatics prior is used as initialisation, but it is not frozen: this is "correctable RDKit structure", the soft version of TD's hard constraint.
- Gap 2: **partially.** Training starts from the RDKit fragment prior plus a learned correction, so there is no conformer matching.
- Gap 3: **addresses implicitly** (all-atom refinement).

## Reported metrics (GeoDiff protocol: δ = 1.25 Å DRUGS, 0.5 Å QM9; COV defined with ≤ δ; 200-molecule test sets)
- DRUGS (Table 3):
  - RDKit: COV-R 45.74/31.75, MAT-R 1.5376/1.4004, COV-P 54.78/59.48, MAT-P 1.3341/1.1996
  - GeoDiff (T = 5000): 89.40/96.86, 0.8571/0.8495, 61.28/65.00, 1.1642/1.1272
  - **EBD (T = 50): 92.60/98.73, 0.8216/0.8279, 66.24/68.39, 1.1237/1.0916**
- QM9 in App. Table 8.
- **Not comparable to TD-protocol numbers.**

## Reusable for FlexiTors
- "Prior + learned correction" instead of "prior frozen": start from RDKit L and learn a *small, bounded* correction. This is exactly the FlexiTors bond-angle relaxation factor.
- A blurring or subspace schedule: resolve torsions early, local detail late. See also subspace diffusion, cited by TD Sec. 2.
