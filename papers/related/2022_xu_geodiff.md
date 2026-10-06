# GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation
- **Citation:** M. Xu, L. Yu, Y. Song, C. Shi, S. Ermon, J. Tang. ICLR 2022. arXiv:2203.02923
- **URL:** https://arxiv.org/abs/2203.02923 · PDF: `2022_xu_geodiff.pdf` · text: `fulltext/2022_xu_geodiff.txt`
- **Tags:** foundation, Cartesian diffusion, benchmark protocol (ConfGF/GeoDiff split)

## Key idea
A roto-translation-equivariant DDPM that runs directly on 3D atomic coordinates for 5000 steps, with an equivariant GNN (GFN) as the score network. Equivariant Markov kernels give an invariant likelihood. The model learns every degree of freedom jointly: bond lengths, angles and torsions.

## Relation to the three gaps
- Gap 1: **addresses in principle.** Nothing is frozen, but the cost is high (5000 steps) and precision is worse. Under TD's protocol, GeoDiff reaches only 24.9% COV-P on DRUGS.
- Gap 2: **does not apply** (no RDKit L).
- Gap 3: **addresses implicitly** through full Cartesian coupling, but has no explicit angle/torsion structure.

## Reported metrics and protocol (important for comparability)
- **Different protocol from TD.** Uses the split of Shi et al. 2021 (ConfGF): 40,000 training molecules × 5 conformers and **200 test molecules** (22,408 QM9 / 14,324 DRUGS test conformers). Thresholds are δ = 0.5 Å for QM9 and **δ = 1.25 Å for DRUGS**, with 2K conformers generated (Sec. 5.1–5.2). RMSD is computed after Kabsch alignment.
- **Own DRUGS results** (δ = 1.25, Table 1), GeoDiff-C: COV-R 89.13/97.88, MAT-R 0.8629/0.8529, COV-P 61.47/64.55, MAT-P 1.1712/1.1232.
- **Own QM9 results** (App. Table 5), GeoDiff-C: 90.07/93.39, 0.2090/0.1988, 52.79/50.29, 0.4448/0.4267.
- **With MMFF post-optimisation, DRUGS** (Table 2): RDKit 60.91/65.70, 1.2026/1.1252, 72.22/88.72, 1.0976/0.9539. GeoDiff+FF: 92.27/100, 0.7618/0.7340, 84.51/95.86, 0.9834/0.9221.
- **Under TD's protocol** (retrained on GeoMol splits, δ = 0.75): DRUGS 42.1/37.8, 0.835/0.809, 24.9/14.5, 1.136/1.090 (TD Table 1).

## Reusable for FlexiTors
- An equivariant Cartesian refinement could serve as an optional last-stage "local relaxation" head. GeoDiff also shows how expensive unconstrained Cartesian diffusion is, which argues for staying low-dimensional.
- **Metric-reference warning:** numbers under the GeoDiff protocol (δ = 1.25 Å, 200-molecule test set) must never be compared with numbers under the TD protocol.
