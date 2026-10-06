# ET-Flow: Equivariant Flow-Matching for Molecular Conformer Generation
- **Citation:** M. Hassan, N. Shenoy, J. Lee, H. Stärk, S. Thaler, D. Beaini (Mila / Valence Labs). NeurIPS 2024. arXiv:2410.22388
- **URL:** https://arxiv.org/abs/2410.22388 · NeurIPS PDF: https://proceedings.neurips.cc/paper_files/paper/2024/file/e8bd617e7dd0394ceadf37b4a7773179-Paper-Conference.pdf · PDF: `2024_hassan_etflow.pdf` · text: `fulltext/2024_hassan_etflow.txt`
- **Tags:** competitor, Cartesian flow matching, harmonic prior, small model

## Key idea
Conditional flow matching in Cartesian space, built from:
- an **equivariant transformer** (TorchMD-NET, 8.3M parameters) as the network;
- a **harmonic prior** (from HarmonicFlow);
- **Kabsch/RMSD alignment of the prior sample to the target** before interpolation, which shortens the paths;
- chirality correction, or an SO(3)-only variant that breaks reflection symmetry (via a cross-product term, App. C.1);
- optional stochastic sampling (ET-Flow-SS).

About 50 steps.

## Relation to the three gaps
- Gap 1: **addresses.** Local structure is learned; QM9 AMR-R of 0.073 Å is far below TD's 0.178 Å. TD's 0.17 Å conformer-matching figure is not a lower bound on AMR-R (review M1).
- Gap 2: **addresses** (no RDKit L).
- Gap 3: **addresses implicitly.**
- Limitations (Sec. 6): recall/diversity, the need for chirality correction, and GEOM-XL is "comparable to MCF-S and TorsionDiff".

## Reported metrics (TD protocol)
- **DRUGS** (δ = 0.75, Table 1), COV-R / AMR-R / COV-P / AMR-P, mean/median:
  - ET-Flow: 79.53/84.57, 0.452/0.419, 74.38/81.04, 0.541/0.470
  - ET-Flow-SS: 79.62/84.63, 0.439/0.406, 75.19/81.66, 0.517/0.442
  - ET-Flow-SO(3): 78.18/83.33, 0.480/0.459, 67.27/71.15, 0.637/0.567
- **QM9** (δ = 0.5, Table 2):
  - ET-Flow: 96.47/100, 0.073/0.047, 94.05/100, 0.098/0.039
  - ET-Flow-SO(3): 95.98/100, 0.076/0.030, 92.10/100, 0.110/0.047
- **GEOM-XL** (Table 8): ET-Flow on all 102 molecules gets AMR-R 2.31/1.93 and AMR-P 3.31/2.84; on 75 molecules, 2.00/1.80 and 2.96/2.63. The authors "encountered 27 failed cases for generation likely due to RDKit failures" when running TD's checkpoint. The table copies MCF's mislabelled baseline rows, so the "GeoDiff" row is really TD's RDKit row.
- **Ablations** (Table 7, reduced training):
  - full: COV-R 75.37, AMR-R 0.557, COV-P 58.90, AMR-P 0.742
  - O(3) without chirality correction: 72.74, 0.576, 54.84, 0.794
  - without alignment: 68.67, 0.622, 47.09, 0.870
  - Gaussian prior instead of harmonic: 66.53, 0.640, 44.41, 0.903
- **OOD** (Table 9): scaffold split on DRUGS 76.06/80.65, 0.644/0.545, 67.83/74.19, 0.511/0.473; on QM9 95.00/100, 0.083/0.029, 90.25/100, 0.124/0.053. Trained on DRUGS and tested on QM9: 86.68/100, 0.218/0.160, 68.69/75.30, 0.369/0.317. (The Table 9 extraction is partly garbled — check the row mapping against the PDF.)
- Reproducibility: EnFlow's reproduction gives COV-P 69.79 vs. the reported 74.38 (see `2025_xu_enflow.md`).

## Reusable for FlexiTors
- **Prior-to-data alignment** (the discrete analogue of TD's conformer matching) and informed priors. For the angle factor, use a prior centred on the RDKit or ideal angle values.
- Its ablation shows prior and alignment choices matter more than architecture. FlexiTors ablations should include an "RDKit-centred angle prior vs. uniform/wide prior" arm.
