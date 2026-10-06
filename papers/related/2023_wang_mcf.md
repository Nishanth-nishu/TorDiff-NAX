# Swallowing the Bitter Pill: Simplified Scalable Conformer Generation (MCF)
- **Citation:** Y. Wang, A. A. Elhag, N. Jaitly, J. M. Susskind, M. Á. Bautista (Apple). ICML 2024 (PMLR 235). arXiv:2311.17932
- **URL:** https://arxiv.org/abs/2311.17932 · PDF: `2023_wang_mcf.pdf` · text: `fulltext/2023_wang_mcf.txt`
- **Tags:** competitor, Cartesian diffusion, scaling, GEOM-XL

## Key idea
**Molecular Conformer Fields:** a conformer is a function from graph nodes (represented by graph-Laplacian eigenvector positional encodings) to 3D coordinates. A DDPM with a non-equivariant PerceiverIO transformer runs directly on atom positions, with **no torsion, local-structure or equivariance priors**. It is scaled to 13M, 64M and 242M parameters and uses 1000 DDPM steps.

## Relation to the three gaps
- Gap 1: **addresses.** All local structure is learned. Specifically, MCF reports that **TD's checkpoint fails to generate 25 of 102 GEOM-XL molecules because RDKit cannot embed them**, and TD "cannot generate conformers in these cases" (Sec. 5.3).
- Gap 2: **addresses** (no RDKit L).
- Gap 3: **addresses implicitly.**

## Reported metrics (TD protocol; δ = 0.75 DRUGS, 0.5 QM9; App. A.2.4 formulas)
- **DRUGS** (Table 2), COV-R / AMR-R / COV-P / AMR-P, mean/median:
  - MCF-S: 79.4/87.5, 0.512/0.492, 57.4/57.6, 0.761/0.715
  - MCF-B: 84.0/91.5, 0.427/0.402, 64.0/66.2, 0.667/0.605
  - MCF-L: 84.7/92.2, 0.390/0.247, 66.8/71.3, 0.618/0.530
- **QM9** (Table 1), MCF-B: 95.0/100, 0.103/0.044, 93.7/100, 0.119/0.055. This is far below TD's AMR-R of 0.178 Å, which is consistent with gap 1. TD's 0.17 Å conformer-matching figure is not a lower bound on AMR-R, so this is not a proof of a floor (review M1).
- **GEOM-XL** (Table 3; values are AMR-R mean/med, AMR-P mean/med, # molecules):
  - All 102 molecules:
    - TD (numbers from TD paper): 2.05/1.86, 2.94/2.78
    - MCF-S: 2.22/1.97, 3.17/2.81
    - MCF-B: 2.01/1.70, 3.03/2.64
    - MCF-L: 1.97/1.60, 2.94/2.43
  - 77-molecule subset where TD succeeds:
    - TD (MCF's own evaluation): 1.93/1.86, 2.84/2.71
    - MCF-S: 2.02/1.87, 2.9/2.69
    - MCF-B: 1.71/1.61, 2.69/2.44
    - MCF-L: 1.64/1.51, 2.57/2.26
  - **Caveat:** in MCF's Table 3 the column header reads "AMR-P AMR-R", the reverse of TD Table 6. The rows labelled "GeoDiff" and "GeoMol" carry TD's RDKit and GeoMol numbers, so the labels are shifted. Read the columns as AMR-R then AMR-P, consistent with TD.
- Protocol inconsistency: App. A.2 says "we split GEOM-QM9 and GEOM-DRUGS randomly ... (80%/10%/10%)", while Sec. 5 says the splits follow Ganea et al. The processed QM9 test set has 995 molecules (noted by S23D).
- Limitations (App. A.1): compute cost (1000 steps; DDIM helps) and a low-data regime.

## Reusable for FlexiTors
- The strongest baseline showing what full-flexibility buys at scale. FlexiTors must beat MCF-S/B at similar size to justify its inductive bias.
- The XL failure mode (RDKit embedding failures) must be handled: fall back to random-coordinate embedding, or generate L.
