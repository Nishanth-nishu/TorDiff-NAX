# Energy-Guided Generative Modeling for Low-Energy Molecular Structure Discovery (EnFlow)
- **Citation:** G. Xu, X. Yi, Z. Meng, P. Zhao, Y. Bian. arXiv:2512.22597 (v1 Dec 2025, titled "Energy-Guided Flow Matching Enables Few-Step Conformer Generation and Ground-State Identification"; v2 May 2026).
- **URL:** https://arxiv.org/abs/2512.22597 · code: https://github.com/Rich-XGK/EnFlow · PDF: `2025_xu_enflow.pdf` · text: `fulltext/2025_xu_enflow.txt`
- **Tags:** competitor, flow matching plus a learned energy model, few-step sampling, ground-state identification

## Key idea
Couples a flow-matching conformer generator with an **explicitly learned energy landscape**. The energy gradient guides sampling toward low-energy regions (1–2 ODE steps possible), and the learned energy ranks generated conformers to pick the ground state. Learned energies align with GFN2-xTB rankings.

## Relation to the three gaps
- Gaps 1–2: **addresses** (Cartesian).
- Gap 3: **partially.** Energy guidance penalises strained torsion and angle combinations.

## Reported metrics (TD protocol; Table 2, DRUGS δ = 0.75; test sets of 1000 molecules)
- **EnFlow (16.6M):** COV-R 77.2/82.3, AMR-R 0.499/0.479, COV-P 70.0/76.5, AMR-P 0.607/0.541. A second step-count variant gets 78.8/84.6, 0.475/0.455, 70.7/76.9, 0.590/0.521. The step labels are garbled in the extraction (5 vs. 50 vs. 100 steps) — check the PDF.
- **2-step EnFlow:** 70.7/74.6, 0.596/0.578, 69.1/75.7, 0.623/0.575.
- **Reproducibility note:** the paper's own reproduction of ET-Flow gets 79.54/85.00, 0.470/0.444, **69.79**/75.53, 0.604/0.538, against ET-Flow's reported COV-P of 74.38.

## Reusable for FlexiTors
- An energy or confidence head for ranking and guidance. In FlexiTors it can **softly penalise bond-angle deviations** (a learned or analytic angle-bending potential) during sampling.
- Ground-state identification metrics (C-RMSD, D-MAE) as extra evaluation.
