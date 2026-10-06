# Flow-Matching Based Refiner for Molecular Conformer Generation
- **Citation:** X. Xu, H. Gao (Iowa State). arXiv:2510.04878 (Oct 2025).
- **URL:** https://arxiv.org/abs/2510.04878 · PDF: `2025_xu_fm_refiner.pdf` · text: `fulltext/2025_xu_fm_refiner.txt`
- **Tags:** post-hoc refinement, **tighter QM9 threshold**

## Key idea
A small (8.3M) flow-matching **refiner** starts from the outputs of an upstream generator (MCF, ET-Flow or DMT). It enters at an intermediate time, skipping the hard low-SNR phase, and fixes the residual errors. The paper analyses dynamics when the upstream sample's effective noise level mismatches the refiner's time.

## Relation to the three gaps
- Gap 1: **partially.** A generic learned local refinement — the "correct the frozen/approximate local structure" step. It could be applied to TD outputs, though the paper does not.
- Gaps 2 and 3: indirect.

## Reported metrics (TD protocol)
- **DRUGS** (δ = 0.75, Table 1), COV-R / AMR-R / COV-P / AMR-P, mean/median:
  - DMT-L + Refiner: **87.47/94.12, 0.349/0.319, 75.91/81.51, 0.497/0.446**
  - MCF-L + Refiner: 86.44/93.68, 0.368/0.330, 72.07/78.41, 0.550/0.480
  - Their own reproduced baselines: MCF-L 85.10/92.86, 0.390/0.343, 66.63/70.00, 0.623/0.546; DMT-L 85.95/91.98, 0.378/0.353, 67.97/71.97, 0.599/0.529
- **QM9** (Table 2): uses **δ = 0.05 Å** because "recent work already achieves 100% median COV at the commonly used threshold δ = 0.5 Å and a median AMR below 0.05 Å". DMT-B + Refiner: COV-R 79.50/89.44, AMR-R 0.070/0.026, COV-P 80.37/97.92, AMR-P 0.076/0.021.

## Reusable for FlexiTors
- **QM9 evaluation at δ = 0.05 Å** (or a threshold sweep) is the right way to see local-structure improvements on QM9.
- The refiner-from-intermediate-time design gives a FlexiTors variant: run the torsional stage, then a short angle-relaxation flow from t = t* — a two-stage alternative to fully joint diffusion.
