# Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow (AvgFlow)
- **Citation:** Z. Cao, M. Geiger, A. dos Santos Costa, D. Reidenbach, K. Kreis, T. Geffner, F. Pellegrini, G. Zhou, E. Kucukbenli (NVIDIA). ICML 2025. arXiv:2507.09785
- **URL:** https://arxiv.org/abs/2507.09785 · PDF: `2025_cao_avgflow.pdf` · text: `fulltext/2025_cao_avgflow.txt`
- **Tags:** competitor, Cartesian flow matching, few-step/one-step sampling

## Key idea
Two ideas:
1. **SO(3)-Averaged Flow:** the training target is the conditional flow averaged over all rotations of the data sample, rather than a single aligned pair. This converges faster than alignment-based ET-Flow-style training.
2. **Reflow and distillation**, for 2-step and 1-step generation.

Backbones are NequIP (4.7M) and DiT (52M / 64M).

## Relation to the three gaps
- Gaps 1–2: **addresses** (learned Cartesian coordinates).
- Gap 3: implicit.
- Notes that one-step AvgFlowDiT-D "surpasses 20 steps of Tor. Diff. ... despite Tor. Diff. starting generation with RDKit-generated conformers".

## Reported metrics (TD protocol)
- **QM9** (δ = 0.5, Table 1):
  - AvgFlowDiT (52M): 96.0/100, 0.082/0.030, 95.0/100, 0.088/0.039
  - AvgFlowNequIP: 96.4/100, 0.089/0.042, 92.8/100, 0.132/0.084
  - Two-step NequIP-R: 95.9/100, 0.151/0.104, 87.7/100, 0.236/0.207
  - One-step NequIP-D: 95.1/100, 0.220/0.195, 84.8/100, 0.304/0.283
- **DRUGS** (δ = 0.75, Table 2):
  - AvgFlowNequIP (about 102 adaptive steps): 76.8/83.6, 0.523/0.511, 60.6/63.5, 0.706/0.670
  - AvgFlowDiT (100 steps): 82.0/86.7, 0.428/0.401, 72.9/78.4, 0.566/0.506
  - AvgFlowDiT-L (64M): 82.0/87.3, 0.409/0.381, 75.7/81.9, 0.516/0.456
- One- and two-step rows are in Table 2's lower sections.

## Reusable for FlexiTors
- Reflow and distillation can make a torus × angle flow a 1–2 step sampler. Straightening is easier in low-dimensional internal coordinates.
- Averaging over a symmetry group instead of aligning — the analogue for FlexiTors is averaging over graph automorphisms or torsion-definition symmetries.
