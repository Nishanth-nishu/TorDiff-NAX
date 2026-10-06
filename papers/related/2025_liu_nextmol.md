# NExT-Mol: 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation (DMT conformer model)
- **Citation:** Z. Liu, Y. Luo, H. Huang, E. Zhang, S. Li, J. Fang, Y. Shi, X. Wang, K. Kawaguchi, T.-S. Chua. ICLR 2025. arXiv:2502.12638
- **URL:** https://arxiv.org/abs/2502.12638 · PDF: `2025_liu_nextmol.pdf` · text: `fulltext/2025_liu_nextmol.txt`
- **Tags:** competitor (conformer-prediction module), transformer diffusion, language-model features

## Key idea
Pairs a 1D molecule LLM (MoLlama, over SELFIES) with a **Diffusion Molecular Transformer (DMT)** for 3D conformer prediction. DMT is a non-equivariant, relational transformer diffusion model on coordinates, and MoLlama representations are injected to improve it.

## Relation to the three gaps
- Gap 1: **addresses** (fully learned coordinates).
- Gap 2: **addresses.**
- Gap 3: **implicit.**

## Reported metrics (TD protocol; Table 4)
The extracted rows are shifted by one label; the values below are re-aligned with the text.
- **DRUGS**, COV-R / AMR-R / COV-P / AMR-P, mean/median:
  - DMT-B (55M): 85.4/92.2, 0.401/0.375, 65.2/67.8, 0.642/0.577
  - DMT-B with PC sampling: 85.5/91.2, 0.396/0.370, 67.6/71.5, 0.623/0.546
  - DMT-L (150M): 85.8/92.3, 0.375/0.346, 67.9/72.5, 0.598/0.527
- **TD + PG reproduced:** 73.8/79.3, 0.566/0.539, 65.2/70.8, 0.680/0.615, vs. the reported 77.0/82.6, 0.543/0.520, 68.9/78.1, 0.656/0.594.
- **QM9** in Table 4b.

## Reusable for FlexiTors
- Pretrained 1D/2D molecular embeddings as extra node features for the score network (cheap to add).
- Evidence that PG results do not fully reproduce: report seeds and confidence intervals.
