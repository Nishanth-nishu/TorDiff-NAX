# A standard transformer and attention with linear biases for molecular conformer generation (S23D)
- **Citation:** V. Gurev, T. Rumbell (IBM Research). arXiv:2506.19834 (Jun 2025).
- **URL:** https://arxiv.org/abs/2506.19834 · PDF: `2025_gurev_s23d.pdf` · text: `fulltext/2025_gurev_s23d.txt`
- **Tags:** competitor, non-equivariant transformer diffusion; **QM9 saturation critique**

## Key idea
"SMILES to 3D": a standard transformer diffusion model with **attention biases linear in shortest-path graph distance** (ALiBi-style) instead of Laplacian eigenvector positional encodings. It uses a two-stage training protocol (heavy atoms first, then fine-tuning with hydrogens), and comes in S (8.6M) and B (24.8M) sizes.

## Relation to the three gaps
- Gaps 1–2: **addresses** (all coordinates learned).
- Gap 3: implicit.

## Reported metrics (TD protocol)
- **DRUGS** (δ = 0.75, Table 1), COV-R / AMR-R / COV-P / AMR-P, mean/median:
  - S23D-B (M) with chirality correction: **87.0/94.0, 0.380/0.356, 69.7/75.0, 0.599/0.532**
  - S23D-B (S) without chirality correction: 84.5/91.7, 0.420/0.387, 62.5/63.4, 0.690/0.626
  - MCF-B for comparison: 84.0/91.5, 0.427/0.402, 64.0/66.8, 0.667/0.605
- **QM9** (δ = 0.5, Table 5), S23D-B (S) with chirality: 96.0/100, 0.090/0.047, 93.8/100, 0.111/0.059.
- **XL** (Table 6), S23D-B: AMR-R 2.07/1.80, AMR-P 3.22/2.83 — comparable to MCF-B.
- **Protocol notes:**
  - Table 1 includes a column "KG" for the number of training conformers per molecule (10, 20 or 30), which varies across papers.
  - MCF's processed QM9 test set has **995 molecules, not 1000**.
  - Hydrogens were added with RDKit after heavy-atom generation in stage 1.

## Protocol critique (App. B.2.1, p. 16)
"Mean AMR values across these models are < 0.1, which is around 10% of the typical bond length ... benchmark results on the QM9 metric are likely to be saturated ... the QM9 benchmark may no longer be appropriate for testing SOTA MCG models."

## Reusable for FlexiTors
- A shortest-path ALiBi bias is a cheap graph-aware attention bias if FlexiTors uses a transformer trunk.
- **Implication for QM9 ablations:** at δ = 0.5 Å, QM9 COV cannot separate methods. Use AMR and tight thresholds (e.g. δ = 0.05–0.1 Å, as the FM-refiner paper does), plus local-geometry metrics.
