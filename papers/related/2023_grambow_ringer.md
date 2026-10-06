# RINGER: Accurate and Efficient Structural Ensemble Generation of Macrocyclic Peptides using Internal Coordinate Diffusion
- **Citation:** C. A. Grambow, H. Weir, N. L. Diamant, G. Scalia, T. Biancalani, K. V. Chuang (Genentech). arXiv:2305.19800 (v2, Aug 2024). An earlier workshop version is titled "RINGER: Rapid Conformer Generation for Macrocycles with Sequence-Conditioned Internal Coordinate Diffusion".
- **URL:** https://arxiv.org/abs/2305.19800 · PDF: `2023_grambow_ringer.pdf` · text: `fulltext/2023_grambow_ringer.txt`
- **Tags:** macrocycles, joint bond-angle + torsion diffusion, direct evidence about TD on macrocycles

## Key idea
Macrocycle conformers are represented in **redundant internal coordinates**: bond distances D (fixed), **bond angles θ and torsions T (both diffused)**. A DDPM with a wrapped-normal prior learns p(θ, T | G; D) using a BERT-style transformer with cyclic relative positional encodings. Cartesians are recovered by solving a small optimisation that enforces ring closure (Eq. 3). Side chains are handled too. Trained and evaluated on CREMP (36k macrocyclic peptides, CREST/GFN2-xTB ensembles).

## Relation to the three gaps
- Gap 1: **addresses for macrocycles.** Ring (backbone) bond angles and torsions are generated rather than taken from RDKit.
- Gap 2: **addresses.** No external local-structure sampler.
- Gap 3: **addresses.** Angles and torsions are denoised jointly ("highly-coupled distributions over internal coordinates").
- **Direct evidence about TD** (Sec. "Structural Analysis", p. ~6; baselines in the Supplement):
  - TD was retrained on CREMP but "relies on RDKit for backbone sampling, and hence produces an identical [Ramachandran] plot" to RDKit.
  - The authors had to replace TD's default RDKit seeding with ETKDGv3 macrocycle settings (useRandomCoords, MMFF), which "very significantly boosted performance of TorDiff". The default seeding "generates low-quality conformers for macrocycles and leads to very poor performance".
  - For training, they seeded local structures from GT xTB conformers followed by MMFF optimisation, rather than TD's conformer matching.

## Reported metrics
- CREMP only, with no GEOM numbers. Coverage thresholds: 0.75 Å all-atom RMSD, 0.1 Å ring-only rRMSD, 0.05 ring-only TFD.
- Recall, precision and F1 of COV and MAT are in Fig. 3 and Table S.7. Text: RINGER "achieves excellent performance across all metrics ... with the best F1-score". The extracted table's column alignment is garbled, so check the PDF before quoting individual numbers.
- After xTB re-optimisation (Table S.10), RINGER still performs best, and its samples need smaller adjustments to reach local minima.

## Reusable for FlexiTors
- A redundant internal-coordinate representation with a ring-closure solve at the end: the path to macrocycles and GEOM-XL cyclic systems.
- Ring-only metrics (rRMSD, rTFD) to report local-structure and ring accuracy separately from global RMSD.
- Practical: when benchmarking TD on large rings, seed with ETKDGv3 macrocycle parameters, or the baseline is unfairly weak.
