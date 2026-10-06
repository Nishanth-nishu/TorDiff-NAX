# CREMP: Conformer-rotamer ensembles of macrocyclic peptides for machine learning
- **Citation:** C. A. Grambow, H. Weir, C. N. Cunningham, T. Biancalani, K. V. Chuang. *Scientific Data* 11:859 (2024), doi:10.1038/s41597-024-03698-y. arXiv:2305.08057 (2023)
- **URL:** https://arxiv.org/abs/2305.08057 · PDF: `2023_grambow_cremp.pdf` (journal version) · text: `fulltext/2023_grambow_cremp.txt`
- **Tags:** dataset, macrocycles

## Key idea
A dataset of about 36k homodetic macrocyclic peptides (4-, 5- and 6-mers, i.e. 12-, 15- and 18-membered backbone rings, with N-methylation and D/L stereochemistry). It contains over 31 million CREST (GFN2-xTB, iMTD-GC) conformers — the macrocycle analogue of GEOM. CREST takes about 14 hours per hexapeptide.

## Relation to the three gaps
- Gap 1: **provides the test bed.** Macrocycle rings are exactly where TD's frozen RDKit local structure (ring conformations are part of L) fails.
- Gaps 2 and 3: indirectly. Ring bond-angle and torsion coupling is strong in macrocycles, so CREMP is the natural stress test for a joint torsion × bond-angle model.

## Metrics
Dataset paper; no model benchmark on GEOM.

## Reusable for FlexiTors
- An out-of-distribution evaluation set for gap 1, complementary to GEOM-XL: train on DRUGS (or fine-tune on CREMP), then test on CREMP macrocycles with RINGER's protocol.
