# Do Deep Learning Methods Really Perform Better in Molecular Conformation Generation?
- **Citation:** G. Zhou, Z. Gao, Z. Wei, H. Zheng, G. Ke (DP Technology). arXiv:2302.07061 (v2, Mar 2023).
- **URL:** https://arxiv.org/abs/2302.07061 · PDF: `2023_zhou_dl_conformation_critique.pdf` · text: `fulltext/2023_zhou_dl_conformation_critique.txt`
- **Related critique (not downloaded):** "Infinite Physical Monkey", arXiv:2304.10494, https://arxiv.org/abs/2304.10494 (URL seen in search results only; not opened).
- **Tags:** evaluation-protocol critique

## Key idea
A parameter-free baseline, **"RDKit + Clustering"**, mixes three samplers — uniform random dihedrals, ETKDG, and ETKDG+MMFF — in a 1:1:4 ratio. It oversamples up to N_e = min(20·N_ref, 2000) energy samples and then clusters them down to the required number of conformers. On the GeoDiff/ConfGF protocol (200 test molecules; δ = 0.5 / 1.25 Å) it beats nearly all deep models on the **recall-only** metrics. Ablations show COV and MAT improve simply by sampling more diversely.

## Relation to the three gaps
- Gaps 1–3: **not addressed.** Its value is as an evaluation-protocol warning.

## Reported metrics (Table 1, recall only; GeoDiff protocol, not the TD protocol)
- QM9: COV 97.65/100, MAT 0.1902/0.1818.
- DRUGS: COV 87.93/100, MAT 0.8086/0.7838.
- For comparison, the paper's GeoDiff row on DRUGS is 88.45/97.09, 0.8651/0.8598.
- TD was excluded because it uses different splits and "is a combination of conventional methods with deep learning".

## Reusable for FlexiTors / evaluation
- Recall metrics (COV-R, MAT-R) can be gamed by pre-sampling many diverse candidates and pruning. **Always report precision (COV-P, AMR-P) alongside recall, fix the generated set at exactly 2K with no oversample-then-filter, and report energy-based checks** (see `2025_nikitin_geom_drugs_revisited.md`).
- The cheap RDKit+Clustering recipe is a strong QM9 baseline to include in FlexiTors ablations.
