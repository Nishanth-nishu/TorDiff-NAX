# Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models
- **Citation:** G. Corso, Y. Xu, V. De Bortoli, R. Barzilay, T. Jaakkola. ICLR 2024. arXiv:2310.13102
- **URL:** https://arxiv.org/abs/2310.13102 · OpenReview: https://openreview.net/forum?id=KqbCvIFBY7 · PDF: `2023_corso_particle_guidance.pdf` · text: `fulltext/2023_corso_particle_guidance.txt`
- **Tags:** TD successor at inference time, diversity, sampling

## Key idea
Samples the 2K conformers jointly rather than i.i.d. A time-dependent repulsive potential between particles is added to the reverse SDE. For TD, the kernel is an RBF over **torsion-angle differences** (wrapped to (−π, π]), made **permutation-invariant** by taking the minimum over graph automorphisms (at most 32 subsampled) (Sec. 6.2; App. D). No retraining is needed.

## Relation to the three gaps
- Gaps 1–3: **not addressed.** Local structure is still RDKit's. It is orthogonal and composable, so FlexiTors can use it at inference.

## Reported metrics
GEOM-DRUGS, TD protocol, δ = 0.75 Å (Table 1; App. Table 3):
- **TD + invariant PG:** COV-R 77.0/82.6, AMR-R 0.543/0.520, COV-P 68.9/78.1, AMR-P 0.656/0.594 (median AMR: −8% recall, −19% precision vs. TD).
- **TD with low-temperature sampling**, tuned for recall: 73.3/77.7, 0.570/0.551, 66.4/73.8, 0.671/0.613.
- **TD + non-invariant PG:** 75.8/81.5, 0.542/0.520, 66.2/72.4, 0.668/0.607.
- Reproduced by NExT-Mol (Table 4a) as **73.8/79.3, 0.566/0.539, 65.2/70.8, 0.680/0.615**, lower than reported.
- The local repo's README shows `--pg_*` flags for this.

## Reusable for FlexiTors
- Drop-in inference-time diversity for the torsional factor. For the bond-angle factor, repulsion is probably unnecessary: that factor should be low-entropy.
- Hyperparameters in `torsional-diffusion/README.md` (pg_weight, pg_repulsive_weight, kernel size, Langevin weight).
