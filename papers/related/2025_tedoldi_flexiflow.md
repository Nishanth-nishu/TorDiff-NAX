# FlexiFlow: decomposable flow matching for generation of flexible molecular ensemble
- **Citation:** R. Tedoldi, O. Engkvist, P. Bryant, H. Azizpour, J. P. Janet, A. Tibo (AstraZeneca / KTH). arXiv:2511.17249 (Nov 2025).
- **URL:** https://arxiv.org/abs/2511.17249 · PDF: `2025_tedoldi_flexiflow.pdf` · text: `fulltext/2025_tedoldi_flexiflow.txt`
- **Tags:** de novo molecule + multi-conformer generation (lower relevance)

## Key idea
Extends 3D de novo flow matching so that a molecule is generated **jointly with several conformations**: two sets of coordinates with a decomposable flow, keeping equivariance and permutation invariance. Evaluated on QM9 and GEOM-Drugs for de novo quality. Conformer ensembles are compared with CREST, RDKit and Adjoint Sampling on 100 molecules (300 conformers each) using COV and AMR threshold sweeps (Fig. 4; App. B.8).

## Relation to the three gaps
- Gaps 1–3: **not targeted.** All coordinates are learned, so there is no frozen L. Relevant mainly as a 2025 multi-conformer ensemble method and for its CREST-referenced evaluation.

## Metrics
No TD-protocol DRUGS table. Coverage sweeps against CREST minima in Fig. 4.

## Reusable for FlexiTors
- Joint ensemble generation (a set of conformers per molecule), as an alternative to particle guidance for diversity.
