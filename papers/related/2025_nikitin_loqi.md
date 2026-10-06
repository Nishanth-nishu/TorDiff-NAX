# LoQI: Scalable Low-Energy Molecular Conformer Generation with Quantum Mechanical Accuracy (+ ChEMBL3D dataset)
- **Citation:** F. Nikitin, D. M. Anstine, R. Zubatyuk, S. G. Paliwal, O. Isayev (CMU / MSU / NVIDIA). ChemRxiv 2025, doi:10.26434/chemrxiv-2025-k4h7v (v2 dated June 2026; an OpenReview entry also exists).
- **URLs:** https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-k4h7v · code and abstract (opened and verified): https://github.com/isayevlab/LoQI
- **PDF: not stored.** ChemRxiv returned HTTP 403 and OpenReview returned a bot-verification page to automated download, so no PDF was fetched. This note is based on the verified abstract in the GitHub README and search-result snippets. Download it manually if needed.
- **Tags:** conformer foundation-model direction, stereochemistry-aware, QM-quality data, macrocycles

## Key idea
Two parts:
- **ChEMBL3D:** over 250 million conformers for 1.8 million drug-like molecules, optimised with the AIMNet2 neural potential (near-DFT accuracy, implicit solvent), covering protonation states and stereoisomers.
- **LoQI:** an all-atom, **stereochemistry-aware** diffusion model built on the Megalodon architecture, with graph augmentation for R/S and E/Z.

Claims "up to tenfold improvement in energy accuracy" over traditional approaches, efficient low-energy conformer search on **macrocycles** and flexible molecules, and validation against crystal structures. Search snippets report 0.956 R/S and 0.992 E/Z stereo accuracy.

## Relation to the three gaps
- Gap 1: **addresses.** Fully learned geometry, trained on QM-quality local structures, and tested on macrocycles.
- Gap 2: **addresses** (no RDKit L).
- Gap 3: implicit.
- Also tackles **E/Z isomerism**, which TD explicitly does not handle (TD App. F.3).

## Metrics
Not verified from the full text (PDF unavailable). It uses its own ChEMBL3D benchmarks, not the TD GEOM protocol.

## Reusable for FlexiTors
- ChEMBL3D as a higher-quality (AIMNet2) source of local structures. A FlexiTors local-structure factor could be pretrained on it.
- Stereo-aware graph augmentation to fix TD's E/Z blind spot, where double bonds are treated as rotatable.
