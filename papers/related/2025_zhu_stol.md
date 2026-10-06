# Chemistry-Enhanced Diffusion-Based Framework for Small-to-Large Molecular Conformation Generation (StoL)
- **Citation:** Y. Zhu, J. Zhang, J. Peng, M. Li, C. Xu, Z. Lan (South China Normal Univ.). arXiv:2511.12182 (Nov 2025).
- **URL:** https://arxiv.org/abs/2511.12182 · PDF: `2025_zhu_stol.pdf` · text: `fulltext/2025_zhu_stol.txt`
- **Tags:** large molecules via fragment assembly (GEOM-XL-like generalisation)

## Key idea
Breaks the input SMILES into overlapping fragments of 6–10 heavy atoms by cutting non-ring single bonds. A diffusion model trained only on small DFT molecules (QCDGE, 438k molecules, B3LYP) generates 3D fragment structures. Fragments are **assembled** by RMSD-matching the shared atoms, then filtered with PoseBusters-style cheminformatics checks. Training adds chemistry-enhanced losses (Sinkhorn and Gumbel-softmax matching, planarity-aware loss).

## Relation to the three gaps
- Gap 1: **partially.** Local structure comes from a DFT-trained generative model rather than RDKit, and large molecules are reached without large-molecule training data.
- Gap 2: not addressed.
- Gap 3: **partially,** through overlap-consistent assembly.

## Metrics
No GEOM COV/AMR. Evaluated on vorinostat and a new StoL25-init set (200 ChEMBL molecules with 16–25 heavy atoms) against DFT-optimised RDKit conformers (energy spread, scaffold RMSD).

## Reusable for FlexiTors
- A fragment-level local-structure model trained on high-quality small-molecule data could replace RDKit as the L-sampler for XL molecules, addressing the RDKit embedding failures reported by MCF and ET-Flow on GEOM-XL.
