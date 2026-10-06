# FoldingDiff: Protein structure generation via folding diffusion
- **Citation:** K. E. Wu, K. K. Yang, R. van den Berg, J. Y. Zou, A. X. Lu, A. P. Amini. arXiv:2209.15611 (v2, Nov 2022).
- **URL:** https://arxiv.org/abs/2209.15611 · PDF: `2022_wu_foldingdiff.pdf` · text: `fulltext/2022_wu_foldingdiff.txt`
- **Tags:** technique, joint bond-angle + dihedral diffusion with wrapped noise

## Key idea
Represents a protein backbone as a sequence of **six internal angles per residue: three dihedrals and three bond angles** (Fig. 1, Table 1). Runs a DDPM whose forward kernel is a **wrapped normal** on [-π, π)^{6(N-1)}, explicitly citing TD for this kernel, with bond lengths held fixed. A transformer denoises, and NeRF rebuilds the Cartesian coordinates.

## Relation to the three gaps
- Gap 1: **addresses, in a different domain.** Bond angles are generated, not frozen.
- Gap 2: **addresses.** There is no external local-structure sampler, so no train/test local-structure mismatch.
- Gap 3: **addresses.** Bond angles and torsions are noised and denoised jointly in one angular space. Caveat: the model is purely intrinsic, with no extrinsic 3D context, so it suffers from the lever-arm problem. TD App. F.4 and the Corso thesis explain why this scales badly for long chains.

## Metrics
Protein backbones (CATH); no GEOM numbers. Reports KL divergence between generated and test angle distributions (Fig. 2).

## Reusable for FlexiTors
- The simplest template for putting **bond angles in the same wrapped-diffusion framework as torsions**. For small molecules, a narrow truncated or wrapped normal centred on the RDKit value is more appropriate than a full-range angle prior.
- Noise scaled per angle type: bond angles need far smaller σ than torsions.
