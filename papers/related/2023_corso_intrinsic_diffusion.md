# Modeling Molecular Structures with Intrinsic Diffusion Models (MIT S.M. thesis)
- **Citation:** G. Corso. MIT S.M. thesis, Feb 2023. arXiv:2302.12255
- **URL:** https://arxiv.org/abs/2302.12255 · PDF: `2023_corso_intrinsic_diffusion.pdf` · text: `fulltext/2023_corso_intrinsic_diffusion.txt`
- **Tags:** framework; TD co-first author's own view of TD's limits and next steps

## Key idea
Generalises TD and DiffDock into **Intrinsic Diffusion Models (IDM)**, built from four components (Sec. 2.4):
1. **Flexibility:** pick a low-dimensional extrinsic manifold that holds most of the entropy, plus a way to project data onto it. Conformer matching is the instance of this projection, and it avoids train/test shift.
2. **Mapping:** a bijection to intrinsic coordinates whose factors are disentangled.
3. **Diffusion:** closed-form heat kernels for each factor.
4. **Score model:** an extrinsic-to-intrinsic network.

## Relation to the three gaps
- Gap 1: **partially, as a proposal.** Future directions (Sec. 5.2, p. 63) state that TD "suffers for larger and more flexible rings, especially for macrocycles". The thesis proposes adding **ring puckering coordinates, modelled as points on hyperspheres**, to the torsional manifold.
- Gap 2: treats conformer matching as the generic "project data onto the manifold" step (Sec. 2.4.1). This confirms that TD's authors see the shift as solved by projection, not as an open problem.
- Gap 3: **partially, as a proposal.** Sec. 5.2 lists "relaxing the condition that the diffusion is done exclusively on the extrinsic manifold, but, instead, using such manifold as a soft constraint or inductive bias to make the full dimensional diffusion more efficient". It also lists designing "forward diffusion processes that more closely align with physical priors".

## Metrics
Reproduces TD's DRUGS, QM9 and XL numbers; no new results.

## Reusable for FlexiTors
- The IDM checklist works as a design spec for FlexiTors: choose the extra factors (a bond-angle relaxation subspace; ring-puckering hyperspheres), check disentanglement and the bijection, derive heat kernels, and keep the extrinsic score network.
- The soft-constraint variant amounts to anisotropic diffusion with small σ on stiff coordinates.
