# Torsional-GFN: a conditional conformation generator for small molecules
- **Citation:** L. N. Ezzine*, A. Volokhova*, P. Gaiński, L. Scimeca, E. Bengio, P. Tossou, Y. Bengio, A. Hernandez-Garcia. Workshop on Generative AI for Biology (ICML 2025 workshop, per the PDF header). arXiv:2507.11759
- **URL:** https://arxiv.org/abs/2507.11759 · PDF: `2025_ezzine_torsional_gfn.pdf` · text: `fulltext/2025_ezzine_torsional_gfn.txt`
- **Tags:** Boltzmann sampling on the torus conditioned on L; generalisation to unseen local structures

## Key idea
A continuous **GFlowNet over the hypertorus of rotatable torsions**, conditioned on the molecular graph G *and its local structure L* (bond lengths and angles). It is trained only from a reward exp(−E/kT), with no data. It decomposes p(c) = p(L | G) · p(τ | G, L) (Sec. 2), obtains L from MD rather than RDKit, and tests **zero-shot generalisation to unseen L** (bond lengths and angles from MD).

## Relation to the three gaps
- Gap 1: **partially.** It argues explicitly against the "local structure is constant" assumption (Sec. 2: "previous works ... rely on the assumption that the local structure of stable conformations is constant ... or they sample local structures from RDKit"). L is still an input, not generated.
- Gap 2: **partially.** It evaluates generalisation to shifted L (MD bond lengths and angles), shows the torsion-mode landscape moves with L, and that the model tracks the shift (Fig. 1b).
- Gap 3: **partially — direct evidence.** It shows p(τ | L) depends on L: modes of the torsional energy landscape shift when bond lengths and angles change. It lists "including the generation of the local structure into the GFlowNet model" as future work.

## Metrics
A handful of small molecules (6 training, 8 for evaluation) against MD Boltzmann samples; no GEOM COV/AMR.

## Reusable for FlexiTors
- Energy/reward-based training on the torus conditioned on L is a route to a FlexiTors Boltzmann generator: TD's App. F.4 limitation 5 asks for a model with exact p(L).
- An evaluation idea for gap 3: **perturb L (MD or xTB angle samples) and measure how torsion marginals shift**, comparing TD's frozen-L score with the FlexiTors joint model.
