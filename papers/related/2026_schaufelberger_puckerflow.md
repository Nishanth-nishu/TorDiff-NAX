# Generating Cyclic Conformers with Flow Matching in Cremer-Pople Coordinates (PuckerFlow)
- **Citation:** L. Schaufelberger*, A. Hartgers*, K. Jorner (ETH Zürich). arXiv:2601.12859 (Jan 2026).
- **URL:** https://arxiv.org/abs/2601.12859 · PDF: `2026_schaufelberger_puckerflow.pdf` · text: `fulltext/2026_schaufelberger_puckerflow.txt`
- **Tags:** ring conformations in internal coordinates; **directly composable with TD**

## Key idea
Flow matching on the **Cremer-Pople puckering space**: N−3 coordinates for an N-membered ring, as amplitude/phase pairs (q_m, φ_m). A geometry-informed prior guarantees closed rings, and a new **cyclic Fourier filter** output layer handles rings of different sizes. Bond lengths and angles for reconstruction come from a dictionary of training-set means keyed by atom and bond pattern (SI A.3). Data: the Folmsbee et al. ring-puckering set (CREST lowest-energy conformers), restricted to 5–8-membered non-aromatic monocycles without substituents (15,204 conformers, 419 rings).

## Relation to the three gaps
- Gap 1: **addresses the ring part of L**, which TD takes from RDKit (TD App. F.4 names ring flexibility as "the largest source of flexibility ... not directly accounted for"). The authors state that TD "relies on RDKit's ETKDG algorithm to sample" ring conformations. They show **PuckerFlow can be integrated into ETKDG or torsional diffusion for the exocyclic torsions** (Sec. 2.6, Fig. 6b), and propose "joint flow matching over the different degrees of freedom to simultaneously sample the cyclic and torsional degrees of freedom" as future work.
- Gap 2: **partially.** Rings are learned from CREST data, not RDKit.
- Gap 3: **not addressed.** Ring bond angles come from a fixed lookup table; coupling between exocyclic torsions and puckering is future work.

## Reported metrics (own ring benchmark; 5 random splits; 2K generated; coverage threshold 0.1 Å)
- **Puckering-displacement RMSD** (Table 1a), PuckerFlow vs. MCF: coverage precision 67.5±1.2% vs. 46.2±1.6%, recall 75.8±0.6% vs. 60.0±1.2%. Against GeoDiff, mean AMR drops from 0.24 to 0.13 Å (precision) and from 0.18 to 0.09 Å (recall).
- **All-atom RMSD** (Table 2a): mean AMR precision 0.18 vs. MCF 0.23 Å; recall 0.15 vs. 0.19 Å. RDKit reaches similar recall with worse precision.
- No GEOM COV/AMR. Limited to ring sizes 5–8; nine or more atoms excluded (fewer than 1% of the data; concavity issues in reconstruction).

## Reusable for FlexiTors
- **Cremer-Pople coordinates as an extra product-manifold factor.** The phases φ_m live on circles (torus-compatible) and the amplitudes q_m ≥ 0 are bounded. This is the concrete realisation of Corso's "ring puckering on hyperspheres" proposal.
- A plug-in ring sampler to replace RDKit's ring conformations before or while diffusing torsions.
