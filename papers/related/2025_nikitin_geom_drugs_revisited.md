# GEOM-Drugs Revisited: Toward More Chemically Accurate Benchmarks for 3D Molecule Generation
- **Citation:** F. Nikitin, I. Dunn, D. R. Koes, O. Isayev (CMU / Pitt). *Digital Discovery* 4(11):3282–3291 (2025). arXiv:2505.00169
- **URL:** https://arxiv.org/abs/2505.00169 · code: https://github.com/isayevlab/geom-drugs-3dgen-evaluation · PDF: `2025_nikitin_geom_drugs_revisited.pdf` · text: `fulltext/2025_nikitin_geom_drugs_revisited.txt`
- **Tags:** evaluation-protocol critique, local-geometry metrics

## Key idea
Audits GEOM-Drugs evaluation, mostly for de novo 3D generation. It finds:
- wrong valency tables and bond-order bugs;
- GEOM molecules fractured by failed GFN2-xTB optimisation;
- reliance on MMFF, which is inconsistent with GEOM's GFN2-xTB geometries — state-of-the-art models now beat MMFF at matching xTB.

It proposes a refined split that drops fractured molecules, and a **GFN2-xTB-based geometry and energy benchmark**. Each generated structure is relaxed with GFN2-xTB, and the paper reports the relaxation energy E_relax plus the mean |Δ bond length|, |Δ bond angle| and |Δ torsion| between the initial and relaxed geometry.

## Relation to the three gaps
- Gap 1: **provides the right metric** for local-structure quality. RMSD-based COV/AMR are dominated by torsions; Δ-bond-angle and E_relax directly measure whether a model's local structure is physically right.
- Gap 3: E_relax and Δ-angle-after-relaxation detect strained, uncoupled torsion/angle combinations.
- Gap 2: not addressed.

## Metrics
De novo models only (EQGAT, Megalodon, SemlaFlow, FlowMol2); no conformer COV/AMR.

## Reusable for FlexiTors
- **Add GFN2-xTB E_relax, Δ-bond-angle and Δ-bond-length metrics** alongside COV/AMR in every FlexiTors ablation. They are more sensitive to the bond-angle factor than RMSD.
- Use GFN2-xTB rather than MMFF as the physics reference, because GEOM is xTB-optimised.
- Apply the corrected GEOM split and molecule filtering, or at least check for fractured molecules.
