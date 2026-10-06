# GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles
- **Citation:** O.-E. Ganea, L. Pattanaik, C. W. Coley, R. Barzilay, K. F. Jensen, W. H. Green, T. S. Jaakkola. NeurIPS 2021. arXiv:2106.07802
- **URL:** https://arxiv.org/abs/2106.07802 · PDF: `2021_ganea_geomol.pdf` · text: `fulltext/2021_ganea_geomol.txt`
- **Tags:** foundation, internal coordinates, origin of the benchmark protocol

## Key idea
A single-pass MPNN that predicts, from the 2D graph plus a stochastic seed, the local 3D structure around every non-terminal atom (bond lengths, bond angles, chirality-aware neighbour placement) and then the torsion angles. It assembles a conformer from these intrinsic coordinates. GeoMol introduced the **train/val/test splits and the 2K-conformer COV/AMR protocol that Torsional Diffusion (TD) adopts**. GeoMol itself used δ = 0.5 Å for QM9 and 1.25 Å for DRUGS (Sec. 5).

## Relation to the three gaps
- Gap 1 (rigid local structure): **partially.** GeoMol *learns* local structures (bond lengths and angles) instead of freezing RDKit's, but it works from the graph only, with no 3D context. TD argues this fails for long-range sterics (TD Sec. 4.3), and GeoMol barely beats RDKit on GEOM-XL (TD Table 6).
- Gap 2 (train/test shift in L): **does not apply.** There is no external L, and training uses ground-truth local structures directly.
- Gap 3 (torsion–angle coupling): **partially.** One network predicts torsions and local structures together, but in a single shot with no iterative steric feedback.

## Reported metrics
- Protocol origin: 2K conformers per K ground-truth conformers; δ = 0.5 Å (QM9) and 1.25 Å (DRUGS).
- Under TD's protocol (δ = 0.75 Å for DRUGS), from TD's tables:
  - **DRUGS:** COV-R 44.6/41.4, AMR-R 0.875/0.834, COV-P 43.0/36.4, AMR-P 0.928/0.841.
  - **QM9 (δ = 0.5):** 91.5/100, 0.225/0.193, 86.7/100, 0.270/0.241.
  - **XL:** AMR-R 2.47/2.39, AMR-P 3.30/3.15.

## Reusable for FlexiTors
- Differentiable construction of local neighbourhoods (bond lengths, angles and chirality) from predicted internal coordinates. Note that it handles chirality by inverting wrong centres after the fact, which TD App. F.3 criticises.
- Canonical data splits: the TD code reuses GeoMol's split files.
