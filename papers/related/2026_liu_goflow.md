# Geometric Flow Matching for Molecular Conformation Generation via Manifold Decomposition (GO-Flow)
- **Citation:** Y. Liu, Y. Zhou, W. Fan (HK PolyU). arXiv:2605.25577 (May 2026).
- **URL:** https://arxiv.org/abs/2605.25577 · PDF: `2026_liu_goflow.pdf` · text: `fulltext/2026_liu_goflow.txt`
- **Tags:** **closest existing work to FlexiTors**: flows on internal coordinates (bond lengths, angles, torsions) × SO(3) × R^3

## Key idea
Decomposes the conformer into three coupled manifolds:
1. translation (centre of mass; linear optimal transport);
2. rotation (unit quaternions; SLERP geodesics on SO(3), with velocity in so(3));
3. **conformation = internal coordinates z = (bond lengths, bond angles, periodic torsions)**, with **entropic optimal-transport** (Sinkhorn) couplings and a barycentric target velocity (Sec. 3.2).

Internal-coordinate velocities are mapped to Cartesian velocities by **explicit Jacobians of bond length, bond angle and dihedral** (Sec. 3.3; App. C). Training is a three-stage curriculum: per-manifold flow matching, then cross-space coupling with a Cartesian consistency loss, then optional likelihood (ODE) fine-tuning with Hutchinson traces (Alg. 1). 50 steps.

## Relation to the three gaps
- Gap 1: **addresses.** Bond lengths and angles are generated, with the stiff-versus-flexible hierarchy stated as motivation.
- Gap 2: **addresses.** No RDKit L, hence no conformer matching.
- Gap 3: **addresses.** Torsions and bond angles live in one conformation manifold, with a coupling stage.
- Ablation (Table 3): removing the conformation manifold ("w/o C", effectively Cartesian flow) drops DRUGS COV-R from 94.82 to 90.26 and MAT-R from 0.7971 to 0.8512 Å; QM9 COV-R from 90.08 to 88.06 and MAT-R from 0.2086 to 0.2159.

## Reported metrics — **GeoDiff/ConfGF protocol, NOT the TD protocol**
The protocol uses the ConfGF split (40k training molecules × 5 conformers, 200 test molecules; App. D) with **δ = 1.25 Å for DRUGS** and 0.5 Å for QM9.
- **DRUGS** (Table 1): **GO-Flow COV-R 94.82/99.26, MAT-R 0.7971/0.7924, COV-P 70.13/72.49, MAT-P 1.1068/1.0001.** The text also quotes, under the same protocol, ET-Flow at COV-R 93.12 and AvgFlow at COV-P 68.56.
- **QM9** (Table 2): COV-R 90.08/94.57, MAT-R 0.2086/0.1845, COV-P 69.42/70.91, MAT-P 0.3587.
- Properties computed with PSI4 (Fig. 3): E MAE 0.1642 eV.
- **Cannot be compared with TD's Table 1** (δ = 0.75 Å, 1000-molecule GeoMol split).
- The table extraction is garbled for baseline rows; only GO-Flow's own numbers and the numbers quoted in the text are reliable here.

## Reusable for FlexiTors
- **Jacobian-based composition** of internal-coordinate velocities into Cartesian updates. This is the practical machinery for letting the extrinsic score network output bond-angle "scores" and applying them in 3D, analogous to TD's torsion updates (Prop. 1).
- A staged curriculum (train per-factor, then couple) to handle the different gradient scales of torsions and angles.
- Main risk to FlexiTors's novelty: FlexiTors must differentiate itself — for example by keeping the RDKit-anchored low-dimensional relaxation subspace, the TD protocol, the likelihood / Boltzmann generator, and rings via Cremer-Pople.
