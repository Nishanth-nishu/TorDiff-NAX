# C-ROBUST-02: Blind continuous L interpolation (S4 without the λ input)

## 1. Idea (scout)
Train exactly like S4, with the same paired cache, the same pair_ok filter and the same λ ~ U[0,1] blend of
RDKit-matched and true L, but give the network no λ input. The question is whether the model must be told the L
quality. At deployment, the quality of a new L source (MMFF, xTB, a learned refiner) is unknown and varies from molecule
to molecule. A blind model adapts per molecule; S4 needs one λ per source, picked on validation molecules.

## 2. Where it comes from (scout)
- Cascaded diffusion amortises the model over the augmentation strength and picks the strength after training
  [E-ROBUST-027]. S4 is built on this.
- Sun et al. (ICML 2025): without noise-level conditioning, most denoising generative models degrade gracefully, and
  some improve [E-ROBUST-037]. The level can be inferred from the input because p(t|z) is concentrated, and that
  concentration depends on data dimensionality [E-ROBUST-038]. Conditioning on a level predicted by a separate
  pre-trained network gave results similar to no conditioning [E-ROBUST-039].

## 3. Why it could matter here (scout)
- Our blind discrete model already infers the regime from L. S3 has no level input and still scores 0.182 on RDKit L
  and 0.033 on true L (1 seed) [E-ROBUST-004].
- S4's λ is one CLI value for every node of every molecule [E-ROBUST-017]. RDKit's ring error is very uneven across
  molecules: 9% of seeds with smallest ring 3 have endocyclic-dihedral error > 10°, against 48-92% for smallest rings
  of 4 to ≥ 7 atoms [E-ROBUST-007]. No single λ describes that L.
- The intermediate regime is where the models differ: B1 overtakes CTRL only at λ ≈ 0.75 [E-ROBUST-005].
- S3 vs S4 is confounded. They differ in the continuum, in the λ input, and in the training conformers (only S4 drops
  pair_ok failures, about 8%) [E-ROBUST-058]. The blind arm shares S4's data path [E-ROBUST-014], so blind vs S4
  isolates the λ input alone.
- It needs no code: S4 conditioning comes from `--lambda_embed_dim 32`, and the default 0 gives the unconditioned
  architecture [E-ROBUST-018].

## 4. Assumptions that may not transfer (scout)
- Sun et al.'s level is the noise of the variable being generated (images). Our λ is the quality of the L part of the
  input, while TD still receives the torsion σ. The two are analogous but not the same [E-ROBUST-037].
- A QM9 molecule (about 18 atoms) is low-dimensional, and the concentration result is dimension-dependent
  [E-ROBUST-038]. p(λ|L) may be broad, so the blind model may average at mid λ and lose to S4 there.
- λ blends are Cartesian interpolations between two geometries, not a real L source. Neither S4 nor the blind arm is
  guaranteed to transfer to MMFF or learned L.
- The round-2 S4 results are pending (about 10 Oct). If S4 is no better than S3 anywhere, this arm answers a question
  nobody needs.

## 5. Minimal experiment (scout)
- **Gate (pre-declared).** Run only if round-2 S4 (3 seeds; λ picked on validation molecules, never on test) beats S3
  by ≥ 0.005 Å AMR-R on MMFF L or on λ = 0.5, or comes closer to B1 at λ = 1 than S3 by ≥ 0.005 Å.
- **Arm.** S4-blind: `paired | --l_interp --log_timing --fail_on_nan`, 100 epochs, 2 seeds.
- **Controls.** S4 (3 seeds, round 2) with λ = 0 on RDKit L, λ = 1 on GT L, and λ picked on validation for MMFF; S3
  (2 seeds). For like-for-like data, all arms are re-scored on S4's molecule subset, as round 2 already does for S4
  (user ruling, round2/DECISION.md) [E-ROBUST-058].
- **Test conditions.** Primary (non-oracle): RDKit, MMFF. Secondary (ORACLE): λ 0/0.25/0.5/0.75/1, A5ring, A5acyc.
- **Primary metric and direction.** AMR-R; blind within ±0.005 Å of S4 at RDKit and true L (equivalence), and ≤ S4 on
  MMFF. If blind is worse than S4 by > 0.005 Å at mid λ, the λ input is needed, and Q1 must also deliver a quality
  estimate.
- **Cost.** 2 × 12-16 GPU-h training plus 2 × 9 × 0.25 GPU-h evaluation, about 29-37 GPU-h. No code change.

## 6. Scout's own call (scout)
Worth trying: MAYBE. It is cheap, has no code risk, and gives a clean answer to a deployment question (does a new L
source need a quality estimate?). Its value depends on S4 beating S3, which is unknown until the round-2 panels land.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
