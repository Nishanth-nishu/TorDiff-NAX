# C-GEOM-05: ML-potential relaxation of seeds (ANI-2x / AIMNet2 / MACE-OFF) as test-time L — not for QM9

## 1. Idea (scout)
Replace MMFF in the `S1_etkdg2L_mmff` recipe by a transferable ML potential (TorchANI ANI-2x, AIMNet2 or MACE-OFF),
relax the ETKDG seeds on GPU, and feed them as L to CTRL/B1. Popular in practice and in recent conformer pipelines;
this card argues it is dominated by GFN2-xTB (C-GEOM-01) for GEOM-QM9 and should not be run.

## 2. Where it comes from (scout)
- MACE-OFF is trained to ωB97M-D3(BJ)/def2-TZVPPD DFT energies and forces (SPICE) [E-GEOM-027].
- GEOM-QM9 references are GFN2-xTB-optimised CREST conformers [E-GEOM-015, E-GEOM-016]; GFN2-xTB re-optimisation of
  GEOM references moves them by ≈ 0, while a different level (MMFF) differs from GFN2-xTB by ~0.011 Å, 1.2° and 4.9°
  (GEOM-Drugs) [E-GEOM-018, E-GEOM-019].
- Package metadata (not install-tested): `aimnet` 0.2.0 needs python ≥ 3.11 and torch ≥ 2.8; `torchani` 2.9.0 needs
  python ≥ 3.10 and torch ≥ 2.0 (2.2.4 is unpinned); `mace-torch` 0.3.16 needs torch ≥ 1.12 but pins `e3nn==0.4.4`
  [E-GEOM-036]. Our TD env is python 3.9, torch 1.13.1, e3nn 0.5.1 [E-GEOM-014].

## 3. Why it could matter here (scout)
- MMFF relaxation helped CTRL (0.178 → 0.152) but not B1 (0.236 → 0.232) [E-GEOM-007]; its residual is the pucker tail
  (20.2% of ring seeds > 10°) with bonds/angles already at λ = 0.5 level [E-GEOM-001, E-GEOM-002]. A better potential
  would only help if it moves seeds into the right pucker basin or brings bonds/angles to the λ = 0.75 level.
- The second effect needs the **reference** level, which is GFN2-xTB, not DFT [E-GEOM-015, E-GEOM-027]: a DFT-level
  ML potential converges to minima that differ systematically from GEOM's (INFERENCE by analogy with the MMFF→GFN2
  offsets in [E-GEOM-019]; DFT↔GFN2 geometry offsets for QM9 were not found in our sources: UNVERIFIED).
- The first effect (escaping a wrong basin) is the same limitation any local optimiser has (C-GEOM-01 §4).

## 4. Assumptions that may not transfer (scout)
- **Wrong target level** (above). The case for ML potentials (DFT accuracy at force-field cost) is about matching DFT;
  our metric matches GFN2-xTB geometries.
- **Cost argument does not apply at QM9 size.** GFN2-xTB is cheap enough here: GEOM ran full CREST for QM9 at 0.5
  core-h per molecule [E-GEOM-016]; a single optimisation per seed is far cheaper (exact runtime UNVERIFIED, measured in
  C-GEOM-01).
- **Installation on our stack** [E-GEOM-036, E-GEOM-014]:
  - AIMNet2 (`aimnet` 0.2.0): **does not install** in the TD env (python ≥ 3.11, torch ≥ 2.8); needs a separate env
    with a CUDA-12 torch (driver UNVERIFIED).
  - TorchANI: current 2.9.0 does not (python ≥ 3.10, torch ≥ 2.0); 2.2.4 declares no torch pin and might work with
    torch 1.13.1 (untested).
  - MACE-OFF (`mace-torch` 0.3.16): python/torch OK on paper, but `e3nn==0.4.4` conflicts with TD's e3nn 0.5.1 ⇒
    separate venv required.
  - By contrast the `xtb` binary needs no Python stack at all [E-GEOM-034, E-GEOM-035].
- Where this card could flip: GEOM-DRUGS-scale molecules where xTB per-seed cost becomes limiting, or a future switch
  of the reference to DFT geometries (brief Q4).

## 5. Minimal experiment (scout)
None recommended. If the panel wants a check anyway, the cheapest is a CPU-only L-error comparison on 100 validation
molecules: ETKDG → {MMFF, MACE-OFF-small (separate venv), GFN2-xTB} relaxation, reporting ring/acyclic RMSD vs GT and
the > 10° pucker share (no GPU, ≈ half a day of setup). Expected: MACE-OFF ≈ xTB on puckers, worse than xTB on
bonds/angles vs GT.

## 6. Scout's own call (scout)
Worth trying: NO — for GEOM-QM9 an ML potential relaxes toward a DFT minimum instead of the GFN2-xTB reference, cannot
fix wrong pucker basins any better than xTB, and costs a separate environment; GFN2-xTB (C-GEOM-01) dominates it.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
