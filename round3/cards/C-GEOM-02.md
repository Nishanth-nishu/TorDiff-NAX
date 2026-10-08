# C-GEOM-02: Training-set ring-template L (OMEGA-style pucker library from GEOM-QM9 training conformers)

## 1. Idea (scout)
Build a ring-conformation library from the **training split's** GEOM conformers: key = ring-system component (ring
atoms and ring bonds with element/bond types; optionally plus first-shell substituent types), value = its distinct
GT ring geometries (puckers) with frequencies. At test time, impose a sampled template pucker on each ETKDG seed's ring
system (constrained re-embedding of ring atoms, or align-and-reattach substituents), fall back to ETKDG when there is no
hit, and optionally xTB-relax the result (C-GEOM-01). Non-oracle: only training conformers are used.

## 2. Where it comes from (scout)
- OMEGA builds molecules from a fragment library, takes ring conformations from that library, detaches substituents,
  re-attaches them to each ring conformation, and enumerates every combination of ring conformations
  [E-GEOM-038, E-GEOM-044].
- On GEOM-QM9, OMEGA's recall AMR median is 0.126 Å vs TD's 0.147 and RDKit's 0.199, and TD attributes this to
  OMEGA's better local structures for small molecules [E-GEOM-025].
- TD itself states that ring conformations are delegated to the L sampler and that this works less well for puckered
  and fused rings [E-GEOM-024].
- Ring conformations fall into relatively few canonical clusters, growing slowly with ring size [E-GEOM-042]; a
  knowledge-based pucker sampler reproduced CREST/GFN2 lowest-energy ring conformers at 0.09 Å RMSD, with the residual
  in bond lengths/angles, which local optimisation would fix [E-GEOM-043].
- RDKit can constrain chosen atoms to given coordinates during embedding (`coordMap`) [E-GEOM-039].

## 3. Why it could matter here (scout)
- Revised after D-201/D-207: on the like-for-like λ = 0 base (955 molecules), true ring geometry alone takes CTRL from
  0.197 to 0.119 and B1 from 0.261 to 0.188 (both gain ≈ 0.075 Å; B1 stays above CTRL) [E-GEOM-013], and the gain sits
  in molecules with few rotors: rigid 0.184 → 0.043, one rotor 0.181 → 0.104 (CTRL) [E-GEOM-006]. A template carries
  GT-level ring geometry (training GT is the same GFN2 level), so on a hit it approximates "true ring geometry from
  another molecule" (INFERENCE).
- The ETKDG pucker tail is large: 28.6% of ring seeds > 10° ring-dihedral RMSD (49.6% excluding the all-3-ring
  molecules, whose ring dihedral is identically 0); 48–92% for molecules whose smallest ring has 4–7 atoms
  [E-GEOM-002, E-GEOM-003]; 22% of the bad seeds are in fused/bridged/spiro systems, which a ring-system-level
  template handles as one unit (unlike monocyclic pucker models) [E-GEOM-011].
- Coverage proxy (key = ring atoms + ring bonds; isomeric / stereo-free): 534 / 421 distinct ring-system components
  in the 885 test ring molecules; even within the 1000-molecule test set, 50.8% / 66.3% of ring molecules have all
  components in another test molecule (45.2% / 58.6% among the 345 molecules with an ETKDG seed > 10°)
  [E-GEOM-012]. The training split is
  ~107× larger, so training coverage should be far higher (INFERENCE, UNVERIFIED).
- Pairs with C-GEOM-01: template fixes the basin, xTB fixes bonds/angles (the residual named in [E-GEOM-043]); that
  combination is the non-learned route to the λ = 0.75 spec (table in C-GEOM-01 §3).

## 4. Assumptions that may not transfer (scout)
- **Substituent effects.** Puckers and ring angles depend on substituents (axial/equatorial preference, exocyclic
  C=O, heteroatoms). A substituent-free key may transplant the wrong preference; a key with first-shell substituents
  lowers coverage. Choose the key level on validation molecules, not test.
- **Coverage is unmeasured** against the real training split [E-GEOM-012]; if it is low for the bad-pucker molecules
  the card degrades to ETKDG.
- **Frequencies are not Boltzmann weights.** Template frequencies reflect how many training molecules/conformers carry a
  pucker, not the test molecule's ensemble; TD will sample puckers in those proportions.
- **Implementation risk.** `coordMap`-constrained embedding can fail or distort substituents for strained cages
  (3-/4-rings, bridged bicycles common in QM9); the align-and-reattach route needs care at ring fusion atoms and
  invertible nitrogens [E-GEOM-038]. Graph/stereo checks as in C-GEOM-01 [E-GEOM-047].
- OMEGA's QM9 advantage [E-GEOM-025] is a whole-tool comparison; that its templates are the cause is an inference.

## 5. Minimal experiment (scout)
- **Step 1, library (CPU):** from the training pickles only (TD split; assert no val/test SMILES), extract ring-system
  components and their GT ring coordinates; cluster per key by ring-atom RMSD after alignment (e.g. < 0.1 Å) and keep
  cluster representatives with counts. Report coverage of test ring molecules overall and among the 345 molecules with
  an ETKDG seed > 10° [E-GEOM-012].
- **Step 2, seeds (CPU):** for the `L_etkdg2L` embeddings, impose a frequency-sampled template per ring system
  (two key levels: ring-only, ring + first shell) → `L_tmpl`, and `L_tmpl_xtb` (if C-GEOM-01's xtb install passes).
  Fallback to the unmodified seed on no hit or failure (count them).
- **Step 3, CPU gate (pre-declared; revised after D-205):** the C-GEOM-01 gate design: all metrics scored on the seeds
  directly (no model) on one common molecule set; rigid AMR-R on the common set of molecules with an empty TD
  `edge_mask`; ETKDG / MMFF / λ references recomputed on that set in the same run [E-GEOM-004]. Go rule (non-oracle):
  `L_tmpl` beats `L_etkdg2L` on the common rigid set by ≥ 0.010 Å AMR-R or lowers the > 10° share (excl. all-3-ring
  basis) by ≥ 5 points; λ references for interpretation only.
- **Step 4, GPU (inference only):** CTRL_rematch s0–2 and B1 s0–2 on the best template source = 6 runs (+ S3/S4 if
  final). Controls: same checkpoints on `L_etkdg2L` / `L_etkdg2L_mmff` (exist), and on `L_etkdg2L_xtb` if C-GEOM-01
  runs. Primary: paired AMR-R, CTRL-on-template vs CTRL-on-ETKDG (expected lower; scout range 0.135–0.165, bounded
  below by A5ring's 0.119); secondary COV-R@0.1, rigid and ring-size strata, B1 − CTRL on `L_tmpl_xtb`.
- **Cost:** 1.5–3 GPU-h; CPU library build over all training conformers (count not established here, UNVERIFIED;
  runtime not measured); ~1–2 days coding
  (library + transplant + tests).

## 6. Scout's own call (scout)
Worth trying: YES — directly attacks the measured pucker tail (true rings help both models by ≈ 0.075 Å) with training data only,
needs no new model, and composes with C-GEOM-01; the CPU gate kills it cheaply if coverage or transplant fidelity is
poor.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
