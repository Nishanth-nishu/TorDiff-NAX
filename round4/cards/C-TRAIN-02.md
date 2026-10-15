# C-TRAIN-02: Train on the L you deploy: xTB-matched training (S5-xTB) and a matched multi-source mixture

## 1. Idea (scout)
Generalise S5 (TD trained on MMFF-relaxed, conformer-matched L) to the next deployed L source, GFN2-xTB-relaxed ETKDG
seeds (round-3 C-GEOM-01): relax every training embedding with xTB before conformer matching, train TD on it, and test
with xTB seeds. Second arm: a per-sample mixture of matched L sources {RDKit, MMFF, xTB} (+ sibling GT from
C-TRAIN-01), all non-own and all torsion-matched, so the model sees L quality vary without ever seeing an L that
encodes its target. The same rule binds the ARCH track: whatever L generator is built, the torsion model is trained on
that generator's samples for the training molecules, re-matched.

## 2. Where it comes from (scout)
- TD: training on true L causes a damaging test-time shift; conformer matching to the sampler's L removes it
  [E-TRAIN-022, E-TRAIN-023]. DAgger: train on the input distribution the deployed system induces, labelled by an
  expert [E-TRAIN-041]; conformer matching is the relabelling (INFERENCE). Cascades: conditioning augmentation helps
  because it removes stage-to-stage train/test mismatch [E-TRAIN-042].
- Training on the upstream model's outputs works at scale: ESM-IF1 trains on millions of AlphaFold2 predictions and a
  higher predicted share helps [E-TRAIN-032], with small noise to avoid learning generator-specific artefacts
  [E-TRAIN-031]. DynamicBind builds training decoys from the upstream (AlphaFold) structure instead of Gaussian noise,
  arguing noise only teaches reverting to the pre-noise state [E-TRAIN-035, E-TRAIN-036].
- In conformer generation, EBD names the RDKit-vs-GT mismatch as harmful [E-TRAIN-037], trains on real RDKit
  conformers matched to GT by linear-sum assignment [E-TRAIN-038], and its model corrects bad RDKit priors most
  [E-TRAIN-039]. Ringer's TD retraining used force-field-processed training L for macrocycles [E-TRAIN-040].
- GEOM's QM9 references are GFN2-xTB minima [E-TRAIN-081], so xTB is the reference-level relaxation.

## 3. Why it could matter here (scout)
- S5 is the only training recipe that improved real use: −0.0041 Å [−0.0064, −0.0019] vs CTRL with MMFF L at test;
  on raw RDKit L it is +0.0031 worse, so the gain is specific to the matched source [E-TRAIN-004].
- CTRL already gains 0.025 Å from MMFF L without retraining [E-TRAIN-003]; MMFF leaves a 35.1 % ring tail vs ETKDG's
  49.6 % [E-TRAIN-013], and RDKit's ring geometry is a known limit of TD [E-TRAIN-082]. If xTB cuts the tail further,
  CTRL's own train/test shift on xTB L grows, and matched training should recover more than S5's 0.004 (INFERENCE).
- The matcher accepts any L-carrying molecule, and standardize_confs.py already has an MMFF relaxation hook between
  embedding and matching (:83-87) [E-TRAIN-007, E-TRAIN-014].
- **Predicted effect.** Round-3 impact predicts CTRL on xTB seeds at 0.146 (0.13–0.155). S5-xTB: −0.002 to −0.008
  vs CTRL on xTB L (MMFF analogue −0.004), i.e. ≈ 0.138–0.150. Mixture arm: within ±0.003 of the best single-source
  model on each source, and better on unseen sources (e.g. template L). Neither touches the rigid stratum, which L
  alone sets [E-TRAIN-012].

## 4. Assumptions that may not transfer (scout)
- Gains are small (S5: 0.004 Å) and need ≥ 3 seeds to detect (CTRL-family seed SD ≈ 0.001–0.002).
- If xTB fixes few more puckers than MMFF, S5-xTB ≈ S5 and adds nothing (the gate below).
- The literature (ESM-IF1, DynamicBind, EBD) trains models that also *correct* their input; TD cannot change L, so only
  the shift-removal half transfers.
- xTB over the training set is CPU-heavy: ≈ 82,895 molecules × ≈ 6.8 conformers × 0.14–0.3 s ≈ 22–47 CPU-h
  (round-3 grounder/engineering estimate; not measured), and needs the race-free xTB runner (round3/grounding.md S-1).
- Mixture arm: one model must cover sources with different error profiles; CDM found that the augmentation type must
  suit the input [E-TRAIN-043].

## 5. Minimal experiment (scout)
- **Gate:** run only if C-GEOM-01 shows CTRL on xTB seeds ≤ CTRL on MMFF seeds − 0.005 Å, or the xTB ring tail
  ≤ 25 % (pre-declared; the gate selects the experiment, not the model).
- **Build (CPU):** `standardize_confs.py --xtb` mirroring `--mmff` (:18, :83-87), calling the S-1 runner per embedding
  before matching; failures counted like MMFF's. VARIANT `xtb` in `slurm/standardize_qm9.sbatch`,
  `featurize_qm9.sbatch`, `ablation_train_array.sbatch`. Inference already has the seed hook (sampling.py:75 uses MMFF;
  the xTB seed path comes from C-GEOM-01).
- **Arms:** S5-xTB (3 seeds). Optional: MIX = per-sample choice among RDKit-, MMFF-, xTB-matched (and sibling, if
  C-TRAIN-01 passes) caches for the same conformer (needs a multi-L paired cache, ≈ +0.4 GB per source).
- **Controls:** CTRL_rematch with xTB seeds, S5 with MMFF and with xTB seeds.
- **Test:** RDKit, MMFF, xTB seeds (primary non-oracle); split-half ORACLE secondary.
- **Primary contrast:** S5-xTB vs CTRL_rematch, both on xTB seeds; expected −0.002 to −0.008, superiority, paired,
  Holm, two-level bootstrap.
- **Cost:** CPU ≈ 25–50 h (xTB) + matching 1–4 h on 32 cores; GPU 3 × 11.7 = 35 + ≈ 3 evaluation; MIX +35.

## 6. Scout's own call (scout)
Worth trying: YES (gated on C-GEOM-01) — the only training principle with a measured real-use gain (S5), and the
rule any new L source (including the ARCH generator) must follow; the MIX arm is MAYBE.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
