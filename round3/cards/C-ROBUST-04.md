# C-ROBUST-04: Classifier-free-guidance-style "L guidance" (L dropout + guided score), or CTRL/B1 score interpolation

## 1. Idea (scout)
Treat accurate L like a condition in classifier-free guidance (CFG). Train one model with the L detail "dropped" part
of the time (replaced by RDKit-type L, which is what S3 does with p = 0.5). At sampling, push toward the L-trusting
score: s = s(τ | L_weak) + w·[s(τ | L) − s(τ | L_weak)], with w > 1. A training-free variant interpolates the scores of
the existing CTRL and B1 models with a weight w chosen per L source.

## 2. Where it comes from (scout)
- CFG trains one network for both scores by replacing the condition with a null token [E-ROBUST-048]. An unconditional
  share of 0.1-0.2 worked equally well, and 0.5 was worse across the whole IS/FID frontier [E-ROBUST-046]. Practitioners
  report 10-20% conditioning dropout [E-ROBUST-055].
- Guidance raises fidelity at the expense of diversity [E-ROBUST-047]. Dieleman calls the diversity cost "great"
  [E-ROBUST-054].
- Mixing or summing scores and running the ordinary reverse process does not sample the composed distribution
  [E-ROBUST-049].

## 3. Why it could matter here (scout)
- Neither existing model is best everywhere. B1 overtakes CTRL only at λ ≈ 0.75, and is 0.045 Å worse at λ = 0.5
  [E-ROBUST-005]. A knob that slides between them per L source looks attractive.
- S3 already contains the "dropout" half: half of its samples carry RDKit L [E-ROBUST-014]. It reaches 0.182 / 0.033
  without any guidance [E-ROBUST-004].

## 4. Assumptions that may not transfer (scout)
- **L is not a droppable token in TD.** CFG's unconditional pass swaps a class label for a null token [E-ROBUST-048].
  TD's score model reads L from the same 3D coordinates that carry the torsions. An "L-free" score at the current
  torsion state would need a second conformer (weak L with the current torsions) built and featurised at every step.
  INFERENCE from the input design [E-ROBUST-016].
- **Wrong direction for our endpoint.** Guidance trades diversity for fidelity [E-ROBUST-047, E-ROBUST-054]. Our primary
  endpoints are recall metrics (AMR-R, COV-R), which reward covering every true conformer. w > 1 is expected to hurt
  them. INFERENCE.
- **Score interpolation is not a principled mixture.** Plain reverse diffusion with combined scores does not sample the
  combined distribution [E-ROBUST-049]. CTRL and B1 were also trained on different L, so their scores disagree on what
  the clean torsions are.
- **The p_uncond result does not transfer.** In CFG the unconditional branch only supplies a guidance direction
  [E-ROBUST-046]. Our "weak-L" branch (RDKit L) is itself a deployment condition and must be accurate on its own.
- **A per-source w is a per-source hyperparameter**, like S4's λ, and carries the same deployment problem
  [E-ROBUST-017].

## 5. Minimal experiment (scout)
Only if the panel overrules the NO. Inference-only diagnostic: CTRL_rematch s0 and B1 s0, score interpolation w ∈ {0.25,
0.5, 0.75}, on RDKit L, MMFF L and λ = 0.5 (ORACLE): 9 runs at about 2× normal inference cost, about 5 GPU-h, plus about
40 lines in `diffusion/sampling.py` (two models, one weighted score). Metric: AMR-R and COV-R@0.1 against the lower
envelope of CTRL and B1 (e.g. 0.136 at λ = 0.5). Expected: at best the lower envelope. It should not beat a trained
mixture (S3/S4) and may lose fine-threshold coverage. A guided (w > 1) single-model variant is not proposed.

## 6. Scout's own call (scout)
Worth trying: NO. The CFG mechanism does not map onto TD's input, because L is not a token that can be dropped.
Guidance gives away the recall we measure. Mixed training (S3, S4, C-ROBUST-01/02) already captures the useful part
without a second network pass.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
