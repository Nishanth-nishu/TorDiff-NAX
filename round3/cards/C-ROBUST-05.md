# C-ROBUST-05: Tune the S3 mixture: ratio sweep or a scheduled-sampling-style curriculum over L quality

## 1. Idea (scout)
Keep S3's mechanism but tune how much true L the model sees. Either sweep p_gt (e.g. 0.25 and 0.75 next to round-2's
0.5), or schedule it during training: start on true L and anneal toward RDKit-heavy mixing, or the reverse. The aim is
to move S3 closer to B1 on true L and to CTRL on RDKit L at the same time.

## 2. Where it comes from (scout)
- Cascaded diffusion augmented 50% of training inputs and none at inference [E-ROBUST-026].
- In CFG, an unconditional share of 0.1-0.2 performed equally, and 0.5 was worse [E-ROBUST-046]. Practitioners use 10-20%
  [E-ROBUST-055].
- Scheduled sampling: the train/inference input discrepancy lets errors accumulate [E-ROBUST-040], and a curriculum
  gradually makes the model deal with its own mistakes [E-ROBUST-041]. Huszár shows the scheduled-sampling objective is
  improper and inconsistent [E-ROBUST-042]. In his analysis the model is fed generated inputs but trained on the
  original targets, and is pushed toward a trivial solution that ignores the conditioning content [E-ROBUST-043].
- ProteinMPNN reports such a trade-off in one protein setting: noise-trained models lose about 3 points of recovery on
  exact inputs while gaining about 1.6 on predicted ones [E-ROBUST-031, E-ROBUST-033]. Whether a trade-off is
  unavoidable for us is INFERENCE, not shown (revised after V2 card note).

## 3. Why it could matter here (scout)
- The headroom at S3's endpoints is small. On the same in-job test sets, S3 (seed 0) is 0.003 Å behind CTRL_rematch on
  RDKit L (0.1816 vs 0.1786) and 0.012 Å behind B1 on true L (0.0331 vs 0.0212, ORACLE) [E-ROBUST-004, E-ROBUST-060]
  (revised after D-109). INFERENCE: since S3 sits between the two specialists, moving p_gt mainly trades one endpoint
  against the other; it cannot plausibly gain more than these gaps at the endpoints.
- The gaps are resolvable, so noise is not the argument. Training-seed ranges of the reference models at these
  endpoints are 0.001-0.007 Å [E-ROBUST-061]. S3's own spread is unknown (1 seed) [E-ROBUST-004]. 3 seeds per arm would
  likely resolve a 0.005 Å shift (revised after D-109; the earlier use of B1's spread under 0.02 Å noise, E-ROBUST-012,
  was the wrong condition and is withdrawn from this card).
- The knob costs nothing in code: `--l_mix_p_gt` takes any value [E-ROBUST-014].

## 4. Assumptions that may not transfer (scout)
- CFG's ratio result is about a guidance branch that only supplies a direction [E-ROBUST-046]. Our RDKit-L branch must be
  accurate on its own, so the CFG optimum (10-20%) says nothing about ours.
- Scheduled sampling targets compounding errors in autoregressive generation [E-ROBUST-040]. TD's L is fixed during
  sampling, so nothing compounds. Huszár's failure comes from generated inputs with the original targets
  [E-ROBUST-042, E-ROBUST-043]. S3's targets are consistent: each L carries its own matched torsions [E-ROBUST-016]. So
  neither scheduled sampling's motivation nor Huszár's failure mode applies (revised after D-103). INFERENCE: a
  curriculum that ends at a fixed mixture optimises that mixture's objective at convergence; only the optimisation path
  differs.
- INFERENCE (not from Huszár): at low p_gt the model would see true L rarely, so by data weighting alone it would drift
  toward CTRL behaviour. A low-p_gt arm would most likely re-trace the S3-CTRL line.

## 5. Minimal experiment (scout)
Only if the panel overrules the NO. p_gt ∈ {0.25, 0.75} × 3 seeds = 6 trainings (about 72-96 GPU-h), tested on RDKit,
GT cycled and λ = 0.5. Expected: a monotone trade-off along the RDKit/true-L axis with changes of at most about
0.003-0.012 Å, and no Pareto gain over p = 0.5. No curriculum arm is proposed.

## 6. Scout's own call (scout)
Worth trying: NO (this round). The headroom is ≤ 0.003 Å on the primary non-oracle endpoint (RDKit L) and ≤ 0.012 Å on
an ORACLE endpoint. A ratio change is expected to trade one against the other (INFERENCE). Curricula have no compounding
error to fix here. The cost is about 2-3× C-ROBUST-01. Revisit only if S3 seed 1 or the S3 panel shows the RDKit-L cost
growing above 0.01 Å, or a large gap at λ = 0.5.

Revision log (scout, 2026-10-08, after verify_V2.md). The call is unchanged; its reasons are corrected.
- §3 "0.004 / 0.013 Å" became the like-for-like 0.003 / 0.012 Å (D-109).
- §3 "Effects that small sit near our noise [E-ROBUST-012]" is withdrawn. E-012 was B1 under 0.02 Å noise, not the
  endpoint seed noise, which is 0.001-0.007 Å [E-ROBUST-061] (D-109).
- §4 "Huszár's 'ignore the conditioning' failure is a real risk at high RDKit share" is withdrawn: his mechanism needs
  un-matched targets. It is replaced by a labelled data-weighting INFERENCE (D-103).
- §2 "Some trade-off is inherent" and §3 "No ratio can gain more" are now labelled INFERENCE.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
