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
  improper and inconsistent [E-ROBUST-042], and that it pushes models toward a trivial solution that ignores the
  conditioning content [E-ROBUST-043].
- ProteinMPNN: robust (noise-trained) models lose about 3 points of recovery on exact inputs while gaining about 1.6 on
  predicted ones [E-ROBUST-031, E-ROBUST-033]. Some trade-off is inherent.

## 3. Why it could matter here (scout)
- The headroom at S3's endpoints is small. S3 is 0.004 Å behind CTRL on RDKit L (0.1816 vs 0.1779) and 0.013 Å behind B1
  on true L (0.0331 vs 0.0199) [E-ROBUST-001, E-ROBUST-002, E-ROBUST-004]. No ratio can gain more than that at the two
  endpoints.
- Effects that small sit near our noise. B1's AMR-R spans 0.092-0.134 across 3 training seeds under 0.02 Å noise
  [E-ROBUST-012]. S3 has 1 seed so far [E-ROBUST-004]. A sweep would need ≥ 3 seeds per arm to resolve 0.005-0.01 Å.
- The knob costs nothing in code: `--l_mix_p_gt` takes any value [E-ROBUST-014].

## 4. Assumptions that may not transfer (scout)
- CFG's ratio result is about a guidance branch that only supplies a direction [E-ROBUST-046]. Our RDKit-L branch must be
  accurate on its own, so the CFG optimum (10-20%) says nothing about ours.
- Scheduled sampling targets compounding errors in autoregressive generation [E-ROBUST-040]. TD's L is fixed during
  sampling, so nothing compounds. Its objective is also inconsistent [E-ROBUST-042]. Our mixed targets are consistent:
  each L carries its own matched torsions [E-ROBUST-016]. A curriculum that ends at a fixed mixture optimises that
  mixture's objective at convergence. Only the optimisation path differs. INFERENCE.
- Huszár's "ignore the conditioning" failure [E-ROBUST-043] is a real risk at high RDKit share (low p_gt). The model may
  stop reading L detail and fall back toward CTRL. A low-p_gt arm would most likely just re-trace the S3-CTRL line.

## 5. Minimal experiment (scout)
Only if the panel overrules the NO. p_gt ∈ {0.25, 0.75} × 3 seeds = 6 trainings (about 72-96 GPU-h), tested on RDKit,
GT cycled and λ = 0.5. Expected: a monotone trade-off along the RDKit/true-L axis with changes ≤ 0.01 Å, and no Pareto
gain over p = 0.5 beyond seed noise. No curriculum arm is proposed.

## 6. Scout's own call (scout)
Worth trying: NO (this round). The endpoint headroom is ≤ 0.004 / ≤ 0.013 Å, curricula have no compounding error to fix
here, and the cost is about 2-3× C-ROBUST-01. Revisit only if S3 seed 1 or the S3 panel shows the RDKit-L cost growing
above 0.01 Å, or a large gap at λ = 0.5.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
