# C-GEOM-04: Dedicated learned ring-pucker component (PuckerFlow / Cremer–Pople factor / joint ring+torsion) — not this round

## 1. Idea (scout)
Brief Q3: build a ring-geometry component in puckering coordinates — either plug a pretrained PuckerFlow sampler in
front of TD (two-stage), or add a Cremer–Pople factor to FlexiTors' product space and train it jointly with torsions
(Corso-thesis / RINGER direction). This card argues that neither should be built in round 3, and states the
conditions that would reopen it.

## 2. Where it comes from (scout)
- PuckerFlow does flow matching in Cremer–Pople coordinates; its authors fed PuckerFlow rings as fixed substructures
  into the pretrained TD (qualitative demo, no TD AMR/COV numbers) [E-GEOM-021].
- PuckerFlow is trained and benchmarked only on 5–8-membered, substituent-free, monocyclic rings [E-GEOM-020].
- On its own benchmark it clearly wins on puckering displacement (recall coverage 75.8% vs ETKDG 53.3%, AMR 0.09 vs
  0.14 Å) [E-GEOM-022], but on all-atom RMSD the margin is small: recall AMR 0.15 vs 0.17 Å unrelaxed, 0.14 vs 0.15 Å
  after MMFF [E-GEOM-023].
- Corso's thesis proposes ring-puckering coordinates on hyperspheres as an extra factor [E-GEOM-032]; GO-Flow finds
  internal coordinates the most important design choice [E-GEOM-031]; RINGER needs a ring-closure step when rings are
  rebuilt from internal coordinates (shown for macrocycles) [E-GEOM-033]. TD flags puckered and fused rings as its weak
  spot [E-GEOM-024].

## 3. Why it could matter here (scout)
- Rings carry the QM9 floor: true ring geometry alone moves CTRL 0.178 → 0.119 [E-GEOM-013], and the ETKDG pucker tail
  (28.6% of ring seeds > 10°) is what separates RDKit/MMFF L from the λ = 0.75 spec [E-GEOM-002, E-GEOM-007]. So the
  target is right; the question is whether a learned pucker model is the cheapest way to hit it.
- Domain mismatch with QM9 (our data): only 223 of 885 ring molecules have all rings isolated and 5–8-membered,
  holding 48% of the bad (> 10°) ETKDG seeds; isolated 4-rings (25% of bad seeds) and fused/bridged/spiro systems
  (22%) are outside PuckerFlow's training domain [E-GEOM-011, E-GEOM-003]; 3-rings have no pucker.
- Substituents matter in QM9 (every test molecule has them), while PuckerFlow is trained without them [E-GEOM-020].

## 4. Assumptions that may not transfer (scout)
- PuckerFlow's gains are on substituent-free isolated rings scored by pucker displacement; on all-atom RMSD the gain over
  ETKDG is ~0.02 Å and shrinks to ~0.01 Å after MMFF [E-GEOM-023] — the metric we use is all-atom-like (heavy-atom
  RMSD), so the expected effect is small even in-domain (INFERENCE).
- Re-training PuckerFlow on QM9 rings (3–7-membered, fused, with substituents) is a new model, not a plug-in; fused
  systems need per-ring Cremer–Pople coordinates plus closure constraints [E-GEOM-020, E-GEOM-033].
- A joint Cremer–Pople × torus factor in FlexiTors is at least the 300+-line change estimated in round 2 for the simpler
  acyclic-angle factor, plus ring closure, plus a training run per seed [E-GEOM-049].
- Cheaper non-learned sources (C-GEOM-01 xTB, C-GEOM-02 templates) attack the same tail with GT-level geometry and no
  training; until they are measured, a learned pucker component has no demonstrated gap to fill.

## 5. Minimal experiment (scout)
None this round. Reopen in round 4 only if **all** hold (pre-declared):
1. After C-GEOM-01/02, the best non-learned source still has > 5% of ring seeds > 10° ring-dihedral RMSD, or
   rigid-subset AMR-R > 0.077 (λ = 0.75 level) [E-GEOM-004];
2. C-GEOM-03 shows a learned model can get the ring tail below 1% (so learning is the missing ingredient);
3. the residual tail is concentrated in ring classes a pucker model can represent (report by ring size/fusion,
   E-GEOM-011 breakdown).
If reopened, the cheapest first step is a PuckerFlow-style model retrained on QM9 training rings (3–7-membered, with
first-shell substituents), used two-stage like [E-GEOM-021], not a joint factor. Cost then: ~1–2 weeks of coding plus
≥ 1 training run (not estimated).

## 6. Scout's own call (scout)
Worth trying: NO (this round) — right target, wrong price: off-domain for half of QM9's pucker errors, small all-atom
gain even in-domain, and two cheaper GT-level sources (C-GEOM-01/02) must be measured first.

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
