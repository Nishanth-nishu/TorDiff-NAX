# Proposal card template

File: `round<N>/cards/C-<AGENT>-<NN>.md`. Sections marked (scout) are filled in P1; (grounder), (analyst), (panel)
are appended later by those agents under their own headings. Nobody rewrites another agent's section.

```
# C-<AGENT>-<NN>: <short title>

## 1. Idea (scout)
<2-3 sentences: what we would change and test.>

## 2. Where it comes from (scout)
<Which papers / blogs, and the mechanism as the source describes it. Every factual sentence ends with [E-...].>

## 3. Why it could matter here (scout)
<Which of OUR findings it targets, with numbers and result files [E-... our-data]. E.g. B1 collapses on RDKit
geometry; ring geometry carries the floor; S2 noise half-works.>

## 4. Assumptions that may not transfer (scout)
<What the source assumes that TD/QM9 may not satisfy (data size, architecture, metric, molecule class).>

## 5. Minimal experiment (scout)
<Arms, control(s), test conditions, seeds, primary metric and expected direction, GPU-h estimate.>

## 6. Scout's own call (scout)
Worth trying: YES / MAYBE / NO — <one-line reason>

## 7. Code grounding (grounder)
## 8. Predicted effect on our project (analyst)
## 9. Panel (judges)
```
