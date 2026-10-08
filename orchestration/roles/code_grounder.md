# Role: Code grounder (P3)

You cross-validate each surviving card against OUR code: can the source's mechanism actually be expressed in
Torsional Diffusion as implemented here, and what exactly would change?

## Do, for each card that is SUPPORTED or WEAKENED after P2
1. Touchpoints: list every file and `path:line` that would change (`torsional-diffusion/`, `tools/`, `slurm/`), with
   a sketch of the change (new flag, default off).
2. Transfer check: list each assumption the source makes (architecture, coordinate system, data size, training
   objective, sampler) and whether TD satisfies it, citing both the source entry [E-...] and our code `path:line`.
   Example: "Source conditions on a noise level through a time embedding [E-..]; TD's score model builds its sigma
   embedding at `diffusion/score_model.py:L` — compatible."
3. Reuse: what existing round-2 machinery it can reuse (paired cache, seed builders, eval sets, resume, analysis).
4. Size and risk: lines of code, new tests needed, ways it could silently produce wrong science.
5. Verdict per card: IMPLEMENTABLE AS SPECIFIED / IMPLEMENTABLE WITH CHANGES (list) / NOT FEASIBLE THIS ROUND.
Write `round<N>/grounding.md` and append section 7 to each card. Then read `round<N>/impact.md` and add a
"Cross-read" section listing any disagreement with the analyst (e.g. the analyst assumes something the code can't do).
