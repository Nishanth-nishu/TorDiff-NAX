# R-SAMPLE data (round 5, Exp 5 headroom)
Inputs (read-only copies from gnode118 `/scratch/nishanth.r/tordiff/results/qm9_default/R{0,1,2}_base_seed*/`
`eval.pkl`, `confs.pkl`, `breakdown.csv`), copied 2026-10-15 to the session scratchpad
(`...\scratchpad\sample\data\`; not committed, 150 MB). Released TD QM9 model, 20 steps, sampling seeds 0/1/2.
Pipeline (run with `python -I`):
1. `curves.py data curves.pkl` — exact expected recall (COV-R at 0.05–0.75 Å, AMR-R) vs sample count n for a random
   n-subset of the pooled 3 x 2K samples (hypergeometric formulas in the docstring). 927 molecules present in all
   three runs with no NaN row (R0 935, R1/R2 931 evaluated; 1 molecule with an additional failure dropped).
2. `alloc.py curves.pkl breakdown.csv alloc.csv` -> `alloc.out` (K-aware framing, expected values).
3. `strata.py curves.pkl breakdown.csv strata_growth.csv` -> `strata.out` (growth with n by n_rot and K strata).
4. `signals.py data curves.pkl signals.pkl` — GT-free signals: rotamer strings, torsions, MMFF94 single-point E.
5. `alloc2.py data curves.pkl signals.pkl breakdown.csv alloc2.csv` -> `alloc2.out` (deployment framing, realised
   prefixes of 5 random pool permutations; K unknown except for the TD rule).
6. `select.py data curves.pkl signals.pkl select.csv` -> `select.out` (protocol-preserving framing: output exactly
   2K, selected from c x 2K proposals).
All numbers are on the 927-molecule success set (failures excluded), so they are NOT TD-convention values; the TD
convention adds 73 zero-coverage molecules for this set. All ORACLE rows use test-set ground truth: analysis only.
