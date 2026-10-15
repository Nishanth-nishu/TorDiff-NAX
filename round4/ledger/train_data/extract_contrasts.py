"""Scout TRAIN (round 4): extract the MAT-R (AMR-R) and COV-R@0.1 union rows of the round-2 paired analysis.
Reads only cluster_sync/round2/results/analysis_r2/contrasts.csv (run from the repo root). Output: contrasts_extract.txt.
Each output line starts with the 1-based line number of the source CSV row (header = line 1)."""
import csv
SRC = 'cluster_sync/round2/results/analysis_r2/contrasts.csv'
OUT = 'round4/ledger/train_data/contrasts_extract.txt'
rows = list(csv.DictReader(open(SRC, newline='')))
with open(OUT, 'w') as f:
    f.write(f'# source: {SRC}; columns: csv_line contrast metric n arm arm_tag ref ref_tag arm_value ref_value diff '
            f'[ci_lo, ci_hi] n_arm_seeds n_ref_seeds p_holm_primary\n')
    for i, r in enumerate(rows):
        if r['universe'] != 'union' or r['metric'] not in ('MAT-R', 'COV-R@0.1'):
            continue
        f.write(f"line {i + 2}: {r['contrast']} {r['metric']} n={r['n']} {r['arm']}:{r['arm_tag']} vs "
                f"{r['ref']}:{r['ref_tag']} arm={float(r['arm_value']):.4f} ref={float(r['ref_value']):.4f} "
                f"diff={float(r['diff']):+.4f} [{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}] "
                f"seeds={r['n_arm_seeds']}v{r['n_ref_seeds']} p_holm={r['p_holm_primary'] or 'NA'}\n")
print('wrote', OUT)
