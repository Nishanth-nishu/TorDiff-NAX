"""V2 (P3 claim check of impact.md 3g): independent parse of round-1 breakdown.log 'by n_true' tables.
Per bin and per run: MAT_R, MAT_P, COV_R(0.5). Compares random-GT-L (gtL / A1) with cycled-GT-L (gtLcycle / A1c),
the model-free 'perfect copier' (A1_gtL_sanity_identity: seeds returned unchanged), and computes the inversion
A_other = cyc + (gtL - cyc) / P0 with exact P0 per bin (from l_error.csv molecule counts) and alternative P0s.
Usage (repo root): python -I round3/ledger/v2_data/v2_ntrue_check.py"""
import csv, glob, os, re, statistics as st
from collections import Counter, defaultdict

R1 = 'cluster_sync/results'
BINS = ['1', '2-3', '4-10', '11-30', '>30']


def table(path, section='by n_true'):
    txt = open(path, encoding='utf-8', errors='replace').read()
    sec = txt.split('### ' + section)[1].split('###')[0]
    out = {}
    for line in sec.splitlines():
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) == 6 and c[0] in BINS:
            out[c[0]] = dict(n=int(c[1]), covr=float(c[2]), matr=float(c[3]), covp=float(c[4]), matp=float(c[5]))
    return out


def runs(pattern):
    fs = sorted(glob.glob(os.path.join(R1, pattern, 'breakdown.log')))
    return fs, [table(f) for f in fs]


def show(name, pattern):
    fs, ts = runs(pattern)
    print(f'{name:28s} [{len(fs)} runs] MAT_R per bin: ' + ' | '.join(
        f'{b}: ' + '/'.join(f'{t[b]["matr"]:.3f}' for t in ts) for b in BINS))
    return ts


# exact P0 per bin from the molecules' GT-conformer counts (noise rows: 2L rows per molecule)
L = Counter()
for r in csv.DictReader(open('cluster_sync/round2/data/QM9/round2_seeds/l_error.csv')):
    if r['cond'] == 'L_noise0.02pa_cyc_ORACLE':
        L[r['smiles']] += 1
nmol = {s: c // 2 for s, c in L.items()}


def b_of(n):
    return '1' if n == 1 else '2-3' if n <= 3 else '4-10' if n <= 10 else '11-30' if n <= 30 else '>30'


def p0(n):
    return 0.0 if n == 1 else (1 - 1 / n) ** (2 * n)


per = defaultdict(list)
for s, n in nmol.items():
    per[b_of(n)].append(n)
P0 = {b: st.mean(p0(n) for n in per[b]) for b in BINS}
print('molecules per bin', {b: len(per[b]) for b in BINS}, ' mean P0', {b: round(P0[b], 4) for b in BINS})
print('n distribution in bin 2-3:', Counter(per['2-3']))

print('\n## MAT_R per bin (round-1 in-job; seeds listed)')
T = {}
T['B1_gtL'] = show('B1 gtL (random GT L)', 'qm9_B1_train_gtL_e100_s*/steps20_seed0_gtL')
T['B1_cyc'] = show('B1 gtLcycle', 'qm9_B1_train_gtL_e100_s*/steps20_seed0_gtLcycle')
T['B1_rdk'] = show('B1 RDKit in-job', 'qm9_B1_train_gtL_e100_s*/steps20_seed0')
T['CB_gtL'] = show('CTRL_base gtL', 'qm9_CTRL_base_100ep_e100_s*/steps20_seed0_gtL')
T['CB_cyc'] = show('CTRL_base gtLcycle', 'qm9_CTRL_base_100ep_e100_s*/steps20_seed0_gtLcycle')
T['CB_rdk'] = show('CTRL_base RDKit in-job', 'qm9_CTRL_base_100ep_e100_s*/steps20_seed0')
T['A1'] = show('released A1 gtL (inf seeds 0,1,2)', 'qm9_default/A1_gtL_model*')
T['A1c'] = show('released A1c cycle (inf 0,1,2)', 'qm9_default/A1c_gtL_cycle_model*')
T['copy'] = show('copier: gtL seeds unchanged', 'qm9_default/A1_gtL_sanity_identity')
T['copyc'] = show('copier: cycle seeds unchanged', 'qm9_default/A1c_gtL_cycle_sanity')
T['R0'] = show('released RDKit L (R0-R2)', 'qm9_default/R*_base_seed*')


def m(key, b, f='matr'):
    return st.mean(t[b][f] for t in T[key])


print('\n## Delta = random-GT minus cycled, per bin (mean over runs); copier Delta = model-free upper reference')
for nm, g, c in (('B1', 'B1_gtL', 'B1_cyc'), ('CTRL_base', 'CB_gtL', 'CB_cyc'), ('released', 'A1', 'A1c'),
                 ('copier', 'copy', 'copyc')):
    print(f'{nm:10s} ' + ' | '.join(f'{b}: {m(g, b) - m(c, b):+.4f}' for b in BINS))
print('released per inference seed (different L draws per seed):')
for i in range(3):
    print('   seed', i, ' | '.join(f'{b}: {T["A1"][i][b]["matr"] - T["A1c"][i][b]["matr"]:+.3f}' for b in BINS))
print('B1 per training seed (SAME L draws, inference seed 0):')
for i in range(3):
    print('   seed', i, ' | '.join(f'{b}: {T["B1_gtL"][i][b]["matr"] - T["B1_cyc"][i][b]["matr"]:+.3f}' for b in BINS))

print('\n## Inversion A_other = cyc + Delta / P0 (P0 = expected, per bin); copier gives d_other = Delta_copy / P0')
for nm, g, c, rd in (('B1', 'B1_gtL', 'B1_cyc', 'B1_rdk'), ('CTRL_base', 'CB_gtL', 'CB_cyc', 'CB_rdk'),
                     ('released', 'A1', 'A1c', 'R0')):
    row = []
    for b in BINS[1:]:
        d = m(g, b) - m(c, b)
        row.append(f'{b}: own {m(c, b):.3f} -> other {m(c, b) + d / P0[b]:.3f} (RDKit {m(rd, b):.3f})')
    print(f'{nm:10s} ' + ' | '.join(row))
print('copier     ' + ' | '.join(f'{b}: d_other {(m("copy", b) - m("copyc", b)) / P0[b]:.3f}' for b in BINS[1:]))
print('B1 Delta / copier Delta: ' + ' | '.join(
    f'{b}: {(m("B1_gtL", b) - m("B1_cyc", b)) / (m("copy", b) - m("copyc", b)):.2f}' for b in BINS[1:]))

print('\n## Copy-number check: MAT_P (per generated sample, nearest target) vs MAT_R on cycled own L')
for key in ('B1_cyc', 'CB_cyc', 'A1c'):
    print(f'{key:8s} ' + ' | '.join(f'{b}: R {m(key, b):.3f} P {m(key, b, "matp"):.3f}' for b in BINS))

# weighted averages over multi-conformer bins, ETKDG-population weights (as the analyst) and in-job counts
W = {'2-3': 227, '4-10': 274, '11-30': 127, '>30': 84}
for nm, g, c, rd in (('B1', 'B1_gtL', 'B1_cyc', 'B1_rdk'), ('CTRL_base', 'CB_gtL', 'CB_cyc', 'CB_rdk')):
    ao = {b: m(c, b) + (m(g, b) - m(c, b)) / P0[b] for b in W}
    wa = sum(ao[b] * W[b] for b in W) / sum(W.values())
    wr = sum(m(rd, b) * W[b] for b in W) / sum(W.values())
    wc = sum(m(c, b) * W[b] for b in W) / sum(W.values())
    wg = sum(m(g, b) * W[b] for b in W) / sum(W.values())
    print(f'{nm}: multi-conformer weighted: cycled {wc:.3f}  gtL {wg:.3f}  implied other-L {wa:.3f}  RDKit in-job {wr:.3f}')
