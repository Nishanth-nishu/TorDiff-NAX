"""Growth of expected recall with sample count, by strata (n_rot_heavy, n_true), from curves.pkl."""
import pickle, sys, numpy as np, pandas as pd
C = pickle.load(open(sys.argv[1], 'rb'))['curves']; bd = pd.read_csv(sys.argv[2]).set_index(['smiles', 'corrected_smiles'])
rows = []
for k, c in C.items():
    nm = c['n_model']; r = dict(n_rot=int(bd.loc[k, 'n_rot_heavy']), K=c['K'], n_heavy=int(bd.loc[k, 'n_heavy']))
    for lab, n in [('n=1', 1), ('n=0.5x', max(1, nm // 2)), ('n=1x', nm), ('n=2x', 2 * nm), ('n=3x', 3 * nm)]:
        r[f'AMR {lab}'] = c['amr'][n]; r[f'COV0.5 {lab}'] = c['cov'][0.5][n]; r[f'COV0.1 {lab}'] = c['cov'][0.1][n]
    r['dAMR 1x->3x'] = c['amr'][3 * nm] - c['amr'][nm]
    r['dCOV0.5 1x->3x'] = c['cov'][0.5][3 * nm] - c['cov'][0.5][nm]
    r['saturated0.5 at 1x'] = c['cov'][0.5][nm] > 0.999
    rows.append(r)
df = pd.DataFrame(rows)
df['rot_bin'] = pd.cut(df.n_rot, [-1, 0, 1, 2, 3, 99], labels=['0', '1', '2', '3', '4+'])
df['K_bin'] = pd.cut(df.K, [0, 1, 3, 10, 30, 10**6], labels=['1', '2-3', '4-10', '11-30', '>30'])
cols = ['AMR n=1x', 'AMR n=3x', 'dAMR 1x->3x', 'COV0.5 n=0.5x', 'COV0.5 n=1x', 'COV0.5 n=3x', 'dCOV0.5 1x->3x', 'COV0.1 n=1x', 'COV0.1 n=3x', 'saturated0.5 at 1x']
pd.set_option('display.width', 250)
for g in ['rot_bin', 'K_bin']:
    t = df.groupby(g, observed=True)[cols].mean(); t.insert(0, 'n', df.groupby(g, observed=True).size())
    print(f'\n### by {g}\n', t.to_string(float_format=lambda v: f'{v:.4f}'))
print('\nall', df[cols].mean().to_string(float_format=lambda v: f'{v:.4f}'))
print('\nshare of total dAMR(1x->3x) from top 20% molecules:',
      df['dAMR 1x->3x'].nsmallest(int(0.2 * len(df))).sum() / df['dAMR 1x->3x'].sum())
print('spearman dAMR vs n_rot', df[['dAMR 1x->3x', 'n_rot']].corr('spearman').iloc[0, 1], ' vs K', df[['dAMR 1x->3x', 'K']].corr('spearman').iloc[0, 1])
df.to_csv(sys.argv[3], index=False)
