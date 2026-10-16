"""Do QM9 test corrected SMILES encode double-bond stereo with '/' or '\\'? (R-CHEM, consult Q-P2-007)
Usage: python -I slash_check.py   (run inside round5/ledger/chem_data; reads the csvs written by stereo_census.py)"""
import pandas as pd

d = pd.read_csv('qm9_test_smiles_from_local_structure_test.csv')
has = d.corrected_smiles.apply(lambda s: ('/' in s) or ('\\' in s))
print('QM9 test molecules whose corrected SMILES contains / or \\:', int(has.sum()), 'of', len(d))
c = pd.read_csv('census_qm9_local.csv')
x = c[c.n_db_stereo_specified > 0]
xs = x.corrected_smiles.apply(lambda s: ('/' in s) or ('\\' in s))
print('molecules "specified" under RDKit 2026.03.6:', len(x), '; of these with a / or \\ in the SMILES:', int(xs.sum()))
print('examples specified without slash:', x.corrected_smiles[~xs].head(5).tolist())
