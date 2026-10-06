import pickle, random
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--confs', type=str, required=True, help='Path to pickle file with generated conformers')
parser.add_argument('--test_csv', type=str, default='./data/DRUGS/test_smiles_corrected.csv', help='Path to csv file with list of smiles')
parser.add_argument('--true_mols', type=str, default='./data/DRUGS/test_mols.pkl', help='Path to pickle file with ground truth conformers')
parser.add_argument('--n_workers', type=int, default=1, help='Numer of parallel workers')
parser.add_argument('--limit_mols', type=int, default=0, help='Limit number of molecules, 0 to evaluate them all')
parser.add_argument('--dataset', type=str, default="drugs", help='Dataset: drugs, qm9 and xl')
parser.add_argument('--filter_mols', type=str, default=None, help='If set, is path to list of smiles to test')
parser.add_argument('--only_alignmol', action='store_true', default=False, help='If set instead of GetBestRMSD, it uses AlignMol (for large molecules)')
# [ablation-hooks]
parser.add_argument('--out_results', type=str, default=None, help='[ablation-hooks] pickle path for the per-molecule RMSD matrices + summary')
parser.add_argument('--report_threshold', type=float, default=None, help='[ablation-hooks] threshold for the one-line summary (default 0.5 for qm9, 0.75 otherwise)')
args = parser.parse_args()

"""
    Evaluates the RMSD of some generated conformers w.r.t. the given set of ground truth
    Part of the code taken from GeoMol https://github.com/PattanaikL/GeoMol
"""

with open(args.confs, 'rb') as f:
    model_preds = pickle.load(f)

test_data = pd.read_csv(args.test_csv)  # this should include the corrected smiles
with open(args.true_mols, 'rb') as f:
    true_mols = pickle.load(f)
threshold = threshold_ranges = np.arange(0, 2.5, .125)


def calc_performance_stats(rmsd_array):
    coverage_recall = np.mean(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0)
    amr_recall = rmsd_array.min(axis=1).mean()
    coverage_precision = np.mean(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(threshold, 1), axis=1)
    amr_precision = rmsd_array.min(axis=0).mean()

    return coverage_recall, amr_recall, coverage_precision, amr_precision


def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]


rdkit_smiles = test_data.smiles.values
corrected_smiles = test_data.corrected_smiles.values

if args.limit_mols:
    rdkit_smiles = rdkit_smiles[:args.limit_mols]
    corrected_smiles = corrected_smiles[:args.limit_mols]

num_failures = 0
results = {}
jobs = []

filter_mols = None
if args.filter_mols:
    with open(args.filter_mols, 'rb') as f:
        filter_mols = pickle.load(f)

for smi, corrected_smi in tqdm(zip(rdkit_smiles, corrected_smiles)):

    if filter_mols is not None and corrected_smi not in filter_mols:
        continue

    if args.dataset == 'xl':
        smi = corrected_smi

    if corrected_smi not in model_preds:
        print('model failure', corrected_smi)
        num_failures += 1
        continue

    true_mols[smi] = true_confs = clean_confs(corrected_smi, true_mols[smi])

    if len(true_confs) == 0:
        print(f'poor ground truth conformers: {corrected_smi}')
        continue

    n_true = len(true_confs)
    n_model = len(model_preds[corrected_smi])
    results[(smi, corrected_smi)] = {
        'n_true': n_true,
        'n_model': n_model,
        'rmsd': np.nan * np.ones((n_true, n_model))
    }
    for i_true in range(n_true):
        jobs.append((smi, corrected_smi, i_true))


def worker_fn(job):
    smi, correct_smi, i_true = job
    true_confs = true_mols[smi]
    model_confs = model_preds[correct_smi]
    tc = true_confs[i_true]

    rmsds = []
    for mc in model_confs:
        try:
            if args.only_alignmol:
                rmsd = AllChem.AlignMol(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
            else:
                rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
            rmsds.append(rmsd)
        except:
            print('Additional failure', smi, correct_smi)
            rmsds = [np.nan] * len(model_confs)
            break
    return smi, correct_smi, i_true, rmsds


def populate_results(res):
    smi, correct_smi, i_true, rmsds = res
    results[(smi, correct_smi)]['rmsd'][i_true] = rmsds


random.shuffle(jobs)
if args.n_workers > 1:
    p = Pool(args.n_workers)
    map_fn = p.imap_unordered
    p.__enter__()
else:
    map_fn = map

for res in tqdm(map_fn(worker_fn, jobs), total=len(jobs)):
    populate_results(res)

if args.n_workers > 1:
    p.__exit__(None, None, None)

stats = []
for res in results.values():
    stats_ = calc_performance_stats(res['rmsd'])
    cr, mr, cp, mp = stats_
    stats.append(stats_)
coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

for i, thresh in enumerate(threshold_ranges):
    print('threshold', thresh)
    coverage_recall_vals = [stat[i] for stat in coverage_recall] + [0] * num_failures
    coverage_precision_vals = [stat[i] for stat in coverage_precision] + [0] * num_failures
    print(f'Recall Coverage: Mean = {np.mean(coverage_recall_vals) * 100:.2f}, Median = {np.median(coverage_recall_vals) * 100:.2f}')
    print(f'Recall AMR: Mean = {np.nanmean(amr_recall):.4f}, Median = {np.nanmedian(amr_recall):.4f}')
    print(f'Precision Coverage: Mean = {np.mean(coverage_precision_vals) * 100:.2f}, Median = {np.median(coverage_precision_vals) * 100:.2f}')
    print(f'Precision AMR: Mean = {np.nanmean(amr_precision):.4f}, Median = {np.nanmedian(amr_precision):.4f}')

print(len(results), 'conformer sets compared', num_failures, 'model failures', np.isnan(amr_recall).sum(),
      'additional failures')

# [ablation-hooks] compact summary at the dataset threshold + optional dump of the raw per-molecule RMSD matrices
report_thr = args.report_threshold if args.report_threshold is not None else (0.5 if args.dataset == 'qm9' else 0.75)
# computed at EXACTLY report_thr from the RMSD matrices (upstream grid is np.arange(0, 2.5, .125): snapping to the
# nearest grid value silently turned e.g. --report_threshold 0.1 into 0.125). Identical to the printed block at 0.5/0.75.
_cr = [float(np.mean(np.min(r['rmsd'], axis=1) < report_thr)) for r in results.values()] + [0] * num_failures
_cp = [float(np.mean(np.min(r['rmsd'], axis=0) < report_thr)) for r in results.values()] + [0] * num_failures
summary = {
    'threshold': float(report_thr),
    'COV-R_mean': float(np.mean(_cr) * 100), 'COV-R_median': float(np.median(_cr) * 100),
    'MAT-R_mean': float(np.nanmean(amr_recall)), 'MAT-R_median': float(np.nanmedian(amr_recall)),
    'COV-P_mean': float(np.mean(_cp) * 100), 'COV-P_median': float(np.median(_cp) * 100),
    'MAT-P_mean': float(np.nanmean(amr_precision)), 'MAT-P_median': float(np.nanmedian(amr_precision)),
    'n_evaluated': len(results), 'n_model_failures': num_failures,
    'n_additional_failures': int(np.isnan(amr_recall).sum()),
}
print('SUMMARY', ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in summary.items() if k != 'sweep'))
# threshold sweep incl. sub-grid values (QM9 at 0.5 A is saturated; 0.05/0.1/0.25 A are more informative).
# Same conventions as above: strict <, failures count as 0 coverage.
sweep = {}
for thr in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25):
    cr = [float(np.mean(np.min(r['rmsd'], axis=1) < thr)) for r in results.values()] + [0] * num_failures
    cp = [float(np.mean(np.min(r['rmsd'], axis=0) < thr)) for r in results.values()] + [0] * num_failures
    sweep[thr] = (100 * np.mean(cr), 100 * np.median(cr), 100 * np.mean(cp), 100 * np.median(cp))
    print(f'SWEEP thr={thr:.3f} COV-R_mean={sweep[thr][0]:.2f} COV-R_median={sweep[thr][1]:.2f} '
          f'COV-P_mean={sweep[thr][2]:.2f} COV-P_median={sweep[thr][3]:.2f}')
summary['sweep'] = sweep
if args.out_results:
    with open(args.out_results, 'wb') as f:
        pickle.dump({'summary': summary, 'threshold_ranges': threshold_ranges, 'num_failures': num_failures,
                     'results': results, 'args': vars(args)}, f)
    print('Saved per-molecule results to', args.out_results)
