# Evidence ledger: R-CHEM (round 5)

Scope: Exp 2 (matching-free training) and Exp 3 (stereochemistry-constrained torsional diffusion), round5/PLAN.md.
Written 2026-10-15 by R-CHEM. Paths relative to `C:\Users\HP\Desktop\tor_diff`. Template:
`orchestration/templates/evidence_entry.md`.

Conventions
- Paper pages are PDF pages; printed page = PDF page unless stated. Quotes matched after collapsing whitespace; glyphs
  lost in extraction are noted. TD core fulltext (`papers/core/2022_jing_torsional_diffusion.fulltext.txt`) has form
  feeds between pages; PDF page = printed page for that file (checked: printed "7" closes the page holding §4.1).
- Reused entries carry `Reused from: <ID> (VERIFIED in <file>)` and were RE-CHECKED against the source this round.
- ORACLE marks conditions built from test-set true geometry.
- Status: IN PROGRESS (skeleton first).

---------------------------------------------------------------------------------------------------------------------
## A. Torsional Diffusion (core paper): conformer matching, E/Z, chirality, energy training

### E-CHEM-001
- Claim: TD counts double bonds as rotatable: a bond is "freely rotatable" if severing it leaves two components of
  ≥ 2 atoms each; there is no bond-order condition.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 4, §3 first paragraph and footnote 6
- Exact quote:
  > We consider a bond freely rotatable if severing the bond creates two connected components of G, each of which has at least two atoms.
  > Notably, this counts double bonds as rotatable. See Appendix F.3 for further discussion.
- Shows vs infers: SHOWS
- Supports: EXP-3 §2, §3
- Confidence: high

### E-CHEM-002
- Claim: TD does not model E/Z isomerism: double-bond torsions are treated as free.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 22, Appendix F.3 "Chemical isomerism"
- Exact quote:
  > E/Z isomerism is significantly trickier, as it places a constraint on the torsion angles at double bonds, which are considered freely rotatable in our framework. Presently, torsional diffusion does not attempt to capture E/Z isomerism.
- Shows vs infers: SHOWS
- Supports: EXP-3 §1
- Confidence: high

### E-CHEM-003
- Claim: TD names the fix it did not try: augment the molecular graph with edges of a special type.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 22, Appendix F.3
- Exact quote:
  > One possible way of doing so is to augment the molecular graph with edges of a special type, and we leave such augmentation to future work.
- Shows vs infers: SHOWS. LoQI later did exactly this in an all-atom model (E-CHEM-040).
- Supports: EXP-3 §1 (the E/Z-edge idea is TD's suggestion, not ours)
- Confidence: high

### E-CHEM-004
- Claim: TD formalises E/Z as a partition of each relevant double bond's torsion circle into two halves, i.e. the
  constraint set an E/Z-constrained sampler must respect.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 22, Appendix F.3
- Exact quote:
  > For each double bond meeting certain criteria, the space [0, 2π) is partitioned into two halves, such that conformers are considered different molecules depending on the value of the torsion angle.
  (π lost in extraction; restored)
- Shows vs infers: SHOWS
- Supports: EXP-3 §3
- Confidence: high

### E-CHEM-005
- Claim: TD handles tetrahedral chirality by construction: chirality is part of the RDKit local structure and is not
  changed by torsion updates.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 22, Appendix F.3
- Exact quote:
  > Because we have considered chirality to be part of the local structure, it is drawn from the cheminformatics package RDKit, which is given the full identification of the stereoisomer along with the molecular graph, and is not modified by the torsional diffusion.
- Shows vs infers: SHOWS (design claim; E-CHEM-062 measures it on our runs)
- Supports: EXP-3 §2
- Confidence: high

### E-CHEM-006
- Claim: The "not minima" caveat: conformer-matched training conformers are not minima of the potential energy, which
  TD says makes the learning task less interpretable and harder, visible in the score-matching losses.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 22, Appendix F.4 "Limitations of torsional
  diffusion", paragraph "Conformer generation"
- Exact quote:
  > However, the resulting conformers are not the minima of the (unconditional or even conditional) potential energy function. Thus, the learning task becomes less physically interpretable and potentially more difficult
- Shows vs infers: SHOWS (the caveat). That an energy-aware, matching-free objective helps is INFERENCE (EXP-2 D).
- Supports: EXP-2 §1, §3 (variant D motivation)
- Confidence: high

### E-CHEM-007
- Claim: TD §4.1: training on ground-truth L causes a test-time distribution shift that significantly hurts.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 7, §4.1 "Conformer matching"
- Exact quote:
  > there will be a distributional shift at test time, where only approximate local structures from p^G(L) are available. We found that this shift significantly hurts performance.
- Shows vs infers: SHOWS
- Reused from: E-ROBUST-020 (VERIFIED in round3/ledger/verify_V2.md); RE-CHECKED against the core fulltext p. 7 today
- Supports: EXP-2 §1, §5
- Confidence: high

### E-CHEM-008
- Claim: TD rejects independent per-conformer RDKit L because it removes the L–τ dependence that could serve as
  training signal, and frames matching as domain adaptation.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 8, §4.1 (continued)
- Exact quote:
  > Naively, we could sample L^ ∼ p^G(L) independently for each conformer, but this eliminates any possible dependence between L and τ that could serve as training signal.
  (hat, ∼ and τ lost in extraction; restored)
- Shows vs infers: SHOWS. Relevant to EXP-2 variants B/C (independent L): TD predicts lost L–τ signal; our B2 and TD
  Table 8 measure the loss as small (E-CHEM-010, E-CHEM-052).
- Supports: EXP-2 §1, §5
- Confidence: high

### E-CHEM-009
- Claim: TD's matching = von-Mises cost matrix, linear-sum assignment, then differential evolution (DE) on torsions;
  the full assignment "guarantees" no L shift between training and inference.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 19, Appendix E, Algorithm 4
- Exact quote:
  > The complete assignment resulting from the linear sum solution guarantees that there is no distributional shift in the local structures seen during training and inference.
- Shows vs infers: SHOWS
- Reused from: E-ROBUST-021 (VERIFIED in round3/ledger/verify_V2.md); RE-CHECKED p. 19 today
- Supports: EXP-2 §3 (variant A)
- Confidence: high

### E-CHEM-010
- Claim: TD Table 8 (DRUGS): random pairing with DE only is marginally worse than full matching (COV-R 72.5 vs 72.7,
  AMR-R 0.588 vs 0.582); training on GT L collapses (34.8 / 0.920), close to random torsions (30.9 / 0.922).
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 27, Table 8 (text PDF p. 26, Appendix H items
  3 and 5)
- Exact quote (table rows):
  > Baseline 72.7 80.0 0.582 0.565 55.2 56.9 0.778 0.729
  > Only D.E. matching 72.5 81.1 0.588 0.569 53.8 56.1 0.794 0.749
  > Train on ground truth L 34.8 22.4 0.920 0.909 22.3 7.8 1.182 1.136
- Shows vs infers: SHOWS
- Reused from: E-TRAIN-024 (round 4, not yet verified) for the item-3 text; rows RE-CHECKED on p. 27 today
- Supports: EXP-2 §2, §5
- Confidence: high

### E-CHEM-011
- Claim: TD's own energy-based training (§3.6) is a matching-free objective: score matching on model samples
  importance-weighted toward the torsional Boltzmann density, instead of on data.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 7, §3.6 "Energy-based training"
- Exact quote:
  > At a high level, we minimize the usual score matching loss, but with simulated samples from the Boltzmann distribution rather than data samples.
- Shows vs infers: SHOWS (TD uses it as a Boltzmann generator at fixed T, DRUGS 3–7 rotatable bonds, §4.5). Using it as a
  regulariser next to data score matching is INFERENCE (EXP-2 D).
- Supports: EXP-2 §1 (variant D mechanism is TD's own), §3
- Confidence: high

### E-CHEM-012
- Claim: TD's energy-training importance weight is w̃_k = p̃_G(τ_k | L) / q_G(τ_k | L) for model samples τ_k.
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 7, Algorithm 1 "Energy-based training epoch"
- Exact quote (glyphs lost; Algorithm 1 lines as extracted):
  > Sample 1, . . . K  qGi ( | Li); for k  1 to K do w~k = p~Gi (k | Li)/qGi (k | Li); Approximate JDSM for p0  p~ using {(w~i, i)}i;
  (reads: sample τ_1..τ_K ∼ q_Gi(·|L_i); w̃_k = p̃_Gi(τ_k|L_i)/q_Gi(τ_k|L_i); approximate J_DSM for p_0 ≈ p̃)
- Shows vs infers: SHOWS
- Supports: EXP-2 §3 (variant D)
- Confidence: high

### E-CHEM-013
- Claim: TD's DSM loss on the torus uses the wrapped-normal perturbation kernel with weights λ(σ) = 1/E‖∇log p_{t|0}‖².
- Source type: paper
- Locator: paper: papers/core/2022_jing_torsional_diffusion.pdf, PDF p. 4, §3.1, Eq. (3)–(4)
- Exact quote:
  > Hence, the perturbation kernel for rescaled Brownian motion on Tm is the wrapped normal distribution on Rm
- Shows vs infers: SHOWS
- Supports: EXP-2 §3 (base loss of all variants); EXP-3 §3 (kernel to constrain)
- Confidence: high

## B. Our code (TD baseline)

### E-CHEM-020
- Claim: TD's rotatable-bond mask is purely topological: every bond whose removal disconnects the (H-explicit) graph
  and whose smaller side has > 1 atom is rotated; bond order is never read, so acyclic C=C, C=N, N=N (and C≡C) bonds
  with substituents on both ends are diffused like single bonds.
- Source type: code
- Locator: code: torsional-diffusion/utils/torsion.py:9-39 (get_transformation_mask; decisive lines 16-20)
- Exact quote (torsion.py:16-20):
  > G2 = G.to_undirected()
  > G2.remove_edge(*edges[i])
  > if not nx.is_connected(G2):
  > l = list(sorted(nx.connected_components(G2), key=len)[0])
  > if len(l) > 1:
- Shows vs infers: SHOWS. Used at training (utils/dataset.py:259) and inference (diffusion/sampling.py:58).
- Supports: EXP-3 §2, §6 (touchpoint for the torsion classifier / mask)
- Confidence: high

### E-CHEM-021
- Claim: The training-side matcher uses the same topological rule (get_torsion_angles), so conformer matching DE-fits
  double-bond torsions too.
- Source type: code
- Locator: code: torsional-diffusion/utils/standardization.py:69-89; called at torsional-diffusion/standardize_confs.py:74
- Exact quote (standardization.py:81-83):
  > if nx.is_connected(G2): continue
  > l = list(sorted(nx.connected_components(G2), key=len)[0])
  > if len(l) < 2: continue
- Shows vs infers: SHOWS
- Supports: EXP-3 §2; EXP-2 §6
- Confidence: high

### E-CHEM-022
- Claim: The model input carries no stereo information: edge features are the bond-type one-hot only (no E/Z / bond
  stereo), and atom chiral tags are computed but never put into the returned Data object.
- Source type: code
- Locator: code: torsional-diffusion/utils/featurization.py:19, :59, :63, :91, :101
- Exact quote:
  > bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
  > chiral_tag.append(chirality[atom.GetChiralTag()])
  > return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z=z)
- Shows vs infers: SHOWS. INFERENCE: for a double bond the network can learn that 0 and π are the modes but cannot
  know which one the requested stereoisomer needs, except through faint imprints in L. (S23D's statement that TD
  includes chirality in its features, E-CHEM-045, is therefore wrong for the released code; TD gets chirality from L.)
- Supports: EXP-3 §2, §3
- Confidence: high

### E-CHEM-023
- Claim: At inference TD discards the seed's torsions: every TD-rotatable torsion, double bonds included, is set to a
  uniform random value before reverse diffusion, so E/Z is re-decided by the model from scratch.
- Source type: code
- Locator: code: torsional-diffusion/diffusion/sampling.py:105-108 (perturb_seeds); called at
  torsional-diffusion/generate_confs.py (sample_confs, `if not args.no_random and n_rotable_bonds > 0.5`)
- Exact quote (sampling.py:107):
  > torsion_updates = np.random.uniform(low=-np.pi,high=np.pi, size=data_conf.edge_mask.sum())
- Shows vs infers: SHOWS
- Supports: EXP-3 §2, §3
- Confidence: high

### E-CHEM-024
- Claim: The reverse step is a plain geodesic random walk per torsion (perturb = g²·eps·score + g·√eps·z), applied
  through modify_conformer; there is no per-bond constraint hook.
- Source type: code
- Locator: code: torsional-diffusion/diffusion/sampling.py:189-204 and :234
- Exact quote (sampling.py:204, :234):
  > perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z
  > conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
- Shows vs infers: SHOWS
- Supports: EXP-3 §3, §6 (where the constrained operator goes)
- Confidence: high

### E-CHEM-025
- Claim: TD's training loss is the wrapped-normal DSM on the torsion noise actually applied (edge_rotate), normalised
  by score_norm(σ).
- Source type: code
- Locator: code: torsional-diffusion/utils/training.py:25-31; noise drawn at torsional-diffusion/utils/dataset.py:71-75
- Exact quote (training.py:31):
  > loss = ((score - pred) ** 2 / score_norm).mean()
- Shows vs infers: SHOWS
- Supports: EXP-2 §3 (base loss of every variant)
- Confidence: high

### E-CHEM-026
- Claim: TD ships a working energy-based training path: `--boltzmann_training` builds a BoltzmannResampler that draws
  `--boltzmann_confs` (default 32) ODE samples with `--boltzmann_steps` (default 5), computes MMFF energies and
  likelihoods, and sets importance weights −E/kT − log q at `--temp` (default 300 K).
- Source type: code
- Locator: code: torsional-diffusion/utils/boltzmann.py:22-56 (weights at :45-46); flags utils/parsing.py:59-64;
  train loop train.py:115-131, :169-184
- Exact quote (boltzmann.py:45-46):
  > energy = mol.mmff_energy
  > logweights.append(-energy / kT - mol.euclidean_dlogp)
- Shows vs infers: SHOWS
- Supports: EXP-2 §3, §6 (variant D reuses it)
- Confidence: high

### E-CHEM-027
- Claim: TD can already sample training conformers by CREST Boltzmann weight (`--boltzmann_weight`), i.e. a pure
  "training-distribution" change with the original objective, the control EXP-2 needs.
- Source type: code
- Locator: code: torsional-diffusion/utils/parsing.py:28; torsional-diffusion/utils/dataset.py:53-54
- Exact quote (dataset.py:54):
  > data.pos = random.choices(data.pos, data.weights, k=1)[0]
- Shows vs infers: SHOWS
- Reused from: E-TRAIN-015 (round 4, unverified); RE-CHECKED today (line numbers unchanged)
- Supports: EXP-2 §5 (distribution-only control)
- Confidence: high

### E-CHEM-028
- Claim: TD's evaluator filters GT conformers with NON-isomeric SMILES and scores with GetBestRMS, so a generated
  conformer with the wrong E/Z is not removed; it simply scores a large RMSD (and still counts toward K = 2L).
- Source type: code
- Locator: code: torsional-diffusion/evaluate_confs.py:47-54 (clean_confs), :115
- Exact quote (evaluate_confs.py:51, :115):
  > conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
  > rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
- Shows vs infers: SHOWS
- Supports: EXP-3 §4 (stereo must be reported separately; AMR alone hides it)
- Confidence: high

### E-CHEM-029
- Claim: In conformer matching the RDKit L pool is embedded from a copy of the GT GEOM mol, and the matched torsions
  (double bonds included) are DE-fitted to the GT conformer, so every training conformer carries the GT E/Z state.
- Source type: code
- Locator: code: torsional-diffusion/standardize_confs.py:73-79 (embedding), :104-107 (DE fit)
- Exact quote (standardize_confs.py:73, :79):
  > mol_rdkit = copy.deepcopy(confs[0]['rd_mol'])
  > AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=n_confs)
- Shows vs infers: SHOWS (code). "Every training conformer carries the GT E/Z" is INFERENCE from DE minimising RMSD
  over all rotatable torsions; not measured on the training pickles.
- Supports: EXP-3 §2
- Confidence: medium

## C. Our data

Scripts written for this round (read-only on their inputs; outputs next to them):
`round5/ledger/chem_data/stereo_census.py` (graph census; TD rule replicated, cross-check vs TD code with
`--check_td` on the cluster), `db_breakdown.py` (stereo double bonds by element pair / specification),
`ez_flip_analysis.py` (per-run E/Z vs chirality consistency, planarity, filtered AMR; cluster). Local runs used RDKit
2026.03.6. QM9 test SMILES were taken from `cluster_sync/results/analysis/local_structure_test.csv`
(997 unique corrected SMILES; 996 parse) and saved as `chem_data/qm9_test_smiles_from_local_structure_test.csv`.
DRUGS test SMILES: GeoMol repo `data/DRUGS/test_smiles_corrected.csv`
(https://raw.githubusercontent.com/PattanaikL/GeoMol/main/data/DRUGS/test_smiles_corrected.csv, accessed 2026-10-15,
1000 rows), saved as `chem_data/drugs_test_smiles_corrected_geomol.csv`; TD states it uses GeoMol's splits (TD PDF p. 8,
"We use the train/val/test splits from Ganea et al. [2021]"), so this is TD's DRUGS test set (identity of the file with
TD's `data/DRUGS/test_smiles.csv` not byte-checked: the 25 GB archive was not downloaded).

### E-CHEM-050
- Claim: QM9 test set (996 molecules): 103 (10.3 %) have ≥ 1 potential stereo double bond (RDKit FindPotentialStereo),
  67 with the stereo specified in the corrected SMILES; every acyclic one is a C=N bond (106 bonds; 70 specified, 36
  unspecified); no acyclic C=C stereo bond exists in the test set. 100 molecules have a stereo double bond inside TD's
  rotatable set (106 bonds = 3.7 % of 2,880 TD-rotatable bonds). Tetrahedral centres: 745 molecules (74.8 %), 1,945
  centres.
- Source type: our-data
- Locator: our-data: round5/ledger/chem_data/census_qm9_local.out (lines 2, 4, 5, 6) and chem_data/db_breakdown_qm9.out (lines 2-3)
- Exact quote:
  > mols with >=1 potential stereo double bond: 103 (10.3%); with >=1 SPECIFIED in SMILES: 67; stereo DB in rings: 6 bonds
  > double_stereo:    106 bonds (3.7%), in 100 mols
  > ('CN', 'specified') 70
  > ('CN', 'unspecified') 36
- Shows vs infers: SHOWS (graph-level counts; the TD-rule replication is cross-checked against TD's own mask in
  the cluster run, E-CHEM-063 when it lands)
- Supports: EXP-3 §2 (QM9 is the wrong test bed for E/Z: ~100 molecules, all C=N)
- Confidence: high

### E-CHEM-051
- Claim: GEOM-DRUGS test set (1000 molecules): 142 (14.2 %) have ≥ 1 potential stereo double bond (163 bonds; 144
  specified); acyclic ones are 58 C=C (all specified), 96 C=N (80 specified) and 3 N=N; 141 molecules have one inside
  TD's rotatable set (157 bonds = 2.0 % of 7,965). Tetrahedral centres: 299 molecules (29.9 %), 461 centres.
- Source type: our-data
- Locator: our-data: round5/ledger/chem_data/census_drugs.out (lines 2-6) and chem_data/db_breakdown_drugs.out (lines 1-5)
- Exact quote:
  > mols with >=1 potential stereo double bond: 142 (14.2%); with >=1 SPECIFIED in SMILES: 127; stereo DB in rings: 6 bonds
  > double_stereo:    157 bonds (2.0%), in 141 mols
  > ('CC', 'specified') 58
- Shows vs infers: SHOWS
- Supports: EXP-3 §2, §4 (DRUGS gives ~140 E/Z molecules per test set: enough for a paired test, not a large one)
- Confidence: high (counts); medium (file identity with TD's csv, see header)

### E-CHEM-052
- Claim: Matching variants already run on QM9 (3 v 3 seeds, paired vs CTRL_rematch): random L–conformer pairing with
  DE only (B2) +0.0021 Å [0.0006, 0.0037]; heavy-atom DE objective (B5) +0.0058 Å [0.0038, 0.0080].
- Source type: our-data
- Locator: our-data: cluster_sync/results/analysis/paired_B_vs_CTRLrematch.md lines 5 and 15
- Exact quote:
  > | B2    | MAT-R      | 934 |  0.1786 |      0.1808 |  0.0021 |  0.0006 |  0.0037 |   0.0096 |             0.0013 | True      |             3 |             3 |   0.0288 |
  > | B5    | MAT-R      | 934 |  0.1786 |      0.1845 |  0.0058 |  0.0038 |  0.008  |   0      |             0.0013 | True      |             3 |             3 |   0      |
- Shows vs infers: SHOWS
- Reused from: E-TRAIN-011 (round 4, unverified); RE-CHECKED against the file today
- Supports: EXP-2 §2 (variant B ≈ B2 already exists)
- Confidence: high

### E-CHEM-053
- Claim: Training without conformer matching on true L (B1) is +0.0536 Å [0.0469, 0.0600] worse than CTRL_base on
  RDKit L (QM9, 3 v 3 seeds), the QM9 analogue of TD Table 8 "Train on ground truth L".
- Source type: our-data
- Locator: our-data: cluster_sync/results/analysis/paired_B1_vs_CTRLbase.md line 5
- Exact quote:
  > | B1           | MAT-R      | 934 |  0.1806 |      0.2342 |   0.0536 |   0.0469 |   0.06   |   0      |             0.0013 | True      |             3 |             3 |   0      |
- Shows vs infers: SHOWS
- Supports: EXP-2 §2
- Confidence: high

### E-CHEM-054
- Claim: Leak-free split-half ORACLE (743 molecules): with another conformer's true L, CTRL_rematch scores AMR-R
  0.2195 vs 0.1873 with RDKit L; B1 (no matching, true L) 0.3046 vs 0.2565. True L that is not coupled to the target
  conformer helps no model.
- Source type: our-data
- Locator: our-data: round4/leakfree_data/leakfree_summary.txt lines 15-16 (columns split AMR-R and ref_rdkitL AMR-R)
- Exact quote:
  > CR      0.2195    0.8564    0.5019  0.1603    0.8891    0.6304 -0.0593 ...     0.1873    0.9447    0.3438 ...
  > B1      0.3046    0.7373    0.4002  0.0303    0.9911    0.9512 -0.2743 ...     0.2565    0.8670    0.2297 ...
  (rows truncated with "..."; the full rows are in the file)
- Shows vs infers: SHOWS (ORACLE). INFERENCE for EXP-2: any matching-free variant must keep the training L
  distribution equal to the deployed one; pairing L with targets it was not built for (the split-half condition) is
  exactly what hurts.
- Supports: EXP-2 §2, §5
- Confidence: high

### E-CHEM-055
- Claim: Assigning stereochemistry from the true 3D structure to the seeds (G1) does not change TD's AMR-R on QM9:
  +0.0001 Å [−0.0035, 0.0038] vs R0 (3 v 3 seeds, 922 molecules).
- Source type: our-data
- Locator: our-data: cluster_sync/results/analysis/paired_vs_R0_qm9_default.md line 65
- Exact quote:
  > | G1       | MAT-R      | 922 |  0.1753 |      0.1755 |   0.0001 |  -0.0035 |   0.0038 |   0.9964 |             0.0008 | True      |             3 |             3 |   1      |
- Shows vs infers: SHOWS. INFERENCE: G1 changes the seed's stereo tags, but TD re-randomises every double-bond torsion
  (E-CHEM-023) and the model has no E/Z input (E-CHEM-022), so G1 could not fix E/Z even in principle; it only
  confirms chirality tags were already right.
- Supports: EXP-3 §2
- Confidence: high

### E-CHEM-056
- Claim: In round-1 stereo checks (cluster RDKit, which flagged only 37–38 QM9 test molecules as having stereo double
  bonds), 42–50 % of those molecules had at least one generated conformer whose 3D stereo matched no GT conformer
  (mean fraction of matching conformers 0.85–0.90), against 4.6–6.8 % for molecules with chiral centres; G1 (true
  stereo on seeds) did not improve it (0.459 vs R0 0.421).
- Source type: our-data
- Locator: our-data: cluster_sync/logs/tordiff_analysis_1089.log lines 28-42 (blocks in loop order R0, A1, G1, A2 of
  slurm/analysis_qm9.sbatch:69)
- Exact quote:
  > with stereo double bonds: n=   38  mean frac of generated confs whose 3D stereo matches a GT conformer = 0.878; mols with ANY mismatch = 0.421
  > with stereo double bonds: n=   37  mean frac of generated confs whose 3D stereo matches a GT conformer = 0.895; mols with ANY mismatch = 0.459
  > with chiral centers: n=  687  mean frac of generated confs whose 3D stereo matches a GT conformer = 0.985; mols with ANY mismatch = 0.048
- Shows vs infers: SHOWS. Caveat: the label compared is the full isomeric SMILES (chirality + E/Z together), so the
  stereo-DB rows also include chirality mismatches; E-CHEM-062 separates them.
- Supports: EXP-3 §2
- Confidence: high (numbers); medium (attribution to E/Z until E-CHEM-062)

## D. Prior art: matching-free / distribution / energy objectives (Exp 2)

### E-CHEM-030
- Claim: GeoMol (2021) already trains a torsional conformer-ensemble model without any per-conformer pairing: an
  OT-based, minimisation-only ensemble loss between generated and ground-truth conformer sets (EMD transport plan).
- Source type: paper
- Locator: paper: papers/related/2021_ganea_geomol.pdf, PDF p. 7, §2.5 "An optimal transport (OT) loss function for
  diverse conformer generation"
- Exact quote:
  > we do not know a priori the number L of true conformers or the matching between generated and true conformers.
  > Our solution is an OT-based, minimization-only, loss function:
  (two-column page: contiguous in the left column of `pdftotext -layout -f 7 -l 7`, lines 63-64; the reading-order
  fulltext interleaves the Figure 6 caption)
- Shows vs infers: SHOWS. This is the direct prior art for EXP-2 variant C's set-level distribution loss (GeoMol is a
  one-shot generator; TD-style score models have not used it — INFERENCE from the papers in papers/INDEX.md, no
  counter-example found in the 2023–2026 search).
- Supports: EXP-2 §1 (novelty of C), §3
- Confidence: high

### E-CHEM-031
- Claim: Distributional Diffusion Models (ICML 2025) replace the diffusion regression loss with a (generalised) energy
  score / kernel scoring rule so that a generator learns the full posterior p(x0 | xt), not its mean.
- Source type: paper
- Locator: paper: papers/related/2025_debortoli_distributional_diffusion.pdf, PDF p. 3, §3
- Exact quote:
  > At a high level, we replace the regression loss (6), used to learn approximation of the conditional mean E[X0|Xt] in diffusion models, with a loss based on scoring rules
- Shows vs infers: SHOWS (Euclidean data, images/robotics). Transfer to the torus with a wrapped distance is INFERENCE.
- Supports: EXP-2 §3 (variant C loss family)
- Confidence: high

### E-CHEM-032
- Claim: Adjoint Sampling (2025) trains a torsional conformer sampler conditioned on bond lengths and angles from an
  energy model alone, with no conformer data and hence no conformer matching.
- Source type: paper
- Locator: paper: papers/related/2025_havens_adjoint_sampling.pdf, PDF p. 12, §5.2
- Exact quote:
  > Torsional Adjoint Sampling (Appendix F.2): We sample only the torsion angles in the molecule, conditioned on elements, bond lengths, and bond angles (3-tuples).
  > This is possible because Adjoint Sampling does not require any data during training, i.e., we do not use the atomic configurations!
- Shows vs infers: SHOWS (SPICE/eSEN; GEOM-DRUGS subset evaluated at δ = 1.25 Å, not TD-comparable)
- Supports: EXP-2 §1 (variant D's "matching-free, energy-aware torsion model" exists as a data-free method)
- Confidence: high

### E-CHEM-033
- Claim: iDEM (ICML 2024) learns a diffusion sampler for an unnormalised Boltzmann density from a replay buffer and an
  energy-derived score estimate, without data.
- Source type: paper
- Locator: paper: papers/related/2024_akhoundsadegh_idem.pdf, PDF p. 1, Figure 1 caption
- Exact quote:
  > iDEM fits a diffusion sampler to a target distribution given by an unnormalized density.
- Shows vs infers: SHOWS (toy n-body systems; Adjoint Sampling p. 8 reports it needs O(100,000) more energy
  evaluations per update, E-CHEM-034)
- Supports: EXP-2 §1
- Confidence: high

### E-CHEM-034
- Claim: Adjoint Sampling reports that importance-weighted / MC-score energy samplers need far more energy evaluations
  per gradient update (iDEM O(100,000) more), which bears on TD-style importance-resampled training cost.
- Source type: paper
- Locator: paper: papers/related/2025_havens_adjoint_sampling.pdf, PDF p. 11, §5.1 "Results"
- Exact quote:
  > iDEM uses O(100, 000) more energy evaluations than Adjoint Sampling per gradient update since it relies on Monte Carlo (MC) estimation of the ground truth score function.
- Shows vs infers: SHOWS
- Supports: EXP-2 §7 (variant D cost risk)
- Confidence: high

### E-CHEM-035
- Claim: EnFlow (2025) adds an explicit energy model trained by Energy Matching plus supervised fine-tuning on GEOM
  energies, and uses it to guide few-step flow sampling; it avoids contrastive-divergence negative sampling.
- Source type: paper
- Locator: paper: papers/related/2025_xu_enflow.pdf, PDF p. 12 (Methods, "Energy Matching"); PDF p. 3 (Introduction)
- Exact quote:
  > We train the energy model using the Energy Matching objective [26], which connects energy-based model training with optimal transport
  > conventional energy-based model training, such as contrastive divergence, requires expensive and often unstable negative sampling.
- Shows vs infers: SHOWS (Cartesian flow model; energy enters as guidance and ranking, not as a training-distribution
  correction of a torsion model)
- Supports: EXP-2 §1 (variant D novelty bound)
- Confidence: high

### E-CHEM-036
- Claim: Torsional-GFN learns p(τ | G, L) from an energy reward with L from MD (not tied to a target conformer) and
  generalises zero-shot to unseen bond lengths and angles: a matching-free, energy-trained torsion model.
- Source type: paper
- Locator: paper: papers/related/2025_ezzine_torsional_gfn.pdf (ICML 2025 GenBio workshop), PDF p. 1, Abstract
- Exact quote:
  > allows for zero-shot generalization to unseen bond lengths and an- gles coming from the MD simulations for such molecules
- Shows vs infers: SHOWS (small MD set, MMFF94s energy)
- Reused from: E-TRAIN-046 (round 4, unverified); RE-CHECKED: quote is contiguous in the left column of
  `pdftotext -layout -f 1 -l 1` (two-column page)
- Supports: EXP-2 §1
- Confidence: medium (workshop paper)

### E-CHEM-037
- Claim: Data-free and data-light Boltzmann approaches are now common for torsions (TD §3.6/§4.5, Torsional-GFN,
  Torsional Adjoint Sampling), so "matching-free + energy-aware torsion model" is not new as a mechanism; what is
  untested is whether a data-plus-energy objective beats conformer matching on the TD recall protocol (AMR-R / COV-R vs
  CREST ensembles).
- Source type: paper (synthesis of E-CHEM-011, -032, -036)
- Locator: as in E-CHEM-011 (TD PDF p. 7), E-CHEM-032 (Adjoint Sampling PDF p. 12), E-CHEM-036 (Torsional-GFN PDF p. 1)
- Exact quote: see the cited entries
- Shows vs infers: INFERENCE (novelty ruling)
- Supports: EXP-2 §0 novelty verdict
- Confidence: medium

## E. Prior art: stereo-aware conformer generation (Exp 3)

### E-CHEM-040
- Claim: LoQI (2025/26, all-atom diffusion on Megalodon) conditions on E/Z by adding auxiliary graph edges between the
  stereo-reference atoms of each stereo double bond (edge classes 5/6 for E/Z; 7/8 for tetrahedral handedness), i.e.
  TD's suggested "special edges" fix, implemented.
- Source type: blog (code)
- Locator: blog: https://github.com/isayevlab/LoQI, commit 04f3caef (2026-10-01), file src/megalodon/data/stereo.py
  lines 1-6 and 177-232 (add_stereo_bonds); accessed 2026-10-15; snapshot papers/blogs/loqi_repo_stereo.txt
- Exact quote (stereo.py:3-4):
  > LoQI uses edge classes 5/6 for E/Z relationships and 7/8 for tetrahedral
  > handedness.
- Shows vs infers: SHOWS (code). The paper PDF could not be fetched (ChemRxiv and OpenReview return 403 to automated
  download, papers/related/2025_nikitin_loqi.md); its reported stereo accuracies (0.956 R/S, 0.992 E/Z in a search
  snippet) are UNVERIFIED.
- Supports: EXP-3 §1 (E/Z conditioning by graph edges is PRIOR ART)
- Confidence: high (code), low (paper numbers)

### E-CHEM-041
- Claim: LoQI's own stereo metric (StereoMetrics) scores a generated molecule E/Z-correct iff its 3D-perceived ordered
  E/Z descriptor string equals the reference's; R/S is counted correct up to global reflection.
- Source type: blog (code)
- Locator: blog: https://github.com/isayevlab/LoQI, commit 04f3caef, src/megalodon/metrics/preserved_stereo.py:47-79;
  accessed 2026-10-15; snapshot papers/blogs/loqi_repo_stereo.txt
- Exact quote:
  > if sr == ref_sr or sr == inv_sr:
  > if ez == ref_ez:
- Shows vs infers: SHOWS. Our per-conformer E/Z check (ez_flip_analysis.py, canonical DB-only SMILES vs the GT set)
  is a stricter, symmetry-safe variant of the same idea.
- Supports: EXP-3 §4 (metric definition, comparability with LoQI)
- Confidence: high

### E-CHEM-042
- Claim: LoQI's README (v1.1.0) scopes its stereo conditioning: tetrahedral control only for centres with four explicit
  neighbours; E/Z edges unchanged.
- Source type: blog (code docs)
- Locator: blog: https://github.com/isayevlab/LoQI README.md lines 33-37 and 150-155, commit 04f3caef, accessed
  2026-10-15; snapshot papers/blogs/loqi_repo_stereo.txt
- Exact quote:
  > stereo conditioning currently supports only tetrahedral centers with four explicit neighbors.
- Shows vs infers: SHOWS
- Supports: EXP-3 §1
- Confidence: high

### E-CHEM-043
- Claim: Swanson et al. (VonMisesNet, ICML 2023) deliberately use RDKit's canonical rotatable-bond definition instead
  of TD's, because it excludes double and triple bonds that may not be freely rotatable.
- Source type: paper
- Locator: paper: papers/related/2023_swanson_vonmises.pdf, PDF p. 2, §1 (right column)
- Exact quote:
  > although we use the canonical rotatable bond definition because it excludes some bonds, such as double or triple bonds, that may not be freely rotatable.
  (two-column page; contiguous in the right column)
- Shows vs infers: SHOWS
- Supports: EXP-3 §1 (excluding double bonds from the torsion set is PRIOR ART), §3 (variant "freeze")
- Confidence: high

### E-CHEM-044
- Claim: Swanson et al. feed RDKit bond-stereo type as a bond feature and enforce consistent double-bond stereo on
  ETKDG embeddings ("ETKDG-Clean": unspecified stereo double bonds are tagged cis so ETKDG keeps them).
- Source type: paper
- Locator: paper: papers/related/2023_swanson_vonmises.pdf, PDF p. 16, Table 3 and Appendix I
- Exact quote:
  > STEREO BOND STEREO TYPE, ONE HOT ENCODED FROM RDKIT BONDSTEREO TYPES 6
  > For each double bond where the bond does not have a cis/trans stereochemistry tag and both endpoint atoms have more than one neighbor, we set the double bond tag to cis in a consistent manner.
- Shows vs infers: SHOWS
- Supports: EXP-3 §1, §3 (bond-stereo features are prior art)
- Confidence: high

### E-CHEM-045
- Claim: S23D (2025) reports that ET-Flow corrects chirality post hoc and that chirality handling materially changes
  benchmark results; it states TD uses chirality in its features (which our code shows is not true for the released
  TD code, E-CHEM-022; TD gets chirality from RDKit L).
- Source type: paper
- Locator: paper: papers/related/2025_gurev_s23d.pdf, PDF p. 5, "Molecule chirality"
- Exact quote:
  > for optimal performance it may be important to include chirality in the feature set of 2D molecular graphs, as was done in Torsional Diffusion and ET-FLow.
- Shows vs infers: SHOWS (their statement). That it is wrong for TD is our code reading (E-CHEM-022).
- Supports: EXP-3 §1, §4 (stereo handling must be stated explicitly for fair comparison)
- Confidence: high

### E-CHEM-046
- Claim: ET-Flow and GeoMol fix tetrahedral chirality post hoc: compare the oriented volume with RDKit tags and invert
  the conformer if it mismatches (no E/Z handling).
- Source type: paper
- Locator: paper: papers/related/2024_hassan_etflow.pdf, PDF p. 5, §3.4 "Chirality Correction"
- Exact quote:
  > To generate conformations with the correct chirality, we propose a simple post hoc trick as done in GeoMol (Ganea et al., 2021).
  > In the case of a mismatch, we simply flip the conformation against the z-axis.
  (two-column page: left column of `pdftotext -layout -f 5 -l 5`, lines 36-43; "con- formations" is a line-break
  hyphen and the right column's Figure text interleaves "post | hoc")
- Shows vs infers: SHOWS
- Supports: EXP-3 §1 (post-hoc stereo filtering/correction is PRIOR ART for chirality; for E/Z, a post-hoc torsion
  flip/filter is the obvious analogue)
- Confidence: high

### E-CHEM-047
- Claim: Search (2023–2026, 2026-10-15) found no torsional-diffusion / flow model that constrains E/Z double-bond
  torsions inside the reverse process (masked or half-circle kernel), and no published E/Z flip rate for TD.
- Source type: paper (search record)
- Locator: WebSearch queries run 2026-10-15 ("diffusion conformer generation E/Z isomerism double bond
  stereochemistry constraint torsion 2024 2025"; "stereoisomer-aware molecular conformer generation cis/trans flips
  ..."; "LoQI ... stereo"); local fulltexts grepped for stereo/E/Z/cis (papers/related/fulltext/*.txt)
- Exact quote: n/a (absence of evidence)
- Shows vs infers: INFERENCE (absence in a limited search; the novelty red-teamer should repeat it). A search snippet
  attributes to LoQI's v2 PDF the statement that TD's open-source code often produces incorrect stereoisomers;
  UNVERIFIED (PDF blocked).
- Supports: EXP-3 §0 novelty verdict
- Confidence: low–medium

## F. RDKit stereo perception

### E-CHEM-070
- Claim: RDKit's rule for a potentially stereogenic double bond: both atoms have ≥ 2 heavy-atom neighbours and the bond
  is not in a ring of fewer than eight atoms. Hence C=N–H imines and terminal =CH2 are not stereo bonds, and ring
  double bonds in rings < 8 (never TD-rotatable anyway) are excluded.
- Source type: blog (official documentation)
- Locator: blog: https://www.rdkit.org/docs/RDKit_Book.html, RDKit developers, section "Stereogenic atoms/bonds" →
  "Stereogenic bonds", accessed 2026-10-08 (snapshot by round-3 scout GEOM), snapshot papers/blogs/rdkit_book.txt
  lines 2504-2508
- Exact quote:
  > A double bond is potentially stereogenic if both atoms have at least two heavy atom neighbors and it’s not present in a ring with less than eight atoms.
- Shows vs infers: SHOWS. Our census uses FindPotentialStereo, which implements these definitions (same section,
  snapshot lines 2458-2462, 2533-2545).
- Supports: EXP-3 §2 (definition of "stereo double bond" in all counts), §3 (torsion classifier)
- Confidence: high

### E-CHEM-071
- Claim: RDKit perceives stereochemistry from 3D coordinates when a 3D conformer is given, which is how generated and
  GT conformers are labelled (AssignStereochemistryFrom3D).
- Source type: blog (official documentation)
- Locator: blog: https://www.rdkit.org/docs/RDKit_Book.html, section "Interpretation of the 2D/3D Flag and
  Stereochemistry", snapshot papers/blogs/rdkit_book.txt line 1888
- Exact quote:
  > When 3D coordinates are provided, stereochemistry is perceived from the
  (line-wrapped in the snapshot; the sentence continues on line 1889)
- Shows vs infers: SHOWS (mol-file context); our use of AssignStereochemistryFrom3D for per-conformer labels is the
  standard API for the same perception
- Supports: EXP-3 §4 (per-step / per-sample stereo verification)
- Confidence: medium (the quoted sentence concerns mol-file parsing)
