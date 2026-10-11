# SC4020 Project 1: BPR-MF vs. LightGCN under unbiased evaluation

Course project (NTU SC4020), due **18 Oct 2026 (week 9)**. Team of 4.

## Hard rules (from the user)
- **No AI attribution anywhere.** Do not add Claude/Anthropic mentions, "Co-Authored-By" lines, "Generated with" footers, or any note of AI contribution in code, comments, notebooks, README, commit messages, PR text, reports or PDFs.
- `CLAUDE.md` and `LOG.md` may be committed and pushed to the team repository (user allowed this on 2026-09-30). Keep them out of the submission `group_XX_code.zip`.
- Do not implement, run heavy jobs, or commit unless asked; plan first when the user says so.

## Goal
Compare BPR-MF and LightGCN on user-item bipartite graphs and test whether LightGCN's advantage holds when exposure bias is removed from the evaluation (decided 2026-10-01).

- **Working title:** *Does LightGCN's Advantage Survive Unbiased Evaluation? BPR-MF vs. LightGCN on KuaiRand-Pure and KuaiRec*
- **Story (agreed 2026-10-01):** short-video feeds are spreading from dedicated apps into e-commerce and social platforms. In a feed the app pushes one item at a time and the user rarely searches, so the recommender decides almost everything the user sees and exposure bias is at its strongest. Short-video data is therefore the natural setting for the question, not a limitation. Scope: both datasets come from one platform, so results apply to feed-style recommendation; do not claim they generalise to e-commerce or product search. Practical takeaway for the Conclusion: before deploying a graph recommender in a feed, evaluate it on unbiased data where available and against a most-popular baseline.
- **Research question:** most test sets contain only items that the platform's own recommender chose to show, so they reward models that recommend what was already popular. Both datasets also contain feedback the recommender did not select. Does LightGCN still beat BPR-MF on that feedback, and by how much compared with a most-popular baseline?
- **Methods (2):** BPR-MF (no graph) and LightGCN (He et al., SIGIR 2020). LightGCN with 0 layers is BPR-MF, so the graph is the only difference.
- **Baseline (required):** most-popular (rank items by training degree). It anchors both tests: it is strong on the biased test and weak on KuaiRand's unbiased test.
- **Datasets (2, both Kuaishou short videos):** KuaiRand-Pure (unbiased test = randomly exposed videos) and KuaiRec (unbiased test = a fully observed block of users and videos).
- **Metrics:** Recall@20 and NDCG@20 on the biased test and on the unbiased test, margin over most-popular, item coverage, popularity of recommendations, training time. On both unbiased tests also GAUC and NDCG@5 over each user's shown videos (exposure-only protocol, decided 2026-10-11, see Conventions).
- **Secondary analyses:** popularity and user-degree breakdown on both tests (07), density sweep (08), ablation (05), layer study (04).

## Deliverables (submit 3 files)
1. `group_XX_report.pdf`: at least 4 pages, 11 or 12 pt font.
2. `group_XX_contribution.pdf`: what each member contributed.
3. `group_XX_code.zip`: source code only. **No datasets, no trained models, no external links.**

Submission rules (from the project brief):
- One submission per group on NTULearn (Assignments -> project 1). Resubmitting is allowed. Submitter: the group coordinator (to be confirmed, see Open decisions).
- **No URLs anywhere in the zip.** The repository `README.md`, `requirements.txt` and `data/README.md` contain links (arXiv, PyTorch wheel index, Zenodo, GitHub). The zip gets its own link-free `README.md` that names the papers and datasets in plain text and gives the install command without a URL; `requirements.txt` in the zip has its URL comment removed. Before zipping, search every file in the package for `http`, including binary files (`grep -a`), and confirm there are no hits.
- The zip README must not point to `data/README.md`, because `data/` is excluded. It states in plain text which datasets to obtain and the file names expected under `data/`.
- Contribution issues had to be reported to the instructor 4 weeks before the deadline (20 Sep 2026). That date has passed, so the team resolves any contribution problem itself.

## Report structure (required headings)
The brief requires these five sections. Owners in brackets.
1. **Abstract** (4, written last from the final numbers)
2. **Introduction** (1): short-video feeds and why evaluation matters there (cite 1-2 real sources for feeds moving into shopping apps), task, exposure bias, research question, related work
3. **Methods**: BPR-MF (2), LightGCN (3), most-popular baseline (4)
4. **Experiments**
   - Datasets and evaluation protocol, including the two unbiased test sets (1)
   - Parameter settings: MF (2), LightGCN (3)
   - Main comparison: biased vs. unbiased test, both datasets (4) — the headline result
   - Ablation (4)
   - Key factors: item popularity and user degree on both tests (3); density sweep (1)
   - Successful and failure cases (3)
   - Strengths and weaknesses (4)
   - Possible improvements (3 drafts, 4 reviews)
5. **Conclusion** (2): answer to the research question, scope (one platform, feed-style recommendation), practical takeaway for deploying a recommender in a feed

References follow the Conclusion as plain citations.

## Required content of the report
- Strengths and weaknesses of both methods.
- Parameter settings (embedding dim, learning rate, L2, number of layers).
- Ablation study of LightGCN components (layer combination mean vs. last, normalization, edge dropout).
- Key factors affecting performance (exposure bias, popularity, user degree, density).
- Performance comparison with discussion: every model on both tests of both datasets.
- Successful and failure cases (recall by user degree group and by item popularity, example users).
- All results are produced by our own runs; nothing is copied from published papers.

## Rubric items beyond the required content
- **Possible solutions to improve the methods (discuss or implement).** Minimum: a discussion subsection grounded in the failure cases from notebook 07. Target: one small implemented fix, chosen after 07 shows where the models fail. Candidates: popularity-aware negative sampling (against popularity bias, fits the research question), or learned layer weights instead of the mean (against over-smoothing). Owner: 3, reviewed by 4. Any implemented fix follows the fairness rules and is reported as a separate row, not mixed into the main comparison.
- **Good visualizations.** Required figures, each saved to `results/` by its notebook:

  | Figure | Notebook | Owner |
  |---|---|---|
  | User and item degree distributions, both datasets (done) | 01 | 1 |
  | Recall@20 vs. number of layers (over-smoothing curve), both tests | 04 | 3 |
  | Ablation deltas, both datasets | 05 | 4 |
  | **Headline:** most-popular, BPR-MF, LightGCN on the biased and unbiased test, both datasets | 06 | 4 |
  | Recall@20 by item-popularity bucket and by user degree group, both tests | 07 | 3 |
  | Recall@20 vs. training density, both models and datasets | 08 | 1 |

- **More than 2 methods is encouraged.** The most-popular baseline is now required (owner 4). ItemKNN stays optional, only if time allows after 11 Oct.
- **Challenging datasets.** Both datasets are sparse real-world logs with exposure bias, and both have an unbiased test set; few course projects can test this.
- **Clear explanation of the task and methods; well organized.** Covered by the report structure above and notebook 00.

## Repository layout
```
CLAUDE.md          this file: conventions and status
LOG.md             running log of every step (append only)
src/               all Python code (importable modules and scripts)
notebooks/         one .ipynb per step, numbered, with outputs kept
results/           JSON/CSV metrics and figures produced by experiments
data/              raw datasets, NOT committed and NOT in the zip
```

### src/
One package per area so each member adds files in their own folder. Notebooks put `src/` on `sys.path` and use `import data, models, training, inference`; packages import each other the same way (`from data import to_pairs`). Each package's `__init__.py` re-exports its public names, so add a new module and export it there.

| Folder | Owner | Contents |
|---|---|---|
| `data/` | 1 | `loaders.py` (paths, `load`), `splits.py` (split rules, unbiased targets, `to_pairs`), `transforms.py` (`sparsify`, `k_core`), `stats.py` (`degrees`, `stats`), `store.py` (`save_processed`, `load_processed`) |
| `inference/` | 1 | `evaluate.py` (`evaluate`; `evaluate_exposed`: GAUC and NDCG@5 over each user's shown videos on `test_unbiased`) |
| `models/` | 2 (MF), 3 (LightGCN), 4 (most-popular) | `core.py`: `MF`, `LightGCN`, `build_norm_adj`, `SpMM` (moved unchanged from `models.py`); `most_popular.py`: `MostPopular` (training-degree scores, matches notebook 01's preview numbers). New models go in their own module. |
| `training/` | 2 | `train.py`: `fit`, `make_model` (moved from `train.py`; `evaluate` now imported from `inference`) |
| `algorithms/` | 2, 3 | `negative_sampling.py`: popularity-aware negative sampling (`popularity_weights`, `sample`; the improvement fix, `neg_alpha` cfg key in `fit`) |
| `ablations/` | 4 | empty: ablation configs, one change at a time |
| `experiments/` | 4 | `run_grid.py`: cached parameter grids for notebooks 03/04 (per-dataset JSONs in `results/`, one key per dataset+config+seed, resumes after interruption; standalone CLI, `select_best` on validation). `density_sweep.py` (owner 1): notebook 08's sweep, both models at the 03/04 selected settings on `keep=<frac>` and `core=<k>` variants, one cached JSON per dataset and variant (`results/08_density_<ds>_<variant>.json`), standalone CLI so chains run in parallel processes. Also the 3-seed runs of 05/06 (`SELECTED`, `ABLATIONS`, `--seeds`, results store `test_exposed`). |
| `analysis/` | 3 | empty: degree and popularity groups, recall breakdowns, figures |
| `utils/` | shared | `paths.py` (`ROOT`, `DATA_DIR`, `RESULTS_DIR`), `io.py` (`save_json`, `load_json`, `savefig` with the no-URL metadata) |

- `data/` (owner 1): `load(name, seed)` for `"kuairand"` and `"kuairec"`. Returns `dict(name, n_users, n_items, train, val, test, test_unbiased, unbiased_exposed, unbiased_items)`; each split is `{user: np.array(items)}`. Both have `test_unbiased` and `unbiased_exposed` (every video each user was shown in the unbiased exposure, clicked or not); `unbiased_items` (the block videos) only for KuaiRec. Users with fewer than 2 positives are dropped and ids are remapped. Raw positives are cached per process, so loading several seeds is fast after the first call (KuaiRec takes about 25 s the first time). Also `sparsify(ds, keep_frac, seed)`, `k_core(ds, k)`, `degrees(ds)`, `stats(ds)`. Notebook 02 saves the result of `load` for seeds 0-2 to `data/processed/<name>_seed<seed>.npz` (not in git); `load_processed(name, seed)` reads it back into the identical dict in under a second.
- `models/core.py` (owner 3 for LightGCN, 2 for MF): `MF` and `LightGCN` (layers, `combine` = mean or last, `norm` = sym, left or none, edge dropout). LightGCN propagates with a CSR adjacency and `SpMM`, an autograd function whose backward uses a CSR copy of the transpose (15 times faster on CPU than the default; gradients checked).
- `training/train.py` (owner 2) and `inference/evaluate.py` (owner 1): BPR training loop with early stopping on validation Recall@20. `fit` scores the best model on `test` and, when the dataset has one, on `test_unbiased`. `evaluate` returns Recall, NDCG, coverage, `popularity` (mean training degree of recommended items) and per-user recall; on KuaiRec's `test_unbiased` it ranks only among `unbiased_items`.
- Planned: multi-seed runner in `experiments/` (owner 4), ablation configs in `ablations/` (owner 4), degree and popularity groups and figures in `analysis/` (owner 3). Done 2026-10-10: most-popular baseline (`models/most_popular.py`, reproduces notebook 02's preview numbers) and the cached parameter-grid runner (`experiments/run_grid.py`).

**Status (2026-10-10, evening): the parameter studies are done and merged. Notebooks 00-04 are complete with outputs; the selected settings are BPR-MF dim=64 lr=1e-3 reg=1e-3 (KuaiRand) / dim=64 lr=3e-4 reg=1e-4 (KuaiRec) and LightGCN layers=3 dim=128 on both. Fairness rule 2 restored at 8-vs-8: notebook 04 selects over an 8-configuration grid (lr fixed to the MF-selected 1e-3), and its 8 lr=5e-3 runs are labelled sensitivity — they ranked strictly lower and change no selection (`results/03_mf_parameters.json`, `03_mf_validation.png`, `results/04_*` including the Colab-produced `.colab.json` files and `04_layers.png`). Notebook 07 is complete on the `member3` branch; 05 and 06 are stubs; 08 has code, not run. Next: 05 (ablation), 06 (3-seed headline with most-popular), 08 (density sweep).**

**Status (2026-10-12): notebooks 05 and 06 are complete (3 seeds, both datasets, both tests, `results/05_*`, `results/06_*`).** LightGCN wins the biased test (+29% / +19% Recall@20) but not the unbiased one: within noise on KuaiRand-Pure's GAUC, reversed on KuaiRec (BPR-MF 0.776 > most-popular 0.761 > LightGCN 0.736). 06 adds a LightGCN dim-64 control row for fairness rule 3. The multi-seed runner is `experiments/run_grid.py` (`SELECTED`, `GRIDS["05"]`/`["06"]`, `--seeds`); the most-popular module, deleted from main by the PR #1 merge, is restored. 08 (member 1) and the improvement fix (member 3) are also merged, so every experiment is done except the improvement rows member 3 asked 06 to add (`neg_alpha=1.0` at the selected settings, 3 seeds, reported separately). Next: report.

**Status (2026-10-11): exposure-only unbiased evaluation (GAUC, NDCG@5) decided; implemented by owner 1 on branch `member1-exposed-eval` (`unbiased_exposed` in both loaders and `store.py`, `inference.evaluate_exposed`, `k_core`); `fit` returns it as `test_exposed` next to `test_unbiased`; still to do: every member regenerating `data/processed/*.npz` (notebook 02's save cell). Merged to main (PR #2). 03/04 are not rerun.**

**Status (2026-10-11, evening): member 3's improvement implemented and run (seed 0, both datasets: `algorithms/negative_sampling.py`, `neg_alpha` in `fit`, summary in `results/07_improvement.json`). Exposure-only GAUC verdict: KuaiRand MF 0.6123 = LightGCN 0.6140, KuaiRec MF 0.7721 > LightGCN 0.7333 - the graph's biased-test advantage disappears on both unbiased tests. The fix itself hurts GAUC everywhere while raising coverage; 06 adds its `neg_alpha=1.0` rows as a separate comparison. Remaining experiments: 05 and 06 (member 4).**

**Status (2026-10-11, afternoon): notebook 08 rewritten (selected settings, exposure-only GAUC, most-popular reference per level, warm/cold users) on branch `member1-density-sweep`; random subsampling finished (seed 0, 16 runs; findings in notebook 08 and LOG.md: LightGCN's lead over BPR-MF grows as density falls because BPR-MF collapses, but on the unbiased test LightGCN never clearly beats most-popular); 5/10-core also finished (no change of sign on KuaiRec; a small unbiased-test lead for LightGCN on KuaiRand's dense core). Notebook 08 complete with outputs; all 24 runs cached in `results/08_density_*.json`.**

### notebooks/
Numbered, in order, one per step. Each notebook imports from `src/`, states its purpose in the first cell, and keeps its outputs saved.
- `00_introduction.ipynb`: text only: theme, exposure bias, why BPR-MF vs. LightGCN, objectives (owner 1).
- `01_data_exploration.ipynb`: raw data of KuaiRand-Pure and KuaiRec: files, columns, feedback signals, watch time and watch ratio, users and videos, how the unbiased parts were collected (owner 1).
- `02_data_processing.ipynb`: positive definitions, repeated interactions, filtering, splits and unbiased test sets, split checks, degree distributions, side-by-side table, popularity of biased vs. unbiased test positives, most-popular preview; saves `data/processed/*.npz` (owner 1).
- `03_mf_parameters.ipynb`, `04_lightgcn_parameters.ipynb`: parameter studies on validation.
- `05_ablation.ipynb`: component ablations.
- `06_main_comparison.ipynb`: most-popular, MF, LightGCN on both tests of both datasets, mean ± std over seeds (headline).
- `07_failure_cases.ipynb`: recall by item popularity and user degree on both tests, example users.
- `08_density_sweep.ipynb`: Recall vs. training density for both models and both datasets.

## Conventions
- **Log every step:** after each piece of work, append an entry to `LOG.md` with date, what was done, the result (numbers or a link to the notebook), and any decision and why. No author is recorded.
- **Sharing results:** `results/` is tracked in the team repository. Commit the small JSON/CSV metrics and figures so other members' notebooks can read them (06 needs 03 and 04). Never commit model weights (`*.pt`, `*.pth` stay ignored).
- **Figures:** save with `fig.savefig(path, dpi=200, metadata={"Software": None})`. Matplotlib otherwise writes its web address into every PNG, which breaks the no-URL rule for the zip.
- **Reproducibility:** fix seeds, save the exact config with every result in `results/`, and report mean ± std over several seeds.
- **Evaluation:** full ranking over all items; mask training items (and validation items at test time); early stopping on validation Recall@20. Every trained model is scored on both `test` and `test_unbiased`. For `test_unbiased`, also mask the user's biased test items (they are known positives). On KuaiRec's `test_unbiased`, rank only among the block videos; Recall@20 there is capped at about 0.32 because users have about 150 unbiased positives, so compare NDCG@20 and the margin over most-popular. **Exposure-only protocol (decided 2026-10-11), on both unbiased tests:** rank only the videos each user was actually shown (KuaiRand: the user's rows in the random log, clicked or not, median 24 per user; KuaiRec: the user's small-matrix videos, about 3,300 per user) and report GAUC (per-user AUC of positives vs. shown non-positives, averaged over users; random = 0.5) and NDCG@5. Full-ranking Recall@20/NDCG@20 stay as a second row. Reason: on KuaiRand's unbiased test full ranking puts thousands of never-shown videos among the candidates, so every model sits near the random floor (Recall@20 about 0.006 vs. 0.003 random), and Recall@20 over shown videos alone is near-trivial (43% of users were shown 20 or fewer). Selection in 03/04 used validation only and is unaffected; their figures keep the full-ranking unbiased numbers and are labelled as such.
- **Splits and positives:**
  - KuaiRand-Pure: positive is `is_click == 1`. `train`/`val`/`test` are a per-user random 80/20 split of the pooled standard (recommender-exposed) logs, then 10% of train for validation. `test_unbiased` is the clicks in the randomly exposed log for the same users and items, without pairs that are clicked in the standard logs.
  - KuaiRec: positive is a (user, video) pair with at least one play of `watch_ratio >= 2`. Same split of the big matrix. `test_unbiased` is the positives in the small matrix (no pair is in both matrices).
  - State this in the report and apply it identically everywhere.
- **Data:** never commit or zip anything under `data/`. Download steps go in `data/README.md`. Downloaded: `KuaiRand-Pure`, `KuaiRec.zip` (read directly from the zip, no need to extract).
- **Environment:** Python 3.13, PyTorch (CPU), numpy, scipy, pandas, matplotlib, scikit-learn, jupyter. No GPU, so keep experiments small enough for 8 CPU cores.
- **Code style:** match the existing files: short docstrings, no unneeded comments, small functions.
- **Sources:** cite the LightGCN paper (arXiv 2002.02126), the KuaiRand paper (arXiv 2208.08696), the KuaiRec paper (arXiv 2202.10842) and BPR (Rendle et al., UAI 2009). Do not copy tables or text from other work.

## Compute budget (proposed 2026-10-01, team to confirm)
Measured: LightGCN about 40 s per epoch and MF about 7 s per epoch on both Kuaishou datasets; a 100-epoch LightGCN run is about an hour, plus a few seconds per evaluation.
- Parameter studies (03, 04): seed 0 only, equal-sized grids for both models, max 100 epochs, evaluate every 5 epochs, patience 4. Tune on KuaiRand-Pure and KuaiRec separately.
- Main comparison (06) and ablation (05): 3 seeds with the best validation settings.
- Density sweep (08): 4 levels, 1 seed first; add seeds only if time allows.
- Spread runs across the four members' machines; every run writes its own JSON to `results/` so nothing has to be rerun.

## Fairness rules for every experiment
1. Identical splits and seeds for every model.
2. Same tuning budget: tune learning rate, L2 and embedding size on validation with equally sized grids.
3. Same embedding dimension, BPR loss and negative sampling, so the graph is the only difference.
4. Same stopping rule: early stopping on validation Recall@20, equal patience and max epochs; never look at test.
5. Several seeds; report mean ± std and say when a gap is within noise.
6. Same evaluation code for every model, on both tests: full ranking (block videos only on KuaiRec's unbiased test), known positives masked.
7. Report training time and memory per epoch next to accuracy.
8. Ablations change one component at a time.
9. The unbiased test sets are for reporting only, never for tuning or early stopping.
10. Document positive definition, split rule and thresholds; say which dataset and which test each number comes from.

## Density sweep (owner: member 1, secondary analysis)
Goal: test whether LightGCN's advantage, on each test, depends on interaction density.
- Randomly remove a fraction of **training** interactions (keep 100%, 50%, 25%, 10%). Validation and both test sets stay fixed.
- Apply the same procedure to both datasets.
- Record users, items, edges and density at every level, and note the cold users that sparsifying creates.
- A k-core filter (5-core, 10-core) is a separate, secondary variant because it also changes the item set; keep it apart from random subsampling.
- `sparsify(ds, keep_frac, seed)` and `k_core(ds, k)` in `src/data/transforms.py`; notebook `08_density_sweep.ipynb`.

## Data status (checked 2026-10-01, seed 0, numbers in `results/02_data_stats.json`)
- **KuaiRand-Pure** (`is_click`): 25,689 users, 7,218 items, 466,490 train / 54,223 val / 131,436 test, density 0.25%. Unbiased test: 194,539 clicks of 21,515 users. 63% of biased test positives are on the 10% most popular items, 14% of unbiased ones. Most-popular baseline: Recall@20 0.114 on the biased test, 0.007 on the unbiased test.
- **KuaiRec** (`watch_ratio >= 2`): 7,173 users, 10,611 items, 609,375 train / 67,773 val / 169,264 test, density 0.80%. Unbiased test: 216,727 positives of 1,408 users over 3,326 block videos. The block videos are popular (31% are in the top 10% by plays), so 55% of unbiased positives are on the top 10% items, like the biased test (56%). Most-popular: Recall@20 0.074 / NDCG@20 0.088 on the biased test, 0.150 / 0.541 on the unbiased test (best possible Recall@20 there is 0.32).
- The two unbiased tests remove different biases: KuaiRand's removes the recommender's choice of which videos are shown; KuaiRec's removes missing feedback on a fixed, popular block.
- KuaiRec's positive definition favours short videos: 11% of plays of videos under 10 s reach watch ratio 2, against 0.2% for videos over 60 s.
- The KuaiRand archive's MD5 differs from the one published in its repo, but it extracts cleanly and the random log has the documented 1,186,059 rows.
- Rejected: KuaiRand-1K (only 43,028 random-exposure interactions, about 4.4M videos, 4.3 GB), Gowalla (too large for CPU, no unbiased test), MicroLens (no unbiased test, very sparse), MovieLens-100K (no unbiased test; dropped 2026-10-10).

## Team roles (everyone touches code, analysis and report)
| Member | Code | Experiment and analysis | Report sections |
|---|---|---|---|
| **1** | Data loaders, splits, unbiased test sets, evaluation, `sparsify` | Dataset statistics; density sweep | Introduction (with related work), datasets, evaluation protocol, density sweep |
| **2** | BPR-MF and training loop | MF parameters (dim, lr, L2) | BPR-MF method, MF parameter discussion, conclusion |
| **3** | LightGCN, `analysis/` | LightGCN parameters (layers 0–4, dim); popularity and user-degree breakdown on both tests; improvement fix | LightGCN method, LightGCN parameter discussion, key factors, success and failure cases, possible improvements |
| **4** | Experiment runner, ablation switches, most-popular baseline | Main comparison on both tests (headline); ablations | Abstract, most-popular baseline, main comparison, ablation, strengths and weaknesses; packaging, contribution PDF, code zip |

Code review pairs: 1 with 2, and 3 with 4. Everyone proofreads the whole report once.

## Notebook and module ownership
The owner creates the notebook, writes the code cells, runs them, saves outputs, saves numbers to `results/`, appends a `LOG.md` entry and writes the matching report section. A contributor adds the cells named in the last column.

| Notebook | Owner | Task | Needs from others |
|---|---|---|---|
| `00_introduction` | **1** | Text only: theme, exposure bias, method rationale, objectives, notebook map. Source for the Introduction. (done) | none |
| `01_data_exploration` | **1** | Raw data of both datasets: files, columns, feedback, watch ratio, users and videos, how the unbiased parts were collected. (done) | none |
| `02_data_processing` | **1** | Positive definitions, repeated interactions, filtering, split sizes with the unbiased tests, split checks, degree distributions, popularity of biased vs. unbiased test positives, most-popular preview. Saves the processed datasets to `data/processed/`. (done) | none |
| `03_mf_parameters` | **2** | BPR-MF study: embedding dim (16/32/64/128), learning rate, L2, on both datasets, selected on validation Recall@20. Report the unbiased test of the selected setting only. | 1's loaders and evaluation |
| `04_lightgcn_parameters` | **3** | LightGCN study: layers 0–4 (over-smoothing curve on both tests), embedding dim, learning rate, L2. Layer 0 is the MF-equivalent control. | 1's loaders, 2's training loop |
| `05_ablation` | **4** | One change at a time: mean vs. last-layer combination, symmetric vs. no normalization, edge dropout on/off. Table of deltas on both datasets and both tests. | 3's LightGCN switches |
| `06_main_comparison` | **4** | Headline: most-popular, BPR-MF and LightGCN with the best validation settings from 03 and 04, 3 seeds, both datasets, both tests. Recall@20, NDCG@20, margin over most-popular, coverage, popularity, training time, mean ± std. Headline figure. | 03, 04 results |
| `07_failure_cases` | **3** | Recall by item-popularity bucket and by user degree group (cold/medium/heavy) on both tests; where LightGCN's gain comes from; example users where it wins and fails. | 1's per-user recall from `evaluate`, 06 models |
| `08_density_sweep` | **1** | Recall vs. training density (100/50/25/10%) for both models on both datasets, scored on both tests, using `sparsify`. Cold users created. Secondary k-core variant. Runner `experiments/density_sweep.py`. (done, seed 0) | 2 and 3 for the models, 03/04 settings |

Module owners: `data/` and `inference/` are 1; `MF` and the training loop are 2; `LightGCN` and `analysis/` are 3; most-popular baseline and `experiments/` (multi-seed runner, saves JSON) and `ablations/` are 4.

Report and packaging: 1 drafts the Introduction (with related work), 2 writes the Conclusion, 4 writes the Abstract and builds the final PDF, the contribution PDF and the code zip (no data, no models, no URLs, no `CLAUDE.md`/`LOG.md`), 1 checks that the zip runs from a clean folder and contains no `http` string.

## Milestones
- 2 Oct: shared interface contract agreed (dataset dict with `test_unbiased`, config keys, result JSON with both tests). Compute budget confirmed.
- 11 Oct: all experiments finished.
- 14 Oct: report drafted (at least 4 pages, the five required headings, all six figures in, improvements subsection written).
- 17 Oct: proofread, build the three files, submit.

## Open decisions
- Confirm the positives: `is_click` for KuaiRand-Pure, `watch_ratio >= 2` (any play) for KuaiRec.
- Confirm the compute budget above.
- Group ID for the file names.
- Who makes the single NTULearn submission (default: the group coordinator).
- Which improvement to implement (decide after notebook 07), or discussion only if time runs out.
- Whether to add ItemKNN as a second baseline.
