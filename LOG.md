# Project log

Append one entry per step: date, what was done, result, decisions and why.

## 2026-09-30
- **Topic chosen:** graph-based recommender systems. Methods: BPR-MF vs. LightGCN. Reason: the pair isolates what graph propagation adds.
- **Novelty check (web search):** LightGCN vs. BPR-MF is heavily published; KuaiRand-Pure is less common with this pair. Decision: keep the pair, use MovieLens-100K + KuaiRand-Pure, and build the analysis around degree groups, popularity and exposure bias.
- **Setup:** downloaded `ml-100k` and the Gowalla train/test files into `data/` (Gowalla to be dropped). Installed CPU PyTorch 2.14.
- **Drafts written, untested:** `src/data.py`, `src/models.py`, `src/train.py`.
- **Team roles assigned** (see CLAUDE.md).
- **Next:** add the KuaiRand-Pure loader, write the interface contract, run a smoke test on MovieLens-100K.

## 2026-09-30 (later)
- Downloaded KuaiRand-Pure (Zenodo record 10439422, 47 MB archive) into `data/`. MD5 differs from the repo's published hash; archive extracts fine and row counts match the paper (random log 1,186,059 rows).
- Dropped Gowalla: deleted its files and its loader.
- Rewrote `src/data.py`: added `load_kuairand` (modes `standard`, `random_test`; positive selectable). Ran it: statistics recorded in CLAUDE.md. Decision: positive = `is_click`, since `is_like` leaves only 15k train rows.
- Added root `README.md`, `data/README.md`, `.gitignore` (excludes CLAUDE.md, LOG.md, data, results).
- Next: interface contract, smoke test of MF and LightGCN on MovieLens-100K, time one KuaiRand epoch.

## 2026-09-30 (decision)
- Considered switching to KuaiRand-1K. Rejected: 1K has only 43,028 random-exposure interactions (vs 1,186,059 in Pure), about 4.4M videos (extreme item-side sparsity, heavy for CPU), a 4.3 GB download, and would not close the density gap with MovieLens. Staying with MovieLens-100K + KuaiRand-Pure (`is_click`).
- Idea to handle the density gap: treat density as a variable (subsampling or k-core levels on both datasets, plot Recall vs. density). Not yet decided.

## 2026-09-30 (plan update)
- Added the density sweep (owner: member 1) and the 10 fairness rules to CLAUDE.md. Moved packaging, contribution PDF and code zip from member 1 to member 4 to balance load.
- Fixed CLAUDE.md layout: environment, code style and sources bullets are back under Conventions.

## 2026-09-30 (assignments)
- Added notebook and module ownership to CLAUDE.md. Moved failure-case notebook (07) and `analysis.py` from member 1 to member 3 to balance load. Notebooks folder is still empty; only the plan exists.
- Review of `src/data.py` found: 5 leaked test items in `random_test` (to remove), CLAUDE.md split wording outdated. Fixes assigned to member 1 (notebook 01).

## 2026-09-30 (fixes and stubs)
- `src/data.py`: `random_test` now removes test pairs already clicked in training and users left without targets. Result: 6 pairs removed, test 198,353, no overlap with train or val. CLAUDE.md split wording and numbers updated.
- Created stub notebooks 01-08 in `notebooks/` (purpose cell with owner/task/inputs/outputs, shared imports and setup cell, empty experiment cell). All parse as valid JSON. Notebook 08 refers to `data.sparsify`, which does not exist yet.

## 2026-09-30 (requirements and repo rules)
- Added `requirements.txt` (torch CPU installed separately, then numpy, scipy, pandas, matplotlib, scikit-learn, jupyter). README setup now uses it. Jupyter is not installed on this machine yet.
- Rule change: CLAUDE.md and LOG.md may now be pushed; removed them from `.gitignore`. They stay out of the submission zip.

## 2026-09-30 (plan checked against the project brief)
- Compared CLAUDE.md, README.md and this log with `project1 technical review.pdf` (14 slides). Topic, 2 methods, 2 datasets, the six analysis items, page and font limits, file names and the deadline all match.
- Gaps found and added to CLAUDE.md:
  - Required headings Abstract/Introduction/Methods/Experiments/Conclusion were not mapped and the Abstract had no owner. Added a report structure with owners; Abstract goes to member 4.
  - "No external links": README.md, requirements.txt and data/README.md contain URLs. Decision: the zip gets a link-free README and requirements file, and member 1 checks the package for `http` before submission. The repository files are unchanged.
  - Rubric item "possible solutions to improve the methods" was missing. Added: discussion as the minimum, one small implemented fix as the target, owner 3, chosen after notebook 07.
  - Rubric item "good visualizations": added a list of five required figures with notebooks and owners.
  - Optional non-learned baseline (most-popular, ItemKNN) noted, only if time allows.
- Also recorded: one submission per group (submitter to confirm); the instructor's cut-off for reporting contribution issues (20 Sep 2026) has passed; log entries must name their author.
- CLAUDE.md corrections: status line now says `data.py` has been run; removed the note about the `code/` folder, which no longer exists.
- New open decisions: submitter, how to share `results/` (currently git-ignored), which improvement to implement, optional baseline, authors of the earlier log entries.
- No code was run or changed.

## 2026-09-30 (report owners, log rule, shared results)
- Report owners changed: Introduction (with related work) is member 1, Conclusion is member 2, Abstract stays with member 4. Team roles table and the packaging line in CLAUDE.md updated to match.
- Log entries do not record an author. This replaces the author requirement noted in the previous entry.
- Decision: share results through the repository. Removed `results/` from `.gitignore`; small JSON/CSV metrics and figures are committed, model weights (`*.pt`, `*.pth`) stay ignored. README layout note updated.

## 2026-09-30 (member 1: data code, evaluation, notebook 01)
- `src/data.py`: added `sparsify(ds, keep_frac, seed)` (random subsample of training pairs; val, test and ids fixed), `k_core(ds, k)` (iterated, remaps ids, filters val/test), `degrees(ds)`; `stats` now also reports cold users and cold items (test users/items with no training interaction).
- `src/train.py`: `evaluate` also returns `popularity` (mean training degree of the recommended items). No other change.
- Checks (all passed): no pair shared between train, val and test in any split; an oracle model gets NDCG@20 = 1 and the expected recall; training items are masked; a random model on KuaiRand-Pure scores recall 0.0027, equal to chance (20 / 7,218); `sparsify` is deterministic per seed and a subset of train; `k_core` leaves every degree >= k.
- Sparsify sizes (seed 0). MovieLens-100K: 50% -> 19,928 pairs, 1 cold user; 25% -> 9,964, 29 cold; 10% -> 3,986, 129 cold. KuaiRand-Pure: 50% -> 233,245, 1,162 cold; 25% -> 116,622, 3,375 cold; 10% -> 46,649, 8,092 cold.
- k-core. MovieLens-100K: 5-core 927 users / 926 items / density 4.5%; 10-core 804 / 719 / 6.3%. KuaiRand-Pure: 5-core 20,952 / 5,301 / 0.41%; 10-core 15,592 / 3,999 / 0.65%.
- Notebook `01_data_exploration` run with outputs; numbers in `results/01_data_stats.json`, figure `results/01_degree_distributions.png`. Findings:
  - Item popularity is more skewed in KuaiRand-Pure: top 1% of items hold 20.5% of training clicks (9.5% in MovieLens-100K); item Gini 0.76 vs 0.71. Median user degree 13 vs 28.
  - MovieLens-100K: 55,375 of 100,000 ratings are >= 4; no duplicate pairs.
  - KuaiRand-Pure click rate is 46.0% in the standard logs and 17.6% in the random log; `is_like` 1.8% / 0.5%; `long_view` 33.2% / 8.5%. `long_view` would give 336,808 train pairs, `is_like` 15,248. `is_click` stays the positive.
  - Removing repeated (user, video) clicks drops 7,291 of 660,342 standard clicks (1.1%) and none in the random log.
  - Exposure bias is visible in the data: 63% of `standard` test clicks are on the 10% most popular items, against 14% of `random_test` clicks (mean item degree 530 vs 111).
  - `random_test`: 208,934 random click pairs, 198,359 with a known user and item, 6 already clicked in training, 198,353 kept.
- Noted: `n_users` includes users with a single positive, who are in no split (1 in MovieLens-100K, 902 in KuaiRand-Pure). Left as is; listed under Open decisions.
- Jupyter installed on this machine (from `requirements.txt`).

## 2026-09-30 (timing of the draft models)
- Default settings (dim 64, lr 1e-3, L2 1e-4, batch 2048), seed 0, MovieLens-100K: BPR-MF test Recall@20 0.315 (best epoch 95, 33 s); LightGCN 3 layers 0.334 (best epoch 195 of 200, not converged, 571 s). Untuned, so not a comparison yet.
- KuaiRand-Pure: MF 5.4 s per epoch; LightGCN 340 s per epoch, which makes the planned experiments impossible on CPU.
- Cause: the backward pass of `torch.sparse.mm` on the COO adjacency (about 1.2 s per step; negative sampling is 1 ms). Tested fix: CSR adjacency and a custom autograd function whose backward multiplies by a CSR copy of the transpose. Gradients match to 6e-9, step time 75 ms (about 15 times faster). Not applied: `models.py` belongs to member 3. Recorded under Open decisions.

## 2026-09-30 (notebook 08: code and MovieLens pilot)
- Notebook `08_density_sweep` written: random subsampling at 100/50/25/10%, 5-core and 10-core variant, per-run JSON with config and graph statistics, warm/cold user recall, mean ± std table, gap table, figure. Finished runs are cached in the JSON and skipped on rerun.
- **Pilot only**: MovieLens-100K, seed 0, untuned shared settings (dim 64, lr 5e-3, L2 1e-4, batch 2048, max 100 epochs, LightGCN 3 layers). Files carry the `_pilot` tag (`results/08_density_sweep_pilot.*`). Not for the report.
- Pilot test Recall@20, MF / LightGCN: 100% 0.301 / 0.340; 50% 0.177 / 0.259; 25% 0.100 / 0.200; 10% 0.063 / 0.132. Relative gain of LightGCN grows from 13% to 111% as density falls. 5-core 0.310 / 0.351; 10-core 0.283 / 0.338.
- Caveats: one seed, so no noise estimate; settings are not tuned per level, and MF may be under-regularised on the sparse levels, which would inflate the gap. LightGCN covers fewer items than MF at every subsampling level (for example 26% vs 41% at 50%).
- Final run needs: best validation settings from notebooks 03 and 04, three seeds, KuaiRand-Pure added, and the LightGCN speed issue resolved. At the current speed the MovieLens part alone takes about 12 minutes per seed; KuaiRand-Pure is not feasible.

## 2026-09-30 (notebook 00: introduction and raw-data tour)
- Added `notebooks/00_introduction.ipynb` (owner 1), run with outputs. Contents: theme and task, BPR-MF vs. LightGCN table and why the pair is a controlled experiment (LightGCN with 0 layers is MF), objectives with a notebook map, then both datasets in tables: file inventory, column meanings, example rows, distributions, and a side-by-side table. Saved `results/00_dataset_overview.csv` and `results/00_kuairand_daily.png`. No URLs in the notebook (the IMDb link column of `u.item` is dropped).
- MovieLens-100K raw: 100,000 ratings, 943 users, 1,682 movies, 20 Sep 1997 to 22 Apr 1998, raw density 6.3%; 55.4% of ratings are >= 4; users have 20 to 737 ratings (median 65); 71% of users are male, 21% students.
- KuaiRand-Pure raw: 1,436,609 standard and 1,186,059 random exposures, 27,285 users, 7,583 videos, 9 Apr to 8 May 2022 (the first file is named 4_08 but starts on 9 April). All videos were uploaded 9 to 11 April 2022; 99% are normal videos, 1% ads.
- Signal rates, standard vs. random: click 46.0% / 17.6%, long view 33.2% / 8.5%, like 1.85% / 0.48%; hate is the only signal more frequent on random videos (0.05% / 0.11%). 99.6% of long views are clicks; 66% of clicks are long views.
- Random exposure is nearly uniform over videos (156 exposures per video, std 31), while standard exposure is heavily skewed (mean 190, median 57, max 10,424). All 27,077 standard users also appear in the random log; 22,335 users clicked in both.
- Found: MovieLens `n_items` is 1,674 but only 1,447 movies have a positive; added to Open decisions with the single-positive users.

## 2026-09-30 (dataset choice kept; notebooks 00 and 01 reorganised)
- Decision: keep MovieLens-100K next to KuaiRand-Pure. The domains differ, but the models only see the graph, and the density sweep controls for density inside each dataset. KuaiRec (same platform as KuaiRand) was the only replacement that would fix the domain mismatch; not adopted because of the time left. It stays a possible third dataset after 11 Oct.
- Moved the raw-data tour, the side-by-side table and the summary from notebook 00 into notebook 01. Notebook 00 is now text only (theme, methods, objectives). Notebook 01 has Part A (raw data) and Part B (training graphs after filtering), and reads the logs once.
- Notebook 01 rerun with outputs; `results/01_data_stats.json` unchanged in content. Result files renamed to `01_dataset_overview.csv` and `01_kuairand_daily.png`; the `00_` copies were removed.

## 2026-09-30 (MovieLens id remap, faster LightGCN)
- `src/data.py`: `load_movielens` now remaps user and movie ids to those with at least one positive, as `load_kuairand` does. MovieLens-100K is now 942 users, 1,447 items, density 2.9% (was 943 / 1,674 / 2.5%); the splits keep the same sizes (39,857 / 4,439 / 11,079). The dropped ids were one user with no rating >= 4 and 227 movies never rated >= 4. Earlier entries in this log use the old counts. README and CLAUDE.md updated.
- `src/models.py` (member 3's module, changed with the user's approval): the adjacency is a CSR tensor and propagation goes through `SpMM`, an autograd function whose backward multiplies by a CSR copy of the transpose. Checked against plain autograd for all norm (sym, left, none) and combine (mean, last) settings with edge dropout: relative gradient difference below 3e-7.
- Speed: KuaiRand-Pure LightGCN about 40 s per epoch (was 340 s), MF 7 s. MovieLens-100K LightGCN about 0.5 s per epoch (was 3 s). Two-epoch KuaiRand recall is unchanged (0.1199).
- Notebook 01 rerun (item Gini for MovieLens now 0.67, top 1% of items hold 8.6% of training positives).
- Notebook 08 pilot rerun on the remapped data, same settings, seed 0; all 12 runs took 196 s in total. Test Recall@20, MF / LightGCN: 100% 0.301 / 0.350; 50% 0.173 / 0.255; 25% 0.100 / 0.210; 10% 0.051 / 0.128. Relative gain 16% to 151%. 5-core 0.310 / 0.351; 10-core 0.283 / 0.338. Same caveats as before: one seed, untuned.
- Remaining (see also the next entry): KuaiRand-Pure still has 902 single-click users in `n_users`; a 100-epoch LightGCN run on KuaiRand-Pure is about an hour, so the KuaiRand budget is an open decision.

## 2026-10-01 (research angle and KuaiRec feasibility check)
- Discussed making exposure bias the headline question: does LightGCN's advantage over BPR-MF hold under unbiased evaluation? Not yet decided; the plan is unchanged. If adopted: same trained model scored on a biased and an unbiased test set, a most-popular baseline becomes required, the density sweep becomes secondary.
- MicroLens considered as a replacement for MovieLens-100K and rejected: same domain as KuaiRand but no unbiased test set, very sparse, heavier on CPU.
- KuaiRec downloaded to `data/KuaiRec.zip` (412 MB, Zenodo record 18164998, not extracted) and checked with a scratch script. Nothing in `src/` changed.
  - Big matrix: 12,530,806 rows, 10,300,969 unique pairs, 7,176 users, 10,728 videos, 5 Jul to 5 Sep 2020. Small matrix: 4,676,570 pairs, 1,411 users, 3,327 videos, 99.6% dense. All small-matrix users and videos are in the big matrix, and no (user, video) pair is in both.
  - Positive = watch ratio >= 2: 846,414 unique pairs in the big matrix (7.5% of rows). Per-user 80/20 split gives 7,175 users, 10,611 items, 609,375 train / 67,773 val / 169,264 test, density 0.80%, median user degree 63. Unbiased test from the small matrix: 216,770 positives for 1,410 users (median 95 per user).
  - Positive = watch ratio >= 1: 2,686,953 train pairs, density 3.5%; LightGCN about 270 s per epoch, too slow.
  - Speed at watch ratio >= 2: MF 7 s per epoch, LightGCN about 40 s per epoch, the same as KuaiRand-Pure.
  - Caveat: the small-matrix videos are themselves popular (27% of them are in the top 10% of training items), so 55% of small-matrix positives fall on the top 10% of items, the same as the standard test (56%). In KuaiRand-Pure the shift is 63% to 14%. The unbiased test on KuaiRec must rank only among the 3,327 small-matrix videos, which needs a candidate restriction in `evaluate`.

## 2026-10-01 (decision: exposure-bias headline, KuaiRec replaces MovieLens-100K)
- **Decision (user):** the headline question is now whether LightGCN's advantage over BPR-MF holds under unbiased evaluation. KuaiRec replaces MovieLens-100K as the second dataset; MovieLens-100K stays only for smoke tests. A most-popular baseline is now required (owner 4). The density sweep becomes a secondary analysis. Why: both Kuaishou datasets have an unbiased part, so the result can be shown on two datasets of the same domain, and the question is more original than a plain BPR-MF vs. LightGCN comparison.
- `src/data.py` rewritten: one loader per dataset (`kuairand`, `kuairec`, `ml100k`), every dataset dict has `train`/`val`/`test`, and the Kuaishou ones add `test_unbiased`. KuaiRand's old `mode` argument (`standard` / `random_test`) is gone: the same trained model is now scored on both tests. KuaiRec is read straight from `data/KuaiRec.zip`. Users with fewer than 2 positives are dropped in all loaders (this removes the 902 single-click KuaiRand users from `n_users`, closing that open decision). Raw positives are cached per process. `sparsify` and `k_core` keep the unbiased test.
- `src/train.py`: `evaluate` ranks only among `unbiased_items` on KuaiRec's unbiased test (coverage over those items). `fit` also returns `test_unbiased`, scored with train, val and biased test items masked.
- `src/models.py`: two PyTorch warnings silenced at the CSR constructor (`check_invariants=False` and a filter for the beta warning). They printed a GitHub address and the local file path into notebook outputs.
- Checks (all passed): splits disjoint including the unbiased test; every unbiased-test user has training items; KuaiRec unbiased items are all block videos; an oracle gets NDCG@20 = 1 on KuaiRec's unbiased test; `fit` returns both tests on both datasets.
- Numbers (seed 0, identical for seeds 0-2):
  - KuaiRand-Pure: 25,689 users, 7,218 items, 466,490 / 54,223 / 131,436, density 0.25%; unbiased test 194,539 clicks, 21,515 users (was 198,353 under the old `random_test` mode, which trained on all standard clicks).
  - KuaiRec: 7,173 users, 10,611 items, 609,375 / 67,773 / 169,264, density 0.80%; unbiased test 216,727 positives, 1,408 users, 3,326 block videos.
  - Most-popular, biased vs. unbiased test, Recall@20: KuaiRand 0.114 vs 0.007; KuaiRec 0.074 vs 0.150 (NDCG@20 0.088 vs 0.541). On KuaiRec's unbiased test the best possible Recall@20 is 0.318.
  - Share of test positives on the 10% most popular items: KuaiRand 63% biased vs 14% unbiased; KuaiRec 56% vs 55%. The KuaiRec block videos are popular (31% in the top 10% by plays; median 2,832 plays vs 243 overall), so KuaiRec's unbiased test removes missing feedback, not popularity.
  - KuaiRec raw: 12,530,806 plays, 2,229,837 repeated plays; 103,863 pairs are positive by their best play but not their worst. The big matrix has no plays in the weeks of 13-26 July and 17-23 August 2020. Watch ratio >= 2 is reached by 11% of plays of videos under 10 s and 0.2% of videos over 60 s.
- Notebook 01 rebuilt and run: Part A KuaiRand-Pure (unchanged) and KuaiRec (new) raw data, side-by-side table; Part B split sizes with unbiased tests, split checks, degree distributions (figure regenerated for KuaiRand-Pure and KuaiRec), positive definitions for both, repeated interactions, biased vs. unbiased popularity with a most-popular preview, summary. MovieLens sections removed.
- Notebook 00 rewritten (text only): theme, exposure bias, the two unbiased parts, method rationale, most-popular baseline, objectives and notebook map.
- Notebooks 02-07: purpose cells rewritten for the new plan (both tests everywhere, 06 is the headline with most-popular, 07 breaks recall down by item popularity and user degree). Notebook 08: purpose updated, records `test_unbiased`; MovieLens pilot rerun, numbers unchanged.
- Figures are now saved with `metadata={"Software": None}`: matplotlib writes its web address into every PNG otherwise. A scan of notebooks, results and `src/` finds no `http` string and no local path.
- Documentation: CLAUDE.md rewritten (goal, research question, report structure, required figures now six, conventions for the two tests, compute budget proposal, fairness rule 9, data status, roles, ownership, milestones, open decisions). README.md and data/README.md rewritten for the new datasets. `.gitignore` now keeps `data/README.md` tracked (it was ignored with the rest of `data/`).
- Open: confirm the positives and the compute budget at the 2 Oct contract meeting.

## 2026-10-01 (story: short-video feeds)
- Framing agreed: the graph comparison stays the subject and exposure bias is the test. Motivation: short-video feeds are spreading into e-commerce and social platforms; in a feed the recommender decides almost everything the user sees, so exposure bias is strongest there. Scope: one platform, feed-style recommendation only, no claim of generalising to e-commerce. Practical takeaway: evaluate a graph recommender on unbiased data and against most-popular before deploying it in a feed.
- Considered and rejected: dropping the graph focus for a pure biased-vs-unbiased study (the brief grades methods: parameters, ablation, strengths and weaknesses). KuaiRand-Pure is not a subset of KuaiRec: separate collections with unlinked ids and different unbiased mechanisms.
- Notebook 00: motivation paragraph in section 1, why short video and scope in section 2, objective 6 (practical takeaway). CLAUDE.md: story added to Goal; Introduction and Conclusion contents updated. The Introduction needs 1-2 real sources for short-video feeds in shopping apps.

## 2026-10-01 (repository)
- First commit pushed to the team repository on GitHub (graph-recsys-exposure-bias, branch main). `data/` contents and the course brief PDF are ignored.

## 2026-10-10 (src/ reorganised into packages)
- `src/` is now one package per area so each member adds files in their own folder: `data/`, `inference/` (owner 1), `models/`, `training/`, `algorithms/`, `ablations/`, `experiments/`, `analysis/`, plus shared `utils/`. Table of folders and owners in CLAUDE.md.
- Member 1 code split up: `data.py` -> `data/loaders.py`, `splits.py`, `transforms.py`, `stats.py`; `evaluate` moved from `train.py` to `inference/evaluate.py`; `to_pairs` now in `data/splits.py`. Code unchanged otherwise. Data paths come from `utils/paths.py`.
- Members 2-4 code moved without edits: `models.py` -> `models/core.py`; `train.py` -> `training/train.py`, whose only changes are dropping `evaluate` and `to_pairs` and importing them from `inference` and `data`. `algorithms/`, `ablations/`, `experiments/`, `analysis/` are empty packages for their owners.
- `utils/io.py`: `save_json`, `load_json` and `savefig` (saves at 200 dpi without matplotlib's web address).
- Notebooks: `import train` -> `import training, inference`; `train.fit` -> `training.fit`, `train.evaluate` -> `inference.evaluate`; "Inputs" lines point to the new folders. Outputs not rerun.
- Check: a short MovieLens run (stats, sparsify, 5-core, 4 epochs of MF and LightGCN with edge dropout) gives the same metrics before and after; the only difference is MF's last loss value in the 8th decimal, which also varies between two runs of the same code (multithreaded float sums). The import cell of every notebook runs.

## 2026-10-10 (MovieLens-100K and the smoke test dropped)
- **Decision (user):** remove MovieLens-100K everywhere so the team works only with the two Kuaishou datasets, and drop the smoke-test notebook; each member checks the pipeline step by step in their own notebook.
- `src/data/loaders.py`: `load_movielens` and the `"ml100k"` key removed; `data.load` takes `"kuairand"` or `"kuairec"`.
- `notebooks/02_smoke_test.ipynb` deleted (it was a stub). Notebook numbers are kept, so there is no 02. Notebook 00's map no longer lists it.
- Notebook 08: pilot removed. `DATASETS` is KuaiRand-Pure and KuaiRec, output files have no `_pilot` tag, all outputs cleared, status cell says not run and that `CONFIGS` needs the settings from 03 and 04. `results/08_density_sweep_pilot*` (4 files) deleted.
- README.md, data/README.md (MovieLens download section) and CLAUDE.md updated; MovieLens-100K is listed under rejected datasets. The saved output of notebook 01's last cell still lists the deleted pilot files until 01 is rerun.
