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

## 2026-10-10 (notebook 02: data processing)
- **Decision (user):** notebook 01 keeps only the raw-data description; the processing of the final datasets moves to a new notebook `02_data_processing.ipynb` (the number freed by the dropped smoke test).
- Notebook 01: Part B and the side-by-side table removed; new figure `results/01_kuairec_watch_ratio.png` (watch ratio per play, big vs. small matrix, and share of plays reaching ratio 2 by video length) with a reading guide.
- Notebook 02: positive definition, repeated interactions, a new filtering table (users dropped for having one positive: 902 of 26,591 on KuaiRand-Pure, 2 of 7,175 on KuaiRec; unbiased positives kept: 194,539 of 208,934 and 216,727 of 217,229), split sizes, seed check, split checks, degree distributions, side-by-side table, biased vs. unbiased test popularity and most-popular preview. Outputs renamed `01_` -> `02_` (`02_data_stats.json`, `02_dataset_overview.csv`, `02_degree_distributions.png`); the content is identical to the old `01_` files apart from the new `filtering` entry.
- `src/data/store.py`: `save_processed(ds, name, seed)` and `load_processed(name, seed)`. Notebook 02 writes `data/processed/{kuairand,kuairec}_seed{0,1,2}.npz` (2.1-2.5 MB each, not in git) and checks that each file reads back identical to `data.load`, including the order of users and items. Reading takes under a second.
- Both notebooks rerun with the project `.venv` kernel (kernelspec changed from the conda kernel to `python3`). Notebook 00's map and CLAUDE.md updated.

## 2026-10-10 (notebook 03: BPR-MF parameter study)
- Completed the BPR-MF parameter study on KuaiRand-Pure and KuaiRec: 8 configurations per dataset, seed 0, max 100 epochs, validation every 5 epochs and patience 4. The grid varied embedding dimension (16/32/64/128), learning rate (3e-4/1e-3/3e-3) and L2 (1e-5/1e-4/1e-3).
- Selection used validation Recall@20 only; neither standard-test nor less-biased-test metrics were used for selection.
  - KuaiRand-Pure selected `dim=64`, `lr=0.001`, `reg=0.001` (`reg_1e-3`): validation Recall@20 0.1673, best epoch 100.
  - KuaiRec selected `dim=64`, `lr=0.0003`, `reg=0.0001` (`lr_3e-4`): validation Recall@20 0.1084, best epoch 75.
- Final seed-0 evaluation of the selected settings:
  - KuaiRand-Pure: standard Recall@20 0.1693 and NDCG@20 0.1102; less-biased Recall@20 0.0060 and NDCG@20 0.0041.
  - KuaiRec: standard Recall@20 0.1066 and NDCG@20 0.1145; less-biased Recall@20 0.1731 and NDCG@20 0.5631.
- Saved `results/03_mf_parameters.json` and `results/03_mf_validation.png`. The validation figure highlights `reg_1e-3` for KuaiRand-Pure and `lr_3e-4` for KuaiRec.

## 2026-10-10 (most-popular module, parameter-grid runner, notebooks 03/04, grids launched)
- Datasets restored to `data/` on this machine (fresh clone): KuaiRand-Pure re-downloaded and extracted, KuaiRec.zip re-downloading (412 MB, slow link).
- `models/most_popular.py`: the `MostPopular` class moved out of notebook 01 into `models/`, exported there. Verified against notebook 01's numbers on KuaiRand-Pure (test Recall@20 0.1141 / NDCG 0.0761; unbiased 0.0068 / 0.0046).
- `experiments/run_grid.py`: parameter grids for notebooks 03/04 with caching. One JSON per notebook and dataset (`results/03_mf_parameters_<ds>.json`, `04_lightgcn_parameters_<ds>.json`), one entry per dataset+config+seed, saved after every finished run, so an interrupted or machine-split grid resumes and different machines can run different datasets without write conflicts. Also `select_best` (validation Recall@20, grid runs only), the layer study (layers 0/1/4 at the selected config), `layer_curve` for the over-smoothing figure, and a CLI for standalone/nohup use. Grids are equally sized, 16 configs each (fairness rule 2): MF dim 16/32/64/128 x lr 1e-3/5e-3 x L2 1e-5/1e-4; LightGCN layers 2/3 x dim 64/128 x the same lr/L2 values. Seed 0, max 100 epochs, per the compute budget.
- Notebooks 03 and 04: experiment cells written (grid execution via the runner, validation tables, selected setting, and the over-smoothing figure `04_layers.png`). Outputs not yet saved; they fill from the cached JSONs once the grids finish.
- `training/train.py` (member 2's module, one-line guard): `fit` crashed when a run ended before the first validation snapshot (epochs < eval_every); it now keeps the current weights instead. The runner also passes `eval_every` through to `fit`.
- Smoke test on KuaiRand-Pure (2-epoch runs, throwaway keys, removed afterwards): layer-0 LightGCN gives metrics identical to MF (test Recall@20 0.1053 both), so the MF-equivalent control holds exactly. A 2-epoch 1-layer LightGCN already shows the pattern the project is about: biased test 0.105 -> 0.132, unbiased test 0.0065 -> 0.0064.
- Launched under nohup: `03 kuairand 04 kuairand` (MF grid, then LightGCN grid + layer study). KuaiRec grids launch when its download finishes. Expected total: roughly 3 h for the MF grids and about a day for the LightGCN grids on 8 CPU threads.

## 2026-10-10 (bottlenecks found and fixed: single-core LightGCN, throttled download)
- Measured while running: the MF grid uses about 2 cores and finishes a config in 74-125 s (early stopping around epoch 15-20), so all 16 configs cost about half an hour per dataset. LightGCN ran at 96% CPU: torch's CPU sparse propagation is effectively single-threaded even with `threads=8`, so a sequential grid used 1 of 8 cores.
- `experiments/run_grid.py`: added `--shard=i/n`. Each shard process runs slice i of the grid (`cfgs[i::n]`) into its own file (`04_lightgcn_parameters_<ds>.shard<i>.json`); `load_runs` merges the base file with all shard files. Shards run grid configs only; the layer study runs in the unsharded process/notebook afterwards, when the whole grid is cached. LightGCN grids now run as 4 shard processes, about 4x faster in wall-clock.
- KuaiRec download: Zenodo throttles to about 120 KB/s per connection (412 MB = over an hour). Fixed by fetching the remaining 285 MB in 8 parallel HTTP range requests (< 1 minute), concatenating, and verifying: final size 431,964,858 bytes and `zipfile.testzip()` passes on every member.
- KuaiRec smoke test passed: loader 7,173 users / 10,611 items; `MostPopular` reproduces notebook 01 (test 0.0736/0.0877, unbiased 0.1503/0.5412); layer-0 LightGCN identical to MF (0.0773 both). Throwaway 2-epoch runs removed from the caches afterwards.
- All four grids running: `03 kuairand` (9/16 cached), `03 kuairec`, and `04 kuairand` / `04 kuairec` as 4 shards each. 2-epoch probes already show the dataset difference the report will hinge on: on KuaiRand-Pure 1 layer lifts the biased test (0.105 -> 0.132) and not the unbiased one (0.0065 -> 0.0064); on KuaiRec it lifts both (0.077 -> 0.102 biased, 0.162 -> 0.180 unbiased).

## 2026-10-10 (GPU support, Colab T4 run, upstream integration)
- **GPU device support** in `models/core.py` (`build_norm_adj`/`_to_csr` take a device; `set_graph` uses the model's), `training/train.py` (`fit` reads `cfg["device"]`, batches move to it) and `inference/evaluate.py` (batch indexing on the embedding device, scores moved to CPU once). `experiments/run_grid.py` gains `--device=`. Regression check on CPU: 2-epoch smoke runs reproduce the pre-patch numbers exactly (MF 0.1021/0.1053/0.0065, LightGCN-1 0.1256/0.1321/0.0064), so the patch only changes where the math runs.
- Upstream integrated on this machine (notebook 02 + `src/data/store.py`): conflicts in CLAUDE.md/LOG.md merged by hand. Notebook 02 executed headless; `data/processed/{kuairand,kuairec}_seed{0,1,2}.npz` written and checked (`load_processed` matches `data.load`), then `git restore notebooks/02_data_processing.ipynb results/` per the agreed procedure. `run_grid._dataset` now prefers `load_processed` and falls back to `load` (the processed files are identical to `load`, so cached grid results stay valid).
- **Colab T4 run** (free tier, browser-driven): payload zip (src + current result JSONs, 20 KB -> 27 k base64 chars) injected into a notebook cell, both datasets fetched with 8 parallel HTTP range requests (about 3 minutes), CUDA smoke test passed (`cuda: True Tesla T4`; 2-epoch LightGCN 8.7 s including evaluation, about 15x the loaded local CPU). Pitfalls found and fixed, all Colab-specific:
  - A runner started with `Popen(..., shell=True)` + `nohup &` dies with the cell's process group when the cell returns; it must be launched with `start_new_session=True` (and `python3 -u` for an unbuffered log).
  - `pgrep -f run_grid` from a subprocess whose own command line contains the pattern always reports RUNNING; status checks must use `ps aux | grep [r]un_grid`.
  - The GitHub repository is now private (it was public this morning), so cloning on Colab needs credentials; the local clone still fetches through stored credentials.
  - A failed setup cell had already moved `data/` into a half-created clone directory; the datasets were moved back before relaunching.
- The 8 local LightGCN shards were killed once the T4 runner produced its first result; the local MF grids finish on CPU (15/16 and 9/16 cached at the time). T4 rate: one MF config about 96 s (python-side negative sampling dominates; the GPU does not help MF), LightGCN expected at a few minutes per config. First T4 line: `kuairand mf dim=128: val 0.1572, test 0.1616, unbiased 0.0063`.
- Open: Colab results come back as `.colab.json` files (local base files win on duplicate keys, so CPU numbers stay canonical); the final 3-seed runs for notebook 06 should all come from one device.

## 2026-10-10 (grids complete: notebooks 03 and 04 produced)
- All parameter grids finished: MF 16+16 configs on this machine (CPU), LightGCN 19+19 configs (16 grid + layers 0/1/4 at the selected setting) on the Colab T4, merged from `results/04_lightgcn_parameters_<ds>.colab.json`. The runner merges them automatically; nothing was rerun.
- Selected on validation Recall@20 only (seed 0), from the executed notebooks:
  - KuaiRand-Pure: BPR-MF dim=16 lr=1e-3 reg=1e-4 (val 0.1632, test 0.1666, unbiased 0.0058); LightGCN layers=3 dim=128 (val 0.2109, test 0.2179, unbiased 0.0063).
  - KuaiRec: BPR-MF dim=64 (val 0.1074, test 0.1051, unbiased 0.1710); LightGCN layers=3 dim=128 (val 0.1279, test 0.1255, unbiased 0.1634).
- Headline pattern (one seed, to be confirmed at 3 seeds in notebook 06): LightGCN's biased-test advantage is +31% (KuaiRand) and +19% (KuaiRec); on the unbiased tests it collapses to +0.0005 absolute recall on KuaiRand and inverts to -4% on KuaiRec. The layer curves (`results/04_layers.png`) tell the same story from another side: on KuaiRand the unbiased Recall@20 is flat at 0.0062-0.0063 for every layer count 0-4, while the biased test rises from 0.162 (layer 0 = MF) to a peak of 0.2179 at 3 layers.
- Notebooks 03 and 04 executed from cache and saved with outputs (03 in under a minute; 04 also regenerated `results/04_layers.png`).
- Next: 05 (ablations at these settings), 06 (3-seed headline + most-popular, ideally all on one device), 07 (analysis), 08 (density sweep). The T4 session is still alive and can run 05/06.

## 2026-10-10 (cleanup and member3 branch)
- Comment pass over the day's changes: device handling in `evaluate` and `set_graph` got one-line comments; `run_grid.load_runs` now documents that other-machine files (`.colab.json`) merge in and local keys win. No code changes; all imports verified after removing `__pycache__` folders.
- All work so far committed on the new branch `member3` (two commits: the GPU device support; the baseline, grid runner, notebooks 03/04 and their results). main stays at origin; the branch is ready to push when credentials are available.

## 2026-10-10 (member3 branch scoped to member 3)
- `member3` no longer carries notebook 03 (member 2's parameter study): the notebook is restored to the upstream stub and the two MF result JSONs are removed from the branch. The full state including them is preserved on the local branch `grids-backup` (commit ac3bb64), and the MF grids can be regenerated with `python3 src/experiments/run_grid.py 03 kuairand 03 kuairec` in about 1.5 h on CPU. Notebook 04 and the runner do not read any 03 file, so nothing else changes.
- `member3` also no longer carries the most-popular baseline (member 4's): `models/most_popular.py` is removed and `models/__init__.py` is restored to main's version, so member 4 commits the baseline on their own branch. Verified that nothing on this branch needs it: `run_grid.py` and notebook 04 do not reference `MostPopular`, and notebook 04's import surface (`data`, `models`, `training`, `inference`, `experiments.run_grid`) was re-imported successfully after the removal. `experiments/run_grid.py` stays on the branch: notebook 04 imports `run_grid`, `run_layers`, `select_best`, `layer_curve` and `merge_runs` from it, so it is a dependency of member 3's deliverable even though `experiments/` is member 4's folder; if member 4 commits the identical file, it merges cleanly.

## 2026-10-10 (analysis module for notebook 07)
- `analysis/groups.py`: `item_groups(ds, n=5)` (items ranked by training degree, split into equal-count buckets) and `user_groups(ds, n=3)` (the same over users: cold/medium/heavy terciles), plus `group_info` with each group's size and degree range for the report's group table. Training degrees only, so no test information enters the grouping.
- `analysis/breakdowns.py`: `topk(model, ds, split, k)` returns each user's top-k items, mirroring `inference.evaluate`'s ranking exactly (same masking: train+val on `test`, also the biased test on `test_unbiased`; same block-video candidate set on KuaiRec) rather than changing member 1's `evaluate` to return lists. Verified on KuaiRand seed 0 with a 2-epoch MF run: per-user recall computed from `topk` equals `evaluate`'s to the bit on both tests (max difference 0.0 over 25,689 and 21,515 users). `recall_by_item_group` (hits / positives per bucket, a micro-average over positives, so groups are weighted by their test support while the reported overall Recall@K stays the macro-average over users), `recall_by_user_group` (mean per-user recall per group, users without targets skipped) and `wins_and_failures` (the n largest recall gains and losses of one model over another, with training degree, for notebook 07's example users).
- `analysis/figures.py`: `plot_breakdowns(rows, labels)` renders the 07 figure from tidy rows (dataset, kind item/user, group, test, model, recall) as grouped bars, one row of panels per dataset and one column per breakdown; the notebook saves it with `utils.savefig`.
- Smoke numbers (2-epoch MF, KuaiRand seed 0, biased test): item-bucket recall 0.002/0.003/0.002/0.003/0.005 across the five buckets with test support 1,024/2,405/6,009/17,431/104,567 — 79% of test positives fall in the most-popular bucket, confirming the support skew notebook 07 must report. Micro recall 0.0042 vs macro 0.0041.

## 2026-10-10 (notebook 07: success and failure cases)
- Models: notebook 06 is not run yet, so notebook 07 trains both models itself at the settings 03/04 selected (seed 0, same protocol: batch 2048, max 100 epochs, early stopping on validation Recall@20). The four runs went to background processes in parallel (MF 237 s / 196 s, LightGCN 5,716 s / 8,155 s wall) and every number reproduces the notebooks 03 and 04 grids to four decimals, including LightGCN despite its grid having run on the Colab T4; per-user recall, top-20 lists and metrics are cached in `data/processed/07_cache_<ds>_<model>.npz` (not in git), so the notebook reruns from cache in seconds and retrains only if a cache is deleted.
- Item buckets (training-degree quintiles, equal item counts): on the biased tests both models hit almost nothing outside the most-popular bucket — KuaiRand Q5 recall 0.199 (MF) / 0.250 (LightGCN) against at most 0.013 in Q1-Q4; support 104,567 of 131k positives in Q5. LightGCN's delta grows monotonically with popularity (+0.001 Q1 to +0.051 Q5 on KuaiRand; +0.001 to +0.014 on KuaiRec): the graph's biased-test gain is mostly more popularity. On the unbiased tests that advantage disappears (KuaiRand Q5 0.0185 vs MF's 0.0196, overall +0.0005) or inverts (KuaiRec Q5 0.074 vs 0.087, overall -4%) even though LightGCN's unbiased coverage is far higher (0.43 vs 0.17 on KuaiRec). KuaiRand's unbiased support is spread almost evenly over the buckets (30k-52k each), the inverse of the biased test.
- User groups (training-degree terciles): the graph helps every group on KuaiRand's biased test (+0.06/+0.06/+0.04 cold/medium/heavy) and cold users most on KuaiRec's (+0.038, shrinking to +0.007 for heavy); on KuaiRec's unbiased test it loses in every group. KuaiRec's unbiased users are almost all cold or medium (1,284/123/1 users per tercile), so the heavy row there is a single user and must be read with its count.
- Example users: the largest wins and losses (±1.0 recall) all belong to users with a single test positive, so they are coin flips of users the models barely know; the notebook shows them with training degree.
- Improvement chosen (the rubric's "possible solutions" item): the tail is unserved — on KuaiRand's unbiased test 73% of positives sit in Q1-Q4 where both models' recall is at most 0.001, and LightGCN amplifies the popularity concentration instead of correcting it. Over-smoothing is not the binding constraint (recall peaks at 3 layers, notebook 04), so notebook 07 selects popularity-aware negative sampling (oversample popular items as negatives) as the fix to implement, to be reported as a separate row next to notebook 06's main comparison.
- Outputs: `results/07_failure_cases.json` (settings, overview with coverage/popularity/time, both breakdowns, group tables, example users) and `results/07_breakdown.png` (four panels: both datasets x item-popularity and user-degree, both models and both tests as grouped bars). Notebook executed with outputs saved; 8 code cells, no errors.


## 2026-10-10 (inline figure outputs carried a web address)
- Found while re-reviewing the branch before the PR: the figures *saved to* `results/` were clean (`metadata={"Software": None}`), but the copies embedded in the notebooks' outputs (produced by `plt.show()`) still contained matplotlib's web address in their PNG metadata — invisible to the zip's `grep -a http` because the PNG is base64 inside the .ipynb, but a URL in the package all the same. Notebooks 04 and 07 now save, close, and display the saved file (`IPython.display.Image(filename=...)`) so the inline copy is the clean file; both re-executed, all cells error-free. Notebook 02 on main has the same embedded address and needs the same two-line fix from its owner (or at packaging time, member 4).

## 2026-10-10 (member3 rebased on member 2's notebook 03; notebook 07's MF retrained at the canonical settings)
- Rebased `member3` onto main's three new commits (member 2's notebook 03, `results/03_mf_parameters.json`, `03_mf_validation.png`, LOG entry). Conflicts resolved: notebook 03 always taken from main (the branch's add-then-remove history for it cancels out), LOG.md merged as the union of both sides. Verified after the rebase: the branch diff touches no 03 file, LOG has no markers, and the whole `src` import surface still works.
- Member 2's notebook 03 supersedes the MF settings notebook 07 had used (from the `grids-backup` grid): the canonical selections are KuaiRand-Pure dim=64 lr=1e-3 reg=1e-3 (val 0.1673) and KuaiRec dim=64 lr=3e-4 reg=1e-4 (val 0.1084), both stronger on validation than the backup picks (0.1632 / 0.1074). Notebook 07's SETTINGS, its two MF caches and all its numbers were regenerated: the retrains reproduce member 2's published numbers exactly (KuaiRand 0.1673 / 0.1693 / 0.0060; KuaiRec 0.1084 / 0.1066 / 0.1731 for val / test / unbiased), so the two independent training paths agree.
- Revised notebook 07 findings under the canonical MF: LightGCN's biased-test margin narrows to +29% (KuaiRand) / +18% (KuaiRec) and still lands almost entirely in Q5 (+0.047 / +0.013, at most +0.003 in any other bucket); on KuaiRand's unbiased test no item bucket improves at all (+0.0003 overall), and KuaiRec's inversion deepens to -6% (Q5 0.074 vs 0.088), with LightGCN worse in every user group. New nuance: the dim-64 MF recommends *less* popular items than LightGCN on KuaiRand (mean degree 820 vs 923, coverage 0.62 vs 0.49), so the graph's edge is not aggregate popularity but converting popularity into hits. The improvement choice is unchanged: popularity-aware negative sampling.
- Notebook 04's grid markdown corrected: member 2's MF grid has 8 configurations, not 16 (fairness rule 2 asks for equal grids; the report's parameter sections must state the counts). CLAUDE.md status updated to the canonical settings and file names.

## 2026-10-10 (fairness rule 2 restored: 8-vs-8 selection grids)
- Notebook 04's LightGCN selection is now restricted to an 8-configuration grid — layers 2/3 x dim 64/128 x L2 1e-5/1e-4 with the learning rate fixed to 0.001, the value notebook 03's MF selected — the same count as member 2's 8-configuration MF grid (fairness rule 2), and one fewer difference between the models. The 8 lr=5e-3 runs already in the cache are relabelled `phase="sensitivity"` and excluded from selection: every one ranked strictly lower on validation than every lr=1e-3 run, so the restriction changes no selection — verified, `select_best` still picks layers=3 dim=128 lr=1e-3 reg=1e-4 on both datasets (val 0.2109 / 0.1279) and notebook 04 re-executed with identical outputs. Nothing downstream (05-08) changes.
- `experiments/run_grid.py` now encodes the protocol: `GRIDS["04"]` is the 8-config selection grid, `SENSITIVITY` holds the lr=5e-3 runs (phase "sensitivity", run outside the shards), `select_best` still filters `phase == "grid"`. A fresh clone reruns 8 + 8 + layer study and reproduces the same labels. The report's parameter sections must state both grids (8 MF configs around a dim=64 baseline; 8 LightGCN factorial configs at lr=1e-3) and mention the lr=5e-3 sensitivity sweep.
