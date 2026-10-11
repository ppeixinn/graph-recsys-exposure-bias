"""Parameter grids for notebooks 03 (BPR-MF) and 04 (LightGCN), and the 3-seed runs of notebooks 05 (ablation)
and 06 (main comparison), cached in results/ per dataset.

Each finished run is stored under a key of dataset, config and seed; rerunning skips finished keys, so an
interrupted grid resumes where it stopped and several machines can each run their own dataset file.
Selection uses validation Recall@20 only; the unbiased test is never looked at (fairness rule 9).

Standalone use from the project root (e.g. under nohup, to fill the cache without a notebook):
    python src/experiments/run_grid.py 03 kuairand 03 kuairec --threads=8
    python src/experiments/run_grid.py 06 kuairand 05 kuairand --seeds=0,1,2 --shard=0/4 --threads=2
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import training  # noqa: E402  (import after the sys.path insert above)
from utils.paths import RESULTS_DIR  # noqa: E402

SHARED = dict(batch=2048, epochs=100)

# Selection grids (fairness rule 2: equally sized). Notebook 04 selects over 8 configurations, the same
# count as notebook 03's BPR-MF grid, with lr fixed to 1e-3 (the value MF selected) so learning rate is not
# a second difference between the models. The 8 additional lr=5e-3 LightGCN runs are a sensitivity check,
# not selection budget: they ranked strictly lower on validation and select_best ignores them.
# The layer sweep is a separate study on top of the grid, not extra tuning budget.
GRIDS = {
    "03": [dict(model="mf", dim=d, lr=l, reg=r, **SHARED)
           for d in (16, 32, 64, 128) for l in (1e-3, 5e-3) for r in (1e-5, 1e-4)],
    "04": [dict(model="lightgcn", layers=k, dim=d, lr=1e-3, reg=r, **SHARED)
           for k in (2, 3) for d in (64, 128) for r in (1e-5, 1e-4)],
}
SENSITIVITY = {"04": [dict(model="lightgcn", layers=k, dim=d, lr=5e-3, reg=r, **SHARED)
                      for k in (2, 3) for d in (64, 128) for r in (1e-5, 1e-4)]}
LAYER_STUDY = (0, 1, 4)  # completes the curve 0-4; layers 2 and 3 are already in the grid
# Notebooks 05/06: the validation selections of notebooks 03 (BPR-MF) and 04 (LightGCN), per dataset.
SELECTED = {
    "kuairand": dict(mf=dict(model="mf", dim=64, lr=1e-3, reg=1e-3, **SHARED),
                     lightgcn=dict(model="lightgcn", layers=3, dim=128, lr=1e-3, reg=1e-4, **SHARED)),
    "kuairec": dict(mf=dict(model="mf", dim=64, lr=3e-4, reg=1e-4, **SHARED),
                    lightgcn=dict(model="lightgcn", layers=3, dim=128, lr=1e-3, reg=1e-4, **SHARED)),
}
# Fairness rule 3 control: LightGCN at BPR-MF's dim=64 (best dim-64 grid config of notebook 04 on both datasets).
LIGHTGCN_DIM64 = dict(model="lightgcn", layers=3, dim=64, lr=1e-3, reg=1e-4, **SHARED)
# Notebook 05: one change at a time from the selected LightGCN; its baseline is notebook 06's LightGCN run.
ABLATIONS = {"last_layer": dict(combine="last"), "no_norm": dict(norm="none"), "edge_drop": dict(edge_drop=0.1)}
GRIDS["06"] = {ds: [s["mf"], s["lightgcn"], LIGHTGCN_DIM64] for ds, s in SELECTED.items()}
GRIDS["05"] = {ds: [dict(s["lightgcn"], **a) for a in ABLATIONS.values()] for ds, s in SELECTED.items()}
FILES = {"03": "03_mf_parameters_{ds}.json", "04": "04_lightgcn_parameters_{ds}.json",
         "05": "05_ablation_{ds}.json", "06": "06_main_comparison_{ds}.json"}

_DS = {}


def _dataset(name, seed):
    if (name, seed) not in _DS:
        import data
        try:
            _DS[(name, seed)] = data.load_processed(name, seed=seed)  # written by notebook 02
        except FileNotFoundError:
            _DS[(name, seed)] = data.load(name, seed=seed)
    return _DS[(name, seed)]


def _runs_path(notebook, dataset):
    return os.path.join(RESULTS_DIR, FILES[notebook].format(ds=dataset))


def load_runs(notebook, dataset):
    """Finished runs of one notebook and dataset: the base file plus shard files and files produced on
    another machine (e.g. `*.colab.json` from the GPU run), all merged. Local keys win on duplicates."""
    runs = {}
    for p in sorted(glob.glob(_runs_path(notebook, dataset)[:-len(".json")] + "*.json")):
        runs.update(json.load(open(p)))
    return runs


def merge_runs(notebook, datasets):
    """Runs of several datasets in one dict (keys already carry the dataset name)."""
    runs = {}
    for ds in datasets:
        runs.update(load_runs(notebook, ds))
    return runs


def _key(dataset, cfg, seed):
    parts = ",".join(f"{k}={cfg[k]}" for k in ("model", "layers", "dim", "lr", "reg", "epochs", "combine", "norm", "edge_drop")
                     if k in cfg)
    return f"{dataset}|{parts}|seed={seed}"


def _run_one(runs, notebook, dataset, cfg, seed, phase, eval_every, threads, out_path, device=None):
    key = _key(dataset, cfg, seed)
    if key in runs:
        return
    cfg = dict(cfg, threads=threads)
    if device:
        cfg["device"] = device
    res, _, _, _ = training.fit(cfg, _dataset(dataset, seed), seed=seed, eval_every=eval_every)
    runs[key] = dict(dataset=dataset, phase=phase, cfg=res["cfg"], seed=seed,
                     val_recall=res["val_recall"], best_epoch=res["best_epoch"], train_time=res["train_time"],
                     test=res["test"], test_unbiased=res.get("test_unbiased"), test_exposed=res.get("test_exposed"),
                     epochs_run=len(res["history"]))
    with open(out_path, "w") as f:
        json.dump(runs, f, indent=1)
    unb = (res.get("test_unbiased") or {}).get("recall")
    print(f"{key}: val {res['val_recall']:.4f}, test {res['test']['recall']:.4f}, "
          f"unbiased {unb if unb is None else round(unb, 4)}, {res['train_time']:.0f}s", flush=True)


def run_grid(notebook, dataset, seeds=(0,), grid=None, eval_every=5, threads=8, shard=None, device=None):
    """Run every config of the grid on the dataset, caching each finished run. Returns the dataset's runs.
    With shard='i/n', run only slice i of the grid into the dataset's shard file: torch's CPU propagation
    is effectively single-threaded, so several shard processes use the cores of one machine properly."""
    runs = load_runs(notebook, dataset)
    cfgs = grid or GRIDS[notebook]
    if isinstance(cfgs, dict):  # notebooks 05/06 have per-dataset configs
        cfgs = cfgs[dataset]
    jobs = [(cfg, seed) for cfg in cfgs for seed in seeds]
    out = _runs_path(notebook, dataset)
    if shard is not None:
        i, n = (int(x) for x in shard.split("/"))
        jobs, out = jobs[i::n], out.replace(".json", f".shard{i}.json")
    for cfg, seed in jobs:
        _run_one(runs, notebook, dataset, cfg, seed, "grid", eval_every, threads, out, device)
    if shard is None:  # sensitivity runs are not selection budget and stay out of the shards
        for cfg in SENSITIVITY.get(notebook, []):
            for seed in seeds:
                _run_one(runs, notebook, dataset, cfg, seed, "sensitivity", eval_every, threads, out, device)
    return runs


def select_best(runs, dataset, seed=0):
    """Highest validation Recall@20 among the dataset's grid runs (phase='grid'; the sensitivity runs and
    the layer study are not selection budget, and the unbiased test is never used)."""
    rs = [r for r in runs.values() if r["dataset"] == dataset and r["seed"] == seed and r["phase"] == "grid"]
    return max(rs, key=lambda r: r["val_recall"])


def run_layers(dataset, seeds=(0,), eval_every=5, threads=8, device=None):
    """At the best grid config of notebook 04, run the missing layers of the over-smoothing curve."""
    runs = load_runs("04", dataset)
    for seed in seeds:
        cfg = select_best(runs, dataset, seed)["cfg"]
        for k in LAYER_STUDY:
            _run_one(runs, "04", dataset, dict(cfg, layers=k), seed, "layers", eval_every, threads,
                     _runs_path("04", dataset), device)
    return runs


def layer_curve(dataset, seed=0):
    """Layers 0-4 at the selected config: validation, biased-test and unbiased-test Recall@20.
    Missing entries (grid still running) are NaN."""
    import numpy as np
    runs = load_runs("04", dataset)
    cfg = select_best(runs, dataset, seed)["cfg"]
    rows = []
    for k in (0, 1, 2, 3, 4):
        r = runs.get(_key(dataset, dict(cfg, layers=k), seed))
        rows.append(dict(layers=k, dim=cfg["dim"], lr=cfg["lr"], reg=cfg["reg"],
                         val_recall=r["val_recall"] if r else np.nan,
                         test_recall=r["test"]["recall"] if r else np.nan,
                         unbiased_recall=r["test_unbiased"]["recall"] if r else np.nan))
    return rows


def main(argv):
    """Usage: python src/experiments/run_grid.py 03 kuairand 04 kuairand [--threads=8] [--shard=i/n] [--device=cuda]
    [--seeds=0,1,2]. Shard processes run grid slices only; run without --shard afterwards to add the 04 layer study."""
    threads, shard, device, seeds, pairs = 8, None, None, (0,), []
    for a in argv[1:]:
        if a.startswith("--seeds="):
            seeds = tuple(int(x) for x in a.split("=")[1].split(","))
        elif a.startswith("--threads="):
            threads = int(a.split("=")[1])
        elif a.startswith("--shard="):
            shard = a.split("=")[1]
        elif a.startswith("--device="):
            device = a.split("=")[1]
        else:
            pairs.append(a)
    for notebook, dataset in zip(pairs[0::2], pairs[1::2]):
        run_grid(notebook, dataset, seeds=seeds, threads=threads, shard=shard, device=device)
        if notebook == "04" and shard is None:
            run_layers(dataset, threads=threads, device=device)


if __name__ == "__main__":
    main(sys.argv)
