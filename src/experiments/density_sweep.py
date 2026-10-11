"""Density sweep for notebook 08: both models at their selected settings on sparser training graphs, cached in results/.

Variants are `keep=<frac>` (random subsampling of training pairs; validation, both tests and the id space stay
fixed) and `core=<k>` (k-core filter; the item set and the test sets change, so it is a separate variant).
Each process writes one file per dataset and variant, so several processes can run side by side; finished keys
are skipped, so an interrupted sweep resumes where it stopped.

Standalone use from the project root (one process per chain of variants):
    python src/experiments/density_sweep.py kuairand keep=1.0 --threads=1
    python src/experiments/density_sweep.py kuairec keep=0.1 core=10 --threads=1
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data  # noqa: E402  (import after the sys.path insert above)
import inference  # noqa: E402
import training  # noqa: E402
from experiments.run_grid import load_runs, select_best  # noqa: E402
from utils.paths import RESULTS_DIR  # noqa: E402

LEVELS = (1.0, 0.5, 0.25, 0.1)
CORES = (5, 10)
VARIANTS = [f"keep={x}" for x in LEVELS] + [f"core={k}" for k in CORES]
MODELS = ("mf", "lightgcn")


def selected_configs(dataset):
    """The settings notebooks 03 (BPR-MF) and 04 (LightGCN) selected on validation, without machine settings."""
    with open(os.path.join(RESULTS_DIR, "03_mf_parameters.json")) as f:
        mf = next(r["config"] for r in json.load(f)["selected"] if r["dataset"] == dataset)
    lgcn = select_best(load_runs("04", dataset), dataset)["cfg"]
    drop = ("threads", "device")
    return {"mf": {k: v for k, v in mf.items() if k not in drop},
            "lightgcn": {k: v for k, v in lgcn.items() if k not in drop}}


def variant(ds, name, seed):
    """Apply one variant (`keep=0.25`, `core=5`) to a full dataset."""
    kind, x = name.split("=")
    return data.sparsify(ds, float(x), seed=seed) if kind == "keep" else data.k_core(ds, int(x))


def graph_stats(ds):
    """`data.stats` plus the users and items left with training pairs and the cold users of the unbiased test."""
    user_deg, item_deg = data.degrees(ds)
    s = data.stats(ds)
    s.update(train_users=int((user_deg > 0).sum()), train_items=int((item_deg > 0).sum()))
    if "test_unbiased" in ds:
        s["cold_users_unbiased"] = int(sum(user_deg[u] == 0 for u, v in ds["test_unbiased"].items() if len(v)))
    return s


def warm_cold(values, user_deg):
    """Mean of a per-user metric over users with and without training pairs (None if a group is empty)."""
    has = ~np.isnan(values)
    warm, cold = has & (user_deg > 0), has & (user_deg == 0)
    return (float(values[warm].mean()) if warm.any() else None,
            float(values[cold].mean()) if cold.any() else None)


def _path(dataset, name):
    return os.path.join(RESULTS_DIR, f"08_density_{dataset}_{name}.json")


def load_sweep():
    """Every finished run of the sweep, all datasets and variants merged."""
    runs = {}
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "08_density_*_*.json"))):
        with open(p) as f:
            runs.update(json.load(f))
    return runs


def run_variant(dataset, name, seed=0, threads=8):
    """Train both models on one variant of the dataset; each finished run is written immediately."""
    path = _path(dataset, name)
    runs = json.load(open(path)) if os.path.exists(path) else {}
    cfgs = selected_configs(dataset)
    ds = None
    for model in MODELS:
        key = f"{dataset}|{name}|{model}|seed={seed}"
        if key in runs:
            continue
        if ds is None:
            ds = variant(data.load_processed(dataset, seed), name, seed)
        res, rec, _, net = training.fit(dict(cfgs[model], threads=threads), ds, seed=seed)
        user_deg, _ = data.degrees(ds)
        _, auc = inference.evaluate_exposed(net, ds)
        rw, rc = warm_cold(rec, user_deg)
        aw, ac = warm_cold(auc, user_deg)
        runs[key] = dict(dataset=dataset, variant=name, model=model, seed=seed, cfg=res["cfg"], stats=graph_stats(ds),
                         val_recall=res["val_recall"], best_epoch=res["best_epoch"],
                         epochs_run=len(res["history"]), train_time=res["train_time"], test=res["test"],
                         test_unbiased=res.get("test_unbiased"), test_exposed=res.get("test_exposed"),
                         recall_warm=rw, recall_cold=rc, gauc_warm=aw, gauc_cold=ac)
        with open(path, "w") as f:
            json.dump(runs, f, indent=1)
        print(f"{key}: val {res['val_recall']:.4f}, test {res['test']['recall']:.4f}, "
              f"GAUC {res['test_exposed']['gauc']:.4f}, best epoch {res['best_epoch']}, "
              f"{res['train_time']:.0f}s", flush=True)
    return runs


def main(argv):
    """Usage: python src/experiments/density_sweep.py <dataset> <variant> [<variant> ...] [--seed=0] [--threads=8]"""
    seed, threads, args = 0, 8, []
    for a in argv[1:]:
        if a.startswith("--seed="):
            seed = int(a.split("=")[1])
        elif a.startswith("--threads="):
            threads = int(a.split("=")[1])
        else:
            args.append(a)
    for name in args[1:]:
        run_variant(args[0], name, seed=seed, threads=threads)


if __name__ == "__main__":
    main(sys.argv)
