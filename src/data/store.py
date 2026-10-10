"""Processed datasets on disk: data/processed/<name>_seed<seed>.npz, written by notebook 02 and read by load_processed."""
import os

import numpy as np

from utils.paths import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLITS = ("train", "val", "test", "test_unbiased")


def processed_path(name, seed):
    return os.path.join(PROCESSED_DIR, f"{name}_seed{seed}.npz")


def save_processed(ds, name, seed):
    """Write a dataset from `load` to data/processed/; user order and item order inside each split are kept."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    arrays = dict(name=np.array(ds["name"]), n_users=np.array(ds["n_users"]), n_items=np.array(ds["n_items"]))
    for s in SPLITS:
        if s in ds:
            arrays[f"{s}_users"] = np.array(list(ds[s]), dtype=np.int64)
            arrays[f"{s}_lengths"] = np.array([len(v) for v in ds[s].values()], dtype=np.int64)
            arrays[f"{s}_items"] = np.concatenate([np.asarray(v, dtype=np.int64) for v in ds[s].values()])
    if "unbiased_items" in ds:
        arrays["unbiased_items"] = np.asarray(ds["unbiased_items"], dtype=np.int64)
    path = processed_path(name, seed)
    np.savez_compressed(path, **arrays)
    return path


def load_processed(name, seed=0):
    """Read data/processed/<name>_seed<seed>.npz into the same dict as `load(name, seed=seed)`, in about a second."""
    with np.load(processed_path(name, seed)) as z:
        ds = dict(name=str(z["name"]), n_users=int(z["n_users"]), n_items=int(z["n_items"]))
        for s in SPLITS:
            if f"{s}_users" in z:
                parts = np.split(z[f"{s}_items"], np.cumsum(z[f"{s}_lengths"])[:-1])
                ds[s] = {int(u): v for u, v in zip(z[f"{s}_users"], parts)}
        if "unbiased_items" in z:
            ds["unbiased_items"] = z["unbiased_items"]
    return ds
