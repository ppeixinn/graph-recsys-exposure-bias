"""Dataset statistics: degrees, split sizes, density, cold users and items."""
import numpy as np


def degrees(ds, split="train"):
    """User and item degree arrays (number of interactions) in one split."""
    user_deg = np.zeros(ds["n_users"], dtype=int)
    item_deg = np.zeros(ds["n_items"], dtype=int)
    for u, v in ds[split].items():
        user_deg[u] = len(v)
        np.add.at(item_deg, v, 1)
    return user_deg, item_deg


def stats(ds):
    """Sizes of the splits, training density, and test users or items that have no training interaction."""
    size = {s: sum(len(v) for v in ds[s].values()) for s in ("train", "val", "test", "test_unbiased") if s in ds}
    user_deg, item_deg = degrees(ds)
    test_items = np.unique(np.concatenate(list(ds["test"].values())))
    return dict(name=ds["name"], users=ds["n_users"], items=ds["n_items"], **size,
                density=size["train"] / (ds["n_users"] * ds["n_items"]),
                cold_users=int(sum(user_deg[u] == 0 for u, v in ds["test"].items() if len(v))),
                cold_items=int((item_deg[test_items] == 0).sum()))
