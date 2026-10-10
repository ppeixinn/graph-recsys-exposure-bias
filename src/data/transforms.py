"""Density transforms for the density sweep: random subsampling of training interactions and k-core filtering."""
import numpy as np

from .splits import _group


def sparsify(ds, keep_frac, seed=0):
    """Keep a random keep_frac of the training interactions; all test sets, validation and the id space stay fixed."""
    if keep_frac >= 1:
        return ds
    rng = np.random.RandomState(seed)
    users = np.concatenate([np.full(len(v), u) for u, v in ds["train"].items()])
    items = np.concatenate(list(ds["train"].values()))
    keep = rng.permutation(len(users))[: int(round(len(users) * keep_frac))]
    train = {u: np.array([], dtype=int) for u in ds["train"]}
    train.update({u: np.array(v) for u, v in _group(users[keep], items[keep]).items()})
    return dict(ds, name=f"{ds['name']} [{keep_frac:.0%} train]", train=train)


def k_core(ds, k):
    """Keep users and items with at least k training interactions (iterated); ids are remapped, so the item set changes."""
    pairs = np.array([(u, i) for u, v in ds["train"].items() for i in v])
    while True:
        du = np.bincount(pairs[:, 0], minlength=ds["n_users"])
        di = np.bincount(pairs[:, 1], minlength=ds["n_items"])
        ok = (du[pairs[:, 0]] >= k) & (di[pairs[:, 1]] >= k)
        if ok.all():
            break
        pairs = pairs[ok]
    uid = {u: n for n, u in enumerate(np.unique(pairs[:, 0]))}
    iid = {i: n for n, i in enumerate(np.unique(pairs[:, 1]))}

    def remap(split):
        out = {uid[u]: np.array([iid[i] for i in v if i in iid], dtype=int) for u, v in split.items() if u in uid}
        return {u: v for u, v in out.items() if len(v)}

    train = {u: np.array(v) for u, v in _group([uid[u] for u in pairs[:, 0]], [iid[i] for i in pairs[:, 1]]).items()}
    out = dict(name=f"{ds['name']} [{k}-core]", n_users=len(uid), n_items=len(iid),
               train=train, val=remap(ds["val"]), test=remap(ds["test"]))
    for s in ("test_unbiased", "unbiased_exposed"):
        if s in ds:
            out[s] = remap(ds[s])
    if "unbiased_items" in ds:
        out["unbiased_items"] = np.array([iid[i] for i in ds["unbiased_items"] if i in iid])
    return out
