"""Dataset loading: builds implicit-feedback splits as {user: np.array(items)}.

Every dataset has `train`, `val` and `test` (random per-user split of the biased logs). KuaiRand-Pure and KuaiRec
also have `test_unbiased`: positives from an exposure that the platform recommender did not choose."""
import os
import zipfile
from functools import lru_cache

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
KUAI_DIR = os.path.join(DATA_DIR, "KuaiRand-Pure", "data")
KUAI_STANDARD = ["log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"]
KUAI_RANDOM = "log_random_4_22_to_5_08_pure.csv"
KUAIREC_ZIP = os.path.join(DATA_DIR, "KuaiRec.zip")
KUAIREC_FILES = "KuaiRec 2.0/data/"


def _split_user_items(user_items, rng, val_frac=0.1):
    """Hold out val_frac of each user's training items (at least 1 if the user has >=3) for validation."""
    train, val = {}, {}
    for u, items in user_items.items():
        items = np.array(items)
        rng.shuffle(items)
        n_val = int(round(len(items) * val_frac))
        n_val = max(n_val, 1) if len(items) >= 3 else 0
        val[u] = items[:n_val]
        train[u] = items[n_val:]
    return train, val


def _random_user_split(by_user, rng, test_frac):
    """Per-user random split of {user: items} into (train, test); every user keeps at least one test item."""
    train, test = {}, {}
    for u, its in by_user.items():
        its = np.array(its)
        if len(its) < 2:
            continue
        rng.shuffle(its)
        n_test = max(1, int(round(len(its) * test_frac)))
        test[u] = its[:n_test]
        train[u] = its[n_test:]
    return train, test


def _group(users, items):
    by_user = {}
    for u, i in zip(users, items):
        by_user.setdefault(int(u), []).append(int(i))
    return by_user


def _build(name, users, items, seed, test_frac):
    """Split unique positive (user, item) pairs. Users with fewer than 2 positives are dropped, ids are remapped.

    Returns the dataset and the raw-id -> index maps of users and items."""
    rng = np.random.RandomState(seed)
    users, items = np.asarray(users), np.asarray(items)
    raw_users, counts = np.unique(users, return_counts=True)
    keep = np.isin(users, raw_users[counts >= 2])
    users, items = users[keep], items[keep]
    uid = {u: k for k, u in enumerate(np.unique(users))}
    iid = {i: k for k, i in enumerate(np.unique(items))}
    by_user = _group([uid[u] for u in users], [iid[i] for i in items])
    train_all, test = _random_user_split(by_user, rng, test_frac)
    train, val = _split_user_items(train_all, rng)
    return dict(name=name, n_users=len(uid), n_items=len(iid), train=train, val=val, test=test), uid, iid


def _unbiased(pos, uid, iid, seen):
    """Unbiased test targets: positives of known users on known items, without pairs in `seen` (biased positives)."""
    pos = pos[pos.user_id.isin(uid) & pos.video_id.isin(iid)]
    pairs = [(uid[u], iid[i]) for u, i in zip(pos.user_id, pos.video_id) if (u, i) not in seen]
    return {u: np.array(v) for u, v in _group(*zip(*pairs)).items()}


def load_movielens(seed=0, min_rating=4, test_frac=0.2):
    """MovieLens-100K (smoke tests only): ratings >= min_rating are positives; random per-user 80/20 split,
    then 10% of train -> val. No unbiased test set."""
    raw = np.loadtxt(os.path.join(DATA_DIR, "ml-100k", "u.data"), dtype=int)
    raw = raw[raw[:, 2] >= min_rating]
    return _build("MovieLens-100K", raw[:, 0], raw[:, 1], seed, test_frac)[0]


@lru_cache(maxsize=None)
def _kuairand_positives(positive):
    import pandas as pd
    cols = ["user_id", "video_id", positive]
    std = pd.concat([pd.read_csv(os.path.join(KUAI_DIR, f), usecols=cols) for f in KUAI_STANDARD])
    rnd = pd.read_csv(os.path.join(KUAI_DIR, KUAI_RANDOM), usecols=cols)
    return tuple(df[df[positive] == 1].drop_duplicates(["user_id", "video_id"])[cols[:2]] for df in (std, rnd))


def load_kuairand(seed=0, positive="is_click", test_frac=0.2):
    """KuaiRand-Pure. A positive is a row with `positive` == 1 (is_click, is_like or long_view).

    train/val/test: per-user random split of the two standard (recommender-exposed) logs.
    test_unbiased:  positives in the randomly exposed log, for the same users and items, without pairs that are
                    also positive in the standard logs."""
    std, rnd = _kuairand_positives(positive)
    ds, uid, iid = _build(f"KuaiRand-Pure ({positive})", std.user_id.values, std.video_id.values, seed, test_frac)
    ds["test_unbiased"] = _unbiased(rnd, uid, iid, set(zip(std.user_id, std.video_id)))
    return ds


@lru_cache(maxsize=None)
def _kuairec_positives(min_ratio):
    import pandas as pd
    cols = ["user_id", "video_id", "watch_ratio"]
    with zipfile.ZipFile(KUAIREC_ZIP) as z:
        big, small = (pd.read_csv(z.open(KUAIREC_FILES + f), usecols=cols) for f in ("big_matrix.csv", "small_matrix.csv"))
    block = np.sort(small.video_id.unique())
    return tuple(df[df.watch_ratio >= min_ratio].drop_duplicates(["user_id", "video_id"])[cols[:2]] for df in (big, small)) + (block,)


def load_kuairec(seed=0, min_ratio=2.0, test_frac=0.2):
    """KuaiRec. A positive is a (user, video) pair with at least one view of watch_ratio >= min_ratio.

    train/val/test: per-user random split of the big matrix (recommender-exposed logs).
    test_unbiased:  positives in the small matrix, where almost every user saw every video of a fixed block.
    unbiased_items: the videos of that block; the unbiased test ranks only among them."""
    big, small, block = _kuairec_positives(min_ratio)
    ds, uid, iid = _build(f"KuaiRec (watch_ratio >= {min_ratio:g})", big.user_id.values, big.video_id.values, seed, test_frac)
    ds["test_unbiased"] = _unbiased(small, uid, iid, set())
    ds["unbiased_items"] = np.array([iid[i] for i in block if i in iid])
    return ds


def load(name, **kw):
    return {"ml100k": load_movielens, "kuairand": load_kuairand, "kuairec": load_kuairec}[name](**kw)


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
    if "test_unbiased" in ds:
        out["test_unbiased"] = remap(ds["test_unbiased"])
    if "unbiased_items" in ds:
        out["unbiased_items"] = np.array([iid[i] for i in ds["unbiased_items"] if i in iid])
    return out


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
