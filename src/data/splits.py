"""Split rules shared by every loader: per-user random train/val/test split and the unbiased test targets."""
import numpy as np


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


def to_pairs(d):
    """{user: items} -> array of (user, item) rows."""
    return np.array([(u, i) for u, its in d.items() for i in its], dtype=np.int64)


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


def _shown(log, uid, iid):
    """Videos each known user was shown in an unbiased log, clicked or not: {user: items}, known items only."""
    log = log[log.user_id.isin(uid) & log.video_id.isin(iid)]
    return {u: np.array(v) for u, v in _group([uid[u] for u in log.user_id], [iid[i] for i in log.video_id]).items()}


def _unbiased(pos, uid, iid, seen):
    """Unbiased test targets: positives of known users on known items, without pairs in `seen` (biased positives)."""
    pos = pos[pos.user_id.isin(uid) & pos.video_id.isin(iid)]
    pairs = [(uid[u], iid[i]) for u, i in zip(pos.user_id, pos.video_id) if (u, i) not in seen]
    return {u: np.array(v) for u, v in _group(*zip(*pairs)).items()}
