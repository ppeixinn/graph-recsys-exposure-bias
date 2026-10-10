"""Recall by item-popularity bucket (over positives) and by user-degree group (over users), plus the example users."""
import numpy as np
import torch

from data import degrees


@torch.no_grad()
def topk(model, ds, split, k=20):
    """Top-k items per user, mirroring inference.evaluate's ranking (same masking and candidate set), so the
    breakdowns line up with the reported Recall@K. `test` masks train+val, `test_unbiased` also the biased test."""
    mask_splits = {"test": ("train", "val"), "test_unbiased": ("train", "val", "test")}.get(split, ("train",))
    model.eval()
    U, I = model.embeddings()
    targets = ds[split]
    users = np.array([u for u, v in targets.items() if len(v)])
    candidates = ds.get("unbiased_items") if split == "test_unbiased" else None
    outside = None
    if candidates is not None:
        outside = np.ones(ds["n_items"], dtype=bool)
        outside[candidates] = False
    tops = {}
    for s in range(0, len(users), 1024):
        batch = users[s:s + 1024]
        scores = (U[torch.as_tensor(batch, device=U.device)] @ I.T).float().cpu().numpy()
        if outside is not None:
            scores[:, outside] = -np.inf
        for r, u in enumerate(batch):
            for ms in mask_splits:
                if u in ds[ms] and len(ds[ms][u]):
                    scores[r, ds[ms][u]] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
        top = np.take_along_axis(top, order, 1)
        for r, u in enumerate(batch):
            tops[u] = top[r]
    return tops


def recall_by_item_group(ds, split, tops, groups, k=20):
    """Recall over each item group's positives: hits / positives (a micro-average over positives, so it weights
    groups by their test support; the reported overall Recall@K stays the macro-average over users)."""
    hits = np.zeros(groups.max() + 1, dtype=int)
    pos = np.zeros(groups.max() + 1, dtype=int)
    for u, items in ds[split].items():
        if not len(items) or u not in tops:
            continue
        np.add.at(pos, groups[items], 1)
        np.add.at(hits, groups[items], np.isin(items, tops[u][:k]).astype(int))
    return [dict(group=int(g), positives=int(pos[g]), hits=int(hits[g]),
                 recall=float(hits[g] / pos[g]) if pos[g] else None)
            for g in range(groups.max() + 1)]


def recall_by_user_group(rec, groups):
    """Mean per-user recall in each degree group; users with no targets in the split (NaN) are skipped."""
    return [dict(group=int(g), users=int((~np.isnan(r)).sum()), recall=float(r[~np.isnan(r)].mean()) if (~np.isnan(r)).any() else None)
            for g in range(groups.max() + 1)
            if len(r := rec[groups == g])]


def wins_and_failures(rec_a, rec_b, ds=None, n=5):
    """The n users with the largest recall gain of b over a and the n with the largest loss: notebook 07's example
    users, with training degree when the dataset is given."""
    d = rec_b - rec_a
    ok = ~np.isnan(d)
    user_deg = degrees(ds)[0] if ds is not None else None
    rows = []
    for sign, kind in ((1, "win"), (-1, "failure")):
        idx = np.where(ok)[0]
        for u in idx[np.argsort(sign * d[idx])[::-1][:n]]:
            rows.append(dict(user=int(u), kind=kind, recall_a=float(rec_a[u]), recall_b=float(rec_b[u]),
                             delta=float(d[u]), train_degree=int(user_deg[u]) if user_deg is not None else None))
    return rows
