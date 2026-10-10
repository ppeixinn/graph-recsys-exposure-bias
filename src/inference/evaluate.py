"""Top-K evaluation: Recall@K, NDCG@K, coverage, popularity of recommendations, per-user results; GAUC and NDCG@K
over each user's shown videos on the unbiased test."""
import numpy as np
import torch
from scipy.stats import rankdata

from data.splits import to_pairs


@torch.no_grad()
def evaluate(model, ds, split, k=20, mask_splits=("train",)):
    """Full-ranking evaluation. Returns metrics, per-user recall (NaN if no targets), item recommendation counts.

    `popularity` is the mean training degree of the recommended items (popularity bias). On `test_unbiased`, a
    dataset with `unbiased_items` ranks only among those items, and coverage is measured over them."""
    model.eval()
    U, I = model.embeddings()
    targets = ds[split]
    users = np.array([u for u, v in targets.items() if len(v)])
    rec = np.full(ds["n_users"], np.nan)
    ndcg = np.full(ds["n_users"], np.nan)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rec_items = np.zeros(ds["n_items"])
    candidates = ds.get("unbiased_items") if split == "test_unbiased" else None
    outside = None
    if candidates is not None:
        outside = np.ones(ds["n_items"], dtype=bool)
        outside[candidates] = False
    for s in range(0, len(users), 1024):
        batch = users[s:s + 1024]
        # score on the embeddings' device, move each batch's scores back to CPU once
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
            t = set(targets[u].tolist())
            hits = np.array([x in t for x in top[r]], dtype=float)
            rec[u] = hits.sum() / len(t)
            ndcg[u] = (hits * disc).sum() / disc[: min(len(t), k)].sum()
            rec_items[top[r]] += 1
    item_deg = np.bincount(to_pairs(ds["train"])[:, 1], minlength=ds["n_items"])
    pool = rec_items if candidates is None else rec_items[candidates]
    m = dict(recall=float(np.nanmean(rec)), ndcg=float(np.nanmean(ndcg)), coverage=float((pool > 0).mean()),
             popularity=float((rec_items * item_deg).sum() / rec_items.sum()))
    return m, rec, rec_items


@torch.no_grad()
def evaluate_exposed(model, ds, k=5, mask_splits=("train", "val", "test")):
    """Exposure-only evaluation of `test_unbiased`: each user's candidates are the videos they were shown
    (`unbiased_exposed`), positives are their unbiased targets, every other shown video is an observed negative.

    Returns GAUC (per-user AUC averaged over users with at least one positive and one negative; ties count half,
    random = 0.5), NDCG@k over the shown videos, the user counts behind each, and per-user AUC (NaN if undefined)."""
    model.eval()
    U, I = model.embeddings()
    targets, shown = ds["test_unbiased"], ds["unbiased_exposed"]
    users = np.array([u for u, v in targets.items() if len(v) and u in shown])
    auc = np.full(ds["n_users"], np.nan)
    ndcg = np.full(ds["n_users"], np.nan)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for s in range(0, len(users), 1024):
        batch = users[s:s + 1024]
        scores = (U[torch.as_tensor(batch, device=U.device)] @ I.T).float().cpu().numpy()
        for r, u in enumerate(batch):
            masked = np.concatenate([ds[ms][u] for ms in mask_splits if u in ds[ms]] + [np.array([], dtype=int)])
            cand = np.setdiff1d(shown[u], masked)
            pos = np.isin(cand, targets[u])
            n_pos = pos.sum()
            if n_pos == 0:
                continue
            sc = scores[r, cand]
            top = np.argsort(-sc, kind="stable")[:k]
            ndcg[u] = (pos[top] * disc[:len(top)]).sum() / disc[: min(n_pos, k)].sum()
            if n_pos < len(cand):
                ranks = rankdata(sc)
                auc[u] = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(cand) - n_pos))
    m = dict(gauc=float(np.nanmean(auc)), ndcg=float(np.nanmean(ndcg)), k=k,
             users_gauc=int((~np.isnan(auc)).sum()), users_ndcg=int((~np.isnan(ndcg)).sum()))
    return m, auc
