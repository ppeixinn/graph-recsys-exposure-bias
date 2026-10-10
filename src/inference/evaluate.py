"""Top-K evaluation: Recall@K, NDCG@K, coverage, popularity of recommendations, per-user results."""
import numpy as np
import torch

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
