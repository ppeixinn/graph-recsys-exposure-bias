"""BPR training loop with early stopping; `evaluate` lives in inference/."""
import time
import numpy as np
import torch

from data import to_pairs
from inference import evaluate
from models import MF, LightGCN


def make_model(cfg, ds):
    if cfg["model"] == "mf":
        return MF(ds["n_users"], ds["n_items"], cfg.get("dim", 64))
    return LightGCN(ds["n_users"], ds["n_items"], cfg.get("dim", 64), cfg.get("layers", 3),
                    cfg.get("combine", "mean"), cfg.get("norm", "sym"))


def fit(cfg, ds, seed=0, verbose=False, eval_every=5, patience=4):
    """Train with BPR loss; keep the epoch with the best validation Recall@20; report test metrics at that epoch."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    torch.set_num_threads(cfg.get("threads", 8))
    model = make_model(cfg, ds).to(cfg.get("device", "cpu"))
    dev = next(model.parameters()).device
    pairs = to_pairs(ds["train"])
    n_items = ds["n_items"]
    train_sets = {u: set(v.tolist()) for u, v in ds["train"].items()}
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    reg, bs, edge_drop = cfg.get("reg", 1e-4), cfg.get("batch", 2048), cfg.get("edge_drop", 0.0)
    model.set_graph(pairs, 0.0, rng)
    best = dict(val=-1, epoch=0, state=None)
    hist, t0, bad = [], time.time(), 0
    for ep in range(1, cfg.get("epochs", 100) + 1):
        model.train()
        if edge_drop > 0:
            model.set_graph(pairs, edge_drop, rng)
        perm = rng.permutation(len(pairs))
        tot = 0.0
        for s in range(0, len(pairs), bs):
            b = pairs[perm[s:s + bs]]
            u, pos = b[:, 0], b[:, 1]
            neg = rng.randint(0, n_items, size=len(b))
            for j in range(len(b)):  # resample negatives that are actually positives
                while neg[j] in train_sets[u[j]]:
                    neg[j] = rng.randint(0, n_items)
            u, pos, neg = (torch.from_numpy(x).to(dev) for x in (u, pos, neg))
            U, I = model.embeddings()
            eu, ep_, en = U[u], I[pos], I[neg]
            loss = torch.nn.functional.softplus((eu * en).sum(1) - (eu * ep_).sum(1)).mean()
            e0u, e0p = model.ego(u, pos)
            _, e0n = model.ego(u, neg)
            l2 = (e0u.pow(2).sum() + e0p.pow(2).sum() + e0n.pow(2).sum()) / (2 * len(b))
            opt.zero_grad()
            (loss + reg * l2).backward()
            opt.step()
            tot += loss.item() * len(b)
        row = dict(epoch=ep, loss=tot / len(pairs), time=time.time() - t0)
        if ep % eval_every == 0:
            model.set_graph(pairs, 0.0, rng)  # full graph for evaluation
            m, _, _ = evaluate(model, ds, "val")
            row["val_recall"] = m["recall"]
            if m["recall"] > best["val"]:
                best.update(val=m["recall"], epoch=ep, state={k: v.clone() for k, v in model.state_dict().items()})
                bad = 0
            else:
                bad += 1
            if verbose:
                print(f"ep {ep} loss {row['loss']:.4f} val_recall {m['recall']:.4f} ({row['time']:.0f}s)", flush=True)
        hist.append(row)
        if bad >= patience:
            break
    if best["state"] is not None:  # stays None only when the run ends before the first evaluation
        model.load_state_dict(best["state"])
    model.set_graph(pairs, 0.0, rng)
    train_time = time.time() - t0
    test, rec, rec_items = evaluate(model, ds, "test", mask_splits=("train", "val"))
    out = dict(cfg=cfg, seed=seed, val_recall=best["val"], best_epoch=best["epoch"], test=test,
               train_time=train_time, history=hist)
    if "test_unbiased" in ds:  # same model, scored on positives the platform recommender did not choose
        out["test_unbiased"] = evaluate(model, ds, "test_unbiased", mask_splits=("train", "val", "test"))[0]
    return out, rec, rec_items, model
