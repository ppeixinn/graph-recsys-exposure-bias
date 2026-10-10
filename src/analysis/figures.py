"""Figures for the breakdowns: recall by item-popularity bucket and by user-degree group, both tests side by side."""
import matplotlib.pyplot as plt
import numpy as np

XLABELS = {"item": "item-popularity bucket (training degree)", "user": "user-degree group (training interactions)"}


def plot_breakdowns(rows, labels=None, title="Recall@20 by group"):
    """Grouped bars from tidy rows with keys dataset, kind ('item' or 'user'), group (int), test ('biased' or
    'unbiased'), model, recall: one row of panels per dataset, one column per kind. Returns the figure."""
    datasets = sorted({r["dataset"] for r in rows})
    kinds = [k for k in ("item", "user") if any(r["kind"] == k for r in rows)]
    combos = sorted({(r["model"], r["test"]) for r in rows})
    width = 0.8 / len(combos)
    fig, axes = plt.subplots(len(datasets), len(kinds), figsize=(4.6 * len(kinds), 3.2 * len(datasets)), squeeze=False)
    for ri, ds in enumerate(datasets):
        for ci, kind in enumerate(kinds):
            ax = axes[ri][ci]
            xs = sorted({r["group"] for r in rows if r["dataset"] == ds and r["kind"] == kind})
            for j, (m, t) in enumerate(combos):
                v = {r["group"]: r["recall"] for r in rows
                     if (r["dataset"], r["kind"], r["model"], r["test"]) == (ds, kind, m, t)}
                heights = [v[x] if x in v and v[x] is not None else np.nan for x in xs]
                ax.bar(np.arange(len(xs)) + (j - len(combos) / 2 + 0.5) * width, heights, width,
                       label=f"{m} ({t} test)")
            xl = (labels or {}).get(kind, [str(x) for x in xs])
            ax.set(xticks=np.arange(len(xs)), xticklabels=xl[:len(xs)], xlabel=XLABELS[kind], ylabel="Recall@20",
                   title=ds)
            ax.grid(alpha=0.3, axis="y")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
