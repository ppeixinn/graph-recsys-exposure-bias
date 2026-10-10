"""Groups for the breakdowns: item-popularity buckets and user-degree groups, both from training degrees only."""
import numpy as np

from data import degrees


def _by_degree(deg, n):
    """Equal-count groups over a degree array: 0 covers the lowest degrees, n-1 the highest."""
    order = np.argsort(deg, kind="stable")
    g = np.empty(len(deg), dtype=int)
    g[order] = np.arange(len(deg)) * n // len(deg)
    return g


def item_groups(ds, n=5):
    """Popularity bucket per item: items ranked by training degree, split into n equal-count buckets."""
    return _by_degree(degrees(ds)[1], n)


def user_groups(ds, n=3):
    """Degree group per user, by default terciles of training interactions (cold / medium / heavy)."""
    return _by_degree(degrees(ds)[0], n)


def group_info(deg, groups):
    """One row per group: size and the training-degree range it covers, for the report's group table."""
    return [dict(group=g, size=int((groups == g).sum()), deg_min=int(d.min()), deg_median=float(np.median(d)),
                 deg_max=int(d.max()))
            for g in range(groups.max() + 1)
            if len(d := deg[groups == g])]
