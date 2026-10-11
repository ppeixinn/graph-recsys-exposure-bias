"""Popularity-aware negative sampling: negatives drawn with probability proportional to training degree^alpha.

BPR's uniform negatives let popular items keep high scores (they are positives often and negatives only at
chance), which notebook 07 shows is where all the recall lives: sampling negatives proportional to popularity
pushes popular embeddings down instead and frees the ranking for tail items. alpha=0 is uniform BPR.
"""
import numpy as np

from data import degrees


def popularity_weights(ds, alpha):
    """Normalised cumulative weights over items for sampling negatives with P(i) ~ (train degree)^alpha.
    Returns None when alpha <= 0, i.e. plain uniform BPR, so the training loop keeps its original path."""
    if alpha <= 0:
        return None
    _, item_deg = degrees(ds)
    w = item_deg.astype(np.float64) ** float(alpha)
    return np.cumsum(w) / w.sum()


def sample(cum_weights, rng, n):
    """Draw n item ids with probability proportional to the weights behind `cum_weights` (a normalised CDF)."""
    return np.searchsorted(cum_weights, rng.random(n))
