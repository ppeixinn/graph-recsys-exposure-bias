"""Recommenders. Each model exposes embeddings() -> (user, item) matrices so inference.evaluate can score it.

core.py holds BPR-MF and LightGCN, most_popular.py the most-popular baseline; new models go in their own
module here and are exported."""
from .core import MF, LightGCN, build_norm_adj, SpMM
from .most_popular import MostPopular
