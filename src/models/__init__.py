"""Recommenders. Each model exposes embeddings() -> (user, item) matrices so inference.evaluate can score it.

core.py holds BPR-MF and LightGCN; add new models (e.g. most-popular) as their own module here and export them."""
from .core import MF, LightGCN, build_norm_adj, SpMM
