"""Reusable algorithm pieces shared by models and training: losses, negative samplers, graph operations,
and the improvement fix (e.g. popularity-aware negative sampling). One module per algorithm."""
from .negative_sampling import popularity_weights, sample
