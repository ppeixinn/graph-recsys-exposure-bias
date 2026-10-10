"""Datasets as {user: np.array(items)} splits: loaders, split rules, density transforms and statistics."""
from utils.paths import DATA_DIR
from .loaders import (KUAI_DIR, KUAI_STANDARD, KUAI_RANDOM, KUAIREC_ZIP, KUAIREC_FILES,
                      load, load_kuairand, load_kuairec)
from .splits import to_pairs
from .transforms import sparsify, k_core
from .stats import degrees, stats
