"""Dataset loading: builds implicit-feedback splits as {user: np.array(items)}.

Both datasets have `train`, `val` and `test` (random per-user split of the biased logs), `test_unbiased`:
positives from an exposure that the platform recommender did not choose, and `unbiased_exposed`: every video each
user was shown in that exposure, positive or not (candidates of the exposure-only protocol)."""
import os
import zipfile
from functools import lru_cache

import numpy as np

from utils.paths import DATA_DIR
from .splits import _build, _shown, _unbiased

KUAI_DIR = os.path.join(DATA_DIR, "KuaiRand-Pure", "data")
KUAI_STANDARD = ["log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"]
KUAI_RANDOM = "log_random_4_22_to_5_08_pure.csv"
KUAIREC_ZIP = os.path.join(DATA_DIR, "KuaiRec.zip")
KUAIREC_FILES = "KuaiRec 2.0/data/"


@lru_cache(maxsize=None)
def _kuairand_positives(positive):
    import pandas as pd
    cols = ["user_id", "video_id", positive]
    std = pd.concat([pd.read_csv(os.path.join(KUAI_DIR, f), usecols=cols) for f in KUAI_STANDARD])
    rnd = pd.read_csv(os.path.join(KUAI_DIR, KUAI_RANDOM), usecols=cols)
    return tuple(df[df[positive] == 1].drop_duplicates(["user_id", "video_id"])[cols[:2]] for df in (std, rnd))


@lru_cache(maxsize=None)
def _kuairand_shown():
    import pandas as pd
    rnd = pd.read_csv(os.path.join(KUAI_DIR, KUAI_RANDOM), usecols=["user_id", "video_id"])
    return rnd.drop_duplicates()


def load_kuairand(seed=0, positive="is_click", test_frac=0.2):
    """KuaiRand-Pure. A positive is a row with `positive` == 1 (is_click, is_like or long_view).

    train/val/test: per-user random split of the two standard (recommender-exposed) logs.
    test_unbiased:  positives in the randomly exposed log, for the same users and items, without pairs that are
                    also positive in the standard logs.
    unbiased_exposed: every video each user was shown in the random log (median 24 per user)."""
    std, rnd = _kuairand_positives(positive)
    ds, uid, iid = _build(f"KuaiRand-Pure ({positive})", std.user_id.values, std.video_id.values, seed, test_frac)
    ds["test_unbiased"] = _unbiased(rnd, uid, iid, set(zip(std.user_id, std.video_id)))
    ds["unbiased_exposed"] = _shown(_kuairand_shown(), uid, iid)
    return ds


@lru_cache(maxsize=None)
def _kuairec_positives(min_ratio):
    import pandas as pd
    cols = ["user_id", "video_id", "watch_ratio"]
    with zipfile.ZipFile(KUAIREC_ZIP) as z:
        big, small = (pd.read_csv(z.open(KUAIREC_FILES + f), usecols=cols) for f in ("big_matrix.csv", "small_matrix.csv"))
    block = np.sort(small.video_id.unique())
    return tuple(df[df.watch_ratio >= min_ratio].drop_duplicates(["user_id", "video_id"])[cols[:2]] for df in (big, small)) + (block,)


@lru_cache(maxsize=None)
def _kuairec_shown():
    import pandas as pd
    with zipfile.ZipFile(KUAIREC_ZIP) as z:
        small = pd.read_csv(z.open(KUAIREC_FILES + "small_matrix.csv"), usecols=["user_id", "video_id"])
    return small.drop_duplicates()


def load_kuairec(seed=0, min_ratio=2.0, test_frac=0.2):
    """KuaiRec. A positive is a (user, video) pair with at least one view of watch_ratio >= min_ratio.

    train/val/test: per-user random split of the big matrix (recommender-exposed logs).
    test_unbiased:  positives in the small matrix, where almost every user saw every video of a fixed block.
    unbiased_items: the videos of that block; the unbiased test ranks only among them.
    unbiased_exposed: the block videos each user was shown in the small matrix (almost all of them)."""
    big, small, block = _kuairec_positives(min_ratio)
    ds, uid, iid = _build(f"KuaiRec (watch_ratio >= {min_ratio:g})", big.user_id.values, big.video_id.values, seed, test_frac)
    ds["test_unbiased"] = _unbiased(small, uid, iid, set())
    ds["unbiased_items"] = np.array([iid[i] for i in block if i in iid])
    ds["unbiased_exposed"] = _shown(_kuairec_shown(), uid, iid)
    return ds


def load(name, **kw):
    return {"kuairand": load_kuairand, "kuairec": load_kuairec}[name](**kw)
