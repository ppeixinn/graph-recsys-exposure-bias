"""Saving results and figures to results/ with the team conventions."""
import json
import os

from .paths import RESULTS_DIR


def save_json(obj, name):
    """Write obj to results/<name>; store the config inside obj so the run can be reproduced."""
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def savefig(fig, name, **kw):
    """Save to results/<name> at 200 dpi without matplotlib's web address in the file metadata."""
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=200, metadata={"Software": None}, **kw)
    return path
