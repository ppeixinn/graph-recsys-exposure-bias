"""The most-popular recommender: ranks items by training degree, the same for every user."""
import torch

from data.stats import degrees


class MostPopular:
    """Non-learned baseline. Anchors both tests: it repeats the exposure choices the platform already made,
    so it is strong on the biased test and weak on KuaiRand's randomly exposed test."""

    def __init__(self, ds):
        self.item_deg = torch.tensor(degrees(ds)[1], dtype=torch.float32)[:, None]
        self.n_users = ds["n_users"]

    def eval(self):
        pass

    def embeddings(self):
        return torch.ones(self.n_users, 1), self.item_deg
