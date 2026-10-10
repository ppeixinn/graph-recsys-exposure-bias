"""BPR-MF and LightGCN (He et al., SIGIR 2020) on a user-item bipartite graph."""
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def build_norm_adj(n_users, n_items, pairs, norm="sym", drop=0.0, rng=None, device=None):
    """Bipartite adjacency A=[[0,R],[R^T,0]], normalised as D^-1/2 A D^-1/2 (norm='sym'),
    D^-1 A ('left') or left unnormalised ('none'). `drop` randomly removes that fraction of edges.
    Returns the matrix and its transpose as CSR tensors (on `device` when given)."""
    u, i = pairs[:, 0], pairs[:, 1]
    if drop > 0:
        keep = rng.rand(len(u)) >= drop
        u, i = u[keep], i[keep]
    n = n_users + n_items
    rows = np.concatenate([u, i + n_users])
    cols = np.concatenate([i + n_users, u])
    A = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    if norm == "sym":
        d = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
        A = sp.diags(d) @ A @ sp.diags(d)
    elif norm == "left":
        d = np.power(deg, -1.0, where=deg > 0, out=np.zeros_like(deg))
        A = sp.diags(d) @ A
    return _to_csr(A, device), _to_csr(A.T, device)


def _to_csr(A, device=None):
    A = A.tocsr()
    t = (lambda a: torch.from_numpy(a).to(device)) if device is not None else torch.from_numpy
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
        return torch.sparse_csr_tensor(t(A.indptr.astype(np.int64)), t(A.indices.astype(np.int64)),
                                       t(A.data.astype(np.float32)), A.shape, check_invariants=False)


class SpMM(torch.autograd.Function):
    """y = A x. The gradient A^T g uses a CSR copy of A^T, which is much faster on CPU than the default backward."""

    @staticmethod
    def forward(ctx, x, adj, adj_t):
        ctx.adj_t = adj_t
        return torch.sparse.mm(adj, x)

    @staticmethod
    def backward(ctx, grad):
        return torch.sparse.mm(ctx.adj_t, grad), None, None


class MF(nn.Module):
    """BPR matrix factorisation: score = <p_u, q_i>. No graph."""

    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def set_graph(self, *a, **k):
        pass

    def embeddings(self):
        return self.user.weight, self.item.weight

    def ego(self, u, i):
        return self.user(u), self.item(i)


class LightGCN(MF):
    """E^(k+1) = A_hat E^(k) (no weights, no nonlinearity); final embedding = combination of layers 0..K."""

    def __init__(self, n_users, n_items, dim=64, n_layers=3, combine="mean", norm="sym"):
        super().__init__(n_users, n_items, dim)
        self.n_layers, self.combine, self.norm = n_layers, combine, norm
        self.adj = self.adj_t = None

    def set_graph(self, pairs, drop=0.0, rng=None):
        # the adjacency lives on the same device as the embeddings
        self.adj, self.adj_t = build_norm_adj(self.n_users, self.n_items, pairs, self.norm, drop, rng,
                                              device=self.user.weight.device)

    def embeddings(self):
        e = torch.cat([self.user.weight, self.item.weight])
        layers = [e]
        for _ in range(self.n_layers):
            e = SpMM.apply(e, self.adj, self.adj_t)
            layers.append(e)
        if self.combine == "mean":
            out = torch.stack(layers).mean(0)
        elif self.combine == "last":
            out = layers[-1]
        else:
            raise ValueError(self.combine)
        return out[: self.n_users], out[self.n_users:]
