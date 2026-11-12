"""Mechanism-guided calibration: adapt only one subspace of the decoder's first-layer output.

The subspace is defined WITHOUT the new user's labels: it is the linear subspace (in whitened coordinates, as in
LEACE) that carries a property in f0's first-layer activations on the SOURCE subjects' recordings.
  sub_low   top-k directions carrying the 6 low-level image properties
  sub_sem   top-k directions carrying the 20 semantic clusters
  sub_rand  k random whitened directions (control)
For a new user only a k x k matrix M is learned:  h -> h + Wp U M U^T W (h - mu),  M = 0 at start (= f0).
k = 6 gives 36 numbers (the channel map has 64 at 8 channels).
"""
import os

import numpy as np
import torch
import torch.nn as nn

from . import props
from . import ubp as U

K = 6


def _basis(H, Z, k, shrinkage=1e-2):
    H = torch.as_tensor(H).double()
    Z = torch.as_tensor(Z).double()
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    mu = H.mean(0)
    Hc = H - mu
    S = Hc.T @ Hc / len(H)
    S = S + shrinkage * S.trace() / S.shape[0] * torch.eye(S.shape[0], dtype=S.dtype)
    ev, V = torch.linalg.eigh(S)
    ev = ev.clamp_min(1e-12)
    W, Wp = V @ torch.diag(ev.rsqrt()) @ V.T, V @ torch.diag(ev.sqrt()) @ V.T
    Uw, s, _ = torch.linalg.svd(W @ (Hc.T @ Z / len(H)), full_matrices=False)
    return mu, W, Wp, Uw[:, :k], s[:k]


def basis(T, kind, k=K, seed=0, v2=False):
    """(mu, W, Wp, U) for `kind`, cached per (target, montage) on Drive.
    v2 (subx adapters): the random basis is drawn inside the activations' support (as in mi/leace.py) and depends on
    the calibration seed; k may exceed the property rank only for sub_rand (sub_low has rank <= 6, sub_sem <= 20)."""
    tag = f"v2_s{seed}" if (v2 and kind == 'sub_rand') else ('v2' if v2 else '')
    path = os.path.join(U.ENV['ckpt_dir'], f"subspace_{kind}{tag}_k{k}_t{T.target:02d}_{T.montage}_{U.ENV['source']}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    dev = U.ENV['device']
    pick = np.random.RandomState(0).choice(len(T.reh_x), min(len(T.reh_x), 8000), replace=False)
    x = torch.as_tensor(T.reh_x[pick]).float()
    with torch.no_grad():
        H = torch.cat([T.f0.model[0](x[i:i + 1024].to(dev).flatten(1)).cpu() for i in range(0, len(x), 1024)])
    paths = T.P[T.reh_t[pick]]
    if kind in ('sub_low', 'sub_rand'):     # sub_rand only uses mu and the whitening; skip the k-means
        Z = props.props_for(paths)
    else:
        km = props.semantic_kmeans(T.P, k=20)
        Z = np.eye(20)[props.semantic_labels(km, paths)]
    rank = {'sub_low': Z.shape[1], 'sub_sem': Z.shape[1] - 1}.get(kind, H.shape[1])
    if k > rank:
        raise ValueError(f'{kind}: k={k} exceeds the property rank {rank}; use sub_rand for larger k')
    mu, W, Wp, Uw, s = _basis(H, Z, k)
    if kind == 'sub_rand':
        g = torch.Generator().manual_seed(seed)
        if v2:      # inside the support of the (unshrunk) activation covariance, like mi/leace.py random_control
            Hc = torch.as_tensor(H).double() - mu.double()
            ev, V = torch.linalg.eigh(Hc.T @ Hc / len(Hc))
            B = V[:, ev > 1e-3 * ev.max()]
            k = min(k, B.shape[1])
            Uw, _ = torch.linalg.qr(B @ torch.randn(B.shape[1], k, generator=g, dtype=torch.float64))
        else:
            Uw, _ = torch.linalg.qr(torch.randn(Uw.shape[0], k, generator=g, dtype=torch.float64))
    out = dict(mu=mu.float(), W=W.float(), Wp=Wp.float(), U=Uw.float(), s=s.float())
    torch.save(out, path)
    return out


class SubspaceEdit(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.register_buffer('mu', b['mu'])
        self.register_buffer('B', (b['U'].T @ b['W']).contiguous())        # k x D: into the subspace
        self.register_buffer('A', (b['Wp'] @ b['U']).contiguous())         # D x k: back out
        self.M = nn.Parameter(torch.zeros(b['U'].shape[1], b['U'].shape[1]))

    def forward(self, h):
        return h + ((h - self.mu) @ self.B.T) @ self.M.T @ self.A.T


def insert(f0, b):
    """Put a SubspaceEdit right after f0's first linear layer; returns (model, [M])."""
    edit = SubspaceEdit(b).to(next(f0.parameters()).device)
    f0.model[0] = nn.Sequential(f0.model[0], edit)
    return f0, [edit.M]
