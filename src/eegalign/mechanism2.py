"""Round 3: calibrate where each kind of information breaks.

Round 2 found two different failures across people:
  * low-level information is MISSING from f0's representation (probe R^2 drops at layer 1 and at the output), so an
    edit after layer 1 cannot restore it; it has to be recovered from the electrodes (input side);
  * category information is PRESENT (probe accuracy ~equal to the within-subject decoder) but category retrieval
    collapses, i.e. it is MISALIGNED with the CLIP target space (output side).
Adapters here (names may end in _lr<lr>; default lr DEFAULT_LR):
  out<kind>_k<k>     output-side subspace edit z -> z + Wp U M U^T W (z - mu), M (k x k) from 0
                       kind: tgt  = output directions carrying the CLIP targets on SOURCE data (cross-cov, whitened)
                             pca  = top output principal components on source data
                             rand = random directions inside the output's support
  pmap<kind>_r<r>    input-side channel map restricted to r spatial patterns: W = I + P M P^T, M (r x r) from 0
                       kind: low  = spatial patterns of the low-level-property cross-covariance on SOURCE EEG
                             rand = random orthonormal patterns
  align+<out...>     channel map first (standard recipe), then the output edit on top (map frozen)
  sulora2+<out...>   SuLoRA-style map first, then the output edit on top
No anchor; lr tuned per family on subject 1 (phase mech2_tune). Everything defined from source data or the
calibration images only.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from . import props
from . import ubp as U
from .subspace import _basis

DEFAULT_LR = 1e-3


def split_lr(name):
    if '_lr' in name:
        fam, lr = name.rsplit('_lr', 1)
        return fam, float(lr)
    return name, DEFAULT_LR


def _src_sample(T, n=8000):
    pick = np.random.RandomState(0).choice(len(T.reh_x), min(len(T.reh_x), n), replace=False)
    return pick, torch.as_tensor(T.reh_x[pick]).float()


@torch.no_grad()
def _embed(f, x, bs=1024):
    dev = U.ENV['device']
    return torch.cat([f(x[i:i + bs].to(dev)).float().cpu() for i in range(0, len(x), bs)])


def out_basis(T, f, kind, k, seed=0, tag='f0'):
    """Output-space basis from SOURCE subjects' outputs of model f (f0, or the map-calibrated model's decoder)."""
    path = os.path.join(U.ENV['ckpt_dir'], f"mech2_out{kind}_k{k}_s{seed}_{tag}_t{T.target:02d}_{T.montage}_"
                                            f"{U.ENV['source']}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    pick, x = _src_sample(T)
    H = _embed(f, x).double()
    Z = T.tgt[T.reh_t[pick]].double().cpu()
    mu, W, Wp, Uw, s = _basis(H, Z, k)
    if kind in ('pca', 'rand'):
        Hc = H - H.mean(0)
        ev, V = torch.linalg.eigh(Hc.T @ Hc / len(Hc))
        if kind == 'pca':       # top-variance directions, expressed in whitened coordinates (orthonormalized)
            Uw, _ = torch.linalg.qr(W.double() @ V[:, -k:])
        else:
            B = V[:, ev > 1e-3 * ev.max()]
            g = torch.Generator().manual_seed(seed)
            Uw, _ = torch.linalg.qr(B @ torch.randn(B.shape[1], k, generator=g, dtype=torch.float64))
    out = dict(mu=mu.float(), W=W.float(), Wp=Wp.float(), U=Uw.float())
    torch.save(out, path)
    return out


def pattern_basis(T, kind, r, seed=0):
    """C x r spatial patterns. 'low': top left singular vectors of the channel x (time, property) cross-covariance
    between SOURCE EEG and the 6 low-level image properties (Haufe-style forward patterns)."""
    if kind == 'rand':
        g = torch.Generator().manual_seed(seed)
        P, _ = torch.linalg.qr(torch.randn(T.C, r, generator=g))
        return P
    path = os.path.join(U.ENV['ckpt_dir'], f"mech2_pat{kind}_r{r}_t{T.target:02d}_{T.montage}_{U.ENV['source']}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    pick, x = _src_sample(T)
    Z = props.props_for(T.P[T.reh_t[pick]]).astype(np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    X = x.double().numpy()
    X = X - X.mean(0)
    S = np.einsum('nct,nj->ctj', X, Z) / len(X)              # C x T x J
    Uc, sv, _ = np.linalg.svd(S.reshape(T.C, -1), full_matrices=False)
    P = torch.as_tensor(Uc[:, :r]).float()
    torch.save(P, path)
    return P


class OutEdit(nn.Module):
    def __init__(self, f0, b):
        super().__init__()
        self.f0 = f0
        dev = next(f0.parameters()).device
        self.register_buffer('mu', b['mu'].to(dev))
        self.register_buffer('B', (b['U'].T @ b['W']).contiguous().to(dev))
        self.register_buffer('A', (b['Wp'] @ b['U']).contiguous().to(dev))
        self.M = nn.Parameter(torch.zeros(b['U'].shape[1], b['U'].shape[1], device=dev))

    def forward(self, x):
        z = self.f0(x)
        return z + ((z - self.mu) @ self.B.T) @ self.M.T @ self.A.T


class PatMap(nn.Module):
    def __init__(self, f0, P):
        super().__init__()
        self.f0 = f0
        dev = next(f0.parameters()).device
        self.register_buffer('P', P.to(dev))
        self.M = nn.Parameter(torch.zeros(P.shape[1], P.shape[1], device=dev))

    def forward(self, x):
        W = torch.eye(self.P.shape[0], device=self.P.device) + self.P @ self.M @ self.P.T
        return self.f0(torch.einsum('cd,ndt->nct', W, x.float()).contiguous())


def is_mech2(name):
    return split_lr(name)[0].split('+')[-1].startswith(('out', 'pmap'))


def adapt2(name, T, X, Tg, seed, steps=200, batch=64):
    import copy
    from .adapters import _train, adapt
    fam, lr = split_lr(name)
    if '+' in fam:                                       # stage 1: the map, standard recipe; then freeze it
        first, fam = fam.split('+')
        m0 = adapt(first, T, X, Tg, seed, steps, batch).requires_grad_(False)
        dec, tag = m0.f0, first        # basis from the map's decoder on raw SOURCE EEG (the map is the target's)
    else:
        m0 = copy.deepcopy(T.f0).requires_grad_(False)
        dec, tag = m0, 'f0'
    if fam.startswith('out'):
        kind, k = fam[3:].split('_k')
        m = OutEdit(m0, out_basis(T, dec, kind, int(k), seed if kind == 'rand' else 0, tag))
    else:
        kind, r = fam[4:].split('_r')
        m = PatMap(m0, pattern_basis(T, kind, int(r), seed))
    return _train(m, T, X, Tg, [m.M], [m.M.detach().clone()], lr, 0.0, 0.0, steps, batch, seed)
