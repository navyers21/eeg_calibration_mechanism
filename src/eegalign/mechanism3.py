"""Calibration aimed at the low-level finding (notebooks/lowlevel.ipynb).

Finding: f0 relies on the same electrodes (Oz and its occipital neighbours) and the same time (80-260 ms) as the
person's own decoder, and its layer 1 captures nearly the same low-level information, but in person-specific
directions: a readout learned from other people recovers less than half of it. So the fix should let each person
learn their OWN low-level directions, exactly where the information is.

Adapters (stages joined by '+', applied in order, earlier stages frozen; a trailing _lr<lr> sets the lr of the
low-level stage; maps use their own recipe, output edits lr OUT_LR):
  llwin_r<r>[_aux]   low-rank update of layer 1 restricted to the occipital electrodes x 80-260 ms:
                     layer1(x) + B A x[occ, 80-260 ms]  (A: r x (n_occ * 45), B: 1024 x r, B = 0 at start)
  llwinctl_r<r>      control: same size, same electrodes, at 600-780 ms (no low-level information there)
  _aux               + auxiliary loss: a 1024 -> 6 head on layer 1 predicts the image's 6 low-level properties
                     (computed from pixels, so free labels), trained jointly; weight AUX_W
  occmap / parmap    channel map restricted to the 8 occipital / 8 parietal electrodes (posterior-17 only)
  sulora2 / align    the existing maps, as a first stage
  outpca_k64         the confirmed output edit, as a last stage
Examples: llwin_r8_aux_lr1e-3, sulora2+llwin_r8_aux+outpca_k64_lr1e-3.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ubp as U

OCC = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']
PAR = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6']
WIN = (20, 65)            # samples at 250 Hz: 80-260 ms
WIN_CTL = (150, 195)      # 600-780 ms, same length
T_LEN = 250
OUT_LR = 1e-4
AUX_W = 1.0
DEFAULT_LR = 1e-3


def is_mech3(name):
    return any(p.startswith(('llwin', 'occmap', 'parmap')) for p in name.split('+'))


def split(name):
    lr = DEFAULT_LR
    if '_lr' in name.split('+')[-1]:
        name, l = name.rsplit('_lr', 1)
        lr = float(l)
    return name.split('+'), lr


class WinAdapter(nn.Module):
    """Wraps layer 1 (nn.Linear on the flattened channels x time input): adds B A x_sub on a channel x time block."""

    def __init__(self, lin, montage, rank, window):
        super().__init__()
        self.lin = lin
        chans = U.MONTAGES[montage]
        ci = [chans.index(c) for c in OCC if c in chans]
        idx = [c * T_LEN + t for c in ci for t in range(*window)]
        dev = lin.weight.device
        self.register_buffer('idx', torch.as_tensor(idx, device=dev))
        self.A = nn.Parameter(torch.randn(rank, len(idx), device=dev) / len(idx) ** 0.5)
        self.B = nn.Parameter(torch.zeros(lin.out_features, rank, device=dev))
        self.last = None

    def forward(self, x):
        h = self.lin(x) + (x[:, self.idx] @ self.A.T) @ self.B.T
        self.last = h
        return h


class SubMap(nn.Module):
    """Channel map that may only change the block of the given electrodes (identity elsewhere)."""

    def __init__(self, f0, montage, names):
        super().__init__()
        self.f0 = f0
        chans = U.MONTAGES[montage]
        ci = [chans.index(c) for c in names if c in chans]
        assert len(ci) == len(names), f'{montage} lacks some of {names}'
        dev = next(f0.parameters()).device
        self.register_buffer('ci', torch.as_tensor(ci, device=dev))
        self.register_buffer('I', torch.eye(len(chans), device=dev))
        self.M = nn.Parameter(torch.zeros(len(ci), len(ci), device=dev))

    def forward(self, x):
        W = self.I.clone()
        W[self.ci[:, None], self.ci[None, :]] = W[self.ci[:, None], self.ci[None, :]] + self.M
        return self.f0(torch.einsum('cd,ndt->nct', W, x.float()).contiguous())


def _decoder(m):
    while hasattr(m, 'f0'):
        m = m.f0
    return m


def _props_z(T):
    """Standardized low-level properties of the calibration images (T.cal_idx is set by calibrate.py)."""
    from . import props
    if not hasattr(T, '_pz'):
        P = props.props_for(T.P).astype(np.float32)
        T._pz = torch.as_tensor((P - P.mean(0)) / (P.std(0) + 1e-6), device=U.ENV['device'])
    return T._pz


def _train(m, T, X, Tg, params, lr, steps, batch, seed, aux=None, n_neg=1024, wd=1e-4):
    """CLIP loss on the calibration pairs (as adapters._train, no anchor), + optional auxiliary low-level loss."""
    dev = U.ENV['device']
    torch.manual_seed(seed)
    X, Tg = torch.as_tensor(X).to(dev).float(), torch.as_tensor(Tg).to(dev).float()
    f = _decoder(m)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    n, bs = len(X), min(batch, len(X))
    m.train()
    for _ in range(steps):
        i = torch.randperm(n, device=dev)[:bs]
        z = m(X[i])
        gal = torch.cat([Tg[i], T.tgt[torch.randint(T.n, (n_neg,), device=dev)]])
        loss = F.cross_entropy(f.softplus(f.logit_scale) * z @ gal.T, torch.arange(len(z), device=dev))
        if aux is not None:
            head, win, yz = aux
            loss = loss + AUX_W * F.mse_loss(head(win.last), yz[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


def adapt3(name, T, X, Tg, seed, steps=200, batch=64):
    from .adapters import adapt
    from .mechanism2 import OutEdit, out_basis
    stages, lr = split(name)
    m, map_tag = None, 'f0'
    for st in stages:
        if st in ('sulora2', 'align'):
            assert m is None, 'a map must be the first stage'
            m, map_tag = adapt(st, T, X, Tg, seed, steps, batch).requires_grad_(False), st
        elif st in ('occmap', 'parmap'):
            assert m is None
            m = SubMap(copy.deepcopy(T.f0).requires_grad_(False), T.montage, OCC if st == 'occmap' else PAR)
            m = _train(m, T, X, Tg, [m.M], lr, steps, batch, seed).requires_grad_(False)
        elif st.startswith('llwin'):
            m = m if m is not None else copy.deepcopy(T.f0).requires_grad_(False)
            dec = _decoder(m)
            r = int(st.split('_r')[1].split('_')[0])
            win = WinAdapter(dec.model[0], T.montage, r, WIN_CTL if st.startswith('llwinctl') else WIN)
            dec.model[0] = win
            params, aux = [win.A, win.B], None
            if st.endswith('_aux'):
                head = nn.Linear(win.lin.out_features, 6).to(U.ENV['device'])
                aux = (head, win, _props_z(T)[torch.as_tensor(T.cal_idx, device=U.ENV['device'])])
                params = params + list(head.parameters())
            m = _train(m, T, X, Tg, params, lr, steps, batch, seed, aux).requires_grad_(False)
            win.last = None
        elif st.startswith('outpca') or st.startswith('outrand'):
            kind, k = st[3:].split('_k')
            # basis from the (map-free) decoder on SOURCE EEG, cached per map type (same files as phases mech2/mech3)
            b = out_basis(T, _decoder(m if m is not None else T.f0), kind, int(k), seed if kind == 'rand' else 0,
                          map_tag)
            m = OutEdit(m if m is not None else copy.deepcopy(T.f0).requires_grad_(False), b)
            m = _train(m, T, X, Tg, [m.M], OUT_LR, steps, batch, seed).requires_grad_(False)
        else:
            raise ValueError(f'unknown stage {st} in {name}')
    return m.eval()
