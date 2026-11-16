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

v2 (fair test after v1 overfit; its controls hurt as much as the method): lower learning rates, an L2 pull of the
update toward zero ('anc'), rank 4, A started from the low-level patterns of the OTHER people's EEG in that window
('P', so with few images the person starts from the cross-person solution, as the learning curve suggests), and
'align': instead of a freely trained head, a FIXED readout fitted on the other people's layer-1 activations, so the
person's low-level information is moved into the directions the rest of the decoder expects (finding D).
  llwinP_r4_anc[_align], llwinctlP_r4_anc_align (control), occmap_align / parmap_align, occmap+outpca_k64
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


ANC_W = 0.1         # v2: L2 pull of the window update (B) toward zero ('anc')


def _train(m, T, X, Tg, params, lr, steps, batch, seed, aux=None, anchor=None, n_neg=1024, wd=1e-4):
    """CLIP loss on the calibration pairs (as adapters._train), + optional low-level loss and L2 anchor.
    aux = (head, get_h, yz): MSE between head(layer-1 output) and the images' standardized low-level properties.
    anchor = list of tensors pulled toward zero with weight ANC_W."""
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
            head, get_h, yz = aux
            loss = loss + AUX_W * F.mse_loss(head(get_h()), yz[i])
        if anchor:
            loss = loss + ANC_W * sum(a.pow(2).sum() for a in anchor)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


class Capture:
    """Keeps the latest output of a module (forward hook)."""

    def __init__(self, mod):
        self.h = None
        self.handle = mod.register_forward_hook(lambda m, i, o: setattr(self, 'h', o))

    def __call__(self):
        return self.h

    def remove(self):
        self.handle.remove()
        self.h = None


def _src_sample(T, n=8000):
    pick = np.random.RandomState(0).choice(len(T.reh_x), min(len(T.reh_x), n), replace=False)
    return pick[np.unique(T.reh_t[pick], return_index=True)[1]]


def window_prior(T, window, r):
    """v2 'P': the r channel x time patterns of the occipital window that carry low-level properties in the OTHER
    people's raw EEG (top singular vectors of the EEG-properties cross-covariance), scaled to unit projection SD."""
    import os
    path = os.path.join(U.ENV['ckpt_dir'], f"mech3_winprior_{window[0]}_{window[1]}_r{r}_t{T.target:02d}_{T.montage}_"
                                            f"{U.ENV['source']}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    from . import props
    chans = U.MONTAGES[T.montage]
    ci = [chans.index(c) for c in OCC if c in chans]
    pick = _src_sample(T)
    Xs = T.reh_x[pick][:, ci, window[0]:window[1]].astype(np.float64).reshape(len(pick), -1)
    Xs = Xs - Xs.mean(0)
    Z = props.props_for(T.P[T.reh_t[pick]]).astype(np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    Uc, sv, _ = np.linalg.svd(Xs.T @ Z / len(Xs), full_matrices=False)
    A0 = Uc[:, :r].T
    A0 = A0 / ((Xs @ A0.T).std(0)[:, None] + 1e-9)
    A0 = torch.as_tensor(A0, dtype=torch.float32)
    torch.save(A0, path)
    return A0


def source_readout(T, dec, tag):
    """v2 'align': a FIXED linear readout layer-1 -> 6 standardized low-level properties, fitted on the OTHER people's
    layer-1 activations of decoder `dec`. Training the person's update to satisfy it moves their low-level information
    into the directions the rest of the decoder was trained on."""
    import os
    path = os.path.join(U.ENV['ckpt_dir'], f"mech3_srcreadout_{tag}_t{T.target:02d}_{T.montage}_{U.ENV['source']}.pt")
    dev = U.ENV['device']
    if os.path.exists(path):
        W, b = torch.load(path, weights_only=False)
    else:
        from sklearn.linear_model import RidgeCV
        from . import props
        pick = _src_sample(T)
        lin = dec.model[0].lin if isinstance(dec.model[0], WinAdapter) else dec.model[0]
        with torch.no_grad():
            H = torch.cat([lin(torch.as_tensor(T.reh_x[pick[j:j + 1024]]).float().to(dev).flatten(1)).cpu()
                           for j in range(0, len(pick), 1024)]).numpy()
        P = props.props_for(T.P).astype(np.float64)
        Z = (props.props_for(T.P[T.reh_t[pick]]) - P.mean(0)) / (P.std(0) + 1e-6)
        r = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(H, Z)
        W, b = torch.as_tensor(r.coef_, dtype=torch.float32), torch.as_tensor(r.intercept_, dtype=torch.float32)
        torch.save((W, b), path)
    head = nn.Linear(W.shape[1], W.shape[0]).to(dev)
    with torch.no_grad():
        head.weight.copy_(W)
        head.bias.copy_(b)
    return head.requires_grad_(False)


def adapt3(name, T, X, Tg, seed, steps=200, batch=64):
    """Stages: sulora2|align (first), occmap|parmap[_align] (first), llwin*/llwinctl*[P]_r<r>[_anc][_aux|_align],
    outpca|outrand_k<k> (last). v1 names behave exactly as before."""
    from .adapters import adapt
    from .mechanism2 import OutEdit, out_basis
    dev = U.ENV['device']
    stages, lr = split(name)
    m, map_tag = None, 'f0'
    yz = lambda: _props_z(T)[torch.as_tensor(T.cal_idx, device=dev)]
    for st in stages:
        tok = st.split('_')
        flags = set(tok[1:])
        if st in ('sulora2', 'align'):
            assert m is None, 'a map must be the first stage'
            m, map_tag = adapt(st, T, X, Tg, seed, steps, batch).requires_grad_(False), st
        elif tok[0] in ('occmap', 'parmap'):
            assert m is None
            m = SubMap(copy.deepcopy(T.f0).requires_grad_(False), T.montage, OCC if tok[0] == 'occmap' else PAR)
            aux, cap = None, None
            if 'align' in flags:
                cap = Capture(m.f0.model[0])
                aux = (source_readout(T, m.f0, 'f0'), cap, yz())
            m = _train(m, T, X, Tg, [m.M], lr, steps, batch, seed, aux).requires_grad_(False)
            if cap:
                cap.remove()
        elif tok[0].startswith('llwin'):
            m = m if m is not None else copy.deepcopy(T.f0).requires_grad_(False)
            dec = _decoder(m)
            r = int([t for t in tok[1:] if t.startswith('r')][0][1:])
            window = WIN_CTL if tok[0].startswith('llwinctl') else WIN
            head = source_readout(T, dec, map_tag) if 'align' in flags else None   # before wrapping layer 1
            win = WinAdapter(dec.model[0], T.montage, r, window)
            if tok[0].endswith('P'):
                with torch.no_grad():
                    win.A.copy_(window_prior(T, window, r).to(dev))
            dec.model[0] = win
            params, aux = [win.A, win.B], None
            get_h = lambda: win.last
            if 'aux' in flags:
                h2 = nn.Linear(win.lin.out_features, 6).to(dev)
                aux, params = (h2, get_h, yz()), params + list(h2.parameters())
            elif 'align' in flags:
                aux = (head, get_h, yz())
            m = _train(m, T, X, Tg, params, lr, steps, batch, seed, aux,
                       anchor=[win.B] if 'anc' in flags else None).requires_grad_(False)
            win.last = None
        elif st.startswith('outpca') or st.startswith('outrand'):
            kind, k = st[3:].split('_k')
            # basis from the decoder on SOURCE EEG, cached per map type (same files as phases mech2/mech3)
            b = out_basis(T, _decoder(m if m is not None else T.f0), kind, int(k), seed if kind == 'rand' else 0,
                          map_tag)
            m = OutEdit(m if m is not None else copy.deepcopy(T.f0).requires_grad_(False), b)
            m = _train(m, T, X, Tg, [m.M], OUT_LR, steps, batch, seed).requires_grad_(False)
        else:
            raise ValueError(f'unknown stage {st} in {name}')
    return m.eval()
