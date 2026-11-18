"""Step 1 of the synthetic-subject proposal: does a simple transform of one person's EEG close the low-level probe gap?

    python scripts/transfer.py tune   --config C [--montages ...]                 # lambda per family, contrastive lr
    python scripts/transfer.py pairs  --config C [--targets ...] [--montages ...] # 1a ceiling + 1b held-out, B x A
    python scripts/transfer.py pool   --config C [--targets ...] [--montages ...] # B -> pool: budgets, 1c split-half
    python scripts/transfer.py retr   --config C [--targets ...] [--montages ...] # 1d: transform in front of f0, top-1
    python scripts/transfer.py report --config C

Setup (all fixed seeds, same for every person):
  eval concepts   the 200 concepts f0 never trains on (ubp.val_concepts), all 10 images = 2,000 trials
  fit concepts    300 of the remaining concepts (3,000 images): transforms are fitted here
  probe concepts  the rest: probes are trained on 8,000 of these images
  probes          ridge (train standardization, PCA 256) from raw EEG (C x 250, flattened) to the 6 low-level
                  properties, same recipe as lowlevel.py. Person probes (one per person), and a pool probe per target
                  (each of the 8,000 images from a random source person other than the target, i.e. the people f0 is
                  trained on). All frozen after fitting.
  transforms      x -> M lag(x), fitted by minimizing the frozen probe's error on the person's fit images
                  (properties standardized) + lambda ||params - identity||^2. M is C x C:
                    scale     exp(theta) I                             (one global gain: only rescales predictions)
                    gain      diag(exp theta)                          (one gain per channel, kept positive)
                    mix       I + alpha L                              (L: row-normalized 3-nearest-neighbour graph)
                    mix_pc    I + diag(alpha) L                        (one mixing coefficient per channel)
                    gainmix   diag(exp theta) (I + alpha L)
                    fullmap   any C x C matrix (upper bound for spatial transforms)
                  lag: shift by tau samples (4 ms) inside 40-200 ms with a 5-sample cosine taper, tau in -6..6 by grid.
                  gainmix_lag: lag first (grid), then gainmix at that lag.
  controls        mix_shuf / mix_pc_shuf / gainmix_shuf: electrode graph randomly permuted (3 permutations)
                  lag_late: same lag range at 500-660 ms;  gainmix_lag_ctl: shuffled graph + late window
                  gain's control is scale: a probe's R^2 can rise just by shrinking its predictions, which says
                  nothing about electrodes. 1c control: the same split-half fits on shuffled property labels (part of
                  each parameter is label-independent and stable by itself), plus how much people differ.
  gap closure     (aligned - cross) / (own - cross); own = the person's own probe on their eval concepts,
                  cross = the other probe on their raw EEG. Suppressed when own - cross < 0.02.
  1d              B -> pool transform fitted on 25 or 200 of B's fit images (3 draws), frozen, put in front of f0;
                  200-way top-1 on the official test set. Arms: probe-fitted vs the same family fitted with f0's CLIP
                  loss on the same images (contrastive) vs f0.
Lambda (per spatial family) and the contrastive lr are tuned on subject 1 (B = 1, A = 2..10 / held top-1) only.
Time axis: samples from epoch start x 4 ms, as in lowlevel.py. Sessions are not in the cached arrays, so 1c uses
concept split-halves; a session split needs the raw files (to do).
Outputs: <dir of results_csv>/transfer/ (one CSV per target x montage per phase, resumable; styles_*.npz hold the
per-person fitted parameters for the generator).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]

PROPS = ['luminance', 'contrast', 'saturation', 'colorfulness', 'high_sf_ratio', 'edge_density']
N_PROBE, N_FIT_CONCEPTS, SPLIT_SEED = 8000, 300, 2028
LAGS = list(range(-6, 7))
WINDOWS = {'early': (10, 50), 'late': (125, 165)}       # samples: 40-200 ms, 500-660 ms
TAPER, N_PERM, MIN_GAP = 5, 3, 0.02
LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
LRS = [1e-3, 3e-3, 1e-2, 3e-2]
CON_ANCHOR, CON_STEPS = 1e-2, 200
BUDGETS = [25, 100, 200, 0]                               # 0 = all 3,000 fit images
RETR_BUDGETS, SEEDS = [25, 200], [0, 1, 2]

SPEC = {
    'identity':        dict(),
    'scale':           dict(sp='scale'),
    'gain':            dict(sp='gain'),
    'mix':             dict(sp='mix', graph='true'),
    'mix_shuf':        dict(sp='mix', graph='shuf'),
    'mix_pc':          dict(sp='mix_pc', graph='true'),
    'mix_pc_shuf':     dict(sp='mix_pc', graph='shuf'),
    'gainmix':         dict(sp='gainmix', graph='true'),
    'gainmix_shuf':    dict(sp='gainmix', graph='shuf'),
    'lag':             dict(lag='early'),
    'lag_late':        dict(lag='late'),
    'gainmix_lag':     dict(sp='gainmix', graph='true', lag='early'),
    'gainmix_lag_ctl': dict(sp='gainmix', graph='shuf', lag='late'),
    'fullmap':         dict(sp='full'),
}
CTRL = {'gain': 'scale', 'mix': 'mix_shuf', 'mix_pc': 'mix_pc_shuf', 'gainmix': 'gainmix_shuf',
        'lag': 'lag_late', 'gainmix_lag': 'gainmix_lag_ctl'}
FAMS = list(SPEC)
RETR_FAMS = ['scale', 'gain', 'mix', 'gainmix', 'lag', 'gainmix_lag', 'fullmap', 'gainmix_shuf', 'lag_late']
SPLIT_FAMS = ['gain', 'mix', 'mix_pc', 'gainmix', 'lag', 'gainmix_lag', 'fullmap']

# approximate 2D scalp positions (10-10 system, top view, nose up); only used to build neighbour graphs
POS = {'P7': (-.71, -.59), 'P5': (-.55, -.62), 'P3': (-.37, -.65), 'P1': (-.18, -.67), 'Pz': (0, -.68),
       'P2': (.18, -.67), 'P4': (.37, -.65), 'P6': (.55, -.62), 'P8': (.71, -.59),
       'PO7': (-.52, -.83), 'PO3': (-.27, -.86), 'POz': (0, -.87), 'PO4': (.27, -.86), 'PO8': (.52, -.83),
       'O1': (-.27, -1.0), 'Oz': (0, -1.02), 'O2': (.27, -1.0),
       'CP3': (-.36, -.33), 'CP4': (.36, -.33), 'C3': (-.37, 0), 'C4': (.37, 0), 'F5': (-.55, .62), 'F6': (.55, .62)}


def dev():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------------------------------------------
# electrode graph, lag
# ---------------------------------------------------------------------------------------------------------------
def graph(chans, k=3):
    """Row-normalized symmetric k-nearest-neighbour adjacency (zero diagonal)."""
    P = np.array([POS[c] for c in chans], float)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    A = np.zeros_like(D)
    for i in range(len(P)):
        A[i, np.argsort(D[i])[:min(k, len(P) - 1)]] = 1
    A = np.maximum(A, A.T)
    return A / A.sum(1, keepdims=True)


def graph_for(L, kind, perm):
    if kind != 'shuf':
        return L
    C = len(L)
    rng = np.random.RandomState(perm + 1)
    p = rng.permutation(C)
    while (p == np.arange(C)).mean() > 0.25:          # make sure it is really scrambled
        p = rng.permutation(C)
    return L[np.ix_(p, p)]


def taper(win, T=250):
    a, b = WINDOWS[win]
    w = np.zeros(T, np.float32)
    w[a:b] = 1
    for i in range(1, TAPER + 1):
        v = 0.5 * (1 + np.cos(np.pi * i / (TAPER + 1)))
        w[a - i] = v
        if b - 1 + i < T:
            w[b - 1 + i] = v
    return w


def shift(x, tau):
    """x[..., t - tau] with edge padding (tau > 0 delays the signal)."""
    if tau == 0:
        return x
    if tau > 0:
        return torch.cat([x[..., :1].expand(*x.shape[:-1], tau), x[..., :-tau]], -1)
    return torch.cat([x[..., -tau:], x[..., -1:].expand(*x.shape[:-1], -tau)], -1)


def lag(x, tau, w):
    if tau == 0 or w is None:
        return x
    return x + w * (shift(x, tau) - x)


# ---------------------------------------------------------------------------------------------------------------
# probes (linear, frozen)
# ---------------------------------------------------------------------------------------------------------------
def fit_probe(X, Y, max_dim=256):
    """lowlevel.readout's recipe, returned as an exact linear map: pred = flatten(x) @ W + b.
    X (n, C, T), Y (n, 6). Returns dict(W (C, T, 6), b (6,), ys (6,), alpha)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    n, C, T = X.shape
    X2 = X.reshape(n, -1).astype(np.float32)
    mu, sd = X2.mean(0), X2.std(0) + 1e-6
    Z = (X2 - mu) / sd
    pca = PCA(min(max_dim, n - 1), random_state=0).fit(Z)
    ym, ys = Y.mean(0), Y.std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(-1, 5, 13), alpha_per_target=True).fit(pca.transform(Z), (Y - ym) / ys)
    A = (pca.components_.T / sd[:, None]).astype(np.float64) @ m.coef_.T            # (D, 6)
    b = (m.intercept_ - (mu / sd + pca.mean_).astype(np.float64) @ pca.components_.T @ m.coef_.T) * ys + ym
    W = (A * ys).reshape(C, T, -1)
    # exactness check against the sklearn pipeline
    chk = m.predict(pca.transform(Z[:5])) * ys + ym
    assert np.allclose(chk, X2[:5].astype(np.float64) @ W.reshape(C * T, -1) + b, atol=1e-3 * (np.abs(chk).max() + 1)), \
        'probe linearization mismatch'
    return dict(W=W.astype(np.float32), b=b.astype(np.float32), ys=ys.astype(np.float32), alpha=m.alpha_)


def r2_of(Y, pred):
    Y, pred = np.asarray(Y, np.float64), np.asarray(pred, np.float64)
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


class ProbeT:
    """Probe tensors on the device."""

    def __init__(self, p):
        d = dev()
        self.W = torch.as_tensor(p['W'], device=d)
        self.b = torch.as_tensor(p['b'], device=d)
        self.ys = torch.as_tensor(p['ys'], device=d)

    def predict(self, x, bs=2048):
        return torch.cat([torch.einsum('nct,ctj->nj', x[i:i + bs], self.W) + self.b for i in range(0, len(x), bs)])

    def G(self, x, bs=1024):
        """G[n, c, d, j] = sum_t W[c, t, j] x[n, d, t], so pred(M x) = einsum('cd,ncdj->nj', M, G) + b."""
        return torch.cat([torch.einsum('ctj,ndt->ncdj', self.W, x[i:i + bs]) for i in range(0, len(x), bs)])


# ---------------------------------------------------------------------------------------------------------------
# transform families
# ---------------------------------------------------------------------------------------------------------------
def init_params(sp, C, d):
    z = lambda *s: torch.zeros(*s, device=d, requires_grad=True)
    if sp == 'scale':
        return {'theta': z(1)}
    if sp == 'gain':
        return {'theta': z(C)}
    if sp == 'mix':
        return {'alpha': z(1)}
    if sp == 'mix_pc':
        return {'alpha': z(C)}
    if sp == 'gainmix':
        return {'theta': z(C), 'alpha': z(1)}
    if sp == 'full':
        return {'W': torch.eye(C, device=d).requires_grad_(True)}
    return {}


def make_M(sp, p, L):
    C = L.shape[0]
    I = torch.eye(C, device=L.device)
    if sp in ('gain', 'scale'):
        return torch.exp(p['theta']) * I if sp == 'scale' else torch.diag(torch.exp(p['theta']))
    if sp == 'mix':
        return I + p['alpha'] * L
    if sp == 'mix_pc':
        return I + p['alpha'][:, None] * L
    if sp == 'gainmix':
        return torch.exp(p['theta'])[:, None] * (I + p['alpha'] * L)
    if sp == 'full':
        return p['W']
    return I


def penalty(sp, p):
    if sp == 'full':
        return (p['W'] - torch.eye(len(p['W']), device=p['W'].device)).pow(2).sum()
    return sum(v.pow(2).sum() for v in p.values()) if p else 0.0


def fit_spatial(G, Y, probe, sp, L, lam, iters=150):
    """L-BFGS on the probe loss (properties standardized) + lam * penalty. Returns (params (numpy), loss)."""
    p = init_params(sp, G.shape[1], G.device)
    opt = torch.optim.LBFGS(list(p.values()), lr=1, max_iter=iters, history_size=20,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-10)

    def loss_fn():
        pred = torch.einsum('cd,ncdj->nj', make_M(sp, p, L), G) + probe.b
        return ((pred - Y) / probe.ys).pow(2).mean() + lam * penalty(sp, p)

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        loss = float(loss_fn())
    return {k: v.detach() for k, v in p.items()}, loss


def best_lag(x, Y, probe, w):
    losses = []
    for tau in LAGS:
        with torch.no_grad():
            pred = probe.predict(lag(x, tau, w))
            losses.append(float(((pred - Y) / probe.ys).pow(2).mean()))
    return LAGS[int(np.argmin(losses))]


class Fitter:
    """Fits every family for one (person's EEG, probe) pair, caching G per lag."""

    def __init__(self, probe, L, lams):
        self.probe, self.L, self.lams = probe, torch.as_tensor(L, dtype=torch.float32, device=dev()), lams
        self.w = {k: torch.as_tensor(taper(k), device=dev()) for k in WINDOWS}

    def fit(self, fam, x, Y, perm=0, G_cache=None):
        s = SPEC[fam]
        w = self.w[s['lag']] if s.get('lag') else None
        sp = s.get('sp')
        L = torch.as_tensor(graph_for(self.L.cpu().numpy(), s.get('graph'), perm), device=dev())
        if not sp:                                      # lag only (or identity): grid over tau
            return dict(fam=fam, sp=None, L=L, tau=best_lag(x, Y, self.probe, w) if w is not None else 0, w=w,
                        params={}, loss=None, perm=perm)
        best = None
        for tau in (LAGS if w is not None else [0]):   # lag + spatial: fit the spatial part at every lag, keep the best
            key = (s.get('lag'), tau)
            if G_cache is not None and key in G_cache:
                G = G_cache[key]
            else:
                G = self.probe.G(lag(x, tau, w))
                if G_cache is not None:
                    G_cache[key] = G
            params, loss = fit_spatial(G, Y, self.probe, sp, L, self.lams.get(sp, 1e-2))
            if best is None or loss < best[2]:
                best = (tau, params, loss)
        tau, params, loss = best
        return dict(fam=fam, sp=sp, L=L, tau=tau, w=w, params=params, loss=loss, perm=perm)

    def apply(self, fit, x):
        with torch.no_grad():
            xl = lag(x, fit['tau'], fit['w'])
            if fit['sp']:
                xl = torch.einsum('cd,ndt->nct', make_M(fit['sp'], fit['params'], fit['L']), xl)
            return xl

    def r2(self, fit, x, Y, probe=None):
        probe = probe or self.probe
        with torch.no_grad():
            return float(np.mean(r2_of(Y.cpu().numpy(), probe.predict(self.apply(fit, x)).cpu().numpy())))


def perms(fam):
    return range(N_PERM) if SPEC[fam].get('graph') == 'shuf' else [0]


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, mon, people):
        from eegalign import props
        from eegalign import ubp as U
        self.mon, self.chans = mon, U.MONTAGES[mon]
        self.X, ref = {}, None
        for s in people:
            d = U.subject_montage(s, mon, keys=('train', 'train_paths', 'train_labels'))
            if ref is None:
                ref = d
                self.paths, self.labels = d['train_paths'], d['train_labels']
                self.X[s] = d['train']
            elif (d['train_paths'] == self.paths).all():
                self.X[s] = d['train']
            else:                                         # reorder to the first person's image order
                pos = {p: i for i, p in enumerate(d['train_paths'])}
                self.X[s] = d['train'][[pos[p] for p in self.paths]]
        self.Y = props.props_for(self.paths).astype(np.float32)
        held = U.val_concepts(self.labels)
        rest = np.setdiff1d(np.unique(self.labels), held)
        rng = np.random.RandomState(SPLIT_SEED)
        fitc = np.sort(rng.choice(rest, N_FIT_CONCEPTS, replace=False))
        probec = np.setdiff1d(rest, fitc)
        self.eval_idx = np.nonzero(np.isin(self.labels, held))[0]
        self.fit_idx = np.nonzero(np.isin(self.labels, fitc))[0]
        pidx = np.nonzero(np.isin(self.labels, probec))[0]
        self.probe_idx = np.sort(rng.choice(pidx, min(N_PROBE, len(pidx)), replace=False))
        h = fitc[rng.permutation(len(fitc))]
        self.half_idx = [np.nonzero(np.isin(self.labels, h[:len(h) // 2]))[0],
                         np.nonzero(np.isin(self.labels, h[len(h) // 2:]))[0]]
        for a, b in ((self.eval_idx, self.fit_idx), (self.eval_idx, self.probe_idx), (self.fit_idx, self.probe_idx)):
            assert not np.isin(self.labels[a], self.labels[b]).any(), 'concept leak between splits'
        self.L = graph(self.chans)

    def x(self, s, idx):
        return torch.as_tensor(self.X[s][idx].astype(np.float32), device=dev())

    def y(self, idx):
        return torch.as_tensor(self.Y[idx], device=dev())


def probes_for(D, base, people, pool_targets):
    """Person probes and pool probes, cached on Drive."""
    from eegalign import ubp as U
    path = os.path.join(base, f'probes_{D.mon}.pt')
    cache = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    changed = False
    Yp = D.Y[D.probe_idx]
    for s in people:
        if ('own', s) not in cache:
            print(f'  probe: person {s}', flush=True)
            cache[('own', s)] = fit_probe(D.X[s][D.probe_idx].astype(np.float32), Yp)
            changed = True
    for t in pool_targets:
        if ('pool', t) not in cache:
            srcs = [s for s in U.ENV['pool_subjects'] if s != t]
            missing = [s for s in srcs if s not in D.X]
            if missing:
                D_extra = Data(D.mon, missing)
                D.X.update(D_extra.X)
            who = np.random.RandomState(t).choice(srcs, len(D.probe_idx))
            Xp = np.stack([D.X[s][i] for s, i in zip(who, D.probe_idx)]).astype(np.float32)
            print(f'  probe: pool for target {t} (sources {srcs})', flush=True)
            cache[('pool', t)] = fit_probe(Xp, Yp)
            changed = True
    if changed:
        torch.save(cache, path)
    return cache


def load_tune(base, mon):
    path = os.path.join(base, 'tune.json')
    if not os.path.exists(path):
        raise SystemExit('run the tune command first')
    return json.load(open(path))[mon]


# ---------------------------------------------------------------------------------------------------------------
# 1a / 1b: pairs
# ---------------------------------------------------------------------------------------------------------------
def run_pairs(D, B, As, probes, lams, out, fams=FAMS, settings=('ceiling', 'heldout')):
    xf, yf, xe, ye = D.x(B, D.fit_idx), D.y(D.fit_idx), D.x(B, D.eval_idx), D.y(D.eval_idx)
    own = float(np.mean(r2_of(ye.cpu().numpy(), ProbeT(probes[('own', B)]).predict(xe).cpu().numpy())))
    rows = []
    for A in As:
        pa = ProbeT(probes[('own', A)])
        ft = Fitter(pa, D.L, lams)
        cross = float(np.mean(r2_of(ye.cpu().numpy(), pa.predict(xe).cpu().numpy())))
        for setting in settings:
            xtr, ytr = (torch.cat([xf, xe]), torch.cat([yf, ye])) if setting == 'ceiling' else (xf, yf)
            cache = {}
            for fam in fams:
                for perm in perms(fam):
                    fit = ft.fit(fam, xtr, ytr, perm, cache)
                    rows.append(dict(B=B, A=A, montage=D.mon, setting=setting, family=fam, perm=perm, own=own,
                                     cross=cross, aligned=ft.r2(fit, xe, ye), tau=fit['tau'], fit_loss=fit['loss'],
                                     n_fit=len(xtr), n_eval=len(xe)))
            del cache
        hr = {r['family']: r['aligned'] for r in rows if r['A'] == A and r['setting'] == 'heldout' and r['perm'] == 0}
        print(f'    B{B} <- A{A}: own {own:.3f} cross {cross:.3f} | held-out ' +
              ' '.join(f'{f} {hr[f]:.3f}' for f in ('gain', 'gainmix', 'lag', 'gainmix_lag', 'fullmap') if f in hr),
              flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------------------------------------------
# pool: budget curve, styles, split-half (1c)
# ---------------------------------------------------------------------------------------------------------------
def param_vec(fit):
    out = {'tau': fit['tau']}
    for k, v in fit['params'].items():
        out[k] = v.cpu().numpy().ravel()
    return out


def run_pool(D, B, probes, lams, out, styles_out):
    pp = ProbeT(probes[('pool', B)])
    ft = Fitter(pp, D.L, lams)
    xf, yf, xe, ye = D.x(B, D.fit_idx), D.y(D.fit_idx), D.x(B, D.eval_idx), D.y(D.eval_idx)
    own = float(np.mean(r2_of(ye.cpu().numpy(), ProbeT(probes[('own', B)]).predict(xe).cpu().numpy())))
    cross = float(np.mean(r2_of(ye.cpu().numpy(), pp.predict(xe).cpu().numpy())))
    rows, styles = [], {}
    for n in BUDGETS:
        for draw in (range(3) if n else [0]):
            sel = np.random.RandomState(draw).permutation(len(D.fit_idx))[:n] if n else np.arange(len(D.fit_idx))
            st = torch.as_tensor(sel, device=dev())
            cache = {}
            for fam in FAMS:
                for perm in perms(fam):
                    fit = ft.fit(fam, xf[st], yf[st], perm, cache)
                    rows.append(dict(B=B, montage=D.mon, part='budget', family=fam, perm=perm, n=len(sel), draw=draw,
                                     own=own, cross=cross, aligned=ft.r2(fit, xe, ye), tau=fit['tau']))
                    if not n and perm == 0:
                        for k, v in param_vec(fit).items():
                            styles[f'{fam}/{k}'] = v
            del cache
        print(f'    B{B} pool n={n or len(D.fit_idx)}: own {own:.3f} cross {cross:.3f} | ' + ' '.join(
            f'{r["family"]} {r["aligned"]:.3f}' for r in rows if r['n'] == (n or len(D.fit_idx)) and r['draw'] == 0
            and r['perm'] == 0 and r['family'] in ('gain', 'gainmix', 'lag', 'gainmix_lag', 'fullmap')), flush=True)
    # 1c: split-half stability. ctrl = the same fits on shuffled labels: part of each parameter does not depend on the
    # image properties (mostly shrinking the probe's predictions toward the mean) and is stable by itself, so the real
    # fits have to be MORE stable than that. The report also compares this to how much people differ (styles files).
    fits = {}
    for kind, seed in (('real', None), ('ctrl', 10)):
        for h, idx in enumerate(D.half_idx):
            x, y = D.x(B, idx), D.y(idx)
            if seed is not None:
                y = y[torch.as_tensor(np.random.RandomState(seed + h).permutation(len(y)), device=dev())]
            for fam in SPLIT_FAMS:
                fits[(kind, fam, h)] = param_vec(ft.fit(fam, x, y))
    for fam in SPLIT_FAMS:
        for kind in ('real', 'ctrl'):
            a0, a1 = fits[(kind, fam, 0)], fits[(kind, fam, 1)]
            row = dict(B=B, montage=D.mon, part='split', family=fam, kind=kind, tau0=a0['tau'], tau1=a1['tau'])
            if 'theta' in a0:
                row['r_gain'] = float(np.corrcoef(a0['theta'], a1['theta'])[0, 1])
            if 'alpha' in a0:
                if len(a0['alpha']) > 1:
                    row['r_alpha'] = float(np.corrcoef(a0['alpha'], a1['alpha'])[0, 1])
                else:
                    row['alpha0'], row['alpha1'] = float(a0['alpha'][0]), float(a1['alpha'][0])
            if 'W' in a0:
                C = int(np.sqrt(len(a0['W'])))
                off = ~np.eye(C, dtype=bool).ravel()
                row['r_map'] = float(np.corrcoef(a0['W'][off], a1['W'][off])[0, 1])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez(styles_out, chans=np.array(D.chans), **styles)


# ---------------------------------------------------------------------------------------------------------------
# 1d: retrieval
# ---------------------------------------------------------------------------------------------------------------
class TransformNet(nn.Module):
    """f0(M lag(x)). Parameters for the contrastive arm; buffers for the probe-fitted arm."""

    def __init__(self, f0, sp, L, tau, w, params=None):
        super().__init__()
        self.f0, self.sp, self.tau = f0, sp, tau
        d = next(f0.parameters()).device
        self.register_buffer('L', torch.as_tensor(L, dtype=torch.float32, device=d))
        self.register_buffer('w', w.to(d) if w is not None else torch.zeros(0, device=d))
        init = params if params is not None else init_params(sp, len(L), d)
        self.p = nn.ParameterDict({k: nn.Parameter(v.detach().clone().to(d)) for k, v in init.items()})

    def forward(self, x):
        x = x.float()
        if self.tau and self.w.numel():
            x = lag(x, self.tau, self.w)
        if self.sp:
            x = torch.einsum('cd,ndt->nct', make_M(self.sp, dict(self.p), self.L), x)
        return self.f0(x.contiguous())


@torch.no_grad()
def con_loss(m, X, Tg, T, n_neg=1024):
    f = m.f0 if hasattr(m, 'f0') else m
    z = m(X)
    g = torch.Generator(device='cpu').manual_seed(0)
    neg = T.tgt[torch.randint(T.n, (n_neg,), generator=g).to(T.tgt.device)]
    gal = torch.cat([Tg, neg])
    return float(F.cross_entropy(f.softplus(f.logit_scale) * z @ gal.T, torch.arange(len(z), device=z.device)))


def contrastive_fit(fam, T, X, Tg, L, w, lr, seed):
    import copy
    from eegalign.adapters import _train
    s = SPEC[fam]
    f0 = copy.deepcopy(T.f0).requires_grad_(False).eval()
    tau = 0
    if s.get('lag'):
        losses = [con_loss(TransformNet(f0, None, L, t, w), X, Tg, T) for t in LAGS]
        tau = LAGS[int(np.argmin(losses))]
    m = TransformNet(f0, s.get('sp'), graph_for(L, s.get('graph'), 0), tau, w)
    if not s.get('sp'):
        return m.eval(), tau
    params = list(m.p.values())
    return _train(m, T, X, Tg, params, [p.detach().clone() for p in params], lr, CON_ANCHOR, 0.0, CON_STEPS, 64,
                  seed), tau


def run_retr(D, B, probes, lams, lrs, out, budgets=RETR_BUDGETS, seeds=SEEDS, fams=RETR_FAMS, arms=('probe', 'con')):
    import copy
    from eegalign.data import Target
    from eegalign.evaluate import evaluate
    T = Target(B, D.mon, train_source=False)
    pos = {p: i for i, p in enumerate(T.P)}
    tix = np.array([pos[p] for p in D.paths[D.fit_idx]])
    assert np.allclose(T.x[tix[:50]], D.X[B][D.fit_idx[:50]].astype(np.float32)), 'Target and Data EEG differ'
    pp = ProbeT(probes[('pool', B)])
    ft = Fitter(pp, D.L, lams)
    f0 = copy.deepcopy(T.f0).requires_grad_(False).eval()
    ev = evaluate(f0, T)
    rows = [dict(B=B, montage=D.mon, family='f0', arm='none', n=0, seed=0, tau=0, test_top1=ev['test_top1'],
                 held_top1=ev['held_top1'])]
    xf, yf = D.x(B, D.fit_idx), D.y(D.fit_idx)
    for n in budgets:
        for seed in seeds:
            sel = np.random.RandomState(seed).permutation(len(D.fit_idx))[:n]
            st = torch.as_tensor(sel, device=dev())
            X, Tg = torch.as_tensor(T.x[tix[sel]], device=dev()).float(), T.tgt[tix[sel]]
            cache = {}
            for fam in fams:
                if 'probe' in arms:
                    fit = ft.fit(fam, xf[st], yf[st], 0, cache)
                    m = TransformNet(f0, fit['sp'], fit['L'].cpu().numpy(), fit['tau'], fit['w'], fit['params']).eval()
                    e = evaluate(m, T)
                    rows.append(dict(B=B, montage=D.mon, family=fam, arm='probe', n=n, seed=seed, tau=fit['tau'],
                                     test_top1=e['test_top1'], held_top1=e['held_top1']))
                if 'con' in arms and fam in CTRL:
                    sp = SPEC[fam].get('sp') or 'gainmix'
                    w = ft.w[SPEC[fam]['lag']] if SPEC[fam].get('lag') else None
                    m, tau = contrastive_fit(fam, T, X, Tg, D.L, w, lrs.get(sp, 1e-2), seed)
                    e = evaluate(m, T)
                    rows.append(dict(B=B, montage=D.mon, family=fam, arm='con', n=n, seed=seed, tau=tau,
                                     test_top1=e['test_top1'], held_top1=e['held_top1']))
                if 'con' in arms and fam == 'fullmap':
                    m, tau = contrastive_fit(fam, T, X, Tg, D.L, None, lrs.get('full', 1e-2), seed)
                    e = evaluate(m, T)
                    rows.append(dict(B=B, montage=D.mon, family=fam, arm='con', n=n, seed=seed, tau=0,
                                     test_top1=e['test_top1'], held_top1=e['held_top1']))
            del cache
            sub = [r for r in rows if r['n'] == n and r['seed'] == seed]
            print(f'    B{B} n={n} seed {seed}: f0 {ev["test_top1"]:.1f} | ' + ' '.join(
                f'{r["family"]}/{r["arm"]} {r["test_top1"]:.1f}' for r in sub), flush=True)
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------------------------------------------
# tune (subject 1 only)
# ---------------------------------------------------------------------------------------------------------------
def tune(cfg, base, mons, others):
    path = os.path.join(base, 'tune.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    for mon in mons:
        if mon in res and 'lr' in res[mon]:
            print(f'[tune] {mon}: done {res[mon]}')
            continue
        D = Data(mon, [1] + others)
        probes = probes_for(D, base, [1] + others, [1])
        rows = []
        for lam in LAMS:
            lams = {k: lam for k in ('scale', 'gain', 'mix', 'mix_pc', 'gainmix', 'full')}
            df = run_pairs(D, 1, others, probes, lams, None,
                           fams=['scale', 'gain', 'mix', 'mix_pc', 'gainmix', 'fullmap'],
                           settings=('heldout',))
            df['lam'] = lam
            rows.append(df)
        df = pd.concat(rows)
        df['sp'] = df.family.map(lambda f: SPEC[f]['sp'])
        tab = df.groupby(['sp', 'lam']).aligned.mean().unstack('lam')
        print(f'\n[tune] {mon}: held-out aligned R^2 (B = 1, mean over A), by lambda\n{tab.round(4).to_string()}')
        best = {sp: float(tab.loc[sp].idxmax()) for sp in tab.index}
        print(f'[tune] {mon}: lambda {best}')
        res[mon] = dict(lam=best)
        # contrastive lr on held top-1 (subject 1, 200 images, seed 0)
        lr_rows = []
        for lr in LRS:
            lrs = {k: lr for k in ('gain', 'mix', 'gainmix', 'full')}
            df = run_retr(D, 1, probes, best, lrs, None, budgets=[200], seeds=[0],
                          fams=['gain', 'mix', 'gainmix', 'fullmap'], arms=('con',))
            df['lr'] = lr
            lr_rows.append(df[df.arm == 'con'])
        df = pd.concat(lr_rows)
        df['sp'] = df.family.map(lambda f: SPEC[f]['sp'])
        tab = df.groupby(['sp', 'lr']).held_top1.mean().unstack('lr')
        print(f'\n[tune] {mon}: contrastive arm, held top-1 by lr\n{tab.round(2).to_string()}')
        res[mon]['lr'] = {sp: float(tab.loc[sp].idxmax()) for sp in tab.index}
        print(f'[tune] {mon}: lr {res[mon]["lr"]}')
        json.dump(res, open(path, 'w'), indent=1)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def closure(d):
    gap = d.own - d.cross
    return np.where(gap > MIN_GAP, (d.aligned - d.cross) / gap.where(gap > MIN_GAP, 1), np.nan)


def paired(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return np.nan, np.nan
    d = a[ok] - b[ok]
    return float(d.mean()), float(stats.ttest_1samp(d, 0).pvalue)


def report(base, mons):
    for mon in mons:
        print(f'\n{"=" * 100}\n{mon}\n{"=" * 100}')
        pf = sorted(glob.glob(os.path.join(base, f'pairs_t*_{mon}.csv')))
        gate = {}
        if pf:
            p = pd.concat([pd.read_csv(f) for f in pf])
            p['gain'] = p.aligned - p.cross
            p['closure'] = closure(p)
            nB = p.B.nunique()
            print(f'\n1a. CEILING (fitted and scored on the same images), {nB} people x {p.A.nunique()} probes. '
                  f'Median gap closure over pairs, mean R^2 gain:')
            c = p[p.setting == 'ceiling'].groupby('family').agg(closure=('closure', 'median'), gain=('gain', 'mean'))
            print(c.loc[[f for f in FAMS if f in c.index]].round(3).to_string())
            h = p[p.setting == 'heldout'].groupby(['family', 'B']).agg(gain=('gain', 'mean'), closure=('closure', 'median'),
                                                                     own=('own', 'mean'), cross=('cross', 'mean'))
            per = h.gain.unstack('B')
            print(f'\n1b. HELD-OUT (fit on fit concepts, scored on eval concepts). R^2 gain per person (mean over the '
                  f'other {p.groupby("B").A.nunique().mean():.0f} people\'s probes; mean over permutations for shuffled controls):')
            per = per.loc[[f for f in FAMS if f in per.index]]
            print(per.round(3).to_string())
            print(f'   own R^2 {h.own.groupby("B").mean().mean():.3f}, cross (single person probe) '
                  f'{h.cross.groupby("B").mean().mean():.3f}')
            print('\n   family        median gain  people > 0  median closure  vs control (mean diff, p)')
            for f in CTRL:
                if f not in per.index:
                    continue
                g = per.loc[f]
                ctl = per.loc[CTRL[f]] if CTRL[f] in per.index else None
                dm, pv = paired(g, ctl) if ctl is not None else (np.nan, np.nan)
                clo = h.closure.unstack('B').loc[f].median()
                print(f'   {f:14s} {g.median():10.3f}  {int((g > 0).sum()):4d}/{len(g)}  {clo:13.2f}   '
                      f'{dm:+.3f} (p={pv:.3f}) vs {CTRL[f]}')
                gate[f] = dict(median=g.median() >= 0.04, positive=(g > 0).sum() >= np.ceil(2 * len(g) / 3),
                               control=(dm > 0) and (pv < 0.05))
            if 'fullmap' in per.index:
                print(f'   {"fullmap":14s} {per.loc["fullmap"].median():10.3f}  (upper bound for spatial transforms)')
            lg = p[(p.setting == 'heldout') & p.family.isin(['lag', 'gainmix_lag'])]
            for f, g in lg.groupby('family'):
                print(f'   {f}: fitted lag at the grid edge (|tau| = {max(LAGS)} samples) in {(g.tau.abs() == max(LAGS)).mean():.0%} '
                      f'of pairs; median |tau| {g.tau.abs().median():.0f} samples ({4 * g.tau.abs().median():.0f} ms)')
        qf = sorted(glob.glob(os.path.join(base, f'pool_t*_{mon}.csv')))
        if qf:
            q = pd.concat([pd.read_csv(f) for f in qf])
            b = q[q.part == 'budget'].copy()
            b['gain'] = b.aligned - b.cross
            per = b.groupby(['family', 'n', 'B']).gain.mean().groupby(['family', 'n']).mean().unstack('n')
            print(f'\nB -> POOL probe (what f0 was trained on). R^2 gain by number of fit images (mean over people and '
                  f'draws). own {b.own.mean():.3f}, cross (pool) {b.cross.mean():.3f}:')
            print(per.loc[[f for f in FAMS if f in per.index]].round(3).to_string())
            s = q[q.part == 'split']
            if len(s):
                print('\n1c. SPLIT-HALF STABILITY (two disjoint halves of the fit concepts). Mean correlation of the '
                      'fitted parameters; ctrl = the same fits on shuffled labels:')
                cols = [c for c in ('r_gain', 'r_alpha', 'r_map') if c in s.columns]
                t = s.groupby(['family', 'kind'])[cols].mean().unstack('kind')
                print(t.round(2).to_string())
                sf = sorted(glob.glob(os.path.join(base, f'styles_t*_{mon}.npz')))
                if len(sf) > 2:
                    st = [np.load(f) for f in sf]
                    print('   between people (all fit images, mean correlation over person pairs; compare with "real"):')
                    for key in ('gain/theta', 'gainmix/theta', 'gainmix_lag/theta', 'mix_pc/alpha'):
                        V = np.stack([z[key] for z in st if key in z])
                        if len(V) > 2:
                            r = np.corrcoef(V)[np.triu_indices(len(V), 1)]
                            print(f'   {key:20s} {r.mean():.2f}')
                    for key in ('lag/tau', 'gainmix_lag/tau', 'mix/alpha', 'gainmix/alpha'):
                        v = [float(np.ravel(z[key])[0]) for z in st if key in z]
                        print(f'   {key:20s} per person: {np.round(v, 2).tolist()}')
                for f in CTRL:
                    gs = s[s.family == f]
                    if not len(gs) or f not in gate:
                        continue
                    real, null = gs[gs.kind == 'real'], gs[gs.kind == 'ctrl']
                    if f in ('gain', 'gainmix', 'gainmix_lag'):
                        a, b, what = real.r_gain.mean(), null.r_gain.mean(), 'gain r'
                    elif f == 'mix_pc':
                        a, b, what = real.r_alpha.mean(), null.r_alpha.mean(), 'alpha r'
                    elif f == 'mix':
                        a = np.corrcoef(real.alpha0, real.alpha1)[0, 1]
                        b = np.corrcoef(null.alpha0, null.alpha1)[0, 1]
                        what = 'alpha r across people'
                    else:
                        a, b, what = (real.tau0 == real.tau1).mean(), (null.tau0 == null.tau1).mean(), 'tau agreement'
                    print(f'   {f:14s} {what}: real {a:.2f}, shuffled labels {b:.2f}' + (
                        f'; tau agree {int((real.tau0 == real.tau1).sum())}/{len(real)}' if SPEC[f].get('lag') else ''))
                    gate[f]['stable'] = bool(a > b)
        rf = sorted(glob.glob(os.path.join(base, f'retr_t*_{mon}.csv')))
        if rf:
            r = pd.concat([pd.read_csv(f) for f in rf])
            f0 = r[r.family == 'f0'].set_index('B').test_top1
            x = r[r.family != 'f0'].groupby(['family', 'arm', 'n', 'B']).test_top1.mean().reset_index()
            x['d'] = x.test_top1 - x.B.map(f0)
            print(f'\n1d. RETRIEVAL: 200-way top-1 change vs f0 (f0 = {f0.mean():.1f}%), mean over people (seeds averaged '
                  f'first), [people > 0], p (paired t vs f0):')
            out = []
            for (fam, arm, n), g in x.groupby(['family', 'arm', 'n']):
                from scipy import stats
                pv = stats.ttest_1samp(g.d, 0).pvalue if len(g) > 2 else np.nan
                out.append(dict(family=fam, arm=arm, n=n, d=round(g.d.mean(), 2), pos=f'{int((g.d > 0).sum())}/{len(g)}',
                                p=round(pv, 3)))
                if arm == 'probe' and n == 25 and fam in gate:
                    gate[fam]['top1'] = g.d.mean() > 0
            print(pd.DataFrame(out).set_index(['family', 'n', 'arm']).unstack('arm').to_string())
            pc = x.pivot_table(index=['family', 'n', 'B'], columns='arm', values='test_top1').dropna()
            if len(pc):
                print('\n   probe-fitted minus contrastive-fitted (same family, same images), mean over people:')
                print((pc['probe'] - pc['con']).groupby(['family', 'n']).mean().unstack('n').round(2).to_string())
        if gate:
            print(f'\nGATE ({mon}): median gain >= 0.04, positive for >= 2/3 of people, beats control (p < .05), '
                  f'split-half above null, top-1 above f0 at 25 images (probe-fitted)')
            for f, g in gate.items():
                ok = all(g.values()) and len(g) == 5
                print(f'   {f:14s} {"PASS" if ok else "fail"}   ' + ' '.join(f'{k}={"y" if v else "n"}' for k, v in g.items())
                      + ('' if len(g) == 5 else '  (missing parts: run every phase)'))


# ---------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['tune', 'pairs', 'pool', 'retr', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument('--montages', nargs='+', default=['posterior17', 'occipital8'])
    a = p.parse_args()
    from eegalign import env
    cfg = env.load(a.config)
    base = os.path.join(os.path.dirname(cfg['results_csv']), 'transfer')
    os.makedirs(base, exist_ok=True)
    if a.cmd == 'report':
        return report(base, a.montages)
    if a.cmd == 'tune':
        return tune(cfg, base, a.montages, a.targets)
    for mon in a.montages:
        tn = load_tune(base, mon)
        todo = [t for t in a.targets if not os.path.exists(os.path.join(base, f'{a.cmd}_t{t:02d}_{mon}.csv'))]
        print(f'[{a.cmd}] {mon}: {len(todo)} people to do', flush=True)
        if not todo:
            continue
        D = Data(mon, a.targets)
        probes = probes_for(D, base, a.targets, a.targets if a.cmd != 'pairs' else [])
        for B in todo:
            out = os.path.join(base, f'{a.cmd}_t{B:02d}_{mon}.csv')
            print(f'[{a.cmd}] B = {B} {mon}', flush=True)
            if a.cmd == 'pairs':
                run_pairs(D, B, [s for s in a.targets if s != B], probes, tn['lam'], out)
            elif a.cmd == 'pool':
                run_pool(D, B, probes, tn['lam'], out, os.path.join(base, f'styles_t{B:02d}_{mon}.npz'))
            else:
                run_retr(D, B, probes, tn['lam'], tn['lr'], out)


if __name__ == '__main__':
    main()
