"""Mechanistic interpretability of cross-subject failure in NeuroBridge, and a synthetic-subject calibration pilot.

    python scripts/nb_interp.py run    --config C [--targets 2 ... 10] [--montages posterior17]   # E1-E5, E7, E8
    python scripts/nb_interp.py tune   --config C                    # channel-map lr / lambda on subject 1
    python scripts/nb_interp.py maps   --config C                    # one end-task channel map per person (E6 input)
    python scripts/nb_interp.py prior  --config C [--targets ...]    # E6: population structure + few-shot priors
    python scripts/nb_interp.py synth  --config C [--targets ...]    # pilot: virtual subjects -> calibrator
    python scripts/nb_interp.py report --config C

Models: f0_B = NeuroBridge trained on the 9 other people (leave-one-out), within_B = trained on B's own pool. Both are
the epoch-50 checkpoints (--which last; NeuroBridge's fixed schedule, no selection: the validation pick lands early
because 4-rep validation trials are much noisier than 80-rep test trials). Stages of f0 (EEGProject + projector):
  x  raw EEG (C x 250)    h1 Linear(C*250 -> 1024)    h2 + Residual(GELU, Linear)    h3 LayerNorm    z projector (512)
Data: the official 200 test images (80 reps; split into even/odd halves for noise ceilings) are only ever scored;
the 200 held-out training concepts (2,000 images, 4 reps, never trained on by any model) are used to estimate
anything that is then scored on test. All 10 people saw the same images, which is what makes patching possible.

  E1 readout    Where does the end-task information sit, and where does it stop transferring? Per stage, a ridge readout
                to f0's image embeddings fitted on B's own activations (own) vs on other people's (cross), scored on
                B's test set. own >> cross at a stage = present but misaligned there.
  E2 patching   Cross-person activation patching. Replace B's activation at a stage, in a k-dim subspace, with the
                source people's activation for the same image; top-1 vs k. Subspaces: top PCs of the B-minus-sources
                difference (diff, estimated on held-out concepts), top PCs of B's activations (var), random (rand).
                Patch sources: mean of 9 people (mean), the same with ~80 total repetitions (snr: controls for the
                noise reduction of averaging 9 people), one person at a time (single). Plus noising (B's component
                into source people). The k where recovery saturates = dimensionality of the harmful difference.
  E3 decompose  B-minus-sources difference per stage: share explained by a constant offset, by a linear map (fitted on
                held-out concepts), and the noise floor (split halves). Then a deployable test: fit offset / per-unit
                affine / linear maps from B's activations to the sources' activations on n of B's calibration images
                (other people saw the same images), apply at that stage, top-1 on test vs n. The linear map is fitted
                on the residual (sources - B), so with few images it shrinks toward the identity.
  E4 stats      Label-free statistics matching per stage (x, h1, h2, h3, z) from U unlabeled trials: mean shift,
                per-unit standardization (AdaBN-style), full covariance alignment (CORAL, Ledoit-Wolf). Control:
                standardization with permuted units.
  E5 RSA        Image x image dissimilarity matrices (200 test images) for every person, raw EEG and f0 output:
                split-half reliability, agreement across people (noise-ceiling corrected), agreement with the image
                embedding geometry.
  E7 occlusion  Which channels and 40 ms windows the end task depends on: f0 on B vs within on B vs f0 on the source
                mean EEG (drop in correct-image margin).
  E8 images     Which images fail: per-image rank of the correct image (f0 vs within), vs embedding density, low-level
                properties, categories; and 2-choice tests on held-out concepts with foils of increasing difficulty
                (unrelated concept, same THINGS category, same concept / other photo).
  E6 prior      End-task channel maps (W_p, fitted with f0_p on p's whole pool) for every person. Are they stable within
                a person (split halves) and different between people? Does the subspace of the other 9 people's maps
                capture B's map? Few-shot calibration of B inside that subspace vs a full map vs the population mean.
  synth         Pilot. Virtual people = a source person's EEG with another style: x_v = W_draw^-1 W_base x_base (so
                W_draw is exactly the map that corrects x_v). Styles: observed maps (obs), samples from the Gaussian of
                the observed maps in their PCA space (sample), random maps of matched size (generic). A calibrator
                learns style from U UNLABELED trials (evoked response + log covariance -> map) on virtual people, then
                predicts B's map from B's unlabeled trials. Arms vs the same calibrator fit on the 9 real people only
                (real9). Then k-label refinement from the predicted map vs from the identity / population mean.
Outputs: <results_dir>/nb_interp/<exp>_tNN_<montage>.csv (+ maps_<montage>.npz, tune.json); report prints tables and
saves figures there. Resumable per (command, target, montage).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
from eegalign import nb as N  # noqa: E402

STAGES = ['h1', 'h2', 'h3', 'z']
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_CAL, CAL_N = 2000, [25, 50, 100, 200, 500, 1000, 2000]
U_LIST = [25, 100, 500, 0]
FEW_K, DRAWS = [5, 25, 50, 200], 3
SYN_U, SYN_N = [50, 200, 1000], 2000
SEED = 0
STEP_SCALE = float(os.environ.get('NB_INTERP_STEP_SCALE', 1))   # < 1 only for smoke tests


def dev():
    return N.ENV['device']


def out_dir():
    d = os.path.join(N.ENV['results_dir'], 'nb_interp')
    os.makedirs(d, exist_ok=True)
    return d


def save(rows, exp, B, mon):
    pd.DataFrame(rows).to_csv(os.path.join(out_dir(), f'{exp}_t{B:02d}_{mon}.csv'), index=False)


def done(exp, B, mon):
    return os.path.exists(os.path.join(out_dir(), f'{exp}_t{B:02d}_{mon}.csv'))


# ---------------------------------------------------------------------------------------------------------------
# model pieces
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def acts(m, X, bs=1024):
    out = {s: [] for s in STAGES}
    for i in range(0, len(X), bs):
        x = torch.as_tensor(X[i:i + bs], device=dev()).float().reshape(min(bs, len(X) - i), -1)
        h1 = m.encoder.model[0](x)
        h2 = m.encoder.model[1](h1)
        h3 = m.encoder.model[2](h2)
        for s, v in zip(STAGES, (h1, h2, h3, m.eeg_projector(h3))):
            out[s].append(v)
    return {s: torch.cat(v) for s, v in out.items()}


def forward_from(m, s, a):
    if s == 'x':
        a, s = m.encoder.model[0](a.reshape(len(a), -1).float()), 'h1'
    if s == 'h1':
        a, s = m.encoder.model[1](a), 'h2'
    if s == 'h2':
        a, s = m.encoder.model[2](a), 'h3'
    if s == 'h3':
        a = m.eeg_projector(a)
    return a


@torch.no_grad()
def top1(z, G):
    return 100 * float(((F.normalize(z, dim=-1) @ G.T).argmax(1) == torch.arange(len(G), device=G.device)).float().mean())


@torch.no_grad()
def margins(z, G):
    s = F.normalize(z, dim=-1) @ G.T
    c = s.diagonal()
    o = s.clone()
    o.fill_diagonal_(-9)
    return c - o.max(1).values, (s.argmax(1) == torch.arange(len(G), device=G.device)).float()


def load_models(B, mon, which):
    f0, _ = N.load_model(N.ckpt_path('loso', mon, B, 2025, which))
    w, _ = N.load_model(N.ckpt_path('within', mon, B, 2025, which))
    return f0.requires_grad_(False), w.requires_grad_(False)


def ridge(X, Y, alphas=np.logspace(-1, 5, 13)):
    """RidgeCV on standardized X, centered Y. Returns predict(Z) -> tensor and the Y mean."""
    from sklearn.linear_model import RidgeCV
    X, Y = X.detach().cpu().numpy().astype(np.float64), Y.detach().cpu().numpy().astype(np.float64)
    mu, sd, ym = X.mean(0), X.std(0) + 1e-6, Y.mean(0)
    m = RidgeCV(alphas=alphas).fit((X - mu) / sd, Y - ym)

    def pred(Z):
        return torch.as_tensor(m.predict((Z.detach().cpu().numpy() - mu) / sd) + ym, dtype=torch.float32,
                               device=dev())
    return pred, torch.as_tensor(ym, dtype=torch.float32, device=dev())


# ---------------------------------------------------------------------------------------------------------------
# data for one target
# ---------------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, B, mon, people, which='last', need_pool=True):
        self.B, self.mon, self.people = B, mon, list(people)
        self.src = [p for p in people if p != B]
        self.f0, self.within = load_models(B, mon, which)
        self.ftr = N.features('train', 'aug').to(dev())
        fte = N.features('test', 'aug').to(dev())
        with torch.no_grad():
            self.G = self.f0.gallery(fte)
            self.Gw = self.within.gallery(fte)
            self.Gtr = self.f0.gallery(self.ftr)
        rng = np.random.RandomState(SEED)
        self.test, self.half, self.snr, self.held, self.cal, self.raw = {}, {}, {}, {}, {}, {}
        for p in people:
            d = N.subject_data(p, mon)
            if p == people[0]:
                self.labels, self.test_paths, self.train_paths = d['train_labels'], d['test_paths'], d['train_paths']
                pool, draws = N.split(self.labels)
                self.pool, self.draws = pool, draws
                self.held_idx = np.concatenate(draws)
                self.cal_idx = np.sort(np.random.RandomState(1).permutation(pool)[:N_CAL])
            tt = d['test_trials'].astype(np.float32)
            self.test[p] = d['test'].astype(np.float32)
            self.half[p] = (tt[:, 0::2].mean(1), tt[:, 1::2].mean(1))
            self.snr[p] = np.stack([tt[i, rng.choice(80, 9, replace=False)].mean(0) for i in range(len(tt))])
            self.held[p] = d['train'][self.held_idx].astype(np.float32)
            self.cal[p] = d['train'][self.cal_idx].astype(np.float32)
            if need_pool and p == B:
                self.raw[p] = d['train']
            elif need_pool:
                self.raw[p] = d['train'][rng.choice(pool, 600, replace=False)]   # for source statistics
            del d, tt
        self.C = self.test[B].shape[1]


# ---------------------------------------------------------------------------------------------------------------
# E1-E5, E7, E8
# ---------------------------------------------------------------------------------------------------------------
def e1_readout(c, A_test_B, A_held):
    rows = []
    rng = np.random.RandomState(SEED)
    pick_src = rng.choice(len(c.src), len(c.held_idx))
    Yh = c.Gtr[c.held_idx]
    for s in STAGES:
        X_own = A_held[c.B][s]
        X_cross = torch.stack([A_held[c.src[j]][s][i] for i, j in enumerate(pick_src)])
        for kind, X in (('own_held', X_own), ('cross_held', X_cross)):
            pred, ym = ridge(X, Yh)
            rows.append(dict(exp='E1', stage=s, readout=kind, top1=top1(pred(A_test_B[s]) - ym, c.G - ym)))
        pool_x = c.raw[c.B][c.pool]
        A_pool = acts(c.f0, pool_x)[s]
        pred, ym = ridge(A_pool, c.Gtr[c.pool])
        rows.append(dict(exp='E1', stage=s, readout='own_pool', top1=top1(pred(A_test_B[s]) - ym, c.G - ym)))
        del A_pool
    return rows


def basis(kind, s, k, D_held, A_held_B, seed=0):
    dim = D_held.shape[1]
    k = min(k, dim)
    if kind == 'diff':
        V = torch.linalg.svd(D_held, full_matrices=False).Vh[:k].T
    elif kind == 'var':
        V = torch.linalg.svd(A_held_B - A_held_B.mean(0), full_matrices=False).Vh[:k].T
    else:
        g = torch.Generator(device='cpu').manual_seed(seed + 17 * k)
        V, _ = torch.linalg.qr(torch.randn(dim, k, generator=g).to(D_held.device))
    return V


def e2_patching(c, A_test, A_snr, A_held):
    rows = []
    f0, B = c.f0, c.B
    rows.append(dict(exp='E2', stage='-', basis='none', k=0, source='none', top1=top1(A_test[B]['z'], c.G)))
    with torch.no_grad():
        zw = c.within(torch.as_tensor(c.test[B], device=dev()))
    rows.append(dict(exp='E2', stage='-', basis='within', k=0, source='none', top1=top1(zw, c.Gw)))
    for s in STAGES:
        mean_src = torch.stack([A_test[p][s] for p in c.src]).mean(0)
        snr_src = torch.stack([A_snr[p][s] for p in c.src]).mean(0)
        D = A_held[B][s] - torch.stack([A_held[p][s] for p in c.src]).mean(0)
        aB = A_test[B][s]
        with torch.no_grad():
            for srcname, tgt in (('mean', mean_src), ('snr', snr_src)):
                rows.append(dict(exp='E2', stage=s, basis='full', k=aB.shape[1], source=srcname,
                                 top1=top1(forward_from(f0, s, tgt), c.G)))
            single_full = np.mean([top1(forward_from(f0, s, A_test[p][s]), c.G) for p in c.src])
            rows.append(dict(exp='E2', stage=s, basis='full', k=aB.shape[1], source='single', top1=single_full))
            for kind in ('diff', 'var', 'rand'):
                for k in [k for k in KS if k < aB.shape[1]]:
                    V = basis(kind, s, k, D, A_held[B][s])
                    P = V @ V.T
                    for srcname, tgt in (('mean', mean_src), ('snr', snr_src)):
                        rows.append(dict(exp='E2', stage=s, basis=kind, k=k, source=srcname,
                                         top1=top1(forward_from(f0, s, aB + (tgt - aB) @ P), c.G)))
                    if kind == 'diff' or k in (16, 64):
                        sg = np.mean([top1(forward_from(f0, s, aB + (A_test[p][s] - aB) @ P), c.G) for p in c.src])
                        rows.append(dict(exp='E2', stage=s, basis=kind, k=k, source='single', top1=sg))
            # noising: B's component into each source person
            base = np.mean([top1(A_test[p]['z'], c.G) for p in c.src])
            for kind, k in (('full', 0), ('diff', 4), ('diff', 16), ('diff', 64)):
                P = torch.eye(aB.shape[1], device=dev()) if kind == 'full' else \
                    (lambda V: V @ V.T)(basis('diff', s, k, D, A_held[B][s]))
                t = np.mean([top1(forward_from(f0, s, A_test[p][s] + (aB - A_test[p][s]) @ P), c.G) for p in c.src])
                rows.append(dict(exp='E2noise', stage=s, basis=kind, k=k, source='B_into_src', top1=t,
                                 src_clean=base))
    return rows


def affine_fit(Xa, Ya, kind):
    """B activations -> source activations: offset, per-unit affine (shrunk), or linear (ridge)."""
    if kind == 'offset':
        mu = (Ya - Xa).mean(0)
        return lambda Z: Z + mu
    if kind == 'diag':
        xm, ym = Xa.mean(0), Ya.mean(0)
        cov = ((Xa - xm) * (Ya - ym)).mean(0)
        var = ((Xa - xm) ** 2).mean(0)
        w = (cov + 1e-3) / (var + 1e-3)
        w = 0.5 * w + 0.5                                       # shrink toward 1 (unit scale)
        return lambda Z: (Z - xm) * w + ym
    pred, _ = ridge(Xa, Ya - Xa)                                 # residual: shrinks toward identity, not the mean
    return lambda Z: Z + pred(Z)


def e3_decompose(c, A_test, A_half, A_held, A_cal):
    rows = []
    f0, B = c.f0, c.B
    for s in STAGES:
        src_test = torch.stack([A_test[p][s] for p in c.src]).mean(0)
        src_held = torch.stack([A_held[p][s] for p in c.src]).mean(0)
        D = A_test[B][s] - src_test
        tot = float((D ** 2).sum(1).mean())
        mu = (src_held - A_held[B][s]).mean(0)
        off = float(((src_test - (A_test[B][s] + mu)) ** 2).sum(1).mean())
        P = affine_fit(A_held[B][s], src_held, 'linear')(A_test[B][s])
        lin = float(((src_test - P) ** 2).sum(1).mean())
        h1, h2 = A_half[B][0][s], A_half[B][1][s]
        noise = float(((h1 - h2) ** 2).sum(1).mean()) / 4
        with torch.no_grad():
            rows.append(dict(exp='E3diag', stage=s, total=tot, frac_offset=1 - off / tot, frac_linear=1 - lin / tot,
                             frac_noise=noise / tot, top1_offset=top1(forward_from(f0, s, A_test[B][s] + mu), c.G),
                             top1_linear=top1(forward_from(f0, s, P), c.G)))
        src_cal = torch.stack([A_cal[p][s] for p in c.src]).mean(0)
        for n in CAL_N:
            for draw in range(DRAWS if n <= 200 else 1):
                ix = torch.as_tensor(np.random.RandomState(100 + draw).permutation(N_CAL)[:n], device=dev())
                for kind in ('offset', 'diag', 'linear'):
                    fn = affine_fit(A_cal[B][s][ix], src_cal[ix], kind)
                    with torch.no_grad():
                        t = top1(forward_from(f0, s, fn(A_test[B][s])), c.G)
                    rows.append(dict(exp='E3cal', stage=s, method=kind, n=n, draw=draw, top1=t))
    return rows


def lw_cov(X):
    from sklearn.covariance import ledoit_wolf
    C, _ = ledoit_wolf(X.detach().cpu().numpy().astype(np.float64))
    return torch.as_tensor(C, dtype=torch.float32, device=dev())


def msqrt(C, p):
    w, V = torch.linalg.eigh(C)
    return (V * w.clamp_min(1e-8) ** p) @ V.T


def e4_stats(c):
    rows = []
    f0, B = c.f0, c.B
    src_raw = np.concatenate([c.raw[p] for p in c.src]).astype(np.float32)
    A_src = acts(f0, src_raw)
    xt = torch.as_tensor(c.test[B], device=dev())
    A_tB = acts(f0, c.test[B])
    with torch.no_grad():
        rows.append(dict(exp='E4', stage='-', method='none', U=0, top1=top1(A_tB['z'], c.G)))
    rng = np.random.RandomState(SEED)
    for U in U_LIST:
        sel = c.pool if U == 0 else rng.choice(c.pool, U, replace=False)
        xb = c.raw[B][sel].astype(np.float32)
        A_B = acts(f0, xb)
        # raw EEG: per-channel standardization and channel-covariance alignment
        xbt = torch.as_tensor(xb, device=dev())
        srt = torch.as_tensor(src_raw, device=dev())
        mb, sb = xbt.mean((0, 2)), xbt.std((0, 2)) + 1e-6
        ms, ss = srt.mean((0, 2)), srt.std((0, 2)) + 1e-6
        with torch.no_grad():
            z = forward_from(f0, 'x', ((xt - mb[None, :, None]) / sb[None, :, None]) * ss[None, :, None] + ms[None, :, None])
            rows.append(dict(exp='E4', stage='x', method='adabn', U=U, top1=top1(z, c.G)))
            cb = torch.einsum('nct,ndt->cd', xbt - xbt.mean(2, keepdim=True), xbt - xbt.mean(2, keepdim=True)) / (xbt.shape[0] * xbt.shape[2])
            cs = torch.einsum('nct,ndt->cd', srt - srt.mean(2, keepdim=True), srt - srt.mean(2, keepdim=True)) / (srt.shape[0] * srt.shape[2])
            Tm = msqrt(cs, 0.5) @ msqrt(cb + 1e-3 * torch.trace(cb) / len(cb) * torch.eye(len(cb), device=dev()), -0.5)
            rows.append(dict(exp='E4', stage='x', method='coral', U=U,
                             top1=top1(forward_from(f0, 'x', torch.einsum('cd,ndt->nct', Tm, xt)), c.G)))
        for s in STAGES:
            a = A_tB[s]
            mB, sB = A_B[s].mean(0), A_B[s].std(0) + 1e-6
            mS, sS = A_src[s].mean(0), A_src[s].std(0) + 1e-6
            perm = torch.as_tensor(np.random.RandomState(7).permutation(len(mS)), device=dev())
            with torch.no_grad():
                for meth, a2 in (('mean', a - mB + mS), ('adabn', (a - mB) / sB * sS + mS),
                                 ('perm', (a - mB) / sB * sS[perm] + mS[perm])):
                    rows.append(dict(exp='E4', stage=s, method=meth, U=U, top1=top1(forward_from(f0, s, a2), c.G)))
            CB, CS = lw_cov(A_B[s]), lw_cov(A_src[s])
            Tm = msqrt(CB, -0.5) @ msqrt(CS, 0.5)
            with torch.no_grad():
                rows.append(dict(exp='E4', stage=s, method='coral', U=U,
                                 top1=top1(forward_from(f0, s, (a - mB) @ Tm + mS), c.G)))
        del A_B
    return rows


def rdm(X):
    X = X.reshape(len(X), -1).double()
    X = (X - X.mean(1, keepdim=True)) / (X.std(1, keepdim=True) + 1e-9)
    return 1 - (X @ X.T) / X.shape[1]


def tri(R):
    i = torch.triu_indices(len(R), len(R), 1)
    return R[i[0], i[1]].cpu().numpy()


def spear(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def e5_rsa(c, A_test, A_half):
    rows = []
    Rg = tri(1 - (c.G @ c.G.T).double())
    Rgw = tri(1 - (c.Gw @ c.Gw.T).double())
    raw = {p: torch.as_tensor(c.test[p], device=dev()) for p in c.people}
    rawh = {p: [torch.as_tensor(h, device=dev()) for h in c.half[p]] for p in c.people}
    T = {('raw', p): tri(rdm(raw[p])) for p in c.people}
    T.update({('z', p): tri(rdm(A_test[p]['z'])) for p in c.people})
    for p in c.people:
        r_raw = spear(tri(rdm(rawh[p][0])), tri(rdm(rawh[p][1])))
        r_z = spear(tri(rdm(A_half[p][0]['z'])), tri(rdm(A_half[p][1]['z'])))
        rows.append(dict(exp='E5', person=p, is_B=p == c.B, space='raw', reliability=2 * r_raw / (1 + r_raw),
                         to_gallery=np.nan, to_B=spear(T[('raw', p)], T[('raw', c.B)]) if p != c.B else np.nan))
        rows.append(dict(exp='E5', person=p, is_B=p == c.B, space='z_f0B', reliability=2 * r_z / (1 + r_z),
                         to_gallery=spear(T[('z', p)], Rg),
                         to_B=spear(T[('z', p)], T[('z', c.B)]) if p != c.B else np.nan))
    with torch.no_grad():
        zw = c.within(raw[c.B])
    rows.append(dict(exp='E5', person=c.B, is_B=True, space='z_within', reliability=np.nan,
                     to_gallery=spear(tri(rdm(zw)), Rgw), to_B=np.nan))
    return rows


def e7_occlusion(c):
    rows = []
    B = c.B
    xs = {'f0_on_B': (c.f0, c.test[B], c.G), 'within_on_B': (c.within, c.test[B], c.Gw),
          'f0_on_srcEEG': (c.f0, np.mean([c.test[p] for p in c.src], 0), c.G)}
    from eegalign.ubp import MONTAGES
    chans = MONTAGES[c.mon]
    for name, (m, X, G) in xs.items():
        X = torch.as_tensor(X, device=dev())
        with torch.no_grad():
            m0, t0 = margins(m(X), G)
            mean = X.mean(0, keepdim=True)
            for ci, ch in enumerate(chans):
                Xo = X.clone()
                Xo[:, ci] = mean[:, ci]
                mo, to = margins(m(Xo), G)
                rows.append(dict(exp='E7', model=name, kind='channel', where=ch, d_margin=float((m0 - mo).mean()),
                                 d_top1=100 * float((t0 - to).mean())))
            for w0 in range(0, 250, 10):
                Xo = X.clone()
                Xo[:, :, w0:w0 + 10] = mean[:, :, w0:w0 + 10]
                mo, to = margins(m(Xo), G)
                rows.append(dict(exp='E7', model=name, kind='window', where=4 * (w0 + 5), d_margin=float((m0 - mo).mean()),
                                 d_top1=100 * float((t0 - to).mean())))
    return rows


def e8_images(c, A_test, A_held, ubp_feat_dir):
    rows, rows2 = [], []
    B = c.B
    with torch.no_grad():
        zw = c.within(torch.as_tensor(c.test[B], device=dev()))
        zs = torch.stack([A_test[p]['z'] for p in c.src]).mean(0)

        def ranks(z, G):
            s = F.normalize(z, dim=-1) @ G.T
            return (s > s.diagonal()[:, None]).sum(1).cpu().numpy()
        r_f0, r_w, r_src = ranks(A_test[B]['z'], c.G), ranks(zw, c.Gw), ranks(zs, c.G)
        S = c.G @ c.G.T
        S.fill_diagonal_(-9)
        dens = S.topk(5, dim=1).values.mean(1).cpu().numpy()
    props = cats = None
    if ubp_feat_dir and not str(c.test_paths[0]).startswith('test/'):
        try:
            from eegalign import props as Pr
            from eegalign import ubp as U
            U.ENV['feat_dir'] = ubp_feat_dir
            props = Pr.props_for(c.test_paths)
        except Exception as e:                                     # noqa: BLE001
            print('   (low-level properties unavailable:', e, ')')
        try:
            from eegalign import categories as Ca
            cats = Ca.labels(c.test_paths, min_concepts=1)
        except Exception as e:                                     # noqa: BLE001
            print('   (categories unavailable:', e, ')')
    for i in range(len(r_f0)):
        row = dict(exp='E8', image=i, rank_f0=int(r_f0[i]), rank_within=int(r_w[i]), rank_f0_srcmean=int(r_src[i]),
                   density=float(dens[i]))
        if props is not None:
            row.update({f'prop{j}': float(v) for j, v in enumerate(props[i])})
        if cats is not None:
            row['category'] = cats[1][cats[0][i]] if cats[0][i] >= 0 else ''
        rows.append(row)
    # 2-choice on held-out concepts (never trained on by either model)
    lab = c.labels[c.held_idx]
    cat_held = None
    if not str(c.train_paths[0]).startswith('train/'):
        try:
            from eegalign import categories as Ca
            cat_held = Ca.labels(c.train_paths[c.held_idx], min_concepts=1)[0]
        except Exception:                                          # noqa: BLE001
            pass
    with torch.no_grad():
        Gh = c.f0.gallery(c.ftr[c.held_idx])
        Ghw = c.within.gallery(c.ftr[c.held_idx])
        zw_h = c.within(torch.as_tensor(c.held[B], device=dev()))
        zsrc_h = torch.stack([A_held[p]['z'] for p in c.src]).mean(0)
    rng = np.random.RandomState(SEED)
    foils = {'sibling': [], 'unrelated': [], 'category': []}
    for i in range(len(lab)):
        same = np.nonzero(lab == lab[i])[0]
        foils['sibling'].append(same[same != i])
        other = np.nonzero(lab != lab[i])[0]
        foils['unrelated'].append(rng.choice(other, 9, replace=False))
        if cat_held is not None and cat_held[i] >= 0:
            cc = np.nonzero((cat_held == cat_held[i]) & (lab != lab[i]))[0]
            foils['category'].append(rng.choice(cc, min(9, len(cc)), replace=False) if len(cc) else np.array([], int))
        else:
            foils['category'].append(np.array([], int))
    for name, z, G in (('f0_on_B', A_held[B]['z'], Gh), ('within_on_B', zw_h, Ghw), ('f0_on_srcmean', zsrc_h, Gh)):
        S = (F.normalize(z, dim=-1) @ G.T).cpu().numpy()
        for kind, fl in foils.items():
            acc = [float((S[i, i] > S[i, f]).mean()) for i, f in enumerate(fl) if len(f)]
            if acc:
                rows2.append(dict(exp='E8choice', model=name, foil=kind, acc=100 * np.mean(acc), n=len(acc)))
    return rows + rows2


def run_target(B, mon, people, which, ubp_feat_dir):
    t0 = time.time()
    c = Ctx(B, mon, people, which)
    A_test = {p: acts(c.f0, c.test[p]) for p in people}
    A_snr = {p: acts(c.f0, c.snr[p]) for p in people}
    A_half = {p: (acts(c.f0, c.half[p][0]), acts(c.f0, c.half[p][1])) for p in people}
    A_held = {p: acts(c.f0, c.held[p]) for p in people}
    A_cal = {p: acts(c.f0, c.cal[p]) for p in people}
    rows = []
    for name, fn in (('E2', lambda: e2_patching(c, A_test, A_snr, A_held)),
                     ('E3', lambda: e3_decompose(c, A_test, A_half, A_held, A_cal)),
                     ('E1', lambda: e1_readout(c, A_test[B], A_held)),
                     ('E4', lambda: e4_stats(c)),
                     ('E5', lambda: e5_rsa(c, A_test, A_half)),
                     ('E7', lambda: e7_occlusion(c)),
                     ('E8', lambda: e8_images(c, A_test, A_held, ubp_feat_dir))):
        t1 = time.time()
        r = fn()
        rows += r
        print(f'   {name}: {len(r)} rows ({time.time() - t1:.0f}s)', flush=True)
    for r in rows:
        r.update(target=B, montage=mon)
    save(rows, 'interp', B, mon)
    print(f'  target {B} {mon}: done in {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# channel maps (E6, synth)
# ---------------------------------------------------------------------------------------------------------------
def fit_map(f0, X, img, Gtr, base, lam, lr, steps, seed, Bm=None, theta0=None, batch=256, n_neg=1024):
    """W = base + reshape(Bm @ theta) (Bm = None: full C x C), theta from theta0 (default 0); loss = CLIP-style
    cross-entropy EEG -> image with in-batch + 1,024 random training-image negatives (f0's gallery), + lam ||theta||^2.
    f0 frozen, eval mode."""
    C = base.shape[0]
    steps = max(3, int(steps * STEP_SCALE))
    torch.manual_seed(seed)
    X = torch.as_tensor(X, device=dev()).float()
    img = torch.as_tensor(img, device=dev())
    r = C * C if Bm is None else Bm.shape[1]
    th = (torch.zeros(r, device=dev()) if theta0 is None else theta0.clone().to(dev())).requires_grad_(True)
    opt = torch.optim.Adam([th], lr=lr)
    scale = f0.scale()
    n, bs = len(X), min(batch, len(X))

    def W():
        return base + (th if Bm is None else Bm @ th).reshape(C, C)
    for _ in range(steps):
        i = torch.randperm(n, device=dev())[:bs]
        z = f0(torch.einsum('cd,ndt->nct', W(), X[i]))
        gal = torch.cat([Gtr[img[i]], Gtr[torch.randint(len(Gtr), (n_neg,), device=dev())]])
        loss = F.cross_entropy(scale * z @ gal.T, torch.arange(bs, device=dev())) + lam * (th ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return W().detach()


@torch.no_grad()
def top1_W(f0, W, X, G):
    return top1(f0(torch.einsum('cd,ndt->nct', W, torch.as_tensor(X, device=dev()).float())), G)


@torch.no_grad()
def held_top1(f0, W, c):
    out = []
    for j, ix in enumerate(c.draws):
        Gh = f0.gallery(c.ftr[ix])
        out.append(top1_W(f0, W, c.held[c.B][j * len(ix):(j + 1) * len(ix)], Gh))
    return float(np.mean(out))


def tune_maps(people, mon, which):
    path = os.path.join(out_dir(), 'tune.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    if mon in res:
        print('[tune]', mon, res[mon])
        return res[mon]
    c = Ctx(1, mon, people, which, need_pool=True)
    I = torch.eye(c.C, device=dev())
    rows = []
    for lr in (3e-4, 1e-3, 3e-3):
        for lam in (1e-3, 1e-2, 1e-1):
            vals = []
            for draw in range(2):
                sel = np.random.RandomState(200 + draw).choice(c.pool, 200, replace=False)
                W = fit_map(c.f0, c.raw[1][sel], sel, c.Gtr, I, lam, lr, 300, draw)
                vals.append(held_top1(c.f0, W, c))
            rows.append(dict(lr=lr, lam=lam, held_top1=float(np.mean(vals))))
            print(f'   lr {lr} lam {lam}: held top-1 {rows[-1]["held_top1"]:.2f}', flush=True)
    d = pd.DataFrame(rows)
    best = d.loc[d.held_top1.idxmax()]
    res[mon] = dict(lr=float(best.lr), lam=float(best.lam), f0_held=held_top1(c.f0, I, c), grid=rows)
    json.dump(res, open(path, 'w'), indent=1)
    print('[tune]', mon, {k: v for k, v in res[mon].items() if k != 'grid'})
    return res[mon]


def person_maps(people, mon, which, hp):
    path = os.path.join(out_dir(), f'maps_{mon}.npz')
    have = dict(np.load(path)) if os.path.exists(path) else {}
    for p in people:
        if f'{p}_full' in have:
            continue
        t0 = time.time()
        f0, _ = N.load_model(N.ckpt_path('loso', mon, p, 2025, which))
        f0.requires_grad_(False)
        d = N.subject_data(p, mon)
        pool, _ = N.split(d['train_labels'])
        ftr = N.features('train', 'aug').to(dev())
        with torch.no_grad():
            Gtr, G = f0.gallery(ftr), f0.gallery(N.features('test', 'aug').to(dev()))
        C = d['test'].shape[1]
        I = torch.eye(C, device=dev())
        X = d['train'][pool].astype(np.float32)
        have[f'{p}_full'] = fit_map(f0, X, pool, Gtr, I, hp['lam'] / 10, hp['lr'], 1500, 0).cpu().numpy()
        lab = d['train_labels'][pool]
        con = np.unique(lab)
        half = np.isin(lab, np.random.RandomState(5).permutation(con)[:len(con) // 2])
        for h, msk in (('a', half), ('b', ~half)):
            have[f'{p}_{h}'] = fit_map(f0, X[msk], pool[msk], Gtr, I, hp['lam'] / 10, hp['lr'], 1000, 1).cpu().numpy()
        have[f'{p}_top1_f0'] = np.array(top1_W(f0, I, d['test'], G))
        have[f'{p}_top1_map'] = np.array(top1_W(f0, torch.as_tensor(have[f'{p}_full'], device=dev()), d['test'], G))
        np.savez(path, **have)
        print(f'  person {p} {mon}: f0 {float(have[f"{p}_top1_f0"]):.1f} -> full map {float(have[f"{p}_top1_map"]):.1f} '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)
    return have


def pca_maps(maps, people, C):
    V = np.stack([(maps[f'{p}_full'] - np.eye(C)).ravel() for p in people])
    m = V.mean(0)
    U_, S, Vt = np.linalg.svd(V - m, full_matrices=False)
    return m, Vt, S / np.sqrt(max(len(V) - 1, 1)), V


def run_prior(B, mon, people, which, hp, maps):
    c = Ctx(B, mon, people, which)
    C, I = c.C, torch.eye(c.C, device=dev())
    others = [p for p in people if p != B]
    m, Vt, sd, V = pca_maps(maps, others, C)
    vB = (maps[f'{B}_full'] - np.eye(C)).ravel()
    rows = []
    d = vB - m
    for r in range(1, len(others)):
        P = Vt[:r].T
        cap = 1 - np.sum((d - P @ (P.T @ d)) ** 2) / np.sum(d ** 2)
        Q, _ = np.linalg.qr(np.random.RandomState(r).randn(len(m), r))
        capr = 1 - np.sum((d - Q @ (Q.T @ d)) ** 2) / np.sum(d ** 2)
        rows.append(dict(exp='E6capture', r=r, capture=cap, capture_rand=capr))
    cosm = float(vB @ m / (np.linalg.norm(vB) * np.linalg.norm(m) + 1e-12))
    within = []
    for p in people:
        a, b = (maps[f'{p}_a'] - np.eye(C)).ravel(), (maps[f'{p}_b'] - np.eye(C)).ravel()
        within.append(float(np.corrcoef(a, b)[0, 1]))
    between = [float(np.corrcoef(vB, (maps[f'{o}_full'] - np.eye(C)).ravel())[0, 1]) for o in others]
    rows.append(dict(exp='E6stable', within_B=within[people.index(B)], within_mean=float(np.mean(within)),
                     between_B=float(np.mean(between)), cos_B_to_mean=cosm,
                     top1_f0=float(maps[f'{B}_top1_f0']), top1_fullmap=float(maps[f'{B}_top1_map'])))
    M = torch.as_tensor(m.reshape(C, C), device=dev()).float() + I
    rows.append(dict(exp='E6few', variant='f0', k=0, draw=0, top1=top1_W(c.f0, I, c.test[B], c.G)))
    rows.append(dict(exp='E6few', variant='mean', k=0, draw=0, top1=top1_W(c.f0, M, c.test[B], c.G)))
    for k in FEW_K:
        for draw in range(DRAWS):
            sel = np.random.RandomState(300 + draw).choice(c.pool, k, replace=False)
            X = c.raw[B][sel]
            for variant in ('full_I', 'full_M', 'sub2', 'sub4', 'sub8', 'rand4'):
                if variant == 'full_I':
                    W = fit_map(c.f0, X, sel, c.Gtr, I, hp['lam'], hp['lr'], 300, draw)
                elif variant == 'full_M':
                    W = fit_map(c.f0, X, sel, c.Gtr, M, hp['lam'], hp['lr'], 300, draw)
                else:
                    r = int(variant[-1])
                    r = min(r, len(others) - 1)
                    Bm = torch.as_tensor(Vt[:r].T if variant.startswith('sub') else
                                         np.linalg.qr(np.random.RandomState(draw).randn(len(m), r))[0],
                                         device=dev()).float()
                    W = fit_map(c.f0, X, sel, c.Gtr, M, hp['lam'], hp['lr'] * 3, 300, draw, Bm=Bm)
                rows.append(dict(exp='E6few', variant=variant, k=k, draw=draw, top1=top1_W(c.f0, W, c.test[B], c.G)))
        print(f'   k={k}: ' + ' '.join(f'{r["variant"]} {r["top1"]:.1f}' for r in rows if r.get('k') == k
                                       and r.get('draw') == 0), flush=True)
    for r in rows:
        r.update(target=B, montage=mon)
    save(rows, 'prior', B, mon)


# ---------------------------------------------------------------------------------------------------------------
# synthetic-subject pilot
# ---------------------------------------------------------------------------------------------------------------
def stats_of(X):
    """Evoked response (C x 25 bins of 40 ms) and covariance of the residual, from trials X (n, C, 250)."""
    X = torch.as_tensor(X, device=dev()).float()
    erp = X.mean(0)
    R = X - erp
    cov = torch.einsum('nct,ndt->cd', R, R) / (R.shape[0] * R.shape[2])
    return erp, cov


def feats(erp, cov):
    C = cov.shape[0]
    e = erp.reshape(C, 25, -1).mean(-1).ravel()
    w, V = torch.linalg.eigh(cov + 1e-4 * torch.trace(cov) / C * torch.eye(C, device=cov.device))
    L = (V * torch.log(w.clamp_min(1e-10))) @ V.T
    iu = torch.triu_indices(C, C)
    return torch.cat([e, L[iu[0], iu[1]]])


def run_synth(B, mon, people, which, hp, maps):
    t0 = time.time()
    c = Ctx(B, mon, people, which)
    C, I = c.C, torch.eye(c.C, device=dev())
    others = [p for p in people if p != B]
    m, Vt, sd, V = pca_maps(maps, others, C)
    Wobs = {p: torch.as_tensor(maps[f'{p}_full'], device=dev()).float() for p in others}
    M = torch.as_tensor(m.reshape(C, C), device=dev()).float() + I
    r = len(others) - 1
    P = torch.as_tensor(Vt[:r], device=dev()).float()
    sdt = torch.as_tensor(sd[:r], device=dev()).float()
    fro = float(np.mean([np.linalg.norm(maps[f'{p}_full'] - np.eye(C)) for p in others]))
    rng = np.random.RandomState(SEED)
    # base people: their own trials (any pool image), 1,000 per person kept on the GPU
    base = {p: torch.as_tensor(N.subject_data(p, mon, keys=('train',))['train'][rng.choice(c.pool, 1000, replace=False)],
                               device=dev()).float() for p in others}
    gen = torch.Generator(device='cpu').manual_seed(SEED)

    def style(kind, b):
        if kind == 'obs':
            d = rng.choice([p for p in others if p != b])
            return Wobs[d]
        if kind == 'sample':
            z = torch.randn(r, generator=gen).to(dev())
            return M + ((z * sdt) @ P).reshape(C, C)
        E = torch.randn(C, C, generator=gen).to(dev())
        return I + E / E.norm() * fro

    def virtual(kind, n):
        Xf, Yf = [], []
        while len(Xf) < n:
            b = rng.choice(others)
            Wd = style(kind, b)
            if torch.linalg.cond(Wd) > 1e3:
                continue
            A = torch.linalg.solve(Wd, Wobs[b])                 # x_v = W_d^-1 W_b x_b
            U = int(rng.choice(SYN_U))
            erp, cov = stats_of(base[b][torch.as_tensor(rng.choice(1000, min(U, 1000), replace=False), device=dev())])
            Xf.append(feats(A @ erp, A @ cov @ A.T))
            Yf.append((Wd - I).ravel())
        return torch.stack(Xf), torch.stack(Yf)

    sets = {k: virtual(k, SYN_N) for k in ('obs', 'sample', 'generic')}
    Xr, Yr = [], []
    for p in others:                                              # real9: each real person with their own map
        for U in SYN_U:
            for _ in range(5):
                erp, cov = stats_of(base[p][torch.as_tensor(rng.choice(1000, U, replace=False), device=dev())])
                Xr.append(feats(erp, cov))
                Yr.append(Wobs[p].ravel() - I.ravel())
    sets['real9'] = (torch.stack(Xr), torch.stack(Yr))
    preds = {k: ridge(X, Y) for k, (X, Y) in sets.items()}
    rows = []
    vB = torch.as_tensor(maps[f'{B}_full'] - np.eye(C), device=dev()).float().ravel()
    rows.append(dict(exp='synth', arm='f0', U=0, draw=0, top1=top1_W(c.f0, I, c.test[B], c.G), cos_to_WB=0.0))
    rows.append(dict(exp='synth', arm='pop_mean', U=0, draw=0, top1=top1_W(c.f0, M, c.test[B], c.G),
                     cos_to_WB=float(F.cosine_similarity((M - I).ravel(), vB, dim=0))))
    rows.append(dict(exp='synth', arm='oracle_WB_alllabels', U=0, draw=0,
                     top1=float(maps[f'{B}_top1_map']), cos_to_WB=1.0))
    # in-distribution check: each predictor on fresh virtual people of its own kind
    for k in ('obs', 'sample', 'generic'):
        Xv, Yv = virtual(k, 200)
        pr, _ = preds[k]
        rows.append(dict(exp='synth_indist', arm=k, cos=float(F.cosine_similarity(pr(Xv), Yv, dim=1).mean())))
    predicted = {}
    for U in SYN_U:
        for draw in range(DRAWS):
            sel = np.random.RandomState(400 + draw).choice(c.pool, U, replace=False)
            f = feats(*stats_of(c.raw[B][sel]))[None]
            for k, (pr, _) in preds.items():
                yh = pr(f)[0]
                W = I + yh.reshape(C, C)
                ok = torch.linalg.cond(W) < 1e3
                rows.append(dict(exp='synth', arm=k, U=U, draw=draw,
                                 top1=top1_W(c.f0, W, c.test[B], c.G) if ok else np.nan,
                                 cos_to_WB=float(F.cosine_similarity(yh, vB, dim=0))))
                if U == 200 and draw == 0:
                    predicted[k] = W
    # labeled refinement from the predicted map (U = 200 unlabeled) vs from I / population mean
    for kk in (25, 200):
        for draw in range(DRAWS):
            sel = np.random.RandomState(500 + draw).choice(c.pool, kk, replace=False)
            X = c.raw[B][sel]
            for start, W0 in [('I', I), ('pop_mean', M)] + [(f'pred_{k}', W) for k, W in predicted.items()]:
                W = fit_map(c.f0, X, sel, c.Gtr, W0, hp['lam'], hp['lr'], 300, draw)
                rows.append(dict(exp='synth_refine', arm=start, k=kk, draw=draw, top1=top1_W(c.f0, W, c.test[B], c.G)))
    for rr in rows:
        rr.update(target=B, montage=mon)
    save(rows, 'synth', B, mon)
    print(f'  synth target {B} {mon}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def paired(a, b):
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    return (float(d.mean()), float(stats.ttest_1samp(d, 0).pvalue)) if len(d) > 2 else (np.nan, np.nan)


def report(mons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pd.set_option('display.width', 220)
    for mon in mons:
        print(f'\n{"=" * 100}\n{mon}\n{"=" * 100}')
        f = sorted(glob.glob(os.path.join(out_dir(), f'interp_t*_{mon}.csv')))
        if f:
            d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
            nT = d.target.nunique()
            e2 = d[d.exp == 'E2']
            f0 = e2[e2.basis == 'none'].set_index('target').top1
            wi = e2[e2.basis == 'within'].set_index('target').top1
            print(f'\n{nT} targets. f0 {f0.mean():.1f}%, within {wi.mean():.1f}% (test top-1)')
            e1 = d[d.exp == 'E1'].pivot_table(index='stage', columns='readout', values='top1', aggfunc='mean')
            print('\nE1 READOUT to f0\'s image embeddings, by stage (own = fitted on B, cross = on other people):')
            print(e1.reindex(STAGES).round(1).to_string())
            print('\nE2 PATCHING (patch source = mean of the 9 others; snr = same with ~80 total reps; top-1 %):')
            for s in STAGES:
                g = e2[e2.stage == s]
                t = g[g.source.isin(['mean', 'snr'])].pivot_table(index='k', columns=['basis', 'source'],
                                                                   values='top1', aggfunc='mean')
                print(f'  stage {s}:')
                print(t.round(1).to_string())
                sg = g[g.source == 'single'].pivot_table(index='k', columns='basis', values='top1', aggfunc='mean')
                print('   single-person patch source:', sg.round(1).to_dict())
            nz = d[d.exp == 'E2noise'].pivot_table(index='stage', columns=['basis', 'k'], values='top1', aggfunc='mean')
            cl = d[d.exp == 'E2noise'].groupby('stage').src_clean.mean()
            print('\n  NOISING: B\'s component patched into each source person (their clean top-1 '
                  f'{cl.mean():.1f}%):')
            print(nz.reindex(STAGES).round(1).to_string())
            fig, ax = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
            for i, s in enumerate(STAGES):
                g = e2[(e2.stage == s)]
                for kind, col in (('diff', 'C3'), ('var', 'C0'), ('rand', 'C7')):
                    for src, ls in (('mean', '-'), ('snr', '--')):
                        h = g[(g.basis == kind) & (g.source == src)].groupby('k').top1.mean()
                        ax[i].plot(h.index, h.values, ls, color=col, label=f'{kind} / {src}')
                ax[i].axhline(f0.mean(), color='k', lw=.8)
                ax[i].axhline(wi.mean(), color='g', lw=.8)
                ax[i].set(xscale='log', title=f'patch at {s}', xlabel='k (dims patched)')
            ax[0].set_ylabel('top-1 % (B test)')
            ax[0].legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir(), f'E2_patching_{mon}.png'), dpi=120)
            plt.close()
            dg = d[d.exp == 'E3diag'].groupby('stage')[['frac_offset', 'frac_linear', 'frac_noise', 'top1_offset',
                                                        'top1_linear']].mean()
            print('\nE3 DIFFERENCE (B minus sources, test set): share explained by an offset / a linear map (fit on '
                  'held-out concepts), noise share; top-1 after correcting:')
            print(dg.reindex(STAGES).round(3).to_string())
            ca = d[d.exp == 'E3cal'].groupby(['stage', 'method', 'n', 'target']).top1.mean().groupby(
                ['stage', 'method', 'n']).mean().unstack('n')
            print('\n  deployable activation matching (map B -> sources on n calibration images), top-1:')
            print(ca.round(1).to_string())
            e4 = d[d.exp == 'E4']
            t4 = e4[e4.method != 'none'].pivot_table(index=['stage', 'method'], columns='U', values='top1',
                                                     aggfunc='mean')
            print(f'\nE4 LABEL-FREE STATISTICS MATCHING (U unlabeled trials; 0 = all), f0 = {f0.mean():.1f}:')
            print(t4.round(1).to_string())
            e5 = d[d.exp == 'E5']
            rB = e5[(e5.space == 'raw') & e5.is_B].set_index('target').reliability
            print('\nE5 RSA: reliability (split-half, Spearman-Brown) and agreement:')
            print(e5.groupby(['space', 'is_B'])[['reliability', 'to_gallery', 'to_B']].mean().round(3).to_string())
            from scipy.stats import spearmanr
            if len(rB) > 3:
                print(f'   across targets: raw-EEG reliability vs f0 top-1 Spearman r = '
                      f'{spearmanr(rB.values, f0.reindex(rB.index).values).correlation:.2f}; vs within top-1 r = '
                      f'{spearmanr(rB.values, wi.reindex(rB.index).values).correlation:.2f}')
            e7 = d[d.exp == 'E7']
            for kind in ('channel', 'window'):
                t = e7[e7.kind == kind].pivot_table(index='where', columns='model', values='d_margin', aggfunc='mean')
                print(f'\nE7 OCCLUSION by {kind} (mean drop in correct-image margin):')
                print(t.round(4).to_string())
                print('   profile correlation f0_on_B vs within_on_B: '
                      f'{t.f0_on_B.corr(t.within_on_B):.2f}; f0_on_B vs f0_on_srcEEG: {t.f0_on_B.corr(t.f0_on_srcEEG):.2f}')
            e8 = d[d.exp == 'E8'].copy()
            e8['gap'] = e8.rank_f0 - e8.rank_within
            print('\nE8 IMAGES: Spearman r of (rank under f0 - rank under within) with image features, pooled:')
            for col in [c for c in e8.columns if c.startswith('prop') or c == 'density']:
                print(f'   {col}: {spearmanr(e8.gap, e8[col], nan_policy="omit").correlation:.3f}   '
                      f'(rank_f0 alone: {spearmanr(e8.rank_f0, e8[col], nan_policy="omit").correlation:.3f})')
            if 'category' in e8:
                cc = e8[e8.category.fillna('') != ''].groupby('category')[['rank_f0', 'rank_within']].mean()
                print(cc.sort_values('rank_f0').round(1).to_string())
            ch = d[d.exp == 'E8choice'].pivot_table(index='model', columns='foil', values='acc', aggfunc='mean')
            print('\n  2-choice accuracy on held-out concepts (chance 50%):')
            print(ch.round(1).to_string())
        f = sorted(glob.glob(os.path.join(out_dir(), f'prior_t*_{mon}.csv')))
        if f:
            d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
            print(f'\nE6 PERSON MAPS ({d.target.nunique()} targets)')
            st = d[d.exp == 'E6stable']
            print(st[['target', 'within_B', 'between_B', 'cos_B_to_mean', 'top1_f0', 'top1_fullmap']].round(3)
                  .to_string(index=False))
            print('   split-half r within a person vs mean r with other people: '
                  f'{st.within_B.mean():.2f} vs {st.between_B.mean():.2f}')
            cp = d[d.exp == 'E6capture'].groupby('r')[['capture', 'capture_rand']].mean()
            print('   share of B\'s map (minus the population mean) inside the top-r subspace of the others\' maps:')
            print(cp.round(3).to_string())
            fw = d[d.exp == 'E6few'].groupby(['variant', 'k', 'target']).top1.mean().groupby(['variant', 'k']).mean()
            print('   few-shot top-1 (k labeled images):')
            print(fw.unstack('k').round(1).to_string())
            base = d[(d.exp == 'E6few')].groupby(['variant', 'k', 'target']).top1.mean()
            for k in FEW_K:
                for v in ('sub4', 'full_M', 'rand4'):
                    try:
                        dm, pv = paired(base[v][k], base['full_I'][k])
                        print(f'   k={k}: {v} - full_I = {dm:+.2f} (p={pv:.3f})')
                    except KeyError:
                        pass
        f = sorted(glob.glob(os.path.join(out_dir(), f'synth_t*_{mon}.csv')))
        if f:
            d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
            print(f'\nSYNTHETIC-SUBJECT PILOT ({d.target.nunique()} targets)')
            ind = d[d.exp == 'synth_indist'].groupby('arm').cos.mean()
            print('   in-distribution recovery (cosine, predicted vs true map on fresh virtual people):',
                  ind.round(3).to_dict())
            s = d[d.exp == 'synth']
            t = s.groupby(['arm', 'U', 'target'])[['top1', 'cos_to_WB']].mean().groupby(['arm', 'U']).mean()
            print('   predicted from U UNLABELED trials of the real target: top-1, cosine to the target\'s full map')
            print(t.round(3).to_string())
            rf = d[d.exp == 'synth_refine'].groupby(['arm', 'k', 'target']).top1.mean()
            print('   k-label refinement, by starting point:')
            print(rf.groupby(['arm', 'k']).mean().unstack('k').round(1).to_string())
            for kk in (25, 200):
                for a in [x for x in rf.index.get_level_values(0).unique() if x != 'I']:
                    dm, pv = paired(rf[a][kk], rf['I'][kk])
                    print(f'   k={kk}: start {a} - start I = {dm:+.2f} (p={pv:.3f})')


# ---------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['run', 'tune', 'maps', 'prior', 'synth', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=list(range(2, 11)))
    p.add_argument('--montages', nargs='+', default=['posterior17'])
    p.add_argument('--which', default='last', choices=['last', 'best'])
    p.add_argument('--ubp_feat_dir', default='', help='UBP image_features dir (low-level properties for E8)')
    a = p.parse_args()
    N.setup(a.config)
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    people = list(range(1, 11))
    if a.cmd == 'report':
        return report(a.montages)
    for mon in a.montages:
        if a.cmd == 'run':
            for B in a.targets:
                if done('interp', B, mon):
                    print(f'[run] {B} {mon}: done')
                    continue
                print(f'[run] target {B} {mon}', flush=True)
                run_target(B, mon, people, a.which, a.ubp_feat_dir)
            continue
        hp = tune_maps(people, mon, a.which)
        if a.cmd == 'tune':
            continue
        maps = person_maps(people, mon, a.which, hp)
        if a.cmd == 'maps':
            continue
        for B in a.targets:
            if done(a.cmd, B, mon):
                print(f'[{a.cmd}] {B} {mon}: done')
                continue
            print(f'[{a.cmd}] target {B} {mon}', flush=True)
            (run_prior if a.cmd == 'prior' else run_synth)(B, mon, people, a.which, hp, maps)


if __name__ == '__main__':
    main()
