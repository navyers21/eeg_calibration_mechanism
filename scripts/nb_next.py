"""Round 2 on NeuroBridge: calibration built on the mechanism found by nb_interp.py.

    python scripts/nb_next.py run    --config C [--targets 2 ... 10] [--montages posterior17]
    python scripts/nb_next.py shared --config C [--targets ...] [--montages ...]   # after `run` for ALL targets
    python scripts/nb_next.py report --config C [--montages posterior17 crown8 occipital8]

What nb_interp found (posterior-17, 9 people): the harmful difference lives in a ~16-dim subspace of f0's output
(patching 16 dims restores 13.9% -> 53.5%), it is mostly a constant offset plus a linear part (noise ~10%), a
label-free layer-1 mean match from 100 unlabeled trials gives +7.3, and mapping a person's activations onto other
people's activations for the same images ("activation matching", AM) gives +22 with 2,000 images.

run (per target B; f0 = leave-B-out NeuroBridge, epoch 50):
  N1  label-free offset. Layer-1 mean match (= subtract B's average evoked response, add the population's, within
      layer 1's row space) from U = 5 ... all unlabeled trials (5 draws for small U), with shrinkage toward 0, stacked
      with an output mean match, and transductive (offset from the 200 unlabeled TEST trials themselves).
  N2  low-rank activation matching. On n = 10 ... 2,000 calibration images (images other people also saw), map B's
      activations at h1 / h2 / h3 / z onto the source people's mean: offset only, offset + reduced-rank ridge
      residual (rank 2 ... 64), full residual ridge. Each with and without the label-free layer-1 offset first (LF).
  N3  what the calibration should match. Rank-16 output edit at z fitted with (a) AM, (b) NeuroBridge's image loss
      (CLIP-style, f0's gallery), (c) AM init then both losses. n = 25 / 100 / 500, all after LF.
  N4  deployment realism. f0, LF, LF+AM (z, rank 16, n = 200), LF+AM (h2, full, n = 2,000), within decoder, scored
      on test trials averaged over R = 1 / 5 / 20 / 80 repetitions (the Crown will not give 80).
  saves shift_tNN_<mon>.npz (output-space quantities for `shared`).
shared (per target B, needs every target's `run`):
  S1  is the harmful shift shared across people? f0 models live in different output spaces, so each other target's
      space is aligned to B's by orthogonal Procrustes on the image-embedding gallery (same 16,540 images; alignment
      R^2 on held-out images reported). Then: cosine between people's offsets, overlap of their 16-dim difference
      subspaces (chance = 16/512).
  S2  zero-shot population correction: add the other people's mean (aligned) offset to B's output, no B data at all.
  S3  population prior for few-shot AM: fit B's correction inside the subspace shared by the others (rank 8/16/32),
      with B's own offset or the population offset, vs B's own reduced-rank fit, n = 10 ... 200.
  S4  synthetic subjects at the level the mechanism points to (output space, pilot 2). Virtual person = a source
      person's f0 output with an observed (aligned) or sampled population correction inverted:
      z_v = z_s - c - ((z_s - m) U) A U^T. A calibrator learns (c, A) from UNLABELED outputs (mean + covariance in
      the shared subspace) on virtual people, then predicts B's correction from B's unlabeled outputs. Arms: observed
      corrections (obs), Gaussian samples around them (sample), random corrections of matched size (generic), the 8
      real people only (real8); vs LF output mean match alone.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
from eegalign import nb as N  # noqa: E402
import nb_interp as I  # noqa: E402

STAGES = I.STAGES
N2_N = [10, 25, 50, 100, 200, 500, 1000, 2000]
RANKS = [2, 4, 8, 16, 32, 64]
U1 = [5, 10, 25, 50, 100, 200, 500, 0]
REPS = [1, 5, 20, 80]
KAPPA = 25


def dev():
    return N.ENV['device']


def ndraw(n):
    return 5 if n <= 50 else (3 if n <= 200 else 1)


# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def acts2(m, X, d1=None, bs=1024):
    """Stage activations with an optional additive shift at h1."""
    out = {s: [] for s in STAGES}
    for i in range(0, len(X), bs):
        x = torch.as_tensor(X[i:i + bs], device=dev()).float()
        h1 = m.encoder.model[0](x.reshape(len(x), -1))
        if d1 is not None:
            h1 = h1 + d1
        h2 = m.encoder.model[1](h1)
        h3 = m.encoder.model[2](h2)
        for s, v in zip(STAGES, (h1, h2, h3, m.eeg_projector(h3))):
            out[s].append(v)
    return {s: torch.cat(v) for s, v in out.items()}


def ridge_svd(X, Y, rel=torch.logspace(-4, 2, 13)):
    """Ridge X -> Y (both centered inside), alpha by generalized cross-validation. Returns (B, x mean, y mean)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ Yc
    s2 = S ** 2
    n = len(X)
    best, best_a = None, None
    for a in (rel.to(X.device) * s2.mean()):
        f = s2 / (s2 + a)
        df = float(f.sum())
        if df >= n - 1:
            continue
        resid = Yc - U @ (f[:, None] * UtY)
        g = float((resid ** 2).sum()) / (1 - df / n) ** 2
        if best is None or g < best:
            best, best_a = g, a
    if best_a is None:
        best_a = rel[-1].to(X.device) * s2.mean()
    B = Vh.T @ ((S / (s2 + best_a))[:, None] * UtY)
    return B, xm, ym


def am_fit(X, Y, method):
    """Activation matching B -> sources. method: 'off' | 'rr<r>' | 'full'. Fitted on the residual Y - X, so small n
    shrinks toward 'offset only' (identity map)."""
    R = Y - X
    if method == 'off':
        c = R.mean(0)
        return lambda Z: Z + c
    B, xm, rm = ridge_svd(X, R)
    if method.startswith('rr'):
        r = int(method[2:])
        Q = torch.linalg.svd((X - xm) @ B, full_matrices=False).Vh[:r].T
        B = B @ Q @ Q.T
    return lambda Z: Z + rm + (Z - xm) @ B


def am_family(X, Y):
    """All AM variants from one ridge fit: offset, reduced rank (RANKS), full."""
    R = Y - X
    c = R.mean(0)
    B, xm, rm = ridge_svd(X, R)
    Qall = torch.linalg.svd((X - xm) @ B, full_matrices=False).Vh
    fns = {'off': (lambda Z: Z + c), 'full': (lambda Z: Z + rm + (Z - xm) @ B)}
    for r in RANKS:
        Q = Qall[:r].T
        fns[f'rr{r}'] = (lambda Br: (lambda Z: Z + rm + (Z - xm) @ Br))(B @ Q @ Q.T)
    return fns


def top1(z, G):
    return I.top1(z, G)


# ---------------------------------------------------------------------------------------------------------------
def n1_labelfree(c, src_raw):
    rows = []
    f0, B = c.f0, c.B
    A_src = acts2(f0, src_raw)
    mS = {s: A_src[s].mean(0) for s in STAGES}
    A_t = acts2(f0, c.test[B])
    rows.append(dict(exp='N1', method='f0', U=0, draw=0, top1=top1(A_t['z'], c.G)))
    for U in U1:
        for draw in range(5 if 0 < U <= 100 else 1):
            sel = c.pool if U == 0 else np.random.RandomState(600 + draw).choice(c.pool, U, replace=False)
            xb = c.raw[B][sel]
            mB1 = acts2(f0, xb)['h1'].mean(0)
            d1 = mS['h1'] - mB1
            with torch.no_grad():
                rows.append(dict(exp='N1', method='h1', U=U, draw=draw,
                                 top1=top1(I.forward_from(f0, 'h1', A_t['h1'] + d1), c.G)))
                w = len(sel) / (len(sel) + KAPPA)
                rows.append(dict(exp='N1', method='h1_shrink', U=U, draw=draw,
                                 top1=top1(I.forward_from(f0, 'h1', A_t['h1'] + w * d1), c.G)))
                Ab = acts2(f0, xb, d1)
                dz = mS['z'] - Ab['z'].mean(0)
                zt = acts2(f0, c.test[B], d1)['z']
                rows.append(dict(exp='N1', method='h1+z', U=U, draw=draw, top1=top1(zt + dz, c.G)))
                rows.append(dict(exp='N1', method='z', U=U, draw=draw,
                                 top1=top1(A_t['z'] + mS['z'] - acts2(f0, xb)['z'].mean(0), c.G)))
    with torch.no_grad():                                   # transductive: the 200 unlabeled test trials
        d1 = mS['h1'] - A_t['h1'].mean(0)
        rows.append(dict(exp='N1', method='transductive_h1', U=200, draw=0,
                         top1=top1(I.forward_from(f0, 'h1', A_t['h1'] + d1), c.G)))
        zt = acts2(f0, c.test[B], d1)['z']
        rows.append(dict(exp='N1', method='transductive_h1+z', U=200, draw=0, top1=top1(zt - zt.mean(0) + mS['z'], c.G)))
    return rows, mS


def n2_lowrank(c, d1_all):
    rows, keep = [], {}
    f0, B = c.f0, c.B
    A_src = [I.acts(f0, c.cal[p]) for p in c.src]
    tgt = {s: torch.stack([a[s] for a in A_src]).mean(0) for s in STAGES}   # independent of B's offset
    del A_src
    for lf in (0, 1):
        d1 = d1_all if lf else None
        Ab = acts2(f0, c.cal[B], d1)
        At = acts2(f0, c.test[B], d1)
        with torch.no_grad():
            rows.append(dict(exp='N2', stage='-', lf=lf, method='none', n=0, draw=0, top1=top1(At['z'], c.G)))
        for s in STAGES:
            for n in N2_N:
                for draw in range(ndraw(n)):
                    ix = torch.as_tensor(np.random.RandomState(700 + draw).permutation(len(c.cal_idx))[:n], device=dev())
                    X, Y = Ab[s][ix], tgt[s][ix]
                    fam = am_family(X, Y)
                    for meth, fn in fam.items():
                        with torch.no_grad():
                            rows.append(dict(exp='N2', stage=s, lf=lf, method=meth, n=n, draw=draw,
                                             top1=top1(I.forward_from(f0, s, fn(At[s])), c.G)))
                        if draw == 0 and lf == 1 and ((s, meth, n) in (('z', 'rr16', 200), ('h2', 'full', 2000),
                                                                     ('z', 'rr16', 25), ('h2', 'rr16', 200))):
                            keep[(s, meth, n)] = fn
            print(f'     N2 lf={lf} stage {s} done', flush=True)
    return rows, keep, tgt


def n3_target(c, d1_all, tgt, steps=300, rank=16):
    rows = []
    f0, B = c.f0, c.B
    zc = acts2(f0, c.cal[B], d1_all)['z']
    zt = acts2(f0, c.test[B], d1_all)['z']
    Ytz = tgt['z']
    img = torch.as_tensor(c.cal_idx, device=dev())
    scale = f0.scale()
    for n in (25, 100, 500):
        for draw in range(3 if n <= 100 else 1):
            ix = torch.as_tensor(np.random.RandomState(700 + draw).permutation(len(c.cal_idx))[:n], device=dev())
            X, Y, im = zc[ix], Ytz[ix], img[ix]
            fam = am_fit(X, Y, f'rr{rank}')
            with torch.no_grad():
                rows.append(dict(exp='N3', variant='AM', n=n, draw=draw, top1=top1(fam(zt), c.G)))
            # factorize the AM edit for the init of (c)
            R = Y - X
            Bm, xm, rm = ridge_svd(X, R)
            Q = torch.linalg.svd((X - xm) @ Bm, full_matrices=False).Vh[:rank].T
            Br = Bm @ Q @ Q.T
            Uu, Ss, Vv = torch.linalg.svd(Br, full_matrices=False)
            for variant in ('CLIP', 'AM+CLIP'):
                torch.manual_seed(draw)
                if variant == 'CLIP':
                    cc = torch.zeros(X.shape[1], device=dev())
                    Uf = (0.01 * torch.randn(X.shape[1], rank, device=dev()))
                    Vf = torch.zeros(X.shape[1], rank, device=dev())
                    m0 = X.mean(0)
                else:
                    cc = rm.clone()                                  # z' = z + cc + (z - m0) U V^T = AM edit
                    Uf = (Uu[:, :rank] * Ss[:rank].sqrt()).clone()
                    Vf = (Vv[:rank].T * Ss[:rank].sqrt()).clone()
                    m0 = xm.clone()
                ps = [p.requires_grad_(True) for p in (cc, Uf, Vf)]
                opt = torch.optim.Adam(ps, lr=1e-3)
                for _ in range(steps):
                    bi = torch.randint(n, (min(n, 256),), device=dev())
                    z2 = X[bi] + cc + (X[bi] - m0) @ Uf @ Vf.T
                    gal = torch.cat([c.Gtr[im[bi]], c.Gtr[torch.randint(len(c.Gtr), (1024,), device=dev())]])
                    loss = F.cross_entropy(scale * z2 @ gal.T, torch.arange(len(bi), device=dev()))
                    if variant == 'AM+CLIP':
                        loss = loss + ((z2 - Y[bi]) ** 2).sum(1).mean() / (Y.var(0).sum() + 1e-6)
                    loss = loss + 1e-4 * sum((p ** 2).sum() for p in ps)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    rows.append(dict(exp='N3', variant=variant, n=n, draw=draw,
                                     top1=top1(zt + cc + (zt - m0) @ Uf @ Vf.T, c.G)))
    return rows


def n4_reps(c, d1_all, keep):
    rows = []
    f0, B = c.f0, c.B
    tt = N.subject_data(B, c.mon, keys=('test_trials',))['test_trials']
    for R in REPS:
        for draw in range(3 if R < 80 else 1):
            rng = np.random.RandomState(800 + draw)
            X = np.stack([tt[i, rng.choice(tt.shape[1], R, replace=False)].astype(np.float32).mean(0)
                          for i in range(len(tt))])
            A0 = acts2(f0, X)
            A1 = acts2(f0, X, d1_all)
            with torch.no_grad():
                zw = c.within(torch.as_tensor(X, device=dev()))
                res = {'f0': top1(A0['z'], c.G), 'LF_h1': top1(A1['z'], c.G), 'within': top1(zw, c.Gw)}
                for (s, meth, n), fn in keep.items():
                    res[f'LF+AM_{s}_{meth}_n{n}'] = top1(I.forward_from(f0, s, fn(A1[s])), c.G)
            for k, v in res.items():
                rows.append(dict(exp='N4', method=k, R=R, draw=draw, top1=v))
    return rows


def run_target(B, mon, people, which):
    t0 = time.time()
    c = I.Ctx(B, mon, people, which)
    src_raw = np.concatenate([c.raw[p] for p in c.src]).astype(np.float32)
    rows, mS = n1_labelfree(c, src_raw)
    print(f'   N1 done ({time.time() - t0:.0f}s)', flush=True)
    d1_all = mS['h1'] - acts2(c.f0, c.raw[B][c.pool])['h1'].mean(0)
    r2, keep, tgt = n2_lowrank(c, d1_all)
    rows += r2
    print(f'   N2 done ({time.time() - t0:.0f}s)', flush=True)
    rows += n3_target(c, d1_all, tgt)
    print(f'   N3 done ({time.time() - t0:.0f}s)', flush=True)
    rows += n4_reps(c, d1_all, keep)
    print(f'   N4 done ({time.time() - t0:.0f}s)', flush=True)
    # output-space quantities for `shared`
    f0 = c.f0
    A_held = {p: I.acts(f0, c.held[p])['z'] for p in people}
    D = A_held[B] - torch.stack([A_held[p] for p in c.src]).mean(0)
    zcal = {p: I.acts(f0, c.cal[p])['z'] for p in people}
    np.savez_compressed(os.path.join(I.out_dir(), f'shift_t{B:02d}_{mon}.npz'),
                        Gtr=c.Gtr.cpu().numpy().astype(np.float16), G=c.G.cpu().numpy(),
                        D=D.cpu().numpy(), zB_cal=zcal[B].cpu().numpy(),
                        zS_cal=torch.stack([zcal[p] for p in c.src]).cpu().numpy().astype(np.float16),
                        zB_test=I.acts(f0, c.test[B])['z'].cpu().numpy(), src=np.array(c.src),
                        cal_idx=c.cal_idx, labels=c.labels, pool=c.pool)
    for r in rows:
        r.update(target=B, montage=mon)
    I.save(rows, 'next', B, mon)
    print(f'  target {B} {mon}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
def procrustes(Ga, Gb, fit_idx, ev_idx):
    """Orthogonal R with Ga R ~ Gb (rows = same images). Returns R and held-out R^2."""
    A, Bm = Ga[fit_idx], Gb[fit_idx]
    U, _, Vh = torch.linalg.svd(A.T @ Bm)
    R = U @ Vh
    E = Ga[ev_idx] @ R - Gb[ev_idx]
    r2 = 1 - float((E ** 2).sum()) / float(((Gb[ev_idx] - Gb[ev_idx].mean(0)) ** 2).sum())
    return R, r2


def topk_basis(D, k):
    return torch.linalg.svd(D, full_matrices=False).Vh[:k].T


def sub_fit(X, Y, U):
    """Correction restricted to subspace U: z' = z + c + ((z - m) U) A U^T, c = mean residual."""
    Rr = Y - X
    A, xm, rm = ridge_svd((X) @ U, Rr @ U)
    m = X.mean(0)
    c = Rr.mean(0)
    return c, A, m


def apply_sub(Z, c, A, m, U):
    return Z + c + ((Z - m) @ U) @ A @ U.T


def run_shared(B, mon, targets, r_list=(8, 16, 32)):
    t0 = time.time()
    path = lambda t: os.path.join(I.out_dir(), f'shift_t{t:02d}_{mon}.npz')
    others = [t for t in targets if t != B and os.path.exists(path(t))]
    L = {t: dict(np.load(path(t))) for t in [B] + others}
    T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev())
    lab, pool = L[B]['labels'], L[B]['pool']
    held = np.setdiff1d(np.arange(len(lab)), pool)
    fit_idx = torch.as_tensor(np.random.RandomState(0).choice(pool, min(6000, len(pool)), replace=False), device=dev())
    ev_idx = torch.as_tensor(held, device=dev())
    GB, G = T(L[B]['Gtr']), T(L[B]['G'])
    DB = T(L[B]['D'])
    muB = DB.mean(0)
    VB = topk_basis(DB, 16)
    zt, zc, zS = T(L[B]['zB_test']), T(L[B]['zB_cal']), T(L[B]['zS_cal'])
    zSm = zS.mean(0)
    rows = [dict(exp='S0', method='f0', top1=top1(zt, G))]
    R, Dal, mus = {}, {}, {}
    for t in others:
        R[t], r2 = procrustes(T(L[t]['Gtr']), GB, fit_idx, ev_idx)
        Dal[t] = T(L[t]['D']) @ R[t]
        mus[t] = Dal[t].mean(0)
        Vt = topk_basis(Dal[t], 16)
        rows.append(dict(exp='S1', other=t, align_r2=r2,
                         cos_offset=float(F.cosine_similarity(mus[t], muB, dim=0)),
                         overlap16=float((VB.T @ Vt).pow(2).sum() / 16), chance16=16 / DB.shape[1]))
    # random-direction baselines for the offset cosine
    g = torch.Generator(device='cpu').manual_seed(0)
    rnd = [float(F.cosine_similarity(torch.randn(DB.shape[1], generator=g).to(dev()), muB, dim=0)) for _ in range(100)]
    rows.append(dict(exp='S1rand', cos_offset_abs_mean=float(np.mean(np.abs(rnd)))))
    mpop = torch.stack(list(mus.values())).mean(0)
    for s in (0.5, 1.0):
        rows.append(dict(exp='S2', method=f'pop_offset_x{s}', top1=top1(zt + s * mpop, G)))
    rows.append(dict(exp='S2', method='own_offset_heldout(oracle)', top1=top1(zt + muB, G)))
    rows.append(dict(exp='S2', method='LF_output_mean(cal, unlabeled)', top1=top1(zt - zc.mean(0) + zSm.mean(0), G)))
    # S3 population-subspace prior for few-shot correction at z
    Dpop = torch.cat([Dal[t] for t in others])
    for n in (10, 25, 50, 100, 200):
        for draw in range(ndraw(n)):
            ix = torch.as_tensor(np.random.RandomState(900 + draw).permutation(len(zc))[:n], device=dev())
            X, Y = zc[ix], zSm[ix]
            for meth in ('off', 'rr16'):
                fn = am_fit(X, Y, meth)
                rows.append(dict(exp='S3', method=f'own_{meth}', n=n, draw=draw, top1=top1(fn(zt), G)))
            for r in r_list:
                U = topk_basis(Dpop, r)
                cc, A, m = sub_fit(X, Y, U)
                rows.append(dict(exp='S3', method=f'pop_sub{r}', n=n, draw=draw, top1=top1(apply_sub(zt, cc, A, m, U), G)))
                rows.append(dict(exp='S3', method=f'pop_sub{r}+popoff', n=n, draw=draw,
                                 top1=top1(apply_sub(zt, mpop, A, m, U), G)))
                Uown = topk_basis(Y - X - (Y - X).mean(0), min(r, n - 1)) if n > 2 else U
                cc, A, m = sub_fit(X, Y, Uown)
                rows.append(dict(exp='S3', method=f'own_sub{r}', n=n, draw=draw, top1=top1(apply_sub(zt, cc, A, m, Uown), G)))
    # S4 synthetic subjects in output space (label-free calibrator)
    rows += s4_synth(B, others, L, R, zt, zc, zS, G, Dpop, mus, T)
    for r in rows:
        r.update(target=B, montage=mon)
    I.save(rows, 'shared', B, mon)
    print(f'  shared {B} {mon}: {(time.time() - t0) / 60:.1f} min', flush=True)


def s4_synth(B, others, L, R, zt, zc, zS, G, Dpop, mus, T, r=16, n_syn=3000):
    rows = []
    rng = np.random.RandomState(0)
    U = topk_basis(Dpop, r)
    d = zt.shape[1]
    # observed corrections, each other target in B's (aligned) output space
    obs = []
    for t in others:
        Xt = T(L[t]['zB_cal']) @ R[t]
        Yt = T(L[t]['zS_cal']).mean(0) @ R[t]
        cc, A, m = sub_fit(Xt, Yt, U)
        obs.append((cc, A))
    V = torch.stack([torch.cat([cc, A.ravel()]) for cc, A in obs])
    vm = V.mean(0)
    Uv, Sv, Vh = torch.linalg.svd(V - vm, full_matrices=False)
    sdv = Sv / max(len(V) - 1, 1) ** 0.5
    cn = float(np.mean([float(cc.norm()) for cc, _ in obs]))
    an = float(np.mean([float(A.norm()) for _, A in obs]))

    def draw_corr(kind):
        if kind == 'obs':
            return obs[rng.randint(len(obs))]
        if kind == 'sample':
            zz = torch.as_tensor(rng.randn(len(sdv)), device=dev()).float()
            v = vm + (zz * sdv) @ Vh
            return v[:d], v[d:].reshape(r, r)
        cc = torch.as_tensor(rng.randn(d), device=dev()).float()
        A = torch.as_tensor(rng.randn(r, r), device=dev()).float()
        return cc / cc.norm() * cn, A / A.norm() * an

    def feats(Z):
        P = (Z - Z.mean(0)) @ U
        C = P.T @ P / max(len(Z) - 1, 1)
        iu = torch.triu_indices(r, r)
        return torch.cat([Z.mean(0), torch.log(torch.diagonal(C).clamp_min(1e-8)), C[iu[0], iu[1]]])

    def virtual(kind, n):
        Xf, Yf = [], []
        for _ in range(n):
            s = rng.randint(zS.shape[0])
            cc, A = draw_corr(kind)
            Zs = zS[s][torch.as_tensor(rng.choice(zS.shape[1], int(rng.choice([50, 200, 1000])), replace=False),
                                       device=dev())]
            m = Zs.mean(0)
            Zv = Zs - cc - ((Zs - m) @ U) @ A @ U.T            # first-order inverse of the correction
            Xf.append(feats(Zv))
            Yf.append(torch.cat([cc, A.ravel()]))
        return torch.stack(Xf), torch.stack(Yf)

    sets = {k: virtual(k, n_syn) for k in ('obs', 'sample', 'generic')}
    Xr, Yr = [], []
    for (cc, A), t in zip(obs, others):
        Zt = T(L[t]['zB_cal']) @ R[t]
        for U_ in (50, 200, 1000):
            for _ in range(10):
                Xr.append(feats(Zt[torch.as_tensor(rng.choice(len(Zt), U_, replace=False), device=dev())]))
                Yr.append(torch.cat([cc, A.ravel()]))
    sets['real8'] = (torch.stack(Xr), torch.stack(Yr))
    preds = {}
    for k, (X, Y) in sets.items():
        Bm, xm, ym = ridge_svd(X, Y)
        preds[k] = (lambda Bm, xm, ym: (lambda Z: ym + (Z - xm) @ Bm))(Bm, xm, ym)
    for k in ('obs', 'sample', 'generic'):
        Xv, Yv = virtual(k, 300)
        rows.append(dict(exp='S4indist', arm=k, cos=float(F.cosine_similarity(preds[k](Xv), Yv, dim=1).mean())))
    zSm = zS.mean(0)
    for U_ in (50, 200, 1000):
        for draw in range(3):
            Zb = zc[torch.as_tensor(np.random.RandomState(950 + draw).choice(len(zc), U_, replace=False), device=dev())]
            rows.append(dict(exp='S4', arm='LF_output_mean', U=U_, draw=draw,
                             top1=top1(zt - Zb.mean(0) + zSm.mean(0), G)))
            f = feats(Zb)[None]
            for k, pr in preds.items():
                y = pr(f)[0]
                cc, A = y[:d], y[d:].reshape(r, r)
                m = Zb.mean(0)
                rows.append(dict(exp='S4', arm=k, U=U_, draw=draw, top1=top1(apply_sub(zt, cc, A, m, U), G)))
                rows.append(dict(exp='S4', arm=f'{k}+LFoffset', U=U_, draw=draw,
                                 top1=top1(apply_sub(zt, zSm.mean(0) - m, A, m, U), G)))
    return rows


# ---------------------------------------------------------------------------------------------------------------
def report(mons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pd.set_option('display.width', 220)
    summary = []
    for mon in mons:
        f = sorted(glob.glob(os.path.join(I.out_dir(), f'next_t*_{mon}.csv')))
        if not f:
            continue
        d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        print(f'\n{"=" * 100}\n{mon}: {d.target.nunique()} targets\n{"=" * 100}')
        n1 = d[d.exp == 'N1']
        f0 = n1[n1.method == 'f0'].set_index('target').top1
        print(f'f0 {f0.mean():.1f}%')
        t = n1[n1.method != 'f0'].groupby(['method', 'U', 'target']).top1.mean().groupby(['method', 'U']).mean()
        print('\nN1 LABEL-FREE OFFSET (U unlabeled trials; 0 = all pool):')
        print(t.unstack('U').round(1).to_string())
        n2 = d[d.exp == 'N2']
        n2 = n2.assign(lf=n2.lf.astype(int))
        per = n2[n2.stage != '-'].groupby(['lf', 'stage', 'method', 'n', 'target']).top1.mean()
        mean = per.groupby(['lf', 'stage', 'method', 'n']).mean()
        for lf in (0, 1):
            print(f'\nN2 ACTIVATION MATCHING ({"after" if lf else "without"} the label-free layer-1 offset), top-1 by n:')
            print(mean[lf].unstack('n').round(1).to_string())
        best = mean.groupby(['n']).idxmax()
        print('\n   best configuration per n:', {int(k): (v[:3], round(float(mean[v]), 1)) for k, v in best.items()})
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for i, lf in enumerate((0, 1)):
            for s, col in zip(STAGES, ('C0', 'C1', 'C2', 'C3')):
                for meth, ls in (('off', ':'), ('rr16', '-'), ('full', '--')):
                    h = mean[lf][s][meth]
                    ax[i].plot(h.index, h.values, ls, color=col, label=f'{s} {meth}')
            ax[i].axhline(f0.mean(), color='k', lw=.8)
            ax[i].set(xscale='log', xlabel='calibration images', ylabel='top-1 %',
                      title=f'activation matching {"after LF offset" if lf else "alone"}')
        ax[1].legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(I.out_dir(), f'N2_curves_{mon}.png'), dpi=120)
        plt.close()
        n3 = d[d.exp == 'N3'].groupby(['variant', 'n', 'target']).top1.mean().groupby(['variant', 'n']).mean()
        print('\nN3 WHAT TO MATCH (rank-16 output edit after LF): AM vs image loss vs both')
        print(n3.unstack('n').round(1).to_string())
        n4 = d[d.exp == 'N4'].groupby(['method', 'R', 'target']).top1.mean().groupby(['method', 'R']).mean()
        print('\nN4 TEST REPETITIONS AVERAGED (R), top-1:')
        print(n4.unstack('R').round(1).to_string())
        g = per.reset_index()
        lfh1 = n1[(n1.method == 'h1') & (n1.U == 100)].groupby('target').top1.mean()
        row = dict(montage=mon, f0=f0.mean(), LF_h1_U100=lfh1.mean())
        for n in (25, 200, 2000):
            gg = g[(g.n == n)].groupby(['lf', 'stage', 'method']).top1.mean()
            row[f'best_AM_n{n}'] = gg.max()
            row[f'best_AM_n{n}_cfg'] = str(gg.idxmax())
        summary.append(row)
        f = sorted(glob.glob(os.path.join(I.out_dir(), f'shared_t*_{mon}.csv')))
        if f:
            s = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
            s1 = s[s.exp == 'S1']
            rb = s[s.exp == 'S1rand'].cos_offset_abs_mean.mean()
            print(f'\nS1 SHARED STRUCTURE (other people aligned into each target\'s output space by Procrustes on the '
                  f'gallery; alignment R^2 {s1.align_r2.mean():.3f}):')
            print(f'   offset cosine between people {s1.cos_offset.mean():.3f} (random direction |cos| {rb:.3f}); '
                  f'16-dim subspace overlap {s1.overlap16.mean():.3f} (chance {s1.chance16.mean():.3f})')
            s2 = s[s.exp.isin(['S0', 'S2'])].groupby(['method', 'target']).top1.mean().groupby('method').mean()
            print('\nS2 ZERO-SHOT POPULATION OFFSET (no data from the target):')
            print(s2.round(1).to_string())
            s3 = s[s.exp == 'S3'].groupby(['method', 'n', 'target']).top1.mean().groupby(['method', 'n']).mean()
            print('\nS3 FEW-SHOT CORRECTION AT THE OUTPUT: own fit vs population-subspace prior')
            print(s3.unstack('n').round(1).to_string())
            ind = s[s.exp == 'S4indist'].groupby('arm').cos.mean()
            print('\nS4 SYNTHETIC SUBJECTS IN OUTPUT SPACE; in-distribution recovery (cos):', ind.round(3).to_dict())
            s4 = s[s.exp == 'S4'].groupby(['arm', 'U', 'target']).top1.mean().groupby(['arm', 'U']).mean()
            print(s4.unstack('U').round(1).to_string())
    if summary:
        print('\nACROSS MONTAGES')
        print(pd.DataFrame(summary).round(1).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['run', 'shared', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=list(range(2, 11)))
    p.add_argument('--montages', nargs='+', default=['posterior17'])
    p.add_argument('--which', default='last', choices=['last', 'best'])
    a = p.parse_args()
    N.setup(a.config)
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'report':
        return report(a.montages)
    people = list(range(1, 11))
    for mon in a.montages:
        for B in a.targets:
            name = 'next' if a.cmd == 'run' else 'shared'
            if I.done(name, B, mon):
                print(f'[{a.cmd}] {B} {mon}: done')
                continue
            print(f'[{a.cmd}] target {B} {mon}', flush=True)
            if a.cmd == 'run':
                run_target(B, mon, people, a.which)
            else:
                run_shared(B, mon, a.targets)


if __name__ == '__main__':
    main()
