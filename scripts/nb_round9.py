"""Round 9: Kazuma's four next steps (Nov 20 team update), on the calibration pipeline.

    python scripts/nb_round9.py figures  --config C            # 1. presentation figures, draw sensitivity, claims table (CPU)
    python scripts/nb_round9.py edit     --config C            # 3. what part of the output edit matters
    python scripts/nb_round9.py budget   --config C            # 4a. equal presentation budgets (100 and 400)
    python scripts/nb_round9.py denoise  --config C            # 4b. linear cross-repeat denoiser + controls
    python scripts/nb_round9.py feats    --config C            # 2. RN50 intermediate image features (needs the image set)
    python scripts/nb_round9.py profile  --config C            # 2. GPU time per model, before committing to the grid
    python scripts/nb_round9.py tune     --config C            # 2. learning rate per encoder x target, person-1 fold only
    python scripts/nb_round9.py encoders --config C [--seeds 2025] [--targets 2 ... 10]
    python scripts/nb_round9.py report   --config C

Ground rules (from the update): every cross-person model is trained without the target person; every basis, predictor
and denoiser is fitted on source people (or on the target's own calibration trials where labeled calibration is the
point); settings are fixed in code or chosen on person 1 only and then frozen. People 2-10 were also used in rounds
3-8, so everything here is DEVELOPMENT REPLICATION, not an untouched confirmation cohort. All on the 63-channel
whitened release: "posterior-17" is a channel subset of whitened data, not a 17-electrode headset.

1. FIGURES. Per-person calibration curves (thin lines) with the cohort mean (thick), per-person contrast dot plots for
   every headline comparison, calibration-draw sensitivity (round 7's 5 draws at k = 500: share of person x draw
   cells where calibration helps, spread across draws), and a claims / evidence / limitation table, recomputed from
   the saved per-person results. Calibration and erasure findings are kept in separate panels and rows.

3. OUTPUT EDIT. Seed-2026 repaug decoder, earlier-day calibration, k = 100 / 200 / 500 / 2000 (round 6 draws), label-
   free layer-1 offset first. Edits of the offset-corrected outputs toward the measured population responses:
     shift      add the mean difference                 scale    shift + one global scale
     rot        shift + rotation (Procrustes, no scale)  rigid    shift + rotation + scale (rounds 6-7)
     pca_r      shift + rank-r linear edit inside the top-r PCs of the SOURCE population responses (r = 4, 16, 64)
     rand_r     same in a random orthonormal basis of rank r (3 bases; rank-matched control)
     randvar_r  same in a random rank-r basis inside the top-64 PCs (closer in captured variance; r = 4, 16)
     full       unrestricted ridge linear map
   Tuning-matched: every method with a strength parameter picks it from the SAME grid by 5-fold cross-validation over
   the calibration images (held-out cosine to the population responses). Evaluated with the population gallery, each
   query scored on its own (cos) and batch CSLS. Consistency: cosine between fitted shift vectors across draws (same
   person) and across people (same k), and overlap of the top-8 directions of each rigid rotation (R - I) across draws
   and across people, against the random-subspace baseline (8 / 512).

4a. BUDGET. Same presentation budget, different split: 100 presentations = 25 x 4 / 50 x 2 / 100 x 1, and 400 = 100 x 4
   / 200 x 2 / 400 x 1 (round 8 covered 1,000). x 4 needs images with 2 session-3 + 2 session-1/2 presentations (THINGS
   never shows an image 4 times in one session), so x 4 spans days. Methods: shift, rigid; 5 draws at 100, 3 at 400.

4b. DENOISE. A linear map from ONE presentation (channels x time, flattened) to the mean of the OTHER presentations of
   the same image (disjoint repeat views), ridge, fitted on the 9 source people (1,500 pool images each). Strength
   chosen on 2 held-out source people. Controls, selected the same way: identity (no denoiser), PCA projection onto the
   top-k components of source single presentations, and shuffled pairs (target = repeats of a DIFFERENT image, which
   can only learn shrinkage). Also a target-calibrated version fitted on the person's own 500 x 2 calibration
   (presentation 1 -> 2 and 2 -> 1). Everything that enters the decoder is denoised (calibration trials and queries);
   then the zero-label pipeline and rigid k = 500, queries at 1 / 4 / 20 presentations and the benchmark. A ridge
   denoiser shrinks its output toward the mean, and the decoder was trained on raw-scale EEG, so every denoiser and
   control is VARIANCE-MATCHED (each output dimension rescaled to the raw input's spread on the fitting data);
   ridge_raw (no rescaling) is kept to show the size of that effect.

2. ENCODERS x VISUAL TARGETS. 3 x 2: targets = degraded-final (NeuroBridge's fused augmented CLIP RN50, current),
   clean-final (CLIP RN50 on clean images), clean-intermediate (RN50 layer-3 output, global-average-pooled, 1024-d,
   same width as the final features); encoders = EEGProject (current) and a compact CNN (temporal conv, average pool,
   spatial conv; NICE / ATM style). Same splits, same repaug recipe (50 epochs, batch 1024), same seeds; learning rate
   picked per cell from {1e-4, 3e-4, 1e-3} on the person-1 fold (source validation concepts), then frozen. Each model is
   evaluated with a pipeline every encoder supports: plain image gallery; zero-label = OUTPUT mean offset + population
   gallery (predicted from that cell's own target features, source outputs only); rigid at k = 500. `profile` times one
   epoch of each cell first; the full grid is 6 cells x 10 folds per seed.
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
from eegalign import nb as N  # noqa: E402
import nb_interp as I  # noqa: E402
import nb_next as X  # noqa: E402
import nb_round3 as R  # noqa: E402
import nb_round4 as R4m  # noqa: E402
import nb_round5 as R5m  # noqa: E402
import nb_round6 as R6m  # noqa: E402
import nb_gallery as GL  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
QK = ['s4R1', 's4R4', 's4R20', 'bench']
CS = 3
SEED = 2026
K_EDIT = [100, 200, 500, 2000]
REL = [0.01, 0.1, 1.0, 10.0, 100.0]
RANKS = [4, 16, 64]
BUDGETS = {100: [(25, 4), (50, 2), (100, 1)], 400: [(100, 4), (200, 2), (400, 1)]}
ENCS = ['eegproject', 'cnn']
TGTS = ['aug', 'clean', 'mid']
LRS = [1e-4, 3e-4, 1e-3]


def dev():
    return N.ENV['device']


def odir(sub=''):
    d = os.path.join(N.ENV['results_dir'], 'nb_round9', sub)
    os.makedirs(d, exist_ok=True)
    return d


def nrm(x):
    return F.normalize(x.float(), dim=-1)


def ctx_for(B, mon, seed=SEED):
    R5m.SEED = seed
    if not os.path.exists(N.ckpt_path('repaug', mon, B, seed, 'last')):
        return None
    return R6m.Ctx('repaug', B, mon)


def score_rows(base, Zq, cells, rows):
    """cells: list of (method, map or None, gallery). Zq: {qk: [z, ...]}."""
    for qk in QK:
        for qi, z0 in enumerate(Zq[qk]):
            for meth, mp, Gt in cells:
                z = z0 if mp is None else mp(z0)
                for rule in ('cos', 'csls_batch'):
                    rows.append(dict(base, method=meth, rule=rule, query=qk, qdraw=qi, top1=R6m.top1(R6m.smat(z, Gt, rule))))


def pstat(a, b):
    """Paired stats over people 2-10 (dict), quiet version of R4m.pstats."""
    from scipy import stats
    idx = sorted(set(a.dropna().index) & set(b.dropna().index) & set(TEST))
    d = (a.reindex(idx) - b.reindex(idx)).values.astype(float)
    if len(d) == 0:
        return None
    rng = np.random.RandomState(0)
    boot = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = np.nan
    return dict(mean=d.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), pos=int((d > 0).sum()),
                neg=int((d < 0).sum()), n=len(d), p=p, per=dict(zip(idx, d)))


# ---------------------------------------------------------------------------------------------------------------
# 1. figures, draw sensitivity, claims table
# ---------------------------------------------------------------------------------------------------------------
def _load(pattern):
    f = sorted(glob.glob(os.path.join(N.ENV['results_dir'], pattern)))
    return pd.concat([pd.read_csv(x) for x in f], ignore_index=True) if f else None


def figures(mon='posterior17'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = odir('figures')
    r6 = _load(f'nb_round6/cal_t*_{mon}.csv')
    conf = _load(f'nb_round7/confirm/cal_t*_{mon}.csv')
    combo = _load(f'nb_round7/combo/combo_t*_{mon}.csv')
    gal = _load(f'nb_gallery/gallery/gal_s*_t*_{mon}.csv')
    scr = _load(f'nb_gallery/scratch/scratch_t*_{mon}.csv')
    claims = []

    def per(df, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k] == v).values
        return df[m].groupby('target').top1.mean()

    def claim(name, a, b, evidence, limitation, kind='calibration'):
        s = pstat(a, b)
        if s:
            claims.append(dict(kind=kind, claim=name, mean=round(s['mean'], 2), ci=f'{s["lo"]:+.2f} to {s["hi"]:+.2f}',
                               improved=f'{s["pos"]}/{s["n"]}', wilcoxon_p=round(s['p'], 4), evidence=evidence,
                               limitation=limitation, **{f't{t}': round(v, 3) for t, v in s['per'].items()}))
    # ---- per-person calibration curves
    if r6 is not None:
        d = r6[(r6.decoder == 'repaug') & (r6.cal_session == CS) & (r6['query'] == 's4R4')]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, rule in zip(axes, ('cos', 'csls_batch')):
            for meth, ro, col in (('zero', 'anchor', '#9e9e9e'), ('rigid', 'anchor', '#555555'), ('ft', 'img', '#2b6cb0')):
                p = d[(d.method == meth) & (d.readout == ro) & (d.rule == rule)].groupby(['target', 'k']).top1.mean().unstack('k')
                p = p[p.index.isin(TEST)]
                for _, row in p.iterrows():
                    ax.plot(p.columns, row.values, color=col, lw=0.6, alpha=0.35)
                ax.plot(p.columns, p.mean().values, color=col, lw=2.5, label=f'{meth} (mean of {len(p)})')
            ax.set_xscale('log')
            ax.set_xlabel('calibration images (x 2 presentations)')
            ax.set_title(f's4R4, {"each query scored alone" if rule == "cos" else "batch CSLS"}')
            ax.grid(alpha=0.3)
        axes[0].set_ylabel('top-1 % (chance 0.5)')
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'calibration_people_{mon}.png'), dpi=200)
        plt.close(fig)
    # ---- draw sensitivity (round 7, 5 draws at k = 500)
    sens = []
    if conf is not None:
        c = conf[(conf.decoder == 'repaug') & (conf.cal_session == CS) & (conf.k == 500) & (conf['query'] == 's4R4')]
        for rule in ('cos', 'csls_batch'):
            a = c[(c.method == 'rigid') & (c.readout == 'anchor') & (c.rule == rule)].groupby(['target', 'draw']).top1.mean()
            z = c[(c.method == 'zero') & (c.readout == 'anchor') & (c.rule == rule)].groupby(['target', 'draw']).top1.mean()
            dd = (a - z).dropna()
            dd = dd[dd.index.get_level_values('target').isin(TEST)]
            g = dd.groupby('target')
            sens.append(dict(rule=rule, cells=len(dd), positive=int((dd > 0).sum()), share_positive=round(float((dd > 0).mean()), 3),
                             mean=round(float(dd.mean()), 2), sd_across_draws_mean=round(float(g.std().mean()), 2),
                             worst_draw_mean=round(float(g.min().mean()), 2), people_all_draws_positive=int((g.min() > 0).sum())))
        pd.DataFrame(sens).to_csv(os.path.join(out, 'draw_sensitivity.csv'), index=False)
        k5 = lambda meth, rule, q='s4R4', cs=CS: per(conf[(conf.decoder == 'repaug') & (conf.k == 500)], cal_session=cs, method=meth, readout='anchor', rule=rule, query=q)  # noqa: E731
        claim('500-image calibration (rigid) beats the zero-label pipeline, s4R4, batch CSLS (locked, seed 2026)',
              k5('rigid', 'csls_batch'), k5('zero', 'csls_batch'), 'round 7 confirm, 5 draws', 'same 9 people as development; one dataset')
        claim('same, each query scored alone', k5('rigid', 'cos'), k5('zero', 'cos'), 'round 7 confirm', 'as above')
        claim('same, single-presentation queries (s4R1)', k5('rigid', 'csls_batch', 's4R1'), k5('zero', 'csls_batch', 's4R1'),
              'round 7 confirm', 'small or no gain: the consumer-like single-view case is not solved')
        claim('same-day calibration', k5('rigid', 'csls_batch', cs=4), k5('zero', 'csls_batch', cs=4), 'round 7 confirm', 'as above')
    if combo is not None:
        cb = lambda k, meth, ro, rule='csls_batch': per(combo, k=k, method=meth, readout=ro, rule=rule, query='s4R4')  # noqa: E731
        claim('fine-tuning + rigid beats fine-tuning alone at 4000 images', cb(4000, 'ft+rigid', 'anchor'), cb(4000, 'ft', 'img'),
              'round 7 combo', '8,000 labeled presentations; 1 draw')
        claim('fine-tuning + rigid vs rigid alone at 500 images', cb(500, 'ft+rigid', 'anchor'), cb(500, 'rigid', 'anchor'),
              'round 7 combo', 'no gain: fine-tuning does not help at small budgets')
    if gal is not None:
        s0 = 2026 if 2026 in set(gal.seed) else int(gal.seed.max())
        gg = lambda g_, meth='zero', rule='cos': per(gal[gal.seed == s0], gallery=g_, method=meth, rule=rule, query='s4R4')  # noqa: E731
        claim('predicted population gallery beats the spectrum-matched image-only gallery (zero labels, scored alone)',
              gg('anchor'), gg('img_spec'), 'round 8 gallery test', 'image-only controls cover whitening, spectrum and rank, not every possible metric', 'gallery')
        claim('predicted population gallery beats the same ridge with pairs shuffled within concept', gg('anchor'), gg('anchor_permconcept'),
              'round 8 gallery test', 'as above', 'gallery')
        claim('measured source templates beat predicted ones (headroom; privileged)', gg('oracle'), gg('anchor'),
              'round 8 gallery test', 'requires earlier people to have seen every candidate', 'gallery')
        claim('retrieval-trained template predictor vs ridge, rigid k = 500', gg('anchor_ret', 'rigid'), gg('anchor_meas', 'rigid'),
              'round 8 gallery test', 'trained on source people and held concepts only', 'gallery')
    if scr is not None and r6 is not None:
        ep = int(scr[scr.target == 1].groupby('epochs').top1.mean().idxmax()) if 1 in set(scr.target) else int(scr.epochs.max())
        sc = per(scr[scr.epochs == ep], k=500, query='s4R4', rule='cos')
        rg = per(r6[(r6.decoder == 'repaug') & (r6.cal_session == CS)], k=500, method='rigid', readout='anchor', rule='cos', query='s4R4')
        claim('calibrated cross-subject decoder beats a decoder trained from scratch on the same 500 images', rg, sc,
              'round 8 scratch control', 'scratch recipe tuned only by epochs (person 1)')
    # ---- contrast dot plot
    if claims:
        cl = pd.DataFrame(claims)
        cl.to_csv(os.path.join(out, 'claims.csv'), index=False)
        tcols = [f't{t}' for t in TEST if f't{t}' in cl]
        fig, ax = plt.subplots(figsize=(10, 0.7 * len(cl) + 1))
        for i, r in cl.iterrows():
            y = len(cl) - i
            ax.scatter([r[t] for t in tcols], [y] * len(tcols), s=12, color='#9e9e9e', zorder=2)
            lo, hi = [float(x) for x in r.ci.replace(' to ', ' ').split()]
            ax.plot([lo, hi], [y, y], color='#2b6cb0', lw=2, zorder=3)
            ax.scatter([r['mean']], [y], s=40, color='#2b6cb0', zorder=4)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_yticks(range(len(cl), 0, -1))
        import textwrap
        ax.set_yticklabels(['\n'.join(textwrap.wrap(c, 60)) for c in cl.claim], fontsize=7)
        ax.set_xlabel('paired difference, top-1 points (grey = one person; blue = mean and 95% CI)')
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'contrasts_people_{mon}.png'), dpi=200)
        plt.close(fig)
        with open(os.path.join(out, 'claims.md'), 'w') as f:
            f.write('| Kind | Claim | Mean (points) | 95% CI | Improved | Wilcoxon p | Evidence | Limitation |\n|---|---|---|---|---|---|---|---|\n')
            for _, r in cl.iterrows():
                f.write(f'| {r.kind} | {r.claim} | {r["mean"]:+.2f} | {r.ci} | {r.improved} | {r.wilcoxon_p} | {r.evidence} | {r.limitation} |\n')
    print(f'figures and tables written to {out}: ' + ', '.join(sorted(os.listdir(out))))


# ---------------------------------------------------------------------------------------------------------------
# 3. output edit decomposition
# ---------------------------------------------------------------------------------------------------------------
def _center(Xs, Ys):
    xm, ym = Xs.mean(0), Ys.mean(0)
    return Xs - xm, Ys - ym, xm, ym


def e_shift(Xs, Ys, lam=None):
    c = (Ys - Xs).mean(0)
    return lambda Z: Z + c


def e_scale(Xs, Ys, lam=None):
    Xc, Yc, xm, ym = _center(Xs, Ys)
    g = min(max(float((Xc * Yc).sum() / ((Xc * Xc).sum() + 1e-8)), 0.5), 2.0)
    return lambda Z: (Z - xm) * g + ym


def e_procrustes(Xs, Ys, lam, scale=True):
    Xc, Yc, xm, ym = _center(Xs, Ys)
    M = Xc.T @ Yc
    s = float(torch.trace(Xc.T @ Xc)) / Xc.shape[1] + 1e-8
    U, _, Vh = R.safe_svd(M + lam * s * torch.eye(M.shape[0], device=M.device))
    Rm = U @ Vh
    g = min(max(float(torch.trace(Rm.T @ M)) / (float(torch.trace(Xc.T @ Xc)) + 1e-8), 0.5), 2.0) if scale else 1.0
    fn = lambda Z: (Z - xm) @ Rm * g + ym  # noqa: E731
    fn.R = Rm
    return fn


def e_lowrank(Xs, Ys, lam, U):
    Xc, Yc, xm, ym = _center(Xs, Ys)
    P = Xc @ U
    Rr = (Yc - Xc) @ U
    G = P.T @ P
    A = torch.linalg.solve(G + lam * float(torch.trace(G)) / len(G) * torch.eye(len(G), device=G.device), P.T @ Rr)
    return lambda Z: (Z - xm) + ym + ((Z - xm) @ U) @ A @ U.T


def e_full(Xs, Ys, lam):
    Xc, Yc, xm, ym = _center(Xs, Ys)
    G = Xc.T @ Xc
    A = torch.linalg.solve(G + lam * float(torch.trace(G)) / len(G) * torch.eye(len(G), device=G.device), Xc.T @ Yc)
    return lambda Z: (Z - xm) @ A + ym


def cv_pick(fit, Xs, Ys, grid, folds=5, seed=0):
    """Strength chosen by k-fold CV over calibration images (held-out cosine to the targets); same grid for every method."""
    if grid is None:
        return None, np.nan
    idx = np.random.RandomState(seed).permutation(len(Xs))
    parts = np.array_split(idx, folds)
    best = (-np.inf, grid[0])
    for lam in grid:
        sc = []
        for p in parts:
            tr = torch.as_tensor(np.setdiff1d(idx, p), device=Xs.device)
            te = torch.as_tensor(p, device=Xs.device)
            fn = fit(Xs[tr], Ys[tr], lam)
            sc.append(float(F.cosine_similarity(fn(Xs[te]), Ys[te], dim=1).mean()))
        if np.mean(sc) > best[0]:
            best = (float(np.mean(sc)), lam)
    return best[1], best[0]


def edit_target(B, mon='posterior17'):
    out = os.path.join(odir('edit'), f'edit_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[edit] t{B}: done')
        return
    ctx = ctx_for(B, mon)
    if ctx is None:
        print(f'[edit] t{B}: no seed-{SEED} repaug checkpoint, skipping')
        return
    t0 = time.time()
    c = ctx.c
    tr = c.ses['train']
    reps = R.load_reps(B, mon)
    cand = np.array([i for i in c.pool if (tr[i] == CS).sum() == 2])
    # fold-specific bases from the SOURCE population responses
    with torch.no_grad():
        P = c.pool_z - c.pool_z.mean(0)
        ev, V = torch.linalg.eigh(P.T @ P / len(P))
        V, ev = V.flip(1), ev.flip(0)
        tot = float(ev.sum())
        D = V.shape[0]
        bases = {}
        for r in RANKS:
            bases[f'pca_{r}'] = V[:, :r]
            for j in range(3):
                g = torch.Generator(device='cpu').manual_seed(100 * r + j)
                Q, _ = torch.linalg.qr(torch.randn(D, r, generator=g).to(dev()))
                bases[f'rand_{r}_b{j}'] = Q
                if r < 64:
                    Qv, _ = torch.linalg.qr(torch.randn(64, r, generator=g).to(dev()))
                    bases[f'randvar_{r}_b{j}'] = V[:, :64] @ Qv
        bvar = {k: float(((P @ U) ** 2).sum() / len(P)) / tot for k, U in bases.items()}
    rows, cons = [], []
    for k in K_EDIT:
        if k > len(cand):
            continue
        for draw in range(R6m.ndraw(k)):
            ims = GL.draw_images(cand, k, draw)
            rr = np.stack([np.nonzero(tr[i] == CS)[0] for i in ims])
            x2 = np.asarray(reps[ims[:, None], rr], np.float32)
            pos = torch.as_tensor(np.array([ctx.pool_pos[int(i)] for i in ims]), device=dev())
            Ys = c.pool_z[pos]
            d1 = c.mS['h1'] - I.acts(c.f0, x2.reshape(-1, *x2.shape[2:]))['h1'].mean(0)
            Xs = X.acts2(c.f0, x2.mean(1), d1)['z']
            Zq = c.zq(d1)
            fits = {'shift': (e_shift, None), 'scale': (e_scale, None),
                    'rot': (lambda a, b, l: e_procrustes(a, b, l, False), REL),
                    'rigid': (lambda a, b, l: e_procrustes(a, b, l, True), REL), 'full': (e_full, REL)}
            for bn, U in bases.items():
                fits[bn] = ((lambda U: (lambda a, b, l: e_lowrank(a, b, l, U)))(U), REL)
            cells = [('zero', None, ctx.Ganc)]
            meta = {}
            for nm, (fit, grid) in fits.items():
                if k < 10 and grid is not None:
                    continue
                lam, cvs = cv_pick(fit, Xs, Ys, grid, seed=draw)
                fn = fit(Xs, Ys, lam)
                cells.append((nm, fn, ctx.Ganc))
                meta[nm] = (lam, cvs)
                if nm == 'rigid':
                    Dm = fn.R - torch.eye(len(fn.R), device=dev())
                    Uq = torch.linalg.svd(Dm.double(), full_matrices=False)[0][:, :8].float().cpu().numpy()
                    cons.append(dict(target=B, k=k, draw=draw, shift=(Ys - Xs).mean(0).cpu().numpy(), rot_dirs=Uq))
            base = dict(k=k, draw=draw)
            rows0 = []
            score_rows(base, Zq, cells, rows0)
            for r_ in rows0:
                lam, cvs = meta.get(r_['method'], (None, np.nan))
                r_.update(lam=lam, cv_cos=cvs, basis_var=bvar.get(r_['method'], np.nan))
            rows += rows0
            print(f'   edit k={k} draw {draw} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    keep = ['k', 'draw', 'method', 'rule', 'query']
    df = df.groupby(keep).agg(top1=('top1', 'mean'), lam=('lam', 'first'), cv_cos=('cv_cos', 'first'),
                              basis_var=('basis_var', 'first')).reset_index()
    df['target'], df['montage'] = B, mon
    df.to_csv(out, index=False)
    np.savez(out.replace('.csv', '_consistency.npz'), target=np.array([x['target'] for x in cons]), k=np.array([x['k'] for x in cons]),
             draw=np.array([x['draw'] for x in cons]), shift=np.stack([x['shift'] for x in cons]), rot_dirs=np.stack([x['rot_dirs'] for x in cons]))
    print(f'  [edit] t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# 4a. equal presentation budgets
# ---------------------------------------------------------------------------------------------------------------
def budget_target(B, mon='posterior17'):
    out = os.path.join(odir('budget'), f'budget_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[budget] t{B}: done')
        return
    ctx = ctx_for(B, mon)
    if ctx is None:
        print(f'[budget] t{B}: no seed-{SEED} repaug checkpoint, skipping')
        return
    t0 = time.time()
    c = ctx.c
    tr = c.ses['train']
    reps = R.load_reps(B, mon)
    cand2 = np.array([i for i in c.pool if (tr[i] == CS).sum() == 2])
    cand4 = np.array([i for i in cand2 if (tr[i] < CS).sum() == 2])
    rows = []
    for budget, splits in BUDGETS.items():
        for k, npres in splits:
            pool_ = cand4 if npres == 4 else cand2
            if len(pool_) < k:
                print(f'   {k}x{npres}: only {len(pool_)} eligible images, skipping')
                continue
            for draw in range(5 if budget <= 100 else 3):
                ims = np.sort(np.random.RandomState(12000 + 31 * k + 7 * npres + draw).choice(pool_, k, replace=False))
                if npres == 4:
                    rr = np.stack([np.r_[np.nonzero(tr[i] == CS)[0], np.nonzero(tr[i] < CS)[0]] for i in ims])
                else:
                    rr = np.stack([np.nonzero(tr[i] == CS)[0][:npres] for i in ims])
                xs = np.asarray(reps[ims[:, None], rr], np.float32)
                pos = torch.as_tensor(np.array([ctx.pool_pos[int(i)] for i in ims]), device=dev())
                Ys = c.pool_z[pos]
                d1 = c.mS['h1'] - I.acts(c.f0, xs.reshape(-1, *xs.shape[2:]))['h1'].mean(0)
                Xs = X.acts2(c.f0, xs.mean(1), d1)['z']
                cells = [('zero', None, ctx.Ganc), ('shift', e_shift(Xs, Ys), ctx.Ganc),
                         ('rigid', e_procrustes(Xs, Ys, 1.0), ctx.Ganc)]
                score_rows(dict(budget=budget, schedule=f'{k}x{npres}', draw=draw), c.zq(d1), cells, rows)
        print(f'   budget {budget} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows).groupby(['budget', 'schedule', 'draw', 'method', 'rule', 'query']).top1.mean().reset_index()
    df['target'], df['montage'] = B, mon
    df.to_csv(out, index=False)
    print(f'  [budget] t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# 4b. linear cross-repeat denoiser
# ---------------------------------------------------------------------------------------------------------------
DN_REL = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
DN_PCA = [16, 64, 256, 1024, 2048]
DN_IMGS = 1500


def _dn_stats(c, mon, seed=0):
    """Per-source second moments for single presentation -> mean of the other presentations (and shuffled pairs)."""
    rng = np.random.RandomState(9100 + seed)
    st = []
    for s_ in c.src:
        ims = np.sort(rng.choice(c.pool, DN_IMGS, replace=False))
        x = torch.as_tensor(np.asarray(R.load_reps(s_, mon)[ims], np.float32), device=dev())   # (n, 4, C, T)
        n, nr = x.shape[:2]
        x = x.reshape(n, nr, -1).double()
        tot = x.sum(1)
        perm = torch.as_tensor(rng.permutation(n), device=dev())
        a = dict(n=0, sx=0, sy=0, sys_=0, xx=0, xy=0, xys=0, yy=0.0, yys=0.0)
        for i in range(nr):
            xi = x[:, i]
            yi = (tot - xi) / (nr - 1)                       # disjoint views: the other presentations
            ysh = yi[perm]                                   # repeats of a different image
            a['n'] += n
            a['sx'] = a['sx'] + xi.sum(0)
            a['sy'] = a['sy'] + yi.sum(0)
            a['sys_'] = a['sys_'] + ysh.sum(0)
            a['xx'] = a['xx'] + xi.T @ xi
            a['xy'] = a['xy'] + xi.T @ yi
            a['xys'] = a['xys'] + xi.T @ ysh
            a['yy'] += float((yi * yi).sum())
            a['yys'] += float((ysh * ysh).sum())
        st.append(a)
        del x, tot
    return st


def _sum(st, idx):
    out = {}
    for k in st[0]:
        out[k] = sum(st[i][k] for i in idx)
    return out


def _cov(a, shuf=False):
    n = a['n']
    mx, my = a['sx'] / n, (a['sys_'] if shuf else a['sy']) / n
    Cxx = a['xx'] / n - torch.outer(mx, mx)
    Cxy = (a['xys'] if shuf else a['xy']) / n - torch.outer(mx, my)
    vy = (a['yys'] if shuf else a['yy']) / n - float(my @ my)
    return Cxx, Cxy, vy, mx, my


def _r2(W, b, val):
    """Held-out R^2 of y_hat = x W + b against the disjoint-view target (unshuffled)."""
    n = val['n']
    mx, my = val['sx'] / n, val['sy'] / n
    Exx, Exy = val['xx'] / n, val['xy'] / n
    Eyy = val['yy'] / n
    pred_sq = float(torch.sum((W.T @ Exx) * W.T)) + 2 * float(mx @ W @ b) + float(b @ b)
    cross = float(torch.sum(W * Exy)) + float(my @ b)
    mse = Eyy - 2 * cross + pred_sq
    return 1 - mse / (Eyy - float(my @ my))


def fit_denoisers(c, mon):
    t0 = time.time()
    st = _dn_stats(c, mon)
    rng = np.random.RandomState(9200)
    vs = rng.choice(len(st), 2, replace=False)
    ts = np.setdiff1d(np.arange(len(st)), vs)
    tr_, va_ = _sum(st, ts), _sum(st, vs)
    Cxx, Cxy, _, mx, my = _cov(tr_)
    _, Cxys, _, _, mys = _cov(tr_, True)
    ev, V = torch.linalg.eigh(Cxx)
    ev = ev.clamp_min(0)
    sc = float(ev.mean())
    res = {}
    ident = (torch.eye(len(Cxx), device=dev(), dtype=Cxx.dtype), torch.zeros(len(Cxx), device=dev(), dtype=Cxx.dtype))
    res['identity'] = (_r2(*ident, va_), None)
    best = {}
    for nm, CXY, MY in (('ridge', Cxy, my), ('shuffled', Cxys, mys)):
        VtC = V.T @ CXY
        for a in DN_REL:
            W = V @ (VtC / (ev + a * sc)[:, None])
            b = MY - mx @ W
            r2 = _r2(W, b, va_)
            if nm not in best or r2 > best[nm][0]:
                best[nm] = (r2, a)
    for kk in DN_PCA:
        if kk >= len(Cxx):
            continue
        Vk = V[:, -kk:]
        W = Vk @ Vk.T
        r2 = _r2(W, mx - mx @ W, va_)
        if 'pca' not in best or r2 > best['pca'][0]:
            best['pca'] = (r2, kk)
    # refit on all 9 sources with the chosen settings
    al = _sum(st, range(len(st)))
    Cxx, Cxy, _, mx, my = _cov(al)
    _, Cxys, _, _, mys = _cov(al, True)
    ev, V = torch.linalg.eigh(Cxx)
    ev = ev.clamp_min(0)
    sc = float(ev.mean())
    maps = {}

    def vm(W, MY):
        # variance-matched: rescale each output dimension to the raw input's spread (the decoder expects raw-scale EEG)
        g = (torch.diagonal(Cxx) / (((Cxx @ W) * W).sum(0) + 1e-12)).clamp_min(0).sqrt()
        Wg = W * g[None]
        return Wg.float(), (MY - mx @ Wg).float()
    for nm, CXY, MY in (('ridge', Cxy, my), ('shuffled', Cxys, mys)):
        W = V @ ((V.T @ CXY) / (ev + best[nm][1] * sc)[:, None])
        maps[nm] = vm(W, MY)
        if nm == 'ridge':
            maps['ridge_raw'] = (W.float(), (MY - mx @ W).float())
    Vk = V[:, -best['pca'][1]:]
    maps['pca'] = vm(Vk @ Vk.T, mx)
    info = {f'{k}_val_r2': round(v[0], 4) for k, v in best.items()}
    info.update({f'{k}_setting': v[1] for k, v in best.items()}, identity_val_r2=round(res['identity'][0], 4))
    print(f'   denoisers fitted ({time.time() - t0:.0f}s): held-out-source R^2 to disjoint repeats {info}', flush=True)
    del st
    return maps, info


def apply_dn(x, mp):
    if mp is None:
        return x
    W, b = mp
    sh = x.shape
    xt = torch.as_tensor(np.asarray(x, np.float32), device=dev()).reshape(sh[0], -1)
    return ((xt @ W + b).reshape(sh)).cpu().numpy()


def fit_target_dn(x2, rel=(0.1, 1.0, 10.0)):
    """Ridge (dual form) on the target's own calibration: presentation 1 -> 2 and 2 -> 1; strength by 2-fold over images."""
    k = len(x2)
    Xa = torch.as_tensor(np.concatenate([x2[:, 0], x2[:, 1]]).reshape(2 * k, -1), device=dev()).double()
    Ya = torch.as_tensor(np.concatenate([x2[:, 1], x2[:, 0]]).reshape(2 * k, -1), device=dev()).double()
    half = np.random.RandomState(0).permutation(k)[:k // 2]
    msk = np.zeros(k, bool)
    msk[half] = True
    msk = torch.as_tensor(np.r_[msk, msk], device=dev())

    def fit(Xt, Yt, a):
        mx, my = Xt.mean(0), Yt.mean(0)
        Xc, Yc = Xt - mx, Yt - my
        Kmat = Xc @ Xc.T
        sc = float(torch.trace(Kmat)) / len(Kmat)
        Al = torch.linalg.solve(Kmat + a * sc * torch.eye(len(Kmat), device=dev(), dtype=Kmat.dtype), Yc)
        W = Xc.T @ Al
        return W, my - mx @ W
    best = (-np.inf, rel[0])
    for a in rel:
        W, b = fit(Xa[msk], Ya[msk], a)
        pr = Xa[~msk] @ W + b
        r2 = 1 - float(((Ya[~msk] - pr) ** 2).sum()) / float(((Ya[~msk] - Ya[~msk].mean(0)) ** 2).sum())
        if r2 > best[0]:
            best = (r2, a)
    W, b = fit(Xa, Ya, best[1])
    Xc = Xa - Xa.mean(0)
    g = (Xc.var(0) / ((Xc @ W).var(0) + 1e-12)).sqrt()                # variance-matched, as for the source denoisers
    Wg = W * g[None]
    return (Wg.float(), (Ya.mean(0) - Xa.mean(0) @ Wg).float()), best


def denoise_target(B, mon='posterior17'):
    out = os.path.join(odir('denoise'), f'denoise_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[denoise] t{B}: done')
        return
    ctx = ctx_for(B, mon)
    if ctx is None:
        print(f'[denoise] t{B}: no seed-{SEED} repaug checkpoint, skipping')
        return
    t0 = time.time()
    c = ctx.c
    tr = c.ses['train']
    reps = R.load_reps(B, mon)
    maps, info = fit_denoisers(c, mon)
    maps['identity'] = None
    cand = np.array([i for i in c.pool if (tr[i] == CS).sum() == 2])
    rows = []
    for draw in range(R6m.ndraw(500)):
        ims = GL.draw_images(cand, 500, draw)
        rr = np.stack([np.nonzero(tr[i] == CS)[0] for i in ims])
        x2 = np.asarray(reps[ims[:, None], rr], np.float32)
        pos = torch.as_tensor(np.array([ctx.pool_pos[int(i)] for i in ims]), device=dev())
        Ys = c.pool_z[pos]
        tmap, tinfo = fit_target_dn(x2)
        variants = dict(maps, target=tmap)
        for nm, mp in variants.items():
            x2d = apply_dn(x2.reshape(-1, *x2.shape[2:]), mp).reshape(x2.shape)
            d1 = c.mS['h1'] - I.acts(c.f0, x2d.reshape(-1, *x2d.shape[2:]))['h1'].mean(0)
            Xs = X.acts2(c.f0, x2d.mean(1), d1)['z']
            Zq = {qk: [X.acts2(c.f0, apply_dn(q, mp), d1)['z'] for q in c.Q[qk]] for qk in QK}
            cells = [('zero', None, ctx.Ganc), ('rigid', e_procrustes(Xs, Ys, 1.0), ctx.Ganc), ('plain_img', None, ctx.Gimg)]
            rr_ = []
            score_rows(dict(denoiser=nm, draw=draw), Zq, cells, rr_)
            for r_ in rr_:
                r_['target_val_r2'] = tinfo[0] if nm == 'target' else np.nan
            rows += rr_
        print(f'   denoise draw {draw} ({time.time() - t0:.0f}s; target-calibrated held-out R^2 {tinfo[0]:.3f})', flush=True)
    df = pd.DataFrame(rows).groupby(['denoiser', 'draw', 'method', 'rule', 'query']).agg(
        top1=('top1', 'mean'), target_val_r2=('target_val_r2', 'first')).reset_index()
    df['target'], df['montage'] = B, mon
    for k_, v_ in info.items():
        df[k_] = v_
    df.to_csv(out, index=False)
    print(f'  [denoise] t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# 2. encoders x visual targets
# ---------------------------------------------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    """Compact temporal-spatial CNN (NICE / ATM 'TSConv' style) -> 1024-d, the same width EEGProject outputs."""

    def __init__(self, n_ch, T=250, f=40, out_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, f, (1, 25)), nn.AvgPool2d((1, 51), (1, 5)), nn.BatchNorm2d(f), nn.ELU(),
            nn.Conv2d(f, f, (n_ch, 1)), nn.BatchNorm2d(f), nn.ELU(), nn.Dropout(0.5))
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 1, n_ch, T)).numel()
        self.head = nn.Sequential(nn.Linear(n, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)).flatten(1))


class EncModel(nn.Module):
    """Same interface as N.NBModel, with a choice of encoder and image-feature width."""

    def __init__(self, enc, n_ch, feat_dim=1024):
        super().__init__()
        self.encoder = N.EEGProject(channels_num=n_ch) if enc == 'eegproject' else CNNEncoder(n_ch)
        self.eeg_projector = N.ProjectorLinear()
        self.img_projector = N.ProjectorLinear(input_dim=feat_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))), requires_grad=False)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.eeg_projector(self.encoder(x.float()))

    def gallery(self, feats):
        return F.normalize(self.img_projector(feats.float()), dim=-1)

    def scale(self):
        return self.softplus(self.logit_scale)


def tgt_feats(kind, split):
    if kind in ('aug', 'clean'):
        return N.features(split, kind)
    p = os.path.join(N.ENV['feat_dir'], f'image_{split}_rn50_l3.npy')
    assert os.path.exists(p), f'{p} missing: run `nb_round9.py feats` first'
    return torch.from_numpy(np.load(p).astype(np.float32))


def feats(local='/content/things_images', hf_repo='Haitao999/things-eeg'):
    """RN50 layer-3 (global-average-pooled, 1024-d) features of every THINGS-EEG2 image, in NeuroBridge order (sorted
    concept folders x sorted files). Order and preprocessing are checked by recomputing the final clean RN50 feature
    and comparing it with NeuroBridge's stored clean features."""
    import open_clip
    from PIL import Image
    from huggingface_hub import list_repo_files, snapshot_download
    import zipfile
    import tarfile
    outp = {s: os.path.join(N.ENV['feat_dir'], f'image_{s}_rn50_l3.npy') for s in ('train', 'test')}
    if all(os.path.exists(p) for p in outp.values()):
        print('intermediate features exist:', outp)
        return
    files = list_repo_files(hf_repo, repo_type='dataset')
    hit = [f for f in files if 'Image_set_Resize' in f]
    assert hit, f'no Image_set_Resize in {hf_repo}'
    pre = hit[0][:hit[0].index('Image_set_Resize')]
    snapshot_download(hf_repo, repo_type='dataset', local_dir=local, allow_patterns=[f'{pre}Image_set_Resize*'], max_workers=4)
    for arc in glob.glob(f'{local}/**/Image_set_Resize*.zip', recursive=True) + glob.glob(f'{local}/**/Image_set_Resize*.tar*', recursive=True):
        print('extracting', arc)
        (zipfile.ZipFile(arc) if arc.endswith('.zip') else tarfile.open(arc)).extractall(os.path.dirname(arc))
    root = next((os.path.join(r, d) for r, ds, _ in os.walk(local) for d in ds if d == 'Image_set_Resize'), None)
    assert root, 'Image_set_Resize folder not found after download'
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai', device=dev())
    model.eval()
    vis = model.visual
    for split, sub, n_exp in (('test', 'test_images', 200), ('train', 'train_images', 16540)):
        d0 = os.path.join(root, sub)
        paths = [os.path.join(d0, cdir, f) for cdir in sorted(os.listdir(d0)) for f in sorted(os.listdir(os.path.join(d0, cdir)))]
        assert len(paths) == n_exp, f'{split}: {len(paths)} images, expected {n_exp}'
        mids, fins = [], []
        with torch.no_grad():
            for i in range(0, len(paths), 128):
                x = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in paths[i:i + 128]]).to(dev())
                x = x.type(vis.conv1.weight.dtype)
                h = vis.act3(vis.bn3(vis.conv3(vis.act2(vis.bn2(vis.conv2(vis.act1(vis.bn1(vis.conv1(x)))))))))
                h = vis.avgpool(h)
                h = vis.layer3(vis.layer2(vis.layer1(h)))
                mids.append(h.mean((2, 3)).float().cpu())
                fins.append(vis.attnpool(vis.layer4(h)).float().cpu())
        mid, fin = torch.cat(mids).numpy(), torch.cat(fins)
        ref = N.features(split, 'clean')
        cosd = F.cosine_similarity(F.normalize(fin, dim=-1), F.normalize(ref, dim=-1), dim=1)
        sub_ = torch.as_tensor(np.random.RandomState(0).choice(len(fin), min(200, len(fin)), replace=False))
        match = float((F.normalize(fin[sub_], dim=-1) @ F.normalize(ref[sub_], dim=-1).T).argmax(1).eq(torch.arange(len(sub_))).float().mean())
        print(f'  {split}: final-layer cosine to NeuroBridge clean features median {float(cosd.median()):.3f} '
              f'(min {float(cosd.min()):.3f}); order check: {100 * match:.1f}% of 200 sampled images match their own row')
        assert match > 0.9, 'image order or preprocessing does not match NeuroBridge: do not use these features'
        np.save(outp[split], mid.astype(np.float32))
    print('saved', outp)


def _r9_ckpt(enc, tgt, mon, B, seed, lr):
    return os.path.join(N.ENV['ckpt_dir'], f'r9_{enc}_{tgt}_{mon}_t{B:02d}_s{seed}_lr{lr:g}.pt')


def _source_data(B, mon):
    Xs, IM, val = [], [], []
    for s in PEOPLE:
        if s == B:
            continue
        d = N.subject_data(s, mon, keys=('train', 'train_labels'))
        pool, draws = N.split(d['train_labels'])
        Xs.append(torch.from_numpy(np.asarray(R.load_reps(s, mon)[pool], np.float16)))
        IM.append(torch.from_numpy(pool))
        val += [(d['train'][ix], ix) for ix in draws[:3]]
        del d
    return torch.cat(Xs), torch.cat(IM), val


def train_cell(enc, tgt, mon, B, seed, lr, epochs=50, batch=1024, data=None, max_epochs=None, verbose=True):
    """repaug recipe (round 5) with a choice of encoder and visual target. Returns (model, val top-1 history)."""
    path = _r9_ckpt(enc, tgt, mon, B, seed, lr)
    if max_epochs is None and os.path.exists(path):
        blob = torch.load(path, map_location=dev())
        m = EncModel(enc, blob['channels'], blob['feat_dim']).to(dev())
        m.load_state_dict(blob['state_dict'])
        return m.eval(), blob['hist']
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xc, IMG, val = data if data is not None else _source_data(B, mon)
    X_ = Xc.to(dev())
    IMG = IMG.to(dev())
    n, _, C, T = X_.shape
    ftr = tgt_feats(tgt, 'train').to(dev())
    m = EncModel(enc, C, ftr.shape[1]).to(dev())
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    kv = torch.tensor([1., 2., 4.], device=dev())
    hist = []
    E = max_epochs or epochs
    for ep in range(1, E + 1):
        m.train()
        perm = torch.randperm(n, device=dev())
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = X_[idx].float()
            b = len(idx)
            k = kv[torch.randint(0, 3, (b,), device=dev())]
            rank = torch.rand(b, 4, device=dev()).argsort(1).argsort(1).float()
            mask = (rank < k[:, None]).float()
            xb = (xb * mask[..., None, None]).sum(1) / k[:, None, None]
            sm = F.avg_pool1d(xb.reshape(-1, 1, T), 5, 1, 2, count_include_pad=False).reshape(b, C, T)
            xb = torch.where(torch.rand(b, C, 1, device=dev()) < 0.3, sm, xb)
            loss = N.nb_loss(m(xb), m.gallery(ftr[IMG[idx]]), m.scale())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == E:
            m.eval()
            with torch.no_grad():
                v = float(np.mean([N.retrieval(m, x, ftr[ix])[0] for x, ix in val]))
            hist.append((ep, v))
            if verbose:
                print(f'    [{enc}/{tgt} lr {lr:g}] ep {ep:3d} val {v:5.1f}', flush=True)
    m.eval()
    if max_epochs is None:
        torch.save(dict(state_dict=m.state_dict(), channels=C, feat_dim=ftr.shape[1], hist=hist, enc=enc, tgt=tgt, lr=lr,
                        seed=seed, epochs=epochs, params=sum(p.numel() for p in m.parameters())), path)
    del X_
    return m, hist


def profile(mon='posterior17', B=1):
    data = _source_data(B, mon)
    rows = []
    for enc in ENCS:
        for tgt in TGTS:
            try:
                tgt_feats(tgt, 'train')
            except AssertionError as e:
                print(f'  {enc}/{tgt}: {e}')
                continue
            if dev() == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            m, _ = train_cell(enc, tgt, mon, B, 0, 1e-4, data=data, max_epochs=1, verbose=False)
            if dev() == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0
            rows.append(dict(encoder=enc, target=tgt, sec_per_epoch=round(dt, 1), min_per_model=round(dt * 50 / 60, 1),
                             params=sum(p.numel() for p in m.parameters())))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if len(df):
        tot = df.min_per_model.sum()
        print(f'one seed, all cells: tune ~{tot * len(LRS):.0f} min (person-1 fold, {len(LRS)} learning rates); '
              f'grid ~{tot * 10:.0f} min for 10 folds; each extra seed ~{tot * 10:.0f} min')
    df.to_csv(os.path.join(odir('encoders'), 'profile.csv'), index=False)


def tune(mon='posterior17', seed=2025):
    data = _source_data(1, mon)
    rows = []
    for enc in ENCS:
        for tgt in TGTS:
            for lr in LRS:
                _, hist = train_cell(enc, tgt, mon, 1, seed, lr, data=data)
                rows.append(dict(encoder=enc, target=tgt, lr=lr, val_top1=hist[-1][1]))
    df = pd.DataFrame(rows)
    pick = df.loc[df.groupby(['encoder', 'target']).val_top1.idxmax()]
    print(df.pivot_table(index=['encoder', 'target'], columns='lr', values='val_top1').round(2).to_string())
    print('chosen (person-1 fold, source validation concepts):\n' + pick.to_string(index=False))
    pick.to_csv(os.path.join(odir('encoders'), 'lr_choice.csv'), index=False)


def _queries(B, mon):
    """Round 4's query sets (session-4 test presentations at R = 1 / 4 / 20, 3 draws below 20; the benchmark)."""
    d = N.subject_data(B, mon, keys=('test', 'test_trials'))
    ses = R4m.load_ses(B)
    tt, te = d['test_trials'], ses['test']
    s4 = [np.nonzero(te[i] == 4)[0] for i in range(200)]
    n4 = min(len(x) for x in s4)
    tt4 = np.stack([tt[i, s4[i][:n4]] for i in range(200)]).astype(np.float32)
    Q = {'bench': [d['test'].astype(np.float32)]}
    for Rq in (1, 4, 20):
        Q[f's4R{Rq}'] = [tt4[:, np.random.RandomState(1000 + q).choice(n4, Rq, replace=False)].mean(1) for q in range(3 if Rq < 20 else 1)]
    return Q, ses


@torch.no_grad()
def _emb(m, x, bs=1024):
    return torch.cat([m(torch.as_tensor(np.asarray(x[i:i + bs], np.float32), device=dev())) for i in range(0, len(x), bs)])


def eval_cell(m, tgt, mon, B):
    """Plain image gallery; zero-label = output mean offset + population gallery; rigid k = 500 (earlier day)."""
    ftr, fte = tgt_feats(tgt, 'train').to(dev()), tgt_feats(tgt, 'test').to(dev())
    Q, ses = _queries(B, mon)
    d = N.subject_data(B, mon, keys=('train_labels',))
    pool, _ = N.split(d['train_labels'])
    # population responses from the SOURCE people (4-presentation averages of pool images)
    Zs = []
    for s in PEOPLE:
        if s == B:
            continue
        Zs.append(_emb(m, N.subject_data(s, mon, keys=('train',))['train'][pool]))
    pool_z = torch.stack(Zs).mean(0)
    msrc = pool_z.mean(0)
    Bm, xm, ym = X.ridge_svd(ftr[torch.as_tensor(pool, device=dev())], pool_z)
    Ganc = nrm((fte - xm) @ Bm + ym)
    Gimg = m.gallery(fte)
    pos_of = {int(i): j for j, i in enumerate(pool)}
    tr = ses['train']
    reps = R.load_reps(B, mon)
    cand = np.array([i for i in pool if (tr[i] == CS).sum() == 2])
    rows = []
    Zq_plain = {qk: [_emb(m, q) for q in Q[qk]] for qk in QK}
    score_rows(dict(draw=-1), Zq_plain, [('plain', None, Gimg)], rows)
    for draw in range(R6m.ndraw(500)):
        ims = GL.draw_images(cand, 500, draw)
        rr = np.stack([np.nonzero(tr[i] == CS)[0] for i in ims])
        x2 = np.asarray(reps[ims[:, None], rr], np.float32)
        off = msrc - _emb(m, x2.reshape(-1, *x2.shape[2:])).mean(0)        # label-free output offset
        Xs = _emb(m, x2.mean(1)) + off
        Ys = pool_z[torch.as_tensor([pos_of[int(i)] for i in ims], device=dev())]
        Zq = {qk: [z + off for z in Zq_plain[qk]] for qk in QK}
        score_rows(dict(draw=draw), Zq, [('zero', None, Ganc), ('rigid', e_procrustes(Xs, Ys, 1.0), Ganc)], rows)
    return rows


def encoders(targets, mon='posterior17', seeds=(2025,)):
    pf = os.path.join(odir('encoders'), 'lr_choice.csv')
    assert os.path.exists(pf), 'run `nb_round9.py tune` first (learning rate per cell, person-1 fold)'
    lrs = {(r.encoder, r.target): float(r.lr) for r in pd.read_csv(pf).itertuples()}
    for seed in seeds:
        for B in targets:
            out = os.path.join(odir('encoders'), f'enc_s{seed}_t{B:02d}_{mon}.csv')
            if os.path.exists(out):
                print(f'[encoders] seed {seed} t{B}: done')
                continue
            t0 = time.time()
            data = _source_data(B, mon)
            rows = []
            for enc in ENCS:
                for tgt in TGTS:
                    if (enc, tgt) not in lrs:
                        continue
                    m, hist = train_cell(enc, tgt, mon, B, seed, lrs[(enc, tgt)], data=data, verbose=False)
                    rr = eval_cell(m, tgt, mon, B)
                    for r_ in rr:
                        r_.update(encoder=enc, visual_target=tgt, lr=lrs[(enc, tgt)], val_top1=hist[-1][1])
                    rows += rr
                    print(f'   [encoders] t{B} {enc}/{tgt}: val {hist[-1][1]:.1f} ({time.time() - t0:.0f}s)', flush=True)
            df = pd.DataFrame(rows).groupby(['encoder', 'visual_target', 'draw', 'method', 'rule', 'query']).agg(
                top1=('top1', 'mean'), lr=('lr', 'first'), val_top1=('val_top1', 'first')).reset_index()
            df['target'], df['montage'], df['seed'] = B, mon, seed
            df.to_csv(out, index=False)
            print(f'  [encoders] seed {seed} t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def _ps(a, b, label):
    try:
        return R4m.pstats(a, b, label)
    except (KeyError, ValueError):
        print(f'  {label}: no data')


def _collapse_bases(df):
    df = df.copy()
    df['method'] = df.method.str.replace(r'_b\d+$', '', regex=True)
    return df


def report(mon='posterior17'):
    pd.set_option('display.width', 250)
    fd = odir('figures')
    if os.path.exists(os.path.join(fd, 'claims.md')):
        print('=' * 110 + '\n1. CLAIMS TABLE (recomputed from saved per-person results; figures in results/nb_round9/figures)\n' + '=' * 110)
        print(open(os.path.join(fd, 'claims.md')).read())
    if os.path.exists(os.path.join(fd, 'draw_sensitivity.csv')):
        print('calibration-draw sensitivity (round 7, rigid - zero at k = 500, s4R4, 5 draws x 9 people):')
        print(pd.read_csv(os.path.join(fd, 'draw_sensitivity.csv')).to_string(index=False))
    # ---- 3
    e = _load(f'nb_round9/edit/edit_t*_{mon}.csv')
    if e is not None:
        e = _collapse_bases(e)
        print('\n' + '=' * 110 + '\n3. OUTPUT EDIT (seed-2026 decoder, population gallery, s4R4; rand / randvar averaged over 3 bases)\n' + '=' * 110)
        per = e.groupby(['k', 'method', 'rule', 'query', 'target']).top1.mean()
        T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3]).mean()
        for rule in ('cos', 'csls_batch'):
            print(f'\nrule {rule}, s4R4 (rows = method, columns = calibration images)')
            print(T.xs((rule, 's4R4'), level=('rule', 'query')).unstack('k').round(2).to_string())
        bv = e.groupby('method').basis_var.mean().dropna()
        print('\nshare of source population-response variance inside each basis: ' + ', '.join(f'{k} {v:.2f}' for k, v in bv.items()))
        print('chosen strength (most common, by method): ' + ', '.join(f'{k} {v}' for k, v in e.dropna(subset=['lam']).groupby('method').lam.agg(lambda s: s.mode().iloc[0]).items()))
        g = lambda k, mth, rule='cos': per.xs((k, mth, rule, 's4R4'), level=('k', 'method', 'rule', 'query'))  # noqa: E731
        for k in (200, 500):
            print(f'\ncomponent contributions at k = {k} (each query scored alone; batch CSLS in the table above)')
            _ps(g(k, 'shift'), g(k, 'zero'), f'shift - zero, k = {k}')
            _ps(g(k, 'scale'), g(k, 'shift'), f'scale (+ global scale) - shift, k = {k}')
            _ps(g(k, 'rot'), g(k, 'shift'), f'rotation - shift, k = {k}')
            _ps(g(k, 'rigid'), g(k, 'rot'), f'rigid - rotation (adds the scale), k = {k}')
            _ps(g(k, 'full'), g(k, 'rigid'), f'full linear - rigid, k = {k}')
            for r in RANKS:
                _ps(g(k, f'pca_{r}'), g(k, f'rand_{r}'), f'pca_{r} - rand_{r} (rank-matched), k = {k}')
                if r < 64:
                    _ps(g(k, f'pca_{r}'), g(k, f'randvar_{r}'), f'pca_{r} - randvar_{r} (inside the top-64 PCs), k = {k}')
                _ps(g(k, f'pca_{r}'), g(k, 'rigid'), f'pca_{r} - rigid, k = {k}')
        f = sorted(glob.glob(os.path.join(odir('edit'), f'edit_t*_{mon}_consistency.npz')))
        if f:
            zs = [np.load(x) for x in f]
            tg = np.concatenate([z['target'] for z in zs])
            kk = np.concatenate([z['k'] for z in zs])
            sh = np.concatenate([z['shift'] for z in zs])
            rd = np.concatenate([z['rot_dirs'] for z in zs])
            shn = sh / np.linalg.norm(sh, axis=1, keepdims=True)
            print('\nconsistency of the fitted edit (people 2-10)')
            for k in (200, 500):
                ii = np.nonzero((kk == k) & np.isin(tg, TEST))[0]
                if len(ii) < 2:
                    continue
                w_s, a_s, w_r, a_r = [], [], [], []
                for x in range(len(ii)):
                    for y in range(x + 1, len(ii)):
                        i, j = ii[x], ii[y]
                        cs_ = float(shn[i] @ shn[j])
                        ov = float(np.sum((rd[i].T @ rd[j]) ** 2) / rd.shape[2])
                        (w_s if tg[i] == tg[j] else a_s).append(cs_)
                        (w_r if tg[i] == tg[j] else a_r).append(ov)
                print(f'  k = {k}: shift-vector cosine within person (across draws) {np.mean(w_s):.2f}, across people {np.mean(a_s):.2f}; '
                      f'top-8 rotation-direction overlap within person {np.mean(w_r):.3f}, across people {np.mean(a_r):.3f} '
                      f'(random {rd.shape[2] / sh.shape[1]:.3f})')
    # ---- 4a
    b = _load(f'nb_round9/budget/budget_t*_{mon}.csv')
    if b is not None:
        print('\n' + '=' * 110 + '\n4a. EQUAL PRESENTATION BUDGETS (seed-2026 decoder, population gallery; x 4 spans days)\n' + '=' * 110)
        per = b.groupby(['budget', 'schedule', 'method', 'rule', 'query', 'target']).top1.mean()
        T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3, 4]).mean()
        print(T.xs('cos', level='rule').unstack('query')[QK].round(2).to_string())
        g = lambda bud, sch, mth, q: per.xs((bud, sch, mth, 'cos', q), level=('budget', 'schedule', 'method', 'rule', 'query'))  # noqa: E731
        for bud, splits in BUDGETS.items():
            (k4, _), (k2, _), (k1, _) = splits
            for q in ('s4R1', 's4R4'):
                for mth in ('rigid', 'shift'):
                    _ps(g(bud, f'{k1}x1', mth, q), g(bud, f'{k2}x2', mth, q), f'more images: {k1}x1 - {k2}x2, {mth}, {q}')
                    _ps(g(bud, f'{k4}x4', mth, q), g(bud, f'{k2}x2', mth, q), f'more repeats: {k4}x4 - {k2}x2, {mth}, {q}')
    # ---- 4b
    dn = _load(f'nb_round9/denoise/denoise_t*_{mon}.csv')
    if dn is not None:
        print('\n' + '=' * 110 + '\n4b. LINEAR CROSS-REPEAT DENOISER (seed-2026 decoder; source-fitted unless "target")\n' + '=' * 110)
        icols = [c for c in dn.columns if c.endswith('_val_r2') or c.endswith('_setting')]
        print('held-out-source R^2 to disjoint repeats (mean over folds): ' +
              ', '.join(f'{c} {dn.groupby("target")[c].first().mean():.3f}' if dn[c].dtype != object else f'{c} {dn[c].mode().iloc[0]}' for c in icols))
        print(f'target-calibrated denoiser, held-out R^2 (mean): {dn[dn.denoiser == "target"].target_val_r2.mean():.3f}')
        per = dn.groupby(['denoiser', 'method', 'rule', 'query', 'target']).top1.mean()
        T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3]).mean()
        for rule in ('cos', 'csls_batch'):
            print(f'\nrule {rule}')
            print(T.xs(rule, level='rule').unstack('query')[QK].round(2).to_string())
        g = lambda d_, mth, q, rule='cos': per.xs((d_, mth, rule, q), level=('denoiser', 'method', 'rule', 'query'))  # noqa: E731
        for q in ('s4R1', 's4R4'):
            for mth in ('zero', 'rigid'):
                _ps(g('ridge', mth, q), g('identity', mth, q), f'source denoiser - none, {mth}, {q}')
                _ps(g('ridge', mth, q), g('pca', mth, q), f'source denoiser - PCA control, {mth}, {q}')
                _ps(g('ridge', mth, q), g('shuffled', mth, q), f'source denoiser - shuffled-pair control, {mth}, {q}')
                _ps(g('target', mth, q), g('identity', mth, q), f'target-calibrated denoiser - none, {mth}, {q}')
                _ps(g('ridge', mth, q), g('ridge_raw', mth, q), f'variance-matched - raw ridge output, {mth}, {q}')
    # ---- 2
    en = _load(f'nb_round9/encoders/enc_s*_t*_{mon}.csv')
    for fn_ in ('profile.csv', 'lr_choice.csv'):
        p = os.path.join(odir('encoders'), fn_)
        if os.path.exists(p):
            print(f'\n{fn_}:\n' + pd.read_csv(p).to_string(index=False))
    if en is not None:
        print('\n' + '=' * 110 + '\n2. ENCODER x VISUAL TARGET (people 2-10; zero = output offset + population gallery)\n' + '=' * 110)
        per = en.groupby(['encoder', 'visual_target', 'method', 'rule', 'query', 'target']).top1.mean()
        T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3, 4]).mean()
        for rule in ('cos', 'csls_batch'):
            print(f'\nrule {rule}')
            print(T.xs(rule, level='rule').unstack('query')[QK].unstack('method').round(2).to_string())
        print(f'seeds: {sorted(en.seed.unique())}; folds: {sorted(en.target.unique())}')
        g = lambda enc, tgt, mth, q='s4R4': per.xs((enc, tgt, mth, 'cos', q), level=('encoder', 'visual_target', 'method', 'rule', 'query'))  # noqa: E731
        for enc in ENCS:
            for tgt in TGTS:
                if (enc, tgt) == ('eegproject', 'aug'):
                    continue
                for mth in ('plain', 'zero', 'rigid'):
                    _ps(g(enc, tgt, mth), g('eegproject', 'aug', mth), f'{enc}/{tgt} - eegproject/aug (current), {mth}, s4R4, cos')
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['figures', 'edit', 'budget', 'denoise', 'feats', 'profile', 'tune', 'encoders', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--seeds', type=int, nargs='+', default=[2025])
    p.add_argument('--reps_dir', default='/content/nb_reps')
    p.add_argument('--montage', default='posterior17')
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'figures':
        return figures(a.montage)
    if a.cmd == 'feats':
        return feats()
    if a.cmd == 'profile':
        return profile(a.montage)
    if a.cmd == 'tune':
        return tune(a.montage, a.seeds[0])
    if a.cmd == 'encoders':
        return encoders(a.targets, a.montage, a.seeds)
    fn = {'edit': edit_target, 'budget': budget_target, 'denoise': denoise_target}.get(a.cmd)
    if fn:
        for B in a.targets:
            print(f'[{a.cmd}] target {B}', flush=True)
            fn(B, a.montage)
        return
    report(a.montage)


if __name__ == '__main__':
    main()
