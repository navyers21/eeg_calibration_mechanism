"""Round 14: is the gap on 4-presentation, later-day queries a calibration problem or a gallery problem?
Two experiments on the FROZEN seed-2026 repaug NeuroBridge decoders (no decoder training).

    python scripts/nb_round14.py selftest                       # synthetic checks (no data)
    python scripts/nb_round14.py feats   --config C             # 14B: RN50 pooled multilevel + layer-3 4x4 grid features
    python scripts/nb_round14.py a       --config C [--targets] # 14A on people 1-10 (1 = dry run only, nothing is tuned)
    python scripts/nb_round14.py b       --config C [--targets] # 14B on people 1-10 (selection is per target, sources only)
    python scripts/nb_round14.py report  --config C

Why. Round 13: label-free recovery from 100 earlier-day images x 2 presentations tied re-centering (+0.21, CI -0.16 to
+0.58), and supplying correct pairs added little (+0.32) with that estimator. Two leads remain. (1) Transductive SCORE
gained +8.2 at 80 presentations but lost at 4; it fits moments, landmarks, rotation AND final CSLS on the evaluation
batch, so it cannot say whether clean data gives a better REUSABLE calibration. (2) Round 8: with the same queries, the
measured source-population gallery scored 19.14% vs 9.20% for the predicted one: gallery construction matters, and that
comparison says nothing about what retraining the query encoder would buy. Encoder training (~90 runs) is held until
these two answers are in.

14A. CLEAN SUPPORT x NOISY QUERY (does clean calibration transfer to unseen 4-presentation queries on another day?)
  The 200 official test identities are split into two fixed halves A / B (seed 14). Calibration uses session-3 EEG of
  the A images and their unordered pool of 100 targets; ONLY session-4 queries of B images are scored, against all 200
  candidates (200-way, chance 0.5%); then A and B swap. Every scored identity is absent from its calibration bank; the
  fitted identities stay in the gallery as distractors. Strictly disjoint 100-way scores (B candidates only) are reported
  as a sensitivity analysis. The official test stimuli are used here as a DIAGNOSTIC; nothing is tuned on them (the
  Round 13 variant was locked on person 1's training images).
  Factors: support presentations per image R_s in {4, 20} (the 4 nested inside the 20; 3 draws of 4) x query
  presentations R_q in {4, 20} (Round 4's session-4 draws: 3 at R_q = 4, 1 at R_q = 20). 100 support identities always.
  Embeddings: label-free layer-1 offset from the support trials (as in rounds 10-13), output z. Gallery = the predicted
  population gallery (Round 10/13 Ganc). Recovery target = measured source-population responses to the support images
  (Round 13's 'pop' target; for test images these are the source people's 80-presentation averages).
  Rows (same queries, same gallery, same offset in every row):
    base          re-centered decoder, no output change (matched baseline)
    center        + output shift fitted on the support (support EEG mean -> target-pool mean): session statistics only
    rec           Round 13 locked recovery (mm0|pop|rho1|q1|s2|ag1), inferred pairs, each query scored alone (cosine)
    rec_bcsls     the same frozen rotation, final scoring by batch CSLS over the query batch (transductive SCORING only)
    rec_true      same landmark rows and weights, true partners;  rec_all: all 100 true pairs, same shrinkage
    sup_rot       supervised rotation with CV-chosen shrinkage (Round 11 'rot'): labelled reference, explicit rule
    rec_cv        EXPLORATORY: the inferred landmark pairs refit with CV-chosen shrinkage (label-free); true_cv: same
                  rows with true partners. With rec_all vs sup_rot these separate correspondence from shrinkage
                  (Round 13's +1.01 oracle used the locked rho = 1; Round 11's CV rotation reached +1.79)
    trans         published transductive SCORE on the query batch (everything fitted on it): reference only
  Agreement filter: support repeats split into two halves; cosine between half-average outputs (label-free).
  PRIMARY (R_q = 4, people 2-10, halves / draws averaged within person, Wilcoxon, Holm over the two):
    P1  rec(R_s = 20) - rec(R_s = 4)            P2  rec(R_s = 20) - base(R_s = 20)
  Cost: R_s = 20 is 2,000 presentations (~6.7 min at 5 Hz) vs 400 (~1.3 min); printed with every gain.
  Decision (printed; practical threshold +3 points over the matched baseline at R_q = 4, R_s = 20, set before running):
    rec_true and rec both >= +3      -> one-time clean calibration is worth pursuing
    rec_true >= +3, rec < +3         -> landmark inference deserves investment (Round 13 did not show this at R_s = 2)
    neither at R_q = 4, gains at 20  -> only cleaner queries help: closes clean support for THIS estimator / protocol
    only rec_bcsls >= +3             -> the gain belongs to transductive retrieval, not a transferable correction

14B. A BETTER DEPLOYABLE GALLERY (can image features recover what makes measured population templates better?)
  Round 8's exact evaluator: R6 context, seed-2026 repaug, k = 500 earlier-day draws (3), label-free offset from the
  calibration trials, session-4 queries; methods zero (offset only, PRIMARY) and rigid k = 500 to each gallery's own
  representation of the calibration images; rules cos (PRIMARY) and csls_batch. Gate: G0 and G* reproduce Round 8.
    G0     current predicted gallery (ridge, final fused CLIP RN50 -> source-mean decoder outputs; Round 8 'anchor')
    G1     pooled multilevel: final features + globally pooled RN50 layer-2 / layer-3 / layer-4 blocks, each block scaled
           to the final block's total variance times a weight in {0, 0.5, 1} (27 settings; all-zero = G0's model)
    G2     G1 + spatial residual: RN50 layer-3 on a 4x4 grid (cell positions kept), channels reduced to 64 by PCA fitted
           on source-training images (-> 16 x 64 = 1,024 dims); ridge to the CONCEPT-CROSS-FITTED residual of G1 (5 folds
           by concept: each training concept's G1 prediction comes from a fit that excluded it); anchor =
           normalize(G1 + s * residual), s in {0.5, 1, 1.5}
    G2shuf dimension-matched control: grid cells independently permuted within each image (independent across images),
           same PCA, s locked to G2's
    G*     measured source-population responses to the test images (PRIVILEGED headroom; never used for selection)
  Selection (per target, sources and training concepts only, no target data): 2 source people held out as pseudo-new
  users; predictors refit on the other 7 people's mean responses to training-pool images; scored by 200-way retrieval of
  the held-out people's 4-presentation outputs on the 200 held-out training concepts (10 image sets). G1 weights, then s,
  chosen by that retrieval (ties -> simpler); final galleries refit on all 9 source people. Ridge strength by GCV.
  PRIMARY: G2 - G0, zero labels, s4R4, cos, people 2-10, Wilcoxon. Secondary (Holm over three): G2 - G1, G2 - G2shuf,
  G1 - G0. Locked practical targets: +3 points (G0 9.20 -> 12.20) and +5 (-> 14.20), about 30% and 50% of the observed
  9.94-point gap to G*; the gap need not be recoverable from images. Diagnostics: anchor cosine to G*, correct-image
  margin over the hardest distractor, and how often G2 fixes the errors G* fixes.

Next (Round 15, only if justified): train a query encoder toward a FIXED teacher space; score baseline and new encoder
against both the predicted and the measured gallery before any multi-seed sweep.
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
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]

MON = 'posterior17'
SEED = 2026
TEST = list(range(2, 11))
PEOPLE = list(range(1, 11))
HALF_SEED = 14
RS = (4, 20)
RQ = (4, 20)
N_SDRAW = 3
THRESH = 3.0
# 14B
K = 500
W_GRID = (0.0, 0.5, 1.0)
S_GRID = (0.5, 1.0, 1.5)
N_PC = 64
GRID = 4
N_FOLD = 5
N_VAL = 2
QK = ['s4R1', 's4R4', 's4R20', 'bench']
POOL_BLOCKS = {'l1': (0, 256), 'l2': (256, 768), 'l3': (768, 1792), 'l4': (1792, 3840)}
G1_BLOCKS = ('l2', 'l3', 'l4')
GALS = ['G0', 'G1', 'G2', 'G2shuf', 'Gstar']


def l2n(x):
    return F.normalize(x.float(), dim=-1)


# ---------------------------------------------------------------------------------------------------------------
# shared pieces (pure torch / numpy; exercised by selftest)
# ---------------------------------------------------------------------------------------------------------------
def halves():
    p = np.random.RandomState(HALF_SEED).permutation(200)
    return np.sort(p[:100]), np.sort(p[100:])


def score_rows(S, tgt, sub=None):
    """S: (n, G) scores, tgt: (n,) correct column per row. Returns top-1 / top-5 (%), MRR, mean margin of the correct
    candidate over the hardest other one, per-row correctness. sub: optional candidate subset (columns)."""
    tgt = torch.as_tensor(np.asarray(tgt), device=S.device)
    if sub is not None:
        sub = torch.as_tensor(np.asarray(sub), device=S.device)
        pos = {int(c): j for j, c in enumerate(sub.tolist())}
        S = S[:, sub]
        tgt = torch.as_tensor([pos[int(t)] for t in tgt.tolist()], device=S.device)
    ar = torch.arange(len(S), device=S.device)
    c = S[ar, tgt]
    rank = (S > c[:, None]).sum(1)
    other = S.clone()
    other[ar, tgt] = -float('inf')
    return dict(top1=100 * float((rank == 0).float().mean()), top5=100 * float((rank < 5).float().mean()),
                mrr=float((1.0 / (rank + 1).float()).mean()),
                margin=float((c - other.max(1).values).mean())), (rank == 0).cpu().numpy()


def nested_reps(n_img, n_have, r, draw):
    """Per image, r of its n_have session repeats; draw 0..: independent subsets of the same pool (r = n_have: all)."""
    if r >= n_have:
        return np.tile(np.arange(n_have), (n_img, 1))
    rng = np.random.RandomState(4000 + 10 * draw)
    return np.stack([np.sort(rng.choice(n_have, r, replace=False)) for _ in range(n_img)])


def ridge_fit(X, Ys, rel=torch.logspace(-4, 2, 13)):
    """Ridge X -> Y for several targets sharing one eigendecomposition; alpha per target by generalized CV on the
    same grid as nb_next.ridge_svd (rel x mean squared singular value). Returns [(B, x mean, y mean)] per target."""
    xm = X.mean(0)
    Xc = (X - xm).double()
    n = len(X)
    e, V = torch.linalg.eigh(Xc.T @ Xc)
    keep = e > e.max() * 1e-10
    e, V = e[keep], V[:, keep]
    S = e.sqrt()
    U = (Xc @ V) / S
    out = []
    for Y in Ys:
        ym = Y.mean(0)
        Yc = (Y - ym).double()
        UtY = U.T @ Yc
        u2 = (UtY ** 2).sum(1)
        tot = float((Yc ** 2).sum())
        best, best_a = None, None
        for a in rel.to(e.device).double() * e.mean():
            f = e / (e + a)
            df = float(f.sum())
            if df >= n - 1:
                continue
            g = (tot - float(((2 * f - f ** 2) * u2).sum())) / (1 - df / n) ** 2
            if best is None or g < best:
                best, best_a = g, a
        if best_a is None:
            best_a = rel[-1].to(e.device).double() * e.mean()
        B = V @ ((S / (e + best_a))[:, None] * UtY)
        out.append((B.float(), xm, ym))
    return out


def apply_ridge(fit, X):
    B, xm, ym = fit
    return (X - xm) @ B + ym


def cell_perm(n, offset=0):
    """Independent permutation of the GRID*GRID cells for each image (seeded by the image's global index)."""
    return np.stack([np.random.RandomState((100003 * (offset + i) + 7) % 2 ** 32).permutation(GRID * GRID) for i in range(n)])


def concept_folds(labels, k=N_FOLD, seed=0):
    """Fold id per row; every concept's images share a fold."""
    u = np.unique(labels)
    f = np.random.RandomState(seed).permutation(len(u)) % k
    m = dict(zip(u.tolist(), f.tolist()))
    return np.array([m[int(c)] for c in labels])


def scale_blocks(blocks, w, ref_var):
    """blocks: name -> (n, d) z-scored features; each kept block scaled to total variance w^2 * ref_var."""
    parts = []
    for name, Z in blocks.items():
        if w.get(name, 0.0) > 0:
            parts.append(Z * (w[name] * (ref_var / Z.shape[1]) ** 0.5))
    return parts


def selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f'  [{"ok" if cond else "FAIL"}] {name} {detail}')
    A, B_ = halves()
    check('halves disjoint and complete', len(set(A) & set(B_)) == 0 and len(set(A) | set(B_)) == 200)
    # scoring on a 200-way gallery with arbitrary row -> column map, and the 100-way subset
    S = torch.randn(100, 200)
    tgt = B_.copy()
    S[torch.arange(100), torch.as_tensor(tgt)] += 10
    m, corr = score_rows(S, tgt)
    m2, _ = score_rows(S, tgt, sub=B_)
    check('score_rows uses the given targets', m['top1'] == 100 and m2['top1'] == 100 and corr.all())
    S2 = S.clone()
    S2[:, torch.as_tensor(A)] += 100                                          # distractors from the other half win
    check('200-way keeps the calibration half as distractors', score_rows(S2, tgt)[0]['top1'] == 0
          and score_rows(S2, tgt, sub=B_)[0]['top1'] == 100)
    # nested support subsets
    r4 = nested_reps(5, 20, 4, 0)
    check('4-of-20 subsets valid', r4.shape == (5, 4) and r4.max() < 20 and all(len(set(x)) == 4 for x in r4))
    check('R_s = 20 uses every repeat', (nested_reps(3, 20, 20, 0) == np.arange(20)).all())
    check('support draws differ', not (nested_reps(5, 20, 4, 0) == nested_reps(5, 20, 4, 1)).all())
    # ridge with GCV equals an explicit SVD implementation (nb_next.ridge_svd's algorithm)
    X = torch.randn(300, 40)
    Y = X @ torch.randn(40, 8) + 0.5 * torch.randn(300, 8)

    def ref(X, Y, rel=torch.logspace(-4, 2, 13)):
        xm, ym = X.mean(0), Y.mean(0)
        U, Sv, Vh = torch.linalg.svd((X - xm).double(), full_matrices=False)
        UtY = U.T @ (Y - ym).double()
        s2 = Sv ** 2
        best = None
        for a in rel.double() * s2.mean():
            f = s2 / (s2 + a)
            df = float(f.sum())
            g = float(((Y - ym).double() - U @ (f[:, None] * UtY)).pow(2).sum()) / (1 - df / len(X)) ** 2
            if best is None or g < best[0]:
                best = (g, a)
        return (Vh.T @ ((Sv / (s2 + best[1]))[:, None] * UtY)).float()
    Bm = ridge_fit(X, [Y])[0][0]
    check('ridge_fit == SVD ridge with GCV', torch.allclose(Bm, ref(X, Y), atol=1e-4),
          f'(max diff {float((Bm - ref(X, Y)).abs().max()):.1e})')
    # cell permutation control: independent across images, a permutation each, deterministic
    P = cell_perm(50)
    check('cell permutations valid and independent', all(sorted(p) == list(range(16)) for p in P)
          and len({tuple(p) for p in P}) > 45 and (cell_perm(50) == P).all())
    # a shared permutation is invertible by a linear map; independent ones are not (why the control is per image)
    n, d = 2000, 16
    Zc = torch.randn(n, d)
    yv = Zc[:, 0:1].clone()
    shared = Zc[:, torch.as_tensor(P[0])]
    indep = torch.stack([Zc[i, torch.as_tensor(P[i % 50])] for i in range(n)])

    def r2(Xf):
        fit = ridge_fit(Xf, [yv])[0]
        return 1 - float((apply_ridge(fit, Xf) - yv).pow(2).mean() / yv.var())
    check('shared permutation is recoverable, per-image permutation is not', r2(shared) > 0.99 and r2(indep) < 0.3,
          f'(R2 {r2(shared):.2f} vs {r2(indep):.2f})')
    # concept folds keep concepts together
    lab = np.repeat(np.arange(100), 10)
    fo = concept_folds(lab)
    check('concept folds group whole concepts', all(len(set(fo[lab == c])) == 1 for c in range(100))
          and len(set(fo)) == N_FOLD)
    # cross-fitted residual: no row's residual uses its own concept
    Xr = torch.randn(1000, 12)
    Yr = Xr @ torch.randn(12, 4)
    lab = np.repeat(np.arange(100), 10)
    fo = concept_folds(lab)
    oof = torch.zeros_like(Yr)
    for f in range(N_FOLD):
        tr_, te_ = np.nonzero(fo != f)[0], np.nonzero(fo == f)[0]
        oof[te_] = apply_ridge(ridge_fit(Xr[tr_], [Yr[tr_]])[0], Xr[te_])
    check('cross-fitted predictions are out of fold', float((oof - Yr).abs().max()) < 1e-2)
    # block scaling: a zero weight drops the block, kept blocks carry the requested variance
    blocks = {'a': torch.randn(500, 30), 'b': torch.randn(500, 60)}
    parts = scale_blocks(blocks, {'a': 0.0, 'b': 0.5}, ref_var=10.0)
    tv = float(parts[0].var(0).sum())
    check('block scaling', len(parts) == 1 and abs(tv - 0.25 * 10.0) < 0.3, f'(total var {tv:.2f})')
    print('SELFTEST PASSED' if ok else 'SELFTEST FAILED')
    return ok


# ---------------------------------------------------------------------------------------------------------------
# data-dependent modules (imported after setup)
# ---------------------------------------------------------------------------------------------------------------
def _m():
    from eegalign import nb as N
    import nb_round3 as R
    import nb_round4 as R4m
    import nb_round5 as R5m
    import nb_round6 as R6m
    import nb_round9 as R9
    import nb_round10 as R10
    import nb_round11 as R11
    import nb_round13 as R13
    import nb_interp as I
    return dict(N=N, R=R, R4m=R4m, R5m=R5m, R6m=R6m, R9=R9, R10=R10, R11=R11, R13=R13, I=I)


def dev():
    return _m()['N'].ENV['device']


def odir(sub=''):
    d = os.path.join(_m()['N'].ENV['results_dir'], 'nb_round14', sub)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------------------------
# 14A
# ---------------------------------------------------------------------------------------------------------------
def locked_variant():
    M = _m()
    p = os.path.join(M['N'].ENV['results_dir'], 'nb_round13', 'locked_settings.json')
    assert os.path.exists(p), f'{p} missing: Round 13 must be locked first'
    s = json.load(open(p))
    return s['variant'], s


class TargetA:
    """Round 13's frozen embeddings for one person plus per-repetition test EEG and source responses to test images."""

    def __init__(self, B):
        M = _m()
        self.M = M
        self.T = M['R13'].Target(B)                      # decoder, source stats, Ganc, Round 4 query draws
        T = self.T
        self.B = B
        tt = M['N'].subject_data(B, MON, keys=('test_trials',))['test_trials']
        te = M['R4m'].load_ses(B)['test']
        self.s3 = [np.nonzero(te[i] == 3)[0] for i in range(200)]
        self.n3 = min(len(x) for x in self.s3)
        assert self.n3 >= max(RS), f'only {self.n3} session-3 repeats per test image'
        self.tt3 = np.stack([tt[i, self.s3[i][:self.n3]] for i in range(200)]).astype(np.float32)   # (200, n3, C, T)
        del tt
        Zt = []
        for s in PEOPLE:
            if s == B:
                continue
            Zt.append(M['R10'].z_of(T.m, M['N'].subject_data(s, MON, keys=('test',))['test'].astype(np.float32)))
        self.Ysrc_test = torch.stack(Zt).mean(0)        # measured source-population responses to the 200 test images

    def support(self, ims, rs, draw):
        M, T = self.M, self.T
        sel = nested_reps(len(ims), self.n3, rs, draw)
        x = self.tt3[ims[:, None], sel]                                           # (100, rs, C, T)
        d1 = T.mS - M['R10'].h1_mean(T.m, x.reshape(-1, *x.shape[2:]))           # label-free, support trials only
        Xs = M['R10'].z_of(T.m, x.mean(1), d1)
        h = rs // 2
        agree = (l2n(M['R10'].z_of(T.m, x[:, :h].mean(1), d1)) * l2n(M['R10'].z_of(T.m, x[:, h:].mean(1), d1))).sum(1)
        Zq = {rq: [M['R10'].z_of(T.m, q, d1) for q in T.Q[f's4R{rq}']] for rq in RQ}
        return Xs, agree, Zq


def run_a(B):
    out = os.path.join(odir('a'), f'r14a_t{B:02d}.csv')
    if os.path.exists(out):
        print(f'[14A] t{B}: done')
        return
    t0 = time.time()
    M = _m()
    R13, R9, R11 = M['R13'], M['R9'], M['R11']
    v, locked = locked_variant()
    TA = TargetA(B)
    T = TA.T
    G = T.Ganc
    # gate: Round 13's no-calibration population-gallery cell (k = 100, draw 0, s4R4) from the same embeddings
    sup = T.support(100, 0)
    mine = [float(R13.metrics(l2n(z) @ l2n(G).T)[0]['top1']) for z in sup['Zq']['s4R4']]
    gmsg = 'Round 13 file not found'
    r13 = os.path.join(M['N'].ENV['results_dir'], 'nb_round13', 'eval', f'r13_t{B:02d}.csv')
    if os.path.exists(r13):
        r = pd.read_csv(r13)
        r = r[(r.k == 100) & (r.draw == 0) & (r.method == 'raw_pop') & (r['query'] == 's4R4')].sort_values('qdraw')
        gmsg = f'max |diff| vs Round 13 raw_pop (k 100, draw 0, s4R4): {np.abs(np.array(mine) - r.top1.values).max():.2e}'
    del sup
    A, Bh = halves()
    eye = torch.eye(G.shape[1], device=G.device)
    rows, lm = [], []
    for hname, fit_ims, score_ims in (('A2B', A, Bh), ('B2A', Bh, A)):
        Ys = TA.Ysrc_test[torch.as_tensor(fit_ims, device=G.device)]
        for rs in RS:
            for draw in range(N_SDRAW if rs < TA.n3 else 1):
                Xs, agree, Zq = TA.support(fit_ims, rs, draw)
                n = len(Xs)
                bank = R13.recover(Xs, Ys, torch.Generator().manual_seed(140000 + 1000 * B + 10 * rs + draw), v, agree)
                prec = float((bank['eeg'] == bank['img']).float().mean()) if len(bank['w']) else np.nan
                lm.append(dict(target=B, half=hname, rs=rs, draw=draw, m=len(bank['w']), precision=prec))
                mm = bank['mm']
                rots = {}
                for cname, (be, bi, bw) in {'rec': (bank['eeg'], bank['img'], bank['w']),
                                            'rec_true': (bank['eeg'], bank['eeg'], bank['w']),
                                            'rec_all': (torch.arange(n, device=Xs.device),) * 2
                                            + (torch.ones(n, device=Xs.device),)}.items():
                    if len(be) >= 2:
                        rots[cname] = R13.fit_rotation(R13.prep(Xs[be], mm), Ys[bi], bw, v['rho'])[:3]
                lam, _ = R9.cv_pick(lambda a, b, l: R11.e_rot_prior(a, b, l, eye), Xs, Ys, R11.REL, seed=draw)
                f_sup = R11.e_rot_prior(Xs, Ys, lam, eye)
                # exploratory: the same landmark rows refit with CV-chosen shrinkage (Round 11 rotation family);
                # rec_cv is label-free (CV over the inferred pseudo-pairs), true_cv uses the true partners
                cvf = {}
                for cname, (be, bi) in (('rec_cv', (bank['eeg'], bank['img'])), ('true_cv', (bank['eeg'], bank['eeg']))):
                    if len(be) >= 10:
                        Xb, Yb = R13.prep(Xs[be], mm), Ys[bi]
                        lc, _ = R9.cv_pick(lambda a, b, l: R11.e_rot_prior(a, b, l, eye), Xb, Yb, R11.REL, seed=draw)
                        cvf[cname] = R11.e_rot_prior(Xb, Yb, lc, eye)
                shift = Ys.mean(0) - Xs.mean(0)
                for rq in RQ:
                    for qd, zall in enumerate(Zq[rq]):
                        z = zall[torch.as_tensor(score_ims, device=zall.device)]
                        S = {'base': l2n(z) @ l2n(G).T, 'center': l2n(z + shift) @ l2n(G).T,
                             'sup_rot': l2n(f_sup(z)) @ l2n(G).T, 'trans': R13.transductive_score(z, G)[0]}
                        for cname, (Rm, mx, my) in rots.items():
                            Qr = R13.apply_rotation(R13.prep(z, mm), Rm, mx, my)
                            S[cname] = l2n(Qr) @ l2n(G).T
                            if cname == 'rec':
                                S['rec_bcsls'] = R13.csls(Qr, G)
                        for cname, fcv in cvf.items():
                            S[cname] = l2n(fcv(R13.prep(z, mm))) @ l2n(G).T
                        for meth, Sm in S.items():
                            m200, _ = score_rows(Sm, score_ims)
                            m100, _ = score_rows(Sm, score_ims, sub=score_ims)
                            rows.append(dict(target=B, half=hname, rs=rs, draw=draw, rq=rq, qdraw=qd, method=meth,
                                             top1=m200['top1'], top5=m200['top5'], mrr=m200['mrr'],
                                             margin=m200['margin'], top1_100=m100['top1'], lam_sup=lam))
        print(f'   t{B} {hname} done ({time.time() - t0:.0f}s)', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(lm).to_csv(out.replace('.csv', '_landmarks.csv'), index=False)
    meta = dict(target=B, variant=locked.get('variant_name'), rho=v['rho'], n3=TA.n3, gate=gmsg,
                minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(out.replace('.csv', '_meta.json'), 'w'), indent=1, default=str)
    print(f'[14A] t{B}: {meta["minutes"]} min | gate: {gmsg}', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# 14B features
# ---------------------------------------------------------------------------------------------------------------
def feat_paths():
    d = _m()['N'].ENV['feat_dir']
    return {s: (os.path.join(d, f'image_{s}_rn50_pool.npy'), os.path.join(d, f'image_{s}_rn50_l3g{GRID}.npy'))
            for s in ('train', 'test')}


def feats(local='/content/things_images', hf_repo='Haitao999/things-eeg'):
    """Globally pooled RN50 layer 1-4 outputs (3,840-d) and the layer-3 output average-pooled to a GRID x GRID grid
    (cells x 1,024, float16) for every THINGS-EEG2 image, NeuroBridge order; order checked as in Round 9."""
    M = _m()
    N = M['N']
    import open_clip
    from PIL import Image
    paths = feat_paths()
    if all(os.path.exists(p) for v in paths.values() for p in v):
        print('14B features exist:', paths)
        return
    root = next((os.path.join(r, d) for r, ds, _ in os.walk(local) for d in ds if d == 'Image_set_Resize'), None)
    if root is None:
        from huggingface_hub import list_repo_files, snapshot_download
        import zipfile
        import tarfile
        files = list_repo_files(hf_repo, repo_type='dataset')
        hit = [f for f in files if 'Image_set_Resize' in f]
        assert hit, f'no Image_set_Resize in {hf_repo}'
        pre = hit[0][:hit[0].index('Image_set_Resize')]
        snapshot_download(hf_repo, repo_type='dataset', local_dir=local, allow_patterns=[f'{pre}Image_set_Resize*'],
                          max_workers=4)
        for arc in glob.glob(f'{local}/**/Image_set_Resize*.zip', recursive=True) + \
                glob.glob(f'{local}/**/Image_set_Resize*.tar*', recursive=True):
            (zipfile.ZipFile(arc) if arc.endswith('.zip') else tarfile.open(arc)).extractall(os.path.dirname(arc))
        root = next((os.path.join(r, d) for r, ds, _ in os.walk(local) for d in ds if d == 'Image_set_Resize'), None)
    assert root, 'Image_set_Resize folder not found'
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai', device=dev())
    model.eval()
    vis = model.visual
    for split, sub, n_exp in (('test', 'test_images', 200), ('train', 'train_images', 16540)):
        d0 = os.path.join(root, sub)
        ps = [os.path.join(d0, cd, f) for cd in sorted(os.listdir(d0)) for f in sorted(os.listdir(os.path.join(d0, cd)))]
        assert len(ps) == n_exp, f'{split}: {len(ps)} images, expected {n_exp}'
        pooled, grid, fins = [], [], []
        with torch.no_grad():
            for i in range(0, len(ps), 128):
                x = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in ps[i:i + 128]]).to(dev())
                x = x.type(vis.conv1.weight.dtype)
                h = vis.act3(vis.bn3(vis.conv3(vis.act2(vis.bn2(vis.conv2(vis.act1(vis.bn1(vis.conv1(x)))))))))
                h = vis.avgpool(h)
                h1 = vis.layer1(h)
                h2 = vis.layer2(h1)
                h3 = vis.layer3(h2)
                h4 = vis.layer4(h3)
                pooled.append(torch.cat([a.mean((2, 3)) for a in (h1, h2, h3, h4)], 1).float().cpu())
                grid.append(F.adaptive_avg_pool2d(h3, GRID).flatten(2).transpose(1, 2).half().cpu())   # (b, cells, 1024)
                fins.append(vis.attnpool(h4).float().cpu())
        fin = torch.cat(fins)
        ref = N.features(split, 'clean')
        sub_ = torch.as_tensor(np.random.RandomState(0).choice(len(fin), min(200, len(fin)), replace=False))
        match = float((l2n(fin[sub_]) @ l2n(ref[sub_]).T).argmax(1).eq(torch.arange(len(sub_))).float().mean())
        print(f'  {split}: order check {100 * match:.1f}% of 200 sampled images match their own row', flush=True)
        assert match > 0.9, 'image order or preprocessing does not match NeuroBridge: do not use these features'
        np.save(paths[split][0], torch.cat(pooled).numpy().astype(np.float32))
        np.save(paths[split][1], torch.cat(grid).numpy())
    print('saved', paths)


# ---------------------------------------------------------------------------------------------------------------
# 14B galleries
# ---------------------------------------------------------------------------------------------------------------
class Feats:
    """Image features for the gallery predictors, all on the device: final (fused aug), pooled blocks (z-scored with
    training-pool statistics), spatial grid (PCA fitted on training-pool cells), and its cell-shuffled control."""

    def __init__(self, c):
        paths = feat_paths()
        for v in paths.values():
            for p in v:
                assert os.path.exists(p), f'{p} missing: run `nb_round14.py feats` first'
        d = dev()
        self.fin = {'train': c.ftr, 'test': c.fte}
        self.ref_var = float(c.ftr[c.pool].var(0).sum())
        pooled = {s: torch.from_numpy(np.load(paths[s][0])).to(d) for s in ('train', 'test')}
        self.blocks = {s: {} for s in ('train', 'test')}
        for name in G1_BLOCKS:
            a, b = POOL_BLOCKS[name]
            mu = pooled['train'][c.pool, a:b].mean(0)
            sd = pooled['train'][c.pool, a:b].std(0) + 1e-6
            for s in ('train', 'test'):
                self.blocks[s][name] = (pooled[s][:, a:b] - mu) / sd
        del pooled
        g = {s: np.load(paths[s][1], mmap_mode='r') for s in ('train', 'test')}
        samp = np.sort(np.random.RandomState(0).choice(c.pool, min(3000, len(c.pool)), replace=False))
        cells = torch.from_numpy(np.asarray(g['train'][samp], np.float32)).to(d).reshape(-1, g['train'].shape[2])
        self.cmu = cells.mean(0)
        _, V = torch.linalg.eigh(torch.cov((cells - self.cmu).T))
        self.Vpc = V[:, -N_PC:]
        del cells
        self.sp, self.sp_shuf = {}, {}
        for s, off in (('train', 0), ('test', 10 ** 6)):
            out, outs = [], []
            P = cell_perm(len(g[s]), off)
            for i in range(0, len(g[s]), 2048):
                x = torch.from_numpy(np.asarray(g[s][i:i + 2048], np.float32)).to(d)
                z = (x - self.cmu) @ self.Vpc                                  # (b, cells, N_PC), positions kept
                out.append(z.flatten(1))
                pi = torch.as_tensor(P[i:i + 2048], device=d)
                outs.append(torch.gather(z, 1, pi[:, :, None].expand(-1, -1, N_PC)).flatten(1))
            self.sp[s], self.sp_shuf[s] = torch.cat(out), torch.cat(outs)

    def g1x(self, s, w, idx=None):
        f = self.fin[s] if idx is None else self.fin[s][idx]
        bl = {k: (v if idx is None else v[idx]) for k, v in self.blocks[s].items()}
        return torch.cat([f] + scale_blocks(bl, w, self.ref_var), 1)


def wkey(w):
    return '|'.join(f'{k}{w[k]:g}' for k in G1_BLOCKS)


def fit_gallery(Fe, pool, Ytargets, w, labels=None, spatial=None):
    """Fit G1 (blocks w) on pool images to each target; optionally the spatial residual head on concept-cross-fitted
    G1 residuals. Returns a list (per target) of dicts: g1 fit, head fits {kind: fit}."""
    ip = torch.as_tensor(pool, device=dev())
    Xp = Fe.g1x('train', w, ip)
    g1 = ridge_fit(Xp, Ytargets)
    out = [dict(g1=f) for f in g1]
    if spatial:
        fo = concept_folds(labels)
        oof = [torch.zeros_like(Y) for Y in Ytargets]
        for f in range(N_FOLD):
            tr_ = torch.as_tensor(np.nonzero(fo != f)[0], device=dev())
            te_ = torch.as_tensor(np.nonzero(fo == f)[0], device=dev())
            fits = ridge_fit(Xp[tr_], [Y[tr_] for Y in Ytargets])
            for j, ft in enumerate(fits):
                oof[j][te_] = apply_ridge(ft, Xp[te_])
        for kind in spatial:
            src = Fe.sp if kind == 'real' else Fe.sp_shuf
            heads = ridge_fit(src['train'][ip], [Y - o for Y, o in zip(Ytargets, oof)])
            for j, h in enumerate(heads):
                out[j].setdefault('head', {})[kind] = h
    return out


def gallery_fn(Fe, fit, w, kind=None, s=1.0):
    """Returns fn(split, idx) -> normalized anchors."""
    def fn(split, idx=None):
        x = Fe.g1x(split, w, idx)
        y = apply_ridge(fit['g1'], x)
        if kind is not None:
            sp = (Fe.sp if kind == 'real' else Fe.sp_shuf)[split]
            y = y + s * apply_ridge(fit['head'][kind], sp if idx is None else sp[idx])
        return l2n(y)
    return fn


def val_score(fn, c, vq):
    """200-way retrieval of held-out source people's outputs on held-concept images (10 image sets)."""
    acc = []
    for j in range(10):
        rows = slice(200 * j, 200 * (j + 1))
        Gv = fn('train', torch.as_tensor(c.held[rows], device=dev()))
        for Zq in vq:
            acc.append(score_rows(l2n(Zq[rows]) @ Gv.T, np.arange(200))[0]['top1'])
    return float(np.mean(acc))


def build_b(c):
    """Selection on source people only, then final galleries on all 9. Returns galleries, fns, info."""
    M = _m()
    I = M['I']
    t0 = time.time()
    Fe = Feats(c)
    Zp = []
    for s in c.src:
        x = M['N'].subject_data(s, MON, keys=('train',))['train']
        Zp.append(torch.cat([I.acts(c.f0, x[c.pool[i:i + 4096]].astype(np.float32))['z']
                             for i in range(0, len(c.pool), 4096)]))
        del x
    Zp = torch.stack(Zp)
    chk = float((Zp.mean(0) - c.pool_z).abs().max())
    vpos = np.sort(np.random.RandomState(c.B).choice(len(c.src), N_VAL, replace=False))
    rest = [i for i in range(len(c.src)) if i not in set(vpos)]
    Yval, Yall = Zp[rest].mean(0), c.pool_z
    vq = [c.Zsrc_held[i] for i in vpos]
    del Zp
    labels = c.labels[c.pool]
    info = dict(pool_z_check=chk, val_people=[c.src[i] for i in vpos])
    # G1: block weights by source validation
    wgrid = [dict(zip(G1_BLOCKS, ws)) for ws in np.array(np.meshgrid(*[W_GRID] * len(G1_BLOCKS))).T.reshape(-1, 3)]
    vals = {}
    for w in wgrid:
        f = fit_gallery(Fe, c.pool, [Yval], w)[0]
        vals[wkey(w)] = (val_score(gallery_fn(Fe, f, w), c, vq), -sum(w.values()), w)
    best = max(vals.values(), key=lambda t: (round(t[0], 6), t[1]))
    w1 = best[2]
    w0 = {k: 0.0 for k in G1_BLOCKS}
    info.update(val_G0model=vals[wkey(w0)][0], val_G1=best[0], w_G1=wkey(w1))
    # G2: spatial residual head, s by source validation
    fv = fit_gallery(Fe, c.pool, [Yval], w1, labels, spatial=('real',))[0]
    sv = {s_: val_score(gallery_fn(Fe, fv, w1, 'real', s_), c, vq) for s_ in S_GRID}
    s_best = max(S_GRID, key=lambda s_: (round(sv[s_], 6), -abs(s_ - 1.0)))
    fvs = fit_gallery(Fe, c.pool, [Yval], w1, labels, spatial=('shuf',))[0]
    info.update(val_G2={f'{k:g}': v for k, v in sv.items()}, s_G2=s_best,
                val_G2shuf=val_score(gallery_fn(Fe, fvs, w1, 'shuf', s_best), c, vq))
    # final fits on all 9 source people
    fa = fit_gallery(Fe, c.pool, [Yall], w1, labels, spatial=('real', 'shuf'))[0]
    fns = {'G1': gallery_fn(Fe, fa, w1), 'G2': gallery_fn(Fe, fa, w1, 'real', s_best),
           'G2shuf': gallery_fn(Fe, fa, w1, 'shuf', s_best)}
    gals = {'G0': c.gal['anchor'][0], 'Gstar': c.gal['oracle'][0]}
    with torch.no_grad():
        for k, fn in fns.items():
            gals[k] = fn('test')
    info['anchor_cos_to_Gstar'] = {k: float((gals[k] * gals['Gstar']).sum(1).mean()) for k in gals}
    info['build_s'] = round(time.time() - t0)
    print(f'   14B galleries ({info["build_s"]}s): val G0-model {info["val_G0model"]:.2f}, G1 {best[0]:.2f} '
          f'[{info["w_G1"]}], G2 {sv[s_best]:.2f} (s {s_best:g}), G2shuf {info["val_G2shuf"]:.2f}; pool_z check '
          f'{chk:.1e}', flush=True)
    return gals, fns, info


def run_b(B):
    out = os.path.join(odir('b'), f'r14b_t{B:02d}.csv')
    if os.path.exists(out):
        print(f'[14B] t{B}: done')
        return
    M = _m()
    N, R, R5m, R6m, I = M['N'], M['R'], M['R5m'], M['R6m'], M['I']
    import nb_next as X
    import nb_gallery as GL
    R5m.SEED = SEED
    if not os.path.exists(N.ckpt_path('repaug', MON, B, SEED, 'last')):
        print(f'[14B] t{B}: no seed-{SEED} repaug checkpoint, skipping')
        return
    t0 = time.time()
    ctx = R6m.Ctx('repaug', B, MON)
    c = ctx.c
    gals, fns, info = build_b(c)
    tr = c.ses['train']
    reps = R.load_reps(B, MON)
    cand = np.array([i for i in c.pool if (tr[i] == GL.CS).sum() == 2])
    rows, corr = [], {}
    for draw in range(R6m.ndraw(K)):
        ims = GL.draw_images(cand, K, draw)
        rr = np.stack([np.nonzero(tr[i] == GL.CS)[0] for i in ims])
        x2 = np.asarray(reps[ims[:, None], rr], np.float32)
        pos = torch.as_tensor(np.array([ctx.pool_pos[int(i)] for i in ims]), device=dev())
        Ymeas = c.pool_z[pos]
        d1 = c.mS['h1'] - I.acts(c.f0, x2.reshape(-1, *x2.shape[2:]))['h1'].mean(0)
        Xs = X.acts2(c.f0, x2.mean(1), d1)['z']
        Zq0 = c.zq(d1)
        iims = torch.as_tensor(ims, device=dev())
        cells = []
        for g in GALS:
            cells.append((g, 'zero', None))
            with torch.no_grad():
                tgt = {'G0': lambda: c.galfn['anchor'](c.ftr[iims]), 'Gstar': lambda: Ymeas}.get(
                    g, lambda: fns[g]('train', iims))()
            cells.append((g, 'rigid', R.m_rigid(Xs, tgt, 1.0)))
        for qk in QK:
            for qi, z0 in enumerate(Zq0[qk]):
                for g, meth, mp in cells:
                    z = z0 if mp is None else mp(z0)
                    for rule in ('cos', 'csls_batch'):
                        Sm = R6m.smat(z, gals[g], rule)
                        mt, cr = score_rows(Sm, np.arange(200))
                        rows.append(dict(draw=draw, gallery=g, method=meth, rule=rule, query=qk, qdraw=qi, **mt))
                        if qk == 's4R4' and meth == 'zero' and rule == 'cos':
                            corr[f'{g}|d{draw}|q{qi}'] = cr
        print(f'   draw {draw} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df['target'] = B
    # gate: G0 and G* reproduce Round 8 (seed 2026, zero, cos, per draw x query, mean over query draws)
    gmsg = 'Round 8 file not found'
    r8 = os.path.join(N.ENV['results_dir'], 'nb_gallery', 'gallery', f'gal_s{SEED}_t{B:02d}_{MON}.csv')
    if os.path.exists(r8):
        r = pd.read_csv(r8)
        r = r[(r.method == 'zero') & (r.rule == 'cos') & r.gallery.isin(['anchor', 'oracle'])].copy()
        r['gallery'] = r.gallery.map({'anchor': 'G0', 'oracle': 'Gstar'})
        mine = df[(df.method == 'zero') & (df.rule == 'cos') & df.gallery.isin(['G0', 'Gstar'])].groupby(
            ['draw', 'gallery', 'query']).top1.mean()
        ref = r.set_index(['draw', 'gallery', 'query']).top1
        j = mine.index.intersection(ref.index)
        gmsg = f'max |diff| vs Round 8 (G0, G*; zero, cos) over {len(j)} cells: {float((mine[j] - ref[j]).abs().max()):.2e}'
    df.to_csv(out, index=False)
    np.savez_compressed(out.replace('.csv', '_correct.npz'), **corr)
    info.update(target=B, gate=gmsg, minutes=round((time.time() - t0) / 60, 1))
    json.dump(info, open(out.replace('.csv', '_meta.json'), 'w'), indent=1, default=str)
    print(f'[14B] t{B}: {info["minutes"]} min | gate: {gmsg}', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def holm(ps):
    keys = sorted(ps, key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = round(run, 4)
    return out


def report_a(R4m):
    fs = sorted(glob.glob(os.path.join(odir('a'), 'r14a_t*.csv')))
    fs = [f for f in fs if not f.endswith('_landmarks.csv')]
    if not fs:
        print('14A: no results yet')
        return
    df = pd.concat(map(pd.read_csv, fs))
    lm = pd.concat(map(pd.read_csv, [f.replace('.csv', '_landmarks.csv') for f in fs]))
    print('=' * 100 + '\nROUND 14A: clean support x noisy query (200-way, B queries vs all 200 candidates; seed 2026)')
    for f in fs:
        m = json.load(open(f.replace('.csv', '_meta.json')))
        print(f'  t{m["target"]}: variant {m["variant"]} (rho {m["rho"]}), {m["n3"]} session-3 repeats; gate: {m["gate"]}')
    pt = df.groupby(['target', 'rs', 'rq', 'method']).agg(top1=('top1', 'mean'), top1_100=('top1_100', 'mean'),
                                                         margin=('margin', 'mean')).reset_index()
    te = pt[pt.target.isin(TEST)]
    print('\nTOP-1 %, people 2-10 (rows = method, columns = support presentations R_s / query presentations R_q)')
    print(te.pivot_table(index='method', columns=['rs', 'rq'], values='top1').round(2).to_string())
    print('\nSENSITIVITY: strictly disjoint 100-way (B candidates only; chance 1%)')
    print(te.pivot_table(index='method', columns=['rs', 'rq'], values='top1_100').round(2).to_string())
    print('\nLANDMARKS (people 2-10): m and precision by R_s')
    print(lm[lm.target.isin(TEST)].groupby('rs')[['m', 'precision']].mean().round(3).to_string())

    def g(meth, rs, rq=4, col='top1'):
        x = pt[(pt.method == meth) & (pt.rs == rs) & (pt.rq == rq)]
        return x.set_index('target')[col]
    cost = {4: '400 presentations, ~1.3 min', 20: '2,000 presentations, ~6.7 min'}
    print(f'\nPRIMARY (R_q = 4; cost: R_s = 4 is {cost[4]}, R_s = 20 is {cost[20]})')
    ps = {'P1 rec 20 vs 4': R4m.pstats(g('rec', 20), g('rec', 4), 'P1 rec(R_s 20) - rec(R_s 4), R_q 4', primary=True),
          'P2 rec vs base': R4m.pstats(g('rec', 20), g('base', 20), 'P2 rec(R_s 20) - base(R_s 20), R_q 4', primary=True)}
    print('  HOLM:', holm(ps))
    print('\nSECONDARY (R_q = 4 unless noted; matched baseline = base at the same R_s)')
    for meth in ('center', 'rec_true', 'rec_all', 'sup_rot', 'rec_bcsls', 'trans', 'rec_cv', 'true_cv'):
        for rs in RS:
            R4m.pstats(g(meth, rs), g('base', rs), f'{meth} - base, R_s {rs}')
    for meth in ('rec_true', 'rec_all', 'sup_rot'):
        R4m.pstats(g(meth, 20), g(meth, 4), f'{meth}: R_s 20 - R_s 4')
    print('  query noise (R_q = 20):')
    for meth in ('rec', 'rec_true', 'center'):
        R4m.pstats(g(meth, 20, 20), g('base', 20, 20), f'{meth} - base, R_s 20, R_q 20')
    R4m.pstats(g('rec', 20, col='top1_100'), g('base', 20, col='top1_100'), 'P2 on the strictly disjoint 100-way set')
    print('  shrinkage reconciliation (exploratory; same landmark rows, CV-chosen vs locked rho):')
    for rs in RS:
        R4m.pstats(g('rec_cv', rs), g('rec', rs), f'rec_cv - rec, R_s {rs}')
        R4m.pstats(g('true_cv', rs), g('rec_true', rs), f'true_cv - rec_true, R_s {rs}')
        R4m.pstats(g('sup_rot', rs), g('rec_all', rs), f'sup_rot - rec_all (all true pairs: CV vs locked rho), R_s {rs}')
    d = {m: float((g(m, 20) - g('base', 20)).reindex(TEST).mean()) for m in ('rec', 'rec_true', 'rec_bcsls')}
    dq = float((g('rec', 20, 20) - g('base', 20, 20)).reindex(TEST).mean())
    print(f'\nDECISION (threshold +{THRESH:g} over matched baseline, R_s 20, R_q 4): rec {d["rec"]:+.2f}, rec_true '
          f'{d["rec_true"]:+.2f}, rec_bcsls {d["rec_bcsls"]:+.2f}; at R_q 20: rec {dq:+.2f}')
    if d['rec'] >= THRESH and d['rec_true'] >= THRESH:
        print('  -> clean support transfers: pursue one-time high-quality calibration')
    elif d['rec_true'] >= THRESH:
        print('  -> true pairs help, inferred do not: landmark inference deserves investment')
    elif d['rec_bcsls'] >= THRESH:
        print('  -> only batch scoring helps: the gain belongs to transductive retrieval, not a transferable correction')
    elif dq >= THRESH:
        print('  -> only cleaner queries help: move resources to gallery / query representations (closes clean '
              'support for this estimator and protocol)')
    else:
        print('  -> no clean-support effect at the threshold for this estimator and protocol')


def report_b(R4m):
    fs = sorted(f for f in glob.glob(os.path.join(odir('b'), 'r14b_t*.csv')))
    if not fs:
        print('14B: no results yet')
        return
    df = pd.concat(map(pd.read_csv, fs))
    print('\n' + '=' * 100 + '\nROUND 14B: deployable galleries on the frozen decoder (Round 8 evaluator, seed 2026)')
    metas = [json.load(open(f.replace('.csv', '_meta.json'))) for f in fs]
    for m in metas:
        print(f'  t{m["target"]}: gate {m["gate"]}; val G0-model {m["val_G0model"]:.2f}, G1 {m["val_G1"]:.2f} '
              f'[{m["w_G1"]}], G2 s {m["s_G2"]} {m["val_G2"]}, G2shuf {m["val_G2shuf"]:.2f}; anchor cos to G* '
              + ', '.join(f'{k} {v:.3f}' for k, v in m['anchor_cos_to_Gstar'].items()))
    pt = df.groupby(['target', 'gallery', 'method', 'rule', 'query']).agg(top1=('top1', 'mean'),
                                                                         margin=('margin', 'mean')).reset_index()
    te = pt[pt.target.isin(TEST)]
    for meth in ('zero', 'rigid'):
        for rule in ('cos', 'csls_batch'):
            x = te[(te.method == meth) & (te.rule == rule)]
            print(f'\n{meth}, {rule}: top-1 % (people 2-10)')
            print(x.pivot_table(index='gallery', columns='query', values='top1').reindex(GALS)[QK].round(2).to_string())
    print('\nmargin of the correct image over the hardest distractor (zero, cos, s4R4):')
    print(te[(te.method == 'zero') & (te.rule == 'cos') & (te['query'] == 's4R4')].groupby('gallery').margin.mean()
          .reindex(GALS).round(4).to_string())

    def g(gal, meth='zero', rule='cos', q='s4R4'):
        x = pt[(pt.gallery == gal) & (pt.method == meth) & (pt.rule == rule) & (pt['query'] == q)]
        return x.set_index('target').top1
    g0 = float(g('G0').reindex(TEST).mean())
    gs = float(g('Gstar').reindex(TEST).mean())
    print(f'\nREPRODUCTION: G0 {g0:.2f} (Round 8: 9.20), G* {gs:.2f} (Round 8: 19.14); gap {gs - g0:.2f}')
    print(f'LOCKED TARGETS: +3 -> {g0 + 3:.2f}, +5 -> {g0 + 5:.2f}')
    print('\nPRIMARY')
    R4m.pstats(g('G2'), g('G0'), 'G2 - G0, zero, s4R4, cos', primary=True)
    print('SECONDARY (Holm over three)')
    ps = {'G2 - G1': R4m.pstats(g('G2'), g('G1'), 'G2 - G1 (spatial beyond pooled multilevel)'),
          'G2 - G2shuf': R4m.pstats(g('G2'), g('G2shuf'), 'G2 - G2shuf (cell positions matter)'),
          'G1 - G0': R4m.pstats(g('G1'), g('G0'), 'G1 - G0 (pooled multilevel)')}
    print('  HOLM:', holm(ps))
    print('OTHER')
    R4m.pstats(g('G2', rule='csls_batch'), g('G0', rule='csls_batch'), 'G2 - G0, csls_batch')
    R4m.pstats(g('G2', 'rigid'), g('G0', 'rigid'), 'G2 - G0, rigid k = 500, cos')
    for q in ('s4R1', 's4R20', 'bench'):
        R4m.pstats(g('G2', q=q), g('G0', q=q), f'G2 - G0, {q}')
    d = float((g('G2') - g('G0')).reindex(TEST).mean())
    print(f'  gap recovered: {100 * d / (gs - g0):.0f}% of the observed G* - G0 gap (not a claim that all of it is '
          f'recoverable from images)')
    # error overlap with the measured gallery
    tot = {'G0wrong_Gstar_right': [0, 0], 'G0wrong_Gstar_wrong': [0, 0]}
    for f in fs:
        if int(os.path.basename(f)[6:8]) not in TEST:
            continue
        z = np.load(f.replace('.csv', '_correct.npz'))
        for key in [k for k in z.files if k.startswith('G0|')]:
            sfx = key[3:]
            g0c, gsc, g2c = z[key], z[f'Gstar|{sfx}'], z[f'G2|{sfx}']
            for name, msk in (('G0wrong_Gstar_right', ~g0c & gsc), ('G0wrong_Gstar_wrong', ~g0c & ~gsc)):
                tot[name][0] += int((msk & g2c).sum())
                tot[name][1] += int(msk.sum())
    for k, (a, b) in tot.items():
        print(f'  G2 fixes {a}/{b} ({100 * a / max(b, 1):.1f}%) of queries with {k.replace("_", ", ")}')


def report():
    M = _m()
    R4m = M['R4m']
    R4m.CONTRASTS.clear()
    report_a(R4m)
    report_b(R4m)
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['selftest', 'feats', 'a', 'b', 'report'])
    p.add_argument('--config')
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    a = p.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    from eegalign import nb as N
    N.setup(a.config)
    N.ENV.setdefault('reps_dir', a.reps_dir)
    if a.cmd == 'feats':
        feats()
    elif a.cmd == 'report':
        report()
    else:
        for t in a.targets:
            (run_a if a.cmd == 'a' else run_b)(t)


if __name__ == '__main__':
    main()
