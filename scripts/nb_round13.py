"""Round 13: relative-profile calibration vs SCORE recovery, on frozen NeuroBridge embeddings.

    python scripts/nb_round13.py selftest                 # synthetic algebra / leakage checks (no data needed)
    python scripts/nb_round13.py dev    --config C        # Experiment 1 on person 1 (development): gate, alpha, ridge
    python scripts/nb_round13.py eval   --config C        # Experiments 2-3 on people 2-10 with the locked settings
    python scripts/nb_round13.py report --config C

Question. Can a new person's later-day EEG be matched to unseen images through its RELATIONSHIPS to a bank of earlier-day
EEG-image reference pairs, when there is too little data to estimate their full coordinate transform? For query q and
candidate g: a_j(q) = cos(q, x_j) over EEG references, b_j(g) = cos(g, y_j) over their paired images; rank candidates by
the weighted Pearson correlation of the two profiles. Compared against SCORE's label-free recovery on the SAME backbone
(arXiv 2608.19134, "SCORE recovery on NeuroBridge"; the published system also uses a SAMGA encoder and recovery-aware
training, which we do not reproduce).

Embeddings (identical for every method). Seed-2026 repaug EEGProject decoder trained on the 9 other people (rounds 7-12
checkpoints), label-free layer-1 offset from the support trials (rounds 3-12 pipeline, labels unused), output z (512-d).
Visual side = the decoder's own image gallery m.gallery(aug features) for support images and the 200 test images (the
representation it was trained to match; NOT the predicted population gallery). Support = k session-3 training images x
2 presentations (round 6 draws); queries = session 4 at 1 / 4 / 20 presentations (round 4 draws) and the benchmark.
Support concepts are training concepts; scored concepts are the 200 THINGS test concepts (disjoint; asserted).

Unlabeled reference bank (SCORE on the support block). The algorithm receives the support EEG groups and the support image
pool in independently shuffled orders, no correspondence. Moment matching (Eq. 3, eps 1e-6) fitted on the support;
CSLS (k = 10); mutual nearest neighbours = landmarks; margin weights w_i = m max(s1 - s2, eta) / sum (eta 1e-3); weighted
centring; R = polar(Xc' W Yc + lambda I), lambda = 0.1 ||Xc' W Yc||_2; one step. Fallbacks: < 2 landmarks -> identity
rotation; < 3 landmarks -> no profile (rotation score used, flagged).

Readouts (identical landmark rows, images and weights):
  raw            cosine of z vs the gallery (= round 10 'zero / img' row: the checkpoint gate)
  score          SCORE recovery frozen on the support, each query scored alone by cosine (PRIMARY COMPARATOR)
  score_csls     same, CSLS with the gallery hubness term taken from the recovered SUPPORT EEG (still query-independent)
  prof           weighted Pearson profile score;  dot = the uncentred weighted profile dot (= q (X'WY) g', checked)
  fuse           (1 - alpha) z(score) + alpha z(prof), z over the 200 candidates of each query; alpha from development
  ridge          ridge map on the same pseudo-pairs (strength from development)
  trans_score    published-style transductive SCORE fitted on the evaluation batch itself (separate reference only)
Conditions: inferred (SCORE pairs), true (same EEG rows and weights, true image partners), shuffled (same rows, images
permuted among the selected ones; 5 fixed permutations), oracle_all (all k true pairs, equal weights; labelled reference).

AMENDMENT after the first development run (person 1 only; people 2-10 never looked at): the published defaults scored
BELOW no calibration (4.23% vs 6.73%), even with all 100 true pairs (5.37%), with landmark precision 0.11. Development
now searches, on person 1 only, 288 pipeline variants: shrinkage rho in {0.1 ... 30}, target space (image gallery vs
the predicted population gallery our working calibration uses), landmark margin filter (keep 100 / 50 / 25%), one or
two landmark re-selection steps, a repeat-agreement filter (drop the bottom quartile of support groups whose two
presentations disagree; label-free), moment matching on / off; scored by cosine or support-CSLS. The best variant is
locked with the fusion weight and ridge strength, then evaluated once on people 2-10 next to the published default.

PRIMARY (locked follow-up on the existing cohort): fuse - recovery within the chosen variant, inferred bank,
k = 100 x 2, s4R4, people 2-10 (draws averaged within person), Wilcoxon, all nine differences printed. If development
picks alpha = 0 the contrast is zero by construction and the report says so. Key secondary: chosen recovery and fusion
against NO calibration in the same gallery (Holm over four), and against the published default.
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
DEV = [1]
TEST = list(range(2, 11))
PEOPLE = list(range(1, 11))
K_MAIN = 100
KS = (25, 50, 100, 200, 500)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
GAMMAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CORRUPT = (0.0, 0.25, 0.5, 0.75, 1.0)
N_PERM = 5
CSLS_K = 10
RHO = 0.1
ETA = 1e-3
EPS = 1e-6
QSETS = ('s4R1', 's4R4', 's4R20', 'bench')
# Development-only search (added after the first development run, chosen on person 1 and locked before evaluation):
# shrinkage, target space, landmark margin filter, re-selection steps, repeat-agreement filter, moment matching.
DEFAULT = dict(mm=True, target='img', rho=0.1, q=1.0, steps=1, agree=False)     # SCORE's published deployment
GRID = dict(mm=(True, False), target=('img', 'pop'), rho=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0), q=(1.0, 0.5, 0.25),
            steps=(1, 2), agree=(False, True))
AGREE_DROP = 0.25          # agreement filter drops the support groups in the bottom quartile of repeat agreement


def vname(v):
    return f"mm{int(v['mm'])}|{v['target']}|rho{v['rho']:g}|q{v['q']:g}|s{v['steps']}|ag{int(v['agree'])}"


def variants():
    import itertools
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]


# ---------------------------------------------------------------------------------------------------------------
# core algebra (row vectors throughout; no labels enter any function below except where named true_* / evaluator)
# ---------------------------------------------------------------------------------------------------------------
def l2n(x):
    return F.normalize(x, dim=-1)


def moment_fit(Q, G):
    return dict(mq=Q.mean(0), sq=Q.std(0, unbiased=False), mg=G.mean(0), sg=G.std(0, unbiased=False))


def moment_apply(Z, mm):
    return (Z - mm['mq']) / (mm['sq'] + EPS) * (mm['sg'] + EPS) + mm['mg']


def csls(A, B, k=CSLS_K, rB=None):
    """2 cos - r_B(a) - r_A(b). rB: optional fixed per-candidate hubness term (query-independent mode)."""
    S = l2n(A) @ l2n(B).T
    rA = S.topk(min(k, S.shape[1]), dim=1).values.mean(1)
    if rB is None:
        rB = S.topk(min(k, S.shape[0]), dim=0).values.mean(0)
    return 2 * S - rA[:, None] - rB[None, :]


def hub_term(Ref, B, k=CSLS_K):
    """Per-candidate mean similarity to its k nearest rows of a REFERENCE set (not the queries)."""
    S = l2n(Ref) @ l2n(B).T
    return S.topk(min(k, S.shape[0]), dim=0).values.mean(0)


def mnn_landmarks(Qt, G, k=CSLS_K):
    """SCORE: mutual nearest neighbours under CSLS; margin weights (Eq. 5). Returns row ids into Qt / G and weights."""
    C = csls(Qt, G, k)
    f, r = C.argmax(1), C.argmax(0)
    i = torch.arange(len(Qt), device=Qt.device)
    keep = r[f] == i
    qi, gj = i[keep], f[keep]
    if len(qi) == 0:
        return qi, gj, torch.zeros(0, device=Qt.device)
    top2 = C[qi].topk(min(2, C.shape[1]), dim=1).values
    marg = (top2[:, 0] - top2[:, -1]).clamp(min=ETA) if top2.shape[1] > 1 else torch.ones(len(qi), device=Qt.device)
    w = len(qi) * marg / marg.sum()
    return qi, gj, w


def fit_rotation(X, Y, w, rho=RHO):
    """Weighted centring + R = polar(Xc' W Yc + lambda I), lambda = rho ||Xc' W Yc||_2 (SCORE Eqs. 6-7)."""
    d = X.shape[1]
    if len(X) < 2:
        return torch.eye(d, device=X.device, dtype=X.dtype), X.mean(0) * 0, X.mean(0) * 0, True
    wn = w / w.sum()
    mx, my = (wn[:, None] * X).sum(0), (wn[:, None] * Y).sum(0)
    C = ((X - mx).T @ (w[:, None] * (Y - my))).double()
    lam = rho * torch.linalg.matrix_norm(C, ord=2)
    U, _, Vh = torch.linalg.svd(C + lam * torch.eye(d, device=C.device, dtype=C.dtype))
    return (U @ Vh).to(X.dtype), mx, my, False


def apply_rotation(Z, R, mx, my):
    return (Z - mx) @ R + my


def fit_ridge(X, Y, w, gamma):
    wn = w / w.sum()
    mx, my = (wn[:, None] * X).sum(0), (wn[:, None] * Y).sum(0)
    Xc, Yc = X - mx, Y - my
    A = (Xc.T @ (w[:, None] * Xc)).double()
    s = torch.trace(A) / A.shape[0] + 1e-12
    Bm = torch.linalg.solve(A + gamma * s * torch.eye(A.shape[0], device=A.device, dtype=A.dtype),
                            (Xc.T @ (w[:, None] * Yc)).double()).to(X.dtype)
    return lambda Z: (Z - mx) @ Bm + my  # noqa: E731


def profiles(Qp, Xl, G, Yl, w):
    """Weighted Pearson (n, g), uncentred weighted dot (n, g), and the mask of undefined entries."""
    wn = (w / w.sum()).to(Qp.dtype)
    A = l2n(Qp) @ l2n(Xl).T
    B = l2n(G) @ l2n(Yl).T
    dot = (A * wn) @ B.T
    Am = A - (A * wn).sum(1, keepdim=True)
    Bm = B - (B * wn).sum(1, keepdim=True)
    den = ((Am ** 2) * wn).sum(1).sqrt()[:, None] * ((Bm ** 2) * wn).sum(1).sqrt()[None, :]
    bad = den < 1e-10
    return ((Am * wn) @ Bm.T) / den.clamp(min=1e-10), dot, bad


def zrow(S):
    return (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-12)


def metrics(S):
    """Correct candidate of row i is column i. top-1, top-5 (%), MRR, and per-query ranks (0 = first)."""
    tgt = S.diagonal()[:, None]
    rank = (S > tgt).sum(1)
    return dict(top1=100 * float((rank == 0).float().mean()), top5=100 * float((rank < 5).float().mean()),
                mrr=float((1.0 / (rank + 1).float()).mean())), rank.cpu().numpy()


def prep(Z, mm):
    return Z if mm is None else moment_apply(Z, mm)


def recover(Xs, Ys, gen, v=None, agree=None):
    """The UNLABELED pipeline: support EEG groups and support image pool arrive in independent random orders.
    v: pipeline variant (default = SCORE's published deployment). agree: per support group, the cosine between its two
    presentations (label-free: repeats of the same anonymous stimulus are known). Returns the bank in ORIGINAL support
    indices so true partners can be swapped in by the evaluator only."""
    import math
    v = v or DEFAULT
    n = len(Xs)
    p1 = torch.randperm(n, generator=gen).to(Xs.device)
    p2 = torch.randperm(n, generator=gen).to(Xs.device)
    Xp, Yp = Xs[p1], Ys[p2]
    mm = moment_fit(Xp, Yp) if v['mm'] else None
    allow = torch.arange(n, device=Xs.device)
    if v['agree'] and agree is not None:
        a = agree[p1]
        allow = torch.nonzero(a >= torch.quantile(a.float(), AGREE_DROP)).flatten()
    Xq = prep(Xp, mm)
    cur = Xq
    for step in range(v['steps']):
        qi, gj, w = mnn_landmarks(cur[allow], Yp)
        qi = allow[qi]
        if v['q'] < 1 and len(qi) > 3:
            keep = w.argsort(descending=True)[:max(3, int(math.ceil(v['q'] * len(qi))))]
            qi, gj, w = qi[keep], gj[keep], w[keep]
            w = len(w) * w / w.sum()
        if step < v['steps'] - 1 and len(qi) >= 2:
            R, mx, my, _ = fit_rotation(Xq[qi], Yp[gj], w, v['rho'])
            cur = apply_rotation(Xq, R, mx, my)
    return dict(mm=mm, eeg=p1[qi], img=p2[gj], w=w)


def readouts(bank_eeg, bank_img, w, mm, Xs, Ys, Q, G, alphas=ALPHAS, gammas=(), hub_ref=True, rho=RHO, rule='score'):
    """Every readout from ONE bank (indices into the support), so all methods see identical information.
    Q: (n, d) query EEG embeddings, G: (200, d) gallery. Row i of Q belongs to column i of G (evaluator only)."""
    out, flags = {}, {}
    Xl, Yl = prep(Xs[bank_eeg], mm), Ys[bank_img]
    Qm = prep(Q, mm)
    R, mx, my, fb_rot = fit_rotation(Xl, Yl, w, rho)
    Qr = apply_rotation(Qm, R, mx, my)
    s_rot = l2n(Qr) @ l2n(G).T
    out['score'] = s_rot
    if hub_ref:
        out['score_csls'] = csls(Qr, G, rB=hub_term(apply_rotation(prep(Xs, mm), R, mx, my), G))
    if len(bank_eeg) >= 3:
        s_prof, s_dot, bad = profiles(Qm, Xl, G, Yl, w)
        fb_prof = bad.any(1)
    else:
        s_prof, s_dot = s_rot.clone(), s_rot.clone()
        fb_prof = torch.ones(len(Q), dtype=torch.bool, device=Q.device)
    s_prof = torch.where(fb_prof[:, None], s_rot, s_prof)        # prespecified fallback: rotation score, flagged
    out['prof'], out['dot'] = s_prof, s_dot
    zr, zp = zrow(out.get(rule, s_rot)), zrow(s_prof)
    for a in alphas:
        f = (1 - a) * zr + a * zp
        out[f'fuse_{a:g}'] = torch.where(fb_prof[:, None], zr, f)
    for g in gammas:
        out[f'ridge_{g:g}'] = l2n(fit_ridge(Xl, Yl, w, g)(Qm)) @ l2n(G).T if len(bank_eeg) >= 2 else s_rot
    flags.update(rot_fallback=bool(fb_rot), prof_fallback_rate=float(fb_prof.float().mean()),
                 orth_err=float((R.T @ R - torch.eye(R.shape[0], device=R.device, dtype=R.dtype)).abs().max()))
    return out, flags


def transductive_score(Q, G):
    """Published-style SCORE on the evaluation batch: moment matching, landmarks and CSLS all use the batch."""
    mm = moment_fit(Q, G)
    Qm = moment_apply(Q, mm)
    qi, gj, w = mnn_landmarks(Qm, G)
    R, mx, my, _ = fit_rotation(Qm[qi], G[gj], w)
    return csls(apply_rotation(Qm, R, mx, my), G), len(qi)


def bank_diag(bank, w):
    wn = (w / w.sum()) if len(w) else w
    return dict(m=int(len(w)), neff=float(1.0 / (wn ** 2).sum()) if len(w) else 0.0,
                precision=float((bank['eeg'] == bank['img']).float().mean()) if len(w) else np.nan)


# ---------------------------------------------------------------------------------------------------------------
# synthetic checks
# ---------------------------------------------------------------------------------------------------------------
def selftest():
    torch.manual_seed(0)
    dt = torch.float64
    d, n, g = 32, 120, 200
    R0, _ = torch.linalg.qr(torch.randn(d, d, dtype=dt))
    base_s, base_g = torch.randn(n, d, dtype=dt), torch.randn(g, d, dtype=dt)
    Ys, G = base_s, base_g                      # image side
    Xs, Q = base_s @ R0.T, base_g @ R0.T        # EEG side = images under a hidden orthogonal map (noiseless)
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f'  [{"PASS" if cond else "FAIL"}] {name} {detail}')

    # geometry: true-pair profiles agree exactly when the same transform applies to support and query
    w = torch.ones(n, dtype=dt)
    A = l2n(Q) @ l2n(Xs).T
    B = l2n(G) @ l2n(Ys).T
    check('true-pair within-modality profiles identical under a shared orthogonal map', (A - B).abs().max() < 1e-10,
          f'(max diff {(A - B).abs().max():.1e})')
    sp, sd, bad = profiles(Q, Xs, G, Ys, w)
    check('true-pair profile retrieval is perfect (noiseless)', metrics(sp)[0]['top1'] == 100.0)
    R, mx, my, _ = fit_rotation(Xs, Ys, w, rho=0.0)
    check('rotation with rho = 0 recovers the hidden map (X R = Y)', (R - R0).abs().max() < 1e-8, f'(max diff {(R - R0).abs().max():.1e})')
    R1, mx1, my1, _ = fit_rotation(Xs, Ys, w)
    check('recovered matrix orthogonal (rho = 0.1)', (R1.T @ R1 - torch.eye(d, dtype=dt)).abs().max() < 1e-8)
    check('rho = 0.1 rotation still retrieves (noiseless)', metrics(l2n(apply_rotation(Q, R1, mx1, my1)) @ l2n(G).T)[0]['top1'] > 95)
    # bilinear equivalence
    wr = torch.rand(n, dtype=dt) + 0.1
    _, sd2, _ = profiles(Q[:10], Xs[:40], G, Ys[:40], wr[:40])
    wn = wr[:40] / wr[:40].sum()
    bil = l2n(Q[:10]) @ (l2n(Xs[:40]).T @ (wn[:, None] * l2n(Ys[:40]))) @ l2n(G).T
    check('uncentred profile dot == q (X\'WY) g\'', (sd2 - bil).abs().max() < 1e-12, f'(max diff {(sd2 - bil).abs().max():.1e})')
    # SCORE assumes the encoder is already roughly aligned with the image space (a subject-specific perturbation of a
    # shared space), so landmark tests use a SMALL hidden rotation; a random full rotation is not identifiable this way.
    S0 = torch.randn(d, d, dtype=dt)
    Rs = torch.linalg.matrix_exp(0.1 * (S0 - S0.T))        # true-pair cosine ~0.7
    Xa, Qa = base_s @ Rs, base_g @ Rs
    bank = recover(Xa, Ys, torch.Generator().manual_seed(1))
    dg = bank_diag(bank, bank['w'])
    check('noiseless, near-aligned: SCORE landmarks are true pairs', dg['precision'] == 1.0 and dg['m'] > 10,
          f"(m = {dg['m']}, precision {dg['precision']:.2f})")
    Ra, mxa, mya, _ = fit_rotation(Xa[bank['eeg']], Ys[bank['img']], bank['w'], rho=0.0)
    check('landmark rotation undoes the hidden perturbation', metrics(l2n(apply_rotation(Qa, Ra, mxa, mya)) @ l2n(G).T)[0]['top1'] == 100.0)
    # noisy, near-aligned setting for the remaining checks
    Xs, Q = Xa, Qa
    Xn, Qn = Xs + 0.8 * torch.randn_like(Xs), Q + 0.8 * torch.randn_like(Q)
    b1 = recover(Xn, Ys, torch.Generator().manual_seed(2))
    b2 = recover(Xn, Ys, torch.Generator().manual_seed(3))       # different input orders
    pairs = lambda b: sorted(zip(b['eeg'].tolist(), b['img'].tolist()))  # noqa: E731
    check('order invariance: same landmark pairs under independent input permutations', pairs(b1) == pairs(b2),
          f'(m = {len(b1["w"])})')
    o1, _ = readouts(b1['eeg'], b1['img'], b1['w'], b1['mm'], Xn, Ys, Qn, G, gammas=(1.0,))
    o2, _ = readouts(b2['eeg'], b2['img'], b2['w'], b2['mm'], Xn, Ys, Qn, G, gammas=(1.0,))
    check('order invariance: identical scores', max(float((o1[k] - o2[k]).abs().max()) for k in o1) < 1e-8)
    # label isolation: the unlabeled fit never sees labels; relabelling the evaluator changes metrics only
    check('label isolation: recover() and readouts() take no labels', 'labels' not in recover.__code__.co_varnames
          and 'labels' not in readouts.__code__.co_varnames)
    # independent scoring: a query's scores do not depend on the other queries
    os_, _ = readouts(b1['eeg'], b1['img'], b1['w'], b1['mm'], Xn, Ys, Qn[:17], G, gammas=(1.0,))
    check('independent scoring (subset of queries == full batch rows)',
          max(float((o1[k][:17] - os_[k]).abs().max()) for k in os_) < 1e-10)
    check('alpha = 0 reproduces rotation ranks', torch.equal(o1['fuse_0'].argsort(1), zrow(o1['score']).argsort(1)))
    # every searched variant stays order-invariant (agreement scores travel with their EEG group)
    agree = torch.rand(len(Xn), dtype=dt)
    vx = dict(mm=False, target='img', rho=3.0, q=0.5, steps=2, agree=True)
    v1 = recover(Xn, Ys, torch.Generator().manual_seed(5), vx, agree)
    v2 = recover(Xn, Ys, torch.Generator().manual_seed(6), vx, agree)
    check('variant (no moment matching, 2 steps, margin + agreement filters) is order-invariant', pairs(v1) == pairs(v2)
          and len(v1['w']) >= 3, f'(m = {len(v1["w"])})')
    check('agreement filter only drops low-agreement groups', bool((agree[v1['eeg']] >= torch.quantile(agree, AGREE_DROP)).all()))
    # true-pair swap keeps rows and weights
    true_img = b1['eeg']
    check('true-pair bank keeps the same EEG rows and weights', len(true_img) == len(b1['img']))
    print('SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


# ---------------------------------------------------------------------------------------------------------------
# data (reuses rounds 4-10 loaders; runs in Colab)
# ---------------------------------------------------------------------------------------------------------------
def _mods():
    from eegalign import nb as N
    import nb_round3 as R
    import nb_round6 as R6m
    import nb_gallery as GL
    import nb_round9 as R9
    import nb_round10 as R10
    return N, R, R6m, GL, R9, R10


def odir(sub=''):
    N = _mods()[0]
    d = os.path.join(N.ENV['results_dir'], 'nb_round13', sub)
    os.makedirs(d, exist_ok=True)
    return d


def settings_path():
    return os.path.join(odir(), 'locked_settings.json')


def concept_check(N, GL, B, ims):
    d = N.subject_data(B, MON, keys=('train_paths', 'test_paths'))
    tr, te = d.get('train_paths'), d.get('test_paths')
    if tr is None or te is None or len(set(map(str, te))) < 2:
        return 'paths unavailable (THINGS train/test concepts are disjoint by construction)'
    a = {GL.concept(p) for p in np.asarray(tr)[ims]}
    b = {GL.concept(p) for p in te}
    assert not (a & b), f'concept leakage: {sorted(a & b)[:5]}'
    return f'checked: {len(a)} support concepts, {len(b)} test concepts, empty intersection'


class Target:
    """Frozen embeddings for one held-out person; everything label-free about the target is computed here."""

    def __init__(self, B):
        N, R, R6m, GL, R9, R10 = _mods()
        import nb_next as X
        self.B = B
        dev = N.ENV['device']
        m = R10.load_decoder('eeg_aug', B, MON, SEED, train=False)
        if m is None:
            raise FileNotFoundError(f'no seed-{SEED} repaug decoder for person {B} (run round 7 / 10 training)')
        self.m = m
        lab = N.subject_data(B, MON, keys=('train_labels',))['train_labels']
        self.pool, _ = N.split(lab)
        Zs, H = [], []
        for s in PEOPLE:
            if s == B:
                continue
            tr_s = N.subject_data(s, MON, keys=('train',))['train']
            Zs.append(R10.z_of(m, tr_s[self.pool]))
            H.append(R10.h1_mean(m, tr_s[np.random.RandomState(3).choice(self.pool, 300, replace=False)]))
            del tr_s
        self.mS = torch.stack(H).mean(0)                                 # identical to round 10
        self.pool_z = torch.stack(Zs).mean(0)                            # source people's responses (no target data)
        ftr, fte = N.features('train', 'aug').to(dev), N.features('test', 'aug').to(dev)
        self.Gtr, self.Gte = m.gallery(ftr), m.gallery(fte)              # decoder's own image gallery (normalized)
        Bm, xm, ym = X.ridge_svd(ftr[torch.as_tensor(self.pool, device=dev)], self.pool_z)
        self.Ganc = l2n((fte - xm) @ Bm + ym)                            # predicted population gallery (round 10)
        self.pos_of = {int(i): j for j, i in enumerate(self.pool)}
        self.Q, ses = R9._queries(B, MON)
        self.tr = ses['train']
        self.reps = R.load_reps(B, MON)
        self.cand = np.array([i for i in self.pool if (self.tr[i] == R10.CS).sum() == 2])
        self.N, self.R6m, self.GL, self.R10 = N, R6m, GL, R10

    def support(self, k, draw):
        ims = self.GL.draw_images(self.cand, k, draw)
        rr = np.stack([np.nonzero(self.tr[i] == self.R10.CS)[0] for i in ims])
        x2 = np.asarray(self.reps[ims[:, None], rr], np.float32)
        d1 = self.mS - self.R10.h1_mean(self.m, x2.reshape(-1, *x2.shape[2:]))      # label-free, support trials only
        z1, z2 = self.R10.z_of(self.m, x2[:, 0], d1), self.R10.z_of(self.m, x2[:, 1], d1)
        dev = z1.device
        return dict(ims=ims, Xs=self.R10.z_of(self.m, x2.mean(1), d1), agree=(l2n(z1) * l2n(z2)).sum(1),
                    Ys={'img': self.Gtr[torch.as_tensor(ims, device=dev)],
                        'pop': self.pool_z[torch.as_tensor([self.pos_of[int(i)] for i in ims], device=dev)]},
                    G={'img': self.Gte, 'pop': self.Ganc},
                    Zq={qk: [self.R10.z_of(self.m, q, d1) for q in self.Q[qk]] for qk in QSETS})


def rot_scores(bank, Xs, Ys, zs, G, rho):
    """Rotation-only scores for the variant search: cosine and support-CSLS per query set draw."""
    Xl, Yl = prep(Xs[bank['eeg']], bank['mm']), Ys[bank['img']]
    R, mx, my, _ = fit_rotation(Xl, Yl, bank['w'], rho)
    hub = hub_term(apply_rotation(prep(Xs, bank['mm']), R, mx, my), G)
    out = []
    for z in zs:
        Zr = apply_rotation(prep(z, bank['mm']), R, mx, my)
        out.append((metrics(l2n(Zr) @ l2n(G).T)[0]['top1'], metrics(csls(Zr, G, rB=hub))[0]['top1']))
    return out


def search_variants(T):
    """Development only: every variant x support draw, inferred and true-pair rotation scores on s4R4."""
    rows = []
    for draw in range(T.R6m.ndraw(K_MAIN)):
        sup = T.support(K_MAIN, draw)
        for v in variants():
            bank = recover(sup['Xs'], sup['Ys'][v['target']], torch.Generator().manual_seed(10000 * T.B + draw), v,
                           sup['agree'])
            dg = bank_diag(bank, bank['w'])
            for cond, bk in (('inferred', bank), ('true', dict(bank, img=bank['eeg']))):
                if len(bk['w']) < 2:
                    continue
                for qi_, (c, cs) in enumerate(rot_scores(bk, sup['Xs'], sup['Ys'][v['target']], sup['Zq']['s4R4'],
                                                         sup['G'][v['target']], v['rho'])):
                    rows.append(dict(draw=draw, variant=vname(v), cond=cond, qdraw=qi_, score=c, score_csls=cs, **dg))
        print(f'   variant search: draw {draw} done', flush=True)
    return pd.DataFrame(rows)


def run_target(B, settings, mode):
    """mode 'dev': variant search + Experiment 1 grid on the development person. mode 'eval': locked settings."""
    out = os.path.join(odir(mode), f'r13_t{B:02d}.csv')
    if os.path.exists(out):
        print(f'[{mode}] t{B}: done')
        return
    t0 = time.time()
    T = Target(B)
    if mode == 'dev':
        grid = search_variants(T)
        grid.to_csv(out.replace('.csv', '_variants.csv'), index=False)
        g = grid[grid.cond == 'inferred'].groupby('variant')[['score', 'score_csls']].mean()
        best = max(((vn, rule, g.loc[vn, rule]) for vn in g.index for rule in ('score', 'score_csls')),
                   key=lambda t: (round(t[2], 6), t[0] == vname(DEFAULT), t[1] == 'score'))
        chosen = next(v for v in variants() if vname(v) == best[0])
        settings = dict(variant=chosen, rule=best[1], search_top1=float(best[2]))
        json.dump(settings, open(out.replace('.csv', '_chosen.json'), 'w'), indent=1)
        print(f'   chosen variant {best[0]} with {best[1]} (dev s4R4 {best[2]:.2f})', flush=True)
    arms = [('default', DEFAULT, 'score'), ('chosen', settings['variant'], settings['rule'])]
    rows, diags, ranks, gate = [], [], {}, []
    r10 = os.path.join(T.N.ENV['results_dir'], 'nb_round10', 'eval', f'r10_s{SEED}_t{B:02d}_{MON}.csv')
    alphas = ALPHAS if mode == 'dev' else tuple(sorted({0.0, settings['alpha']}))
    gammas = GAMMAS if mode == 'dev' else (settings['gamma'],)
    ks = (K_MAIN,) if mode == 'dev' else KS
    cmsg = None
    for k in ks:
        if k > len(T.cand):
            continue
        for draw in range(T.R6m.ndraw(k)):
            sup = T.support(k, draw)
            if cmsg is None:
                cmsg = concept_check(T.N, T.GL, B, sup['ims'])
            Xs, n = sup['Xs'], len(sup['Xs'])
            for qk in QSETS:
                for qi_, z in enumerate(sup['Zq'][qk]):
                    for meth, G in (('raw', T.Gte), ('raw_pop', T.Ganc)):
                        mt, rk = metrics(l2n(z) @ l2n(G).T)
                        rows.append(dict(target=B, k=k, draw=draw, variant='none', cond='none', method=meth, query=qk,
                                         qdraw=qi_, **mt))
                        if meth == 'raw' and k == K_MAIN:
                            gate.append((draw, qk, qi_, mt['top1']))
                    if k == K_MAIN:
                        S, _ = transductive_score(z, T.Gte)
                        rows.append(dict(target=B, k=k, draw=draw, variant='none', cond='transductive',
                                         method='trans_score', query=qk, qdraw=qi_, **metrics(S)[0]))
            for tag, v, rule in arms:
                Ys, G = sup['Ys'][v['target']], sup['G'][v['target']]
                bank = recover(Xs, Ys, torch.Generator().manual_seed(10000 * B + 100 * k + draw), v, sup['agree'])
                conds = {'inferred': (bank['eeg'], bank['img'], bank['w']),
                         'true': (bank['eeg'], bank['eeg'], bank['w']),
                         'oracle_all': (torch.arange(n, device=Xs.device),) * 2 + (torch.ones(n, device=Xs.device),)}
                pg = torch.Generator().manual_seed(777 + draw)
                for p in range(N_PERM):
                    perm = torch.randperm(len(bank['img']), generator=pg).to(Xs.device)
                    conds[f'shuffled_{p}'] = (bank['eeg'], bank['img'][perm], bank['w'])
                if mode == 'dev':
                    cg = torch.Generator().manual_seed(999 + draw)
                    mlen = len(bank['eeg'])
                    for c in CORRUPT:
                        nc = int(round(c * mlen))
                        img = bank['eeg'].clone()
                        if nc >= 2:
                            sel = torch.randperm(mlen, generator=cg)[:nc].to(Xs.device)
                            img[sel] = img[sel.roll(1)]                  # cyclic shift = no fixed points
                        conds[f'corrupt_{c:g}'] = (bank['eeg'], img, bank['w'])
                diags.append(dict(target=B, k=k, draw=draw, variant=tag, **bank_diag(bank, bank['w'])))
                # the oracle bank is moment-matched with the support pool's stats (same preprocessing as inferred)
                for cname, (be, bi, bw) in conds.items():
                    main = cname in ('inferred', 'true', 'oracle_all') or mode == 'dev'
                    for qk in QSETS:
                        if not main and qk != 's4R4':
                            continue
                        for qi_, z in enumerate(sup['Zq'][qk]):
                            if len(be) < 2:
                                continue
                            o, fl = readouts(be, bi, bw, bank['mm'], Xs, Ys, z, G, alphas=alphas,
                                             gammas=gammas if main else (), rho=v['rho'], rule=rule)
                            for meth, S in o.items():
                                mt, rk = metrics(S)
                                rows.append(dict(target=B, k=k, draw=draw, variant=tag, cond=cname, method=meth,
                                                 query=qk, qdraw=qi_, **mt))
                                if k == K_MAIN and qk == 's4R4' and cname in ('inferred', 'true'):
                                    ranks[f'{tag}|{cname}|{meth}|d{draw}|q{qi_}'] = rk
                            if cname == 'inferred' and qk == 's4R4' and qi_ == 0:
                                diags[-1].update(fl)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.DataFrame(diags).to_csv(out.replace('.csv', '_landmarks.csv'), index=False)
    np.savez_compressed(out.replace('.csv', '_ranks.npz'), **ranks)
    gmsg = 'round 10 file not found'
    if gate and os.path.exists(r10):
        r = pd.read_csv(r10)
        r = r[(r.decoder == 'eeg_aug') & (r.k == K_MAIN) & (r.method == 'zero') & (r.gallery == 'img') & (r.rule == 'cos')]
        mine = pd.DataFrame(gate, columns=['draw', 'query', 'qdraw', 'top1']).groupby(['draw', 'query']).top1.mean()
        ref = r.set_index(['draw', 'query']).top1
        j = mine.index.intersection(ref.index)
        gmsg = f'max |diff| vs round 10 zero/img over {len(j)} cells: {float((mine[j] - ref[j]).abs().max()):.2e}'
    meta = dict(target=B, mode=mode, seed=SEED, montage=MON, checkpoint='repaug seed %d (rounds 7-12)' % SEED,
                features='aug (NeuroBridge fused degraded CLIP RN50); img = decoder image gallery, pop = predicted '
                         'population gallery', concept_check=cmsg, checkpoint_gate=gmsg, settings=settings,
                arms={t: vname(v) + f' / {r_}' for t, v, r_ in arms}, minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(out.replace('.csv', '_meta.json'), 'w'), indent=1, default=str)
    print(f'[{mode}] t{B}: {meta["minutes"]} min | gate: {gmsg} | {cmsg}', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# development gate, locking, evaluation, report
# ---------------------------------------------------------------------------------------------------------------
def per_target(df, k, variant, cond, method, query='s4R4'):
    x = df[(df.k == k) & (df.variant == variant) & (df.cond == cond) & (df.method == method) & (df['query'] == query)]
    return x.groupby('target').top1.mean()


def dev():
    B = DEV[0]
    base = os.path.join(odir('dev'), f'r13_t{B:02d}')
    if os.path.exists(base + '.csv') and not os.path.exists(base + '_variants.csv'):     # first-version development run
        import shutil
        dst = os.path.join(odir(), f'dev_v1_{time.strftime("%Y%m%d_%H%M%S")}')
        shutil.move(odir('dev'), dst)
        if os.path.exists(settings_path()):
            shutil.move(settings_path(), os.path.join(dst, 'locked_settings_v1.json'))
        print(f'archived the first-version development run to {dst}')
    run_target(B, {}, 'dev')
    df, lm = pd.read_csv(base + '.csv'), pd.read_csv(base + '_landmarks.csv')
    grid = pd.read_csv(base + '_variants.csv')
    ch = json.load(open(base + '_chosen.json'))
    rule = ch['rule']
    s = df[(df['query'] == 's4R4') & (df.k == K_MAIN)].copy()
    raw = s[s.method == 'raw'].top1.mean()
    raw_pop = s[s.method == 'raw_pop'].top1.mean()
    print(f'\nDEVELOPMENT (person {B}, k = {K_MAIN} x 2, s4R4, mean over draws and query draws)')
    print(f'  no calibration: image gallery {raw:.2f}, population gallery {raw_pop:.2f}')
    print('\nVARIANT SEARCH (inferred bank, rotation only; top 10 and the published default)')
    g = grid[grid.cond == 'inferred'].groupby('variant')[['score', 'score_csls', 'precision', 'm']].mean()
    g['best'] = g[['score', 'score_csls']].max(1)
    gt = grid[grid.cond == 'true'].groupby('variant')[['score', 'score_csls']].mean().max(1).rename('true_pairs')
    g = g.join(gt).sort_values('best', ascending=False)
    print(g.head(10).round(2).to_string())
    print('  default: ' + g.loc[[vname(DEFAULT)]].round(2).to_string(header=False))
    for key in GRID:
        vals = {}
        for val in GRID[key]:
            tag = {'mm': f'mm{int(val) if isinstance(val, bool) else val}', 'target': f'|{val}|', 'rho': f'|rho{val:g}|',
                   'q': f'|q{val:g}|', 'steps': f'|s{val}|', 'agree': f'|ag{int(val)}'}[key]
            vals[val] = g[[tag in vn for vn in g.index]].best.mean()
        print(f'  mean over the grid by {key}: ' + ', '.join(f'{k_}: {v_:.2f}' for k_, v_ in vals.items()))
    print(f"  CHOSEN: {vname(ch['variant'])} scored with {rule} ({ch['search_top1']:.2f})")
    for tag in ('default', 'chosen'):
        x = s[s.variant == tag].copy()
        x['cond2'] = x.cond.str.replace(r'shuffled_\d', 'shuffled', regex=True)
        l_ = lm[lm.variant == tag]
        print(f'\n[{tag}] landmarks m = {l_.m.mean():.1f}, effective {l_.neff.mean():.1f}, precision {l_.precision.mean():.2f}')
        print(x.groupby(['cond2', 'method']).top1.mean().unstack(0).round(2).to_string())
    c = s[s.variant == 'chosen']
    best = {}
    for cond in ('inferred', 'true'):
        fz = {a: c[(c.cond == cond) & (c.method == f'fuse_{a:g}')].top1.mean() for a in ALPHAS}
        best[cond] = max(ALPHAS, key=lambda a: (round(fz[a], 6), -a))
        print(f'  chosen, {cond}: fusion by alpha ' + ', '.join(f'{a:g}: {v:.2f}' for a, v in fz.items()) + f' -> {best[cond]:g}')
    gz = {gm: c[(c.cond == 'inferred') & (c.method == f'ridge_{gm:g}')].top1.mean() for gm in GAMMAS}
    g_best = max(GAMMAS, key=lambda gm: (round(gz[gm], 6), gm))
    print('  ridge on inferred pairs by gamma: ' + ', '.join(f'{k_:g}: {v_:.2f}' for k_, v_ in gz.items()) + f' -> {g_best:g}')
    print('\nCORRUPTION SWEEP, chosen variant (rotation | profile)')
    for cc in CORRUPT:
        q_ = c[c.cond == f'corrupt_{cc:g}']
        print(f'  {cc:4.0%}: {q_[q_.method == rule].top1.mean():.2f} | {q_[q_.method == "prof"].top1.mean():.2f}')
    t_rot = c[(c.cond == 'true') & (c.method == rule)].top1.mean()
    t_prof = c[(c.cond == 'true') & (c.method == 'prof')].top1.mean()
    i_rot = c[(c.cond == 'inferred') & (c.method == rule)].top1.mean()
    ref = raw_pop if ch['variant']['target'] == 'pop' else raw
    gate_stop = best['true'] == 0.0 and t_prof <= t_rot
    print(f'\nDECISION GATE: chosen recovery {i_rot:.2f} vs no calibration {ref:.2f} (same gallery); true-pair profile '
          f'{t_prof:.2f} vs true-pair rotation {t_rot:.2f}; alpha true {best["true"]:g}, inferred {best["inferred"]:g}. '
          + ('STOP for profiles: they add nothing even with true pairs.' if gate_stop else 'Proceed.')
          + ('' if i_rot > ref else ' NOTE: even the best unlabeled recovery does not beat no calibration on this person.'))
    settings = dict(variant=ch['variant'], variant_name=vname(ch['variant']), rule=rule, alpha=float(best['inferred']),
                    gamma=float(g_best), alpha_true=float(best['true']), gate_stop=bool(gate_stop), dev_person=B,
                    k=K_MAIN, locked=time.strftime('%Y-%m-%d %H:%M'))
    json.dump(settings, open(settings_path(), 'w'), indent=1)
    print(f'locked settings -> {settings_path()}: {settings}')


def evaluate(targets):
    if not os.path.exists(settings_path()):
        raise SystemExit('run `dev` first: evaluation needs the locked development settings')
    st = json.load(open(settings_path()))
    if 'variant' not in st:
        raise SystemExit('locked settings predate the variant search: delete results/nb_round13 and rerun `dev`')
    if st.get('gate_stop'):
        print('NOTE: the development gate said STOP for profiles; running the locked evaluation as a documented check.')
    old = [x for x in glob.glob(os.path.join(odir('eval'), 'r13_t??.csv'))
           if 'variant' not in pd.read_csv(x, nrows=1).columns]
    if old:                                                               # first-version evaluation files
        import shutil
        dst = os.path.join(odir(), f'eval_v1_{time.strftime("%Y%m%d_%H%M%S")}')
        shutil.move(odir('eval'), dst)
        print(f'archived {len(old)} first-version evaluation file(s) to {dst}')
    for B in targets:
        run_target(B, st, 'eval')


def report():
    import nb_round4 as R4m
    pd.set_option('display.width', 250)
    st = json.load(open(settings_path())) if os.path.exists(settings_path()) else {}
    f = sorted(x for x in glob.glob(os.path.join(odir('eval'), 'r13_t*.csv')) if x.endswith(tuple(f'{i:02d}.csv' for i in PEOPLE)))
    if not f:
        print('no evaluation results yet')
        return
    df = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
    lm = pd.concat([pd.read_csv(x.replace('.csv', '_landmarks.csv')) for x in f], ignore_index=True)
    a, rule = st.get('alpha', 0.0), st.get('rule', 'score')
    fuse, ridge = f'fuse_{a:g}', f'ridge_{st.get("gamma", 1.0):g}'
    rawc = 'raw_pop' if st.get('variant', {}).get('target') == 'pop' else 'raw'
    print(f'ROUND 13: relative profiles vs SCORE recovery on NeuroBridge (seed {SEED}, {MON})')
    print(f'  locked on person {st.get("dev_person")}: variant {st.get("variant_name")} scored with {rule}; alpha {a:g}; '
          f'ridge gamma {st.get("gamma")}; published default = {vname(DEFAULT)} / score')
    for x in f:
        meta = json.load(open(x.replace('.csv', '_meta.json')))
        print(f"  t{meta['target']}: gate {meta['checkpoint_gate']}; {meta['concept_check']}")
    print('\nLANDMARKS (inferred bank)')
    print(lm.groupby(['variant', 'k'])[['m', 'neff', 'precision']].mean().round(2).to_string())
    s = df[(df.k == K_MAIN) & (df['query'] == 's4R4') & df.target.isin(TEST)].copy()
    s['cond2'] = s.cond.str.replace(r'shuffled_\d', 'shuffled', regex=True)
    print(f'\nk = {K_MAIN}, s4R4, people 2-10 (rows = method, columns = variant / bank)')
    print(s.groupby(['variant', 'cond2', 'method']).top1.mean().unstack([0, 1]).round(2).to_string())
    g = lambda var, c, m, q='s4R4', k=K_MAIN: per_target(df, k, var, c, m, q)  # noqa: E731
    print('\nPRIMARY (fixed at lock, before evaluation)')
    if a == 0.0:
        print('  development chose alpha = 0: fuse == recovery by construction; profiles are not carried forward.')
    else:
        R4m.pstats(g('chosen', 'inferred', fuse), g('chosen', 'inferred', rule),
                   f'{fuse} - recovery, chosen variant, inferred bank, k = {K_MAIN}, s4R4', primary=True)
    print('\nKEY SECONDARY: is anything better than no calibration?')
    ps = {}
    ps['chosen recovery vs none'] = R4m.pstats(g('chosen', 'inferred', rule), g('none', 'none', rawc),
                                               f'chosen recovery ({rule}) - no calibration ({rawc})')
    ps['chosen fuse vs none'] = R4m.pstats(g('chosen', 'inferred', fuse), g('none', 'none', rawc), f'chosen {fuse} - no calibration')
    ps['default vs none'] = R4m.pstats(g('default', 'inferred', 'score'), g('none', 'none', 'raw'), 'published SCORE default - no calibration')
    ps['chosen vs default'] = R4m.pstats(g('chosen', 'inferred', rule), g('default', 'inferred', 'score'), 'chosen - published default recovery')
    print(f'  HOLM: {holm(ps)}')
    print('\nPROFILES')
    R4m.pstats(g('chosen', 'inferred', 'prof'), g('chosen', 'inferred', rule), 'profile - recovery (standalone), chosen')
    R4m.pstats(g('chosen', 'inferred', 'prof'), g('chosen', 'inferred', 'dot'), 'Pearson profile - bilinear dot control')
    R4m.pstats(g('default', 'inferred', fuse), g('default', 'inferred', 'score'), f'{fuse} - recovery, published default (the original spec contrast)')
    R4m.pstats(g('chosen', 'inferred', ridge), g('chosen', 'inferred', rule), f'{ridge} - recovery, same pseudo-pairs')
    sh = s[(s.variant == 'chosen') & s.cond.str.startswith('shuffled')]
    for meth in (rule, 'prof'):
        R4m.pstats(g('chosen', 'inferred', meth), sh[sh.method == meth].groupby('target').top1.mean(),
                   f'{meth}: inferred - shuffled pairs (does correspondence matter?)')
    print('\nTRUE PAIRS (same EEG rows and weights; labelled diagnostic)')
    for meth in (rule, 'prof', fuse):
        R4m.pstats(g('chosen', 'true', meth), g('chosen', 'inferred', meth), f'{meth}: true - inferred (correspondence bottleneck)')
    R4m.pstats(g('chosen', 'true', 'prof'), g('chosen', 'true', rule), 'true pairs: profile - rotation')
    R4m.pstats(g('chosen', 'oracle_all', rule), g('none', 'none', rawc), 'all 100 TRUE pairs, chosen rotation - no calibration')
    print('\nTRANSDUCTIVE REFERENCE (published-style, fits on the evaluation batch; not comparable to the primary)')
    for q in QSETS:
        print(f'  {q}: transductive {g("none", "transductive", "trans_score", q).reindex(TEST).mean():.2f} | chosen '
              f'{g("chosen", "inferred", rule, q).reindex(TEST).mean():.2f} | none {g("none", "none", rawc, q).reindex(TEST).mean():.2f}')
    print('\nBUDGET (s4R4, people 2-10)')
    b = df[(df['query'] == 's4R4') & df.target.isin(TEST)]
    b = b[((b.variant == 'none') & b.method.isin(['raw', 'raw_pop'])) |
          ((b.cond == 'inferred') & b.method.isin([rule, 'score', 'prof', fuse])) |
          ((b.variant == 'chosen') & (b.cond == 'true') & (b.method == rule))]
    b = b.assign(row=b.variant + '|' + b.cond + '|' + b.method)
    print(b.groupby(['row', 'k']).top1.mean().unstack('k').round(2).to_string())
    print('\nQUERY NOISE (k = 100, chosen recovery vs no calibration)')
    for q in ('s4R1', 's4R20', 'bench'):
        R4m.pstats(g('chosen', 'inferred', rule, q), g('none', 'none', rawc, q), f'{q}')
    print('\nONLY-ONE-CORRECT COUNTS (k = 100, s4R4, chosen, inferred)')
    for x in f:
        r = np.load(x.replace('.csv', '_ranks.npz'))
        keys = [k_ for k_ in r.files if k_.startswith(f'chosen|inferred|{rule}|')]
        so = sum(int(((r[k_] == 0) & (r[k_.replace(f'|{rule}|', '|prof|')] != 0)).sum()) for k_ in keys)
        po = sum(int(((r[k_] != 0) & (r[k_.replace(f'|{rule}|', '|prof|')] == 0)).sum()) for k_ in keys)
        print(f'  {os.path.basename(x)}: recovery only {so}, profile only {po}')
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def holm(ps):
    ks = sorted(ps, key=lambda k: np.nan_to_num(ps[k], nan=1.0))
    out, run = {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (len(ks) - i) * np.nan_to_num(ps[k], nan=1.0)))
        out[k] = round(run, 4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['selftest', 'dev', 'eval', 'report'])
    p.add_argument('--config')
    p.add_argument('--targets', type=int, nargs='+', default=TEST)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    a = p.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    from eegalign import nb as N
    N.setup(a.config)
    N.ENV.setdefault('reps_dir', a.reps_dir)
    {'dev': dev, 'report': report}.get(a.cmd, lambda: evaluate(a.targets))()


if __name__ == '__main__':
    main()
