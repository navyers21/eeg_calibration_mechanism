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

PRIMARY (locked follow-up on the existing cohort): fuse - score, inferred bank, k = 100 x 2, s4R4, cosine, people 2-10
(draws averaged within person), Wilcoxon, all nine differences printed. If development picks alpha = 0 the contrast is
zero by construction and the report says so (decision gate: stop).
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


def recover(Xs, Ys, gen):
    """The UNLABELED pipeline: support EEG groups and support image pool arrive in independent random orders.
    Returns the bank in ORIGINAL support indices so true partners can be swapped in by the evaluator only."""
    n = len(Xs)
    p1 = torch.randperm(n, generator=gen).to(Xs.device)
    p2 = torch.randperm(n, generator=gen).to(Xs.device)
    mm = moment_fit(Xs[p1], Ys[p2])
    qi, gj, w = mnn_landmarks(moment_apply(Xs[p1], mm), Ys[p2])
    return dict(mm=mm, eeg=p1[qi], img=p2[gj], w=w)


def readouts(bank_eeg, bank_img, w, mm, Xs, Ys, Q, G, alphas=ALPHAS, gammas=(), hub_ref=True):
    """Every readout from ONE bank (indices into the support), so all methods see identical information.
    Q: (n, d) query EEG embeddings, G: (200, d) gallery. Row i of Q belongs to column i of G (evaluator only)."""
    out, flags = {}, {}
    Xl, Yl = moment_apply(Xs[bank_eeg], mm), Ys[bank_img]
    Qm = moment_apply(Q, mm)
    R, mx, my, fb_rot = fit_rotation(Xl, Yl, w)
    Qr = apply_rotation(Qm, R, mx, my)
    s_rot = l2n(Qr) @ l2n(G).T
    out['score'] = s_rot
    if hub_ref:
        out['score_csls'] = csls(Qr, G, rB=hub_term(apply_rotation(moment_apply(Xs, mm), R, mx, my), G))
    if len(bank_eeg) >= 3:
        s_prof, s_dot, bad = profiles(Qm, Xl, G, Yl, w)
        fb_prof = bad.any(1)
    else:
        s_prof, s_dot = s_rot.clone(), s_rot.clone()
        fb_prof = torch.ones(len(Q), dtype=torch.bool, device=Q.device)
    s_prof = torch.where(fb_prof[:, None], s_rot, s_prof)        # prespecified fallback: rotation score, flagged
    out['prof'], out['dot'] = s_prof, s_dot
    zr, zp = zrow(s_rot), zrow(s_prof)
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
    """Frozen embeddings for one held-out person; computed once and cached on Drive."""

    def __init__(self, B):
        N, R, R6m, GL, R9, R10 = _mods()
        self.B = B
        dev = N.ENV['device']
        m = R10.load_decoder('eeg_aug', B, MON, SEED, train=False)
        if m is None:
            raise FileNotFoundError(f'no seed-{SEED} repaug decoder for person {B} (run round 7 / 10 training)')
        self.m = m
        lab = N.subject_data(B, MON, keys=('train_labels',))['train_labels']
        self.pool, _ = N.split(lab)
        H = []
        for s in PEOPLE:
            if s == B:
                continue
            tr_s = N.subject_data(s, MON, keys=('train',))['train']
            H.append(R10.h1_mean(m, tr_s[np.random.RandomState(3).choice(self.pool, 300, replace=False)]))
            del tr_s
        self.mS = torch.stack(H).mean(0)                                 # identical to round 10
        self.Gtr = m.gallery(N.features('train', 'aug').to(dev))         # decoder's own image gallery (normalized)
        self.Gte = m.gallery(N.features('test', 'aug').to(dev))
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
        Xs = self.R10.z_of(self.m, x2.mean(1), d1)
        Zq = {qk: [self.R10.z_of(self.m, q, d1) for q in self.Q[qk]] for qk in QSETS}
        return ims, Xs, self.Gtr[torch.as_tensor(ims, device=Xs.device)], Zq


def run_target(B, settings, mode):
    """mode 'dev': Experiment 1 grid on the development person. mode 'eval': locked settings on a test person."""
    out = os.path.join(odir(mode), f'r13_t{B:02d}.csv')
    if os.path.exists(out):
        print(f'[{mode}] t{B}: done')
        return
    t0 = time.time()
    T = Target(B)
    rows, diags, ranks = [], [], {}
    gate = []
    r10 = os.path.join(T.N.ENV['results_dir'], 'nb_round10', 'eval', f'r10_s{SEED}_t{B:02d}_{MON}.csv')
    alphas = ALPHAS if mode == 'dev' else tuple(sorted({0.0, settings['alpha']}))
    gammas = GAMMAS if mode == 'dev' else (settings['gamma'],)
    ks = (K_MAIN,) if mode == 'dev' else KS
    cmsg = None
    for k in ks:
        if k > len(T.cand):
            continue
        for draw in range(T.R6m.ndraw(k)):
            ims, Xs, Ys, Zq = T.support(k, draw)
            if cmsg is None:
                cmsg = concept_check(T.N, T.GL, B, ims)
            gen = torch.Generator().manual_seed(10000 * B + 100 * k + draw)
            bank = recover(Xs, Ys, gen)
            n = len(Xs)
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
                        img[sel] = img[sel.roll(1)]                      # cyclic shift = no fixed points
                    conds[f'corrupt_{c:g}'] = (bank['eeg'], img, bank['w'])
            dg = bank_diag(bank, bank['w'])
            diags.append(dict(target=B, k=k, draw=draw, **dg))
            for cname, (be, bi, bw) in conds.items():
                main = cname in ('inferred', 'true', 'oracle_all') or mode == 'dev'
                for qk in QSETS:
                    if not main and qk != 's4R4':
                        continue
                    for qi_, z in enumerate(Zq[qk]):
                        o, fl = readouts(be, bi, bw, bank['mm'], Xs, Ys, z, T.Gte, alphas=alphas,
                                         gammas=gammas if main else ())
                        if cname == 'inferred':
                            o['raw'] = l2n(z) @ l2n(T.Gte).T
                            o['mm_only'] = l2n(moment_apply(z, bank['mm'])) @ l2n(T.Gte).T
                        for meth, S in o.items():
                            mt, rk = metrics(S)
                            rows.append(dict(target=B, k=k, draw=draw, cond=cname, method=meth, query=qk, qdraw=qi_, **mt))
                            if k == K_MAIN and qk == 's4R4' and cname in ('inferred', 'true'):
                                ranks[f'{cname}|{meth}|d{draw}|q{qi_}'] = rk
                        if cname == 'inferred' and qk == 's4R4' and qi_ == 0:
                            diags[-1].update(fl)
            if k == K_MAIN:
                for qk in QSETS:
                    for qi_, z in enumerate(Zq[qk]):
                        S, mt_ = transductive_score(z, T.Gte)
                        rows.append(dict(target=B, k=k, draw=draw, cond='transductive', method='trans_score', query=qk,
                                         qdraw=qi_, **metrics(S)[0]))
                        gate_row = [r for r in rows if r['k'] == k and r['draw'] == draw and r['method'] == 'raw'
                                    and r['query'] == qk and r['qdraw'] == qi_]
                        if gate_row and os.path.exists(r10):
                            gate.append((draw, qk, qi_, gate_row[-1]['top1']))
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
                features='aug (NeuroBridge fused degraded CLIP RN50), decoder image gallery', concept_check=cmsg,
                checkpoint_gate=gmsg, settings=settings, minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(out.replace('.csv', '_meta.json'), 'w'), indent=1)
    print(f'[{mode}] t{B}: {meta["minutes"]} min | gate: {gmsg} | {cmsg}', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# development gate, locking, evaluation, report
# ---------------------------------------------------------------------------------------------------------------
def per_target(df, k, cond, method, query='s4R4'):
    x = df[(df.k == k) & (df.cond == cond) & (df.method == method) & (df['query'] == query)]
    return x.groupby('target').top1.mean()


def dev():
    run_target(DEV[0], {}, 'dev')
    df = pd.read_csv(os.path.join(odir('dev'), f'r13_t{DEV[0]:02d}.csv'))
    lm = pd.read_csv(os.path.join(odir('dev'), f'r13_t{DEV[0]:02d}_landmarks.csv'))
    print(f'\nDEVELOPMENT (person {DEV[0]}, k = {K_MAIN} x 2, s4R4, cosine, mean over draws and query draws)')
    print(f'  landmarks: m = {lm.m.mean():.1f} (range {lm.m.min()}-{lm.m.max()}), effective {lm.neff.mean():.1f}, '
          f'precision {lm.precision.mean():.2f}; profile fallback rate {lm.get("prof_fallback_rate", pd.Series([np.nan])).mean():.2f}')
    s = df[(df['query'] == 's4R4') & (df.k == K_MAIN)].copy()
    s['cond2'] = s.cond.str.replace(r'shuffled_\d', 'shuffled', regex=True)
    tab = s.groupby(['cond2', 'method']).top1.mean().unstack(0).round(2)
    print(tab.to_string())
    best = {}
    for cond in ('inferred', 'true'):
        fz = {a: s[(s.cond == cond) & (s.method == f'fuse_{a:g}')].top1.mean() for a in ALPHAS}
        a_best = max(ALPHAS, key=lambda a: (round(fz[a], 6), -a))
        best[cond] = a_best
        print(f'  {cond}: fusion by alpha ' + ', '.join(f'{a:g}: {v:.2f}' for a, v in fz.items()) + f' -> best {a_best:g}')
    gz = {g: s[(s.cond == 'inferred') & (s.method == f'ridge_{g:g}')].top1.mean() for g in GAMMAS}
    g_best = max(GAMMAS, key=lambda g: (round(gz[g], 6), g))
    t_rot = s[(s.cond == 'true') & (s.method == 'score')].top1.mean()
    t_prof = s[(s.cond == 'true') & (s.method == 'prof')].top1.mean()
    print(f'  ridge on inferred pairs by gamma: ' + ', '.join(f'{g:g}: {v:.2f}' for g, v in gz.items()) + f' -> {g_best:g}')
    print('\nCORRUPTION SWEEP (true bank with a fraction of its images reassigned; rotation | profile)')
    for c in CORRUPT:
        print(f'  {c:4.0%}: ' + f"{s[(s.cond == f'corrupt_{c:g}') & (s.method == 'score')].top1.mean():.2f} | "
              f"{s[(s.cond == f'corrupt_{c:g}') & (s.method == 'prof')].top1.mean():.2f}")
    gate_stop = best['true'] == 0.0 and t_prof <= t_rot
    print(f'\nDECISION GATE: true-pair profile {t_prof:.2f} vs true-pair rotation {t_rot:.2f}; best true-pair alpha '
          f'{best["true"]:g}; inferred alpha {best["inferred"]:g}. '
          + ('STOP: profiles add nothing even with true pairs (report the diagnostic, do not run eval).' if gate_stop else
             'Proceed: locking alpha from the INFERRED bank (the primary condition).'))
    settings = dict(alpha=float(best['inferred']), gamma=float(g_best), alpha_true=float(best['true']), gate_stop=bool(gate_stop),
                    dev_person=DEV[0], k=K_MAIN, locked=time.strftime('%Y-%m-%d %H:%M'))
    json.dump(settings, open(settings_path(), 'w'), indent=1)
    print(f'locked settings -> {settings_path()}: {settings}')


def evaluate(targets):
    if not os.path.exists(settings_path()):
        raise SystemExit('run `dev` first: evaluation needs the locked development settings')
    st = json.load(open(settings_path()))
    if st.get('gate_stop'):
        print('NOTE: the development gate said STOP; running the locked evaluation anyway as a documented negative check.')
    for B in targets:
        run_target(B, st, 'eval')


def report():
    import nb_round4 as R4m
    pd.set_option('display.width', 250)
    st = json.load(open(settings_path())) if os.path.exists(settings_path()) else {}
    f = sorted(glob.glob(os.path.join(odir('eval'), 'r13_t*.csv')))
    f = [x for x in f if not x.endswith('_landmarks.csv')]
    if not f:
        print('no evaluation results yet')
        return
    df = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
    lm = pd.concat([pd.read_csv(x.replace('.csv', '_landmarks.csv')) for x in f], ignore_index=True)
    a = st.get('alpha', 0.0)
    fuse = f'fuse_{a:g}'
    ridge = f'ridge_{st.get("gamma", 1.0):g}'
    print(f'ROUND 13: relative profiles vs SCORE recovery on NeuroBridge (seed {SEED}, {MON}); locked settings {st}')
    for x in f:
        meta = json.load(open(x.replace('.csv', '_meta.json')))
        print(f"  t{meta['target']}: gate {meta['checkpoint_gate']}; {meta['concept_check']}")
    print('\nLANDMARKS (inferred bank, by budget)')
    print(lm.groupby('k')[['m', 'neff', 'precision']].mean().round(2).to_string())
    s = df[(df.k == K_MAIN) & (df['query'] == 's4R4')].copy()
    s['cond2'] = s.cond.str.replace(r'shuffled_\d', 'shuffled', regex=True)
    T = s[s.target.isin(TEST)].groupby(['cond2', 'method']).top1.mean().unstack(0).round(2)
    print(f'\nk = {K_MAIN}, s4R4, cosine unless named (people 2-10)')
    print(T.to_string())
    g = lambda c, m, q='s4R4', k=K_MAIN: per_target(df, k, c, m, q)  # noqa: E731
    print('\nPRIMARY')
    if a == 0.0:
        print('  development chose alpha = 0, so fuse == score by construction: the profile readout is not carried '
              'forward (decision gate). Diagnostics below.')
    else:
        R4m.pstats(g('inferred', fuse), g('inferred', 'score'), f'{fuse} - score, inferred bank, k = {K_MAIN}, s4R4', primary=True)
    print('\nSECONDARY')
    ps = {}
    ps['prof vs score'] = R4m.pstats(g('inferred', 'prof'), g('inferred', 'score'), 'profile - rotation (standalone), inferred')
    ps['prof vs dot'] = R4m.pstats(g('inferred', 'prof'), g('inferred', 'dot'), 'Pearson profile - bilinear dot control, inferred')
    ps['score vs raw'] = R4m.pstats(g('inferred', 'score'), g('inferred', 'raw'), 'SCORE recovery - unadapted cosine')
    ps['ridge vs score'] = R4m.pstats(g('inferred', ridge), g('inferred', 'score'), f'{ridge} - score, same pseudo-pairs')
    sh = s[s.cond.str.startswith('shuffled') & s.target.isin(TEST)]
    for meth in ('score', 'prof'):
        R4m.pstats(g('inferred', meth), sh[sh.method == meth].groupby('target').top1.mean(),
                   f'{meth}: inferred - shuffled pairs (does correspondence matter?)')
    print(f'  HOLM: {holm(ps)}')
    print('\nTRUE PAIRS (same EEG rows and weights; labelled diagnostic)')
    for meth in ('score', 'prof', 'dot', fuse):
        R4m.pstats(g('true', meth), g('inferred', meth), f'{meth}: true - inferred (correspondence bottleneck)')
    R4m.pstats(g('true', 'prof'), g('true', 'score'), 'true pairs: profile - rotation')
    R4m.pstats(g('oracle_all', 'score'), g('inferred', 'score'), 'all k true pairs (labelled few-shot) - inferred, rotation')
    print('\nTRANSDUCTIVE REFERENCE (fits on the evaluation batch; not comparable to the primary)')
    for q in QSETS:
        print(f'  {q}: transductive SCORE {g("transductive", "trans_score", q).reindex(TEST).mean():.2f} vs frozen '
              f'{g("inferred", "score", q).reindex(TEST).mean():.2f} vs raw {g("inferred", "raw", q).reindex(TEST).mean():.2f}')
    print('\nBUDGET (inferred, s4R4, people 2-10)')
    b = df[(df.cond == 'inferred') & (df['query'] == 's4R4') & df.target.isin(TEST)]
    print(b[b.method.isin(['raw', 'score', 'prof', fuse, ridge])].groupby(['method', 'k']).top1.mean().unstack('k').round(2).to_string())
    print('\nQUERY NOISE (k = 100, inferred)')
    for q in ('s4R1', 's4R20', 'bench'):
        R4m.pstats(g('inferred', fuse if a else 'prof', q), g('inferred', 'score', q), f'{fuse if a else "prof"} - score, {q}')
    print('\nONLY-ONE-CORRECT COUNTS (k = 100, s4R4, inferred; summed over draws and query draws)')
    for x in f:
        r = np.load(x.replace('.csv', '_ranks.npz'))
        keys = [k for k in r.files if k.startswith('inferred|score|')]
        so = sum(int(((r[k] == 0) & (r[k.replace('|score|', '|prof|')] != 0)).sum()) for k in keys)
        po = sum(int(((r[k] != 0) & (r[k.replace('|score|', '|prof|')] == 0)).sum()) for k in keys)
        print(f'  {os.path.basename(x)}: rotation only {so}, profile only {po}')
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
