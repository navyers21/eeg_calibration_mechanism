"""Round 12: learn WHERE a new person's rotation may move, not how it moves.

    python scripts/nb_round12.py eval   --config C        # needs round 11's pair decoders; no new training
    python scripts/nb_round12.py report --config C

Why. Round 9: the useful correction is a rotation (+3.44 over a shift), and editing only the 16 highest-variance output
directions was no better than random ones (+0.42). Round 11: a single rotation learned from other people adds nothing,
because people's rotations barely agree in direction (cosine 0.05). But round 9 also found that the DIRECTIONS in which
outputs move overlap across people (0.17, vs 0.016 for random). So previous people may tell us where a new person's
correction is reproducible, even though they cannot tell us its sign or size. With few calibration images, the
rotation is pulled hard toward "no rotation" everywhere (CV picks lambda 10-100); a weaker pull along reproducible
directions should keep more of the real correction there and less of the noise elsewhere.

Coordinates and sign. Each source's rotation lives in its own pair decoder; it is carried into the target decoder's
output basis by a rotation fitted between the two decoders' POPULATION responses on the same images (no person-specific
data; fit quality is reported). The pooled quantity sym(D1 D2') is unchanged if a person's whole correction flips sign
(both halves flip), so pooling keeps WHICH directions move reproducibly, and each target supplies its own sign and size.

Reproducible directions (source people only, round 11's leave-two-out pair decoders, so every source is new to the
decoder it is measured in and nothing from the target is used). For target B and each source j, in decoder D_Bj:
  - j's outputs (label-free layer-1 offset, 2-presentation averages) and the 8-person population responses for
    4,000 pool images, split into two halves by CONCEPT;
  - a rotation fitted on each half (shrunk toward the identity, strength by CV on that half);
  - M_j = sym(D1 D2'), D = R - I: large where the two halves move the same direction the same way;
  - mapped into the target decoder's space (round 11's population-only map Q_j) and summed over the 9 sources.
  U_rep = the 16 leading eigenvectors of the sum ("correction reproducibility", not variance).
Comparison bases, same rank 16, same penalty form: U_pca (16 top-variance directions of the population responses;
round 9's failed edit used these), U_diff (16 leading directions of the person-minus-population DIFFERENCES of the
source people: a matched-rank CONTROL in the spirit of the patching analysis, computed source-only. Round 3's patching
controls showed that the correct same-image donor values mattered more than any particular basis, so patching is not
treated as having identified the right calibration basis; the motivation here is the reproducibility of FITTED
rotations), U_rand (random).

Calibration (seed-2026 repaug decoder, population gallery, earlier-day calibration, round 6 draws, true image pairs):
  zero      layer-1 offset only
  ordinary  rotation with no shrinkage
  uniform   rotation shrunk toward identity, strength by 5-fold CV over 13 settings including none (round 10/11 method)
  dir_*     rotation shrunk toward identity with a WEAKER pull (alpha in {0.1, 0.3}) along the basis U and the full pull
            elsewhere; 13 settings (strength x alpha, plus none), chosen by the same 5-fold CV: the same tuning budget
            as uniform. Every direction can still move given enough evidence; sign and size come from the target alone.
  One scoring path for every arm; alpha = 1 reproduces the uniform rotation exactly (checked and printed). Note on
  scale: the rigid map is (z - x_mean) R g + y_mean, so g scales only the centered part before the population mean is
  added back; that is why it changes cosine rankings, unlike a scalar on the final vector.
  rigid     rounds 6-7 reference
k = 16 / 25 / 50 / 100 / 200 / 500 images x 2 presentations (16 = the SA-LBC comparison point). Queries: session 4 at
1 / 4 / 20 presentations and the benchmark; each query scored on its own (cos) and batch CSLS. The correction is always
fitted on training-concept images and scored on the 200 disjoint test concepts (a different gallery), so every number is
the "fit on one gallery, score on another" transfer test.

PRIMARY (fixed before running; a locked follow-up on the existing cohort): dir_rep - uniform, k = 100 images x 2
presentations, earlier-session calibration, s4R4, each query scored on its own, people 2-10 (draws averaged within
person), Wilcoxon; all nine paired differences printed. Practical bar: +1 point. Secondary: the same at k = 500 (does the
advantage shrink with more evidence?), dir_rep - ordinary, uniform - ordinary.
Secondary (Holm over three): dir_rep - dir_rand (any low-rank preference?), dir_rep - dir_pca (reproducibility beyond
variance?), dir_rep - dir_diff (beyond patching-style difference directions?). Then every k, batch CSLS, other queries.
Claim wording: a population-informed rotation prior, motivated by the interpretability analysis; it counts as
'where you regularize matters' only if dir_rep beats uniform AND dir_rand. Diagnostic: how much of the target's own 500-image
rotation lies in each basis (privileged; for interpretation only).

Added branches (each with its own fixed-in-advance comparison; none changes the primary):
  REPEAT-AGREEMENT WEIGHTING. Each calibration image's two presentations are encoded separately; their cosine agreement
    (after removing the person's calibration mean) sets its weight in the rotation fit: bottom quartile 0.5, the rest 1,
    all images kept, same strength as the unweighted fit. Validity check first, on source people new to their pair
    decoder: does agreement predict a smaller angle to the independent population anchor, also after partialling out the
    response norm? Prespecified: agreement weights - the same weights shuffled across images, k = 100. Then a 2 x 2 with
    the directional prior.
  CNN FINE-TUNING TOWARD POPULATION RESPONSES (command cnnft; round 10's CNN decoders, seeds 2025-2027, k = 500, the same
    draws): ordinary fine-tuning; fine-tuning with an added pull toward the measured population responses of the
    calibration images (round 7's ft_anchor recipe); each followed by the rotation. Source EEG for test images is never
    used. Round 7's "+2.48 for the anchor loss" compared different readouts (population vs image gallery); with the same
    readout it added ~0. So every comparison here keeps the downstream identical. Prespecified: ft_pop + rotation -
    ft + rotation, s4R4, cos; same with the population gallery and no rotation; single presentation secondary. Optional:
    a backbone follow-up, not a new calibration method.
  Dropped: choosing calibration images by reliability / diversity (round 3, X4: all panel types within 0.3 points of
    random at 5-25 images).
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
from eegalign import nb as N  # noqa: E402
import nb_round3 as R  # noqa: E402
import nb_round4 as R4m  # noqa: E402
import nb_round5 as R5m  # noqa: E402
import nb_round6 as R6m  # noqa: E402
import nb_gallery as GL  # noqa: E402
import nb_round9 as R9  # noqa: E402
import nb_round10 as R10  # noqa: E402
import nb_round11 as R11  # noqa: E402
import nb_round7 as R7  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
QK = ['s4R1', 's4R4', 's4R20', 'bench']
SEED = 2026
KS = [16, 25, 50, 100, 200, 500]
CS = 3
RANK = 16
ALPHAS = (0.1, 0.3)
UNI = [0.0] + [float(x) for x in np.logspace(-2, 3, 12)]          # 13 settings; 0 = ordinary rotation
BASES = ('rep', 'pca', 'diff', 'rand')


def dev():
    return N.ENV['device']


def odir(sub=''):
    d = os.path.join(N.ENV['results_dir'], 'nb_round12', sub)
    os.makedirs(d, exist_ok=True)
    return d


def e_rot_dir(Xs, Ys, lam, alpha, U):
    """Rotation with a direction-dependent pull toward the identity: weaker (alpha) along U."""
    D = Xs.shape[1]
    I_ = torch.eye(D, device=Xs.device)
    P = I_ if U is None else (I_ - U @ U.T) + alpha * (U @ U.T)
    return R11.e_rot_prior(Xs, Ys, lam, P)


def repeat_agreement(z1, z2):
    """Cosine between a person's two single-presentation outputs per image, after removing their calibration mean."""
    m = torch.cat([z1, z2]).mean(0)
    return torch.nn.functional.cosine_similarity(z1 - m, z2 - m, dim=1)


def _rank(x):
    return torch.as_tensor(pd.Series(x.detach().cpu().numpy()).rank().values, dtype=torch.float64)


def spearman_partial(a, b, c):
    """Spearman(a, b), and the same with c (response norm) partialled out of both ranks."""
    ra, rb, rc = _rank(a), _rank(b), _rank(c)
    corr = lambda x, y: float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()).norm() * (y - y.mean()).norm() + 1e-12))  # noqa: E731
    res = lambda x: x - (((x - x.mean()) * (rc - rc.mean())).sum() / ((rc - rc.mean()) ** 2).sum()) * (rc - rc.mean()) - x.mean()  # noqa: E731
    return corr(ra, rb), corr(res(ra), res(rb))


def agreement_weights(agr):
    """Conservative rule: bottom agreement quartile weighted 0.5, the rest 1; every image kept."""
    w = torch.ones_like(agr)
    w[agr <= torch.quantile(agr, 0.25)] = 0.5
    return w


def e_rot_w(Xs, Ys, lam, alpha, U, w):
    """e_rot_dir with per-image weights in the means and the cross-covariance (weights rescaled to mean 1)."""
    w = (w / w.mean()).float()[:, None]
    xm, ym = (w * Xs).sum(0) / w.sum(), (w * Ys).sum(0) / w.sum()
    Xc, Yc = Xs - xm, Ys - ym
    D = Xs.shape[1]
    I_ = torch.eye(D, device=Xs.device)
    P = I_ if U is None else (I_ - U @ U.T) + alpha * (U @ U.T)
    s_ = float(((w * Xc) * Xc).sum()) / D + 1e-8
    Uu, _, Vh = R.safe_svd((w * Xc).T @ Yc + lam * s_ * P)
    Rm = (Uu @ Vh).float()
    return lambda Z: (Z - xm) @ Rm + ym


def top_eig(M, r=RANK):
    ev, V = torch.linalg.eigh((M + M.T) / 2)
    return V[:, -r:].float(), ev.flip(0)


def build_bases(B, mon, ctx, cache):
    c = ctx.c
    pool = c.pool
    ims = np.sort(np.random.RandomState(11000).choice(pool, R11.N_IMG, replace=False))
    sub300 = np.random.RandomState(3).choice(pool, 300, replace=False)
    lab = c.labels[ims]
    con = np.unique(lab)
    h1c = set(np.random.RandomState(12000).choice(con, len(con) // 2, replace=False))
    half = np.array([x in h1c for x in lab])
    PB = c.pool_z[torch.as_tensor([ctx.pool_pos[int(i)] for i in ims], device=dev())]
    Dm = PB.shape[1]
    Id = torch.eye(Dm, device=dev())
    Mrep = torch.zeros(Dm, Dm, device=dev(), dtype=torch.float64)
    Cdiff = torch.zeros(Dm, Dm, device=dev(), dtype=torch.float64)
    agree, vc = [], []
    for j in PEOPLE:
        if j == B:
            continue
        p = R11.pair_path(B, j, mon)
        if not os.path.exists(p):
            print(f'   missing pair decoder ({B}, {j}): run round 11 train first')
            return None, None
        Dn = N.load_model(p)[0].to(dev()).eval().requires_grad_(False)
        Zp, H = [], []
        for q in PEOPLE:
            if q in (B, j):
                continue
            if q not in cache:
                tr = N.subject_data(q, mon, keys=('train',))['train']
                cache[q] = (np.asarray(tr[ims], np.float32), np.asarray(tr[sub300], np.float32))
                del tr
            Zp.append(R10.z_of(Dn, cache[q][0]))
            H.append(R10.h1_mean(Dn, cache[q][1]))
        PD, mSD = torch.stack(Zp).mean(0), torch.stack(H).mean(0)
        rj = np.asarray(R.load_reps(j, mon)[ims][:, :2], np.float32)
        d1 = mSD - R10.h1_mean(Dn, rj.reshape(-1, *rj.shape[2:]))
        Xj = R10.z_of(Dn, rj.mean(1), d1)
        # repeat-agreement validity check (source person j, new to this decoder): does agreement between j's two
        # presentations predict a smaller angle between j's averaged output and the independent population anchor?
        z1, z2 = R10.z_of(Dn, rj[:, 0], d1), R10.z_of(Dn, rj[:, 1], d1)
        agr = repeat_agreement(z1, z2)
        err = 1 - torch.nn.functional.cosine_similarity(Xj - Xj.mean(0), PD - PD.mean(0), dim=1)
        nrm_ = (Xj - Xj.mean(0)).norm(dim=1)
        vc.append(spearman_partial(agr, err, nrm_))
        lam_q, _ = R9.cv_pick(lambda a, b, l: R11.e_rot_prior(a, b, l, Id), PD, PB, R11.REL, seed=j)
        Q = R11.e_rot_prior(PD, PB, lam_q, Id).R
        Ds = []
        for h in (half, ~half):
            ht = torch.as_tensor(np.nonzero(h)[0], device=dev())
            lam_h, _ = R9.cv_pick(lambda a, b, l: R11.e_rot_prior(a, b, l, Id), Xj[ht], PD[ht], R11.REL, seed=j)
            Ds.append(R11.e_rot_prior(Xj[ht], PD[ht], lam_h, Id).R - Id)
        Mj = (Ds[0] @ Ds[1].T + Ds[1] @ Ds[0].T) / 2
        Mrep += (Q.T @ Mj @ Q).double()
        agree.append(float((Ds[0] * Ds[1]).sum() / (Ds[0].norm() * Ds[1].norm() + 1e-12)))
        E = (Xj - Xj.mean(0)) - (PD - PD.mean(0))
        Cdiff += (Q.T @ (E.T @ E / len(E)) @ Q).double()
        del Dn, rj
    U_rep, ev_rep = top_eig(Mrep)
    U_diff, _ = top_eig(Cdiff)
    P = c.pool_z - c.pool_z.mean(0)
    U_pca, _ = top_eig((P.T @ P / len(P)).double())
    g = torch.Generator(device='cpu').manual_seed(700 + B)
    U_rand, _ = torch.linalg.qr(torch.randn(Dm, RANK, generator=g))
    bases = {'rep': U_rep, 'pca': U_pca, 'diff': U_diff, 'rand': U_rand.to(dev())}
    pos = ev_rep.clamp_min(0)
    ov = lambda A, Bm: float(((A.T @ Bm) ** 2).sum() / RANK)  # noqa: E731
    info = dict(target=B, split_half_agreement=float(np.mean(agree)),
                rep_top16_share=float(pos[:RANK].sum() / (pos.sum() + 1e-12)),
                overlap_rep_pca=ov(U_rep, U_pca), overlap_rep_diff=ov(U_rep, U_diff), overlap_rep_rand=ov(U_rep, bases['rand']),
                agree_err_spearman=float(np.mean([v[0] for v in vc])), agree_err_partial_norm=float(np.mean([v[1] for v in vc])))
    return bases, info


def eval_target(B, mon, cache):
    out = os.path.join(odir('eval'), f'r12_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[eval] t{B}: done')
        return
    R5m.SEED = SEED
    if not os.path.exists(N.ckpt_path('repaug', mon, B, SEED, 'last')):
        print(f'[eval] t{B}: no seed-{SEED} repaug decoder, skipping')
        return
    t0 = time.time()
    ctx = R6m.Ctx('repaug', B, mon)
    c = ctx.c
    bases, info = build_bases(B, mon, ctx, cache)
    if bases is None:
        return
    print(f'   bases ({time.time() - t0:.0f}s): {info}', flush=True)
    tr = c.ses['train']
    reps = R.load_reps(B, mon)
    cand = np.array([i for i in c.pool if (tr[i] == CS).sum() == 2])
    grid_dir = [(0.0, 1.0)] + [(l, a) for l in R11.REL for a in ALPHAS]     # 13 settings, same budget as uniform
    rows = []
    for k in KS:
        for draw in range(R6m.ndraw(k)):
            ims = GL.draw_images(cand, k, draw)
            rr = np.stack([np.nonzero(tr[i] == CS)[0] for i in ims])
            x2 = np.asarray(reps[ims[:, None], rr], np.float32)
            d1 = c.mS['h1'] - R10.h1_mean(c.f0, x2.reshape(-1, *x2.shape[2:]))
            Xs = R10.z_of(c.f0, x2.mean(1), d1)
            Ys = c.pool_z[torch.as_tensor([ctx.pool_pos[int(i)] for i in ims], device=dev())]
            Zq = c.zq(d1)
            lam_u, _ = R9.cv_pick(lambda a, b, l: e_rot_dir(a, b, l, 1.0, None), Xs, Ys, UNI, seed=draw)
            cells = [('zero', None, {}), ('ordinary', e_rot_dir(Xs, Ys, 0.0, 1.0, None), dict(lam=0.0)),
                     ('uniform', e_rot_dir(Xs, Ys, lam_u, 1.0, None), dict(lam=lam_u)),
                     ('rigid', R9.e_procrustes(Xs, Ys, 1.0), {})]
            if k == KS[0] and draw == 0:
                # one scoring path: alpha = 1 (any basis) and the round-10 rotation must give the same map at equal lambda
                zt = Zq['s4R4'][0]
                ref = R9.e_procrustes(Xs, Ys, 10.0, False)(zt)
                chk = max(float((e_rot_dir(Xs, Ys, 10.0, 1.0, bases['rep'])(zt) - ref).abs().max()),
                          float((e_rot_dir(Xs, Ys, 10.0, 1.0, None)(zt) - ref).abs().max()))
                info['scoring_path_max_diff'] = chk
                print(f'   scoring-path check (alpha = 1 vs round-10 rotation, lambda 10): max |diff| {chk:.2e}', flush=True)
            bests = {}
            for bn in BASES:
                U = bases[bn]
                best, _ = R9.cv_pick(lambda a, b, la, U=U: e_rot_dir(a, b, la[0], la[1], U), Xs, Ys, grid_dir, seed=draw)
                bests[bn] = best
                cells.append((f'dir_{bn}', e_rot_dir(Xs, Ys, best[0], best[1], U), dict(lam=best[0], alpha=best[1])))
            if k in (100, 500):
                # repeat-agreement weighting: same strength as the unweighted fit, so only the weights differ
                agr = repeat_agreement(R10.z_of(c.f0, x2[:, 0], d1), R10.z_of(c.f0, x2[:, 1], d1))
                w = agreement_weights(agr)
                ws = w[torch.as_tensor(np.random.RandomState(14000 + draw).permutation(len(w)), device=w.device)]
                br = bests['rep']
                cells += [('w_agree', e_rot_w(Xs, Ys, lam_u, 1.0, None, w), dict(lam=lam_u)),
                          ('w_shuffle', e_rot_w(Xs, Ys, lam_u, 1.0, None, ws), dict(lam=lam_u)),
                          ('w_agree_dir', e_rot_w(Xs, Ys, br[0], br[1], bases['rep'], w), dict(lam=br[0], alpha=br[1])),
                          ('w_shuffle_dir', e_rot_w(Xs, Ys, br[0], br[1], bases['rep'], ws), dict(lam=br[0], alpha=br[1]))]
            for meth, mp, meta in cells:
                for qk in QK:
                    for qi, z0 in enumerate(Zq[qk]):
                        z = z0 if mp is None else mp(z0)
                        for rule in ('cos', 'csls_batch'):
                            rows.append(dict(k=k, draw=draw, method=meth, rule=rule, query=qk, qdraw=qi,
                                             lam=meta.get('lam', np.nan), alpha=meta.get('alpha', np.nan),
                                             top1=R6m.top1(R6m.smat(z, ctx.Ganc, rule))))
            if k == 500 and draw == 0:
                Dt = e_rot_dir(Xs, Ys, 0.1, 1.0, None).R - torch.eye(Xs.shape[1], device=dev())
                tot = float((Dt ** 2).sum()) + 1e-12
                for bn in BASES:
                    U = bases[bn]
                    info[f'target_motion_in_{bn}'] = float(((U.T @ Dt) ** 2).sum() + ((Dt @ U) ** 2).sum()) / (2 * tot)
        print(f'   k={k} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows).groupby(['k', 'draw', 'method', 'rule', 'query']).agg(
        top1=('top1', 'mean'), lam=('lam', 'first'), alpha=('alpha', 'first')).reset_index()
    df['target'], df['montage'] = B, mon
    df.to_csv(out, index=False)
    with open(out.replace('.csv', '_bases.json'), 'w') as f:
        json.dump(info, f)
    print(f'[eval] t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


def evaluate(targets, mon='posterior17'):
    cache = {}
    for B in targets:
        print(f'[eval] target {B}', flush=True)
        eval_target(B, mon, cache)


# ---------------------------------------------------------------------------------------------------------------
# CNN fine-tuning toward population responses (round 10's CNN decoders, k = 500)
# ---------------------------------------------------------------------------------------------------------------
def cnnft_target(B, mon, seed):
    out = os.path.join(odir('cnnft'), f'cnnft_s{seed}_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[cnnft] seed {seed} t{B}: done')
        return
    m = R10.load_decoder('cnn_mid', B, mon, seed, train=False)
    if m is None:
        print(f'[cnnft] seed {seed} t{B}: no CNN decoder (round 10), skipping')
        return
    t0 = time.time()
    Q, ses = R9._queries(B, mon)
    ftr, fte = R9.tgt_feats('mid', 'train').to(dev()), R9.tgt_feats('mid', 'test').to(dev())
    pool, _ = N.split(N.subject_data(B, mon, keys=('train_labels',))['train_labels'])
    Zs, H = [], []
    for s_ in PEOPLE:
        if s_ == B:
            continue
        tr_s = N.subject_data(s_, mon, keys=('train',))['train']
        Zs.append(R10.z_of(m, tr_s[pool]))
        H.append(R10.h1_mean(m, tr_s[np.random.RandomState(3).choice(pool, 300, replace=False)]))
        del tr_s
    pool_z, mS = torch.stack(Zs).mean(0), torch.stack(H).mean(0)
    Bm, xm, ym = R10.X.ridge_svd(ftr[torch.as_tensor(pool, device=dev())], pool_z)
    Ganc = R10.nrm((fte - xm) @ Bm + ym)
    pos_of = {int(i): j for j, i in enumerate(pool)}
    tr = ses['train']
    reps = R.load_reps(B, mon)
    cand = np.array([i for i in pool if (tr[i] == CS).sum() == 2])
    rows = []
    for draw in range(R6m.ndraw(500)):
        ims = GL.draw_images(cand, 500, draw)
        rr = np.stack([np.nonzero(tr[i] == CS)[0] for i in ims])
        x2 = np.asarray(reps[ims[:, None], rr], np.float32)
        Ys = pool_z[torch.as_tensor([pos_of[int(i)] for i in ims], device=dev())]
        variants = {'ft': R6m.finetune(m, x2, ims, ftr, seed=draw),
                    'ft_pop': R7.finetune_anchor(m, x2, ims, ftr, Ys, seed=draw)}
        for nm, mm in variants.items():
            with torch.no_grad():
                Zc = mm(torch.as_tensor(x2.mean(1), device=dev()))
                Gi = mm.gallery(fte)
            lam, _ = R9.cv_pick(lambda a, b, l: R11.e_rot_prior(a, b, l, torch.eye(Zc.shape[1], device=dev())), Zc, Ys, UNI, seed=draw)
            rot = R11.e_rot_prior(Zc, Ys, lam, torch.eye(Zc.shape[1], device=dev()))
            for qk in QK:
                for qi, q in enumerate(Q[qk]):
                    with torch.no_grad():
                        z = mm(torch.as_tensor(q, device=dev()))
                    for meth, zz, G in ((nm, z, Gi), (f'{nm}+rot', rot(z), Ganc), (f'{nm}|pop_gallery', z, Ganc)):
                        for rule in ('cos', 'csls_batch'):
                            rows.append(dict(seed=seed, draw=draw, method=meth, rule=rule, query=qk, qdraw=qi,
                                             top1=R6m.top1(R6m.smat(zz, G, rule))))
        del variants
    df = pd.DataFrame(rows).groupby(['seed', 'draw', 'method', 'rule', 'query']).top1.mean().reset_index()
    df['target'], df['montage'] = B, mon
    df.to_csv(out, index=False)
    print(f'[cnnft] seed {seed} t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


def report(mon='posterior17'):
    pd.set_option('display.width', 250)
    f = sorted(glob.glob(os.path.join(odir('eval'), f'r12_t*_{mon}.csv')))
    if not f:
        print('no round 12 results yet')
        return
    d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
    bi = pd.DataFrame([json.load(open(x)) for x in sorted(glob.glob(os.path.join(odir('eval'), f'r12_t*_{mon}_bases.json')))])
    print(f'ROUND 12: direction-dependent rotation prior ({mon}, seed-{SEED} decoders, people 2-10; top-1 %)')
    if len(bi):
        t = bi[bi.target.isin(TEST)]
        print('\nREPRODUCIBLE DIRECTIONS (source people only; means over targets)')
        print(f'  split-half agreement of a source person\'s rotation (cosine of departures, 0 = none): {t.split_half_agreement.mean():.3f}')
        print(f'  share of positive reproducibility in the top {RANK} directions: {t.rep_top16_share.mean():.3f}')
        print(f'  overlap of U_rep with U_pca {t.overlap_rep_pca.mean():.3f}, with U_diff {t.overlap_rep_diff.mean():.3f}, '
              f'with random {t.overlap_rep_rand.mean():.3f} (random baseline {RANK / 512:.3f})')
        mc = [x for x in t.columns if x.startswith('target_motion_in_')]
        if mc:
            print('  share of the TARGET\'s own 500-image rotation inside each basis (privileged diagnostic): '
                  + ', '.join(f'{x.replace("target_motion_in_", "")} {t[x].mean():.3f}' for x in mc))
    per = d.groupby(['k', 'method', 'rule', 'query', 'target']).top1.mean()
    T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3]).mean()
    for rule in ('cos', 'csls_batch'):
        for q in ('s4R4', 's4R1', 'bench'):
            print(f'\n{q}, {rule} (rows = method, columns = calibration images)')
            print(T.xs((rule, q), level=('rule', 'query')).unstack('k').round(2).to_string())
    print('\nchosen settings (most common): ' + d.dropna(subset=['lam']).groupby(['method', 'k']).apply(
        lambda x: f'{x.lam.mode().iloc[0]:g}' + (f'/a{x.alpha.mode().iloc[0]:g}' if x.alpha.notna().any() else '')).unstack('k').to_string())
    g = lambda k, m, rule='cos', q='s4R4': per.xs((k, m, rule, q), level=('k', 'method', 'rule', 'query'))  # noqa: E731
    print('\nPRIMARY')
    p0 = R4m.pstats(g(100, 'dir_rep'), g(100, 'uniform'), 'dir_rep - uniform, k = 100, s4R4, cos (locked follow-up, existing cohort)', primary=True)
    ps = {'vs rand': R4m.pstats(g(100, 'dir_rep'), g(100, 'dir_rand'), 'dir_rep - dir_rand (same rank and strengths, random basis)'),
          'vs pca': R4m.pstats(g(100, 'dir_rep'), g(100, 'dir_pca'), 'dir_rep - dir_pca (reproducibility beyond variance?)'),
          'vs diff': R4m.pstats(g(100, 'dir_rep'), g(100, 'dir_diff'), 'dir_rep - dir_diff (source-only difference directions)')}
    print(f'  HOLM over the secondaries: {R10.holm(ps)}')
    print('\nREGULARIZATION VS LOCATION (k = 100)')
    R4m.pstats(g(100, 'uniform'), g(100, 'ordinary'), 'uniform shrinkage - ordinary rotation (does shrinking help at all?)')
    R4m.pstats(g(100, 'dir_rep'), g(100, 'ordinary'), 'dir_rep - ordinary rotation')
    R4m.pstats(g(100, 'dir_rep', 'csls_batch'), g(100, 'uniform', 'csls_batch'), 'primary with batch CSLS')
    print('\nSECONDARY: does the advantage shrink with more evidence? (k = 500)')
    R4m.pstats(g(500, 'dir_rep'), g(500, 'uniform'), 'dir_rep - uniform, k = 500')
    R4m.pstats(g(500, 'dir_rep'), g(500, 'dir_rand'), 'dir_rep - dir_rand, k = 500')
    print('\nEVERY BUDGET (cos, s4R4)')
    for k in KS:
        for b in ('rep', 'rand', 'pca', 'diff'):
            R4m.pstats(g(k, f'dir_{b}'), g(k, 'uniform'), f'dir_{b} - uniform, k = {k}')
        R4m.pstats(g(k, 'uniform'), g(k, 'zero'), f'uniform - zero, k = {k}')
    print('\nOTHER QUERIES (k = 100)')
    for q in ('s4R1', 's4R20', 'bench'):
        R4m.pstats(g(100, 'dir_rep', q=q), g(100, 'uniform', q=q), f'dir_rep - uniform, {q}')
    if 'scoring_path_max_diff' in bi:
        print(f'\nscoring-path check (alpha = 1 vs round-10 rotation): max |diff| over targets {bi.scoring_path_max_diff.max():.2e}')
    if p0 is not None and not np.isnan(p0):
        dm = (g(100, 'dir_rep') - g(100, 'uniform')).reindex(TEST).mean()
        dr = (g(100, 'dir_rep') - g(100, 'dir_rand')).reindex(TEST).mean()
        print(f'\nDECISION: primary {"PASSES" if p0 < 0.05 and dm > 0 else "does not pass"} (mean {dm:+.2f}, p = {p0:.4f}; '
              f'practical bar +1); vs random basis {dr:+.2f}. Reading: beats ordinary but ties uniform = the gain is '
              f'regularization; beats uniform AND random = where you regularize matters.')
    # ---- repeat-agreement weighting
    print('\n' + '=' * 100 + '\nREPEAT-AGREEMENT WEIGHTING (bottom agreement quartile weighted 0.5; same strength as unweighted)\n' + '=' * 100)
    gate = None
    if len(bi) and 'agree_err_spearman' in bi:
        t = bi[bi.target.isin(TEST)]
        gate = t.agree_err_spearman.mean()
        print(f'VALIDITY CHECK on source people (new to their pair decoder): Spearman(agreement, angular error to the population '
              f'anchor) = {gate:+.3f}; with response norm partialled out = {t.agree_err_partial_norm.mean():+.3f}. '
              f'Negative = more agreement predicts smaller error. ' + ('PASSES' if gate < -0.05 and t.agree_err_partial_norm.mean() < 0 else
                                                                        'FAILS: read the weighting rows below as exploratory'))
    for k in (100, 500):
        R4m.pstats(g(k, 'w_agree'), g(k, 'w_shuffle'), f'agreement weights - shuffled weights, k = {k} (prespecified at 100)')
        R4m.pstats(g(k, 'w_agree'), g(k, 'uniform'), f'agreement weights - equal weights, k = {k}')
    print('2 x 2 at k = 100 (weights x directional prior):')
    for a_, b_ in (('w_agree_dir', 'dir_rep'), ('w_agree_dir', 'w_agree'), ('w_agree_dir', 'uniform'), ('w_agree_dir', 'w_shuffle_dir')):
        R4m.pstats(g(100, a_), g(100, b_), f'{a_} - {b_}')
    # ---- CNN fine-tuning
    f = sorted(glob.glob(os.path.join(odir('cnnft'), f'cnnft_s*_t*_{mon}.csv')))
    if f:
        cf = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        print('\n' + '=' * 100 + f'\nCNN FINE-TUNING TOWARD POPULATION RESPONSES (k = 500; seeds {sorted(cf.seed.unique())}; averaged per person)\n' + '=' * 100)
        pc = cf.groupby(['method', 'rule', 'query', 'target']).top1.mean()
        T = pc[pc.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2]).mean()
        print(T.unstack('query')[QK].round(2).to_string())
        h = lambda m_, q='s4R4', rule='cos': pc.xs((m_, rule, q), level=('method', 'rule', 'query'))  # noqa: E731
        R4m.pstats(h('ft_pop+rot'), h('ft+rot'), 'anchor loss, identical downstream: ft_pop + rotation - ft + rotation, s4R4, cos (prespecified)')
        R4m.pstats(h('ft_pop|pop_gallery'), h('ft|pop_gallery'), 'anchor loss, both read with the population gallery, no rotation')
        R4m.pstats(h('ft_pop'), h('ft'), 'anchor loss, both read with their own image gallery')
        R4m.pstats(h('ft+rot'), h('ft'), 'rotation after ordinary ft - ordinary ft (best previous: 14.8%)')
        R4m.pstats(h('ft_pop+rot', 's4R1'), h('ft+rot', 's4R1'), 'anchor loss, identical downstream, single presentation (secondary)')
        R4m.pstats(h('ft_pop+rot', rule='csls_batch'), h('ft+rot', rule='csls_batch'), 'anchor loss, identical downstream, batch CSLS')
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['eval', 'cnnft', 'report'])
    p.add_argument('--seeds', type=int, nargs='+', default=[2025, 2026, 2027])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    p.add_argument('--montage', default='posterior17')
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'eval':
        return evaluate(a.targets, a.montage)
    if a.cmd == 'cnnft':
        for seed in a.seeds:
            for B in a.targets:
                cnnft_target(B, a.montage, seed)
        return
    report(a.montage)


if __name__ == '__main__':
    main()
