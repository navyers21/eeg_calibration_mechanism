"""Round 11: a population rotation prior, learned from how OTHER people differ from the population.

    python scripts/nb_round11.py train  --config C [--epochs 50]   # 45 pair decoders (each leaves out 2 people)
    python scripts/nb_round11.py eval   --config C                 # priors + calibration with them
    python scripts/nb_round11.py report --config C

Why. Round 9 broke the calibration correction into parts: the rotation does the work (+3.4 over a shift), the scale
hurts, and low-rank edits are no better than random. It also found that the rotations fitted for different people
share structure (top-8 direction overlap 0.17 across people vs 0.016 for random directions; shift vectors cosine 0.79).
Round 10: rotation-only calibration beats rigid most at small budgets, but its strength is chosen by cross-validation
and it shrinks hard toward the IDENTITY (lambda = 10 almost always), so with 25-100 images most of the possible
rotation is thrown away. If part of the rotation is shared across people, it can be learned from the source people and
used (a) with zero labels and (b) as the thing a few labeled images are shrunk toward, instead of the identity.

How the prior is learned without leaking the target. A source person j can only show how a NEW person differs if the
decoder has never seen j. The usual leave-one-out decoder for target B was trained on j, and the leave-one-out decoder
for j was trained on B, so neither can be used. So for every pair (B, j) we train a decoder on the other 8 people
(45 decoders, same recipe and seed as the target decoder: repaug, seed 2026). For target B and each source j:
  1. in decoder D_Bj, j is a new person: label-free layer-1 offset from j's trials, then j's outputs for 4,000 pool
     images (2-presentation averages, like a calibration session) and the 8-person population responses;
  2. R_j = rotation from j's outputs to the population responses (4,000 images), shrunk toward the identity with the
     strength chosen by 5-fold CV on j's own images, so low-signal dimensions are not rotated at random;
  3. the space of D_Bj is mapped onto the target decoder's space by a rotation Q_j fitted between the two decoders'
     population responses on the same images (no person-specific data; same shrinkage rule), and R_j is carried over:
     Q_j' R_j Q_j;
  4. R_pop = the rotation closest to the average of the 9 carried-over R_j.
Nothing from the target person enters R_pop.

Methods on the target (seed-2026 repaug decoder, population gallery, earlier-day calibration, round 6 draws):
  zero         layer-1 offset only (reference)
  pop0         ZERO LABELS: offset, then R_pop applied about the target's unlabeled output mean and the population mean
  pop0_half    same with half of R_pop's rotation (exploratory: in case the learned rotation overshoots)
  rot          rotation-only, shrunk toward the identity, strength by 5-fold CV over the calibration images (round 10)
  rot_prior    rotation-only, shrunk toward R_pop instead, same CV grid (plus a stronger setting)
  rot_rand     CONTROL: shrunk toward a rotation with R_pop's angles but random axes (P R_pop P')
  rigid        rounds 6-7 reference (lambda 1, with scale)
k = 25 / 50 / 100 / 200 / 500 images x 2 presentations; queries s4R1 / s4R4 / s4R20 / benchmark; cos and batch CSLS.
Diagnostics: how far R_pop is from the identity; how much the 9 source rotations agree; and whether the target's own
rotation (fitted on its 500-image calibration) points the same way as R_pop more than chance.

PRIMARY (fixed before running): rot_prior - rot, k = 100, s4R4, cos, people 2-10, Wilcoxon.
Secondary (Holm over four): pop0 - zero (zero labels); rot_prior - rot at k = 25; rot_prior - rot_rand at k = 100
(do the learned axes matter, or just the amount of rotation); the primary with batch CSLS.
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
from eegalign import nb as N  # noqa: E402
import nb_round3 as R  # noqa: E402
import nb_round4 as R4m  # noqa: E402
import nb_round5 as R5m  # noqa: E402
import nb_round6 as R6m  # noqa: E402
import nb_gallery as GL  # noqa: E402
import nb_round9 as R9  # noqa: E402
import nb_round10 as R10  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
QK = ['s4R1', 's4R4', 's4R20', 'bench']
SEED = 2026
KS = [25, 50, 100, 200, 500]
CS = 3
N_IMG = 4000
REL = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def dev():
    return N.ENV['device']


def odir(sub=''):
    d = os.path.join(N.ENV['results_dir'], 'nb_round11', sub)
    os.makedirs(d, exist_ok=True)
    return d


def pair_path(a, b, mon, seed=SEED):
    a, b = min(a, b), max(a, b)
    return N.ckpt_path(f'pairx{b:02d}', mon, a, seed, 'last')


# ---------------------------------------------------------------------------------------------------------------
# pair decoders (repaug recipe, 8 training people)
# ---------------------------------------------------------------------------------------------------------------
def train_on(sources, mon, seed, path, epochs=50, batch=1024, lr=1e-4, wd=1e-4):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xs, IM = [], []
    for s in sources:
        d = N.subject_data(s, mon, keys=('train_labels',))
        pool, _ = N.split(d['train_labels'])
        Xs.append(torch.from_numpy(np.asarray(R.load_reps(s, mon)[pool], np.float16)))
        IM.append(torch.from_numpy(pool))
    X = torch.cat(Xs).to(dev())
    IMG = torch.cat(IM).to(dev())
    del Xs
    n, _, C, T = X.shape
    ftr = N.features('train', 'aug').to(dev())
    m = N.NBModel(C).to(dev())
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    kv = torch.tensor([1., 2., 4.], device=dev())
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n, device=dev())
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = X[idx].float()
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
    m.eval()
    N.save_model(m, path, dict(kind='pair', sources=list(sources), seed=seed, epochs=epochs, montage=mon), [], epochs)
    del X, IMG
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (time.time() - t0) / 60


def train(mon='posterior17', epochs=50):
    pairs = [(a, b) for a in PEOPLE for b in PEOPLE if a < b]
    todo = [p for p in pairs if not os.path.exists(pair_path(*p, mon))]
    print(f'{len(pairs) - len(todo)}/{len(pairs)} pair decoders exist; training {len(todo)}', flush=True)
    for i, (a, b) in enumerate(todo):
        mins = train_on([p for p in PEOPLE if p not in (a, b)], mon, SEED, pair_path(a, b, mon), epochs)
        print(f'  [{i + 1}/{len(todo)}] pair ({a}, {b}): {mins:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------------------------------------------
def procrustes(Xs, Ys):
    """Orthogonal R minimizing ||(X - xm) R - (Y - ym)|| (rows are samples)."""
    Xc, Yc = Xs - Xs.mean(0), Ys - Ys.mean(0)
    U, _, Vh = R.safe_svd(Xc.T @ Yc)
    return (U @ Vh).float()


def polar(M):
    U, _, Vh = R.safe_svd(M)
    return (U @ Vh).float()


def e_rot_prior(Xs, Ys, lam, P):
    """Rotation shrunk toward P: argmax tr(R'(Xc'Yc + lam s P)) over orthogonal R; applied about the means."""
    xm, ym = Xs.mean(0), Ys.mean(0)
    Xc, Yc = Xs - xm, Ys - ym
    s = float(torch.trace(Xc.T @ Xc)) / Xc.shape[1] + 1e-8
    U, _, Vh = R.safe_svd(Xc.T @ Yc + lam * s * P)
    Rm = (U @ Vh).float()
    fn = lambda Z: (Z - xm) @ Rm + ym  # noqa: E731
    fn.R = Rm
    return fn


def dev_cos(A, B):
    """Cosine between two rotations' departures from the identity."""
    I_ = torch.eye(len(A), device=A.device)
    a, b = (A - I_).flatten(), (B - I_).flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def build_prior(B, mon, ctx, cache):
    """R_pop for target B from the 9 source people, each as a new person in its own pair decoder."""
    c = ctx.c
    pool = c.pool
    ims = np.sort(np.random.RandomState(11000).choice(pool, N_IMG, replace=False))
    sub300 = np.random.RandomState(3).choice(pool, 300, replace=False)
    PB = c.pool_z[torch.as_tensor([ctx.pool_pos[int(i)] for i in ims], device=dev())]
    Rts, info = [], []
    for j in PEOPLE:
        if j == B:
            continue
        p = pair_path(B, j, mon)
        if not os.path.exists(p):
            print(f'   missing pair decoder ({B}, {j}): run train first')
            return None, None
        D = N.load_model(p)[0].to(dev()).eval().requires_grad_(False)
        train_people = [q for q in PEOPLE if q not in (B, j)]
        Zp, H = [], []
        for q in train_people:
            if q not in cache:
                tr = N.subject_data(q, mon, keys=('train',))['train']
                cache[q] = (np.asarray(tr[ims], np.float32), np.asarray(tr[sub300], np.float32))
                del tr
            Zp.append(R10.z_of(D, cache[q][0]))
            H.append(R10.h1_mean(D, cache[q][1]))
        PD, mSD = torch.stack(Zp).mean(0), torch.stack(H).mean(0)
        rj = np.asarray(R.load_reps(j, mon)[ims][:, :2], np.float32)               # (N_IMG, 2, C, T)
        d1 = mSD - R10.h1_mean(D, rj.reshape(-1, *rj.shape[2:]))
        Xj = R10.z_of(D, rj.mean(1), d1)
        # rotations shrunk toward the identity, strength by 5-fold CV on that source's own images: dimensions with
        # little signal stay put instead of rotating at random (an unshrunk 512-d Procrustes fits noise there)
        Id = torch.eye(Xj.shape[1], device=dev())
        lam_j, _ = R9.cv_pick(lambda a, b, l: e_rot_prior(a, b, l, Id), Xj, PD, REL, seed=j)
        Rj = e_rot_prior(Xj, PD, lam_j, Id).R
        lam_q, _ = R9.cv_pick(lambda a, b, l: e_rot_prior(a, b, l, Id), PD, PB, REL, seed=j)
        Q = e_rot_prior(PD, PB, lam_q, Id).R                                          # D_Bj space -> target space
        fitQ = float(F.cosine_similarity((PD - PD.mean(0)) @ Q, PB - PB.mean(0), dim=1).mean())
        Rt = Q.T @ Rj @ Q
        Rts.append(Rt)
        info.append(dict(source=j, lam_source=lam_j, lam_space=lam_q, space_fit_cos=fitQ, dist_from_I=float((Rj - torch.eye(len(Rj), device=dev())).norm()) / len(Rj) ** 0.5))
        del D, rj
    Rpop = polar(torch.stack(Rts).mean(0))
    agree = [dev_cos(Rts[a], Rts[b]) for a in range(len(Rts)) for b in range(a + 1, len(Rts))]
    I_ = torch.eye(len(Rpop), device=dev())
    summary = dict(target=B, rpop_dist_from_I=float((Rpop - I_).norm()) / len(Rpop) ** 0.5,
                   source_dist_from_I=float(np.mean([x['dist_from_I'] for x in info])),
                   source_agreement=float(np.mean(agree)), space_fit_cos=float(np.mean([x['space_fit_cos'] for x in info])),
                   lam_source=float(np.median([x['lam_source'] for x in info])), lam_space=float(np.median([x['lam_space'] for x in info])))
    return Rpop, summary


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------
def eval_target(B, mon, cache):
    out = os.path.join(odir('eval'), f'r11_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[eval] t{B}: done')
        return
    R5m.SEED = SEED
    if not os.path.exists(N.ckpt_path('repaug', mon, B, SEED, 'last')):
        print(f'[eval] t{B}: no seed-{SEED} repaug decoder (round 7), skipping')
        return
    t0 = time.time()
    ctx = R6m.Ctx('repaug', B, mon)
    c = ctx.c
    Rpop, summ = build_prior(B, mon, ctx, cache)
    if Rpop is None:
        return
    g = torch.Generator(device='cpu').manual_seed(500 + B)
    Pr, _ = torch.linalg.qr(torch.randn(len(Rpop), len(Rpop), generator=g))
    Pr = Pr.to(dev())
    Rrand = Pr @ Rpop @ Pr.T                                                          # same angles, random axes
    ybar = c.pool_z.mean(0)
    from scipy.linalg import expm, logm
    Rhalf = torch.as_tensor(np.real(expm(0.5 * logm(Rpop.double().cpu().numpy()))), dtype=torch.float32, device=dev())
    tr = c.ses['train']
    reps = R.load_reps(B, mon)
    cand = np.array([i for i in c.pool if (tr[i] == CS).sum() == 2])
    rows = []
    for k in KS:
        for draw in range(R6m.ndraw(k)):
            ims = GL.draw_images(cand, k, draw)
            rr = np.stack([np.nonzero(tr[i] == CS)[0] for i in ims])
            x2 = np.asarray(reps[ims[:, None], rr], np.float32)
            singles = x2.reshape(-1, *x2.shape[2:])
            d1 = c.mS['h1'] - R10.h1_mean(c.f0, singles)
            Xs = R10.z_of(c.f0, x2.mean(1), d1)
            Ys = c.pool_z[torch.as_tensor([ctx.pool_pos[int(i)] for i in ims], device=dev())]
            Zq = c.zq(d1)
            xbar_u = R10.z_of(c.f0, singles, d1).mean(0)                                # unlabeled mean
            pop0 = lambda Z: (Z - xbar_u) @ Rpop + ybar  # noqa: E731
            pop0h = lambda Z: (Z - xbar_u) @ Rhalf + ybar  # noqa: E731
            lam_i, _ = R9.cv_pick(lambda a, b, l: e_rot_prior(a, b, l, torch.eye(len(Rpop), device=dev())), Xs, Ys, REL, seed=draw)
            lam_p, _ = R9.cv_pick(lambda a, b, l: e_rot_prior(a, b, l, Rpop), Xs, Ys, REL, seed=draw)
            lam_r, _ = R9.cv_pick(lambda a, b, l: e_rot_prior(a, b, l, Rrand), Xs, Ys, REL, seed=draw)
            f_rot = e_rot_prior(Xs, Ys, lam_i, torch.eye(len(Rpop), device=dev()))
            f_pri = e_rot_prior(Xs, Ys, lam_p, Rpop)
            cells = [('zero', None), ('pop0', pop0), ('pop0_half', pop0h), ('rot', f_rot), ('rot_prior', f_pri),
                     ('rot_rand', e_rot_prior(Xs, Ys, lam_r, Rrand)), ('rigid', R9.e_procrustes(Xs, Ys, 1.0))]
            base = dict(k=k, draw=draw, lam_rot=lam_i, lam_prior=lam_p, lam_rand=lam_r)
            for meth, mp in cells:
                for qk in QK:
                    for qi, z0 in enumerate(Zq[qk]):
                        z = z0 if mp is None else mp(z0)
                        for rule in ('cos', 'csls_batch'):
                            rows.append(dict(base, method=meth, rule=rule, query=qk, qdraw=qi,
                                             top1=R6m.top1(R6m.smat(z, ctx.Ganc, rule))))
            if k == 500 and draw == 0:                                                # does the target rotate like R_pop?
                Rt = e_rot_prior(Xs, Ys, 0.1, torch.eye(len(Rpop), device=dev())).R
                summ.update(target_vs_prior=dev_cos(Rt, Rpop), target_vs_random_axes=dev_cos(Rt, Rrand),
                            prior_rot_vs_target=dev_cos(f_pri.R, Rt))
        print(f'   k={k} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows).groupby(['k', 'draw', 'method', 'rule', 'query']).agg(
        top1=('top1', 'mean'), lam_rot=('lam_rot', 'first'), lam_prior=('lam_prior', 'first'), lam_rand=('lam_rand', 'first')).reset_index()
    df['target'], df['montage'] = B, mon
    df.to_csv(out, index=False)
    with open(out.replace('.csv', '_prior.json'), 'w') as f:
        json.dump(summ, f)
    print(f'[eval] t{B}: {(time.time() - t0) / 60:.1f} min; prior {summ}', flush=True)


def evaluate(targets, mon='posterior17'):
    cache = {}
    for B in targets:
        print(f'[eval] target {B}', flush=True)
        eval_target(B, mon, cache)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def report(mon='posterior17'):
    pd.set_option('display.width', 250)
    f = sorted(glob.glob(os.path.join(odir('eval'), f'r11_t*_{mon}.csv')))
    if not f:
        print('no round 11 results yet')
        return
    d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
    pj = [json.load(open(x)) for x in sorted(glob.glob(os.path.join(odir('eval'), f'r11_t*_{mon}_prior.json')))]
    pr = pd.DataFrame(pj)
    print(f'ROUND 11: population rotation prior ({mon}, seed-{SEED} decoders, people 2-10; top-1 %)')
    if len(pr):
        t = pr[pr.target.isin(TEST)]
        print('\nTHE PRIOR (means over targets)')
        print(f'  R_pop distance from identity {t.rpop_dist_from_I.mean():.3f}; single sources {t.source_dist_from_I.mean():.3f}')
        print(f'  agreement between the 9 source rotations (cosine of departures from identity): {t.source_agreement.mean():.3f} (0 = unrelated)')
        print(f'  pair-decoder -> target-decoder space fit (cosine): {t.space_fit_cos.mean():.3f}')
        if 'lam_source' in t:
            print(f'  chosen shrinkage (median): source rotations {t.lam_source.median()}, space map {t.lam_space.median()}')
        if 'target_vs_prior' in t:
            print(f'  target\'s own 500-image rotation vs R_pop: {t.target_vs_prior.mean():.3f}; vs same angles on random axes: '
                  f'{t.target_vs_random_axes.mean():.3f}  (per target: ' + ', '.join(f't{int(r.target)} {r.target_vs_prior:.2f}' for r in t.itertuples()) + ')')
    per = d.groupby(['k', 'method', 'rule', 'query', 'target']).top1.mean()
    T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3]).mean()
    for rule in ('cos', 'csls_batch'):
        for q in ('s4R4', 's4R1', 'bench'):
            print(f'\n{q}, {rule} (rows = method, columns = calibration images)')
            print(T.xs((rule, q), level=('rule', 'query')).unstack('k').round(2).to_string())
    lam = d.groupby('k')[['lam_rot', 'lam_prior', 'lam_rand']].agg(lambda x: x.mode().iloc[0])
    print('\nchosen strength (most common): \n' + lam.to_string())
    g = lambda k, m, rule='cos', q='s4R4': per.xs((k, m, rule, q), level=('k', 'method', 'rule', 'query'))  # noqa: E731
    print('\nPRIMARY')
    p0 = R4m.pstats(g(100, 'rot_prior'), g(100, 'rot'), 'rot_prior - rot, k = 100, s4R4, cos', primary=True)
    ps = {'pop0 - zero': R4m.pstats(g(500, 'pop0'), g(500, 'zero'), 'ZERO LABELS: pop0 - zero (unlabeled trials of the k = 500 draws)'),
          'k=25': R4m.pstats(g(25, 'rot_prior'), g(25, 'rot'), 'rot_prior - rot, k = 25'),
          'axes': R4m.pstats(g(100, 'rot_prior'), g(100, 'rot_rand'), 'rot_prior - rot_rand (learned vs random axes), k = 100'),
          'csls': R4m.pstats(g(100, 'rot_prior', 'csls_batch'), g(100, 'rot', 'csls_batch'), 'primary with batch CSLS')}
    print(f'  HOLM over the secondaries: {R10.holm(ps)}')
    print('\nEVERY BUDGET (cos, s4R4)')
    for k in KS:
        R4m.pstats(g(k, 'rot_prior'), g(k, 'rot'), f'rot_prior - rot, k = {k}')
        R4m.pstats(g(k, 'rot_prior'), g(k, 'zero'), f'rot_prior - zero, k = {k}')
        R4m.pstats(g(k, 'rot_prior'), g(k, 'rot_rand'), f'rot_prior - rot_rand, k = {k}')
    R4m.pstats(g(500, 'pop0_half'), g(500, 'zero'), 'exploratory: pop0_half - zero, s4R4')
    print('\nZERO LABELS, OTHER QUERIES')
    for q in ('s4R1', 's4R20', 'bench'):
        R4m.pstats(g(500, 'pop0', q=q), g(500, 'zero', q=q), f'pop0 - zero, {q}')
    for q in ('s4R1', 'bench'):
        R4m.pstats(g(100, 'rot_prior', q=q), g(100, 'rot', q=q), f'rot_prior - rot, k = 100, {q}')
    if p0 is not None and not np.isnan(p0):
        dm = (g(100, 'rot_prior') - g(100, 'rot')).reindex(TEST).mean()
        print(f'\nDECISION: primary {"PASSES" if p0 < 0.05 and dm > 0 else "does not pass"} (mean {dm:+.2f}, Wilcoxon p = {p0:.4f})')
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'eval', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    p.add_argument('--montage', default='posterior17')
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'train':
        return train(a.montage, a.epochs)
    if a.cmd == 'eval':
        return evaluate(a.targets, a.montage)
    report(a.montage)


if __name__ == '__main__':
    main()
