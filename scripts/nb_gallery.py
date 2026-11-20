"""Round 8: strengthen the two results we have. (1) Does predicting population EEG add anything beyond a better image
metric, and can a better template predictor close the gap to measured templates? (2) What does the cross-subject decoder
buy over a decoder trained from scratch on the same calibration data? (3) Should a new user see more images or the same
images more often?

    python scripts/nb_gallery.py audit   --config C                     # candidate concepts vs every fit (fast)
    python scripts/nb_gallery.py gallery --config C [--seeds 2026 2025] # A: gallery test, zero labels + rigid k = 500
    python scripts/nb_gallery.py scratch --config C                     # B: within-person decoder from scratch on k images
    python scripts/nb_gallery.py schedule --config C                    # C: 1000 x 1 vs 500 x 2 vs 250 x 4 presentations
    python scripts/nb_gallery.py report  --config C

A. GALLERY TEST. Same frozen repaug decoder, same calibration draws (round 6, earlier day, k = 500, 3 draws), same
label-free layer-1 offset (estimated from the calibration trials, labels unused), same queries; only the gallery the
query is compared against changes. Every gallery is built from source people and training concepts only; the 200 test
concepts are disjoint from the 1,654 training concepts (checked by `audit`, which also checks the ridge / GCV and
predictor-selection sets).
  img          the decoder's own CLIP gallery (reference)
  img_white    whitened CLIP gallery (image-only)
  img_spec     CLIP eigenvectors with the anchor predictor's eigenvalues (image-only, spectrum-matched)
  img_rank     CLIP gallery projected on its top-r PCs, r = effective rank (participation ratio) of the predictor
  anchor       predicted population responses (ridge CLIP -> source-mean decoder outputs, round 3, LOCKED)
  anchor_permconcept   the same ridge fitted on image/response pairs shuffled WITHIN concept (same geometry learner,
               image-level correspondence broken)
  oracle       measured source-population responses to the test images (PRIVILEGED: needs every candidate seen by
               earlier people; a headroom estimate for prediction, not a deployable gallery; measured templates are
               noisy too, so not a strict ceiling)
  anchor_ret   template predictor trained for retrieval: linear CLIP -> template map initialized at the ridge solution,
               trained with InfoNCE where the queries are SOURCE people's 1-, 2- and 4-presentation decoder outputs for
               the 2,000 held-concept images (images the decoder never trained on; each source person acts as a
               simulated new user), positives = that image's template, negatives = the other images in the batch,
               plus a penalty toward the ridge predictions. Penalty (0.1 / 1 / 10) and stopping step are chosen on 2
               held-out source people x 50 held-out concepts (trained on the other 7 people x 150 concepts); if no
               setting beats ridge there, ridge is kept. Nothing from the target person is used. Round 4's
               contrastive head differed: its queries were clean 4-presentation outputs on images the decoder had
               trained on, selection used the same people, and it stopped after ~20 steps (no gain over ridge).
Methods: zero labels (offset only), rigid alignment at k = 500 to each gallery's own representation of the
calibration images (oracle and anchor_meas: to the measured population responses, as in rounds 6-7).
Rules: cos = each query scored on its own (PRIMARY); csls_batch = round 4-7 batch rule (secondary).
Decoders: repaug seed 2026 (round 7's fresh decoder, PRIMARY) and seed 2025 (replication).
  PRIMARY: seed 2026, zero labels, s4R4, cos, people 2-10, Wilcoxon: anchor - img_spec.
  Secondary (Holm over the four): anchor - img_rank; anchor - best image-only gallery per person (conservative:
  a per-person max over four noisy cells is biased against anchor); anchor - anchor_permconcept; oracle - anchor.
  Then the same with csls_batch, at rigid k = 500, at s4R1 / s4R20 / bench, and on seed 2025.
  Decision rule (printed): anchor beats img_spec with cos -> lead with the gallery; only with csls_batch -> batch
  retrieval only; neither -> the paper is the calibration study.
  Template predictor (prespecified): anchor_ret - anchor, rigid k = 500, s4R4, cos, seed 2026; practical threshold
  +2 points (4 more correct per 200), set before running. Also at zero labels and with csls_batch.

B. FROM-SCRATCH CONTROL. A NeuroBridge decoder trained only on the person's k calibration images (2 presentations
each, the same images and draws as rounds 6-7; 1-or-2-presentation averaging and smoothing as in repaug), random init,
snapshots at 25 / 50 / 100 / 200 / 400 epochs; the epoch is chosen on person 1 (s4R4, cos) and applied to people 2-10.
Compared, per person, with the calibrated cross-subject decoder on the same images: rigid (round 6) and ft+rigid
(round 7), and with the within-person decoder trained on all of the person's data (round 6 ceiling).

C. CALIBRATION SCHEDULE. Same budget of 1,000 calibration presentations, seed-2026 decoder, rigid to measured
population responses (rounds 6-7 method), 3 draws each:
  1000x1   1,000 images, one session-3 presentation each
  500x2    500 images x 2 session-3 presentations (round 6-7 draws)
  250x4    250 images x 4 presentations: their 2 session-3 presentations plus 2 from sessions 1-2 (THINGS-EEG2 never
           shows an image 4 times in one session, so this schedule spans days; read it with that caveat)
Queries: session 4 at 1 / 4 / 20 presentations and the benchmark. Answers: does a new user gain more from another
image or from seeing the same image again, and is the weak single-presentation result a calibration or a query problem?
"""
import argparse
import glob
import os
import re
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
import nb_next as X  # noqa: E402
import nb_round3 as R  # noqa: E402
import nb_round4 as R4m  # noqa: E402
import nb_round5 as R5m  # noqa: E402
import nb_round6 as R6m  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
QK = ['s4R1', 's4R4', 's4R20', 'bench']
K = 500
CS = 3                                   # earlier-day calibration session
SEEDS = (2026, 2025)
IMG_ONLY = ['img', 'img_white', 'img_spec', 'img_rank']
GALS = IMG_ONLY + ['anchor', 'anchor_permconcept', 'anchor_ret', 'oracle']
K_SCRATCH = [500, 1000, 2000, 4000]
SCHEDULES = {'1000x1': (1000, 1), '500x2': (500, 2), '250x4': (250, 4)}
RET_LAMS = (0.1, 1.0, 10.0)
RET_STEPS = 2000
EPOCHS = [25, 50, 100, 200, 400]


def dev():
    return N.ENV['device']


def odir(sub):
    d = os.path.join(N.ENV['results_dir'], 'nb_gallery', sub)
    os.makedirs(d, exist_ok=True)
    return d


def nrm(x):
    return F.normalize(x.float(), dim=-1)


def draw_images(cand, k, draw, cs=CS):
    """Round 6's calibration draws (identical seeds)."""
    return np.sort(np.random.RandomState(7000 + 31 * k + draw + 1000 * cs).choice(cand, k, replace=False))


def concept(path):
    """THINGS paths: .../<00001_aardvark>/<aardvark_01b.jpg>; falls back to the file name if there is no folder."""
    p = str(path).replace('\\', '/')
    d = os.path.basename(os.path.dirname(p))
    if not d:
        d = os.path.splitext(os.path.basename(p))[0].rsplit('_', 1)[0]
    return re.sub(r'^\d+_', '', d).lower()


# ---------------------------------------------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------------------------------------------
def audit(B=1, mon='posterior17'):
    d = N.subject_data(B, mon, keys=('train_labels', 'train_paths', 'test_paths'))
    lab = d['train_labels']
    pool, draws = N.split(lab)
    held = np.concatenate(draws)
    tr_c = np.array([concept(p) for p in d['train_paths']])
    te_c = set(concept(p) for p in d['test_paths'])
    ok = True
    pool_c, held_c = set(tr_c[pool]), set(tr_c[held])
    print(f'AUDIT (person {B}): {len(set(tr_c))} training concepts, {len(te_c)} test concepts; '
          f'example paths: {d["train_paths"][0]} | {d["test_paths"][0]}')
    if len(te_c) < 150 or len(set(tr_c)) < 1000:
        print('  WARNING: could not parse concept names from the paths; check the examples above')
        ok = False
    for nm, s in (('ridge fit (pool)', pool_c), ('predictor selection / beta (held)', held_c), ('all training', set(tr_c))):
        n = len(te_c & s)
        ok &= n == 0
        print(f'  test concepts in {nm}: {n}' + ('' if n == 0 else f'  <-- e.g. {sorted(te_c & s)[:5]}'))
    print(f'  pool and held concepts disjoint: {len(pool_c & held_c) == 0}')
    print(f'  ridge GCV runs on pool images only (fit_ridge(f[pool], pool_z)); calibration images are pool images')
    print('  AUDIT ' + ('PASSED' if ok else 'FAILED: stop and fix before reading any gallery result'))
    return ok


# ---------------------------------------------------------------------------------------------------------------
# A. gallery test
# ---------------------------------------------------------------------------------------------------------------
def fit_ret(c, mon, seed=0, verbose=True):
    """Retrieval-trained template predictor (see docstring). Uses source people and held concepts only."""
    t0 = time.time()
    f, held = c.ftr, c.held
    rng = np.random.RandomState(8000 + seed)
    # source queries: decoder outputs for held images at 1, 2 and 4 presentations (no offset: sources are in-distribution)
    combos = [(0,), (1,), (2,), (3,), (0, 1), (2, 3), (0, 1, 2, 3)]
    ksz = torch.tensor([len(x) for x in combos], device=dev())
    Qs = []
    for s_ in c.src:
        rp = np.asarray(R.load_reps(s_, mon)[held], np.float32)          # (2000, 4, C, T)
        Qs.append(torch.stack([X.acts2(c.f0, rp[:, list(cb)].mean(1))['z'] for cb in combos], 1))
        del rp
    Qs = nrm(torch.stack(Qs))                                             # (9, 2000, 7, D)
    lab = c.labels[held]
    con = np.unique(lab)
    vcon = rng.choice(con, 50, replace=False)
    vi = np.nonzero(np.isin(lab, vcon))[0]
    ti = np.nonzero(~np.isin(lab, vcon))[0]
    vs = rng.choice(len(c.src), 2, replace=False)
    ts = np.setdiff1d(np.arange(len(c.src)), vs)
    Bm, xm, ym = X.ridge_svd(f[c.pool], c.pool_z)
    ridge = lambda fe: (fe - xm) @ Bm + ym  # noqa: E731
    fh = f[torch.as_tensor(held, device=dev())]
    poolt = torch.as_tensor(c.pool, device=dev())

    @torch.no_grad()
    def val(fn):
        G = nrm(fn(fh[torch.as_tensor(vi, device=dev())]))
        t = torch.arange(len(vi), device=dev())
        acc = [(Qs[s_][torch.as_tensor(vi, device=dev()), j] @ G.T).argmax(1).eq(t).float().mean().item()
               for s_ in vs for j in (0, 4, 6)]                            # 1, 2 and 4 presentations
        return 100 * float(np.mean(acc))
    v_ridge = val(ridge)
    best = (v_ridge, None, 'ridge', 0)
    tit = torch.as_tensor(ti, device=dev())
    for lam in RET_LAMS:
        torch.manual_seed(seed)
        lin = torch.nn.Linear(f.shape[1], Bm.shape[1]).to(dev())
        with torch.no_grad():
            lin.weight.copy_(Bm.T)
            lin.bias.copy_(ym - xm @ Bm)
        opt = torch.optim.AdamW(lin.parameters(), 1e-4, weight_decay=1e-4)
        for st in range(1, RET_STEPS + 1):
            ii = tit[torch.randperm(len(ti), device=dev())[:256]]
            ss = torch.as_tensor(ts, device=dev())[torch.randint(0, len(ts), (len(ii),), device=dev())]
            kk = torch.randint(0, 3, (len(ii),), device=dev())            # 1 / 2 / 4 presentations, equally often
            jj = torch.where(kk == 0, torch.randint(0, 4, (len(ii),), device=dev()), torch.where(kk == 1, 4 + torch.randint(0, 2, (len(ii),), device=dev()), torch.full_like(kk, 6)))
            q = Qs[ss, ii, jj]
            g = nrm(lin(fh[ii]))
            pb = poolt[torch.randint(0, len(c.pool), (512,), device=dev())]
            with torch.no_grad():
                r0 = ridge(f[pb])
            loss = F.cross_entropy(q @ g.T / 0.05, torch.arange(len(ii), device=dev())) + \
                lam * (1 - F.cosine_similarity(lin(f[pb]), r0, dim=1).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if st % 100 == 0:
                v = val(lambda fe: lin(fe))
                if v > best[0]:
                    best = (v, {k: x.detach().clone() for k, x in lin.state_dict().items()}, lam, st)
    info = dict(ret_val_ridge=v_ridge, ret_val_best=best[0], ret_lam=best[2], ret_step=best[3])
    if verbose:
        print(f'   anchor_ret ({time.time() - t0:.0f}s): source-only val top-1 ridge {v_ridge:.2f} -> {best[0]:.2f} '
              f'(lambda {best[2]}, step {best[3]}; chance {100 / len(vi):.2f})', flush=True)
    if best[1] is None:
        return (lambda fe: nrm(ridge(fe))), info
    lin = torch.nn.Linear(f.shape[1], Bm.shape[1]).to(dev())
    lin.load_state_dict(best[1])
    lin.requires_grad_(False)
    return (lambda fe: nrm(lin(fe))), info


def build_galleries(c, mon):
    """name -> (test gallery (200, D), fn(CLIP features) -> gallery vectors, or None)."""
    G = {g: (c.gal[g][0], c.galfn[g]) for g in ('img', 'img_white', 'img_spec', 'anchor', 'anchor_permconcept')}
    with torch.no_grad():
        Pp = c.galfn['anchor'](c.ftr[c.pool])
        lp = torch.linalg.eigvalsh(torch.cov(Pp.T)).clamp_min(0)
        r = int(round(float(lp.sum() ** 2 / (lp ** 2).sum())))
        Gp = c.f0.gallery(c.ftr[c.pool])
        mu = Gp.mean(0)
        _, V = torch.linalg.eigh(torch.cov(Gp.T))
        Vr = V[:, -r:]
        fn = lambda fe: nrm((c.f0.gallery(fe) - mu) @ Vr @ Vr.T)  # noqa: E731
        G['img_rank'] = (fn(c.fte), fn)
    G['oracle'] = (c.gal['oracle'][0], None)
    fn_ret, info = fit_ret(c, mon)
    with torch.no_grad():
        G['anchor_ret'] = (fn_ret(c.fte), fn_ret)
    info['eff_rank'] = r
    return G, info


def gallery_target(B, seed, mon='posterior17'):
    out = os.path.join(odir('gallery'), f'gal_s{seed}_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[gallery] seed {seed} t{B}: done')
        return
    R5m.SEED = seed
    if not os.path.exists(N.ckpt_path('repaug', mon, B, seed, 'last')):
        print(f'[gallery] seed {seed} t{B}: no repaug checkpoint (run round 5 / round 7 train_seed first), skipping')
        return
    t0 = time.time()
    ctx = R6m.Ctx('repaug', B, mon)
    c = ctx.c
    tr = c.ses['train']
    reps = R.load_reps(B, mon)
    G, info = build_galleries(c, mon)
    cand = np.array([i for i in c.pool if (tr[i] == CS).sum() == 2])
    rows = []
    for draw in range(R6m.ndraw(K)):
        ims = draw_images(cand, K, draw)
        rr = np.stack([np.nonzero(tr[i] == CS)[0] for i in ims])
        x2 = np.asarray(reps[ims[:, None], rr], np.float32)
        xa = x2.mean(1)
        pos = torch.as_tensor(np.array([ctx.pool_pos[int(i)] for i in ims]), device=dev())
        Ymeas = c.pool_z[pos]
        d1 = c.mS['h1'] - I.acts(c.f0, x2.reshape(-1, *x2.shape[2:]))['h1'].mean(0)
        Xs = X.acts2(c.f0, xa, d1)['z']
        Zq0 = c.zq(d1)
        fims = c.ftr[torch.as_tensor(ims, device=dev())]
        cells = []                                       # (gallery name, method, map, test gallery)
        for g, (Gt, fn) in G.items():
            cells.append((g, 'zero', None, Gt))
            with torch.no_grad():
                tgt = Ymeas if fn is None else fn(fims)
            cells.append((g, 'rigid', R.m_rigid(Xs, tgt, 1.0), Gt))
        cells.append(('anchor_meas', 'rigid', R.m_rigid(Xs, Ymeas, 1.0), G['anchor'][0]))   # rounds 6-7 method
        for qk in QK:
            for qi, z0 in enumerate(Zq0[qk]):
                for g, meth, mp, Gt in cells:
                    z = z0 if mp is None else mp(z0)
                    for rule in ('cos', 'csls_batch'):
                        rows.append(dict(draw=draw, gallery=g, method=meth, rule=rule, query=qk, qdraw=qi,
                                         top1=R6m.top1(R6m.smat(z, Gt, rule))))
        print(f'   draw {draw} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows).groupby(['draw', 'gallery', 'method', 'rule', 'query']).top1.mean().reset_index()
    df['target'], df['montage'], df['seed'] = B, mon, seed
    for k_, v_ in info.items():
        df[k_] = v_
    df.to_csv(out, index=False)
    print(f'  [gallery] seed {seed} t{B}: {(time.time() - t0) / 60:.1f} min (predictor effective rank {info["eff_rank"]})', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# B. from-scratch control
# ---------------------------------------------------------------------------------------------------------------
def train_scratch(x2, img, ftr, C, seed=0, lr=1e-4, wd=1e-4, eval_fn=None):
    torch.manual_seed(seed)
    m = N.NBModel(C).to(dev())
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    X2 = torch.as_tensor(x2, device=dev())
    IM = torch.as_tensor(img, device=dev())
    k, T = len(X2), X2.shape[-1]
    bs = min(256, k)
    res = {}
    for ep in range(1, max(EPOCHS) + 1):
        m.train()
        perm = torch.randperm(k, device=dev())
        for i in range(0, k, bs):
            idx = perm[i:i + bs]
            both = torch.rand(len(idx), device=dev()) < 0.5
            pick = torch.randint(0, 2, (len(idx),), device=dev())
            xb = torch.where(both[:, None, None], X2[idx].mean(1), X2[idx, pick])
            sm = F.avg_pool1d(xb.reshape(-1, 1, T), 5, 1, 2, count_include_pad=False).reshape(xb.shape)
            xb = torch.where(torch.rand(len(idx), xb.shape[1], 1, device=dev()) < 0.3, sm, xb)
            loss = N.nb_loss(m(xb), m.gallery(ftr[IM[idx]]), m.scale())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep in EPOCHS:
            m.eval()
            res[ep] = eval_fn(m)
    return res


def scratch_target(B, mon='posterior17'):
    out = os.path.join(odir('scratch'), f'scratch_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[scratch] t{B}: done')
        return
    t0 = time.time()
    R5m.SEED = 2025
    ctx = R6m.Ctx('repaug', B, mon)                     # queries, session table, features only
    c = ctx.c
    tr = c.ses['train']
    reps = R.load_reps(B, mon)
    cand = np.array([i for i in c.pool if (tr[i] == CS).sum() == 2])
    rows = []

    def eval_fn(m):
        o = {}
        with torch.no_grad():
            Gt = m.gallery(c.fte)
            for qk in QK:
                for rule in ('cos', 'csls_batch'):
                    o[(qk, rule)] = float(np.mean([R6m.top1(R6m.smat(m(torch.as_tensor(q, device=dev())), Gt, rule))
                                                   for q in c.Q[qk]]))
        return o
    for k in K_SCRATCH:
        if k > len(cand):
            continue
        for draw in range(R6m.ndraw(k)):
            ims = draw_images(cand, k, draw)
            rr = np.stack([np.nonzero(tr[i] == CS)[0] for i in ims])
            x2 = np.asarray(reps[ims[:, None], rr], np.float32)
            res = train_scratch(x2, ims, c.ftr, x2.shape[2], seed=draw, eval_fn=eval_fn)
            for ep, o in res.items():
                for (qk, rule), v in o.items():
                    rows.append(dict(k=k, draw=draw, epochs=ep, query=qk, rule=rule, top1=v))
            print(f'   scratch k={k} draw {draw} ({time.time() - t0:.0f}s): s4R4 cos by epoch '
                  + ', '.join(f'{ep}:{o[("s4R4", "cos")]:.1f}' for ep, o in res.items()), flush=True)
    df = pd.DataFrame(rows)
    df['target'], df['montage'] = B, mon
    df.to_csv(out, index=False)
    print(f'  [scratch] t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# C. calibration schedule (same budget of 1,000 presentations)
# ---------------------------------------------------------------------------------------------------------------
def schedule_target(B, mon='posterior17', seed=2026):
    out = os.path.join(odir('schedule'), f'sched_s{seed}_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[schedule] t{B}: done')
        return
    R5m.SEED = seed
    if not os.path.exists(N.ckpt_path('repaug', mon, B, seed, 'last')):
        print(f'[schedule] seed {seed} t{B}: no repaug checkpoint, skipping')
        return
    t0 = time.time()
    ctx = R6m.Ctx('repaug', B, mon)
    c = ctx.c
    tr = c.ses['train']
    reps = R.load_reps(B, mon)
    cand2 = np.array([i for i in c.pool if (tr[i] == CS).sum() == 2])
    cand4 = np.array([i for i in cand2 if (tr[i] < CS).sum() == 2])       # other 2 presentations on earlier days
    print(f'   images with 2 session-{CS} presentations: {len(cand2)}; of those, other 2 in sessions 1-2: {len(cand4)}')
    rows = []
    for name, (k, npres) in SCHEDULES.items():
        pool_ = cand4 if npres == 4 else cand2
        if len(pool_) < k:
            print(f'   {name}: only {len(pool_)} eligible images, skipping')
            continue
        for draw in range(3):
            if name == '500x2':
                ims = draw_images(cand2, K, draw)                          # identical to rounds 6-7
            else:
                ims = np.sort(np.random.RandomState(9000 + 31 * k + draw).choice(pool_, k, replace=False))
            if npres == 4:
                rr = np.stack([np.r_[np.nonzero(tr[i] == CS)[0], np.nonzero(tr[i] < CS)[0]] for i in ims])
            else:
                rr = np.stack([np.nonzero(tr[i] == CS)[0][:npres] for i in ims])
            xs = np.asarray(reps[ims[:, None], rr], np.float32)            # (k, npres, C, T)
            pos = torch.as_tensor(np.array([ctx.pool_pos[int(i)] for i in ims]), device=dev())
            Ymeas = c.pool_z[pos]
            d1 = c.mS['h1'] - I.acts(c.f0, xs.reshape(-1, *xs.shape[2:]))['h1'].mean(0)
            Xs = X.acts2(c.f0, xs.mean(1), d1)['z']
            rig = R.m_rigid(Xs, Ymeas, 1.0)
            Zq0 = c.zq(d1)
            for qk in QK:
                for qi, z0 in enumerate(Zq0[qk]):
                    for meth, z in (('zero', z0), ('rigid', rig(z0))):
                        for gal, Gt in (('anchor', ctx.Ganc), ('img', ctx.Gimg)):
                            for rule in ('cos', 'csls_batch'):
                                rows.append(dict(schedule=name, draw=draw, method=meth, gallery=gal, rule=rule, query=qk,
                                                 qdraw=qi, top1=R6m.top1(R6m.smat(z, Gt, rule))))
        print(f'   {name} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows).groupby(['schedule', 'draw', 'method', 'gallery', 'rule', 'query']).top1.mean().reset_index()
    df['target'], df['montage'], df['seed'] = B, mon, seed
    df.to_csv(out, index=False)
    print(f'  [schedule] t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def holm(ps):
    ks = sorted(ps, key=lambda k: (np.nan_to_num(ps[k], nan=1.0)))
    out, run = {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (len(ks) - i) * np.nan_to_num(ps[k], nan=1.0)))
        out[k] = round(run, 4)
    return out


def report(mon='posterior17'):
    pd.set_option('display.width', 250)
    audit()
    # ---- A
    f = sorted(glob.glob(os.path.join(odir('gallery'), f'gal_s*_t*_{mon}.csv')))
    if f:
        d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        per = d.groupby(['seed', 'gallery', 'method', 'rule', 'query', 'target']).top1.mean()
        print('\n' + '=' * 110 + f'\nA. GALLERY TEST ({mon}, repaug, k = {K} earlier-day draws; people 2-10; top-1 %)\n' + '=' * 110)
        print('predictor effective rank (img_rank uses it): '
              + ', '.join(f's{s}: {v:.1f}' for s, v in d.groupby('seed').eff_rank.mean().items()))
        T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3, 4]).mean()
        for seed in sorted(d.seed.unique(), reverse=True):
            for rule in ('cos', 'csls_batch'):
                print(f'\nseed {seed}, rule {rule}')
                print(T.xs((seed, rule), level=('seed', 'rule')).unstack('query')[QK].unstack('method').round(2).to_string())

        def g(gal, meth='zero', rule='cos', q='s4R4', seed=2026):
            return per.xs((seed, gal, meth, rule, q), level=('seed', 'gallery', 'method', 'rule', 'query'))

        def best_img(meth='zero', rule='cos', q='s4R4', seed=2026):
            return pd.concat([g(x, meth, rule, q, seed) for x in IMG_ONLY], axis=1).max(axis=1)
        s0 = 2026 if 2026 in set(d.seed) else int(d.seed.max())
        print(f'\nPRIMARY (seed {s0}, zero labels, s4R4, each query scored on its own)')
        p0 = R4m.pstats(g('anchor', seed=s0), g('img_spec', seed=s0), 'anchor - img_spec (predicted population vs '
                        'spectrum-matched image-only gallery)', primary=True)
        ps = {'anchor - img_rank': R4m.pstats(g('anchor', seed=s0), g('img_rank', seed=s0), 'anchor - img_rank (rank-matched)'),
              'anchor - best image-only': R4m.pstats(g('anchor', seed=s0), best_img(seed=s0),
                                                     'anchor - best image-only gallery per person (conservative)'),
              'anchor - permconcept': R4m.pstats(g('anchor', seed=s0), g('anchor_permconcept', seed=s0),
                                                 'anchor - anchor_permconcept (image-level correspondence)'),
              'oracle - anchor': R4m.pstats(g('oracle', seed=s0), g('anchor', seed=s0),
                                            'oracle - anchor (measured vs predicted; PRIVILEGED)')}
        print(f'  HOLM over the secondaries: {holm(ps)}')
        print('\nBATCH RULE (csls_batch) and other queries')
        pc = R4m.pstats(g('anchor', rule='csls_batch', seed=s0), g('img_spec', rule='csls_batch', seed=s0),
                        'anchor - img_spec, csls_batch, s4R4')
        for q in ('s4R1', 's4R20', 'bench'):
            R4m.pstats(g('anchor', q=q, seed=s0), g('img_spec', q=q, seed=s0), f'anchor - img_spec, cos, {q}')
        print('\nWITH 500 LABELS (rigid to each gallery\'s own representation of the calibration images)')
        pr = R4m.pstats(g('anchor', 'rigid', seed=s0), g('img_spec', 'rigid', seed=s0), 'anchor - img_spec, rigid k = 500, cos, s4R4')
        R4m.pstats(g('anchor', 'rigid', seed=s0), best_img('rigid', seed=s0), 'anchor - best image-only, rigid k = 500, cos, s4R4')
        R4m.pstats(g('anchor_meas', 'rigid', seed=s0), g('img_spec', 'rigid', seed=s0),
                   'anchor (rigid to measured responses, rounds 6-7) - img_spec, rigid k = 500, cos, s4R4')
        R4m.pstats(g('anchor', 'rigid', 'csls_batch', seed=s0), g('img_spec', 'rigid', 'csls_batch', seed=s0),
                   'anchor - img_spec, rigid k = 500, csls_batch, s4R4')
        print('\nCALIBRATION WITH EACH QUERY SCORED ON ITS OWN (rigid - zero, same gallery, s4R4)')
        for gal in ('anchor', 'img_spec', 'img'):
            R4m.pstats(g(gal, 'rigid', seed=s0), g(gal, seed=s0), f'{gal}: rigid k = 500 - zero labels, cos')
        R4m.pstats(g('anchor_meas', 'rigid', seed=s0), g('anchor', seed=s0), 'anchor_meas (round 7 method): rigid - zero, cos')
        R4m.pstats(g('anchor_meas', 'rigid', 'csls_batch', seed=s0), g('anchor', rule='csls_batch', seed=s0),
                   'anchor_meas (round 7 method): rigid - zero, csls_batch (should reproduce round 7, +3.89)')
        R4m.pstats(g('anchor_meas', 'rigid', q='s4R1', seed=s0), g('anchor', q='s4R1', seed=s0),
                   'anchor_meas: rigid - zero, cos, s4R1 (single presentation)')
        if 2025 in set(d.seed) and s0 != 2025:
            print('\nREPLICATION ON THE SEED-2025 DECODER')
            R4m.pstats(g('anchor', seed=2025), g('img_spec', seed=2025), 'anchor - img_spec, zero, cos, s4R4')
            R4m.pstats(g('anchor', seed=2025), best_img(seed=2025), 'anchor - best image-only, zero, cos, s4R4')
            R4m.pstats(g('anchor', 'rigid', seed=2025), g('img_spec', 'rigid', seed=2025), 'anchor - img_spec, rigid k = 500, cos, s4R4')
        print('\nTEMPLATE PREDICTOR (headroom, then the retrieval-trained predictor)')
        R4m.pstats(g('oracle', 'rigid', seed=s0), g('anchor_meas', 'rigid', seed=s0),
                   'headroom: oracle - anchor (rounds 6-7 method), rigid k = 500, cos, s4R4 (PRIVILEGED)')
        if 'ret_val_best' in d:
            sel = d[d.seed == s0].groupby('target')[['ret_val_ridge', 'ret_val_best', 'ret_lam', 'ret_step']].first()
            kept = int((sel.ret_lam.astype(str) == 'ridge').sum())
            print(f'  source-only selection: val top-1 ridge {sel.ret_val_ridge.mean():.2f} -> {sel.ret_val_best.mean():.2f}; '
                  f'ridge kept for {kept}/{len(sel)} folds; chosen lambdas {sel.ret_lam.astype(str).value_counts().to_dict()}')
        ret_p = R4m.pstats(g('anchor_ret', 'rigid', seed=s0), g('anchor_meas', 'rigid', seed=s0),
                           'anchor_ret - anchor (rounds 6-7 method), rigid k = 500, cos, s4R4 (prespecified; threshold +2)')
        R4m.pstats(g('anchor_ret', 'rigid', seed=s0), g('anchor', 'rigid', seed=s0), 'anchor_ret - anchor, both rigid to their own templates, cos, s4R4')
        R4m.pstats(g('anchor_ret', seed=s0), g('anchor', seed=s0), 'anchor_ret - anchor, zero labels, cos, s4R4')
        R4m.pstats(g('anchor_ret', rule='csls_batch', seed=s0), g('anchor', rule='csls_batch', seed=s0), 'anchor_ret - anchor, zero labels, csls_batch, s4R4')
        R4m.pstats(g('anchor_ret', 'rigid', 'csls_batch', seed=s0), g('anchor_meas', 'rigid', 'csls_batch', seed=s0),
                   'anchor_ret - anchor (rounds 6-7 method), rigid k = 500, csls_batch, s4R4')
        R4m.pstats(g('anchor_ret', seed=s0), g('img_rank', seed=s0), 'anchor_ret - img_rank, zero labels, cos, s4R4')
        dr = (g('anchor_ret', 'rigid', seed=s0) - g('anchor_meas', 'rigid', seed=s0)).reindex(TEST).mean()
        print(f'  -> anchor_ret {"CLEARS" if dr >= 2 and ret_p is not None and ret_p < 0.05 else "does not clear"} the '
              f'+2-point threshold (mean {dr:+.2f})')
        print('\nDECISION RULE')
        sig = lambda p: p is not None and not np.isnan(p) and p < 0.05  # noqa: E731
        da = (g('anchor', seed=s0) - g('img_spec', seed=s0)).reindex(TEST).mean()
        dc = (g('anchor', rule='csls_batch', seed=s0) - g('img_spec', rule='csls_batch', seed=s0)).reindex(TEST).mean()
        if sig(p0) and da > 0:
            print('  anchor beats the image-only control with each query scored on its own -> lead with "other people\'s '
                  'EEG builds a better retrieval gallery for images no one has recorded"; cross-day calibration second.')
        elif sig(pc) and dc > 0:
            print('  anchor beats image-only only under batch CSLS -> the gallery claim is limited to batch retrieval.')
        else:
            print('  image-only geometry explains the gallery gain -> the paper is the calibration study.')
        if pr is not None and not np.isnan(pr):
            print(f'  with 500 labels: anchor - img_spec Wilcoxon p = {pr:.4f} (does the gallery advantage survive calibration?)')
    # ---- B
    f = sorted(glob.glob(os.path.join(odir('scratch'), f'scratch_t*_{mon}.csv')))
    if f:
        s = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        print('\n' + '=' * 110 + f'\nB. FROM-SCRATCH CONTROL ({mon}, same calibration images as rounds 6-7, earlier day; top-1 %)\n' + '=' * 110)
        sp = s.groupby(['k', 'epochs', 'query', 'rule', 'target']).top1.mean()
        if 1 in set(s.target):
            p1 = sp.xs(('s4R4', 'cos', 1), level=('query', 'rule', 'target')).groupby('epochs').mean()
            ep = int(p1.idxmax())
            print(f'epochs chosen on person 1 (s4R4, cos): {ep}   person-1 by epoch: ' + ', '.join(f'{e}:{v:.2f}' for e, v in p1.items()))
        else:
            ep = int(s.epochs.max())
            print(f'WARNING: no person-1 scratch run; using the longest training ({ep} epochs) instead of a selected one')
        T = sp[sp.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3]).mean()
        print('people 2-10, all epochs (descriptive):')
        print(T.unstack('query')[QK].round(2).to_string())
        scr = lambda k, q='s4R4', rule='cos': sp.xs((k, ep, q, rule), level=('k', 'epochs', 'query', 'rule'))  # noqa: E731
        f6 = sorted(glob.glob(os.path.join(N.ENV['results_dir'], 'nb_round6', f'cal_t*_{mon}.csv')))
        f7 = sorted(glob.glob(os.path.join(N.ENV['results_dir'], 'nb_round7', 'combo', f'combo_t*_{mon}.csv')))
        if f6:
            r6 = pd.concat([pd.read_csv(x) for x in f6], ignore_index=True)
            r6 = r6[(r6.decoder.isin(['repaug', 'within'])) & (r6.cal_session.isin([CS, 0]))]
            p6 = r6.groupby(['k', 'method', 'readout', 'rule', 'query', 'target']).top1.mean()
            cal = lambda k, meth, ro, q='s4R4', rule='cos': p6.xs((k, meth, ro, rule, q), level=('k', 'method', 'readout', 'rule', 'query'))  # noqa: E731
            within = lambda q='s4R4', rule='cos': p6.xs((0, 'within', 'img', rule, q), level=('k', 'method', 'readout', 'rule', 'query'))  # noqa: E731
            print('\ncross-subject calibrated vs scratch on the same images (s4R4):')
            for k in K_SCRATCH:
                for rule in ('cos', 'csls_batch'):
                    try:
                        R4m.pstats(cal(k, 'rigid', 'anchor', rule=rule), scr(k, rule=rule), f'rigid (cross-subject) - scratch, k = {k}, {rule}')
                        R4m.pstats(cal(k, 'ft', 'img', rule=rule), scr(k, rule=rule), f'fine-tune (cross-subject) - scratch, k = {k}, {rule}')
                    except KeyError:
                        pass
            for k in K_SCRATCH:
                try:
                    R4m.pstats(scr(k), within(), f'scratch k = {k} - within-person decoder on all data (cos)')
                except KeyError:
                    pass
            w = within().reindex(TEST).mean()
            r500 = cal(500, 'rigid', 'anchor').reindex(TEST).mean()
            lv = {k: scr(k).reindex(TEST).mean() for k in K_SCRATCH if (k, ep) in set(zip(s.k, s.epochs))}
            print(f'\nlevels (s4R4, cos): within-person (all data) {w:.2f}; rigid at 500 {r500:.2f}; scratch '
                  + ', '.join(f'{k}: {v:.2f}' for k, v in lv.items()))
            hit = [k for k, v in lv.items() if v >= r500]
            print('  smallest scratch k that reaches rigid at 500: ' + (str(min(hit)) if hit else f'none up to {max(lv)}'))
        if f7:
            r7 = pd.concat([pd.read_csv(x) for x in f7], ignore_index=True)
            p7 = r7.groupby(['k', 'method', 'readout', 'rule', 'query', 'target']).top1.mean()
            for k in (500, 2000, 4000):
                for rule in ('cos', 'csls_batch'):
                    try:
                        R4m.pstats(p7.xs((k, 'ft+rigid', 'anchor', rule, 's4R4'), level=('k', 'method', 'readout', 'rule', 'query')),
                                   scr(k, rule=rule), f'ft+rigid (cross-subject) - scratch, k = {k}, {rule}')
                    except KeyError:
                        pass
    # ---- C
    f = sorted(glob.glob(os.path.join(odir('schedule'), f'sched_s*_t*_{mon}.csv')))
    if f:
        sc = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        print('\n' + '=' * 110 + f'\nC. CALIBRATION SCHEDULE ({mon}, seed 2026, 1,000 presentations each; top-1 %, people 2-10)\n' + '=' * 110)
        pp = sc.groupby(['schedule', 'method', 'gallery', 'rule', 'query', 'target']).top1.mean()
        T = pp[pp.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3, 4]).mean()
        print(T.xs('anchor', level='gallery').unstack('query')[QK].unstack('schedule').round(2).to_string())

        def h(sch, q, meth='rigid', rule='cos'):
            return pp.xs((sch, meth, 'anchor', rule, q), level=('schedule', 'method', 'gallery', 'rule', 'query'))
        for q in ('s4R1', 's4R4'):
            for rule in ('cos', 'csls_batch'):
                try:
                    R4m.pstats(h('1000x1', q, rule=rule), h('500x2', q, rule=rule), f'more images: 1000x1 - 500x2, rigid, {q}, {rule}')
                    R4m.pstats(h('250x4', q, rule=rule), h('500x2', q, rule=rule), f'more repeats: 250x4 - 500x2, rigid, {q}, {rule} (spans days)')
                except KeyError:
                    pass
        for sch in SCHEDULES:
            try:
                R4m.pstats(h(sch, 's4R1'), h(sch, 's4R1', 'zero'), f'{sch}: rigid - zero, s4R1, cos')
            except KeyError:
                pass
        try:
            R4m.pstats(h('500x2', 's4R4'), h('500x2', 's4R1'), 'query repetition: s4R4 - s4R1, 500x2 rigid, cos (for scale)')
        except KeyError:
            pass
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(''), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['audit', 'gallery', 'scratch', 'schedule', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--reps_dir', default='/content/nb_reps')
    p.add_argument('--montage', default='posterior17')
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'audit':
        ok = audit(mon=a.montage)
        sys.exit(0 if ok else 1)
    if a.cmd == 'gallery':
        for seed in a.seeds:
            for B in a.targets:
                print(f'[gallery] seed {seed} target {B}', flush=True)
                gallery_target(B, seed, a.montage)
        return
    if a.cmd == 'scratch':
        for B in a.targets:
            print(f'[scratch] target {B}', flush=True)
            scratch_target(B, a.montage)
        return
    if a.cmd == 'schedule':
        for B in a.targets:
            print(f'[schedule] target {B}', flush=True)
            schedule_target(B, a.montage)
        return
    report(a.montage)


if __name__ == '__main__':
    main()
