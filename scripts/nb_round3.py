"""Round 3 (NeuroBridge): matched-access calibration, transferable structure, and the Crown acquisition budget.

    python scripts/nb_round3.py prep   --config C [--subjects 1 ... 10]            # per-repetition cache (local disk)
    python scripts/nb_round3.py run    --config C [--targets 1 ... 10] [--montages posterior17 occipital8 crown8]
    python scripts/nb_round3.py report --config C [--montages ...]

Why this round. An external review of rounds 1-2 asked for (a) the incremental value of activation matching (AM)
over the strongest translation / rigid alignment at the SAME calibration access, (b) whether the "16-dim" structure
transfers from other people to a new person prospectively, not just under same-image patching, (c) whether the
Crown-position failure is signal, calibration data or session drift, all evaluated on a LATER session with realistic
repetitions, people as the unit, and comparisons fixed before looking.

Sessions. NeuroBridge's preprocessing concatenates sessions in order: test repetitions 0-19 / 20-39 / 40-59 / 60-79
are sessions 1 / 2 / 3 / 4. Each training image was shown in two sessions, 2 repetitions each; repetitions 0-1 come
from the earlier of the two (sessions 1-3), 2-3 from the later. So calibration data drawn from training repetition 0
(or 0-1) always precedes the session-4 test repetitions used as queries: a clean cross-session evaluation.
`prep` rebuilds the per-repetition training EEG (the Drive cache only kept the 4-repetition average) on local disk.

Evaluation (Track B, the paper evaluation): calibrate on k known images x r presentations from earlier sessions;
query = the 200 test images averaged over R session-4 presentations (R = 1, 2, 4, 8, 20; 3 query draws averaged);
also the 80-repetition benchmark for continuity. Top-1 (and MRR). Targets 2-10 are scored; target 1 is run the same way
and used ONLY to select ranks / shrinkage / regularization (the selection rule is applied in `report`).
Prespecified primary contrasts (posterior-17 unless stated; k = 25 images x 1 presentation; query R = 4 session-4):
  P1 calibration: the subject-1-selected low-rank AM vs the subject-1-selected best of {shrunk translation, rigid}.
  P2 mechanism:  AM inside the source-only mismatch basis vs inside source-activation PCs, same rank.
  P3 deployment: on Crown-8, the selected calibrated method vs the strongest simple baseline.
  People are the units: per-person means over support draws, all 9 differences, bootstrap CI, exact sign test,
  Wilcoxon, leave-one-person-out range, Holm over P1-P3.

Experiments per target (f0 = leave-target-out NeuroBridge, epoch 50; anchors = mean f0 output of the 9 source people
on the same calibration image, from their 4-repetition averages):
  X1 factorial. Calibration target {neural anchors, image embeddings (scaled)} x map {translation shrunk (kappa 0/5/25),
     rigid = identity-regularized orthogonal Procrustes + global scale (lambda 0.1/1), reduced-rank residual ridge
     (rank 1/2/4/8/16), full residual ridge} x label-free layer-1 offset {off, on} x readout {image gallery,
     predicted population anchors (CHORUS-style: ridge from image features to source anchors)}. k = 5/10/25/50/200/2000
     (x 1 presentation; 20 support draws up to 50) and a 4-presentation benchmark track. Controls: anchors shuffled
     within the support (centered maps), anchors from different images (translation).
  X1L layer test: AM at h3 vs an image-loss adapter at h3 and at z with the same rank / init / steps.
  X1T team baseline: channel map + rank-64 output edit (CLIP loss).
  X1F label-free competitors: layer-1 evoked offset, output mean / mean+scale matching, sensor Euclidean alignment;
     transductive (use the unlabeled query batch): offset from the batch, batch whitening, CSLS hubness correction,
     SCORE-lite (mutual-nearest-neighbour landmarks + identity-regularized orthogonal map, iterated).
  X2 basis. Bases from source people only, in f0's frozen output: centered / uncentered mismatch (each source person
     minus the other sources' mean, same images), source-activation PCs, gallery PCs, variance-matched random, isotropic
     random; ranks 2-32. Deployable: target fits offset + U A U^T (A full ridge or diagonal) from k = 5-50 images.
     Oracle panel: same-image single donor, repeat-matched source mean, wrong-image donor, same-category wrong-image
     donor, independent target repeat, removal of the target's component, offset-only. Subspace overlap via principal
     angles on disjoint image halves.
  X3 acquisition. All montages: calibration designs at fixed cost (40 = 40x1 / 20x2 / 10x4; 20 = 20x1 / 10x2 / 5x4) x
     query repetitions (1-20, session 4) x methods {f0, LF offset, shrunk translation, rigid, AM rank 2 / 4, old map +
     fresh offset, within-person decoder}. Signal diagnostics: model-free image identification from raw EEG (split
     sessions, grand evoked response removed) within person and from source anchors.
  X4 anchor panels: reliability-selected, diversity-selected, both, vs random panels (k = 5 / 10 / 25).
Outputs: <results>/nb_round3/<exp>_tNN_<mon>.csv + manifest_tNN_<mon>.json; report tables, primary statistics, figures.
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
import nb_interp as I  # noqa: E402
import nb_next as X  # noqa: E402

PEOPLE = list(range(1, 11))
N_CAL = 2000
QR = [1, 2, 4, 8, 20]
QKEYS = ['bench'] + [f's4R{r}' for r in QR]
K_B = [5, 10, 25, 50, 200, 2000]
RANKS = [1, 2, 4, 8, 16]
BASIS_RANKS = [2, 4, 8, 16, 32]
U_LF = 100


def dev():
    return N.ENV['device']


def odir():
    d = os.path.join(N.ENV['results_dir'], 'nb_round3')
    os.makedirs(d, exist_ok=True)
    return d


def ndr(k):
    return 20 if k <= 50 else (10 if k <= 200 else 3)


def reps_path(sub, mon):
    return os.path.join(N.ENV.get('reps_dir', '/content/nb_reps'), f'sub-{sub:02d}', f'{mon}.npy')


# ---------------------------------------------------------------------------------------------------------------
# prep: per-repetition training EEG on local disk
# ---------------------------------------------------------------------------------------------------------------
def prep(subjects, montages=('posterior17', 'crown8'), local='/content/nb_raw'):
    from eegalign import ubp as U
    import nb_data
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
    ch = json.load(open(os.path.join(N.ENV['cache_dir'], 'info.json')))['ch_names']
    for s in subjects:
        if all(os.path.exists(reps_path(s, m)) for m in montages):
            print(f'  sub-{s:02d}: per-repetition cache present')
            continue
        t0 = time.time()
        f = nb_data.dl(f'things_eeg/preprocessed_eeg/sub-{s:02d}/train.npy', local)
        tr = np.load(f, mmap_mode='r')                       # (1654, 10, 4, 63, 250)
        for m in montages:
            idx = [ch.index(c) for c in U.MONTAGES[m]]
            out = np.empty((16540, 4, len(idx), 250), np.float16)
            for i in range(0, 1654, 64):
                blk = np.asarray(tr[i:i + 64, :, :, idx, :250], np.float32)
                out[i * 10:(i + blk.shape[0]) * 10] = blk.reshape(-1, 4, len(idx), 250)
            os.makedirs(os.path.dirname(reps_path(s, m)), exist_ok=True)
            np.save(reps_path(s, m), out)
        del tr
        os.remove(f)
        print(f'  sub-{s:02d}: per-repetition cache built ({time.time() - t0:.0f}s)', flush=True)


def load_reps(sub, mon):
    from eegalign import ubp as U
    p = reps_path(sub, mon)
    if os.path.exists(p):
        return np.load(p, mmap_mode='r')
    sp = reps_path(sub, 'posterior17')
    if os.path.exists(sp) and set(U.MONTAGES[mon]) <= set(U.MONTAGES['posterior17']):
        idx = [U.MONTAGES['posterior17'].index(c) for c in U.MONTAGES[mon]]
        return np.load(sp, mmap_mode='r')[:, :, idx]
    raise FileNotFoundError(f'{p} missing: run `nb_round3.py prep` first')


# ---------------------------------------------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def score(z, G):
    s = F.normalize(z, dim=-1) @ G.T
    r = (s > s.diagonal()[:, None]).sum(1).float()
    return 100 * float((r == 0).float().mean()), float((1 / (r + 1)).mean())


def T(a):
    return torch.as_tensor(np.asarray(a, np.float32), device=dev())


def safe_svd(M, full_matrices=False):
    """SVD in float64, with a tiny jitter retry (degenerate matrices can fail to converge in float32)."""
    Md = M.double()
    try:
        U, S, Vh = torch.linalg.svd(Md, full_matrices=full_matrices)
    except Exception:                                                  # noqa: BLE001
        Md = Md + 1e-9 * Md.abs().mean() * torch.randn_like(Md)
        U, S, Vh = torch.linalg.svd(Md, full_matrices=full_matrices)
    return U.to(M.dtype), S.to(M.dtype), Vh.to(M.dtype)


def ortho(A):
    return torch.linalg.qr(A)[0]


def topk(D, k, center=True):
    D = D - D.mean(0) if center else D
    return safe_svd(D)[2][:k].T


def p_angles(U1, U2):
    s = torch.linalg.svdvals(U1.T @ U2)
    return float((s ** 2).mean())


# maps at one stage: X (k x d) B's support activations, Y (k x d) targets. Each returns fn(Z) -> Z'
def m_trans(X_, Y_, kappa):
    c = (Y_ - X_).mean(0) * (len(X_) / (len(X_) + kappa))
    return lambda Z: Z + c


def m_rigid(X_, Y_, lam):
    xm, ym = X_.mean(0), Y_.mean(0)
    Xc, Yc = X_ - xm, Y_ - ym
    M = Xc.T @ Yc
    s = float(torch.trace(Xc.T @ Xc)) / Xc.shape[1] + 1e-8
    U, _, Vh = safe_svd(M + lam * s * torch.eye(M.shape[0], device=M.device))
    R = U @ Vh
    g = float(torch.trace(R.T @ M)) / (float(torch.trace(Xc.T @ Xc)) + 1e-8)
    g = min(max(g, 0.5), 2.0)
    return lambda Z: (Z - xm) @ R * g + ym


def m_sub(X_, Y_, U, diag=False):
    """offset + U A U^T residual map restricted to basis U."""
    R = Y_ - X_
    c = R.mean(0)
    xm = X_.mean(0)
    P, Q = (X_ - xm) @ U, (R - c) @ U
    if diag:
        a = (P * Q).sum(0) / ((P * P).sum(0) + 1e-3 * (P * P).sum(0).mean() + 1e-8)
        return lambda Z: Z + c + (((Z - xm) @ U) * a) @ U.T
    A, pm, qm = X.ridge_svd(P, Q)
    return lambda Z: Z + c + (((Z - xm) @ U - pm) @ A + qm) @ U.T


# ---------------------------------------------------------------------------------------------------------------
# per-target context
# ---------------------------------------------------------------------------------------------------------------
class R3:
    def __init__(self, B, mon, which='last'):
        t0 = time.time()
        self.B, self.mon = B, mon
        self.src = [p for p in PEOPLE if p != B]
        self.f0, self.w = I.load_models(B, mon, which)
        self.ftr, self.fte = N.features('train', 'aug').to(dev()), N.features('test', 'aug').to(dev())
        with torch.no_grad():
            self.G, self.Gw, self.Gtr = self.f0.gallery(self.fte), self.w.gallery(self.fte), self.f0.gallery(self.ftr)
        d = N.subject_data(B, mon)
        self.labels, self.test_paths, self.train_paths = d['train_labels'], d['test_paths'], d['train_paths']
        self.pool, self.draws = N.split(self.labels)
        self.held = np.concatenate(self.draws)
        self.cal = np.sort(np.random.RandomState(1).permutation(self.pool)[:N_CAL])
        tt = d['test_trials'].astype(np.float32)                      # (200, 80, C, T)
        self.C = tt.shape[2]
        # queries: benchmark (80 reps) + session-4 averages of R reps (3 draws each, 1 for R = 20)
        self.Q = {'bench': [d['test'].astype(np.float32)]}
        for R in QR:
            self.Q[f's4R{R}'] = [tt[:, 60 + np.random.RandomState(1000 + q).choice(20, R, replace=False)].mean(1)
                                 for q in range(3 if R < 20 else 1)]
        self.s12, self.s34 = tt[:, :40].mean(1), tt[:, 40:].mean(1)
        self.tt4 = tt[:, 60:80]
        del tt
        # target per-repetition training EEG (earlier sessions)
        reps = load_reps(B, mon)
        self.cal_reps = np.asarray(reps[self.cal], np.float32)          # (2000, 4, C, T)
        pick = np.random.RandomState(2).choice(np.setdiff1d(self.pool, self.cal), 1000, replace=False)
        self.unl = np.asarray(reps[pick, 0], np.float32)                # unlabeled single presentations
        self.held_B = d['train'][self.held].astype(np.float32)
        del reps, d
        # source people: anchors on calibration images, pool outputs for the anchor predictor, held outputs
        acc_cal = {s: 0 for s in ('h1', 'h3', 'z')}
        self.Zsrc_cal, self.Zsrc_held, self.Zsrc_test, self.Zsrc_s34 = [], [], [], []
        pool_z = torch.zeros(len(self.pool), 512, device=dev())
        raw_stats = []
        for s in self.src:
            ds = N.subject_data(s, mon)
            A = I.acts(self.f0, ds['train'][self.cal].astype(np.float32))
            for k in acc_cal:
                acc_cal[k] = acc_cal[k] + A[k] / len(self.src)
            self.Zsrc_cal.append(A['z'])
            self.Zsrc_held.append(I.acts(self.f0, ds['train'][self.held].astype(np.float32))['z'])
            self.Zsrc_test.append(I.acts(self.f0, ds['test'].astype(np.float32))['z'])
            tts = ds['test_trials']
            self.Zsrc_s34.append(I.acts(self.f0, tts[:, 40:].astype(np.float32).mean(1))['z'])
            for i in range(0, len(self.pool), 4096):
                pool_z[i:i + 4096] += I.acts(self.f0, ds['train'][self.pool[i:i + 4096]].astype(np.float32))['z'] / len(self.src)
            raw_stats.append(ds['train'][np.random.RandomState(3).choice(self.pool, 300, replace=False)].astype(np.float32))
            self.raw_test_src = getattr(self, 'raw_test_src', 0) + ds['test'].astype(np.float32) / len(self.src)
            del ds, tts
        self.anchor = acc_cal                                           # mean source activations (h1, h3, z) per cal image
        self.Zsrc_cal, self.Zsrc_held = torch.stack(self.Zsrc_cal), torch.stack(self.Zsrc_held)
        self.Zsrc_test, self.Zsrc_s34 = torch.stack(self.Zsrc_test), torch.stack(self.Zsrc_s34)
        self.src_raw = np.concatenate(raw_stats)
        # source statistics for label-free offsets
        As = I.acts(self.f0, self.src_raw)
        self.mS = {k: As[k].mean(0) for k in As}
        self.sS = {k: As[k].std(0) + 1e-6 for k in As}
        self.znorm = float(self.anchor['z'].norm(dim=1).mean())
        # CHORUS-style anchor predictor: image features -> population anchor (ridge on pool images)
        self.pool_z = pool_z
        Bm, xm, ym = X.ridge_svd(self.ftr[self.pool], pool_z)
        self.P = F.normalize((self.fte - xm) @ Bm + ym, dim=-1)
        self.P_cal_check = float(F.cosine_similarity((self.ftr[self.cal] - xm) @ Bm + ym, self.anchor['z'], dim=1).mean())
        print(f'   context ready ({time.time() - t0:.0f}s); anchor-predictor cos on cal images {self.P_cal_check:.3f}',
              flush=True)

    def lf_delta(self, U=U_LF, seed=0):
        sel = np.random.RandomState(500 + seed).choice(len(self.unl), U, replace=False)
        return self.mS['h1'] - I.acts(self.f0, self.unl[sel])['h1'].mean(0)

    def cal_input(self, r):
        """B's calibration EEG with r presentations per image (r = 1: repetition 0; 2: 0-1; 4: all)."""
        return self.cal_reps[:, :r].mean(1)

    def qacts(self, d1=None, stage='z'):
        out = {}
        for k, lst in self.Q.items():
            out[k] = [X.acts2(self.f0, q, d1)[stage] for q in lst]
        return out


def eval_fn(c, fn, Qz, G):
    """Apply fn to every query set (stage z) and score; returns dict of top-1 per query key (+ MRR for 2 keys)."""
    res = {}
    with torch.no_grad():
        for k, lst in Qz.items():
            v = [score(fn(z), G) for z in lst]
            res[k] = float(np.mean([a for a, _ in v]))
            if k in ('bench', 's4R4'):
                res[f'mrr_{k}'] = float(np.mean([b for _, b in v]))
    return res


def manifest(c):
    return dict(target=c.B, montage=c.mon, decoder_training_people=c.src, decoder_training_concepts='pool (1454)',
                anchor_people=c.src, anchor_images='2000 pool images (cal), sources\' 4-rep averages',
                basis_people=c.src, basis_images='200 held-out concepts (2000 images)',
                target_calibration='cal images, training repetitions 0 (r=1), 0-1 (r=2), 0-3 (r=4); sessions 1-3 for r<=2',
                target_unlabeled='1000 other pool images, repetition 0', queries='official test images, session 4 '
                'repetitions (R = 1-20) and 80-rep benchmark', selection='target 1 only (applied in report)',
                anchor_predictor='ridge from augmented image features to source anchors, pool images')


# ---------------------------------------------------------------------------------------------------------------
# X1 factorial (+ controls)
# ---------------------------------------------------------------------------------------------------------------
def x1(c):
    rows = []
    f0 = c.f0
    Qz = {lf: None for lf in (0, 1)}
    d1_all = c.lf_delta(U_LF, 0)
    Qz[0], Qz[1] = c.qacts(None), c.qacts(d1_all)
    for lf in (0, 1):
        for ro, G in (('img', c.G), ('anchor', c.P)):
            rows.append(dict(exp='X1', track='-', k=0, draw=0, cal='-', lf=lf, map='none', readout=ro,
                             **eval_fn(c, lambda Z: Z, Qz[lf], G)))
    Ysrc = c.anchor['z']
    Yimg = c.Gtr[c.cal] * c.znorm
    for track, r, ks in (('B', 1, K_B), ('bench4', 4, [25, 200])):
        Zc = {0: X.acts2(f0, c.cal_input(r))['z'], 1: X.acts2(f0, c.cal_input(r), d1_all)['z']}
        for k in ks:
            for draw in range(ndr(k)):
                ix = torch.as_tensor(np.random.RandomState(700 + draw).permutation(N_CAL)[:k], device=dev())
                other = torch.as_tensor(np.random.RandomState(900 + draw).permutation(N_CAL)[:k], device=dev())
                for tgt, Y in (('neural', Ysrc), ('image', Yimg)):
                    for lf in (0, 1):
                        Xs, Ys = Zc[lf][ix], Y[ix]
                        fns = {f'trans_k{kp}': m_trans(Xs, Ys, kp) for kp in (0, 5, 25)}
                        fns.update({f'rigid_l{lm}': m_rigid(Xs, Ys, lm) for lm in (0.1, 1.0)})
                        B_, xm, rm = X.ridge_svd(Xs, Ys - Xs)
                        Qall = safe_svd((Xs - xm) @ B_)[2]
                        for rk in RANKS:
                            Qr = Qall[:rk].T
                            fns[f'rr{rk}'] = (lambda Br, xm=xm, rm=rm: (lambda Z: Z + rm + (Z - xm) @ Br))(B_ @ Qr @ Qr.T)
                        fns['full'] = (lambda Br, xm=xm, rm=rm: (lambda Z: Z + rm + (Z - xm) @ Br))(B_)
                        if tgt == 'neural':
                            perm = torch.as_tensor(np.random.RandomState(950 + draw).permutation(k), device=dev())
                            Bs, xs, rs = X.ridge_svd(Xs, Ys[perm] - Xs)
                            Qs = safe_svd((Xs - xs) @ Bs)[2][:2].T
                            fns['rr2_shuffled'] = (lambda Br, xs=xs, rs=rs: (lambda Z: Z + rs + (Z - xs) @ Br))(Bs @ Qs @ Qs.T)
                            fns['trans_k5_otherimgs'] = m_trans(Xs, Y[other], 5)
                        if tgt == 'image':
                            fns = {m: fns[m] for m in ('trans_k5', 'rigid_l1.0', 'rr2', 'rr16', 'full') if m in fns}
                        for name, fn in fns.items():
                            ros = (('img', c.G), ('anchor', c.P)) if (tgt == 'neural' and k in (25, 200)) else (('img', c.G),)
                            for ro, G in ros:
                                rows.append(dict(exp='X1', track=track, k=k, draw=draw, cal=tgt, lf=lf, map=name,
                                                 readout=ro, **eval_fn(c, fn, Qz[lf], G)))
            print(f'     X1 track {track} k={k} done', flush=True)
    return rows, d1_all


def lowrank_train(c, stage, Xs, img, rank, Zq_stage, steps=300, seed=0):
    """Image-loss low-rank residual adapter at `stage` (h3 or z), trained through the rest of f0."""
    torch.manual_seed(seed)
    d = Xs.shape[1]
    Ua = (0.01 * torch.randn(d, rank, device=dev())).requires_grad_(True)
    Va = torch.zeros(d, rank, device=dev()).requires_grad_(True)
    cb = torch.zeros(d, device=dev()).requires_grad_(True)
    m0 = Xs.mean(0)
    opt = torch.optim.Adam([Ua, Va, cb], lr=1e-3)
    scale = c.f0.scale()
    for _ in range(steps):
        bi = torch.randint(len(Xs), (min(len(Xs), 256),), device=dev())
        a = Xs[bi] + cb + (Xs[bi] - m0) @ Ua @ Va.T
        z = I.forward_from(c.f0, stage, a)
        gal = torch.cat([c.Gtr[img[bi]], c.Gtr[torch.randint(len(c.Gtr), (1024,), device=dev())]])
        loss = F.cross_entropy(scale * z @ gal.T, torch.arange(len(bi), device=dev())) + 1e-4 * (
            Ua.pow(2).sum() + Va.pow(2).sum() + cb.pow(2).sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    Ua, Va, cb = Ua.detach(), Va.detach(), cb.detach()
    return lambda A: A + cb + (A - m0) @ Ua @ Va.T


def x1_layer_and_team(c, d1_all):
    rows = []
    f0 = c.f0
    img_all = torch.as_tensor(c.cal, device=dev())
    Ah3 = X.acts2(f0, c.cal_input(1), d1_all)
    Qh3 = c.qacts(d1_all, 'h3')
    Qz = c.qacts(d1_all, 'z')
    for k in (25, 200):
        for draw in range(3):
            ix = torch.as_tensor(np.random.RandomState(700 + draw).permutation(N_CAL)[:k], device=dev())
            for rk in (4, 16):
                B_, xm, rm = X.ridge_svd(Ah3['h3'][ix], c.anchor['h3'][ix] - Ah3['h3'][ix])
                Qr = safe_svd((Ah3['h3'][ix] - xm) @ B_)[2][:rk].T
                Br = B_ @ Qr @ Qr.T
                fam = lambda A, Br=Br, xm=xm, rm=rm: A + rm + (A - xm) @ Br
                fim = lowrank_train(c, 'h3', Ah3['h3'][ix], img_all[ix], rk, Qh3, seed=draw)
                fimz = lowrank_train(c, 'z', Ah3['z'][ix], img_all[ix], rk, Qz, seed=draw)
                for name, stage, fn in (('AM_h3', 'h3', fam), ('imageloss_h3', 'h3', fim), ('imageloss_z', 'z', fimz)):
                    Qs = Qh3 if stage == 'h3' else Qz
                    with torch.no_grad():
                        res = {kq: float(np.mean([score(I.forward_from(f0, stage, fn(a)), c.G)[0] for a in lst]))
                               for kq, lst in Qs.items() if kq in ('bench', 's4R1', 's4R4', 's4R20')}
                    rows.append(dict(exp='X1L', k=k, draw=draw, rank=rk, method=name, **res))
            # team baseline: channel map (CLIP loss) + rank-64 output edit (CLIP loss)
            Xraw = c.cal_input(1)[ix.cpu().numpy()]
            W = I.fit_map(f0, Xraw, c.cal[ix.cpu().numpy()], c.Gtr, torch.eye(c.C, device=dev()), 0.1, 3e-3, 300, draw)
            with torch.no_grad():
                zc = f0(torch.einsum('cd,ndt->nct', W, torch.as_tensor(Xraw, device=dev()).float()))
            Upc = topk(c.Zsrc_held.reshape(-1, 512), 64)
            cb = torch.zeros(512, device=dev()).requires_grad_(True)
            Ma = torch.zeros(64, 64, device=dev()).requires_grad_(True)
            opt = torch.optim.Adam([cb, Ma], lr=1e-3)
            mu = zc.mean(0)
            for _ in range(300):
                bi = torch.randint(k, (min(k, 256),), device=dev())
                z2 = zc[bi] + cb + ((zc[bi] - mu) @ Upc) @ Ma @ Upc.T
                gal = torch.cat([c.Gtr[img_all[ix][bi]], c.Gtr[torch.randint(len(c.Gtr), (1024,), device=dev())]])
                loss = F.cross_entropy(f0.scale() * z2 @ gal.T, torch.arange(len(bi), device=dev())) + 1e-4 * (Ma.pow(2).sum() + cb.pow(2).sum())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            res = {}
            with torch.no_grad():
                for kq, lst in c.Q.items():
                    if kq not in ('bench', 's4R1', 's4R4', 's4R20'):
                        continue
                    vals = []
                    for q in lst:
                        z = f0(torch.einsum('cd,ndt->nct', W, torch.as_tensor(q, device=dev()).float()))
                        vals.append(score(z + cb + ((z - mu) @ Upc) @ Ma @ Upc.T, c.G)[0])
                    res[kq] = float(np.mean(vals))
            rows.append(dict(exp='X1T', k=k, draw=draw, rank=64, method='channelmap+outedit', **res))
        print(f'     X1L/X1T k={k} done', flush=True)
    return rows


def x1_labelfree(c):
    rows = []
    f0 = c.f0
    keys = ('bench', 's4R1', 's4R4', 's4R20')
    base = {k: c.Q[k] for k in keys}

    def run(name, fn_q, U=0, draw=0, transductive=False):
        res = {}
        with torch.no_grad():
            for k in keys:
                res[k] = float(np.mean([score(fn_q(q), c.G)[0] for q in base[k]]))
        rows.append(dict(exp='X1F', method=name, U=U, draw=draw, transductive=transductive, **res))
    run('f0', lambda q: X.acts2(f0, q)['z'])
    for draw in range(3):
        for U in (25, 100, 400):
            d1 = c.lf_delta(U, draw)
            run('LF_h1', lambda q, d1=d1: X.acts2(f0, q, d1)['z'], U, draw)
            sel = np.random.RandomState(600 + draw).choice(len(c.unl), U, replace=False)
            Zu = I.acts(f0, c.unl[sel])['z']
            mz, sz = Zu.mean(0), Zu.std(0) + 1e-6
            run('out_mean', lambda q, mz=mz: X.acts2(f0, q)['z'] - mz + c.mS['z'], U, draw)
            run('out_meanscale', lambda q, mz=mz, sz=sz: (X.acts2(f0, q)['z'] - mz) / sz * c.sS['z'] + c.mS['z'], U, draw)
            xb = torch.as_tensor(c.unl[sel], device=dev())
            xs = torch.as_tensor(c.src_raw, device=dev())

            def ccov(x):
                x = x - x.mean(2, keepdim=True)
                C = torch.einsum('nct,ndt->cd', x, x) / (x.shape[0] * x.shape[2])
                return C + 0.1 * torch.trace(C) / len(C) * torch.eye(len(C), device=dev())
            Tm = I.msqrt(ccov(xs), 0.5) @ I.msqrt(ccov(xb), -0.5)
            run('EA_sensor', lambda q, Tm=Tm: f0(torch.einsum('cd,ndt->nct', Tm, torch.as_tensor(q, device=dev()).float())), U, draw)
    # transductive: use the unlabeled query batch itself
    run('LF_h1_batch', lambda q: (lambda A: I.forward_from(f0, 'h1', A['h1'] - A['h1'].mean(0) + c.mS['h1']))(X.acts2(f0, q)),
        transductive=True)

    def whiten(q):
        z = X.acts2(f0, q)['z']
        zc = z - z.mean(0)
        C = zc.T @ zc / len(zc)
        return zc @ I.msqrt(C + 0.1 * torch.trace(C) / len(C) * torch.eye(len(C), device=dev()), -0.5)
    run('batch_whiten', whiten, transductive=True)

    def csls(q, kk=10):
        z = F.normalize(X.acts2(f0, q)['z'], dim=-1)
        S = z @ c.G.T
        rq = S.topk(kk, dim=1).values.mean(1, keepdim=True)
        rg = S.topk(kk, dim=0).values.mean(0, keepdim=True)
        return 2 * S - rq - rg

    def run_scores(name, Sfn):
        res = {}
        with torch.no_grad():
            for k in keys:
                vals = []
                for q in base[k]:
                    S = Sfn(q)
                    vals.append(100 * float((S.argmax(1) == torch.arange(len(S), device=dev())).float().mean()))
                res[k] = float(np.mean(vals))
        rows.append(dict(exp='X1F', method=name, U=0, draw=0, transductive=True, **res))
    run_scores('CSLS', csls)

    def score_lite(q, iters=5, lam=0.3):
        z = X.acts2(f0, q)['z']
        Rm = torch.eye(z.shape[1], device=dev())
        g = c.G
        for _ in range(iters):
            zz = F.normalize((z - z.mean(0)) @ Rm, dim=-1)
            S = zz @ g.T
            a, b = S.argmax(1), S.argmax(0)
            mutual = torch.nonzero(b[a] == torch.arange(len(a), device=dev())).flatten()
            if len(mutual) < 5:
                break
            Xc = F.normalize(z[mutual] - z.mean(0), dim=-1)
            Yc = g[a[mutual]]
            M = Xc.T @ Yc
            U_, _, Vh = safe_svd(M + lam * len(mutual) / z.shape[1] * torch.eye(z.shape[1], device=dev()))
            Rm = U_ @ Vh
        return F.normalize((z - z.mean(0)) @ Rm, dim=-1) @ g.T
    run_scores('SCORE_lite', score_lite)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# X2 bases: transferable structure
# ---------------------------------------------------------------------------------------------------------------
def x2(c, d1_all):
    rows = []
    f0 = c.f0
    S = c.Zsrc_held                                                     # (9, 2000, 512)
    mis = torch.cat([S[i] - torch.cat([S[:i], S[i + 1:]]).mean(0) for i in range(len(S))])
    mis_c = torch.cat([m - m.mean(0) for m in mis.split(S.shape[1])])
    src_act = S.reshape(-1, 512)
    Cs = (src_act - src_act.mean(0)).T @ (src_act - src_act.mean(0)) / len(src_act)
    g = torch.Generator(device='cpu').manual_seed(0)
    bases = {}
    for r in BASIS_RANKS:
        bases[('mis_centered', r)] = topk(mis_c, r, center=False)
        bases[('mis_uncentered', r)] = topk(mis, r, center=False)
        bases[('src_pc', r)] = topk(src_act, r)
        bases[('gallery_pc', r)] = topk(c.Gtr[c.pool], r)
        Lc = torch.linalg.cholesky(Cs + 1e-6 * torch.eye(512, device=dev()))
        bases[('var_matched_rand', r)] = ortho(Lc @ torch.randn(512, r, generator=g).to(dev()))
        bases[('iso_rand', r)] = ortho(torch.randn(512, r, generator=g).to(dev()))
    # variance captured + overlap with the target's own difference basis on disjoint halves (diagnostic, uses B)
    ZB_held = I.acts(f0, c.held_B)['z']
    D_B = ZB_held - S.mean(0)
    half = len(D_B) // 2
    V_B = topk(D_B[:half], 16, center=False)
    Smis_b = torch.cat([m[half:] for m in mis.split(S.shape[1])])
    for (name, r), U in bases.items():
        if r != 16:
            continue
        Ub = topk(Smis_b, 16, center=False) if name == 'mis_uncentered' else U
        rows.append(dict(exp='X2geom', basis=name, rank=r,
                         var_src=float(((src_act - src_act.mean(0)) @ U).pow(2).sum() / (src_act - src_act.mean(0)).pow(2).sum()),
                         overlap_targetdiff=p_angles(V_B, Ub), chance=16 / 512))
    # deployable: fit offset + U A U^T on k target images (track B, r = 1, after LF)
    Zc = X.acts2(f0, c.cal_input(1), d1_all)['z']
    Qz = c.qacts(d1_all)
    Qz = {k: v for k, v in Qz.items() if k in ('bench', 's4R1', 's4R4', 's4R20')}
    for k in (5, 10, 25, 50):
        for draw in range(20):
            ix = torch.as_tensor(np.random.RandomState(700 + draw).permutation(N_CAL)[:k], device=dev())
            Xs, Ys = Zc[ix], c.anchor['z'][ix]
            rows.append(dict(exp='X2dep', basis='offset_only', rank=0, diag=False, k=k, draw=draw,
                             **eval_fn(c, m_trans(Xs, Ys, 0), Qz, c.G)))
            for (name, r), U in bases.items():
                for diag in (False, True):
                    rows.append(dict(exp='X2dep', basis=name, rank=r, diag=diag, k=k, draw=draw,
                                     **eval_fn(c, m_sub(Xs, Ys, U, diag), Qz, c.G)))
        print(f'     X2 deployable k={k} done', flush=True)
    # oracle panel (session 3-4 target query; donors with matched repetitions)
    zq = I.acts(f0, c.s34)['z']
    zrep = I.acts(f0, c.s12)['z']
    donors_single = c.Zsrc_s34                                         # (9, 200, 512), 40 reps each
    snr = []
    for s_i, s in enumerate(c.src):
        tts = N.subject_data(s, c.mon, keys=('test_trials',))['test_trials']
        sel = np.random.RandomState(40 + s).choice(80, 4, replace=False)
        snr.append(I.acts(f0, tts[:, sel].astype(np.float32).mean(1))['z'])
    snr = torch.stack(snr).mean(0)                                     # ~36 total presentations
    src_mean = donors_single.mean(0)
    perm = torch.as_tensor(np.random.RandomState(5).permutation(200), device=dev())
    try:
        from eegalign import categories as Ca
        cat, _ = Ca.labels(c.test_paths, min_concepts=1)
        same_cat = []
        rng = np.random.RandomState(6)
        for i in range(200):
            cand = np.nonzero((cat == cat[i]) & (np.arange(200) != i))[0] if cat[i] >= 0 else []
            same_cat.append(rng.choice(cand) if len(cand) else rng.choice(np.delete(np.arange(200), i)))
        same_cat = torch.as_tensor(np.array(same_cat), device=dev())
    except Exception:                                                  # noqa: BLE001
        same_cat = None
    base0 = score(zq, c.G)
    rows.append(dict(exp='X2oracle', basis='-', rank=0, donor='none', top1=base0[0], mrr=base0[1]))
    for (name, r), U in bases.items():
        if r not in (4, 16):
            continue
        P = U @ U.T
        dons = {'single_same_image': None, 'repeat_matched_mean': snr, 'wrong_image_mean': src_mean[perm],
                'target_repeat': zrep}
        if same_cat is not None:
            dons['same_category_wrong_image'] = src_mean[same_cat]
        with torch.no_grad():
            for dn, dz in dons.items():
                if dn == 'single_same_image':
                    v = [score(zq + (donors_single[i] - zq) @ P, c.G) for i in range(len(c.src))]
                    t1, mr = float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))
                else:
                    t1, mr = score(zq + (dz - zq) @ P, c.G)
                rows.append(dict(exp='X2oracle', basis=name, rank=r, donor=dn, top1=t1, mrr=mr))
            t1, mr = score(zq - (zq - zq.mean(0)) @ P, c.G)
            rows.append(dict(exp='X2oracle', basis=name, rank=r, donor='remove_target_component', top1=t1, mrr=mr))
            t1, mr = score(zq + (src_mean.mean(0) - zq.mean(0)) @ P, c.G)
            rows.append(dict(exp='X2oracle', basis=name, rank=r, donor='offset_only', top1=t1, mrr=mr))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# X3 acquisition budget and signal; X4 anchor panels
# ---------------------------------------------------------------------------------------------------------------
def raw_ident(Xa, Xb):
    """Model-free identification: correlate image patterns (grand evoked response removed) across two halves."""
    a = Xa.reshape(len(Xa), -1) - Xa.reshape(len(Xa), -1).mean(0)
    b = Xb.reshape(len(Xb), -1) - Xb.reshape(len(Xb), -1).mean(0)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-9)
    R = a @ b.T / a.shape[1]
    return 100 * float((R.argmax(1) == np.arange(len(R))).mean()), float(np.diag(R).mean() - R.mean())


def x3(c, d1_all):
    rows = []
    f0 = c.f0
    keys = QKEYS
    Qz0, Qz1 = c.qacts(None), c.qacts(d1_all)
    with torch.no_grad():
        base_w = {k: float(np.mean([score(c.w(torch.as_tensor(q, device=dev())), c.Gw)[0] for q in c.Q[k]])) for k in keys}
        base_f = {k: float(np.mean([score(z, c.G)[0] for z in Qz0[k]])) for k in keys}
        base_lf = {k: float(np.mean([score(z, c.G)[0] for z in Qz1[k]])) for k in keys}
    rows.append(dict(exp='X3', design='-', cost=0, images=0, reps=0, draw=0, method='within', **base_w))
    rows.append(dict(exp='X3', design='-', cost=0, images=0, reps=0, draw=0, method='f0', **base_f))
    rows.append(dict(exp='X3', design='-', cost=0, images=0, reps=0, draw=0, method='LF_offset', **base_lf))
    # fresh (transductive) offset on the query session
    fresh = {}
    for k in keys:
        fresh[k] = [(lambda A: I.forward_from(f0, 'h1', A['h1'] - A['h1'].mean(0) + c.mS['h1']))(X.acts2(f0, q)) for q in c.Q[k]]
    with torch.no_grad():
        rows.append(dict(exp='X3', design='-', cost=0, images=0, reps=0, draw=0, method='fresh_offset(transductive)',
                         **{k: float(np.mean([score(z, c.G)[0] for z in fresh[k]])) for k in keys}))
    Zr = {r: X.acts2(f0, c.cal_input(r), d1_all)['z'] for r in (1, 2, 4)}
    for cost, designs in ((40, ((40, 1), (20, 2), (10, 4))), (20, ((20, 1), (10, 2), (5, 4)))):
        for (k, r) in designs:
            for draw in range(10):
                ix = torch.as_tensor(np.random.RandomState(1700 + draw).permutation(N_CAL)[:k], device=dev())
                Xs, Ys = Zr[r][ix], c.anchor['z'][ix]
                fns = {'trans_k5': m_trans(Xs, Ys, 5), 'rigid_l1': m_rigid(Xs, Ys, 1.0)}
                B_, xm, rm = X.ridge_svd(Xs, Ys - Xs)
                Qall = safe_svd((Xs - xm) @ B_)[2]
                for rk in (2, 4):
                    Qr = Qall[:rk].T
                    fns[f'rr{rk}'] = (lambda Br, xm=xm, rm=rm: (lambda Z: Z + rm + (Z - xm) @ Br))(B_ @ Qr @ Qr.T)
                for name, fn in fns.items():
                    rows.append(dict(exp='X3', design=f'{k}x{r}', cost=cost, images=k, reps=r, draw=draw, method=name,
                                     **{kk: v for kk, v in eval_fn(c, fn, Qz1, c.G).items() if not kk.startswith('mrr')}))
                # old map + fresh offset (map fitted on LF-shifted support, applied to fresh-centered queries)
                with torch.no_grad():
                    rows.append(dict(exp='X3', design=f'{k}x{r}', cost=cost, images=k, reps=r, draw=draw,
                                     method='rr2+fresh_offset(transductive)',
                                     **{kk: float(np.mean([score(fns['rr2'](z), c.G)[0] for z in fresh[kk]])) for kk in keys}))
    # signal diagnostics (model-free, raw EEG)
    acc_w, gap_w = raw_ident(c.s12, c.s34)
    acc_x, gap_x = raw_ident(c.raw_test_src, c.s34)
    rows.append(dict(exp='X3signal', within_person_raw_ident=acc_w, within_person_pattern_gap=gap_w,
                     from_source_anchors_raw_ident=acc_x, from_source_pattern_gap=gap_x))
    return rows


def x4(c, d1_all):
    rows = []
    S = c.Zsrc_cal                                                     # (9, 2000, 512)
    Sn = F.normalize(S - S.mean(1, keepdim=True), dim=-1)
    rel = torch.stack([(Sn[i] * torch.cat([Sn[:i], Sn[i + 1:]]).mean(0)).sum(1) for i in range(len(S))]).mean(0)

    def kcenter(cands, k):
        Y = F.normalize(c.anchor['z'][cands] - c.anchor['z'].mean(0), dim=-1)
        chosen = [int(torch.argmax(rel[cands]))]
        dmin = 1 - Y @ Y[chosen[0]]
        for _ in range(k - 1):
            j = int(torch.argmax(dmin))
            chosen.append(j)
            dmin = torch.minimum(dmin, 1 - Y @ Y[j])
        return cands[torch.as_tensor(chosen, device=dev())]
    allc = torch.arange(N_CAL, device=dev())
    top_rel = torch.argsort(rel, descending=True)
    Zc = X.acts2(c.f0, c.cal_input(1), d1_all)['z']
    Qz = {k: v for k, v in c.qacts(d1_all).items() if k in ('bench', 's4R1', 's4R4')}
    for k in (5, 10, 25):
        panels = {'reliability': top_rel[:k], 'diversity': kcenter(allc, k), 'reliability+diversity': kcenter(top_rel[:N_CAL // 2], k)}
        for draw in range(20):
            panels[f'random{draw}'] = torch.as_tensor(np.random.RandomState(2700 + draw).permutation(N_CAL)[:k], device=dev())
        for pname, ix in panels.items():
            Xs, Ys = Zc[ix], c.anchor['z'][ix]
            for mname, fn in (('trans_k5', m_trans(Xs, Ys, 5)), ('rr2', X.am_family(Xs, Ys)['rr2'] if k > 2 else m_trans(Xs, Ys, 5))):
                rows.append(dict(exp='X4', k=k, panel='random' if pname.startswith('random') else pname,
                                 draw=int(pname[6:]) if pname.startswith('random') else 0, map=mname,
                                 **{kk: v for kk, v in eval_fn(c, fn, Qz, c.G).items() if not kk.startswith('mrr')}))
    return rows


def run_target(B, mon, which):
    t0 = time.time()
    c = R3(B, mon, which)
    json.dump(manifest(c), open(os.path.join(odir(), f'manifest_t{B:02d}_{mon}.json'), 'w'), indent=1)
    out = {}
    r1, d1_all = x1(c)
    out['x1'] = r1
    print(f'   X1 done ({time.time() - t0:.0f}s)', flush=True)
    for name, fn in (('x1f', lambda: x1_labelfree(c)), ('x2', lambda: x2(c, d1_all)), ('x3', lambda: x3(c, d1_all)),
                     ('x4', lambda: x4(c, d1_all)), ('x1l', lambda: x1_layer_and_team(c, d1_all))):
        out[name] = fn()
        print(f'   {name} done ({time.time() - t0:.0f}s)', flush=True)
    for name, rows in out.items():
        df = pd.DataFrame(rows)
        df['target'], df['montage'] = B, mon
        df.to_csv(os.path.join(odir(), f'{name}_t{B:02d}_{mon}.csv'), index=False)
    print(f'  target {B} {mon}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def load(exp, mon):
    f = sorted(glob.glob(os.path.join(odir(), f'{exp}_t*_{mon}.csv')))
    return pd.concat([pd.read_csv(x) for x in f], ignore_index=True) if f else None


def person_stats(a, b, label):
    """a, b: Series indexed by target (people 2-10). Paired difference statistics."""
    from scipy import stats
    idx = sorted(set(a.index) & set(b.index) - {1})
    d = (a.reindex(idx) - b.reindex(idx)).values
    rng = np.random.RandomState(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    pos = int((d > 0).sum())
    sign_p = float(stats.binomtest(pos, int((d != 0).sum()), 0.5).pvalue) if (d != 0).any() else 1.0
    try:
        wil = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        wil = np.nan
    lopo = [np.delete(d, i).mean() for i in range(len(d))]
    print(f'  {label}: mean {d.mean():+.2f} (95% CI {np.percentile(boot, 2.5):+.2f} to {np.percentile(boot, 97.5):+.2f}), '
          f'median {np.median(d):+.2f}, {pos}/{len(d)} positive, sign p={sign_p:.4f}, Wilcoxon p={wil:.4f}, '
          f'leave-one-person-out {min(lopo):+.2f} to {max(lopo):+.2f}')
    print('     per person: ' + ', '.join(f't{t}:{v:+.1f}' for t, v in zip(idx, d)))
    return wil if not np.isnan(wil) else sign_p


def select_on_t1(x1d):
    """Subject-1 selection for the primary contrasts (track B, k = 25, neural target, image readout, query s4R4)."""
    g = x1d[(x1d.person == 1) & (x1d.track == 'B') & (x1d.k == 25) & (x1d.cal == 'neural') & (x1d.readout == 'img')]
    g = g.groupby(['map', 'lf']).s4R4.mean()
    am = g[[i for i in g.index if i[0].startswith('rr') and 'shuffled' not in i[0]]].idxmax()
    simple = g[[i for i in g.index if i[0].startswith(('trans_k0', 'trans_k5', 'trans_k25', 'rigid'))]].idxmax()
    return am, simple


def report(mons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pd.set_option('display.width', 230)
    primary_p = {}
    for mon in mons:
        f = sorted(glob.glob(os.path.join(odir(), f'x1_t*_{mon}.csv')))
        if not f:
            continue
        x1d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        x1d['person'] = x1d['target']
        people = sorted(x1d.person.unique())
        print(f'\n{"=" * 110}\n{mon}: people {people}\n{"=" * 110}')
        base = x1d[(x1d['map'] == 'none') & (x1d.lf == 0) & (x1d.readout == 'img')].set_index('person')
        print('frozen f0 (top-1 by query):', base[QKEYS].drop(index=1, errors='ignore').mean().round(1).to_dict())
        sub = x1d[(x1d.track == 'B') & (x1d.readout == 'img') & (x1d.person != 1)]
        per = sub.groupby(['cal', 'lf', 'map', 'k', 'person']).s4R4.mean()
        tab = per.groupby(['cal', 'lf', 'map', 'k']).mean().unstack('k')
        print('\nX1 (Track B: k images x 1 presentation, earlier sessions; query = session 4, 4 presentations), top-1:')
        print(tab.round(1).to_string())
        ro = x1d[(x1d.track == 'B') & (x1d.cal == 'neural') & (x1d.k.isin([25, 200])) & (x1d.person != 1)]
        print('\n  readout: image gallery vs predicted population anchors (neural target, query s4R4):')
        print(ro.groupby(['readout', 'lf', 'map', 'k']).s4R4.mean().unstack('k').unstack('readout').round(1).to_string())
        bt = x1d[(x1d.track == 'bench4') & (x1d.readout == 'img') & (x1d.person != 1)]
        print('\n  benchmark track (4-presentation calibration, 80-rep queries), top-1:')
        print(bt.groupby(['cal', 'lf', 'map', 'k']).bench.mean().unstack('k').round(1).to_string())
        if 1 in people:
            am, simple = select_on_t1(x1d)
            print(f'\nSELECTED ON SUBJECT 1 (k = 25, query s4R4): AM = {am[0]} (LF {am[1]}), simple = {simple[0]} (LF {simple[1]})')
            sel = x1d[(x1d.track == 'B') & (x1d.k == 25) & (x1d.cal == 'neural') & (x1d.readout == 'img')]
            a = sel[(sel['map'] == am[0]) & (sel.lf == am[1])].groupby('person').s4R4.mean()
            b = sel[(sel['map'] == simple[0]) & (sel.lf == simple[1])].groupby('person').s4R4.mean()
            f0p = base.s4R4
            print('PRIMARY P1' if mon == 'posterior17' else 'P1-equivalent', f'({mon}):')
            p = person_stats(a, b, f'{am[0]} minus {simple[0]}')
            person_stats(a, f0p, f'{am[0]} minus frozen f0')
            person_stats(b, f0p, f'{simple[0]} minus frozen f0')
            if mon == 'posterior17':
                primary_p['P1'] = p
            if mon == 'crown8':
                best_simple = max([(sel[(sel['map'] == m) & (sel.lf == lf)].groupby('person').s4R4.mean(), m, lf)
                                   for m in ('trans_k0', 'trans_k5', 'trans_k25') for lf in (0, 1)],
                                  key=lambda t: t[0].get(1, -1))
                print('PRIMARY P3 (Crown-8): selected AM vs strongest simple baseline (subject-1 choice):')
                primary_p['P3'] = person_stats(a, best_simple[0], f'{am[0]} minus {best_simple[1]} (LF {best_simple[2]})')
            sh = sel[(sel['map'] == 'rr2_shuffled') & (sel.lf == am[1])].groupby('person').s4R4.mean()
            ot = sel[(sel['map'] == 'trans_k5_otherimgs') & (sel.lf == am[1])].groupby('person').s4R4.mean()
            person_stats(a, sh, 'AM minus AM with shuffled anchors (pairing control)')
            person_stats(b, ot, 'simple minus translation from other images (pairing control)')
        # figure: per-person lines at k = 25 for f0 / shrunk translation / rigid / rank-2 AM, queries R1 and R20
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        sel = x1d[(x1d.track == 'B') & (x1d.k == 25) & (x1d.cal == 'neural') & (x1d.readout == 'img') & (x1d.lf == 1)]
        meths = [('f0', None), ('trans_k5', 'trans_k5'), ('rigid_l1.0', 'rigid_l1.0'), ('rr2', 'rr2')]
        for i, q in enumerate(('s4R1', 's4R20')):
            vals = []
            for lab, m in meths:
                v = (base[q] if m is None else sel[sel['map'] == m].groupby('person')[q].mean()).drop(index=1, errors='ignore')
                vals.append(v)
            M = pd.concat(vals, axis=1)
            M.columns = [l for l, _ in meths]
            for _, row in M.iterrows():
                ax[i].plot(range(4), row.values, '-o', color='0.6', ms=3, lw=.8)
            ax[i].plot(range(4), M.mean().values, '-o', color='C3', lw=2)
            ax[i].set(xticks=range(4), xticklabels=M.columns, title=f'{mon}, k=25 x 1, session-4 query R={q[3:]}',
                      ylabel='top-1 %')
        plt.tight_layout()
        plt.savefig(os.path.join(odir(), f'R3_primary_{mon}.png'), dpi=130)
        plt.close()
        # X1F label-free
        lf = load('x1f', mon)
        if lf is not None:
            lf = lf[lf.target != 1]
            print('\nX1F LABEL-FREE COMPETITORS (top-1):')
            print(lf.groupby(['transductive', 'method', 'U'])[['bench', 's4R1', 's4R4', 's4R20']].mean().round(1).to_string())
        # X1L / X1T
        ll = load('x1l', mon)
        if ll is not None:
            ll = ll[ll.target != 1]
            print('\nX1L LAYER TEST + X1T TEAM BASELINE (track B, after LF for X1L):')
            print(ll.groupby(['method', 'rank', 'k'])[['bench', 's4R1', 's4R4', 's4R20']].mean().round(1).to_string())
        # X2
        x2d = load('x2', mon)
        if x2d is not None:
            x2d = x2d[x2d.target != 1]
            print('\nX2 GEOMETRY (rank 16): variance of source activations captured, overlap with the target\'s own '
                  'difference basis (principal-angle cos^2, disjoint halves):')
            print(x2d[x2d.exp == 'X2geom'].groupby('basis')[['var_src', 'overlap_targetdiff', 'chance']].mean().round(3).to_string())
            dep = x2d[x2d.exp == 'X2dep']
            print('\nX2 DEPLOYABLE (offset + U A U^T fitted on k target images; query s4R4):')
            print(dep[dep.diag == False].groupby(['basis', 'rank', 'k']).s4R4.mean().unstack('k').round(1).to_string())  # noqa: E712
            print('   diagonal A:')
            print(dep[dep.diag == True].groupby(['basis', 'rank', 'k']).s4R4.mean().unstack('k').round(1).to_string())  # noqa: E712
            if 1 in people:
                rstar = int(''.join(ch for ch in select_on_t1(x1d)[0][0] if ch.isdigit()) or 2)
                rb = min(BASIS_RANKS, key=lambda r: abs(r - rstar))
                dd = load('x2', mon)
                dd = dd[(dd.exp == 'X2dep') & (dd.k == 25) & (dd.diag == False) & (dd['rank'] == rb)]  # noqa: E712
                a = dd[dd.basis == 'mis_centered'].groupby('target').s4R4.mean()
                b = dd[dd.basis == 'src_pc'].groupby('target').s4R4.mean()
                if mon == 'posterior17':
                    print(f'PRIMARY P2 (rank {rb}, k = 25): mismatch basis vs source-activation PCs')
                    primary_p['P2'] = person_stats(a, b, 'mis_centered minus src_pc')
            orc = x2d[x2d.exp == 'X2oracle']
            print('\nX2 ORACLE PANEL (target = sessions 3-4; top-1):')
            print(orc.groupby(['basis', 'rank', 'donor']).top1.mean().unstack('donor').round(1).to_string())
        # X3
        x3d = load('x3', mon)
        if x3d is not None:
            x3d = x3d[x3d.target != 1]
            print('\nX3 ACQUISITION (query = session 4 with R presentations; bench = 80):')
            print(x3d[x3d.exp == 'X3'].groupby(['cost', 'design', 'method'])[QKEYS].mean().round(1).to_string())
            print('\n  signal (model-free raw-EEG image identification, %; chance 0.5):')
            print(x3d[x3d.exp == 'X3signal'][['within_person_raw_ident', 'from_source_anchors_raw_ident',
                                              'within_person_pattern_gap', 'from_source_pattern_gap']].mean().round(3).to_string())
        x4d = load('x4', mon)
        if x4d is not None:
            x4d = x4d[x4d.target != 1]
            print('\nX4 ANCHOR PANELS (track B; query s4R4):')
            print(x4d.groupby(['map', 'k', 'panel']).s4R4.mean().unstack('panel').round(1).to_string())
    if primary_p:
        from statsmodels.stats.multitest import multipletests
        keys = sorted(primary_p)
        adj = multipletests([primary_p[k] for k in keys], method='holm')[1]
        print('\nHOLM-ADJUSTED PRIMARY CONTRASTS:', {k: round(float(a), 4) for k, a in zip(keys, adj)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['prep', 'run', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--subjects', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--montages', nargs='+', default=['posterior17', 'occipital8', 'crown8'])
    p.add_argument('--which', default='last')
    p.add_argument('--reps_dir', default='/content/nb_reps')
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'prep':
        return prep(a.subjects)
    if a.cmd == 'report':
        return report(a.montages)
    for mon in a.montages:
        for B in a.targets:
            if os.path.exists(os.path.join(odir(), f'x1l_t{B:02d}_{mon}.csv')):
                print(f'[run] {B} {mon}: done')
                continue
            print(f'[run] target {B} {mon}', flush=True)
            run_target(B, mon, a.which)


if __name__ == '__main__':
    main()
