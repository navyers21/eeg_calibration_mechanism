"""Round 4: how much later-day accuracy before the user gives any EEG, and what does a warm-up add?

    python scripts/nb_round4.py sessions  --config /content/nb_config.json      # session audit (UBP files, once)
    python scripts/nb_round4.py run       --config /content/nb_config.json --montages posterior17 occipital8 crown8
    python scripts/nb_round4.py placement --config /content/nb_config.json      # Crown placement, model-free
    python scripts/nb_round4.py report    --config /content/nb_config.json

Built on round 3 (scripts/nb_round3.py; context class R3 reused, nothing re-tuned on outcomes). Round 3 already has the
gallery x offset 2 x 2 (posterior-17, s4R4: image 5.1 / anchors 7.0 / image+LF 5.4 / anchors+LF 7.2), so round 4 asks
what the anchor gain is made of, and gives the warm-up one fair test.

Sessions. `sessions` downloads each person's UBP preprocessed files (Haitao999/things-eeg; they carry a session id per
(image, repetition)), matches every NeuroBridge single trial to its UBP trial (17 x 17 channel map, then Hungarian
matching on trial correlations), and writes a session table in NeuroBridge order to <cache_dir>/sessions. This is the
chronology audit the reviews asked for: it checks "test reps 60-79 = session 4" and finds which training trials were
recorded in session 4. Without the table, `run` falls back to round 3's assumptions and skips G-fresh and D.

Locked settings. Decoder, anchor predictor (ridge, GCV), U = 100 offset and query draws are round 3's. New knobs (CSLS
k = 10, inverted-softmax beta, kernel / MLP predictors) are chosen on SOURCE people only. People 2-10 are the test set
(1 kept for continuity); this is a locked reanalysis of people who informed rounds 1-3, not an independent cohort.

Experiments (per target x montage):
  G  gallery x scoring rule x stress, zero target labels.
     galleries: img (f0 image gallery) | img_white (source-pool ZCA) | img_spec (image gallery with its eigen-spectrum
     matched to the anchor predictor's: geometry without neural pairing) | anchor (round-3 ridge, LOCKED) |
     anchor_permall / anchor_permconcept (ridge on shuffled image-neural pairs, across all images / within concept) |
     anchor_krr, anchor_mlp, anchor_sel (best of ridge/krr/mlp by held-out-concept cosine on source data) |
     oracle (measured source outputs on the test images: labeled oracle, not deployable).
     rules: cos | csls_src (candidate density from a fixed bank of source query outputs: inductive) | csls_unl
     (bank = target's earlier-session unlabeled trials: causal) | csls_batch (bank = the query batch: transductive) |
     isoft_src (inverted softmax over the source bank, beta chosen on source people).
     stress: balanced 200 | +1000 distractor candidates (held-out concepts) | 100 of 200 images queried |
     300 Zipf-distributed repeated queries.  offsets: none, old_100.  metrics: top-1, MRR, share of predictions going
     to the 10 most-predicted candidates, normalized prediction entropy.
  O  warm-up. Gallery frozen (img / anchor / anchor_sel; cos and csls_src). Offsets from U = 25-400 single trials:
     old (sessions 1-3) vs fresh (session 4 training stimuli, disjoint from the test images), MATCHED on image identity
     (same images: one earlier-session presentation and one session-4 presentation), blend (both, budget 2U), old with
     2U (budget match for blend), any (random sessions: permuted-session control), round-3 unmatched old, round-3
     transductive batch offset. 10 draws. Ocross: reciprocal 100 / 100 test-image cross-fit (warm-up from session-4
     trials of 100 images, queries = the other 100, full 200-image gallery); labeled as a cross-fit, not a warm-up.
  D  drift. Mean h1 of 200 single trials per session, disjoint halves: within-session vs across-session distance
     (same trial count, image-disjoint halves in both), and session-4 retrieval with an offset from session 1, 2, 3, 4.
  placement (all people, raw test EEG, no decoder): Crown-8, occipital-8, posterior-17, Crown with 1/2/4/6 sites
     swapped for posterior sites, source-selected best 8; template = sessions 1-2 mean, query = session-4 R = 4 / 20
     or sessions 3-4 mean; label-permutation null; evoked SNR per montage. Channel rankings use source people only.

Primary contrasts (posterior-17, query s4R4, people 2-10, Wilcoxon, Holm over the two):
  Q1 gallery:    anchor / cos  minus  img / csls_src     (does the neural anchor gallery beat the best inductive
                 hubness correction of the image gallery, both with zero target EEG?)
  Q2 adaptation: anchor / cos, fresh_100 minus old_100, matched images (does a same-day warm-up beat an equal earlier one?)
Secondary, prespecified: anchor - img (cos); anchor - img_spec; anchor - anchor_permconcept; old_100 - none with anchors
(equivalence margin +-1 point, 90% CI); occipital-8 replication of Q1 and Q2; Crown replacement curve.
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
import nb_round3 as R  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
QK = ['s4R1', 's4R4', 's4R20', 'bench']
UGRID = [25, 50, 100, 200, 400]
MARGIN = 1.0
UBP_REPO = 'Haitao999/things-eeg'


def dev():
    return N.ENV['device']


def odir():
    d = os.path.join(N.ENV['results_dir'], 'nb_round4')
    os.makedirs(d, exist_ok=True)
    return d


def ses_path(s):
    return os.path.join(N.ENV['cache_dir'], 'sessions', f'sub-{s:02d}.npz')


def nrm(x):
    return F.normalize(x.float(), dim=-1)


# ---------------------------------------------------------------------------------------------------------------
# sessions: chronology audit + session table in NeuroBridge order
# ---------------------------------------------------------------------------------------------------------------
def _tload(p):
    try:
        return torch.load(p, weights_only=False, mmap=True, map_location='cpu')
    except Exception:                                                  # noqa: BLE001
        return torch.load(p, weights_only=False, map_location='cpu')


def _pkey(p):
    return '/'.join(str(p).replace('\\', '/').split('/')[-2:])


def _z(a):
    a = a.reshape(*a.shape[:-2], -1).astype(np.float32)
    a = a - a.mean(-1, keepdims=True)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def _match(nb, ub, W):
    """nb (n, R, C, T), ub (n, R2, C, T): Hungarian match of every NB trial to a UBP trial of the same image.
    Returns perm (n, R) = UBP rep for each NB rep, matched corr (n, R), best unmatched corr (n, R)."""
    from scipy.optimize import linear_sum_assignment
    a, b = _z(nb), _z(np.einsum('cd,nrdt->nrct', W, ub))
    Cm = np.einsum('nrk,nsk->nrs', a, b)
    perm = np.zeros(Cm.shape[:2], int)
    for i in range(len(Cm)):
        perm[i] = linear_sum_assignment(-Cm[i])[1]
    mc = np.take_along_axis(Cm, perm[..., None], 2)[..., 0]
    other = Cm.copy()
    np.put_along_axis(other, perm[..., None], -9, 2)
    return perm, mc, other.max(2)


def sessions(subjects, local='/content/ubp_raw'):
    from huggingface_hub import hf_hub_download, list_repo_files
    from eegalign import ubp as U
    import nb_check_order as C
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
    files = list_repo_files(UBP_REPO, repo_type='dataset')
    idx = [U.CHANNELS.index(c) for c in U.MONTAGES['posterior17']]
    summ = {}
    for s in subjects:
        if os.path.exists(ses_path(s)):
            print(f'  sub-{s:02d}: session table present')
            summ[s] = json.load(open(ses_path(s)[:-4] + '.json'))
            continue
        t0 = time.time()
        got = {}
        for key in ('Preprocessed_data_250Hz_whiten', 'Preprocessed_data_250Hz_no_whiten'):
            for mode in ('test', 'train'):
                hit = [f for f in files if f.endswith(f'{key}/sub-{s:02d}/{mode}.pt')]
                if not hit or mode in got:
                    continue
                d = _tload(hf_hub_download(UBP_REPO, hit[0], repo_type='dataset', local_dir=local))
                if 'session' not in d:
                    print(f'  {hit[0]}: no session field')
                    continue
                got[mode] = dict(eeg=np.asarray(d['eeg'][:, :, idx, :250], np.float32), ses=np.asarray(d['session']).astype(int),
                                 img=np.asarray(d['img']))
                del d
        assert set(got) == {'test', 'train'}, f'sub-{s:02d}: UBP files with session ids not found'
        if min(got['train']['ses'].min(), got['test']['ses'].min()) == 0:      # store sessions as 1-4
            for k in got:
                got[k]['ses'] = got[k]['ses'] + 1
        nbd = N.subject_data(s, 'posterior17', keys=['test_trials', 'train_paths', 'test_paths'])
        # image order
        ui = {_pkey(p): i for i, p in enumerate(got['train']['img'][:, 0])}
        tr_map = np.array([ui[_pkey(p)] for p in nbd['train_paths']])
        ut = {_pkey(p): i for i, p in enumerate(got['test']['img'][:, 0])}
        te_map = np.array([ut[_pkey(p)] for p in nbd['test_paths']])
        reps = R.load_reps(s, 'posterior17')                            # (16540, 4, 17, 250) NB order
        rng = np.random.RandomState(0)
        fit = rng.choice(len(tr_map), 1500, replace=False)
        W = C.fit_map(got['train']['eeg'][tr_map[fit]].mean(1), np.asarray(reps[fit], np.float32).mean(1))
        # test: 80 x 80 per image
        pt, mt, ot = _match(nbd['test_trials'].astype(np.float32), got['test']['eeg'][te_map], W)
        test_ses = np.take_along_axis(got['test']['ses'][te_map], pt, 1)
        # train: 4 x 4 per image, chunked
        train_ses = np.zeros((len(tr_map), 4), int)
        mtr, otr, ident = [], [], []
        for a in range(0, len(tr_map), 2000):
            sl = slice(a, a + 2000)
            p_, m_, o_ = _match(np.asarray(reps[sl], np.float32), got['train']['eeg'][tr_map[sl]], W)
            train_ses[sl] = np.take_along_axis(got['train']['ses'][tr_map[sl]], p_, 1)
            mtr.append(m_)
            otr.append(o_)
            ident.append((p_ == np.arange(4)).all(1))
        mtr, otr, ident = np.concatenate(mtr), np.concatenate(otr), np.concatenate(ident)
        assume = np.repeat(np.arange(4) + 1, 20)[None]
        info = dict(subject=s,
                    test_match_corr=float(np.median(mt)), test_best_other=float(np.median(ot)),
                    test_confident=float((mt - ot > 0.05).mean()),
                    test_reps_as_assumed=float((test_ses == assume).mean()),
                    test_session_counts={int(k): int(v) for k, v in zip(*np.unique(test_ses, return_counts=True))},
                    train_match_corr=float(np.median(mtr)), train_best_other=float(np.median(otr)),
                    train_confident=float((mtr - otr > 0.05).mean()), train_identity_perm=float(ident.mean()),
                    train_rep01_before_rep23=float((train_ses[:, :2].max(1) < train_ses[:, 2:].min(1)).mean()),
                    train_imgs_with_session4=int((train_ses == 4).any(1).sum()),
                    train_session_counts={int(k): int(v) for k, v in zip(*np.unique(train_ses, return_counts=True))})
        os.makedirs(os.path.dirname(ses_path(s)), exist_ok=True)
        np.savez(ses_path(s), train=train_ses, test=test_ses, train_conf=(mtr - otr), test_conf=(mt - ot))
        json.dump(info, open(ses_path(s)[:-4] + '.json', 'w'), indent=1)
        summ[s] = info
        del got, reps
        for f in glob.glob(os.path.join(local, '**', '*.pt'), recursive=True):
            os.remove(f)
        print(f'  sub-{s:02d}: {json.dumps(info)} ({time.time() - t0:.0f}s)', flush=True)
    json.dump(summ, open(os.path.join(odir(), 'session_audit.json'), 'w'), indent=1)


def load_ses(B):
    p = ses_path(B)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    tr, te = d['train'].astype(int), d['test'].astype(int)
    if min(tr.min(), te.min()) == 0:                                    # UBP session ids may be 0-based
        tr, te = tr + 1, te + 1
    return {'train': tr, 'test': te}


# ---------------------------------------------------------------------------------------------------------------
# anchor predictors (source data only)
# ---------------------------------------------------------------------------------------------------------------
def fit_ridge(Xtr, Ytr):
    Bm, xm, ym = X.ridge_svd(Xtr, Ytr)
    return lambda f: (f - xm) @ Bm + ym


def fit_krr(Xtr, Ytr, Xva, Yva, n_land=3000, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Xn = nrm(Xtr)
    L = Xn[torch.randperm(len(Xn), generator=g)[:n_land].to(Xn.device)]
    d2 = torch.cdist(Xn[:2000], Xn[:2000]).pow(2)
    gam = 1.0 / float(d2[d2 > 0].median())
    phi = lambda f: torch.exp(-gam * torch.cdist(nrm(f), L).pow(2))  # noqa: E731
    P, ym = phi(Xtr), Ytr.mean(0)
    pm = P.mean(0)
    Pc = P - pm
    G_, R_ = Pc.T @ Pc, Pc.T @ (Ytr - ym)
    best = (-9, None)
    for lam in torch.logspace(-3, 3, 13):
        Wt = torch.linalg.solve(G_ + lam * G_.trace() / len(G_) * torch.eye(len(G_), device=G_.device), R_)
        cs = float(F.cosine_similarity((phi(Xva) - pm) @ Wt + ym, Yva, dim=1).mean())
        if cs > best[0]:
            best = (cs, Wt)
    Wt = best[1]
    return lambda f: (phi(f) - pm) @ Wt + ym


def fit_mlp(Xtr, Ytr, Xva, Yva, epochs=200, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 1024), torch.nn.GELU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(1024, Ytr.shape[1])).to(Xtr.device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    best = (-9, None)
    for ep in range(epochs):
        net.train()
        for b in torch.randperm(len(Xtr), device=Xtr.device).split(512):
            loss = 1 - F.cosine_similarity(net((Xtr[b] - mu) / sd), Ytr[b], dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad():
                cs = float(F.cosine_similarity(net((Xva - mu) / sd), Yva, dim=1).mean())
            if cs > best[0]:
                best = (cs, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval().requires_grad_(False)
    return lambda f: net((f - mu) / sd)


# ---------------------------------------------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------------------------------------------
class R4(R.R3):
    def __init__(self, B, mon, which='last'):
        super().__init__(B, mon, which)
        t0 = time.time()
        self.ses = load_ses(B)
        reps = R.load_reps(B, mon)
        f = self.ftr
        # ---- queries from audited session-4 test repetitions
        if self.ses is not None:
            tt = N.subject_data(B, mon, keys=['test_trials'])['test_trials']
            te = self.ses['test']
            s4 = [np.nonzero(te[i] == 4)[0] for i in range(200)]
            n4 = min(len(x) for x in s4)
            self.tt4 = np.stack([tt[i, s4[i][:n4]] for i in range(200)]).astype(np.float32)
            for Rq in R.QR:
                if Rq <= n4:
                    self.Q[f's4R{Rq}'] = [self.tt4[:, np.random.RandomState(1000 + q).choice(n4, Rq, replace=False)].mean(1)
                                          for q in range(3 if Rq < 20 else 1)]
            self.q_rebuilt = int(not (te == np.repeat(np.arange(4) + 1, 20)[None]).all())
            del tt
        else:
            self.q_rebuilt = -1
        # ---- warm-up trial sets
        cand = np.setdiff1d(np.concatenate([self.pool, self.held]), self.cal)
        if self.ses is not None:
            tr = self.ses['train']
            ok = (tr[cand] == 4).any(1) & (tr[cand] < 4).any(1)
            m = np.random.RandomState(4).permutation(cand[ok])[:2000]
            assert len(m) > 100, f'only {len(m)} images seen in session 4 and earlier: session values {np.unique(tr)}'
            old_r = np.array([np.nonzero(tr[i] < 4)[0][0] for i in m], dtype=int)
            new_r = np.array([np.nonzero(tr[i] == 4)[0][0] for i in m], dtype=int)
            self.m_imgs = m
            self.old_m = np.asarray(reps[m, old_r], np.float32)
            self.fresh_m = np.asarray(reps[m, new_r], np.float32)
            self.by_ses = {}
            for s in (1, 2, 3, 4):
                ok = (tr[cand] == s).any(1)
                ims = np.random.RandomState(10 + s).permutation(cand[ok])[:800]
                rr = np.array([np.nonzero(tr[i] == s)[0][0] for i in ims], dtype=int)
                self.by_ses[s] = np.asarray(reps[ims, rr], np.float32)
        else:
            self.m_imgs, self.old_m, self.fresh_m, self.by_ses = None, self.unl, None, None
        del reps
        # ---- galleries
        held_y = self.Zsrc_held.mean(0)
        self.dis_pos = np.random.RandomState(5).choice(len(self.held), 1000, replace=False)
        fdis = f[self.held[self.dis_pos]]
        Gp = self.Gtr[self.pool]
        mu = Gp.mean(0)
        Cg = (Gp - mu).T @ (Gp - mu) / len(Gp)
        Wz = I.msqrt(Cg + 0.1 * Cg.trace() / len(Cg) * torch.eye(len(Cg), device=dev()), -0.5)
        preds = {'ridge': fit_ridge(f[self.pool], self.pool_z),
                 'krr': fit_krr(f[self.pool], self.pool_z, f[self.held], held_y),
                 'mlp': fit_mlp(f[self.pool], self.pool_z, f[self.held], held_y)}
        with torch.no_grad():
            self.pred_cos = {k: float(F.cosine_similarity(p(f[self.held]), held_y, dim=1).mean()) for k, p in preds.items()}
        sel = max(self.pred_cos, key=self.pred_cos.get)
        self.pred_sel = sel
        lab = self.labels[self.pool]
        perm_c = np.arange(len(self.pool))
        for cc in np.unique(lab):
            ii = np.nonzero(lab == cc)[0]
            perm_c[ii] = np.random.RandomState(int(cc)).permutation(ii)
        preds['permall'] = fit_ridge(f[self.pool][torch.as_tensor(np.random.RandomState(6).permutation(len(self.pool)), device=dev())], self.pool_z)
        preds['permconcept'] = fit_ridge(f[self.pool][torch.as_tensor(perm_c, device=dev())], self.pool_z)
        # spectrum-matched image gallery: image eigenvectors, anchor-predictor eigenvalues
        with torch.no_grad():
            Pp = preds['ridge'](f[self.pool])
            lp = torch.linalg.eigvalsh((Pp - Pp.mean(0)).T @ (Pp - Pp.mean(0)) / len(Pp)).flip(0).clamp_min(1e-9)
            lg, Vg = torch.linalg.eigh(Cg)
            lg, Vg = lg.flip(0).clamp_min(1e-9), Vg.flip(1)
            Ms = Vg @ torch.diag((lp / lg).sqrt()) @ Vg.T
            gal = lambda fe: self.f0.gallery(fe)  # noqa: E731
            self.gal, self.galfn = {}, {}
            for name, fn in (('img', lambda fe: gal(fe)), ('img_white', lambda fe: nrm((gal(fe) - mu) @ Wz)),
                             ('img_spec', lambda fe: nrm((gal(fe) - mu) @ Ms)),
                             ('anchor', lambda fe: nrm(preds['ridge'](fe))), ('anchor_krr', lambda fe: nrm(preds['krr'](fe))),
                             ('anchor_mlp', lambda fe: nrm(preds['mlp'](fe))), ('anchor_sel', lambda fe: nrm(preds[sel](fe))),
                             ('anchor_permall', lambda fe: nrm(preds['permall'](fe))),
                             ('anchor_permconcept', lambda fe: nrm(preds['permconcept'](fe)))):
                self.gal[name] = (fn(self.fte), fn(fdis))
                self.galfn[name] = fn
            self.held_oracle = nrm(held_y)
            self.gal['oracle'] = (nrm(self.Zsrc_test.mean(0)), nrm(held_y[torch.as_tensor(self.dis_pos, device=dev())]))
            self.anchor_check = float((self.gal['anchor'][0] - self.P).abs().max())
            bank = nrm(self.Zsrc_held.reshape(-1, 512))
            self.bank_src = bank[torch.as_tensor(np.random.RandomState(7).choice(len(bank), 6000, replace=False), device=dev())]
        self.beta = {g: self.pick_beta(g) for g in self.gal}
        print(f'   R4 context ({time.time() - t0:.0f}s): predictor held cos {self.pred_cos} -> {sel}; '
              f'anchor reproduces round 3 (max diff {self.anchor_check:.1e}); queries rebuilt {self.q_rebuilt}; '
              f'session table {"yes" if self.ses is not None else "NO (fallback)"}', flush=True)

    @torch.no_grad()
    def pick_beta(self, g):
        """inverted-softmax beta on source people: held images, one source as queries, the others as the bank."""
        Gh = self.held_oracle if g == 'oracle' else self.galfn[g](self.ftr[self.held])
        return self._beta_gal_from(Gh)

    def _beta_gal_from(self, Gh):
        best = (-1, 20.0)
        S = self.Zsrc_held
        for beta in (5.0, 10.0, 20.0, 40.0, 80.0):
            acc = []
            for i in range(3):
                q = nrm(S[i])
                bk = nrm(torch.cat([S[:i], S[i + 1:]]).reshape(-1, 512))[::3]
                Sc = beta * q @ Gh.T - torch.logsumexp(beta * bk @ Gh.T, dim=0, keepdim=True)
                acc.append(float((Sc.argmax(1) == torch.arange(len(q), device=dev())).float().mean()))
            if np.mean(acc) > best[0]:
                best = (float(np.mean(acc)), beta)
        return best[1]

    def delta(self, Xtr):
        return self.mS['h1'] - I.acts(self.f0, Xtr)['h1'].mean(0)

    def zq(self, d1, keys=QK):
        return {k: [X.acts2(self.f0, q, d1)['z'] for q in self.Q[k]] for k in keys}


@torch.no_grad()
def evaluate(zq, tgt, G, rule, bank=None, beta=20.0):
    q = nrm(zq)
    S = q @ G.T
    if rule.startswith('csls'):
        S = 2 * S - (nrm(bank) @ G.T).topk(10, dim=0).values.mean(0, keepdim=True)
    elif rule == 'isoft_src':
        S = beta * S - torch.logsumexp(beta * nrm(bank) @ G.T, dim=0, keepdim=True)
    t = torch.as_tensor(tgt, device=dev())
    st = S[torch.arange(len(S), device=dev()), t]
    rk = (S > st[:, None]).sum(1).float()
    pred = S.argmax(1)
    cnt = torch.bincount(pred, minlength=G.shape[0]).float()
    p = cnt / cnt.sum()
    ent = float(-(p[p > 0] * p[p > 0].log()).sum() / np.log(G.shape[0]))
    return dict(top1=100 * float((rk == 0).float().mean()), mrr=float((1 / (rk + 1)).mean()),
                hub10=float(cnt.topk(10).values.sum() / cnt.sum()), ent=ent)


# ---------------------------------------------------------------------------------------------------------------
# G: gallery x rule x stress
# ---------------------------------------------------------------------------------------------------------------
def exp_g(c):
    rows = []
    offs = {'none': None, 'old_100': c.delta(c.old_m[np.random.RandomState(3000).choice(len(c.old_m), 100, replace=False)])}
    rng = np.random.RandomState(7)
    miss = np.sort(rng.choice(200, 100, replace=False))
    order = np.random.RandomState(8).permutation(200)
    w = 1 / np.arange(1, 201) ** 1.1
    zipf = order[np.random.RandomState(9).choice(200, 300, p=w / w.sum())]
    for oname, d1 in offs.items():
        Z = c.zq(d1)
        bank_unl = X.acts2(c.f0, c.unl[:400], d1)['z']
        for qk in QK:
            for di, z in enumerate(Z[qk]):
                stresses = {'balanced': (z, np.arange(200), False), 'distract': (z, np.arange(200), True),
                            'missing': (z[miss], miss, False), 'zipf': (z[zipf], zipf, False)}
                for gname, (G200, Gd) in c.gal.items():
                    for sname, (zz, tgt, dis) in stresses.items():
                        G = torch.cat([G200, Gd]) if dis else G200
                        for rule, bank in (('cos', None), ('csls_src', c.bank_src), ('csls_unl', bank_unl),
                                           ('csls_batch', zz), ('isoft_src', c.bank_src)):
                            rows.append(dict(exp='G', offset=oname, query=qk, draw=di, gallery=gname, stress=sname,
                                             rule=rule, **evaluate(zz, tgt, G, rule, bank, c.beta[gname])))
    rows.append(dict(exp='Gpred', **{f'cos_{k}': v for k, v in c.pred_cos.items()}, selected=c.pred_sel,
                     **{f'beta_{k}': v for k, v in c.beta.items()}, q_rebuilt=c.q_rebuilt))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# O: warm-up
# ---------------------------------------------------------------------------------------------------------------
def exp_o(c):
    rows = []
    gals = ('img', 'anchor', 'anchor_sel')

    def score_all(d1, cond, U, budget, draw, keys=QK):
        Z = c.zq(d1, keys)
        for g in gals:
            G = c.gal[g][0]
            for rule in ('cos', 'csls_src'):
                res = {}
                for qk in keys:
                    v = [evaluate(z, np.arange(200), G, rule, c.bank_src) for z in Z[qk]]
                    res[qk] = float(np.mean([x['top1'] for x in v]))
                    if qk == 's4R4':
                        res['mrr_s4R4'] = float(np.mean([x['mrr'] for x in v]))
                rows.append(dict(exp='O', cond=cond, U=U, budget=budget, draw=draw, gallery=g, rule=rule, **res))
    score_all(None, 'none', 0, 0, 0)
    have_fresh = c.fresh_m is not None
    n = len(c.old_m)
    allt = np.concatenate([c.by_ses[s] for s in (1, 2, 3, 4)]) if have_fresh else None
    for U in UGRID:
        for draw in range(10):
            r = np.random.RandomState(3000 + 17 * U + draw)
            sel = r.choice(n, min(U, n), replace=False)
            score_all(c.delta(c.old_m[sel]), 'old', U, U, draw)
            if have_fresh:
                sel2 = r.choice(n, min(2 * U, n), replace=False)
                df = c.delta(c.fresh_m[sel])
                score_all(df, 'fresh', U, U, draw)
                score_all(0.5 * (df + c.delta(c.old_m[sel])), 'blend', U, 2 * U, draw)
                score_all(c.delta(c.old_m[sel2]), 'old_2U', U, 2 * U, draw)
                score_all(c.delta(allt[r.choice(len(allt), U, replace=False)]), 'any_session', U, U, draw)
    for draw in range(10):
        sel = np.random.RandomState(4000 + draw).choice(len(c.unl), 100, replace=False)
        score_all(c.delta(c.unl[sel]), 'old_unmatched_r3', 100, 100, draw)
    # round-3 transductive batch offset
    for g in gals:
        for rule in ('cos', 'csls_src'):
            res = {}
            for qk in QK:
                v = []
                for q in c.Q[qk]:
                    A = X.acts2(c.f0, q)
                    z = I.forward_from(c.f0, 'h1', A['h1'] - A['h1'].mean(0) + c.mS['h1'])
                    v.append(evaluate(z, np.arange(200), c.gal[g][0], rule, c.bank_src)['top1'])
                res[qk] = float(np.mean(v))
            rows.append(dict(exp='O', cond='batch_transductive', U=0, budget=0, draw=0, gallery=g, rule=rule, **res))
    # reciprocal 100/100 cross-fit (session-4 test trials of the other half)
    perm = np.random.RandomState(11).permutation(200)
    halves = (perm[:100], perm[100:])
    old100 = c.delta(c.old_m[np.random.RandomState(3000).choice(n, 100, replace=False)])
    for h, (warm, qi) in enumerate(((halves[0], halves[1]), (halves[1], halves[0]))):
        n4 = c.tt4.shape[1]
        conds = {'none': None, 'old_100': old100,
                 'cross_all': c.delta(c.tt4[warm].reshape(-1, *c.tt4.shape[2:])),
                 'cross_100': c.delta(c.tt4[warm, np.random.RandomState(12 + h).randint(0, n4, 100)])}
        for cond, d1 in conds.items():
            Z = c.zq(d1, ['s4R1', 's4R4', 's4R20'])
            for g in gals:
                res = {qk: float(np.mean([evaluate(z[qi], qi, c.gal[g][0], 'cos')['top1'] for z in Z[qk]]))
                       for qk in Z}
                rows.append(dict(exp='Ocross', cond=cond, half=h, gallery=g, rule='cos', **res))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# D: drift
# ---------------------------------------------------------------------------------------------------------------
def exp_d(c):
    rows = []
    if c.by_ses is None:
        return rows
    for draw in range(5):
        m = {}
        for s in (1, 2, 3, 4):
            p = np.random.RandomState(5000 + 10 * draw + s).permutation(len(c.by_ses[s]))
            m[s] = [I.acts(c.f0, c.by_ses[s][p[:200]])['h1'].mean(0), I.acts(c.f0, c.by_ses[s][p[200:400]])['h1'].mean(0)]
        scale = float((c.mS['h1'] - m[4][0]).norm())
        for s in (1, 2, 3, 4):
            for t in (1, 2, 3, 4):
                dd = float((m[s][0] - m[t][1]).norm())
                rows.append(dict(exp='Ddist', draw=draw, s=s, t=t, dist=dd, rel=dd / scale, same=int(s == t)))
        for s in (1, 2, 3, 4):
            d1 = c.mS['h1'] - m[s][0]
            Z = c.zq(d1, ['s4R4', 's4R20'])
            for g in ('img', 'anchor'):
                rows.append(dict(exp='Dret', draw=draw, offset_session=s, gallery=g,
                                 **{qk: float(np.mean([evaluate(z, np.arange(200), c.gal[g][0], 'cos')['top1'] for z in Z[qk]]))
                                    for qk in Z}))
    return rows



# ---------------------------------------------------------------------------------------------------------------
# M: methods that try to RAISE zero-label accuracy, built on the round-3 findings (selection on source people only)
#   head   contrastive gallery head: the ridge anchor predictor fine-tuned with InfoNCE against individual source
#          people's outputs (ridge fits the population MEAN; retrieval needs the right ranking under person noise)
#   shrink query shrinkage onto the anchor subspace (Wiener weights on the predicted-anchor PCs): the useful signal
#          lives in the high-variance subspace and low-repeat queries are noise-dominated
#   mix    per-source galleries (ridge per source person) scored as a mixture: lse = soft max over sources;
#          sel = sources weighted by how well each explains the query batch (transductive, no labels)
# ---------------------------------------------------------------------------------------------------------------
def fit_head(c, steps=600, tau=0.05, seed=0):
    torch.manual_seed(seed)
    f, cal, held = c.ftr, c.cal, c.held
    Bm, xm, ym = X.ridge_svd(f[c.pool], c.pool_z)
    lin = torch.nn.Linear(f.shape[1], 512).to(dev())
    with torch.no_grad():
        lin.weight.copy_(Bm.T)
        lin.bias.copy_(ym - xm @ Bm)
    opt = torch.optim.AdamW(lin.parameters(), 1e-4, weight_decay=1e-4)
    Zc = nrm(c.Zsrc_cal)                                                # (9, 2000, 512) individual source outputs
    Zh = c.Zsrc_held

    @torch.no_grad()
    def val():
        G = nrm(lin(f[held]))
        return float(np.mean([(nrm(Zh[i]) @ G.T).argmax(1).eq(torch.arange(len(G), device=dev())).float().mean().item()
                              for i in range(len(Zh))]))
    best = (val(), {k: v.clone() for k, v in lin.state_dict().items()}, 0)
    for st in range(1, steps + 1):
        ii = torch.randperm(len(cal), device=dev())[:512]
        ss = torch.randint(0, len(Zc), (len(ii),), device=dev())
        q = Zc[ss, ii]
        k = nrm(lin(f[torch.as_tensor(cal, device=dev())[ii]]))
        lg = q @ k.T / tau
        t = torch.arange(len(ii), device=dev())
        pb = torch.as_tensor(np.random.RandomState(st).choice(len(c.pool), 512, replace=False), device=dev())
        loss = (F.cross_entropy(lg, t) + F.cross_entropy(lg.T, t)) / 2 + \
            (1 - F.cosine_similarity(lin(f[torch.as_tensor(c.pool, device=dev())[pb]]), c.pool_z[pb], dim=1).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % 50 == 0:
            v = val()
            if v > best[0]:
                best = (v, {kk: vv.clone() for kk, vv in lin.state_dict().items()}, st)
    lin.load_state_dict(best[1])
    lin.requires_grad_(False)
    return lambda fe: nrm(lin(fe)), best[0], best[2]


@torch.no_grad()
def fit_shrink(c, galfn):
    """Wiener shrinkage of (query - source query mean) on the predicted-anchor PCs; tau chosen on source held queries."""
    Pp = c.galfn['anchor'](c.ftr[c.pool])
    mu = Pp.mean(0)
    a, V = torch.linalg.eigh((Pp - mu).T @ (Pp - mu) / len(Pp))
    mq = c.Zsrc_held.reshape(-1, 512).mean(0)
    Gh = galfn(c.ftr[c.held])
    best = (-1, None)
    for tau in [0.0] + [float(x) for x in (a.max().item() * torch.logspace(-4, 0, 9))]:
        w = a / (a + tau) if tau > 0 else torch.ones_like(a)
        fn = (lambda z, w=w: (z - mq) @ V @ torch.diag(w) @ V.T + mq)
        acc = float(np.mean([(nrm(fn(c.Zsrc_held[i])) @ Gh.T).argmax(1).eq(torch.arange(len(Gh), device=dev())).float().mean().item()
                             for i in range(len(c.Zsrc_held))]))
        if acc > best[0]:
            best = (acc, fn, tau)
    return best[1], best[2]


def exp_m(c):
    rows = []
    head, head_val, head_step = fit_head(c)
    with torch.no_grad():
        ridge_val = float(np.mean([(nrm(c.Zsrc_held[i]) @ c.galfn['anchor'](c.ftr[c.held]).T).argmax(1)
                                   .eq(torch.arange(len(c.held), device=dev())).float().mean().item() for i in range(len(c.Zsrc_held))]))
        Ghead = head(c.fte)
        cal_t = torch.as_tensor(c.cal, device=dev())
        Gs = []
        for i in range(len(c.Zsrc_cal)):
            pr = fit_ridge(c.ftr[cal_t], c.Zsrc_cal[i])
            Gs.append(nrm(pr(c.fte)))
        Gs = torch.stack(Gs)                                            # (9, 200, 512)
    sh_a, tau_a = fit_shrink(c, c.galfn['anchor'])
    sh_h, tau_h = fit_shrink(c, head)
    with torch.no_grad():                                               # anchor PCs for the label-free Wiener rule
        Pp = c.galfn['anchor'](c.ftr[c.pool])
        a_sig, Vp = torch.linalg.eigh((Pp - Pp.mean(0)).T @ (Pp - Pp.mean(0)) / len(Pp))
    nrep = {'s4R1': 1, 's4R4': 4, 's4R20': 20, 'bench': 80}
    rows.append(dict(exp='Mfit', head_val_top1=head_val, ridge_val_top1=ridge_val, head_step=head_step,
                     shrink_tau_anchor=tau_a, shrink_tau_head=tau_h))
    offs = {'none': None, 'old_100': c.delta(c.old_m[np.random.RandomState(3000).choice(len(c.old_m), 100, replace=False)])}
    t200 = np.arange(200)

    @torch.no_grad()
    def mix_scores(z, mode):
        S = torch.einsum('qd,sjd->sqj', nrm(z), Gs)                     # (9, q, 200)
        if mode == 'lse':
            return torch.logsumexp(20 * S, 0)
        fit = S.max(2).values.mean(1)                                   # how well each source explains the batch
        w = torch.softmax(fit / 0.01, 0)
        return (w[:, None, None] * S).sum(0)
    for oname, d1 in offs.items():
        Z = c.zq(d1)
        with torch.no_grad():                                           # signal / noise per anchor PC from the
            n4 = c.tt4.shape[1]                                         # person's own repeats (label-free, uses
            zt = X.acts2(c.f0, c.tt4.reshape(-1, *c.tt4.shape[2:]), d1)['z'].reshape(200, n4, -1)  # repeat grouping)
            pt = zt @ Vp
            noise1 = pt.var(1).mean(0)
            a_s = (pt.mean(1).var(0) - noise1 / n4).clamp_min(1e-9)
            mu_u = zt.mean((0, 1))
        for qk in QK:
            wR = a_s / (a_s + noise1 / nrep[qk])                       # Wiener weight for an R-repeat query
            shR = (lambda z, wR=wR: (z - mu_u) @ Vp @ torch.diag(wR) @ Vp.T + mu_u)
            for di, z in enumerate(Z[qk]):
                with torch.no_grad():
                    cands = {'anchor+shrinkR': (shR(z), c.gal['anchor'][0]), 'head+shrinkR': (shR(z), Ghead),
                             'anchor+center': (z - z.mean(0), c.gal['anchor'][0]), 'head+center': (z - z.mean(0), Ghead),'anchor': (z, c.gal['anchor'][0]), 'img': (z, c.gal['img'][0]), 'head': (z, Ghead),
                             'anchor+shrink': (sh_a(z), c.gal['anchor'][0]), 'head+shrink': (sh_h(z), Ghead),
                             'img+shrink': (sh_a(z), c.gal['img'][0])}
                    for name, (zz, G) in cands.items():
                        r = evaluate(zz, t200, G, 'cos')
                        rows.append(dict(exp='M', offset=oname, query=qk, draw=di, method=name, top1=r['top1'], mrr=r['mrr']))
                    for mode in ('lse', 'sel'):
                        S = mix_scores(z, mode)
                        st = S[t200, t200]
                        rk = (S > st[:, None]).sum(1).float()
                        rows.append(dict(exp='M', offset=oname, query=qk, draw=di, method=f'mix_{mode}',
                                         top1=100 * float((rk == 0).float().mean()), mrr=float((1 / (rk + 1)).mean())))
    return rows


def run_target(B, mon, which):
    t0 = time.time()
    c = R4(B, mon, which)
    out = {'g': exp_g(c)}
    print(f'   G done ({time.time() - t0:.0f}s)', flush=True)
    out['o'] = exp_o(c)
    print(f'   O done ({time.time() - t0:.0f}s)', flush=True)
    out['d'] = exp_d(c)
    out['m'] = exp_m(c)
    for name, rows in out.items():
        df = pd.DataFrame(rows)
        df['target'], df['montage'] = B, mon
        df.to_csv(os.path.join(odir(), f'{name}_t{B:02d}_{mon}.csv'), index=False)
    print(f'  target {B} {mon}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# placement (model-free, raw test EEG)
# ---------------------------------------------------------------------------------------------------------------
def placement(which_people=None, n_perm=200):
    which_people = which_people or PEOPLE
    from eegalign import ubp as U
    crown, post = U.MONTAGES['crown8'], U.MONTAGES['posterior17']
    data, ses = {}, {}
    for p in PEOPLE:
        a = N.subject_data(p, 'posterior17', keys=['test_trials'])['test_trials']
        b = N.subject_data(p, 'crown8', keys=['test_trials'])['test_trials']
        data[p] = np.concatenate([a, b[:, :, [crown.index(ch) for ch in crown if ch not in post]]], 2)
        st = load_ses(p)
        ses[p] = st['test'] if st is not None else np.repeat(np.arange(4) + 1, 20)[None].repeat(200, 0)
    chs = post + [ch for ch in crown if ch not in post]
    ci = {ch: i for i, ch in enumerate(chs)}

    def halves(p, R_=None, draw=0):
        x, s = data[p], ses[p]
        tm = np.stack([x[i, np.isin(s[i], (1, 2))].astype(np.float32).mean(0) for i in range(200)])
        if R_ is None:
            q = np.stack([x[i, np.isin(s[i], (3, 4))].astype(np.float32).mean(0) for i in range(200)])
        else:
            q = np.stack([x[i, np.random.RandomState(100 * draw + i).permutation(np.nonzero(s[i] == 4)[0])[:R_]]
                          .astype(np.float32).mean(0) for i in range(200)])
        return tm, q
    # single-channel ranks per person (sessions 1-2 vs 3-4)
    single = {p: {} for p in PEOPLE}
    H = {p: halves(p) for p in PEOPLE}
    for p in PEOPLE:
        for ch in chs:
            single[p][ch] = R.raw_ident(H[p][0][:, [ci[ch]]], H[p][1][:, [ci[ch]]])[0]
    rows = []
    for B in which_people:
        rank = {ch: np.mean([single[p][ch] for p in PEOPLE if p != B]) for ch in chs}
        cr = sorted(crown, key=lambda ch: rank[ch])                       # worst first
        po = sorted([ch for ch in post if ch not in crown], key=lambda ch: -rank[ch])
        mons = {'crown8': crown, 'occipital8': U.MONTAGES['occipital8'], 'posterior17': post,
                'crown_po2': ['PO3', 'PO4'], 'best8_src': sorted(chs, key=lambda ch: -rank[ch])[:8]}
        for m in (1, 2, 4, 6):
            mons[f'crown_swap{m}'] = cr[m:] + po[:m]
        HQ = {(R_, d): halves(B, R_, d) for R_ in (4, 20) for d in range(3 if R_ < 20 else 1)}
        for mname, mc in mons.items():
            ix = [ci[ch] for ch in mc]
            tm, q = H[B]
            acc, gap = R.raw_ident(tm[:, ix], q[:, ix])
            null = []
            for k in range(n_perm):
                pp = np.random.RandomState(k).permutation(200)
                null.append(R.raw_ident(tm[pp][:, ix], q[:, ix])[0])
            res = dict(exp='placement', target=B, montage=mname, channels=' '.join(mc), q='s34_40', top1=acc, gap=gap,
                       null_mean=float(np.mean(null)), null_95=float(np.percentile(null, 95)))
            rows.append(res)
            for R_ in (4, 20):
                v = [(lambda h: R.raw_ident(h[0][:, ix], h[1][:, ix])[0])(HQ[(R_, d)]) for d in range(3 if R_ < 20 else 1)]
                rows.append(dict(exp='placement', target=B, montage=mname, channels=' '.join(mc), q=f's4R{R_}', top1=float(np.mean(v))))
            # evoked SNR: grand ERP power over time / single-trial residual power, mean over channels
            x = data[B][:, :, ix].astype(np.float32)
            erp = x.mean((0, 1))
            res_var = (x - erp).var(-1).mean()
            rows.append(dict(exp='snr', target=B, montage=mname, snr=float(erp.var(-1).mean() / res_var)))
        rows.append(dict(exp='rank', target=B, **{f'r_{ch}': rank[ch] for ch in chs}))
        print(f'  placement target {B} done', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(odir(), 'placement.csv'), index=False)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
CONTRASTS = []


def pstats(a, b, label, primary=False, margin=None):
    """a, b: Series by target. People 2-10. Ties reported; sign test on non-ties; Wilcoxon separately."""
    from scipy import stats
    idx = sorted(set(a.dropna().index) & set(b.dropna().index) & set(TEST))
    d = (a.reindex(idx) - b.reindex(idx)).values.astype(float)
    if len(d) == 0:
        print(f'  {label}: no data')
        return np.nan
    rng = np.random.RandomState(0)
    boot = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
    pos, zer, neg = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    sign_p = float(stats.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else 1.0
    try:
        wil = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        wil = np.nan
    lopo = [np.delete(d, i).mean() for i in range(len(d))]
    eq = ''
    if margin is not None:
        lo, hi = np.percentile(boot, 5), np.percentile(boot, 95)
        eq = f'; 90% CI {lo:+.2f} to {hi:+.2f} -> {"EQUIVALENT within" if (lo > -margin and hi < margin) else "not shown equivalent at"} +-{margin}'
    print(f'  {"[PRIMARY] " if primary else ""}{label}: mean {d.mean():+.2f} (95% CI {np.percentile(boot, 2.5):+.2f} to '
          f'{np.percentile(boot, 97.5):+.2f}), +{pos}/0:{zer}/-{neg}, sign p={sign_p:.4f} (non-ties), Wilcoxon p={wil:.4f}, '
          f'LOPO {min(lopo):+.2f} to {max(lopo):+.2f}{eq}')
    print('     per person: ' + ', '.join(f't{t}:{v:+.3f}' for t, v in zip(idx, d)))
    CONTRASTS.append(dict(label=label, primary=primary, mean=d.mean(), pos=pos, zero=zer, neg=neg, sign_p=sign_p,
                          wilcoxon_p=wil, **{f't{t}': v for t, v in zip(idx, d)}))
    return wil


def load(exp, mon):
    f = sorted(glob.glob(os.path.join(odir(), f'{exp}_t*_{mon}.csv')))
    return pd.concat([pd.read_csv(x) for x in f], ignore_index=True) if f else None


def report(mons):
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    prim = {}
    au = os.path.join(odir(), 'session_audit.json')
    if os.path.exists(au):
        a = pd.DataFrame(json.load(open(au))).T
        for sub in a.index:                                              # label-based stats recomputed on 1-4 ids
            st = load_ses(int(sub))
            if st is not None:
                a.loc[sub, 'test_reps_as_assumed'] = float((st['test'] == np.repeat(np.arange(4) + 1, 20)[None]).mean())
                a.loc[sub, 'train_rep01_before_rep23'] = float((st['train'][:, :2].max(1) < st['train'][:, 2:].min(1)).mean())
                a.loc[sub, 'train_imgs_with_session4'] = int((st['train'] == 4).any(1).sum())
        print('SESSION AUDIT (per person):')
        cols = ['test_match_corr', 'test_best_other', 'test_confident', 'test_reps_as_assumed', 'train_match_corr',
                'train_best_other', 'train_confident', 'train_identity_perm', 'train_rep01_before_rep23',
                'train_imgs_with_session4']
        print(a[[c for c in cols if c in a]].to_string())
    else:
        have = [p for p in PEOPLE if os.path.exists(ses_path(p))]
        print(f'SESSION AUDIT: no audit summary; session tables present for people {have or "none"}'
              + ('' if have else ' (fallback assumptions in use; fresh warm-up and drift skipped)'))
    for mon in mons:
        g = load('g', mon)
        if g is None:
            continue
        print(f'\n{"=" * 120}\n{mon}\n{"=" * 120}')
        gp = g[g.exp == 'Gpred']
        if len(gp):
            print('anchor predictors, held-out-concept cosine to source outputs (source data only):')
            print(gp[[c for c in gp if c.startswith('cos_')]].mean().round(4).to_string(), '\n  selected:',
                  gp.selected.value_counts().to_dict(), '\n  queries rebuilt from audited sessions:', gp.q_rebuilt.value_counts().to_dict())
        g = g[g.exp == 'G']
        per = g.groupby(['offset', 'query', 'gallery', 'stress', 'rule', 'target'])[['top1', 'mrr', 'hub10', 'ent']].mean()
        T_ = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3, 4]).mean()

        def cell(off, q, gal, st, rule, col='top1'):
            return per.loc[(off, q, gal, st, rule)][col]
        print('\nG: zero-label galleries, offset none, balanced 200-way (top-1 %; hub10 and entropy at s4R4):')
        tb = T_.xs(('none', 'balanced'), level=('offset', 'stress'))['top1'].unstack('query')[QK]
        hb = T_.xs(('none', 's4R4', 'balanced'), level=('offset', 'query', 'stress'))[['hub10', 'ent']]
        print(pd.concat([tb, hb], axis=1).round(3).to_string())
        print('\nG stress, s4R4, offset none (top-1 %):')
        print(T_.xs(('none', 's4R4'), level=('offset', 'query'))['top1'].unstack('stress').round(2).to_string())
        print('\nG with old_100 offset, balanced (top-1 %):')
        print(T_.xs(('old_100', 'balanced'), level=('offset', 'stress'))['top1'].unstack('query')[QK].round(2).to_string())
        tag = 'PRIMARY' if mon == 'posterior17' else 'replication'
        print(f'\nQ1 ({tag}, s4R4, offset none, balanced):')
        A = cell('none', 's4R4', 'anchor', 'balanced', 'cos')
        p = pstats(A, cell('none', 's4R4', 'img', 'balanced', 'csls_src'), 'Q1 anchor/cos - img/csls_src',
                   primary=(mon == 'posterior17'))
        if mon == 'posterior17':
            prim['Q1'] = p
        pstats(A, cell('none', 's4R4', 'img', 'balanced', 'cos'), 'anchor/cos - img/cos (B - A)')
        pstats(A, cell('none', 's4R4', 'img_spec', 'balanced', 'cos'), 'anchor - spectrum-matched image gallery')
        pstats(A, cell('none', 's4R4', 'img_white', 'balanced', 'cos'), 'anchor - whitened image gallery')
        pstats(A, cell('none', 's4R4', 'anchor_permconcept', 'balanced', 'cos'), 'anchor - anchor with pairs shuffled within concept')
        pstats(cell('none', 's4R4', 'anchor', 'balanced', 'csls_src'), cell('none', 's4R4', 'img', 'balanced', 'csls_src'),
               'anchor - img, both csls_src')
        pstats(cell('none', 's4R4', 'anchor_sel', 'balanced', 'cos'), A, 'selected predictor - ridge (both cos)')
        pstats(cell('none', 's4R4', 'oracle', 'balanced', 'cos'), A, 'oracle (measured source test EEG) - anchor')
        pstats(cell('none', 's4R4', 'anchor', 'distract', 'cos'), cell('none', 's4R4', 'img', 'distract', 'csls_src'),
               'Q1 under +1000 distractors')
        pstats(cell('none', 's4R4', 'anchor', 'zipf', 'cos'), cell('none', 's4R4', 'img', 'zipf', 'csls_src'),
               'Q1 under Zipf repeated queries')
        # O
        o = load('o', mon)
        if o is not None:
            ob = o[o.exp == 'O']
            perO = ob.groupby(['gallery', 'rule', 'cond', 'U', 'target'])[['s4R1', 's4R4', 's4R20', 'bench']].mean()
            TO = perO[perO.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3]).mean()
            print('\nO: warm-up curves, anchor gallery, cos (top-1 %):')
            print(TO.xs(('anchor', 'cos'), level=('gallery', 'rule')).round(2).to_string())
            print('\nO: image gallery, cos (top-1 %):')
            print(TO.xs(('img', 'cos'), level=('gallery', 'rule'))[['s4R4', 's4R20']].unstack('cond').round(2).to_string())

            def oc(cond, U, q='s4R4', gal='anchor', rule='cos'):
                return perO.loc[(gal, rule, cond, U)][q]
            none = perO.loc[('anchor', 'cos', 'none', 0)]['s4R4']
            if ('anchor', 'cos', 'fresh', 100) in perO.index.droplevel('target'):
                print(f'\nQ2 ({tag}, anchor gallery, s4R4, matched images):')
                p = pstats(oc('fresh', 100), oc('old', 100), 'Q2 fresh_100 - old_100', primary=(mon == 'posterior17'))
                if mon == 'posterior17':
                    prim['Q2'] = p
                pstats(oc('fresh', 100), none, 'fresh_100 - none')
                pstats(oc('blend', 100), oc('old_2U', 100), 'blend (50 old + 50 fresh per 100) - old_200, budget 200')
                pstats(oc('fresh', 100, 's4R20'), oc('old', 100, 's4R20'), 'fresh_100 - old_100 at s4R20')
                pstats(oc('fresh', 400), oc('old', 400), 'fresh_400 - old_400')
                pstats(oc('any_session', 100), oc('old', 100), 'random-session control - old_100')
            pstats(oc('old', 100), none, 'old_100 - none (anchors; D - B)', margin=MARGIN)
            pstats(oc('old_unmatched_r3', 100), none, 'round-3 style old_100 - none (anchors)', margin=MARGIN)
            oc_ = o[o.exp == 'Ocross']
            if len(oc_):
                pc = oc_[oc_.gallery == 'anchor'].groupby(['cond', 'target'])[['s4R1', 's4R4', 's4R20']].mean()
                print('\nOcross (reciprocal 100/100 cross-fit, anchor gallery, cos; queries = the other 100 images):')
                print(pc[pc.index.get_level_values('target').isin(TEST)].groupby(level=0).mean().round(2).to_string())
                pstats(pc.loc['cross_100']['s4R4'], pc.loc['old_100']['s4R4'], 'cross-fit 100 session-4 trials - old_100 (s4R4)')
                pstats(pc.loc['cross_all']['s4R20'], pc.loc['old_100']['s4R20'], 'cross-fit all session-4 trials - old_100 (s4R20)')
        d = load('d', mon)
        if d is not None and len(d):
            dd = d[(d.exp == 'Ddist') & d.target.isin(TEST)]
            print('\nD: distance between session means of h1 (200 trials each, image-disjoint halves), relative to |source - session 4|:')
            print(dd.groupby(['s', 't']).rel.mean().unstack('t').round(3).to_string())
            w = dd[dd.same == 1].groupby('target').rel.mean()
            acr = dd[(dd.same == 0) & ((dd.s == 4) | (dd.t == 4))].groupby('target').rel.mean()
            pstats(acr, w, 'across-session (vs session 4) - within-session distance')
            dr = d[(d.exp == 'Dret') & d.target.isin(TEST)]
            print('\nD: session-4 retrieval with an offset estimated in session s (top-1 %):')
            print(dr.groupby(['gallery', 'offset_session'])[['s4R4', 's4R20']].mean().unstack('gallery').round(2).to_string())
    for mon in mons:
        m = load('m', mon)
        if m is None:
            continue
        print(f'\nM ({mon}): methods to raise zero-label accuracy (exploratory; settings chosen on source people only)')
        mf = m[(m.exp == 'Mfit') & m.target.isin(TEST)]
        print(mf[['head_val_top1', 'ridge_val_top1', 'head_step', 'shrink_tau_anchor', 'shrink_tau_head']].mean().round(4).to_string())
        mm = m[m.exp == 'M']
        pm = mm.groupby(['offset', 'method', 'query', 'target']).top1.mean()
        tm = pm[pm.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2]).mean().unstack('query')
        print(tm[[q for q in QK if q in tm]].round(2).to_string())
        base = pm.loc[('none', 'anchor', 's4R4')]
        for meth in ('head', 'anchor+shrink', 'anchor+shrinkR', 'anchor+center', 'head+center', 'mix_lse', 'mix_sel'):
            pstats(pm.loc[('none', meth, 's4R4')], base, f'{meth} - anchor (s4R4, no offset)')
        pstats(pm.loc[('none', 'head+center', 's4R20')], pm.loc[('none', 'anchor', 's4R20')], 'head+center - anchor (s4R20)')
        pstats(pm.loc[('none', 'head+center', 's4R4')], pm.loc[('none', 'img', 's4R4')], 'head+center - f0 image gallery (s4R4)')
    pl = os.path.join(odir(), 'placement.csv')
    if os.path.exists(pl):
        p_ = pd.read_csv(pl)
        pp = p_[(p_.exp == 'placement') & p_.target.isin(TEST)]
        print('\nPLACEMENT (model-free raw-EEG image identification, %; chance 0.5):')
        print(pp.groupby(['montage', 'q']).top1.mean().unstack('q').round(2).to_string())
        nl = pp[pp.q == 's34_40'].groupby('montage')[['null_mean', 'null_95']].mean()
        sn = p_[(p_.exp == 'snr') & p_.target.isin(TEST)].groupby('montage').snr.mean()
        print(pd.concat([nl, sn], axis=1).round(3).to_string())
        t = pp[pp.q == 's34_40'].pivot_table(index='target', columns='montage', values='top1')
        for m in ('crown_swap1', 'crown_swap2', 'crown_swap4', 'best8_src'):
            if m in t:
                pstats(t[m], t['crown8'], f'{m} - crown8 (s34_40)')
        if 'occipital8' in t:
            gap = t['occipital8'] - t['crown8']
            for m in ('crown_swap1', 'crown_swap2', 'crown_swap4'):
                if m in t:
                    print(f'  {m}: closes {100 * float(((t[m] - t["crown8"]) / gap).reindex(TEST).mean()):.0f}% of the occipital-8 gap (mean over people)')
    if prim:
        from statsmodels.stats.multitest import multipletests
        keys = sorted(prim)
        adj = multipletests([prim[k] for k in keys], method='holm')[1]
        print('\nHOLM-ADJUSTED PRIMARIES:', {k: round(float(v), 4) for k, v in zip(keys, adj)})
    print('\nDECISION RULES (prespecified):\n'
          '  Q1 > 0: the neural anchor gallery is a contribution beyond hubness correction -> lead with it, freeze the predictor.\n'
          '  Q1 ~ 0 but anchor > img/cos: the gain is gallery geometry; the paper compares two cheap calibration-free readouts.\n'
          '  anchor ~ img_spec: neural pairing is not needed for the gain; narrow the claim to gallery spectrum.\n'
          '  Q2 > 0 at s4R4 by a worthwhile amount: recommend a same-day warm-up. Only at s4R20: high-repeat use only.\n'
          '  old_100 - none equivalent within +-1 at s4R4: drop the earlier-session warm-up from the default pipeline.\n'
          '  Crown swap closes most of the occipital gap with 1-2 sites: placement trade-off, test that montage on hardware.')
    pd.DataFrame(CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['sessions', 'run', 'placement', 'report'])
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
    if a.cmd == 'sessions':
        return sessions(a.subjects)
    if a.cmd == 'placement':
        return placement(a.targets)
    if a.cmd == 'report':
        return report(a.montages)
    for mon in a.montages:
        for B in a.targets:
            mf = os.path.join(odir(), f'm_t{B:02d}_{mon}.csv')
            if os.path.exists(os.path.join(odir(), f'd_t{B:02d}_{mon}.csv')):
                if os.path.exists(mf):
                    print(f'[run] {B} {mon}: done')
                    continue
                print(f'[run] target {B} {mon}: methods only', flush=True)
                c = R4(B, mon, a.which)
                df = pd.DataFrame(exp_m(c))
                df['target'], df['montage'] = B, mon
                df.to_csv(mf, index=False)
                continue
            print(f'[run] target {B} {mon}', flush=True)
            run_target(B, mon, a.which)


if __name__ == '__main__':
    main()
