"""Round 6: what does a short labeled calibration session buy, on top of the best zero-label readout?

    python scripts/nb_round6.py run    --config C --montages posterior17 occipital8
    python scripts/nb_round6.py report --config C

Rounds 3-5: maps fitted on single presentations do not beat an offset, but with averaged calibration they did
(round 3 benchmark track: rigid at 200 images 22.7 vs translation 19.4; activation matching at 2,000 images 37.1),
and THINGS-EEG2's 5 Hz presentation makes averaged calibration cheap (200 images x 2 presentations = 80 s of
viewing). Round 5: f0 trained on 1-4 presentation averages (repaug) doubles the zero-label pipeline at s4R4
(5.1 -> 10.2) but loses ~1 point on 80-presentation queries.

Calibration session (honest version). THINGS-EEG2 shows each training image twice per session, so a session gives
k images x 2 presentations. Using the audited session table, calibration images come from ONE session:
  cal_session 3 = an earlier day (the day before the session-4 queries); cal_session 4 = the same day as the queries
  (training images of session 4, disjoint from the test images: calibrate at the start, then use).
k = 25 / 100 / 200 / 500 / 1000 / 2000 / 4000 pool images (presentation time 2k x 0.2 s: 10 s ... 27 min).
Queries: session-4 test images at R = 1 / 4 / 20 and the 80-presentation benchmark.

Decoders: loso (f0) and repaug (round 5), each with its own source anchors; ens = sum of the two decoders' score
matrices (same method on both).
Methods (all with the label-free h1 offset estimated from the calibration trials themselves):
  zero      no labels (offset only); readout anchors + batch CSLS = round-5 best pipeline (reference)
  trans     shrunk translation (kappa 5) to the source anchors
  rigid     identity-regularized Procrustes + scale (lambda 1)
  rr4 / rr16 / full   activation matching to the source anchors (offset + reduced-rank ridge, GCV)
  ft        fine-tune all of the decoder on the calibration trials with the NeuroBridge loss (image gallery,
            presentations re-averaged 1-2 at random, AdamW 3e-5, 300 steps, fixed in advance)
  ft_head   same, projector + last encoder block only
Readouts: image gallery and predicted anchors (maps, zero), image gallery for ft (its output space moved); rules
cosine and batch CSLS. Within-person decoder (trained on the person's own 4-presentation pool data, all sessions)
is printed as the ceiling.

Primary (prespecified; posterior-17, repaug, s4R4, cal_session 3, k = 200, people 2-10, Wilcoxon):
  best calibrated method (chosen on subject 1 among trans / rigid / rr4 / rr16 / full / ft / ft_head, each with
  its best readout x rule on subject 1)  minus  zero-label best pipeline.
Secondary: the same at every k (the curve), same day vs earlier day at k = 200, ens vs the better single decoder,
occipital-8 replication.
"""
import argparse
import copy
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
import nb_next as X  # noqa: E402
import nb_round3 as R  # noqa: E402
import nb_round4 as R4m  # noqa: E402
import nb_round5 as R5m  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
KS = [25, 100, 200, 500, 1000, 2000, 4000]
QK = ['s4R1', 's4R4', 's4R20', 'bench']
DECODERS = ['loso', 'repaug']
CAL_SESSIONS = (3, 4)
FT_VARIANTS = (('ft', False), ('ft_head', True))
PRIMARY = dict(montage='posterior17', decoder='repaug', query='s4R4', cal_session=3, k=200)


def dev():
    return N.ENV['device']


def odir():
    d = os.path.join(N.ENV['results_dir'], 'nb_round6')
    os.makedirs(d, exist_ok=True)
    return d


def ndraw(k):
    return 5 if k <= 200 else (3 if k <= 1000 else 1)


def nrm(x):
    return F.normalize(x.float(), dim=-1)


@torch.no_grad()
def smat(z, G, rule):
    S = nrm(z) @ G.T
    if rule == 'csls_batch':
        S = 2 * S - S.topk(10, dim=0).values.mean(0, keepdim=True)
    return S


def top1(S):
    return 100 * float((S.argmax(1) == torch.arange(len(S), device=S.device)).float().mean())


def finetune(f0, x2, img, ftr, head_only=False, steps=300, lr=3e-5, seed=0):
    """x2: (k, 2, C, T) calibration presentations; img: (k,) train-image indices."""
    torch.manual_seed(seed)
    m = copy.deepcopy(f0).requires_grad_(False)
    params = list(m.eeg_projector.parameters()) + list(m.encoder.model[-1].parameters()) if head_only else \
        list(m.encoder.parameters()) + list(m.eeg_projector.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    X2 = torch.as_tensor(x2, device=dev())
    IM = torch.as_tensor(img, device=dev())
    k = len(X2)
    bs = min(256, k)
    m.train()
    for _ in range(steps):
        idx = torch.randint(0, k, (bs,), device=dev())
        both = torch.rand(bs, device=dev()) < 0.5
        pick = torch.randint(0, 2, (bs,), device=dev())
        xb = torch.where(both[:, None, None], X2[idx].mean(1), X2[idx, pick])
        with torch.no_grad():
            g = m.gallery(ftr[IM[idx]])
        loss = N.nb_loss(m(xb), g, m.scale())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    m.eval().requires_grad_(False)
    return m


class Ctx:
    """One decoder's round-4 context plus everything round 6 needs."""

    def __init__(self, kind, B, mon):
        R5m.use_kind(kind)
        self.kind = kind
        self.c = R4m.R4(B, mon, 'last')
        c = self.c
        self.pool_pos = {int(i): j for j, i in enumerate(c.pool)}
        self.Gimg, self.Ganc = c.gal['img'][0], c.gal['anchor'][0]


def run_target(B, mon):
    out = os.path.join(odir(), f'cal_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[run] {B} {mon}: done')
        return
    t0 = time.time()
    ctx = {k: Ctx(k, B, mon) for k in DECODERS if os.path.exists(N.ckpt_path(k, mon, B, R5m.SEED, 'last'))}
    c0 = next(iter(ctx.values())).c
    assert c0.ses is not None, 'round 6 needs the session table (nb_round4.py sessions)'
    tr = c0.ses['train']
    reps = R.load_reps(B, mon)
    ftr = c0.ftr
    rows = []

    def record(base, S_by_dec):
        for dec, S in S_by_dec.items():
            rows.append(dict(base, decoder=dec, top1=top1(S)))
        if len(S_by_dec) == 2:
            rows.append(dict(base, decoder='ens', top1=top1(sum(S_by_dec.values()))))
    # within-person ceiling
    with torch.no_grad():
        for qk in QK:
            for rule in ('cos', 'csls_batch'):
                v = [top1(smat(c0.w(torch.as_tensor(q, device=dev())), c0.Gw, rule)) for q in c0.Q[qk]]
                rows.append(dict(cal_session=0, k=0, draw=0, method='within', readout='img', rule=rule, query=qk,
                                 decoder='within', top1=float(np.mean(v))))
    for cs in CAL_SESSIONS:
        cand = np.array([i for i in c0.pool if (tr[i] == cs).sum() == 2])
        for k in KS:
            if k > len(cand):
                continue
            for draw in range(ndraw(k)):
                ims = np.sort(np.random.RandomState(7000 + 31 * k + draw + 1000 * cs).choice(cand, k, replace=False))
                rr = np.stack([np.nonzero(tr[i] == cs)[0] for i in ims])
                x2 = np.asarray(reps[ims[:, None], rr], np.float32)   # (k, 2, C, T)
                xa = x2.mean(1)
                pos = np.array([ctx[DECODERS[0]].pool_pos[int(i)] for i in ims])
                fits = {}
                for dec, cx in ctx.items():
                    c = cx.c
                    d1 = c.mS['h1'] - I.acts(c.f0, x2.reshape(-1, *x2.shape[2:]))['h1'].mean(0)
                    Xs = X.acts2(c.f0, xa, d1)['z']
                    Ys = c.pool_z[torch.as_tensor(pos, device=dev())]
                    fns = {'zero': lambda Z: Z, 'trans': R.m_trans(Xs, Ys, 5)}
                    if k >= 10:
                        fns['rigid'] = R.m_rigid(Xs, Ys, 1.0)
                        am = X.am_family(Xs, Ys)
                        fns.update({'rr4': am['rr4'], 'rr16': am['rr16'], 'full': am['full']})
                    Zq = c.zq(d1)
                    fts = {}
                    if k >= 25:
                        for nm, ho in FT_VARIANTS:
                            fts[nm] = finetune(c.f0, x2, ims, ftr, head_only=ho, seed=draw)
                    fits[dec] = (fns, Zq, fts)
                meths = list(fits[DECODERS[0]][0]) + list(fits[DECODERS[0]][2])
                for meth in meths:
                    for readout in ('img', 'anchor'):
                        if meth.startswith('ft') and readout == 'anchor':
                            continue
                        for rule in ('cos', 'csls_batch'):
                            for qk in QK:
                                nq = len(c0.Q[qk])
                                for qi in range(nq):
                                    Sd = {}
                                    for dec, cx in ctx.items():
                                        fns, Zq, fts = fits[dec]
                                        if meth in fns:
                                            z = fns[meth](Zq[qk][qi])
                                            G = cx.Gimg if readout == 'img' else cx.Ganc
                                        else:
                                            m = fts[meth]
                                            z = m(torch.as_tensor(cx.c.Q[qk][qi], device=dev()))
                                            G = m.gallery(cx.c.fte)
                                        Sd[dec] = smat(z, G, rule)
                                    record(dict(cal_session=cs, k=k, draw=draw, method=meth, readout=readout, rule=rule,
                                                query=qk, qdraw=qi), Sd)
                print(f'   cal session {cs} k={k} done ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df = df.groupby([c for c in df.columns if c not in ('qdraw', 'top1')], dropna=False).top1.mean().reset_index()
    df['target'], df['montage'] = B, mon
    df.to_csv(out, index=False)
    print(f'  target {B} {mon}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
def report(mons):
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    for mon in mons:
        f = sorted(glob.glob(os.path.join(odir(), f'cal_t*_{mon}.csv')))
        if not f:
            continue
        d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        per = d.groupby(['decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query', 'target']).top1.mean()
        T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=list(range(7))).mean()
        print(f'\n{"=" * 120}\n{mon} (people 2-10, top-1 %)\n{"=" * 120}')
        w = T.xs('within', level='decoder').xs('img', level='readout')
        print('within-person ceiling:', w.unstack('query').round(2).reset_index(drop=True).to_string())
        best_cell = T.drop('within', level='decoder', errors='ignore')
        # best readout x rule per method (mean over people; descriptive)
        for q in ('s4R4', 's4R20', 'bench'):
            for cs in (3, 4):
                sub = best_cell.xs((cs, q), level=('cal_session', 'query'))
                tab = sub.groupby(level=['decoder', 'k', 'method']).max().unstack('k')
                print(f'\nquery {q}, calibration session {cs} ({"earlier day" if cs == 3 else "same day"}): best readout x rule '
                      f'per method (descriptive), columns = k images x 2 presentations')
                print(tab.round(2).to_string())
        # prespecified primary
        p = PRIMARY
        if mon == p['montage'] and 1 in set(d.target):
            s1 = per.xs((p['decoder'], p['cal_session'], p['k'], p['query'], 1),
                        level=('decoder', 'cal_session', 'k', 'query', 'target'))
            cal = s1.drop('zero', level='method', errors='ignore')
            meth, ro, rule = cal.idxmax()
            print(f'\nSELECTED ON SUBJECT 1 (k = {p["k"]}, {p["query"]}, {p["decoder"]}): {meth} / {ro} / {rule}')
            a = per.xs((p['decoder'], p['cal_session'], p['k'], meth, ro, rule, p['query']),
                       level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
            z = per.xs((p['decoder'], p['cal_session'], p['k'], 'zero', 'anchor', 'csls_batch', p['query']),
                       level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
            R4m.pstats(a, z, f'{meth}/{ro}/{rule} minus zero-label best pipeline (k = {p["k"]}, earlier day)', primary=True)
            for k in KS:
                try:
                    ak = per.xs((p['decoder'], p['cal_session'], k, meth, ro, rule, p['query']),
                                level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
                    zk = per.xs((p['decoder'], p['cal_session'], k, 'zero', 'anchor', 'csls_batch', p['query']),
                                level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
                    R4m.pstats(ak, zk, f'  curve k = {k}: {meth} - zero')
                except KeyError:
                    pass
            for q in ('s4R20', 'bench'):
                a2 = per.xs((p['decoder'], p['cal_session'], p['k'], meth, ro, rule, q),
                            level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
                z2 = per.xs((p['decoder'], p['cal_session'], p['k'], 'zero', 'anchor', 'csls_batch', q),
                            level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
                R4m.pstats(a2, z2, f'  {meth} - zero at {q}')
            sd = per.xs((p['decoder'], 4, p['k'], meth, ro, rule, p['query']),
                        level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
            R4m.pstats(sd, a, '  same-day minus earlier-day calibration (same method)')
            for dec in ('loso', 'ens'):
                try:
                    o = per.xs((dec, p['cal_session'], p['k'], meth, ro, rule, p['query']),
                               level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
                    R4m.pstats(o, a, f'  decoder {dec} minus repaug (same method)')
                except KeyError:
                    pass
            for big in (2000, 4000):
                for q in ('s4R4', 's4R20', 'bench'):
                    try:
                        bb = T.xs((p['decoder'], 3, big, q), level=('decoder', 'cal_session', 'k', 'query'))
                        print(f'  k = {big}, {q}: best cell {bb.idxmax()} = {bb.max():.2f}')
                    except KeyError:
                        pass
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['run', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--montages', nargs='+', default=['posterior17'])
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'report':
        return report(a.montages)
    for mon in a.montages:
        for B in a.targets:
            print(f'[run] target {B} {mon}', flush=True)
            run_target(B, mon)


if __name__ == '__main__':
    main()
