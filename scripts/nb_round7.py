"""Round 7: close out the checkpoint. Confirm the operating point on a fresh decoder, and combine fine-tuning with
the population-response readout.

    python scripts/nb_round7.py train_seed --config C                  # repaug f0, seed 2026, posterior-17 (~9 min)
    python scripts/nb_round7.py confirm    --config C                  # locked k = 500 analysis on the seed-2026 decoders
    python scripts/nb_round7.py combo      --config C                  # fine-tuning + population readout, k = 500 / 2000 / 4000
    python scripts/nb_round7.py report     --config C

Confirm (locked before seeing it). Round 6's prespecified k = 200 primary missed; from k = 500 every person
improved (posterior-17 +4.23, occipital-8 +3.49, both 9/9). k = 500 (1,000 presentations, 3.3 min) is locked as the
operating point and re-tested on a NEW decoder: repaug retrained with seed 2026 (new initialization, new
presentation sampling, new smoothing draws), so the decoder, its anchors, the zero-label pipeline and the calibration
map are all refitted. Calibration images and queries use round 6's draws (same audited sessions). Method locked from
round 6 (rigid alignment + predicted-response readout + batch CSLS, selected on person 1).
  PRIMARY: posterior-17, s4R4, earlier-day calibration, people 2-10, Wilcoxon: rigid - zero-label pipeline at k = 500.
  Secondary: same at s4R20 and bench; same-day calibration; fine-tuning at k = 500.

Combo (exploratory). Round 6: rigid alignment wins up to ~2,000 images, fine-tuning wins above (18.3% at 4,000).
  ft+rigid   fine-tune, then rigidly align the fine-tuned outputs to the population responses (fitted on the same
             calibration images), read out with predicted responses
  ft+trans   same with a translation
  ft_anchor  fine-tune with the image loss plus a pull toward the population responses of the calibration images
             (1 - cosine, weight 1), so the output stays in the space the predicted responses live in
  ft, rigid  round 6's methods on the same draws (paired references)
Prespecified comparisons: ft+rigid - ft at k = 4000 and ft+rigid - rigid at k = 500 (s4R4, anchor readout, CSLS).
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
import nb_round6 as R6m  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
SEED2 = 2026
K_LOCK = 500
K_COMBO = [500, 2000, 4000]
QK = ['s4R1', 's4R4', 's4R20', 'bench']


def dev():
    return N.ENV['device']


def odir(sub):
    d = os.path.join(N.ENV['results_dir'], 'nb_round7', sub)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------------------------
# confirm: seed-2026 repaug decoders, round-6 pipeline at k = 500 only
# ---------------------------------------------------------------------------------------------------------------
def use_seed2():
    R5m.SEED = SEED2
    R5m.odir = lambda: odir('train')
    R6m.odir = lambda: odir('confirm')
    R6m.KS = [K_LOCK]
    R6m.DECODERS = ['repaug']
    R6m.ndraw = lambda k: 5


def train_seed(targets, mon='posterior17'):
    use_seed2()
    for B in targets:
        print(f'[train repaug seed {SEED2}] {mon} t{B}', flush=True)
        R5m.train_rep('repaug', mon, B)


def confirm(targets, mon='posterior17'):
    use_seed2()
    for B in targets:
        print(f'[confirm] target {B} {mon}', flush=True)
        R6m.run_target(B, mon)


# ---------------------------------------------------------------------------------------------------------------
# combo: fine-tuning + population readout (seed-2025 repaug, round-6 draws)
# ---------------------------------------------------------------------------------------------------------------
def finetune_anchor(f0, x2, img, ftr, Y, lam=1.0, steps=300, lr=3e-5, seed=0):
    torch.manual_seed(seed)
    m = copy.deepcopy(f0).requires_grad_(False)
    params = list(m.encoder.parameters()) + list(m.eeg_projector.parameters())
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
        z = m(xb)
        loss = N.nb_loss(z, g, m.scale()) + lam * (1 - F.cosine_similarity(z, Y[idx], dim=1).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    m.eval().requires_grad_(False)
    return m


def combo_target(B, mon='posterior17', cs=3):
    out = os.path.join(odir('combo'), f'combo_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'[combo] {B} {mon}: done')
        return
    t0 = time.time()
    ctx = R6m.Ctx('repaug', B, mon)
    c = ctx.c
    tr = c.ses['train']
    reps = R.load_reps(B, mon)
    cand = np.array([i for i in c.pool if (tr[i] == cs).sum() == 2])
    rows = []
    for k in K_COMBO:
        if k > len(cand):
            continue
        for draw in range(2 if k <= 500 else 1):
            ims = np.sort(np.random.RandomState(7000 + 31 * k + draw + 1000 * cs).choice(cand, k, replace=False))
            rr = np.stack([np.nonzero(tr[i] == cs)[0] for i in ims])
            x2 = np.asarray(reps[ims[:, None], rr], np.float32)
            xa = x2.mean(1)
            pos = torch.as_tensor(np.array([ctx.pool_pos[int(i)] for i in ims]), device=dev())
            Ys = c.pool_z[pos]
            d1 = c.mS['h1'] - I.acts(c.f0, x2.reshape(-1, *x2.shape[2:]))['h1'].mean(0)
            Xs = X.acts2(c.f0, xa, d1)['z']
            Zq0 = c.zq(d1)
            ft = R6m.finetune(c.f0, x2, ims, c.ftr, seed=draw)
            fta = finetune_anchor(c.f0, x2, ims, c.ftr, Ys, seed=draw)
            with torch.no_grad():
                Zft_cal = ft(torch.as_tensor(xa, device=dev()))
                Zfta_cal = fta(torch.as_tensor(xa, device=dev()))
            maps = {'rigid': R.m_rigid(Xs, Ys, 1.0), 'zero': lambda Z: Z,
                    'ft+rigid': R.m_rigid(Zft_cal, Ys, 1.0), 'ft+trans': R.m_trans(Zft_cal, Ys, 5),
                    'ft_anchor+rigid': R.m_rigid(Zfta_cal, Ys, 1.0)}
            Gft, Gfta = ft.gallery(c.fte), fta.gallery(c.fte)
            Ganc, Gimg = ctx.Ganc, ctx.Gimg
            for qk in QK:
                for qi, q in enumerate(c.Q[qk]):
                    with torch.no_grad():
                        qt = torch.as_tensor(q, device=dev())
                        zft, zfta = ft(qt), fta(qt)
                        z0 = Zq0[qk][qi]
                        cells = {('zero', 'anchor'): (z0, Ganc), ('rigid', 'anchor'): (maps['rigid'](z0), Ganc),
                                 ('ft', 'img'): (zft, Gft), ('ft+rigid', 'anchor'): (maps['ft+rigid'](zft), Ganc),
                                 ('ft+trans', 'anchor'): (maps['ft+trans'](zft), Ganc),
                                 ('ft_anchor', 'anchor'): (zfta, Ganc), ('ft_anchor', 'img'): (zfta, Gfta),
                                 ('ft_anchor+rigid', 'anchor'): (maps['ft_anchor+rigid'](zfta), Ganc)}
                        for (meth, ro), (z, G) in cells.items():
                            for rule in ('cos', 'csls_batch'):
                                rows.append(dict(k=k, draw=draw, method=meth, readout=ro, rule=rule, query=qk, qdraw=qi,
                                                 top1=R6m.top1(R6m.smat(z, G, rule))))
            print(f'   combo k={k} draw {draw} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df = df.groupby(['k', 'draw', 'method', 'readout', 'rule', 'query']).top1.mean().reset_index()
    df['target'], df['montage'] = B, mon
    df.to_csv(out, index=False)
    print(f'  [combo] target {B}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
def report():
    pd.set_option('display.width', 250)
    # 1. confirmatory run
    f = sorted(glob.glob(os.path.join(odir('confirm'), 'cal_t*.csv')))
    if f:
        d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        per = d.groupby(['decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query', 'target']).top1.mean()
        print(f'CONFIRMATORY RUN (repaug seed {SEED2}, k = {K_LOCK}, posterior-17, people 2-10)')

        def g(meth, ro, rule, q, cs=3):
            return per.xs(('repaug', cs, K_LOCK, meth, ro, rule, q),
                          level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
        T = per[per.index.get_level_values('target').isin(TEST)].xs('repaug', level='decoder')
        print(T.groupby(level=['cal_session', 'method', 'readout', 'rule', 'query']).mean().unstack('query').round(2).to_string())
        R4m.pstats(g('rigid', 'anchor', 'csls_batch', 's4R4'), g('zero', 'anchor', 'csls_batch', 's4R4'),
                   'rigid - zero-label pipeline, k = 500, s4R4, earlier day (locked)', primary=True)
        for q in ('s4R1', 's4R20', 'bench'):
            R4m.pstats(g('rigid', 'anchor', 'csls_batch', q), g('zero', 'anchor', 'csls_batch', q), f'  same at {q}')
        R4m.pstats(g('rigid', 'anchor', 'csls_batch', 's4R4', 4), g('zero', 'anchor', 'csls_batch', 's4R4', 4),
                   '  same-day calibration: rigid - zero')
        R4m.pstats(g('ft', 'img', 'csls_batch', 's4R4'), g('zero', 'anchor', 'csls_batch', 's4R4'), '  fine-tuning - zero (s4R4)')
        z = g('zero', 'anchor', 'csls_batch', 's4R4')
        f0p = per.xs(('repaug', 3, K_LOCK, 'zero', 'img', 'cos', 's4R4'),
                     level=('decoder', 'cal_session', 'k', 'method', 'readout', 'rule', 'query'))
        print(f'  levels (s4R4): seed-2026 decoder plain {f0p.reindex(TEST).mean():.2f}, zero-label pipeline '
              f'{z.reindex(TEST).mean():.2f}, rigid at k = 500 {g("rigid", "anchor", "csls_batch", "s4R4").reindex(TEST).mean():.2f}')
    # 2. occipital-8 at the locked k, from round 6
    f = sorted(glob.glob(os.path.join(N.ENV['results_dir'], 'nb_round6', 'cal_t*_occipital8.csv')))
    if f:
        d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        d = d[(d.decoder == 'repaug') & (d.cal_session == 3) & (d.k == K_LOCK) & (d['query'] == 's4R4')]
        a = d[(d.method == 'rigid') & (d.readout == 'anchor') & (d.rule == 'csls_batch')].groupby('target').top1.mean()
        z = d[(d.method == 'zero') & (d.readout == 'anchor') & (d.rule == 'csls_batch')].groupby('target').top1.mean()
        print('\nOCCIPITAL-8 REPLICATION (round 6 data, repaug seed 2025, k = 500, s4R4)')
        R4m.pstats(a, z, 'rigid - zero-label pipeline')
    # 3. combo
    f = sorted(glob.glob(os.path.join(odir('combo'), 'combo_t*.csv')))
    if f:
        d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        per = d.groupby(['k', 'method', 'readout', 'rule', 'query', 'target']).top1.mean()
        T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3, 4]).mean()
        print('\nCOMBO (fine-tuning + population readout; batch CSLS; posterior-17, repaug seed 2025, earlier day)')
        print(T.xs('csls_batch', level='rule').unstack('query')[QK].unstack('k').round(2).to_string())

        def g(k, meth, ro, q='s4R4'):
            return per.xs((k, meth, ro, 'csls_batch', q), level=('k', 'method', 'readout', 'rule', 'query'))
        R4m.pstats(g(4000, 'ft+rigid', 'anchor'), g(4000, 'ft', 'img'), 'ft+rigid - ft, k = 4000, s4R4 (prespecified)')
        R4m.pstats(g(500, 'ft+rigid', 'anchor'), g(500, 'rigid', 'anchor'), 'ft+rigid - rigid, k = 500, s4R4 (prespecified)')
        for k in K_COMBO:
            for meth, ro in (('ft_anchor', 'anchor'), ('ft_anchor+rigid', 'anchor'), ('ft+trans', 'anchor')):
                try:
                    R4m.pstats(g(k, meth, ro), g(k, 'ft', 'img'), f'  {meth} - ft, k = {k}, s4R4')
                except KeyError:
                    pass
        for q in ('s4R20', 'bench'):
            R4m.pstats(g(4000, 'ft+rigid', 'anchor', q), g(4000, 'ft', 'img', q), f'  ft+rigid - ft, k = 4000, {q}')
    # 4. smooth4 control (round 5)
    f = sorted(glob.glob(os.path.join(N.ENV['results_dir'], 'nb_round5', 'e_*_posterior17.csv')))
    if f:
        e = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        e = e[(e.exp == 'M2') & (e.offset == 'none') & (e.gallery == 'anchor') & (e['query'] == 's4R4')]
        if 'smooth4' in set(e.kind):
            print('\nSMOOTH4 CONTROL (round 5; anchors, no offset, s4R4)')
            p = e.groupby(['kind', 'rule', 'target']).top1.mean()
            for rule in ('cos', 'csls_batch'):
                R4m.pstats(p.loc[('repaug', rule)], p.loc[('smooth4', rule)], f'repaug - smooth4 ({rule})')
                R4m.pstats(p.loc[('smooth4', rule)], p.loc[('loso', rule)], f'smooth4 - loso ({rule})')
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(''), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train_seed', 'confirm', 'combo', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    p.add_argument('--montage', default='posterior17')
    p.add_argument('--epochs', type=int, default=50)
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'train_seed':
        if a.epochs != 50:
            _tr = R5m.train_rep
            R5m.train_rep = lambda kind, mon, B: _tr(kind, mon, B, a.epochs)
        return train_seed(a.targets, a.montage)
    if a.cmd == 'confirm':
        return confirm(a.targets, a.montage)
    if a.cmd == 'combo':
        for B in a.targets:
            print(f'[combo] target {B}', flush=True)
            combo_target(B, a.montage)
        return
    report()


if __name__ == '__main__':
    main()
