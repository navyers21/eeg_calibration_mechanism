"""Where is low-level visual information lost across people?

    python scripts/lowlevel.py run    --config C [--targets 2 3 ...] [--montages posterior17 occipital8]
    python scripts/lowlevel.py report --config C

Known so far: a decoder trained on the person (within-subject) reads low-level image properties from that person's
EEG much better than the cross-subject decoder f0 does. Since the within-subject decoder's layer 1 is a linear map of
the raw EEG, the information IS in the person's EEG. The question is why f0's layer 1, trained on other people,
misses it. Four analyses, all scored on the target person's 200 held-out concepts (2,000 trials, concepts never used
to train any probe or model):

  A. time    Raw-EEG probe per 40 ms window (step 20 ms). 'own': probe trained on the person's own EEG (their
             training images); 'cross': trained on the other 9 people's EEG of the same images. own high + cross low
             in a window = the low-level signal there differs between people.
  B. chan    Same, one electrode at a time (all time points).
  C. occl    Occlusion: replace one time window (or one electrode) of the held-out EEG by its average over trials,
             and measure how much the low-level readout of layer 1 drops, for f0 and for the within-subject decoder
             (readout trained on the person's unoccluded training trials). Shows where each model's layer 1 looks.
  D. trans   Transfer probe on layer 1 and the output: train on the 9 other people's activations (f0) vs on the
             person's own, test on the person. Fails across people but works on their own = shifted, not missing.

Scores: R^2 of a linear (ridge) readout of the 6 low-level properties, averaged over properties (per property too).
Time axis: UBP's preprocessed data is 250 Hz and the model uses the first 250 samples (1 s); we label samples as ms
from epoch start (sample x 4). Caveat: the EEG is whitened across all 63 channels before the montage is selected, so
single-electrode results (B, C by channel) are approximate.
Outputs: <dir of results_csv>/lowlevel/t<NN>_<montage>.csv (one row per analysis x condition), resumable.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]

PROPS = ['luminance', 'contrast', 'saturation', 'colorfulness', 'high_sf_ratio', 'edge_density']
WIN, STEP, MS = 10, 5, 4          # 40 ms windows, 20 ms steps, 4 ms per sample
N_TRAIN = 8000                    # training trials per probe (own and cross use the same images)


def readout(Xtr, Ytr, Xte, Yte, max_dim=256):
    """Ridge readout fitted on train (train-set standardization, PCA if wider than max_dim).
    Returns (R^2 per property on the test set, predict function for new inputs)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    pca = PCA(max_dim, random_state=0).fit((Xtr - mu) / sd) if Xtr.shape[1] > max_dim else None
    f = lambda X: pca.transform((X - mu) / sd) if pca is not None else (X - mu) / sd
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(-1, 5, 13), alpha_per_target=True).fit(f(Xtr), (Ytr - ym) / ys)
    predict = lambda X: m.predict(f(X)) * ys + ym
    return r2_of(Yte, predict(Xte)), predict


def r2_of(Y, pred):
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def score(r2):
    return dict(r2=float(np.mean(r2)), **{f'r2_{p}': float(v) for p, v in zip(PROPS, r2)})


@torch.no_grad()
def acts(model, X, layer, bs=512):
    """Layer-1 output (model[0]) or final output of a UBP EEGProject decoder, (n, 1024)."""
    from eegalign import ubp as U
    dev, out, store = U.ENV['device'], [], []
    h = model.model[0].register_forward_hook(lambda m, i, o: store.append(o.float().cpu())) if layer == 'layer1' \
        else None
    for i in range(0, len(X), bs):
        z = model(torch.as_tensor(np.asarray(X[i:i + bs], np.float32)).to(dev)).float().cpu()
        out.append(store.pop() if h else z)
    if h:
        h.remove()
    return torch.cat(out).numpy()


def run_one(t, mon, out_csv):
    from eegalign import props
    from eegalign.data import Target
    from eegalign.source import ensure_within
    from eegalign import ubp as U
    T = Target(t, mon, train_source=False)
    rng = np.random.RandomState(0)
    # held-out concepts: the evaluation set for everything
    Xh = np.concatenate(T.held_x).astype(np.float32)                       # (2000, C, 250)
    Yh = props.props_for([p for ps in T.held_paths for p in ps]).astype(np.float64)
    # own vs cross training sets on the SAME images: the other people's trials T.reh_x index pool images by reh_t
    pick = rng.choice(len(T.reh_x), min(N_TRAIN, len(T.reh_x)), replace=False)
    pick = pick[np.abs(T.reh_x[pick].astype(np.float32)).reshape(len(pick), -1).sum(1) > 0]   # drop empty rows
    pick = pick[np.unique(T.reh_t[pick], return_index=True)[1]]   # one trial per image: no duplicated own rows
    Xc = T.reh_x[pick].astype(np.float32)                                   # other people's EEG
    Xo = T.x[T.reh_t[pick]].astype(np.float32)                              # this person's EEG, same images
    Ytr = props.props_for(T.P[T.reh_t[pick]]).astype(np.float64)
    chans = U.MONTAGES[mon]
    rows = []
    add = lambda **k: rows.append(dict(target=t, montage=mon, **k))
    # A. time-resolved raw-EEG probe
    starts = list(range(0, 250 - WIN + 1, STEP))
    for s in starts:
        sl = slice(s, s + WIN)
        for kind, Xtr in (('own', Xo), ('cross', Xc)):
            r2, *_ = readout(Xtr[:, :, sl].reshape(len(Xtr), -1), Ytr, Xh[:, :, sl].reshape(len(Xh), -1), Yh)
            add(analysis='time', train=kind, ms=(s + WIN / 2) * MS, channel='', **score(r2))
    for kind, Xtr in (('own', Xo), ('cross', Xc)):          # whole epoch, all channels: the reference
        r2, *_ = readout(Xtr.reshape(len(Xtr), -1), Ytr, Xh.reshape(len(Xh), -1), Yh)
        add(analysis='all', train=kind, ms=np.nan, channel='', **score(r2))
    print(f'  t{t} {mon}: time done', flush=True)
    # B. per electrode
    for ci, ch in enumerate(chans):
        for kind, Xtr in (('own', Xo), ('cross', Xc)):
            r2, *_ = readout(Xtr[:, ci, :], Ytr, Xh[:, ci, :], Yh)
            add(analysis='chan', train=kind, ms=np.nan, channel=ch, **score(r2))
    print(f'  t{t} {mon}: electrodes done', flush=True)
    # C. occlusion on layer 1 of f0 and of the within-subject decoder (readout trained on own unoccluded trials)
    models = {'f0': T.f0, 'within': ensure_within(t, mon, train=False)}
    mean_h = Xh.mean(0, keepdims=True)
    for name, m in models.items():
        # readout trained on this person's unoccluded training trials, applied to (occluded) held-out trials
        r2_base, predict = readout(acts(m, Xo, 'layer1'), Ytr, acts(m, Xh, 'layer1'), Yh, max_dim=10 ** 6)
        add(analysis='occl', train=name, ms=np.nan, channel='none', **score(r2_base))
        for s in starts[::2]:                                # 40 ms steps for occlusion
            Xm = Xh.copy()
            Xm[:, :, s:s + WIN] = mean_h[:, :, s:s + WIN]
            add(analysis='occl_time', train=name, ms=(s + WIN / 2) * MS, channel='',
                **score(r2_base - r2_of(Yh, predict(acts(m, Xm, 'layer1')))))
        for ci, ch in enumerate(chans):
            Xm = Xh.copy()
            Xm[:, ci, :] = mean_h[:, ci, :]
            add(analysis='occl_chan', train=name, ms=np.nan, channel=ch,
                **score(r2_base - r2_of(Yh, predict(acts(m, Xm, 'layer1')))))
    print(f'  t{t} {mon}: occlusion done', flush=True)
    # D. transfer probe on f0's layer 1 and output: trained on other people's activations vs the person's own
    for layer in ('layer1', 'output'):
        Hh = acts(T.f0, Xh, layer)
        for kind, Xtr in (('own', Xo), ('cross', Xc)):
            r2, *_ = readout(acts(T.f0, Xtr, layer), Ytr, Hh, Yh)
            add(analysis='trans', train=kind, ms=np.nan, channel=layer, **score(r2))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'  saved {out_csv}', flush=True)


CURVE_N = [25, 50, 100, 200, 500, 1000, 2000, 0]      # 0 = all available


def curve_one(t, mon, out_csv):
    """E. How many labeled images does a person-specific low-level readout on f0's layer 1 need?"""
    from eegalign import props
    from eegalign.data import Target
    from eegalign.source import ensure_within
    T = Target(t, mon, train_source=False)
    rng = np.random.RandomState(0)
    Xh = np.concatenate(T.held_x).astype(np.float32)
    Yh = props.props_for([p for ps in T.held_paths for p in ps]).astype(np.float64)
    pick = rng.choice(len(T.reh_x), min(N_TRAIN, len(T.reh_x)), replace=False)
    pick = pick[np.abs(T.reh_x[pick].astype(np.float32)).reshape(len(pick), -1).sum(1) > 0]
    pick = pick[np.unique(T.reh_t[pick], return_index=True)[1]]
    Xo, Xc = T.x[T.reh_t[pick]].astype(np.float32), T.reh_x[pick].astype(np.float32)
    Ytr = props.props_for(T.P[T.reh_t[pick]]).astype(np.float64)
    Ho, Hh = acts(T.f0, Xo, 'layer1'), acts(T.f0, Xh, 'layer1')
    rows = []
    for n in CURVE_N:
        for draw in range(3 if n else 1):
            ix = np.random.RandomState(draw).permutation(len(Ho))[:n] if n else np.arange(len(Ho))
            r2, _ = readout(Ho[ix], Ytr[ix], Hh, Yh, max_dim=min(256, len(ix) - 1))
            rows.append(dict(target=t, montage=mon, readout='f0_own', n=len(ix), draw=draw, **score(r2)))
    r2, _ = readout(acts(T.f0, Xc, 'layer1'), Ytr, Hh, Yh)
    rows.append(dict(target=t, montage=mon, readout='f0_cross', n=len(Xc), draw=0, **score(r2)))
    W = ensure_within(t, mon, train=False)
    r2, _ = readout(acts(W, Xo, 'layer1'), Ytr, acts(W, Xh, 'layer1'), Yh)
    rows.append(dict(target=t, montage=mon, readout='within_own', n=len(Xo), draw=0, **score(r2)))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'  saved {out_csv}', flush=True)


def curve_report(base):
    import glob
    files = sorted(glob.glob(os.path.join(base, 'curve_t*_*.csv')))
    if not files:
        raise SystemExit('no learning-curve results yet: run the curve command first')
    d = pd.concat([pd.read_csv(f) for f in files])
    for mon, g in d.groupby('montage'):
        o = g[g.readout == 'f0_own']
        per = o.groupby(['target', 'n']).r2.mean().reset_index()
        per['n'] = np.where(per.n > 2000, 'all', per.n.astype(str))
        tab = per.groupby('n', sort=False).r2.agg(['mean', 'sem']).round(3)
        ref = g[g.readout != 'f0_own'].groupby('readout').r2.mean().round(3).to_dict()
        print(f'\n=== {mon} ({g.target.nunique()} subjects): low-level R^2 of a readout on f0 layer 1, trained on n of '
              f'the person\'s own images ===')
        print(tab.to_string())
        print(f'reference: f0 layer 1 with a readout trained on the other 9 people = {ref.get("f0_cross")}, '
              f'within-subject decoder layer 1 (own, all images) = {ref.get("within_own")}')


# ---- calibration aimed at the finding (src/eegalign/mechanism3.py), run through calibrate.py --------------------
LL_TREAT = ['llwin_r8', 'llwin_r8_aux', 'occmap', 'sulora2+llwin_r8_aux', 'sulora2+llwin_r8_aux+outpca_k64']
LL_CTRL = {'llwinctl_r8': 'llwin_r8', 'parmap': 'occmap',
           'sulora2+llwinctl_r8_aux+outpca_k64': 'sulora2+llwin_r8_aux+outpca_k64'}   # control -> its treatment's lr
LL_LRS = ['1e-4', '3e-4', '1e-3', '3e-3']
POSTERIOR_ONLY = ('occmap', 'parmap')


def _call(args):
    import subprocess
    cmd = [sys.executable, os.path.join(ROOT, 'calibrate.py'), *args]
    print('$', ' '.join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


def ll_tune(config, montages):
    for mon in montages:
        fams = [f for f in LL_TREAT if mon == 'posterior17' or f not in POSTERIOR_ONLY]
        _call(['--config', config, '--phase', 'll_tune', '--target', '1', '--montage', mon, '--adapters', 'f0',
               *[f'{f}_lr{lr}' for f in fams for lr in LL_LRS], '--budgets', '50', '--seeds', '0', '1', '2'])


def ll_best(csv, mon):
    df = pd.read_csv(csv)
    df = df[(df.phase == 'll_tune') & (df.montage == mon)]
    best = {}
    for f in LL_TREAT:
        if mon != 'posterior17' and f in POSTERIOR_ONLY:
            continue
        g = df[df.adapter.str.startswith(f + '_lr') & ~df.adapter.str[len(f) + 3:].str.contains('_')]
        g = g[g.adapter.str.replace(r'_lr[0-9.e-]+$', '', regex=True) == f].groupby('adapter').held_top1.mean()
        if len(g) < len(LL_LRS):
            raise SystemExit(f'll_tune not finished for {f} {mon}: run the tuning cell first')
        best[f] = g.idxmax().rsplit('_lr', 1)[1]
    for c, f in LL_CTRL.items():
        if f in best:
            best[c] = best[f]
    return best


def ll_calib(config, montages, targets):
    csv = json.load(open(config))['results_csv']
    for mon in montages:
        best = ll_best(csv, mon)
        print(f'{mon}: learning rates chosen on subject 1 (held-out top-1):', best, flush=True)
        for t in targets:
            _call(['--config', config, '--phase', 'llcal', '--target', str(t), '--montage', mon,
                   '--adapters', *[f'{f}_lr{lr}' for f, lr in best.items()], '--budgets', '25', '50', '200',
                   '--seeds', '0', '1', '2'])


LL_ARMS = [('f0', 'subspace', 'f0'), ('Channel map', 'subspace', 'align'), ('SuLoRA-style map', 'subspace', 'sulora2'),
           ('SuLoRA-style map + output edit (confirmed best)', 'mech3', 'sulora2+outpca_k64'),
           ('Window adapter (occipital, 80-260 ms)', 'llcal', 'llwin_r8'),
           ('  control: window adapter at 600-780 ms', 'llcal', 'llwinctl_r8'),
           ('Window adapter + low-level aux loss', 'llcal', 'llwin_r8_aux'),
           ('Occipital-only map', 'llcal', 'occmap'), ('  control: parietal-only map', 'llcal', 'parmap'),
           ('SuLoRA-style map + window adapter + aux', 'llcal', 'sulora2+llwin_r8_aux'),
           ('Best + window adapter + aux', 'llcal', 'sulora2+llwin_r8_aux+outpca_k64'),
           ('  control: best + 600-780 ms window + aux', 'llcal', 'sulora2+llwinctl_r8_aux+outpca_k64')]
LL_CMP = [('Window adapter (occipital, 80-260 ms)', '  control: window adapter at 600-780 ms'),
          ('Window adapter + low-level aux loss', 'Window adapter (occipital, 80-260 ms)'),
          ('Occipital-only map', '  control: parietal-only map'), ('Occipital-only map', 'Channel map'),
          ('SuLoRA-style map + window adapter + aux', 'SuLoRA-style map'),
          ('Best + window adapter + aux', 'SuLoRA-style map + output edit (confirmed best)'),
          ('Best + window adapter + aux', '  control: best + 600-780 ms window + aux')]


def ll_report(csv):
    from scipy.stats import ttest_rel
    df = pd.read_csv(csv)
    df['arm'] = df.adapter.str.replace(r'_lr[0-9.e-]+$', '', regex=True)
    pd.set_option('display.width', 250, 'display.max_colwidth', 70)
    for mon in ['posterior17', 'occipital8']:
        per = {}
        for label, phase, arm in LL_ARMS:
            g = df[(df.phase == phase) & (df.arm == arm) & (df.montage == mon)]
            if g.empty:
                continue
            s = g.groupby(['n_images', 'target']).test_top1.mean()
            per[label] = {n: s.xs(0, level='n_images') for n in (25, 50, 200)} if arm == 'f0' else \
                {n: s.xs(n, level='n_images') for n in (25, 50, 200) if n in s.index.get_level_values(0)}
        if 'f0' not in per:
            continue
        print(f'\n=== {mon}: test top-1 % (gain over f0), mean over subjects ===')
        rows = []
        for label, d in per.items():
            r = {'arm': label}
            for n, v in d.items():
                r[f'{n} img'] = f'{v.mean():5.1f}' + ('' if label == 'f0' else
                                                     f' ({(v - per["f0"][n].reindex(v.index)).mean():+.1f})')
                r['n'] = len(v)
            rows.append(r)
        print(pd.DataFrame(rows).set_index('arm').to_string())
        print(f'--- {mon}: paired comparisons (difference, p across subjects, subjects better) ---')
        for a, b in LL_CMP:
            if a not in per or b not in per:
                continue
            out = []
            for n in (25, 50, 200):
                if n in per[a] and n in per[b]:
                    x, y = per[a][n].align(per[b][n], join='inner')
                    out.append(f'{n}: {(x - y).mean():+.2f} (p={ttest_rel(x, y).pvalue:.3f}, '
                               f'{int(((x - y) > 0).sum())}/{len(x)})')
            print(f'  {a.strip()}  vs  {b.strip()}:  ' + ' | '.join(out))


def report(base):
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(base, 't*_*.csv')))
    if not files:
        raise SystemExit('no lowlevel results yet: run first')
    d = pd.concat([pd.read_csv(f) for f in files])
    pd.set_option('display.width', 200, 'display.max_columns', 20)
    for mon, g in d.groupby('montage'):
        n = g.target.nunique()
        print(f'\n================ {mon} ({n} subjects) ================')
        a = g[g.analysis == 'all'].pivot_table(index='target', columns='train', values='r2')
        print(f'Whole epoch, all electrodes, low-level R^2: own {a["own"].mean():.3f}, cross {a["cross"].mean():.3f} '
              f'(cross / own = {(a["cross"] / a["own"]).mean():.2f})')
        t = g[g.analysis == 'time'].pivot_table(index='ms', columns='train', values='r2', aggfunc='mean')
        pk = t['own'].idxmax()
        print(f'\nA. Time course (mean over subjects): own peaks at {pk:.0f} ms (R^2 {t["own"].max():.3f}); '
              f'cross at {t["cross"].idxmax():.0f} ms (R^2 {t["cross"].max():.3f})')
        tp = g[g.analysis == 'time'].groupby(['target', 'train']).apply(lambda x: x.loc[x.r2.idxmax(), 'ms'])
        tp = tp.unstack()
        print('   own-probe peak per subject (ms):', {int(k): int(v) for k, v in tp['own'].items()},
              f'| spread (SD) {tp["own"].std():.0f} ms')
        print(t.round(3).iloc[::2].to_string())
        c = g[g.analysis == 'chan'].pivot_table(index='channel', columns='train', values='r2', aggfunc='mean')
        print('\nB. Per electrode (mean R^2):')
        print(c.sort_values('own', ascending=False).round(3).to_string())
        o = g[g.analysis == 'occl'].pivot_table(index='target', columns='train', values='r2')
        print(f'\nC. Layer-1 low-level readout, unoccluded: within {o["within"].mean():.3f}, f0 {o["f0"].mean():.3f}')
        ot = g[g.analysis == 'occl_time'].pivot_table(index='ms', columns='train', values='r2', aggfunc='mean')
        print('   R^2 lost when a 40 ms window is blanked (bigger = the model relies on it):')
        print(ot.round(4).to_string())
        oc = g[g.analysis == 'occl_chan'].pivot_table(index='channel', columns='train', values='r2', aggfunc='mean')
        print('   R^2 lost when one electrode is blanked:')
        print(oc.sort_values('within', ascending=False).round(4).to_string())
        tr = g[g.analysis == 'trans'].pivot_table(index='channel', columns='train', values='r2', aggfunc='mean')
        print('\nD. Transfer probe on f0 (rows: layer; own = trained on this person, cross = on the other 9):')
        print(tr.round(3).to_string())
        # figures
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        for kind, col in (('own', 'C0'), ('cross', 'C1')):
            s = g[(g.analysis == 'time') & (g.train == kind)].groupby('ms').r2
            ax[0].plot(s.mean().index, s.mean(), col, label=f'trained on {kind} EEG')
            ax[0].fill_between(s.mean().index, s.mean() - s.sem(), s.mean() + s.sem(), color=col, alpha=.2)
        ax[0].set(title='A. Low-level information in raw EEG over time', xlabel='ms from epoch start',
                  ylabel='R$^2$ (mean of 6 properties)')
        ax[0].legend()
        c.sort_values('own', ascending=False).plot.bar(ax=ax[1], title='B. Per electrode', ylabel='R$^2$')
        for kind, col in (('within', 'C2'), ('f0', 'C3')):
            s = g[(g.analysis == 'occl_time') & (g.train == kind)].groupby('ms').r2
            ax[2].plot(s.mean().index, s.mean(), col, label=f'{kind} layer 1')
        ax[2].set(title='C. Layer-1 R$^2$ lost when a 40 ms window is blanked', xlabel='ms from epoch start',
                  ylabel='R$^2$ lost')
        ax[2].legend()
        plt.tight_layout()
        png = os.path.join(base, f'lowlevel_{mon}.png')
        plt.savefig(png, dpi=120)
        plt.show()
        print('figure ->', png)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['run', 'report', 'curve', 'curve_report', 'tune', 'calib', 'calib_report'])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument('--montages', nargs='+', default=['posterior17', 'occipital8'])
    a = p.parse_args()
    from eegalign import env
    cfg = env.load(a.config)
    base = os.path.join(os.path.dirname(cfg['results_csv']), 'lowlevel')
    os.makedirs(base, exist_ok=True)
    if a.cmd == 'report':
        return report(base)
    if a.cmd == 'curve_report':
        return curve_report(base)
    if a.cmd == 'tune':
        return ll_tune(a.config, a.montages)
    if a.cmd == 'calib':
        return ll_calib(a.config, a.montages, a.targets)
    if a.cmd == 'calib_report':
        return ll_report(cfg['results_csv'])
    if a.cmd == 'curve':
        for mon in a.montages:
            for t in a.targets:
                out = os.path.join(base, f'curve_t{t:02d}_{mon}.csv')
                if os.path.exists(out):
                    print(f'[curve] t{t} {mon}: done, skipping', flush=True)
                    continue
                print(f'[curve] t{t} {mon}: running', flush=True)
                curve_one(t, mon, out)
        return
    for mon in a.montages:
        for t in a.targets:
            out = os.path.join(base, f't{t:02d}_{mon}.csv')
            if os.path.exists(out):
                print(f'[lowlevel] t{t} {mon}: done, skipping', flush=True)
                continue
            print(f'[lowlevel] t{t} {mon}: running', flush=True)
            run_one(t, mon, out)


if __name__ == '__main__':
    main()
