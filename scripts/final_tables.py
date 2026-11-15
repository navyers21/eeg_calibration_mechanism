"""Tables for notebooks/calibration_mech.ipynb, built from the results CSVs (no training).

    python scripts/final_tables.py calib   --config C   # calibration: gains over f0 + matched-control comparisons
    python scripts/final_tables.py oracle  --config C   # linear readouts on f0's output
    python scripts/final_tables.py scaling --config C   # per-user data from 200 images to the whole pool
    python scripts/final_tables.py confirm --config C   # confirmation on new calibration images (seeds 3-5)

Each calibration arm is read from the phase where it was run with its tuned learning rate (lr is part of the adapter
name, so arms from different phases never mix). Paired t-tests across subjects, seeds averaged first.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# (label, phase, adapter name without lr). Learning rates were tuned per montage on subject 1.
ARMS = [
    ('f0 (no calibration)', 'subspace', 'f0'),
    ('Channel map', 'subspace', 'align'),
    ('SuLoRA-style map', 'subspace', 'sulora2'),
    ('SuLoRA-style map + output edit (top output PCs, k=64)', 'mech3', 'sulora2+outpca_k64'),
    ('  control: SuLoRA-style map + random output edit (k=64)', 'final', 'sulora2+outrand_k64'),
    ('Output edit only (top output PCs, k=64)', 'mech3', 'outpca_k64'),
    ('  control: random output edit (k=64)', 'final', 'outrand_k64'),
    ('Pattern map, low-level patterns (r=4, 16 numbers)', 'mech2', 'pmaplow_r4'),
    ('  control: random patterns (r=4)', 'mech2', 'pmaprand_r4'),
]
COMPARE = [
    ('SuLoRA-style map + output edit (top output PCs, k=64)', 'SuLoRA-style map'),
    ('SuLoRA-style map + output edit (top output PCs, k=64)', '  control: SuLoRA-style map + random output edit (k=64)'),
    ('Output edit only (top output PCs, k=64)', '  control: random output edit (k=64)'),
    ('Output edit only (top output PCs, k=64)', 'Channel map'),
    ('Pattern map, low-level patterns (r=4, 16 numbers)', '  control: random patterns (r=4)'),
]
BUDGETS = [25, 50, 200]


def per_subject(df):
    rows = {}
    for label, phase, ad in ARMS:
        g = df[(df.phase == phase) & (df.adapter.str.replace(r'_lr[0-9.e-]+$', '', regex=True) == ad)]
        if g.empty:
            continue
        if ad == 'f0':
            s = g.groupby(['montage', 'target']).test_top1.mean()
            for n in BUDGETS:
                for (mon, t), v in s.items():
                    rows[(label, mon, n, t)] = v
        else:
            for (mon, n, t), v in g.groupby(['montage', 'n_images', 'target']).test_top1.mean().items():
                if n in BUDGETS:
                    rows[(label, mon, n, t)] = v
    s = pd.Series(rows)
    s.index.names = ['arm', 'montage', 'n_images', 'target']
    return s


def calib(df):
    s = per_subject(df)
    order = [a[0] for a in ARMS if a[0] in s.index.get_level_values('arm')]
    missing = [a[0] for a in ARMS if a[0] not in order]
    if missing:
        print('NOT RUN YET:', missing)
    for mon in ['posterior17', 'occipital8']:
        f0 = s.xs(('f0 (no calibration)', mon), level=['arm', 'montage'])
        print(f'\n=== {mon}: test top-1 (%) and gain over f0, mean over subjects (n = subjects) ===')
        out = []
        for arm in order:
            r = {'arm': arm}
            for n in BUDGETS:
                try:
                    v = s.xs((arm, mon, n), level=['arm', 'montage', 'n_images'])
                except KeyError:
                    continue
                d = v - f0.xs(n, level='n_images').reindex(v.index)
                r[f'{n} img'] = f'{v.mean():5.1f} ({d.mean():+.1f})' if arm != 'f0 (no calibration)' else f'{v.mean():5.1f}'
                r['n'] = len(v)
            out.append(r)
        print(pd.DataFrame(out).set_index('arm').to_string())
        print(f'\n--- {mon}: paired comparisons (difference in top-1 points, p across subjects) ---')
        cmp = []
        for a, b in COMPARE:
            r = {'comparison': f'{a.strip()}  vs  {b.strip()}'}
            for n in BUDGETS:
                try:
                    x = s.xs((a, mon, n), level=['arm', 'montage', 'n_images'])
                    y = s.xs((b, mon, n), level=['arm', 'montage', 'n_images'])
                except KeyError:
                    continue
                x, y = x.align(y, join='inner')
                r[f'{n} img'] = f'{(x - y).mean():+.2f} (p={ttest_rel(x, y).pvalue:.3f})'
            cmp.append(r)
        print(pd.DataFrame(cmp).set_index('comparison').to_string())


def oracle(path):
    o = pd.read_csv(path)
    ref = o[o.readout == 'none'].groupby(['montage', 'features']).test_top1.mean().unstack()
    print('Reference (test top-1, %):')
    print(ref.round(1).to_string())
    # n >= 1000 only: before the step-0 fix (commit after b0fd4d6) the contrastive readout could not fall back to f0
    # itself, so its small-n numbers in the existing CSV are underestimates; from 1,000 images it is far above f0.
    keep = o[(o.features == 'f0_out') & o.readout.isin(['contrastive', 'centered']) & (o.n >= 1000)]
    t = keep.groupby(['montage', 'readout', 'n']).test_top1.agg(['mean', 'sem']).round(1)
    print('\nLinear readout trained on the user\'s own images, on top of f0\'s output (test top-1, %):')
    print(t.to_string())
    print('\nRead the largest n: if the readout on f0\'s output gets close to the within-subject decoder, the '
          'information is in f0 and only its mapping to CLIP space is off.')


def scaling(df):
    s = df[df.phase == 'scaling']
    if s.empty:
        print('no scaling runs yet')
        return
    t = s.groupby(['montage', 'adapter', 'n_images', 'target']).test_top1.mean()
    c = t.groupby(['montage', 'adapter', 'n_images']).agg(['mean', 'count']).round(1)
    tab = c['mean'].unstack('n_images')
    cnt = c['count'].unstack('n_images')
    for mon in tab.index.get_level_values(0).unique():
        print(f'\n=== {mon}: test top-1 (%) by number of calibration images ===')
        print(tab.loc[mon].to_string())
        print('subjects finished per cell:')
        print(cnt.loc[mon].to_string())
    print('\nf0 = no calibration; scratch = decoder trained only on the user\'s own n images. Where scratch catches '
          'the cross-subject models, starting from other people stops helping.')


def confirm(df):
    """Confirmation on new calibration images (phase 'confirm', seeds 3-5), with the criteria fixed in advance."""
    c = df[df.phase == 'confirm'].copy()
    if c.empty:
        print('no confirmation runs yet')
        return
    c['arm'] = c.adapter.str.replace(r'_lr[0-9.e-]+$', '', regex=True).map(
        {'sulora2': 'map', 'sulora2+outpca_k64': 'map+edit', 'sulora2+outrand_k64': 'map+random'})
    per = c.groupby(['montage', 'arm', 'n_images', 'target']).test_top1.mean()
    f0 = df[(df.phase == 'subspace') & (df.adapter == 'f0')].groupby(['montage', 'target']).test_top1.mean()
    res = {}
    for mon in ['posterior17', 'occipital8']:
        print(f'\n=== {mon}: test top-1 (%), new calibration images (seeds 3-5) ===')
        rows = []
        for n in BUDGETS:
            try:
                m, e, r = (per.xs((mon, arm, n), level=['montage', 'arm', 'n_images'])
                           for arm in ('map', 'map+edit', 'map+random'))
            except KeyError:
                continue
            m, e = m.align(e, join='inner')
            r = r.reindex(e.index)
            d1, d2 = e - m, e - r
            p1, p2 = ttest_rel(e, m).pvalue, ttest_rel(e, r).pvalue
            res[(mon, n)] = (d1.mean(), p1, d2.mean(), p2, len(e))
            rows.append({'images': n, 'subjects': len(e), 'f0': round(f0[mon].reindex(e.index).mean(), 1),
                         'SuLoRA-style map': round(m.mean(), 1), 'map + output edit': round(e.mean(), 1),
                         'map + random edit': round(r.mean(), 1),
                         'edit vs map': f'{d1.mean():+.2f} (p={p1:.3f}, {int((d1 > 0).sum())}/{len(d1)} subj)',
                         'edit vs random': f'{d2.mean():+.2f} (p={p2:.3f})'})
        print(pd.DataFrame(rows).to_string(index=False))
    print('\n=== Criteria fixed before running ===')
    def chk(name, keys, test):
        if not all(k in res for k in keys):
            print(f'{name}: NOT ALL RUNS FINISHED')
            return
        ok = all(test(res[k]) for k in keys)
        print(f'{name}: {"PASS" if ok else "fail"}')
    both = lambda n: [('posterior17', n), ('occipital8', n)]
    chk('1. Main: map + edit beats the SuLoRA-style map at 200 images, both montages, p < .05', both(200),
        lambda v: v[0] > 0 and v[1] < .05)
    chk('2. Small budget: gain over the map is positive at 50 images, both montages', both(50), lambda v: v[0] > 0)
    chk('3. Directions matter: map + edit beats map + random edit at 200 images, both montages, p < .05',
        both(200), lambda v: v[2] > 0 and v[3] < .05)
    print('If 1 passes, the +1.6-point gain over the SuLoRA-style map is confirmed; if not, report it as exploratory.')


CONFIRM = [('SuLoRA-style map', 'sulora2'), ('SuLoRA-style map + output edit', 'sulora2+outpca_k64'),
           ('SuLoRA-style map + random output edit', 'sulora2+outrand_k64')]


def confirm(df):
    """Pre-stated criteria: (1) at 200 images, map + output edit > SuLoRA-style map on both montages, p < .05;
    (2) at 50 images, the gain over the map is positive on both montages (p reported); (3) at 200 images,
    map + output edit > map + random output edit on both montages, p < .05."""
    c = df[df.phase == 'confirm'].copy()
    if c.empty:
        print('no confirmation runs yet: run the confirm phase first')
        return
    c['arm'] = c.adapter.str.replace(r'_lr[0-9.e-]+$', '', regex=True)
    per = c.groupby(['montage', 'arm', 'n_images', 'target']).test_top1.mean()
    f0 = df[(df.phase == 'subspace') & (df.adapter == 'f0')].groupby(['montage', 'target']).test_top1.mean()
    verdict = {}
    for mon in ['posterior17', 'occipital8']:
        if mon not in per.index.get_level_values('montage'):
            continue
        print(f'\n=== {mon}: NEW calibration images (seeds 3-5), test top-1 % (gain over f0) ===')
        rows = []
        for label, arm in CONFIRM:
            r = {'arm': label}
            for n in BUDGETS:
                try:
                    v = per.loc[(mon, arm, n)]
                except KeyError:
                    continue
                r[f'{n} img'] = f'{v.mean():5.1f} ({(v - f0.loc[mon].reindex(v.index)).mean():+.1f})'
                r['subjects'] = len(v)
            rows.append(r)
        print(pd.DataFrame(rows).set_index('arm').to_string())
        print(f'--- {mon}: paired comparisons ---')
        for a, b, key in [('sulora2+outpca_k64', 'sulora2', 'vs map'),
                          ('sulora2+outpca_k64', 'sulora2+outrand_k64', 'vs random edit')]:
            line = []
            for n in BUDGETS:
                try:
                    x, y = per.loc[(mon, a, n)].align(per.loc[(mon, b, n)], join='inner')
                except KeyError:
                    continue
                d, p = (x - y).mean(), ttest_rel(x, y).pvalue
                verdict[(mon, key, n)] = (d, p)
                line.append(f'{n} img: {d:+.2f} (p={p:.3f}, {int(((x - y) > 0).sum())}/{len(x)} subjects)')
            print(f'  map + output edit {key:15s}', ' | '.join(line))
    mons = ['posterior17', 'occipital8']
    ok = lambda key, n, need_p: all((m, key, n) in verdict and verdict[(m, key, n)][0] > 0 and
                                    (not need_p or verdict[(m, key, n)][1] < 0.05) for m in mons)
    print('\nPre-stated criteria')
    print(f'  1. 200 img, beats SuLoRA-style map on both montages, p < .05:      {"PASS" if ok("vs map", 200, True) else "fail"}')
    print(f'  2. 50 img, positive gain over the map on both montages:           {"PASS" if ok("vs map", 50, False) else "fail"}')
    print(f'  3. 200 img, beats the random output edit on both montages, p < .05: {"PASS" if ok("vs random edit", 200, True) else "fail"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['calib', 'oracle', 'scaling', 'confirm'])
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg = json.load(open(a.config))
    pd.set_option('display.width', 250, 'display.max_columns', 20, 'display.max_colwidth', 80)
    if a.cmd == 'oracle':
        oracle(os.path.join(os.path.dirname(cfg['results_csv']), 'oracle_readout2.csv'))
        return
    df = pd.read_csv(cfg['results_csv'])
    df['n_images'] = df.n_images.astype(int)
    {'calib': calib, 'scaling': scaling, 'confirm': confirm}[a.cmd](df)


if __name__ == '__main__':
    main()
