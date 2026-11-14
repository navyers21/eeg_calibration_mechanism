"""Tables for notebooks/final.ipynb, built from the results CSVs (no training).

    python scripts/final_tables.py calib   --config C   # calibration: gains over f0 + matched-control comparisons
    python scripts/final_tables.py oracle  --config C   # linear readouts on f0's output
    python scripts/final_tables.py scaling --config C   # per-user data from 200 images to the whole pool

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['calib', 'oracle', 'scaling'])
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg = json.load(open(a.config))
    pd.set_option('display.width', 250, 'display.max_columns', 20, 'display.max_colwidth', 80)
    if a.cmd == 'oracle':
        oracle(os.path.join(os.path.dirname(cfg['results_csv']), 'oracle_readout2.csv'))
        return
    df = pd.read_csv(cfg['results_csv'])
    df['n_images'] = df.n_images.astype(int)
    calib(df) if a.cmd == 'calib' else scaling(df)


if __name__ == '__main__':
    main()
