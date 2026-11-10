"""What breaks across people, and what does calibration restore?

    python gap_analysis.py --config C --montage posterior17 [--eval_set held] [--n_images 50]

Reads the run_interp outputs for every finished target and reports, per measure, mean over targets with SE:
  * retrieval (baseline, score half): top-1, category accuracy (semantic), within-category top-1 (fine detail)
  * probes at the first layer: mean R^2 of the 6 low-level properties, semantic-cluster accuracy
  * reliance: accuracy lost by LEACE beyond norm-matched random erasure, low-level vs semantic
For each: lost = within - f0, restored = mode - f0, fraction restored = restored / lost (pooled over targets).
Decision: a subspace story needs the fraction restored to differ clearly between low-level and semantic measures.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_run(d):
    out = {}
    s = json.load(open(os.path.join(d, 'summary.json')))['baseline']
    out.update(top1=s['top1'], category=s['cluster_acc'], fine=s['within_cluster_top1'])
    pr = os.path.join(d, 'probes.csv')
    if os.path.exists(pr):
        p = pd.read_csv(pr)
        p = p[p.layer == 'input_linear']
        out['probe_low_r2'] = p[p.kind == 'low'].score.mean()
        out['probe_sem_acc'] = p[p.kind == 'semantic'].score.mean()
    iv = pd.read_csv(os.path.join(d, 'interventions.csv'))
    iv = iv[iv.layer == 'input_linear']
    iv['kind'] = np.where(iv.variant == 'leace', 'leace', 'random')
    for c, lab in (('low_level_all', 'low'), ('semantic_cluster', 'sem')):
        g = iv[iv.concept == c].groupby('kind')[['d_cluster_acc', 'd_within_cluster_top1']].mean()
        if len(g) == 2:
            out[f'rely_{lab}_category'] = g.loc['leace', 'd_cluster_acc'] - g.loc['random', 'd_cluster_acc']
            out[f'rely_{lab}_fine'] = g.loc['leace', 'd_within_cluster_top1'] - g.loc['random', 'd_within_cluster_top1']
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--montage', default='posterior17')
    p.add_argument('--eval_set', default='held')
    p.add_argument('--n_images', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='')
    a = p.parse_args()
    cfg = json.load(open(a.config))
    base = os.path.join(os.path.dirname(cfg['results_csv']), 'interp', a.eval_set, a.montage)
    rows = []
    for tdir in sorted(glob.glob(os.path.join(base, 't*'))):
        for d in glob.glob(os.path.join(tdir, '*')):
            if os.path.exists(os.path.join(d, 'interventions.csv')):
                mode = os.path.basename(d)
                if mode not in ('within', 'f0') and f'_n{a.n_images}_s{a.seed}' not in mode:
                    continue
                rows.append(dict(target=int(os.path.basename(tdir)[1:]), mode=mode.split('_n')[0], **load_run(d)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f'no finished interp runs under {base}')
    measures = [c for c in df.columns if c not in ('target', 'mode')]
    wide = df.pivot_table(index='target', columns='mode', values=measures)
    print(f'{a.montage} | {a.eval_set} | n_images {a.n_images} | targets {sorted(df.target.unique())}\n')
    print('Mean over targets:')
    print(df.groupby('mode')[measures].mean().T.round(4).to_string())
    res = []
    for m in measures:
        if ('within' not in wide[m]) or ('f0' not in wide[m]):
            continue
        lost = (wide[m]['within'] - wide[m]['f0']).dropna()
        for mode in [c for c in wide[m].columns if c not in ('within', 'f0')]:
            rest = (wide[m][mode] - wide[m]['f0']).dropna()
            k = rest.index.intersection(lost.index)
            res.append(dict(measure=m, mode=mode, targets=len(k), lost=lost[k].mean(), restored=rest[k].mean(),
                            frac_restored=rest[k].sum() / lost[k].sum() if lost[k].sum() != 0 else np.nan,
                            p_restored=ttest_1samp(rest[k], 0).pvalue if len(k) >= 3 else np.nan))
    out = pd.DataFrame(res)
    print('\nLost across people (within - f0) and restored by calibration (mode - f0):')
    print(out.round(4).to_string(index=False))
    print('\nRead: compare frac_restored for fine (within-category) vs category, and probe_low_r2 vs probe_sem_acc.')
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
