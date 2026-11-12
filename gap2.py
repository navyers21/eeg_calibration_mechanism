"""Decision 1, v2: what breaks across people and what does the map restore, with THINGS categories.

    python gap2.py --config C --montage posterior17 [--eval_set held] [--layer input_linear] [--out results/x.csv]

Reads run_probes2 outputs (one CSV per target). Per measure and calibrated mode:
  lost = within - f0, restored = mode - f0 (paired over targets), frac = sum(restored) / sum(lost),
  95% bootstrap CI of frac (resampling targets), paired t-test p of restored vs 0, Holm-adjusted over the table.
The preregistered question is a CONTRAST: does the map restore a different fraction of low-level information
(lowres_r2: low-level beyond what the object category explains) than of semantic information (cat_acc, category)?
Reported as frac(low) - frac(semantic) with a bootstrap CI over targets. Retrieval measures do not depend on layer.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

MEASURES = ['top1', 'category', 'fine', 'low_r2', 'lowres_r2', 'cat_acc']
CONTRASTS = [('lowres_r2', 'cat_acc'), ('lowres_r2', 'category'), ('fine', 'category')]


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]).clip(max=1)
    out[np.flatnonzero(ok)[idx]] = adj
    return out


def frac_boot(lost, rest, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(lost), (n_boot, len(lost)))
    with np.errstate(divide='ignore', invalid='ignore'):
        return rest[i].sum(1) / lost[i].sum(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--montage', default='posterior17')
    p.add_argument('--eval_set', default='held')
    p.add_argument('--layer', default='input_linear')
    p.add_argument('--out', default='')
    a = p.parse_args()
    cfg = json.load(open(a.config))
    files = sorted(glob.glob(os.path.join(os.path.dirname(cfg['results_csv']), 'interp2', a.eval_set, a.montage,
                                          't*.csv')))
    if not files:
        raise SystemExit('no probes2 outputs yet: run the probes2 phase first')
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[df.layer == a.layer]
    w = df.pivot_table(index='target', columns='mode', values=MEASURES)
    print(f'{a.montage} | {a.eval_set} | layer {a.layer} | targets {list(w.index)} | '
          f'categorized images per target ~{int(df.n_cat.mean())} in {int(df.n_categories.mean())} categories')
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print('\nMean over targets:')
    print(df.groupby('mode')[MEASURES + ['cat_chance', 'cat_acc_shuffled']].mean().T.round(4).to_string())
    rows, boots = [], {}
    for mode in ['f0_map', 'sulora_map']:
        for ms in MEASURES:
            lost = (w[(ms, 'within')] - w[(ms, 'f0')]).values
            rest = (w[(ms, mode)] - w[(ms, 'f0')]).values
            b = frac_boot(lost, rest)
            boots[(mode, ms)] = b
            rows.append(dict(measure=ms, mode=mode, targets=len(lost), lost=lost.mean(),
                             p_lost=ttest_1samp(lost, 0).pvalue, restored=rest.mean(),
                             frac=rest.sum() / lost.sum(), frac_lo=np.nanpercentile(b, 2.5),
                             frac_hi=np.nanpercentile(b, 97.5), p_restored=ttest_1samp(rest, 0).pvalue))
    out = pd.DataFrame(rows)
    out['p_lost_holm'] = holm(out.p_lost)
    out['p_restored_holm'] = holm(out.p_restored)
    print('\nLost across people (within - f0) and restored by calibration (mode - f0), Holm over this table:')
    print(out.round(4).to_string(index=False))
    print('\nDecision 1 contrast: frac restored (A) - frac restored (B), 95% bootstrap CI over targets:')
    crow = []
    for mode in ['f0_map', 'sulora_map']:
        for A, B in CONTRASTS:
            d = boots[(mode, A)] - boots[(mode, B)]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            pt = out.set_index(['mode', 'measure']).frac
            crow.append(dict(mode=mode, A=A, B=B, frac_A=pt[(mode, A)], frac_B=pt[(mode, B)],
                             diff=pt[(mode, A)] - pt[(mode, B)], ci_lo=lo, ci_hi=hi,
                             clear=bool(lo > 0 or hi < 0)))
    crow = pd.DataFrame(crow)
    print(crow.round(3).to_string(index=False))
    print('\nRead: a subspace story needs the lowres_r2 vs cat_acc / category contrasts to be clear (CI excludes 0).'
          '\nA frac is unreliable when the measure is barely lost (check p_lost_holm).')
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        out.to_csv(a.out, index=False)
        crow.to_csv(a.out.replace('.csv', '_contrast.csv'), index=False)


if __name__ == '__main__':
    main()
