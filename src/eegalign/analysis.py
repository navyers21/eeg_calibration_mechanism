"""Summary tables: per-subject paired gains over f0 and over B's full adapter, and the go/no-go check."""
import numpy as np
import pandas as pd


def _p(d):
    """Two-sided paired t-test across subjects (seeds averaged first); NaN with fewer than 3 subjects."""
    from scipy.stats import ttest_1samp
    d = np.asarray(d, float)
    return float(ttest_1samp(d, 0.0).pvalue) if len(d) >= 3 and d.std() > 0 else float('nan')


def _per_target(df, metric):
    """Seed-averaged metric per (montage, adapter, n_images, target)."""
    return df.groupby(['montage', 'adapter', 'n_images', 'target'])[metric].mean()


def summary(csv_path, phase, metric='test_top1'):
    df = pd.read_csv(csv_path)
    df = df[df.phase.isin(phase.split(','))]          # 'subspace,subx': maps from one phase, subx from the other
    if df.empty or 'f0' not in set(df.adapter):
        raise SystemExit(f'No finished {phase} runs with an f0 row yet: run the phase first.')
    df['n_images'] = df.n_images.astype(int)
    df['adapter'] = df.adapter.str.replace(r'_lr[0-9.e-]+$', '', regex=True)   # tuned lr is part of the name
    per = _per_target(df, metric)
    f0 = per.xs('f0', level='adapter').droplevel('n_images')
    full = per.xs('full', level='adapter') if 'full' in df.adapter.values else None
    rows = []
    for (mon, ad, n), g in per.groupby(level=['montage', 'adapter', 'n_images']):
        if ad == 'f0':
            continue
        g = g.droplevel(['montage', 'adapter', 'n_images'])
        d = g - f0.loc[mon].reindex(g.index)
        r = dict(montage=mon, adapter=ad, n_images=n, presentations=4 * n, subjects=len(g), top1=g.mean(),
                 f0=f0.loc[mon].reindex(g.index).mean(), gain_f0=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)),
                 wins_f0=f'{int((d > 0).sum())}/{len(d)}', p_f0=_p(d))
        if full is not None and ad != 'full' and (mon, n) in full.droplevel('target').index:
            e = (g - full.loc[(mon, n)].reindex(g.index)).dropna()
            r.update(vs_full=e.mean(), wins_full=f'{int((e > 0).sum())}/{len(e)}', p_full=_p(e))
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(['montage', 'adapter', 'n_images'])
    curves = df[df.adapter != 'f0'].groupby(['montage', 'adapter', 'n_images'])[metric].mean().unstack('n_images')
    return out, curves


def go_check(out, adapter='align', budgets=(25, 50, 100), min_frac=0.7):
    """Go: `adapter` beats f0 on >= 70% of subjects, mean gain > 2 SE, and beats `full`, at every budget listed."""
    lines = []
    for mon, g in out[(out.adapter == adapter) & (out.n_images.isin(budgets))].groupby('montage'):
        for _, r in g.iterrows():
            w, k = map(int, r.wins_f0.split('/'))
            ok = w >= np.ceil(min_frac * k) and r.gain_f0 > 2 * r.se and r.get('vs_full', 0) > 0
            lines.append(f"{mon:12s} n={r.n_images:4d}: gain {r.gain_f0:+.2f} ± {r.se:.2f}, wins {r.wins_f0}, "
                         f"vs full {r.get('vs_full', float('nan')):+.2f} -> {'PASS' if ok else 'fail'}")
    return lines
