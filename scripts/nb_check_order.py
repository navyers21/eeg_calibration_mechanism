"""Is the NeuroBridge image order the same as the UBP image order (sorted paths)?

    python scripts/nb_check_order.py --config /content/nb_config.json --ubp_cache <UBP montage_cache/hf_whiten> [--subjects 1 2]

The two datasets come from the same raw recordings but are whitened differently (63-channel MVNN vs UBP's
whitening), so the same image's EEG is a different mix of channels in each, and a raw per-image correlation
can look like noise even when the order is right. This check first fits a 17 x 17 channel map from UBP to
NeuroBridge EEG on half of the images (time points as samples), then on the OTHER half compares:
  test    200 images, 80 repetitions averaged (high SNR): correlation of matched vs shuffled images
  train   per image (4 reps): matched vs shuffled, and within each concept, how often the best-matching UBP image
          is the same index (identity; chance 10%), which detects a within-concept reordering
  lag     the time shift (samples) that maximizes the matched correlation (0 if the epochs line up)
Verdict: order OK if matched >> shuffled on both splits and the within-concept identity rate is high.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]


def zrow(a):
    a = a.reshape(len(a), -1)
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8)


def fit_map(U, B):
    """Least squares channel map W (C x C) with B ~ W U, using every (image, time point) as a sample."""
    Uf = U.transpose(1, 0, 2).reshape(U.shape[1], -1)
    Bf = B.transpose(1, 0, 2).reshape(B.shape[1], -1)
    return Bf @ Uf.T @ np.linalg.inv(Uf @ Uf.T + 1e-3 * np.trace(Uf @ Uf.T) / len(Uf) * np.eye(len(Uf)))


def corr_rows(a, b):
    return (zrow(a) * zrow(b)).mean(1)


def check(xb, xu, rng):
    """xb, xu: (n, C, T) NeuroBridge and UBP EEG in the order to test. Map fitted on half, scored on the rest."""
    n = len(xb)
    p = rng.permutation(n)
    fit, ev = p[: n // 2], p[n // 2:]
    W = fit_map(xu[fit], xb[fit])
    pu = np.einsum('cd,ndt->nct', W, xu[ev])
    matched = corr_rows(xb[ev], pu)
    shuffled = corr_rows(xb[ev], pu[rng.permutation(len(ev))])
    lags = {}
    for lag in range(-25, 26, 5):
        a, b = (xb[ev][..., lag:], pu[..., :pu.shape[-1] - lag]) if lag >= 0 else (xb[ev][..., :lag], pu[..., -lag:])
        lags[lag] = float(corr_rows(a, b).mean())
    return float(matched.mean()), float(shuffled.mean()), max(lags, key=lags.get), lags, W


def main():
    from eegalign import nb as N
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ubp_cache', required=True)
    p.add_argument('--subjects', type=int, nargs='+', default=[1, 2])
    p.add_argument('--montage', default='posterior17')
    a = p.parse_args()
    N.setup(a.config)
    out = {}
    for s in a.subjects:
        up = os.path.join(a.ubp_cache, f'sub-{s:02d}', f'{a.montage}.npz')
        if not os.path.exists(up):
            print(f'sub-{s:02d}: no UBP cache at {up}')
            continue
        u = np.load(up)
        b = N.subject_data(s, a.montage)
        rng = np.random.RandomState(0)
        # test
        ou = np.argsort(u['test_paths'])
        mt, st, lt, lags_t, _ = check(b['test'].astype(np.float32), u['test'][ou].astype(np.float32), rng)
        # train: subsample concepts to keep it light
        ou = np.argsort(u['train_paths'])
        xu, xb = u['train'][ou].astype(np.float32), b['train'].astype(np.float32)
        concepts = rng.choice(1654, 300, replace=False)
        idx = np.concatenate([np.arange(c * 10, c * 10 + 10) for c in concepts])
        mr, sr, lr_, lags_r, W = check(xb[idx], xu[idx], rng)
        # within-concept identity: map fitted on all sampled images, then 10 x 10 matching per concept
        pu = np.einsum('cd,ndt->nct', W, xu[idx])
        za, zb = zrow(xb[idx]), zrow(pu)
        hits = 0
        for k in range(len(concepts)):
            sl = slice(k * 10, k * 10 + 10)
            R = za[sl] @ zb[sl].T / za.shape[1]
            hits += int((R.argmax(1) == np.arange(10)).sum())
        ident = hits / (10 * len(concepts))
        ok = (mt > st + 0.1) and (mr > sr + 0.05) and ident > 0.5
        out[s] = dict(test_matched=round(mt, 3), test_shuffled=round(st, 3), test_best_lag=lt,
                      train_matched=round(mr, 3), train_shuffled=round(sr, 3), train_best_lag=lr_,
                      within_concept_identity=round(ident, 3), order_ok=bool(ok))
        print(f'sub-{s:02d}: test matched {mt:.3f} vs shuffled {st:.3f} (best lag {lt}) | train matched {mr:.3f} vs '
              f'shuffled {sr:.3f} (best lag {lr_}) | within-concept identity {ident:.0%} (chance 10%) -> '
              f'{"ORDER OK" if ok else "ORDER DIFFERS or check inconclusive"}', flush=True)
        print(f'   test correlation by lag (samples): {json.dumps({k: round(v, 3) for k, v in lags_t.items()})}')
    path = os.path.join(N.ENV['cache_dir'], 'order_check.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('saved', path)


if __name__ == '__main__':
    main()
