"""Calibrate one (target, montage): every adapter x budget x seed, one CSV row each.

    python calibrate.py --config /content/align_config.json --phase pilot --target 1 --montage occipital8 \
        --adapters f0 ea align align_bias align_full full --budgets 25 50 100 200 400 800 --seeds 0 1

A budget is a number of calibration images (each = 4 presentations, all repetitions averaged) or 'full' for the
whole pool. For each seed the images are the first n of one fixed random order, so budgets are nested and every
adapter sees the same images. f0 and ea need no labeled images: one row each (n_images 0, seed 0).
Resumable: runs whose tag is already in the CSV are skipped.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from eegalign import env, registry  # noqa: E402
from eegalign.adapters import adapt, w_shift  # noqa: E402
from eegalign.data import Target  # noqa: E402
from eegalign.evaluate import evaluate  # noqa: E402

LABEL_FREE = ('f0', 'ea', 'sulora_f0', 'sulora2_f0')


def plan(a):
    jobs = [(ad, 0, 0) for ad in a.adapters if ad in LABEL_FREE]
    for s in a.seeds:
        for b in a.budgets:
            for ad in a.adapters:
                if ad not in LABEL_FREE:
                    jobs.append((ad, b, s))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--phase', required=True)
    p.add_argument('--target', type=int, required=True)
    p.add_argument('--montage', required=True)
    p.add_argument('--adapters', nargs='+', required=True)
    p.add_argument('--budgets', nargs='+', default=['25', '50', '100', '200', '400', '800'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    cfg = env.load(a.config)
    csv_path = cfg['results_csv']
    done = registry.done_tags(csv_path)
    tag = lambda ad, b, s: registry.tag(a.phase, a.target, a.montage, ad, b, s)
    todo = [j for j in plan(a) if tag(*j) not in done]
    print(f'[{a.phase}] target {a.target} {a.montage}: {len(todo)} runs to do', flush=True)
    if not todo:
        return
    T = Target(a.target, a.montage)
    orders = {s: np.random.RandomState(s).permutation(T.n) for s in a.seeds}
    for ad, b, s in todo:
        t0 = time.time()
        n = T.n if b == 'full' else int(b)
        idx = orders.get(s, orders[a.seeds[0]])[:n]
        big = n > 800 or (a.phase == 'scaling' and n >= 800)   # scaling: every adapter gets 1000 steps from 800 up
        m = adapt(ad, T, T.x[idx], T.tgt[idx], s, steps=1000 if big else 200, batch=1024 if big else 64)
        ev = evaluate(m, T)
        registry.append(csv_path, dict(tag=tag(ad, b, s), phase=a.phase, target=a.target, montage=a.montage,
                                       adapter=ad, n_images=n, presentations=4 * n, seed=s, w_shift=w_shift(m),
                                       wall_s=round(time.time() - t0, 1), **ev))
        if hasattr(m, 'W'):
            np.save(os.path.join(cfg['maps_dir'], tag(ad, b, s).replace('/', '_').replace('|', '__') + '.npy'),
                    m.W.detach().cpu().numpy())
        print(f'  {ad:11s} n={n:5d} seed {s} | test {ev["test_top1"]:5.1f} ± {ev["test_se"]:.1f} | held '
              f'{ev["held_top1"]:5.2f} | W shift {w_shift(m):.3f} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
