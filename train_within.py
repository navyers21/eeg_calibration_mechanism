"""Train (or reuse) within-subject decoders: same architecture, blur targets and recipe as the cross-subject f0,
trained on the subject's own training images (the 200 held-out concepts excluded). Checkpoints go to Drive.

    python train_within.py --config C --targets 2 3 4 --montages posterior17 occipital8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.eegalign import env  # noqa: E402
from src.eegalign import ubp as U  # noqa: E402
from src.eegalign.source import ensure_within, within_path  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', required=True)
    p.add_argument('--montages', nargs='+', required=True)
    a = p.parse_args()
    env.load(a.config)
    for t in a.targets:
        for mon in a.montages:
            had = os.path.exists(within_path(t, mon))
            m = ensure_within(t, mon)
            d = U.subject_montage(t, mon, keys=('test', 'test_paths'))
            top1 = U.retrieval(m, d['test'], U.feats_for(d['test_paths'], 'uniform:11', 'RN50'), np.arange(200))[0]
            print(f'  subject {t:2d} {mon:12s} {"reused " if had else "trained"} | within-subject test top-1 {top1:.1f}',
                  flush=True)
        U.free_raw()


if __name__ == '__main__':
    main()
