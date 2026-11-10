"""Train (or check) the leave-one-subject-out source decoders f0 for the given targets and montages.

    python train_source.py --config /content/align_config.json --targets 1 2 3 --montages occipital8 posterior17

Checkpoints go to the Drive checkpoint folder; ones that already exist (including those from the B experiment)
are reused. Each prints its test-set top-1 on the held-out target.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from eegalign import env  # noqa: E402
from eegalign import ubp as U  # noqa: E402
from eegalign.source import ckpt_path, ensure_source  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', required=True)
    p.add_argument('--montages', nargs='+', required=True)
    a = p.parse_args()
    env.load(a.config)
    for t in a.targets:
        for mon in a.montages:
            had = os.path.exists(ckpt_path(t, mon))
            m = ensure_source(t, mon)
            d = U.subject_montage(t, mon)
            top1 = U.retrieval(m, d['test'], U.feats_for(d['test_paths'], 'uniform:11', 'RN50'), np.arange(200))[0]
            print(f'  target {t:2d} {mon:12s} {"reused " if had else "trained"} | f0 test top-1 {top1:.1f}', flush=True)
        U.free_raw()


if __name__ == '__main__':
    main()
