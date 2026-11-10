"""Results CSV: one row per (phase, target, montage, adapter, n_images, seed). Finished runs are skipped by tag."""
import csv
import os
import time

from . import VERSION
from . import ubp as U

COLUMNS = ['tag', 'phase', 'target', 'montage', 'adapter', 'n_images', 'presentations', 'seed', 'test_top1',
           'test_top5', 'test_se', 'test_clean_top1', 'held_top1', 'held_se', 'w_shift', 'source', 'wall_s',
           'version', 'time']


def tag(phase, target, montage, adapter, n_images, seed):
    return f"{phase}/{montage}-{adapter}|n{n_images}|t{target:02d}|seed{seed}|{U.ENV['source']}"


def done_tags(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {r['tag'] for r in csv.DictReader(f)}


def append(path, row):
    new = not os.path.exists(path)
    row = dict(row, source=U.ENV['source'], version=VERSION, time=time.strftime('%Y-%m-%d %H:%M'))
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
