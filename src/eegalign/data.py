"""Everything one (target, montage) needs: calibration pool, two evaluation sets, source statistics.

  * pool: the target's training images outside the 200 held-out concepts (calibration images are drawn from here)
  * test set (primary): the official 200 test images, 80 repetitions averaged, never used for anything else
  * held-out concepts (secondary, for continuity with B): 200 concepts x 10 images, 4-rep averages, 10 draws
  * source statistics: other subjects' EEG on the pool images, for EA (mean covariance) and rehearsal
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import TARGET_SPEC
from . import ubp as U
from .source import ensure_source


def cov(x):
    """Mean channel covariance of (n, C, T) trials, each trial centred over time."""
    x = torch.as_tensor(np.asarray(x, np.float32))
    x = x - x.mean(-1, keepdim=True)
    return (torch.einsum('nct,ndt->cd', x, x) / (x.shape[0] * x.shape[-1])).numpy()


class Target:
    def __init__(self, target, montage, n_draws=10, n_rehearse=20000, train_source=True):
        dev = U.ENV['device']
        self.target, self.montage, self.C = target, montage, len(U.MONTAGES[montage])
        self.f0 = ensure_source(target, montage, train=train_source)
        d = U.subject_montage(target, montage)
        labels, paths = d['train_labels'], d['train_paths']
        pool, _ = U.split_indices(labels, 'dev')
        held = np.setdiff1d(np.unique(labels), np.unique(labels[pool]))
        assert len(held) == 200 and not np.isin(labels[pool], held).any(), 'held-out concepts leak into the pool'
        self.P, self.n = paths[pool], len(pool)
        self.x = d['train'][pool].astype(np.float32)
        self.tgt = U.feats_for(self.P, TARGET_SPEC, 'RN50').to(dev)
        self.y = np.arange(200)
        self.test_x = d['test'].astype(np.float32)
        self.test_g = U.feats_for(d['test_paths'], TARGET_SPEC, 'RN50')
        self.test_gc = U.feats_for(d['test_paths'], 'none', 'RN50')
        ev = [np.array([np.nonzero(labels == c)[0][j] for c in held]) for j in range(n_draws)]
        self.held_x = [d['train'][ix].astype(np.float32) for ix in ev]
        self.held_g = [U.feats_for(paths[ix], TARGET_SPEC, 'RN50') for ix in ev]
        self.held_paths = [paths[ix] for ix in ev]
        self.held_labels = [labels[ix] for ix in ev]
        self.test_paths = d['test_paths']
        self._sources(n_rehearse)
        self.R_tgt = cov(self.x)          # unlabeled target EEG only (label-free alignment)

    def _sources(self, n_rehearse):
        pos = {p: i for i, p in enumerate(self.P)}
        srcs = [s for s in U.ENV['pool_subjects'] if s != self.target]
        rng, acc_r, rx, rt, rs = np.random.RandomState(0), 0, [], [], []
        for s in srcs:
            d = U.subject_montage(s, self.montage, keys=('train', 'train_paths'))
            rows = np.array([pos.get(p, -1) for p in d['train_paths']])
            x = np.zeros((self.n,) + d['train'].shape[1:], np.float32)
            x[rows[rows >= 0]] = d['train'][rows >= 0].astype(np.float32)
            acc_r = acc_r + cov(x)
            pick = rng.choice(self.n, min(self.n, n_rehearse // len(srcs)), replace=False)
            rx.append(x[pick].astype(np.float16)); rt.append(pick); rs.append(np.full(len(pick), s))
        self.R_src = acc_r / len(srcs)
        self.reh_x, self.reh_t, self.reh_s = np.concatenate(rx), np.concatenate(rt), np.concatenate(rs)
