"""Adapter between Karan's model-agnostic toolkit (mi/) and our UBP models.

One adapter object per (mode, montage, n_images, seed, eval set). Modes, all scored on the SAME target images:
  within      decoder trained on the target's own data (the "before")
  f0          cross-subject decoder (trained on the 9 other subjects)
  f0_map      f0 + the target's channel map fitted on n calibration images
  sulora_map  SuLoRA-style jointly tuned f0 + the target's map (starting from the mean source map)
  sub_low / sub_sem / sub_rand   f0 + subspace-only calibration (see src/eegalign/subspace.py)
Eval sets: 'held' = the 200 held-out training concepts x 10 images (4-rep averages; never trained on by any mode),
           'test' = the official 200 test images (80-rep averages; analysis only, never used to choose anything).
"""
from collections import OrderedDict

import numpy as np
import torch

from src.eegalign import ubp as U
from src.eegalign.adapters import Aligned, adapt
from src.eegalign.data import Target
from src.eegalign.source import ensure_within

_T = {}


def target(t, montage):
    key = (t, montage)
    if key not in _T:
        _T.clear()
        _T[key] = Target(t, montage, train_source=False)
    return _T[key]


class UBPAdapter:
    def __init__(self, mode, montage, n_images=50, seed=0, eval_set='held'):
        self.mode, self.montage, self.n, self.seed, self.eval_set = mode, montage, n_images, seed, eval_set

    def load_model(self, subject, device):
        T = target(int(subject), self.montage)
        if self.mode == 'within':
            return ensure_within(int(subject), self.montage, train=False)
        if self.mode == 'f0':
            return adapt('f0', T, None, None, self.seed)
        idx = np.random.RandomState(self.seed).permutation(T.n)[:self.n]
        name = dict(f0_map='align', sulora_map='sulora2').get(self.mode, self.mode)
        return adapt(name, T, T.x[idx], T.tgt[idx], self.seed).eval()

    def get_layers(self, model):
        b = model.f0 if isinstance(model, Aligned) else model
        return OrderedDict(input_linear=b.model[0], res_block=b.model[1], output=b.model[2]), {}

    def encode(self, model, eeg):
        return model(eeg.contiguous())

    def load_eval_set(self, subject):
        T = target(int(subject), self.montage)
        if self.eval_set == 'test':
            paths, groups = list(T.test_paths), np.arange(len(T.test_paths))
            eeg, gal = torch.as_tensor(T.test_x), T.test_g
        else:
            paths = [p for ps in T.held_paths for p in ps]
            groups = np.concatenate(T.held_labels)
            eeg, gal = torch.as_tensor(np.concatenate(T.held_x)), torch.cat(T.held_g)
        return dict(eeg=eeg.float(), image_paths=paths, groups=groups, gallery_feats=gal,
                    cluster_feats=U.feats_for(paths, 'none', 'RN50').numpy())

    def load_train_image_feats(self):
        T = next(iter(_T.values()))
        return U.feats_for(T.P, 'none', 'RN50').numpy()
