"""Source decoder f0: UBP EEGProject trained on every subject except the target (leave one subject out).

The run config is identical to the B experiment's (`al_source`), so checkpoints already on Drive are reused.
Early stopping uses the source subjects' copies of the 200 held-out training concepts; the primary metric here is
the official test set, which f0 never sees in any form.
"""
import os

import torch

from . import TARGET_SPEC
from . import ubp as U


def source_cfg(target, montage):
    return U.make_cfg('al_source', montage, TARGET_SPEC, lr=1e-5, early_stop='val', single_trial=False, save=True,
                      loso_target=target, label=f'{montage}-proj-{TARGET_SPEC}-source')


def ckpt_path(target, montage):
    tag = U.run_tag(source_cfg(target, montage))
    return os.path.join(U.ENV['ckpt_dir'], tag.replace('/', '_').replace('|', '__') + '.pt')


def ensure_source(target, montage, train=True):
    """Load f0 for (target, montage), training it first if the checkpoint is missing."""
    ck = ckpt_path(target, montage)
    if not os.path.exists(ck):
        if not train:
            raise FileNotFoundError(f'no source model for target {target} {montage}: run the sources phase first')
        cfg = source_cfg(target, montage)
        print(f'  training source model {U.run_tag(cfg)}', flush=True)
        U.append_result(U.run(cfg))
    m = U.build_model('EEGProjectLayer', len(U.MONTAGES[montage]), U.CLIP_MAP['RN50'][1]).to(U.ENV['device'])
    m.load_state_dict(torch.load(ck, weights_only=False, map_location=U.ENV['device'])['state_dict'])
    return m.eval()


# ----------------------------------------------------------------------------------------------------------------
# Within-subject decoder: same architecture, targets and recipe as f0, trained on the target's OWN training images
# (the 200 held-out concepts are excluded by protocol 'dev'). The "before" model of the mechanism analysis.
# ----------------------------------------------------------------------------------------------------------------
def within_cfg(subject, montage):
    return U.make_cfg('within', montage, TARGET_SPEC, single_trial=False, save=True, subject=subject,
                      label=f'{montage}-proj-{TARGET_SPEC}-within')


def within_path(subject, montage):
    tag = U.run_tag(within_cfg(subject, montage))
    return os.path.join(U.ENV['ckpt_dir'], tag.replace('/', '_').replace('|', '__') + '.pt')


def ensure_within(subject, montage, train=True):
    ck = within_path(subject, montage)
    if not os.path.exists(ck):
        if not train:
            raise FileNotFoundError(f'no within-subject model for subject {subject} {montage}: run the within phase')
        cfg = within_cfg(subject, montage)
        print(f'  training within-subject model {U.run_tag(cfg)}', flush=True)
        U.append_result(U.run(cfg))
    m = U.build_model('EEGProjectLayer', len(U.MONTAGES[montage]), U.CLIP_MAP['RN50'][1]).to(U.ENV['device'])
    m.load_state_dict(torch.load(ck, weights_only=False, map_location=U.ENV['device'])['state_dict'])
    return m.eval()
