"""Load the exported checkpoints (checkpoints branch) as plain torch modules. Needs only torch + numpy.

    from ckpt_loader import load_stage
    m = load_stage('checkpoints', person=3, stage='rot500')    # f0 | zero | rot100 | rot500 | within | within_rep
    z = m(x)                  # x: (n, 17, 250) EEG in our preprocessing (posterior17, NeuroBridge order); z: (n, 512)
    G = m.gallery_test        # (200, 512) the gallery this stage is scored against (rows = THINGS-EEG2 test images)
    top1 = (F.normalize(z, dim=-1) @ G.T).argmax(1)

Every stage has the same layers (encoder.model[0] Linear 4250->1024, Residual(GELU, Linear 1024->1024), LayerNorm,
eeg_projector Linear 1024->512) plus `out`, a 512->512 Linear that is the identity except for the rot stages. So hooks
on the same layer names work across stages. For calibrated stages the label-free layer-1 offset is already folded into
encoder.model[0].bias, so intermediate activations are the calibrated ones.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

T_LEN, FEAT, PROJ = 250, 1024, 512


class _Res(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class _Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(FEAT, PROJ)

    def forward(self, x):
        return self.linear(x)


class _Enc(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(n_ch * T_LEN, FEAT),
                                   _Res(nn.Sequential(nn.GELU(), nn.Linear(FEAT, FEAT), nn.Dropout(0.3))),
                                   nn.LayerNorm(FEAT))

    def forward(self, x):
        return self.model(x.reshape(len(x), -1))


class Stage(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.encoder = _Enc(n_ch)
        self.eeg_projector = _Proj()
        self.out = nn.Linear(PROJ, PROJ)

    def forward(self, x):
        return self.out(self.eeg_projector(self.encoder(x)))


def load_stage(root, person, stage, device='cpu'):
    man = json.load(open(os.path.join(root, 'manifest.json')))
    rec = man['people'][str(person)]['stages'][stage]
    ck = torch.load(os.path.join(root, rec['file']), map_location='cpu', weights_only=False)
    if 'delta' in ck:                    # calibrated stage: f0's weights + the changed layer-1 bias and output map
        f0 = man['people'][str(person)]['stages'][ck['base']]['file']
        sd = torch.load(os.path.join(root, f0), map_location='cpu', weights_only=False)['state_dict']
        sd = dict(sd, **ck['delta'])
    else:
        sd = ck['state_dict']
    sd = {k: v.float() for k, v in sd.items()}
    n_ch = sd['encoder.model.0.weight'].shape[1] // T_LEN
    m = Stage(n_ch)
    m.load_state_dict(sd)
    m.gallery_test = F.normalize(ck['gallery_test'].float(), dim=-1)
    m.info = {k: v for k, v in ck.items() if k not in ('state_dict', 'delta', 'gallery_test')}
    return m.to(device).eval().requires_grad_(False)
