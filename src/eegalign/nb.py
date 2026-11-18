"""NeuroBridge backend: data cache, model, training recipe, evaluation.

Faithful to the official code (github.com/feroooooo/NeuroBridge @ 4f59fbb, intra-subject THINGS-EEG script):
  encoder     EEGProject: Linear(C*250 -> 1024) -> Residual(GELU, Linear(1024, 1024), Dropout 0.3) -> LayerNorm
              (same as UBP's EEGProject; module names match NeuroBridge so its checkpoints load)
  projectors  linear 1024 -> 512 on BOTH sides (Shared Semantic Projector), both trained
  loss        symmetric InfoNCE; image embeddings L2-normalized, EEG embeddings not; logit scale
              softplus(log(1/0.07)) ~ 2.73, fixed (--softplus, temperature not learnable)
  priors      image side: the fused feature of 4 augmented views (GaussianBlur, GaussianNoise, LowResolution, Mosaic),
              for training AND test (--image_aug --image_test_aug). EEG side: one-time random smoothing of the
              training EEG (kernel 5, p = 0.3 per trial x channel; --eeg_aug smooth --frozen_eeg_prior)
  optimizer   AdamW lr 1e-4, wd 1e-4, batch 1024 (drop last), 50 epochs, 4 repetitions averaged (--data_average)
Differences from the official code, on purpose:
  * model selection on held-out TRAINING concepts (the 200 concepts of ubp.val_concepts, same split as the UBP
    pipeline), never on the test set. NeuroBridge keeps the epoch with the lowest TEST loss; Nathan's earlier
    decoders used the last epoch. Both are logged for comparison, only the validation pick is used downstream.
  * the leave-one-subject-out model (f0) uses the same encoder and recipe on the 9 other people (the paper's
    inter-subject table uses TSConv on 63 channels; we keep EEGProject so calibration acts on the same architecture
    as within-person decoders).
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT_DIM, PROJ_DIM, T_LEN = 1024, 512, 250
INIT_T = 0.07
AUG_NAME = 'GaussianBlur-GaussianNoise-LowResolution-Mosaic'
ENV = dict(device='cuda' if torch.cuda.is_available() else 'cpu')


def setup(config_path):
    cfg = json.load(open(config_path))
    ENV.update(cfg)
    for k in ('ckpt_dir', 'results_dir'):
        os.makedirs(cfg[k], exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------------------
class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class EEGProject(nn.Module):
    def __init__(self, feature_dim=FEAT_DIM, eeg_sample_points=T_LEN, channels_num=17):
        super().__init__()
        self.input_dim = eeg_sample_points * channels_num
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, feature_dim),
            ResidualAdd(nn.Sequential(nn.GELU(), nn.Linear(feature_dim, feature_dim), nn.Dropout(0.3))),
            nn.LayerNorm(feature_dim))

    def forward(self, x):
        return self.model(x.reshape(x.shape[0], self.input_dim))


class ProjectorLinear(nn.Module):
    def __init__(self, input_dim=FEAT_DIM, output_dim=PROJ_DIM):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class NBModel(nn.Module):
    """forward(x) -> EEG embedding (512, unnormalized). gallery(f) -> L2-normalized image embeddings.
    `.model` is the encoder's Sequential and `.softplus(.logit_scale)` the logit scale, so code written for UBP's
    EEGProject (hooks on model[0], the CLIP-style loss) works unchanged with gallery = self.gallery(features)."""

    def __init__(self, n_ch):
        super().__init__()
        self.encoder = EEGProject(channels_num=n_ch)
        self.eeg_projector = ProjectorLinear()
        self.img_projector = ProjectorLinear()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / INIT_T)), requires_grad=False)
        self.softplus = nn.Softplus()

    @property
    def model(self):
        return self.encoder.model

    def forward(self, x):
        return self.eeg_projector(self.encoder(x.float()))

    def gallery(self, feats):
        return F.normalize(self.img_projector(feats.float()), dim=-1)

    def scale(self):
        return self.softplus(self.logit_scale)


def nb_loss(z_eeg, g_img, scale):
    logits = scale * z_eeg @ g_img.T
    lab = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2


def save_model(m, path, cfg, history, epoch):
    """NeuroBridge's checkpoint keys (+ ours), so eegdistill.checkpoints.load_run can also read it."""
    torch.save({'model_state_dict': m.encoder.state_dict(), 'eeg_projector_state_dict': m.eeg_projector.state_dict(),
                'img_projector_state_dict': m.img_projector.state_dict(), 'epoch': epoch, 'cfg': cfg,
                'history': history}, path)


def load_model(path, n_ch=None, device=None):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    n_ch = n_ch or ck['model_state_dict']['model.0.weight'].shape[1] // T_LEN
    m = NBModel(n_ch)
    m.encoder.load_state_dict(ck['model_state_dict'])
    m.eeg_projector.load_state_dict(ck['eeg_projector_state_dict'])
    m.img_projector.load_state_dict(ck['img_projector_state_dict'])
    return m.to(device or ENV['device']).eval(), ck


def ckpt_path(kind, montage, subject, seed, which):
    return os.path.join(ENV['ckpt_dir'], f'{kind}_{montage}_t{subject:02d}_s{seed}_{which}.pt')


# ---------------------------------------------------------------------------------------------------------------
# data (cache built by scripts/nb_data.py)
# ---------------------------------------------------------------------------------------------------------------
_FEATS = {}


def features(split, kind='aug'):
    """Image features in NeuroBridge order: (16540, 1024) train / (200, 1024) test. kind: 'aug' (fused augmented
    views, what NeuroBridge trains and tests on) or 'clean'."""
    key = (split, kind)
    if key not in _FEATS:
        d = ENV['feat_dir']
        p = os.path.join(d, AUG_NAME, f'{split}.npy') if kind == 'aug' else os.path.join(d, f'image_{split}.npy')
        f = np.load(p).astype(np.float32)
        if kind == 'aug':
            f = f.mean(0) if f.ndim == 4 else f          # (R, obj, img, D) -> (obj, img, D); R = 1 after fusing
        _FEATS[key] = torch.from_numpy(f.reshape(-1, f.shape[-1]))
    return _FEATS[key]


def subject_data(sub, montage, keys=None):
    """Cached arrays for one subject x montage. A montage without its own file is cut from a cached superset
    (occipital-8 from posterior-17), so Drive only stores posterior17 and crown8."""
    from .ubp import MONTAGES
    base = os.path.join(ENV['cache_dir'], f'sub-{sub:02d}')
    path = os.path.join(base, f'{montage}.npz')
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files if keys is None or k in keys}
    for sup in ('posterior17', 'crown8', 'nested32', 'all'):
        sp = os.path.join(base, f'{sup}.npz')
        if os.path.exists(sp) and set(MONTAGES[montage]) <= set(MONTAGES[sup]):
            d = np.load(sp)
            idx = [list(d['channels']).index(c) for c in MONTAGES[montage]]
            out = {}
            for k in d.files:
                if keys is not None and k not in keys:
                    continue
                v = d[k]
                if k in ('train', 'test'):
                    v = v[:, idx]
                elif k == 'test_trials':
                    v = v[:, :, idx]
                elif k == 'channels':
                    v = np.array(MONTAGES[montage])
                out[k] = v
            return out
    raise FileNotFoundError(f'{path} missing (and no cached superset): run scripts/nb_data.py for subject {sub}')


def val_concepts(labels, seed=2026):
    """Same 200 held-out concepts as the UBP pipeline (ubp.val_concepts)."""
    concepts = np.unique(labels)
    return np.sort(np.random.RandomState(seed).choice(concepts, 200, replace=False))


def split(labels):
    held = val_concepts(labels)
    pool = np.nonzero(~np.isin(labels, held))[0]
    draws = [np.array([np.nonzero(labels == c)[0][j] for c in held]) for j in range(10)]
    return pool, draws


def random_smooth(x, kernel_size=5, prob=0.3, seed=0, chunk=2048):
    """NeuroBridge's RandomSmooth applied once (frozen prior), vectorized (eegdistill.data.random_smooth):
    each (trial, channel) is replaced with probability `prob` by an edge-truncated centered moving average."""
    x = np.array(x, dtype=np.float32, copy=True)
    n, c, t = x.shape
    h = kernel_size // 2
    lo, hi = np.clip(np.arange(t) - h, 0, t), np.clip(np.arange(t) + h + 1, 0, t)
    width = (hi - lo).astype(np.float64)
    mask = np.random.default_rng(seed).random((n, c)) < prob
    for i in range(0, n, chunk):
        xb = x[i:i + chunk]
        cs = np.concatenate([np.zeros(xb.shape[:2] + (1,)), np.cumsum(xb, axis=-1, dtype=np.float64)], axis=-1)
        sm = ((cs[..., hi] - cs[..., lo]) / width).astype(np.float32)
        mb = mask[i:i + chunk]
        xb[mb] = sm[mb]
    return x


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def embed(m, X, bs=2048):
    m.eval()
    dev = ENV['device']
    return torch.cat([m(torch.as_tensor(X[i:i + bs]).to(dev).float()) for i in range(0, len(X), bs)])


@torch.no_grad()
def retrieval(m, X, feats, labels=None):
    """200-way (or len(feats)-way) top-1 / top-5, %, cosine between EEG and projected image embeddings."""
    z = F.normalize(embed(m, X), dim=-1)
    g = m.gallery(feats.to(z.device))
    y = torch.arange(len(g), device=z.device) if labels is None else torch.as_tensor(labels, device=z.device)
    top = (z @ g.T).topk(5, dim=-1).indices
    return 100 * (top[:, 0] == y).float().mean().item(), 100 * (top == y[:, None]).any(1).float().mean().item()


@torch.no_grad()
def test_loss(m, X, feats):
    z = embed(m, X)
    return float(nb_loss(z, m.gallery(feats.to(z.device)), m.scale()))
