"""Calibration adapters: how f0 is updated on a new user's calibration images.

| name       | trained parameters                                     | init           |
| ---------- | ------------------------------------------------------ | -------------- |
| f0         | none (the uncalibrated cross-subject decoder)          |                |
| ea         | none: W = R_src^1/2 R_tgt^-1/2 from unlabeled EEG       | label-free     |
| align      | C x C channel map W in front of the frozen f0          | identity       |
| align_bias | W plus a per-channel offset                            | identity, 0    |
| align_full | align, then every f0 layer (B's adapter) on top of it  | identity       |
| full       | every f0 layer, L2 pull to f0, rehearsal (B's frozen)  | f0             |
| first_layer| f0's first linear layer only (the only channel-mixing layer), L2 pull 10 | f0   |
| later_layers | f0's residual block + LayerNorm only, L2 pull 1       | f0             |
| lowrank_first | rank-8 update B @ A to the first layer (LoRA)        | B @ A = 0      |
| svd_first  | k x k update inside the top-16 singular directions of the first layer | 0  |
| sulora_f0  | none: f0 jointly re-tuned on the sources with one channel map per source subject, | |
|            | applied with the mean source map (zero-shot)           | label-free     |
| sulora     | the new user's map in front of that jointly tuned f0   | mean source map|

All learned adapters use the CLIP loss against the calibration images' blurred CLIP features, with 1,024 extra
negatives drawn from the pool's image features (features only, no EEG), and a relative L2 pull to the start point.
Rehearsal samples (other subjects' EEG) always bypass the user's channel map.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ubp as U

ALIGN = dict(lr=1e-3, anchor=1.0, rehearse=0.0)
FULL = dict(lr=1e-4, anchor=1.0, rehearse=0.5)        # B's frozen adapter: all-a1-lr0.0001-s200-reh0.5
ADAPTERS = ['f0', 'ea', 'align', 'align_bias', 'align_full', 'full', 'first_layer', 'later_layers', 'lowrank_first',
            'svd_first', 'sulora_f0', 'sulora', 'lora_first', 'sulora2_f0', 'sulora2', 'sub_low', 'sub_sem', 'sub_rand']
SUB = dict(lr=1e-3, anchor=1.0)      # subspace calibration: k x k matrix M from zero, absolute L2 pull to zero
# v2 subspace calibration ('subx_<low|sem|rand>_k<k>[_lr<lr>]'): no pull to zero, lr tuned on subject 1 (subx_tune)
SUBX = dict(lr=1e-2, anchor=0.0)


def parse_subx(name):
    """'subx_low_k6_lr1e-2' -> ('sub_low', 6, 0.01)."""
    parts = name.split('_')
    kind, k = 'sub_' + parts[1], int(parts[2][1:])
    lr = float(parts[3][2:]) if len(parts) > 3 else SUBX['lr']
    return kind, k, lr
LAYER = dict(first_layer=dict(lr=1e-4, anchor=10.0), later_layers=dict(lr=1e-4, anchor=1.0),
             lowrank_first=dict(lr=1e-3, anchor=0.0), svd_first=dict(lr=1e-3, anchor=0.0))
LAYER['lora_first'] = dict(lr=1e-4, anchor=0.0)     # v2 LoRA: standard init (A random, B = 0), 10x lower lr
JOINT = dict(steps=2000, batch=256, lr_f0=1e-4, lr_w=1e-3, anchor_f0=1.0, anchor_w=1.0)
# v2 joint tuning: every source subject's full pool (not a 2,200-image sample), f0 at its own training lr (1e-5)
JOINT2 = dict(steps=1000, batch=256, lr_f0=1e-5, lr_w=1e-3, anchor_f0=1.0, anchor_w=1.0)


def msqrt(R, p):
    w, v = np.linalg.eigh(R + 1e-6 * np.trace(R) / len(R) * np.eye(len(R)))
    return (v * w.clip(1e-12) ** p) @ v.T


class Aligned(nn.Module):
    """f0(W @ x + b): a channel map in front of the decoder. f0 stays a submodule, so stage 2 can train it."""

    def __init__(self, f0, W0, bias=False):
        super().__init__()
        self.f0 = f0
        self.register_buffer('W0', torch.as_tensor(W0, dtype=torch.float32))
        self.W = nn.Parameter(self.W0.clone())
        self.b = nn.Parameter(torch.zeros(len(W0), 1)) if bias else None

    def forward(self, x):
        x = torch.einsum('cd,ndt->nct', self.W, x.float())
        if self.b is not None:
            x = x + self.b
        return self.f0(x.contiguous())


class LowRankLinear(nn.Module):
    """W0 + B @ A on a frozen linear layer (LoRA). A starts at 0, so the layer starts exactly at W0."""

    def __init__(self, lin, rank=8):
        super().__init__()
        self.register_buffer('W0', lin.weight.detach().clone())
        self.register_buffer('b0', lin.bias.detach().clone())
        self.A = nn.Parameter(torch.zeros(rank, lin.in_features, device=lin.weight.device))
        self.B = nn.Parameter(torch.randn(lin.out_features, rank, device=lin.weight.device) * 0.01)

    def forward(self, x):
        return F.linear(x, self.W0 + self.B @ self.A, self.b0)


class SVDLinear(nn.Module):
    """W0 + U_k M V_k^T: a k x k update restricted to the top-k singular directions of the frozen layer."""

    def __init__(self, lin, k=16):
        super().__init__()
        self.register_buffer('W0', lin.weight.detach().clone())
        self.register_buffer('b0', lin.bias.detach().clone())
        U_, S_, Vh = torch.linalg.svd(self.W0, full_matrices=False)
        self.register_buffer('U', U_[:, :k].contiguous())
        self.register_buffer('Vh', Vh[:k].contiguous())
        self.M = nn.Parameter(torch.zeros(k, k, device=lin.weight.device))

    def forward(self, x):
        return F.linear(x, self.W0 + self.U @ self.M @ self.Vh, self.b0)


class LoRALinear(nn.Module):
    """Standard LoRA: W0 + (alpha / r) B @ A with A ~ N(0, 1/in) and B = 0, so the layer starts exactly at W0."""

    def __init__(self, lin, rank=8, alpha=8.0):
        super().__init__()
        self.register_buffer('W0', lin.weight.detach().clone())
        self.register_buffer('b0', lin.bias.detach().clone())
        self.A = nn.Parameter(torch.randn(rank, lin.in_features, device=lin.weight.device) / lin.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(lin.out_features, rank, device=lin.weight.device))
        self.scale = alpha / rank

    def forward(self, x):
        return F.linear(x, self.W0 + self.scale * self.B @ self.A, self.b0)


def base(m):
    """The underlying UBP decoder (has logit_scale), through any wrappers (channel maps, output / pattern edits)."""
    while hasattr(m, 'f0'):
        m = m.f0
    return m


def _train(m, T, X, Tg, params, starts, lr, anchor, rehearse, steps, batch, seed, n_neg=1024, wd=1e-4):
    """CLIP loss on calibration pairs (+ rehearsal through f0 only) + relative L2 pull of `params` to `starts`."""
    dev = U.ENV['device']
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    X, Tg = torch.as_tensor(X).to(dev).float(), torch.as_tensor(Tg).to(dev).float()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    n, bs = len(X), min(batch, len(X))
    n_reh, f = int(round(rehearse * batch)), base(m)
    norm0 = sum(p0.pow(2).sum() for p0 in starts).clamp_min(1.0)   # starts at 0: absolute L2
    m.train()
    for _ in range(steps):
        idx = torch.randperm(n, device=dev)[:bs]
        z, t = m(X[idx]), Tg[idx]
        if n_reh:
            j = rng.randint(0, len(T.reh_x), n_reh)
            z = torch.cat([z, f(torch.as_tensor(T.reh_x[j]).to(dev).float())])
            t = torch.cat([t, T.tgt[T.reh_t[j]]])
        gal = torch.cat([t, T.tgt[torch.randint(T.n, (n_neg,), device=dev)]])
        loss = F.cross_entropy(f.softplus(f.logit_scale) * z @ gal.T, torch.arange(len(z), device=dev))
        if anchor > 0:
            loss = loss + anchor * sum((p - p0).pow(2).sum() for p, p0 in zip(params, starts)) / norm0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


def _unfreeze(m):
    params = [p for p in base(m).parameters()]
    for p in params:
        p.requires_grad_(True)
    return params, [p.detach().clone() for p in params]


def adapt(name, T, X, Tg, seed, steps=200, batch=64):
    """Return the calibrated model for adapter `name` given calibration inputs X (n, C, t) and targets Tg."""
    dev = U.ENV['device']
    f0 = copy.deepcopy(T.f0).requires_grad_(False)
    if name == 'f0':
        return f0.eval()
    if name == 'scratch':      # within-subject decoder from scratch on the n images only (UBP lr, 1000 steps)
        m = U.build_model('EEGProjectLayer', T.C, U.CLIP_MAP['RN50'][1]).to(dev)
        torch.manual_seed(seed)
        params = list(m.parameters())
        return _train(m, T, X, Tg, params, [p.detach().clone() for p in params], 1e-4, 0.0, 0.0, 1000, batch, seed)
    if name == 'full':
        params, starts = _unfreeze(f0)
        return _train(f0, T, X, Tg, params, starts, FULL['lr'], FULL['anchor'], FULL['rehearse'], steps, batch, seed)
    from . import mechanism2
    if mechanism2.is_mech2(name):
        return mechanism2.adapt2(name, T, X, Tg, seed, steps, batch)
    if name.startswith('subx_'):
        from . import subspace
        kind, k, lr = parse_subx(name)
        m, params = subspace.insert(f0, subspace.basis(T, kind, k=k, seed=seed, v2=True))
        return _train(m, T, X, Tg, params, [p.detach().clone() for p in params], lr, SUBX['anchor'], 0.0,
                      steps, batch, seed)
    if name.startswith('sub_'):
        from . import subspace
        m, params = subspace.insert(f0, subspace.basis(T, name))
        return _train(m, T, X, Tg, params, [p.detach().clone() for p in params], SUB['lr'], SUB['anchor'], 0.0,
                      steps, batch, seed)
    if name in LAYER:
        c = LAYER[name]
        if name in ('lowrank_first', 'svd_first', 'lora_first'):
            cls = dict(lowrank_first=LowRankLinear, svd_first=SVDLinear, lora_first=LoRALinear)[name]
            f0.model[0] = cls(f0.model[0]).to(dev)
            params = [p for p in f0.model[0].parameters()]
        else:
            mods = [f0.model[0]] if name == 'first_layer' else [f0.model[1], f0.model[2]]
            params = [p for mod in mods for p in mod.parameters()]
            for p in params:
                p.requires_grad_(True)
        return _train(f0, T, X, Tg, params, [p.detach().clone() for p in params], c['lr'], c['anchor'], 0.0,
                      steps, batch, seed)
    if name in ('sulora_f0', 'sulora', 'sulora2_f0', 'sulora2'):
        jf0, Wp = joint_source(T) if name in ('sulora_f0', 'sulora') else joint_source2(T)
        m = Aligned(copy.deepcopy(jf0).requires_grad_(False), Wp).to(dev)
        if name.endswith('_f0'):
            return m.eval()
        return _train(m, T, X, Tg, [m.W], [m.W.detach().clone()], ALIGN['lr'], ALIGN['anchor'], 0.0, steps, batch,
                      seed)
    W0 = msqrt(T.R_src, 0.5) @ msqrt(T.R_tgt, -0.5) if name == 'ea' else np.eye(T.C)
    m = Aligned(f0, W0, bias=name == 'align_bias').to(dev)
    if name == 'ea':
        return m.eval()
    params = [m.W] + ([m.b] if m.b is not None else [])
    m = _train(m, T, X, Tg, params, [p.detach().clone() for p in params], ALIGN['lr'], ALIGN['anchor'], 0.0,
               steps, batch, seed)
    if name == 'align_full':
        m.W.requires_grad_(False)
        params, starts = _unfreeze(m)
        m = _train(m, T, X, Tg, params, starts, FULL['lr'], FULL['anchor'], FULL['rehearse'], steps, batch, seed)
    return m


def joint_source(T):
    """SuLoRA-style joint tuning, cached per (target, montage): start from f0, give every source subject its own
    channel map W_s (from identity) and train the maps and f0 together on the source subjects' pool images
    (the rehearsal samples, ~2,200 images per subject). Returns (jointly tuned f0, mean source map)."""
    import os
    dev = U.ENV['device']
    path = os.path.join(U.ENV['ckpt_dir'], f"sulora_t{T.target:02d}_{T.montage}_{U.ENV['source']}.pt")
    f = copy.deepcopy(T.f0)
    if os.path.exists(path):
        ck = torch.load(path, weights_only=False, map_location=dev)
        f.load_state_dict(ck['state_dict'])
        return f.eval(), ck['W_prior']
    subs = np.unique(T.reh_s)
    Ws = nn.Parameter(torch.eye(T.C, device=dev).repeat(len(subs), 1, 1))
    params = list(f.parameters())
    for p in params:
        p.requires_grad_(True)
    starts = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW([dict(params=params, lr=JOINT['lr_f0']), dict(params=[Ws], lr=JOINT['lr_w'])],
                            weight_decay=1e-4)
    norm0, I = sum(p0.pow(2).sum() for p0 in starts), torch.eye(T.C, device=dev)
    sid = torch.as_tensor(np.searchsorted(subs, T.reh_s), device=dev)
    torch.manual_seed(0)
    f.train()
    for _ in range(JOINT['steps']):
        j = torch.randint(len(T.reh_x), (JOINT['batch'],), device=dev)
        x = torch.as_tensor(T.reh_x[j.cpu().numpy()]).to(dev).float()
        x = torch.einsum('ncd,ndt->nct', Ws[sid[j]], x).contiguous()
        t = T.tgt[torch.as_tensor(T.reh_t, device=dev)[j]]
        gal = torch.cat([t, T.tgt[torch.randint(T.n, (1024,), device=dev)]])
        loss = F.cross_entropy(f.softplus(f.logit_scale) * f(x) @ gal.T, torch.arange(len(x), device=dev))
        loss = loss + JOINT['anchor_f0'] * sum((p - p0).pow(2).sum() for p, p0 in zip(params, starts)) / norm0 \
            + JOINT['anchor_w'] * (Ws - I).pow(2).sum() / (len(subs) * T.C)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    W_prior = Ws.detach().mean(0).cpu().numpy()
    torch.save(dict(state_dict=f.state_dict(), W_prior=W_prior), path)
    print(f'  joint source tuning saved: {path} | mean source map shift {np.linalg.norm(W_prior - np.eye(T.C)) / np.sqrt(T.C):.3f}',
          flush=True)
    return f.eval(), W_prior


def joint_source2(T):
    """v2 of the SuLoRA-style joint tuning: same idea (one channel map per source subject, trained together with f0),
    but on every source subject's FULL pool and with f0 at its own training learning rate (1e-5), so f0 is refined
    rather than overwritten. Cached per (target, montage)."""
    import os
    dev = U.ENV['device']
    path = os.path.join(U.ENV['ckpt_dir'], f"sulora2_t{T.target:02d}_{T.montage}_{U.ENV['source']}.pt")
    f = copy.deepcopy(T.f0)
    if os.path.exists(path):
        ck = torch.load(path, weights_only=False, map_location=dev)
        f.load_state_dict(ck['state_dict'])
        return f.eval(), ck['W_prior']
    pos = {p: i for i, p in enumerate(T.P)}
    srcs = [s for s in U.ENV['pool_subjects'] if s != T.target]
    xs, ts, ss = [], [], []
    for k, s in enumerate(srcs):
        d = U.subject_montage(s, T.montage)
        rows = np.array([pos.get(p, -1) for p in d['train_paths']])
        keep = rows >= 0
        xs.append(d['train'][keep].astype(np.float16)); ts.append(rows[keep]); ss.append(np.full(keep.sum(), k))
    X, tix, sid = np.concatenate(xs), torch.as_tensor(np.concatenate(ts), device=dev), torch.as_tensor(np.concatenate(ss), device=dev)
    Ws = nn.Parameter(torch.eye(T.C, device=dev).repeat(len(srcs), 1, 1))
    params = list(f.parameters())
    for p in params:
        p.requires_grad_(True)
    starts = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW([dict(params=params, lr=JOINT2['lr_f0']), dict(params=[Ws], lr=JOINT2['lr_w'])],
                            weight_decay=1e-4)
    norm0, I = sum(p0.pow(2).sum() for p0 in starts), torch.eye(T.C, device=dev)
    torch.manual_seed(0)
    f.train()
    for _ in range(JOINT2['steps']):
        j = torch.randint(len(X), (JOINT2['batch'],))
        x = torch.as_tensor(X[j.numpy()]).to(dev).float()
        jd = j.to(dev)
        x = torch.einsum('ncd,ndt->nct', Ws[sid[jd]], x).contiguous()
        t = T.tgt[tix[jd]]
        gal = torch.cat([t, T.tgt[torch.randint(T.n, (1024,), device=dev)]])
        loss = F.cross_entropy(f.softplus(f.logit_scale) * f(x) @ gal.T, torch.arange(len(x), device=dev))
        loss = loss + JOINT2['anchor_f0'] * sum((p - p0).pow(2).sum() for p, p0 in zip(params, starts)) / norm0 \
            + JOINT2['anchor_w'] * (Ws - I).pow(2).sum() / (len(srcs) * T.C)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    W_prior = Ws.detach().mean(0).cpu().numpy()
    torch.save(dict(state_dict=f.state_dict(), W_prior=W_prior), path)
    print(f'  joint source tuning v2 saved: {path}', flush=True)
    return f.eval(), W_prior


def w_shift(m):
    """How far the learned channel map moved: ||W - W0||_F / ||W0||_F (0 for adapters without a map)."""
    if not isinstance(m, Aligned):
        return 0.0
    return float((m.W.detach() - m.W0).norm() / m.W0.norm())
