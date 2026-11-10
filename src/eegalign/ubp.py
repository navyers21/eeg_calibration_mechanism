"""ubpkd: channel-reduction + distillation experiments on top of the official UBP code (CVPR 2025).

Reuses UBP's EEG encoders (base/eeg_backbone.py) and blur transforms (base/inpating_data.py) and
re-implements its training loop in plain PyTorch so that we control channel selection, the validation
split, the image-feature cache and the distillation losses.

Fidelity to UBP's main.py (commit 2e513ae):
  * loss: CLIP loss on the UNnormalised EEG embedding vs L2-normalised image features, scale = softplus(logit_scale)
  * AdamW(lr 1e-4, wd 1e-4), batch 1024, max 50 epochs, early stopping on epoch train loss (min_delta 1e-3, patience 5)
  * UBP = FoveaBlur(kernel 51, exp curve, system_g 3) with uncertainty-aware kernels 51-c / 51 / 51+c, c = 6,
    moving average gamma 0.3, alpha 0.05; labels take effect from the next epoch (as with UBP's dataloader workers)
  * the test gallery uses the medium blur (as UBP); we ALSO score every run against an unblurred gallery
Deliberate differences (fixes): channel selection can happen before whitening (source='clean_mvnn'); a held-out
validation split by concept (protocol='dev'); LOSO model selection on held-out training concepts instead of the
other subjects' test images; the image-feature cache key includes the blur settings; UBP's validation pass, which
overwrote the uncertainty state of the first 200 training samples with test-set similarities, is not replicated.
"""
import os, sys, json, time, csv, random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

VERSION = 'ubpkd-0.1'

# ----------------------------------------------------------------------------------------------------------------
# Channels and montages
# ----------------------------------------------------------------------------------------------------------------
CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
            'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
            'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
            'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
            'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
            'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'O1', 'Oz', 'O2']
POSTERIOR17 = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
CROWN8 = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']   # verify against the Neurosity spec
NESTED16 = CROWN8 + ['O1', 'Oz', 'O2', 'POz', 'Pz', 'P3', 'P4', 'Cz']
NESTED32 = NESTED16 + ['PO7', 'PO8', 'P7', 'P8', 'P1', 'P2', 'P5', 'P6', 'CP1', 'CP2', 'CPz', 'C1', 'C2', 'FCz', 'F3', 'F4']
MONTAGES = {
    'all': CHANNELS,
    'nested32': NESTED32,
    'nested16': NESTED16,
    'posterior17': POSTERIOR17,
    'occipital8': ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8'],
    'crown8': CROWN8,
    'crown_po2': ['PO3', 'PO4'],
    'crown_ant6': ['CP3', 'C3', 'F5', 'F6', 'C4', 'CP4'],
}
for _m, _chs in MONTAGES.items():
    assert len(set(_chs)) == len(_chs) and all(c in CHANNELS for c in _chs), _m

# UBP paper, Table 5 (EEGProject + CLIP RN50, 17 posterior channels, top-1 per subject 1..10)
PAPER_TABLE5 = {
    'vanilla': [30.8, 39.5, 42.2, 42.8, 33.8, 45.5, 40.7, 49.8, 41.7, 52.5],
    'ubp': [41.2, 50.7, 50.2, 49.3, 43.3, 58.7, 49.8, 57.8, 44.5, 62.3],
}

CLIP_MAP = {
    'RN50': ('openai', 1024), 'RN101': ('openai', 512),
    'ViT-B-16': ('laion2b_s34b_b88k', 512), 'ViT-B-32': ('laion2b_s34b_b79k', 512),
    'ViT-L-14': ('laion2b_s32b_b82k', 768), 'ViT-H-14': ('laion2b_s32b_b79k', 1024),
}
UBP_KERNEL, UBP_G, UBP_C, UBP_GAMMA, UBP_ALPHA = 51, 3, 6, 0.3, 0.05
T_LEN = 250

# ----------------------------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------------------------
ENV = dict(ubp_dir='/content/ubp', data_root=None, image_root=None, whiten_dir=None, nowhiten_dir=None,
           cache_dir='/content/cache', feat_dir=None, ckpt_dir=None, results_csv=None, source='hf_whiten',
           device='cuda' if torch.cuda.is_available() else 'cpu', ubp_commit='unknown', num_workers=8)


def setup(**kw):
    ENV.update(kw)
    if ENV['ubp_dir'] not in sys.path:
        sys.path.insert(0, ENV['ubp_dir'])
    for k in ('cache_dir', 'feat_dir', 'ckpt_dir'):
        if ENV.get(k):
            os.makedirs(ENV[k], exist_ok=True)
    return ENV


def _eeg_dir(source):
    return ENV['whiten_dir'] if source == 'hf_whiten' else ENV['nowhiten_dir']


# ----------------------------------------------------------------------------------------------------------------
# Montage-specific multivariate noise normalisation (the fix for channel leakage)
# ----------------------------------------------------------------------------------------------------------------
def lw_cov_batched(X):
    """sklearn.discriminant_analysis._cov(x, shrinkage='auto') for a batch. X: (n, T, C) float64 tensor.
    = StandardScaler -> Ledoit-Wolf -> rescale, exactly as THINGS-EEG2 / UBP preprocessing does per epoch."""
    n_s, p = X.shape[1], X.shape[2]
    Xc = X - X.mean(1, keepdim=True)
    scale = Xc.pow(2).mean(1).sqrt()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    Xs = Xc / scale[:, None, :]
    XtX = Xs.transpose(1, 2) @ Xs
    emp = XtX / n_s
    X2 = Xs ** 2
    emp_tr = X2.sum(1) / n_s
    mu = emp_tr.sum(-1) / p
    beta_ = (X2.transpose(1, 2) @ X2).sum((1, 2))
    delta_ = (XtX ** 2).sum((1, 2)) / n_s ** 2
    beta = 1.0 / (p * n_s) * (beta_ / n_s - delta_)
    delta = (delta_ - 2.0 * mu * emp_tr.sum(-1) + p * mu ** 2) / p
    beta = torch.minimum(beta, delta)
    shrink = torch.where(beta == 0, torch.zeros_like(beta), beta / delta)
    eye = torch.eye(p, dtype=X.dtype, device=X.device)
    shrunk = (1 - shrink)[:, None, None] * emp + (shrink * mu)[:, None, None] * eye
    return scale[:, :, None] * shrunk * scale[:, None, :]


def inv_sqrt(S):
    w, V = torch.linalg.eigh(S)
    return (V * w.clamp_min(1e-12).rsqrt()) @ V.T


def mvnn_montage(train_eeg, train_ses, test_eeg, test_ses, chunk=4096):
    """Per-session whitening fitted on the TRAINING trials of the selected channels only.
    train_eeg: (n_img, n_rep, C, T) float16 numpy; *_ses: session id per (img, rep)."""
    dev = ENV['device']
    out_tr = np.empty(train_eeg.shape, np.float16)
    out_te = np.empty(test_eeg.shape, np.float16)
    for s in np.unique(train_ses):
        mi, mr = np.nonzero(train_ses == s)
        acc, n = None, 0
        for a in range(0, len(mi), chunk):
            x = torch.from_numpy(train_eeg[mi[a:a + chunk], mr[a:a + chunk]].astype(np.float64)).to(dev)
            c = lw_cov_batched(x.transpose(1, 2)).sum(0)
            acc = c if acc is None else acc + c
            n += x.shape[0]
        W = inv_sqrt(acc / n).float()
        for eeg, ses, out in ((train_eeg, train_ses, out_tr), (test_eeg, test_ses, out_te)):
            ii, rr = np.nonzero(ses == s)
            for a in range(0, len(ii), chunk):
                x = torch.from_numpy(eeg[ii[a:a + chunk], rr[a:a + chunk]].astype(np.float32)).to(dev)
                out[ii[a:a + chunk], rr[a:a + chunk]] = torch.einsum('ij,njt->nit', W, x).cpu().numpy().astype(np.float16)
    return out_tr, out_te


# ----------------------------------------------------------------------------------------------------------------
# EEG data
# ----------------------------------------------------------------------------------------------------------------
_RAW = OrderedDict()


def _load_raw(sub, mode, source):
    key = (sub, mode, source)
    if key not in _RAW:
        while len(_RAW) >= 2:
            _RAW.popitem(last=False)
        path = os.path.join(_eeg_dir(source), f'sub-{sub:02d}', f'{mode}.pt')
        _RAW[key] = torch.load(path, weights_only=False)
    return _RAW[key]


def subject_montage(sub, montage, source=None, keys=None):
    """Cached per-subject arrays for one montage: averaged train / test, single-trial test, paths, labels.
    keys: load only these arrays from the cache (saves RAM at 63 channels)."""
    source = source or ENV['source']
    base = ENV.get('local_cache_dir') if montage in ENV.get('local_montages', ()) else ENV['cache_dir']
    path = os.path.join(base, source, f'sub-{sub:02d}', f'{montage}.npz')
    if not os.path.exists(path):
        idx = [CHANNELS.index(c) for c in MONTAGES[montage]]
        tr, te = _load_raw(sub, 'train', source), _load_raw(sub, 'test', source)
        names = list(tr.get('ch_names', CHANNELS))
        assert names == CHANNELS, f'unexpected channel order in data: {names[:5]}...'
        tr_eeg = np.asarray(tr['eeg'])[:, :, idx, :T_LEN].astype(np.float16)
        te_eeg = np.asarray(te['eeg'])[:, :, idx, :T_LEN].astype(np.float16)
        if source == 'clean_mvnn':
            tr_eeg, te_eeg = mvnn_montage(tr_eeg, np.asarray(tr['session']), te_eeg, np.asarray(te['session']))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 train=tr_eeg.astype(np.float32).mean(1).astype(np.float16),
                 test=te_eeg.astype(np.float32).mean(1).astype(np.float16),
                 test_trials=te_eeg,
                 train_paths=np.asarray(tr['img'])[:, 0].astype(str),
                 train_labels=np.asarray(tr['label'])[:, 0].astype(np.int64),
                 test_paths=np.asarray(te['img'])[:, 0].astype(str))
    d = np.load(path)
    return {k: d[k] for k in d.files if keys is None or k in keys}


def free_raw():
    _RAW.clear()


def val_concepts(labels, seed=2026):
    concepts = np.unique(labels)
    n_val = 200 if len(concepts) >= 1000 else max(2, len(concepts) // 8)
    return np.sort(np.random.RandomState(seed).choice(concepts, n_val, replace=False))


def split_indices(labels, protocol):
    """dev: hold out ALL images of 200 concepts; validate 200-way on the first image of each held-out concept."""
    if protocol == 'paper':
        return np.arange(len(labels)), None
    vc = val_concepts(labels)
    is_val = np.isin(labels, vc)
    val = np.array([np.nonzero(labels == c)[0][0] for c in vc])
    return np.nonzero(~is_val)[0], val


def cal_indices(labels, train_idx, frac=0.1, seed=2027):
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(train_idx, int(round(frac * len(train_idx))), replace=False))


def build_data(cfg, montage):
    """Training / validation / test arrays for a run, in a deterministic order (shared by teacher and student)."""
    proto = cfg['protocol']
    if cfg.get('loso_target') is None:
        d = subject_montage(cfg['subject'], montage)
        tr, va = split_indices(d['train_labels'], proto)
        out = dict(x_train=d['train'][tr], p_train=d['train_paths'][tr])
        if va is not None:
            out.update(x_val=d['train'][va], p_val=d['train_paths'][va], val_gallery=d['train_paths'][va],
                       y_val=np.arange(len(va)))
        out.update(x_test=d['test'], x_test_trials=d['test_trials'], p_test=d['test_paths'])
        return out
    tgt, cal = cfg['loso_target'], cfg.get('cal')
    xs, ps, xv, yv = [], [], [], []
    gallery = None
    srcs = [] if cal == 'cal10only' else [s for s in cfg['pool_subjects'] if s != tgt]
    # Memory-lean (needed at 63 channels): load only the training arrays, write each source straight into one
    # preallocated array instead of concatenating copies. Same data and order as before.
    lab = {s: subject_montage(s, montage, keys=('train_labels',))['train_labels'] for s in srcs}
    sizes = [len(split_indices(lab[s], 'dev')[0]) for s in srcs]
    X0 = None
    for k, s in enumerate(srcs):
        d = subject_montage(s, montage, keys=('train', 'train_paths', 'train_labels'))
        tr, va = split_indices(d['train_labels'], 'dev')
        if X0 is None:
            X0 = np.empty((sum(sizes),) + d['train'].shape[1:], d['train'].dtype)
        X0[sum(sizes[:k]):sum(sizes[:k + 1])] = d['train'][tr]
        ps.append(d['train_paths'][tr])
        xv.append(d['train'][va]); yv.append(np.arange(len(va)))
        gallery = d['train_paths'][va]
        del d
    if X0 is not None:
        xs.append(X0)
    d = subject_montage(tgt, montage)
    tr_t, va_t = split_indices(d['train_labels'], 'dev')
    if cal in ('cal10', 'cal10only'):
        ci = cal_indices(d['train_labels'], tr_t)
        xs.append(d['train'][ci]); ps.append(d['train_paths'][ci])
    if cal == 'cal10only':          # validate on the target's own held-out concepts (no test images)
        xv.append(d['train'][va_t]); yv.append(np.arange(len(va_t))); gallery = d['train_paths'][va_t]
    return dict(x_train=xs[0] if len(xs) == 1 else np.concatenate(xs), p_train=np.concatenate(ps), x_val=np.concatenate(xv),
                y_val=np.concatenate(yv), val_gallery=gallery,
                x_test=d['test'], x_test_trials=d['test_trials'], p_test=d['test_paths'])


# ----------------------------------------------------------------------------------------------------------------
# Image features (CLIP), with the blur in the cache key
# ----------------------------------------------------------------------------------------------------------------
def method_specs(method):
    """Returns (training spec(s), test-gallery spec). Specs: 'none' | 'fovea:K:G' | 'uniform:K'."""
    fov = lambda k: f'fovea:{k}:{UBP_G}'
    if method == 'vanilla':
        return ['none'], 'none'
    if method == 'fovea':
        return [fov(UBP_KERNEL)], fov(UBP_KERNEL)
    if method == 'ubp':
        return [fov(UBP_KERNEL - UBP_C), fov(UBP_KERNEL), fov(UBP_KERNEL + UBP_C)], fov(UBP_KERNEL)
    if method.startswith('uniform:'):
        k = int(method.split(':')[1])
        s = 'none' if k == 0 else f'uniform:{k}'
        return [s], s
    raise ValueError(method)


def _make_blur(spec):
    if spec == 'none':
        return None
    from base.inpating_data import FoveaBlur, UniformBlur
    kind, *args = spec.split(':')
    if kind == 'fovea':
        return FoveaBlur(h=224, w=224, blur_kernel_size=int(args[0]), curve_type='exp', system_g=float(args[1]))
    if kind == 'uniform':
        return UniformBlur(int(args[0]))
    raise ValueError(spec)


def all_image_paths():
    paths = []
    for split in ('train_images', 'test_images'):
        root = os.path.join(ENV['image_root'], split)
        for folder in sorted(os.listdir(root)):
            fp = os.path.join(root, folder)
            if os.path.isdir(fp):
                paths += [f'{split}/{folder}/{f}' for f in sorted(os.listdir(fp))
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return paths


class _ImgDS(torch.utils.data.Dataset):
    def __init__(self, paths, spec):
        from torchvision import transforms
        self.paths, self.blur = paths, _make_blur(spec)
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(os.path.join(ENV['image_root'], self.paths[i])).convert('RGB')
        if self.blur is not None:
            img = self.blur(img)
        return self.tf(img)


@torch.no_grad()
def _encode_images(paths, spec, vision):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(vision, pretrained=CLIP_MAP[vision][0], device=ENV['device'])
    model.eval()
    dl = torch.utils.data.DataLoader(_ImgDS(paths, spec), batch_size=128, num_workers=ENV['num_workers'])
    out = []
    for x in dl:
        f = model.encode_image(x.to(ENV['device']))
        out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out)


_FEATS = {}


def image_features(spec, vision='RN50'):
    key = (spec, vision)
    if key not in _FEATS:
        path = os.path.join(ENV['feat_dir'], f"{vision}__{spec.replace(':', '-')}.pt")
        if os.path.exists(path):
            d = torch.load(path)
        else:
            t0 = time.time()
            paths = all_image_paths()
            d = {'paths': paths, 'feats': _encode_images(paths, spec, vision).half()}
            torch.save(d, path)
            print(f'  image features {vision} {spec}: {len(paths)} images, {time.time() - t0:.0f}s', flush=True)
        d['index'] = {p: i for i, p in enumerate(d['paths'])}
        _FEATS[key] = d
    return _FEATS[key]


def feats_for(paths, spec, vision):
    d = image_features(spec, vision)
    rows = torch.tensor([d['index'][str(p)] for p in paths])
    return d['feats'][rows].float()


# ----------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------
def build_model(backbone, c_num, z_dim):
    from base import eeg_backbone as eb
    return getattr(eb, backbone)(z_dim=z_dim, c_num=c_num, timesteps=[0, T_LEN])


def first_linear(model):
    from base import eeg_backbone as eb
    return model.model[0] if isinstance(model, eb.EEGProjectLayer) else model.project[1]


def channel_param(model, c_num):
    """Name of the only parameter whose shape depends on the input channels."""
    from base import eeg_backbone as eb
    if isinstance(model, eb.EEGProjectLayer):
        return 'model.0.weight'
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and tuple(m.kernel_size) == (c_num, 1):
            return name + '.weight'
    raise ValueError('no spatial layer found')


def init_from_teacher(student, s_ch, teacher, t_ch):
    """Copy all teacher weights; for the channel-dependent layer copy shared channels, keep the rest random."""
    from base import eeg_backbone as eb
    sd_s, sd_t = student.state_dict(), teacher.state_dict()
    ks, kt = channel_param(student, len(s_ch)), channel_param(teacher, len(t_ch))
    assert ks == kt
    n_shared = 0
    for k, v in sd_t.items():
        if k == ks:
            w = sd_s[k].clone()
            for i, ch in enumerate(s_ch):
                if ch in t_ch:
                    j = t_ch.index(ch); n_shared += 1
                    if isinstance(student, eb.EEGProjectLayer):
                        w[:, i * T_LEN:(i + 1) * T_LEN] = v[:, j * T_LEN:(j + 1) * T_LEN]
                    else:
                        w[:, :, i] = v[:, :, j]
            sd_s[k] = w
        elif k in sd_s and sd_s[k].shape == v.shape:
            sd_s[k] = v.clone()
    student.load_state_dict(sd_s)
    return n_shared


@torch.no_grad()
def embed(model, X, bs=4096, hidden=False):
    model.eval()
    zs, hs = [], []
    store = {}
    h = first_linear(model).register_forward_hook(lambda m, i, o: store.__setitem__('h', o)) if hidden else None
    for a in range(0, len(X), bs):
        x = torch.as_tensor(X[a:a + bs]).to(ENV['device']).float()
        zs.append(model(x).float())
        if hidden:
            hs.append(store['h'].float())
    if h is not None:
        h.remove()
    return (torch.cat(zs), torch.cat(hs)) if hidden else torch.cat(zs)


@torch.no_grad()
def retrieval(model, X, gallery, labels):
    z = F.normalize(embed(model, X), dim=-1)
    g = F.normalize(gallery.to(z.device), dim=-1)
    top = (z @ g.T).topk(5, dim=-1).indices.cpu()
    y = torch.as_tensor(labels)
    return 100 * (top[:, 0] == y).float().mean().item(), 100 * (top == y[:, None]).any(1).float().mean().item()


# ----------------------------------------------------------------------------------------------------------------
# Run definitions
# ----------------------------------------------------------------------------------------------------------------
BB_SHORT = {'EEGProjectLayer': 'proj', 'TSconv': 'tsconv', 'Shallownet': 'shallow', 'EEGnet': 'eegnet', 'Deepnet': 'deep'}


def teacher_spec(montage, method='ubp', backbone='EEGProjectLayer', seed=0, short=None):
    return dict(montage=montage, method=method, backbone=backbone, seed=seed,
                short=short or {'posterior17': 't17', 'all': 't63'}.get(montage, f'ba-{montage}'))


def make_cfg(group, montage, method='ubp', backbone='EEGProjectLayer', kd='none', teacher=None, init=False,
             ua=None, baseline=None, cal=None, label=None, **kw):
    ua = ua or ('student' if method == 'ubp' else 'none')
    name = label or '-'.join(x for x in [montage, BB_SHORT[backbone], method,
                                         None if ua in ('student', 'none') else f'ua_{ua}',
                                         None if kd == 'none' else kd,
                                         teacher['short'] if teacher else None,
                                         'init' if init else None, cal] if x)
    cfg = dict(group=group, name=f'{group}/{name}', montage=montage, method=method, backbone=backbone, kd=kd,
               teacher=teacher, init=init, ua=ua, baseline=baseline, cal=cal, vision='RN50', protocol='dev',
               epochs=50, lr=1e-4, batch=1024, early_stop='train_loss', seed=0, subject=None, loso_target=None,
               pool_subjects=list(ENV.get('pool_subjects', range(1, 11))), single_trial=True, save=False, kd_tau=4.0)
    cfg.update(kw)
    return cfg


def run_tag(cfg):
    who = f"loso{cfg['loso_target']:02d}" if cfg.get('loso_target') else f"s{cfg['subject']:02d}"
    return f"{cfg['name']}|{who}|seed{cfg['seed']}|{cfg['protocol']}|{ENV['source']}"


def teacher_cfg(spec, student_cfg):
    c = make_cfg('teachers', spec['montage'], method=spec['method'], backbone=spec['backbone'],
                 label=f"{spec['short']}-{spec['montage']}-{BB_SHORT[spec['backbone']]}-{spec['method']}",
                 seed=spec['seed'], save=True, single_trial=False)
    c.update(ENV.get('overrides', {}))
    for k in ('subject', 'loso_target', 'pool_subjects', 'protocol', 'vision'):
        c[k] = student_cfg[k]
    c['cal'] = None                 # LOSO teachers are trained on the other subjects only
    if c['loso_target'] is not None:
        c.update(lr=student_cfg['lr'], early_stop='val')
    return c


_TEACHERS = OrderedDict()


def ensure_teacher(spec, student_cfg):
    tc = teacher_cfg(spec, student_cfg)
    tag = run_tag(tc)
    if tag in _TEACHERS:
        return _TEACHERS[tag]
    ck = os.path.join(ENV['ckpt_dir'], tag.replace('/', '_').replace('|', '__') + '.pt')
    if not os.path.exists(ck):
        print(f'  training teacher {tag}', flush=True)
        res = run(tc)
        append_result(res)
    blob = torch.load(ck, weights_only=False)
    m = build_model(spec['backbone'], len(MONTAGES[spec['montage']]), CLIP_MAP[tc['vision']][1]).to(ENV['device'])
    m.load_state_dict(blob['state_dict'])
    m.eval()
    while len(_TEACHERS) >= 4:
        _TEACHERS.popitem(last=False)
    _TEACHERS[tag] = (m, tc, blob.get('metrics', {}))
    return _TEACHERS[tag]


# ----------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------
def _seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run(cfg):
    t0 = time.time()
    dev = ENV['device']
    _seed_all(cfg['seed'])
    chans = MONTAGES[cfg['montage']]
    z_dim = CLIP_MAP[cfg['vision']][1]
    data = build_data(cfg, cfg['montage'])
    X = torch.from_numpy(data['x_train']).to(dev)                  # float16, cast per batch
    N = X.shape[0]
    train_specs, test_spec = method_specs(cfg['method'])
    Ftr = torch.stack([feats_for(data['p_train'], s, cfg['vision']) for s in train_specs]).to(dev)  # (L, N, D)
    medium = len(train_specs) // 2
    gallery = {'method': feats_for(data['p_test'], test_spec, cfg['vision']),
               'clean': feats_for(data['p_test'], 'none', cfg['vision'])}
    y_test = np.arange(len(data['p_test']))
    val_gallery = feats_for(data['val_gallery'], test_spec, cfg['vision']) if 'x_val' in data else None

    teacher_info, T_z, T_h, t_scale, tm = {}, None, None, None, None
    if cfg['teacher'] is not None:
        tm, tc, tmet = ensure_teacher(cfg['teacher'], cfg)
        teacher_info = dict(teacher_tag=run_tag(tc), teacher_top1=tmet.get('test_top1'))
        tdata = build_data(cfg, cfg['teacher']['montage'])
        assert (tdata['p_train'] == data['p_train']).all()
        need_h = cfg['kd'] in ('feature', 'all')
        out = embed(tm, tdata['x_train'], hidden=need_h)
        T_z, T_h = out if need_h else (out, None)
        t_scale = tm.softplus(tm.logit_scale).item()
    _seed_all(cfg['seed'])          # re-seed: training a teacher above consumes RNG state
    model = build_model(cfg['backbone'], len(chans), z_dim).to(dev)
    if cfg['init'] and tm is not None:
        teacher_info['init_shared_ch'] = init_from_teacher(model, chans, tm, MONTAGES[cfg['teacher']['montage']])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)

    # uncertainty-aware state
    sim = np.ones(N)
    labels = np.full(N, 1, dtype=np.int64)            # 0 low blur / 1 medium / 2 high blur
    if cfg['ua'] == 'teacher':
        s_t = t_scale * (T_z * Ftr[medium]).sum(-1).cpu().numpy()
        z = norm.ppf(1 - UBP_ALPHA / 2)
        lo, hi = s_t.mean() - z * s_t.std(ddof=1), s_t.mean() + z * s_t.std(ddof=1)
        labels[s_t > hi], labels[s_t < lo] = 0, 2
    hidden = {}
    hook = first_linear(model).register_forward_hook(lambda m, i, o: hidden.__setitem__('h', o)) \
        if cfg['kd'] in ('feature', 'all') else None

    best_loss, best_val, wait, best_state, best_state_epoch, hist = float('inf'), -1.0, 0, None, None, []
    epoch = 0
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        tags = torch.from_numpy(labels.copy() if len(train_specs) == 3 else np.zeros(N, np.int64)).to(dev)
        perm = torch.randperm(N, device=dev)
        tot, cnt = 0.0, 0
        for a in range(0, N, cfg['batch']):
            idx = perm[a:a + cfg['batch']]
            x = X[idx].float()
            tgt = Ftr[tags[idx], idx]
            z = model(x)
            scale = model.softplus(model.logit_scale)
            logits = scale * z @ tgt.T
            lab = torch.arange(len(idx), device=dev)
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            if cfg['ua'] == 'student':
                ii = idx.cpu().numpy()
                diag = logits.diagonal().detach().cpu().numpy()
                bs_ = UBP_GAMMA * diag + (1 - UBP_GAMMA) * sim[ii]
                if len(ii) > 1:
                    zc = norm.ppf(1 - UBP_ALPHA / 2)
                    m_, sd_ = bs_.mean(), bs_.std(ddof=1)
                    ml = np.ones(len(ii), np.int64)
                    ml[diag > m_ + zc * sd_] = 0
                    ml[diag < m_ - zc * sd_] = 2
                    labels[ii] = ml
                sim[ii] = bs_
            if T_z is not None and cfg['kd'] != 'none':
                tz = T_z[idx]
                if cfg['kd'] in ('emb', 'response', 'imitate', 'all'):
                    l_emb = (1 - F.cosine_similarity(z, tz, dim=-1)).mean()
                    loss = l_emb if cfg['kd'] == 'imitate' else loss + l_emb
                if cfg['kd'] in ('kl', 'response', 'all'):
                    tau = cfg['kd_tau']
                    t_logits = t_scale * tz @ tgt.T
                    loss = loss + F.kl_div(F.log_softmax(logits / tau, -1), F.softmax(t_logits / tau, -1),
                                           reduction='batchmean') * tau ** 2
                if cfg['kd'] in ('feature', 'all'):
                    loss = loss + (1 - F.cosine_similarity(hidden['h'], T_h[idx], dim=-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx); cnt += len(idx)
        ep_loss = tot / cnt
        rec = dict(epoch=epoch, loss=ep_loss)
        if val_gallery is not None:
            rec['val_top1'], rec['val_top5'] = retrieval(model, data['x_val'], val_gallery, data['y_val'])
            if rec['val_top1'] > best_val:
                best_val = rec['val_top1']
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_state_epoch = epoch
        hist.append(rec)
        if cfg['early_stop'] == 'train_loss':
            if best_loss - ep_loss > 1e-3:
                best_loss, wait = ep_loss, 0
            else:
                wait += 1
        elif cfg['early_stop'] == 'val':
            wait = 0 if rec['val_top1'] >= best_val else wait + 1
        if cfg['early_stop'] in ('train_loss', 'val') and wait >= 5:
            break
    if hook is not None:
        hook.remove()

    res = dict(tag=run_tag(cfg), epochs_run=epoch, train_loss=hist[-1]['loss'], **teacher_info)
    if cfg['early_stop'] == 'val' and best_state is not None:      # LOSO: select on held-out training concepts
        model.load_state_dict(best_state)
    res['test_top1'], res['test_top5'] = retrieval(model, data['x_test'], gallery['method'], y_test)
    res['test_clean_top1'], res['test_clean_top5'] = retrieval(model, data['x_test'], gallery['clean'], y_test)
    if val_gallery is not None:
        res['val_top1'], res['val_top5'] = hist[-1]['val_top1'], hist[-1]['val_top5']
        res['best_val_top1'], res['best_val_epoch'] = best_val, best_state_epoch
    if cfg['single_trial']:
        xt = data['x_test_trials']
        res['st_top1'], res['st_top5'] = retrieval(model, xt.reshape(-1, *xt.shape[2:]), gallery['method'],
                                                   np.repeat(y_test, xt.shape[1]))
    if cfg['early_stop'] != 'val' and best_state is not None and cfg['protocol'] == 'dev':
        final = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        res['test_top1_at_best_val'], _ = retrieval(model, data['x_test'], gallery['method'], y_test)
        model.load_state_dict(final)
    if cfg['ua'] != 'none' and len(train_specs) == 3:
        res['ua_frac_low'], res['ua_frac_high'] = float((labels == 0).mean()), float((labels == 2).mean())
    res['wall_s'] = round(time.time() - t0, 1)
    if cfg['save']:
        ck = os.path.join(ENV['ckpt_dir'], res['tag'].replace('/', '_').replace('|', '__') + '.pt')
        torch.save({'state_dict': model.state_dict(), 'cfg': cfg, 'metrics': res, 'history': hist}, ck)
    res['cfg'] = cfg
    return res


# ----------------------------------------------------------------------------------------------------------------
# Results registry
# ----------------------------------------------------------------------------------------------------------------
COLUMNS = ['tag', 'group', 'name', 'baseline', 'subject', 'loso_target', 'seed', 'montage', 'n_ch', 'backbone',
           'method', 'ua', 'kd', 'teacher', 'init', 'cal', 'protocol', 'source', 'vision', 'lr', 'epochs_run',
           'train_loss', 'val_top1', 'val_top5', 'best_val_top1', 'best_val_epoch', 'test_top1', 'test_top5',
           'test_clean_top1', 'test_clean_top5', 'test_top1_at_best_val', 'st_top1', 'st_top5', 'teacher_tag',
           'teacher_top1', 'init_shared_ch', 'ua_frac_low', 'ua_frac_high', 'wall_s', 'ubp_commit', 'version', 'time']


def done_tags():
    p = ENV['results_csv']
    if not p or not os.path.exists(p):
        return set()
    with open(p) as f:
        return {r['tag'] for r in csv.DictReader(f)}


def append_result(res):
    cfg = res['cfg']
    row = {k: res.get(k, '') for k in COLUMNS}
    row.update(group=cfg['group'], name=cfg['name'], baseline=cfg['baseline'] or '', subject=cfg['subject'] or '',
               loso_target=cfg['loso_target'] or '', seed=cfg['seed'], montage=cfg['montage'],
               n_ch=len(MONTAGES[cfg['montage']]), backbone=cfg['backbone'], method=cfg['method'], ua=cfg['ua'],
               kd=cfg['kd'], teacher=cfg['teacher']['short'] if cfg['teacher'] else '', init=int(cfg['init']),
               cal=cfg['cal'] or '', protocol=cfg['protocol'], source=ENV['source'], vision=cfg['vision'],
               lr=cfg['lr'], ubp_commit=ENV['ubp_commit'], version=VERSION, time=time.strftime('%Y-%m-%d %H:%M'))
    for k, v in row.items():
        if isinstance(v, float):
            row[k] = round(v, 4)
    p = ENV['results_csv']
    new = not os.path.exists(p)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------------------------------------------
# Experiment groups
# ----------------------------------------------------------------------------------------------------------------
T17, T63 = teacher_spec('posterior17'), teacher_spec('all')
CH_MONTAGES = ['all', 'nested32', 'nested16', 'posterior17', 'occipital8', 'crown8', 'crown_po2', 'crown_ant6']
SHORT_MONTAGES = ['all', 'posterior17', 'occipital8', 'crown8', 'crown_po2']
BLUR_KERNELS = [0, 5, 11, 21, 31, 41]


def ch_name(montage, method, backbone='EEGProjectLayer'):
    return f'channels/{montage}-{BB_SHORT[backbone]}-{method}'


def group_cfgs(group):
    """Run configurations for one subject (seeds and subject are filled in by run_group)."""
    G = []
    if group == 'repro':      # UBP paper protocol (no held-out split, early stopping on train loss)
        for m in ('vanilla', 'ubp'):
            G.append(make_cfg('repro', 'posterior17', m, protocol='paper', label=f'posterior17-proj-{m}-paper'))
    elif group == 'channels':
        for mon in CH_MONTAGES:
            for m in ('vanilla', 'ubp'):
                G.append(make_cfg('channels', mon, m, baseline=None if m == 'vanilla' else ch_name(mon, 'vanilla')))
    elif group == 'kd':
        for mon in ('crown8', 'occipital8'):
            base = ch_name(mon, 'ubp')
            for kd, t, init in [('response', T63, False), ('response', T17, False), ('response', T17, True),
                                ('response', T63, True), ('none', T63, True), ('feature', T63, False)]:
                G.append(make_cfg('kd', mon, 'ubp', kd=kd, teacher=t, init=init, baseline=base))
            if mon == 'crown8':
                G.append(make_cfg('kd', mon, 'ubp', kd='response', teacher=teacher_spec(mon, seed=100), baseline=base))
    elif group == 'tsconv':
        for mon in SHORT_MONTAGES:
            for m in ('vanilla', 'ubp'):
                G.append(make_cfg('channels', mon, m, backbone='TSconv',
                                  baseline=None if m == 'vanilla' else ch_name(mon, 'vanilla', 'TSconv')))
    elif group == 'blur':
        for mon in SHORT_MONTAGES:
            for k in BLUR_KERNELS:
                G.append(make_cfg('blur', mon, f'uniform:{k}',
                                  baseline=None if k == 0 else f'blur/{mon}-proj-uniform:0'))
    elif group == 'tblur':
        for mon in ('crown8', 'occipital8'):
            for t in (T63, T17):
                G.append(make_cfg('tblur', mon, 'ubp', ua='teacher', teacher=t, baseline=ch_name(mon, 'ubp')))
    elif group == 'loso':
        for mon in ('crown8', 'occipital8', 'posterior17'):
            G.append(make_cfg('loso', mon, 'ubp', lr=1e-5, early_stop='val', single_trial=False))
            G.append(make_cfg('loso', mon, 'ubp', lr=1e-5, early_stop='val', cal='cal10', single_trial=False,
                              baseline=f'loso/{mon}-proj-ubp'))
            G.append(make_cfg('loso', mon, 'ubp', lr=1e-4, early_stop='val', cal='cal10only', single_trial=False))
        G.append(make_cfg('loso', 'crown8', 'ubp', lr=1e-5, early_stop='val', cal='cal10', kd='response',
                          teacher=T17, init=True, single_trial=False, baseline='loso/crown8-proj-ubp-cal10'))
    else:
        raise ValueError(group)
    return G


def run_group(group, subjects, seeds, only=None, dry_run=False):
    done = done_tags()
    for s in subjects:
        todo = []
        for seed in seeds:
            for c in group_cfgs(group):
                if only and not any(o in c['name'] for o in only):
                    continue
                c = dict(c, seed=seed, **ENV.get('overrides', {}))
                if group == 'loso':
                    c['loso_target'] = s
                else:
                    c['subject'] = s
                if run_tag(c) not in done:
                    todo.append(c)
        n_all = len(group_cfgs(group)) * len(seeds) if not only else None
        print(f'[{group}] subject {s}: {len(todo)} to run' + (f' of {n_all}' if n_all else ''), flush=True)
        if dry_run:
            continue
        for i, c in enumerate(todo, 1):
            res = run(c)
            append_result(res)
            done.add(res['tag'])
            print(f"  [{i}/{len(todo)}] {res['tag']}  ep {res['epochs_run']}  "
                  f"test {res['test_top1']:.1f}/{res['test_top5']:.1f}  clean {res['test_clean_top1']:.1f}"
                  + (f"  val {res['val_top1']:.1f}" if 'val_top1' in res else '')
                  + (f"  st {res['st_top1']:.1f}" if 'st_top1' in res else '') + f"  ({res['wall_s']:.0f}s)",
                  flush=True)
        free_raw()


# ----------------------------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------------------------
def summary(csv_path=None, metric='test_top1', source=None, protocol=None):
    import pandas as pd
    df = pd.read_csv(csv_path or ENV['results_csv'])
    if source:
        df = df[df.source == source]
    if protocol:
        df = df[df.protocol == protocol]
    df = df[df.group != 'teachers']
    df['who'] = df['subject'].fillna(df['loso_target'])
    key = ['who', 'seed', 'protocol', 'source']
    base = df[key + ['name', metric]].rename(columns={'name': 'baseline', metric: 'base_metric'})
    m = df.merge(base, on=key + ['baseline'], how='left')
    m['gain'] = m[metric] - m['base_metric']
    rows = []
    for name, g in m.groupby('name', sort=True):
        r = dict(config=name, ch=int(g.n_ch.iloc[0]), subj=g.who.nunique(), n=len(g),
                 **{metric: f'{g[metric].mean():.1f} ± {g[metric].std():.1f}' if len(g) > 1 else f'{g[metric].mean():.1f}'})
        for extra in ('test_clean_top1', 'val_top1', 'st_top1'):
            if extra in g and g[extra].notna().any() and extra != metric:
                r[extra] = f'{g[extra].mean():.1f}'
        gg = g.dropna(subset=['gain'])
        if len(gg):
            r['gain'] = f'{gg.gain.mean():+.1f} ± {gg.gain.std():.1f}' if len(gg) > 1 else f'{gg.gain.mean():+.1f}'
            r['wins'] = f'{int((gg.gain > 0).sum())}/{len(gg)}'
        rows.append(r)
    out = pd.DataFrame(rows).fillna('')
    return out


def repro_table(csv_path=None):
    import pandas as pd
    df = pd.read_csv(csv_path or ENV['results_csv'])
    df = df[df.group == 'repro']
    t = df.pivot_table(index='subject', columns='method', values='test_top1', aggfunc='mean').round(1)
    for m in ('vanilla', 'ubp'):
        if m in t:
            t[f'{m}_paper'] = [PAPER_TABLE5[m][int(s) - 1] for s in t.index]
    return t