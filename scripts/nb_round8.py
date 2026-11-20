"""Round 8: leak-free layout transfer. Virtual electrodes and distillation from a dense decoder to sparse layouts,
with the round-4-7 calibration pipeline on top, on data with no cross-channel whitening leak.

    python scripts/nb_round8.py prep    --config C --prep z                     # rebuild montages from the unwhitened release
    python scripts/nb_round8.py train   --config C --prep z  --montages posterior17 crown_po2
    python scripts/nb_round8.py distill --config C --prep z  --montages crown_po2
    python scripts/nb_round8.py eval    --config C --prep z  --montages crown_po2 --kinds repaug distill ve veo
    python scripts/nb_round8.py eval    --config C --prep nb --montages crown_po2 --kinds repaug    # leak-size reference
    python scripts/nb_round8.py report  --config C

Why. Every subset result so far was cut from a release whitened across all 63 channels: a whitened channel is a mix
of all 63, so "Crown-8" inputs carry signal from electrodes the Crown does not have, and reconstructing posterior
channels from them would be circular. This round rebuilds every montage from the UNWHITENED release
(Haitao999/things-eeg, Preprocessed_data_250Hz_no_whiten, which carries session ids) in NeuroBridge image order:
  prep z     per-channel z-scoring per session, statistics from that session's training trials (PRIMARY)
  prep mvnn  noise normalization fitted per session on each montage's OWN channels (optional, secondary)
  prep nb    the original 63-channel-whitened caches (reference, to measure the leak)
Each prep has its own data, per-repetition cache, session table, checkpoints and results.

Decoders (per prep x montage x target, people 1-10, 1 = selection only):
  repaug    f0 trained on the 9 other people, 1/2/4-presentation sampling (round 5)
  within    the person's own decoder (standard recipe), the ceiling
  distill   sparse-input student: image loss + pull toward the posterior-17 repaug teacher's output for the SAME
            trial (identical presentation subset); image projector copied from the teacher and frozen, so the
            student answers in the teacher's space
Virtual electrodes (montage names ve_<S> / veo_<S>): a time-invariant spatiotemporal ridge map from the sparse
channels to posterior-17 (lags 0 or +-10 samples = +-40 ms, ridge strength and lag chosen on held-out SOURCE trials),
  ve   fitted on the 9 source people's single presentations (deployable: the new user needs no dense data)
  veo  fitted on the target's own training presentations (upper bound, not deployable)
The new user's sparse EEG is expanded to virtual posterior-17 and read by the frozen posterior-17 repaug decoder
(and the within-person posterior-17 decoder as a ceiling); the round-6 pipeline runs on top.

Evaluation = round 6 at k = 500 (earlier-day calibration, 3 draws): zero-label pipeline (layer-1 offset from the
calibration trials, predicted population responses, batch CSLS), plain decoder, rigid alignment, fine-tuning,
within-person ceiling; queries session 4 at R = 1 / 4 / 20 and the 80-presentation benchmark.
Signal (per target): raw image identification (no decoder) on each montage and on the virtual channels, and the
virtual channels' R^2 against the real posterior-17 test data.

PRIMARY (prep z, Crown PO3/PO4, s4R4, zero-label pipeline, people 2-10, Wilcoxon, Holm over the two):
  Q1 ve - repaug (sparse)    Q2 distill - repaug (sparse)
Secondary: Crown-8 and occipital-8; with calibration (rigid at k = 500); veo; leak size (z - nb per montage).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
from eegalign import nb as N  # noqa: E402
import nb_round3 as R  # noqa: E402
import nb_round4 as R4m  # noqa: E402
import nb_round5 as R5m  # noqa: E402
import nb_round6 as R6m  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
SPARSE = ['crown_po2', 'crown8', 'occipital8']
ALL_MONT = ['posterior17'] + SPARSE
UBP_REPO = 'Haitao999/things-eeg'
SEED = 2025
K = 500
ORIG = {}


def dev():
    return N.ENV['device']


# ---------------------------------------------------------------------------------------------------------------
# environment per preprocessing
# ---------------------------------------------------------------------------------------------------------------
def setup_prep(prep, drive_base, reps_base='/content'):
    ORIG.update({k: N.ENV[k] for k in ('cache_dir', 'ckpt_dir', 'results_dir')}) if not ORIG else None
    ORIG.setdefault('reps_dir', N.ENV.get('reps_dir', '/content/nb_reps'))
    if prep == 'nb':
        N.ENV.update(cache_dir=ORIG['cache_dir'], ckpt_dir=ORIG['ckpt_dir'], reps_dir=ORIG['reps_dir'])
    else:
        lb = os.path.join(drive_base, 'leakfree', prep)
        N.ENV.update(cache_dir=os.path.join(lb, 'data'), ckpt_dir=os.path.join(lb, 'checkpoints'),
                     reps_dir=os.path.join(reps_base, f'nb_reps_{prep}'))
        for k in ('cache_dir', 'ckpt_dir'):
            os.makedirs(N.ENV[k], exist_ok=True)
    out = os.path.join(ORIG['results_dir'], 'nb_round8', prep)
    os.makedirs(out, exist_ok=True)
    N.ENV['results_dir'] = out
    R5m.odir = lambda: out
    R6m.KS, R6m.CAL_SESSIONS, R6m.FT_VARIANTS = [K], (3,), (('ft', False),)
    R6m.ndraw = lambda k: 3
    return out


# ---------------------------------------------------------------------------------------------------------------
# prep: unwhitened release -> leak-free montages in NeuroBridge image order
# ---------------------------------------------------------------------------------------------------------------
def _tload(p):
    try:
        return torch.load(p, weights_only=False, mmap=True, map_location='cpu')
    except Exception:                                                  # noqa: BLE001
        return torch.load(p, weights_only=False, map_location='cpu')


def _pkey(p):
    return '/'.join(str(p).replace('\\', '/').split('/')[-2:])


def zscore_by_session(x_tr, ses_tr, x_te, ses_te):
    out_tr, out_te = np.empty(x_tr.shape, np.float16), np.empty(x_te.shape, np.float16)
    for s in np.unique(ses_tr):
        m = ses_tr == s
        sel = x_tr[m]                                                    # (n, C, T)
        mu = sel.mean((0, 2), keepdims=True)
        sd = sel.std((0, 2), keepdims=True) + 1e-12
        out_tr[m] = ((sel - mu) / sd).astype(np.float16)
        mt = ses_te == s
        if mt.any():
            out_te[mt] = ((x_te[mt] - mu) / sd).astype(np.float16)
    return out_tr, out_te


def prep_data(subjects, prep, local='/content/ubp_raw_nw'):
    from huggingface_hub import hf_hub_download, list_repo_files
    from eegalign import ubp as U
    U.ENV['device'] = dev()
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
    files = list_repo_files(UBP_REPO, repo_type='dataset')
    union = list(dict.fromkeys(sum([U.MONTAGES[m] for m in ALL_MONT], [])))
    idx = [U.CHANNELS.index(c) for c in union]
    for s in subjects:
        npz = {m: os.path.join(N.ENV['cache_dir'], f'sub-{s:02d}', f'{m}.npz') for m in ALL_MONT}
        rp = {m: os.path.join(N.ENV['reps_dir'], f'sub-{s:02d}', f'{m}.npy') for m in ALL_MONT}
        sp = os.path.join(N.ENV['cache_dir'], 'sessions', f'sub-{s:02d}.npz')
        if all(os.path.exists(p) for p in list(npz.values()) + list(rp.values()) + [sp]):
            print(f'  sub-{s:02d}: {prep} caches present')
            continue
        t0 = time.time()
        got = {}
        for mode in ('train', 'test'):
            hit = [f for f in files if f.endswith(f'Preprocessed_data_250Hz_no_whiten/sub-{s:02d}/{mode}.pt')]
            assert hit, f'no unwhitened {mode} file for sub-{s:02d} on {UBP_REPO}'
            d = _tload(hf_hub_download(UBP_REPO, hit[0], repo_type='dataset', local_dir=local))
            got[mode] = dict(eeg=np.asarray(d['eeg'][:, :, idx, :250], np.float32), ses=np.asarray(d['session']).astype(int),
                             img=np.asarray(d['img']))
            del d
        nbd = np.load(os.path.join(ORIG['cache_dir'], f'sub-{s:02d}', 'posterior17.npz'))
        ui = {_pkey(p): i for i, p in enumerate(got['train']['img'][:, 0])}
        tr_map = np.array([ui[_pkey(p)] for p in nbd['train_paths']])
        ut = {_pkey(p): i for i, p in enumerate(got['test']['img'][:, 0])}
        te_map = np.array([ut[_pkey(p)] for p in nbd['test_paths']])
        tr, te = got['train']['eeg'][tr_map], got['test']['eeg'][te_map]
        ses_tr, ses_te = got['train']['ses'][tr_map], got['test']['ses'][te_map]
        if min(ses_tr.min(), ses_te.min()) == 0:
            ses_tr, ses_te = ses_tr + 1, ses_te + 1
        del got
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        np.savez(sp, train=ses_tr, test=ses_te)
        for m in ALL_MONT:
            ci = [union.index(c) for c in U.MONTAGES[m]]
            xtr, xte = tr[:, :, ci], te[:, :, ci]
            if prep == 'z':
                otr, ote = zscore_by_session(xtr, ses_tr, xte, ses_te)
            elif prep == 'mvnn':
                sc = float(xtr[:, 0].std()) + 1e-30                     # raw scale -> ~1 before float16
                otr, ote = U.mvnn_montage((xtr / sc).astype(np.float16), ses_tr, (xte / sc).astype(np.float16), ses_te)
            else:
                raise ValueError(prep)
            os.makedirs(os.path.dirname(npz[m]), exist_ok=True)
            np.savez(npz[m], train=otr.astype(np.float32).mean(1).astype(np.float16),
                     test=ote.astype(np.float32).mean(1).astype(np.float16), test_trials=ote,
                     train_labels=nbd['train_labels'], train_paths=nbd['train_paths'], test_paths=nbd['test_paths'],
                     channels=np.array(U.MONTAGES[m]))
            os.makedirs(os.path.dirname(rp[m]), exist_ok=True)
            np.save(rp[m], otr)
        del tr, te
        for f in glob.glob(os.path.join(local, '**', '*.pt'), recursive=True):
            os.remove(f)
        print(f'  sub-{s:02d}: {prep} montages built ({time.time() - t0:.0f}s)', flush=True)


def ensure_reps(subjects):
    """per-repetition caches live on the runtime's local disk; rebuild them from Drive npz is impossible (averaged),
    so a fresh runtime reruns prep for the per-rep part only (prep skips subjects whose files all exist)."""
    missing = [s for s in subjects if not all(os.path.exists(os.path.join(N.ENV['reps_dir'], f'sub-{s:02d}', f'{m}.npy'))
                                              for m in ALL_MONT)]
    return missing


# ---------------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------------
def train_within(mon, B, epochs=50):
    if os.path.exists(N.ckpt_path('within', mon, B, SEED, 'last')):
        print(f'  [within] {mon} t{B}: done')
        return
    import nb_train as TR
    TR.train_one('within', mon, B, SEED, PEOPLE, epochs, 0)


def train_distill(mon, B, lam=1.0, epochs=50, batch=1024, lr=1e-4, wd=1e-4):
    path = N.ckpt_path('distill', mon, B, SEED, 'last')
    if os.path.exists(path):
        print(f'  [distill] {mon} t{B}: done')
        return
    tp = N.ckpt_path('repaug', 'posterior17', B, SEED, 'last')
    assert os.path.exists(tp), f'teacher missing: train repaug posterior17 t{B} first'
    t0 = time.time()
    torch.manual_seed(SEED)
    teacher = N.load_model(tp)[0].to(dev()).eval().requires_grad_(False)
    Xs, Xd, IM, val = [], [], [], []
    for s in PEOPLE:
        if s == B:
            continue
        d = N.subject_data(s, mon, keys=('train', 'train_labels'))
        pool, draws = N.split(d['train_labels'])
        Xs.append(torch.from_numpy(np.asarray(R.load_reps(s, mon)[pool], np.float16)))
        Xd.append(torch.from_numpy(np.asarray(R.load_reps(s, 'posterior17')[pool], np.float16)))
        IM.append(torch.from_numpy(pool))
        val += [(d['train'][ix], ix) for ix in draws[:3]]
    XS, XD, IMG = torch.cat(Xs).to(dev()), torch.cat(Xd).to(dev()), torch.cat(IM).to(dev())
    del Xs, Xd
    n, _, C, T = XS.shape
    ftr = N.features('train', 'aug').to(dev())
    m = N.NBModel(C).to(dev())
    m.img_projector.load_state_dict(teacher.img_projector.state_dict())
    m.img_projector.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    kvals = torch.tensor([1., 2., 4.], device=dev())
    hist = []
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n, device=dev())
        tot, nb = 0.0, 0
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            b = len(idx)
            k = kvals[torch.randint(0, 3, (b,), device=dev())]
            mask = (torch.rand(b, 4, device=dev()).argsort(1).argsort(1).float() < k[:, None]).float()
            xs = (XS[idx].float() * mask[..., None, None]).sum(1) / k[:, None, None]
            with torch.no_grad():
                zt = teacher((XD[idx].float() * mask[..., None, None]).sum(1) / k[:, None, None])
            sm = F.avg_pool1d(xs.reshape(-1, 1, T), 5, 1, 2, count_include_pad=False).reshape(b, C, T)
            xs = torch.where(torch.rand(b, C, 1, device=dev()) < 0.3, sm, xs)
            z = m(xs)
            with torch.no_grad():
                g = m.gallery(ftr[IMG[idx]])
            loss = N.nb_loss(z, g, m.scale()) + lam * (1 - F.cosine_similarity(z, zt, dim=1).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        rec = dict(kind='distill', montage=mon, target=B, epoch=ep, train_loss=tot / max(nb, 1))
        if ep % 10 == 0 or ep == epochs:
            m.eval()
            with torch.no_grad():
                rec['val_top1'] = float(np.mean([N.retrieval(m, x, ftr[ix])[0] for x, ix in val]))
            print(f'    ep {ep:3d} loss {rec["train_loss"]:.3f} | val {rec["val_top1"]:5.1f}', flush=True)
        hist.append(rec)
    m.eval()
    cfg = dict(kind='distill', montage=mon, target=B, seed=SEED, lam=lam, teacher=tp)
    N.save_model(m, path, cfg, hist, epochs)
    N.save_model(m, N.ckpt_path('distill', mon, B, SEED, 'best'), cfg, hist, epochs)
    print(f'  => [distill] {mon} t{B} [{(time.time() - t0) / 60:.1f} min]', flush=True)
    del XS, XD, IMG
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------------------------------------------
# virtual electrodes
# ---------------------------------------------------------------------------------------------------------------
def lagged(x, L):
    """x (n, C, T) -> (n, T, C * (2L + 1) + 1) lagged features with a bias column."""
    xp = F.pad(x, (L, L))
    f = xp.unfold(2, 2 * L + 1, 1).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
    return torch.cat([f, torch.ones(*f.shape[:2], 1, device=x.device)], 2)


def fit_ve(xs_tr, xd_tr, xs_va, xd_va):
    """time-invariant spatiotemporal ridge sparse -> dense; lag and ridge chosen on the validation trials."""
    best = (-9, None)
    for L in (0, 10):
        A, Bm = 0, 0
        for i in range(0, len(xs_tr), 512):
            Fm = lagged(torch.as_tensor(xs_tr[i:i + 512], device=dev()).float(), L).reshape(-1, xs_tr.shape[1] * (2 * L + 1) + 1)
            Y = torch.as_tensor(xd_tr[i:i + 512], device=dev()).float().permute(0, 2, 1).reshape(-1, xd_tr.shape[1])
            A = A + Fm.T.double() @ Fm.double()
            Bm = Bm + Fm.T.double() @ Y.double()
        Fv = lagged(torch.as_tensor(xs_va, device=dev()).float(), L).double()
        Yv = torch.as_tensor(xd_va, device=dev()).double().permute(0, 2, 1)
        sst = ((Yv - Yv.mean((0, 1), keepdim=True)) ** 2).sum((0, 1))
        for rel in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            W = torch.linalg.solve(A + rel * A.trace() / len(A) * torch.eye(len(A), device=dev(), dtype=A.dtype), Bm)
            r2 = float((1 - ((Fv @ W - Yv) ** 2).sum((0, 1)) / sst).mean())
            if r2 > best[0]:
                best = (r2, (W.float(), L, rel))
    W, L, rel = best[1]

    def apply(x):
        out = []
        for i in range(0, len(x), 1024):
            xb = torch.as_tensor(np.asarray(x[i:i + 1024], np.float32), device=dev())
            out.append((lagged(xb, L) @ W).permute(0, 2, 1).cpu().numpy().astype(np.float16))
        return np.concatenate(out)
    return apply, dict(val_r2=best[0], lag=L, ridge=rel)


def sample_pairs(subjects, S, n_tr=2500, n_va=500, seed=0):
    xs_tr, xd_tr, xs_va, xd_va = [], [], [], []
    for s in subjects:
        rs, rd = R.load_reps(s, S), R.load_reps(s, 'posterior17')
        r = np.random.RandomState(seed + s)
        tot = min(n_tr + n_va, len(rs))
        n_va_s = max(1, int(round(tot * n_va / (n_tr + n_va))))
        ims = np.sort(r.choice(len(rs), tot, replace=False))
        rep = r.randint(0, 4, len(ims))
        a, b = np.asarray(rs[ims, rep], np.float32), np.asarray(rd[ims, rep], np.float32)
        p = r.permutation(len(ims))
        cut = tot - n_va_s
        xs_tr.append(a[p[:cut]]), xd_tr.append(b[p[:cut]]), xs_va.append(a[p[cut:]]), xd_va.append(b[p[cut:]])
    return (np.concatenate(xs_tr), np.concatenate(xd_tr), np.concatenate(xs_va), np.concatenate(xd_va))


VE = {}


def build_ve(B, S, oracle):
    """virtual posterior-17 for target B from its sparse montage S; installs patched data loaders."""
    srcs = [B] if oracle else [p for p in PEOPLE if p != B]
    apply, info = fit_ve(*sample_pairs(srcs, S, n_tr=6000 if oracle else 2500, n_va=1000 if oracle else 500))
    d = N.subject_data(B, S)
    out = {k: d[k] for k in ('train_labels', 'train_paths', 'test_paths')}
    out['train'] = apply(d['train'])
    out['test'] = apply(d['test'])
    tt = d['test_trials']
    out['test_trials'] = apply(tt.reshape(-1, *tt.shape[2:])).reshape(*tt.shape[:2], 17, tt.shape[-1])
    out['channels'] = np.array(['ve'] * 17)
    reps = R.load_reps(B, S)
    rp = os.path.join(N.ENV['reps_dir'], f'sub-{B:02d}', f'{"veo" if oracle else "ve"}_{S}.npy')
    mm = np.lib.format.open_memmap(rp, mode='w+', dtype=np.float16, shape=(reps.shape[0], 4, 17, reps.shape[-1]))
    for i in range(0, len(reps), 2048):
        blk = np.asarray(reps[i:i + 2048], np.float32)
        mm[i:i + len(blk)] = apply(blk.reshape(-1, *blk.shape[2:])).reshape(len(blk), 4, 17, blk.shape[-1])
    mm.flush()
    del mm
    # signal: R^2 of the virtual channels against the real posterior-17 test data
    real = N.subject_data(B, 'posterior17', keys=('test', 'test_trials'))
    yv, yr = out['test'].astype(np.float32), real['test'].astype(np.float32)
    info['test_avg_r2'] = float(np.mean(1 - ((yv - yr) ** 2).sum((0, 2)) / ((yr - yr.mean((0, 2), keepdims=True)) ** 2).sum((0, 2))))
    ts, tr_ = out['test_trials'][:, :4].astype(np.float32), real['test_trials'][:, :4].astype(np.float32)
    info['single_trial_r2'] = float(np.mean(1 - ((ts - tr_) ** 2).sum((0, 1, 3)) /
                                            ((tr_ - tr_.mean((0, 1, 3), keepdims=True)) ** 2).sum((0, 1, 3))))
    mon = f'{"veo" if oracle else "ve"}_{S}'
    VE.update(B=B, mon=mon, data=out, reps=rp)
    return mon, info


_SD, _LR, _CK = N.subject_data, R.load_reps, N.ckpt_path


def _sd(sub, mon, keys=None):
    if mon.startswith(('ve_', 'veo_')):
        if VE.get('B') == sub and VE.get('mon') == mon:
            return {k: v for k, v in VE['data'].items() if keys is None or k in keys}
        return _SD(sub, 'posterior17', keys)
    return _SD(sub, mon, keys)


def _lr(sub, mon):
    if mon.startswith(('ve_', 'veo_')):
        if VE.get('B') == sub and VE.get('mon') == mon:
            return np.load(VE['reps'], mmap_mode='r')
        return _LR(sub, 'posterior17')
    return _LR(sub, mon)


def _ck(kind, mon, sub, seed, which):
    return _CK(kind, 'posterior17' if mon.startswith(('ve_', 've0_', 'veo_')) else mon, sub, seed, which)


N.subject_data, R.load_reps, N.ckpt_path = _sd, _lr, _ck


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------
def signal_rows(B, mon):
    d = N.subject_data(B, mon, keys=('test_trials',))
    st = R4m.load_ses(B)
    te = st['test'] if st is not None else np.repeat(np.arange(4) + 1, 20)[None].repeat(200, 0)
    x = d['test_trials']
    tm = np.stack([x[i, np.isin(te[i], (1, 2))].astype(np.float32).mean(0) for i in range(200)])
    q34 = np.stack([x[i, np.isin(te[i], (3, 4))].astype(np.float32).mean(0) for i in range(200)])
    row = dict(raw_ident_s34=R.raw_ident(tm, q34)[0])
    for Rq in (4, 20):
        q = np.stack([x[i, np.nonzero(te[i] == 4)[0][:Rq]].astype(np.float32).mean(0) for i in range(200)])
        row[f'raw_ident_s4R{Rq}'] = R.raw_ident(tm, q)[0]
    return row


def eval_one(prep, kind, S, B):
    sub = 'repaug' if kind in ('ve', 'veo') else kind
    mon = S if kind in ('repaug', 'distill') else f'{kind}_{S}'
    out_dir = os.path.join(N.ENV['results_dir'], kind)
    os.makedirs(out_dir, exist_ok=True)
    R6m.odir = lambda: out_dir
    if os.path.exists(os.path.join(out_dir, f'cal_t{B:02d}_{mon}.csv')):
        print(f'  [eval {prep}/{kind}] {mon} t{B}: done')
        return
    if not os.path.exists(N.ckpt_path(sub, S if kind in ('repaug', 'distill') else 'posterior17', B, SEED, 'last')):
        print(f'  [eval {prep}/{kind}] {mon} t{B}: no checkpoint, skipping')
        return
    t0 = time.time()
    info = {}
    if kind in ('ve', 'veo'):
        mon, info = build_ve(B, S, oracle=(kind == 'veo'))
    R6m.DECODERS = [sub]
    R6m.run_target(B, mon)
    sig = dict(signal_rows(B, mon), **info, prep=prep, kind=kind, montage=S, target=B)
    pd.DataFrame([sig]).to_csv(os.path.join(out_dir, f'signal_t{B:02d}_{mon}.csv'), index=False)
    if kind in ('ve', 'veo') and os.path.exists(VE.get('reps', '')):
        os.remove(VE['reps'])
    VE.clear()
    print(f'  [eval {prep}/{kind}] {mon} t{B}: {(time.time() - t0) / 60:.1f} min {info}', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def report(base):
    pd.set_option('display.width', 250)
    rows = []
    for f in glob.glob(os.path.join(base, 'nb_round8', '*', '*', 'cal_t*.csv')):
        prep, kind = f.split(os.sep)[-3], f.split(os.sep)[-2]
        d = pd.read_csv(f)
        d['prep'], d['kind'] = prep, kind
        d['sparse'] = d['montage'].str.replace('veo_', '').str.replace('ve_', '')
        rows.append(d)
    if not rows:
        raise SystemExit('nothing evaluated yet')
    d = pd.concat(rows, ignore_index=True)
    cells = {'zero-label pipeline': ('zero', 'anchor', 'csls_batch'), 'plain decoder': ('zero', 'img', 'cos'),
             'rigid k=500': ('rigid', 'anchor', 'csls_batch'), 'fine-tune k=500': ('ft', 'img', 'csls_batch'),
             'within (ceiling)': ('within', 'img', 'cos')}
    per = {}
    for name, (m, ro, ru) in cells.items():
        x = d[(d.method == m) & (d.readout == ro) & (d.rule == ru)]
        per[name] = x.groupby(['prep', 'kind', 'sparse', 'query', 'target']).top1.mean()
    tab = pd.concat({k: v[v.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3]).mean()
                     for k, v in per.items()}, axis=1)
    print('TOP-1 % (people 2-10). kinds: repaug = plain sparse decoder, distill = distilled student, ve = virtual '
          'electrodes (population prior), veo = virtual electrodes fitted on the person (upper bound)')
    for q in ('s4R4', 's4R20', 'bench'):
        print(f'\nquery {q}')
        print(tab.xs(q, level='query').round(2).to_string())

    def g(prep, kind, S, cell='zero-label pipeline', q='s4R4'):
        try:
            return per[cell].xs((prep, kind, S, q), level=('prep', 'kind', 'sparse', 'query'))
        except KeyError:
            return None
    prim = {}
    print('\nPRIMARIES (prep z, Crown PO3/PO4, s4R4, zero-label pipeline)')
    for key, kind in (('Q1', 've'), ('Q2', 'distill')):
        a, b = g('z', kind, 'crown_po2'), g('z', 'repaug', 'crown_po2')
        if a is not None and b is not None:
            prim[key] = R4m.pstats(a, b, f'{key} {kind} - plain sparse decoder', primary=True)
    if prim:
        from statsmodels.stats.multitest import multipletests
        ks = sorted(prim)
        print('HOLM:', dict(zip(ks, [round(float(v), 4) for v in multipletests([prim[k] for k in ks], method='holm')[1]])))
    print('\nSECONDARY')
    for S in SPARSE:
        for kind in ('ve', 'veo', 'distill'):
            for cell in ('zero-label pipeline', 'rigid k=500'):
                a, b = g('z', kind, S, cell), g('z', 'repaug', S, cell)
                if a is not None and b is not None:
                    R4m.pstats(a, b, f'[{S}] {kind} - plain, {cell}')
        a, b = g('z', 'repaug', S, 'rigid k=500'), g('z', 'repaug', S)
        if a is not None and b is not None:
            R4m.pstats(a, b, f'[{S}] calibration (rigid k=500) - zero-label, plain sparse decoder')
    print('\nLEAK SIZE (z - nb, same people, same pipeline)')
    for S in ALL_MONT:
        for cell in ('plain decoder', 'zero-label pipeline', 'within (ceiling)'):
            for q in ('s4R4', 'bench'):
                a, b = g('z', 'repaug', S, cell, q), g('nb', 'repaug', S, cell, q)
                if a is not None and b is not None:
                    R4m.pstats(a, b, f'[{S}] {cell}, {q}: leak-free - 63-ch whitened')
    sig = [pd.read_csv(f) for f in glob.glob(os.path.join(base, 'nb_round8', '*', '*', 'signal_t*.csv'))]
    if sig:
        s = pd.concat(sig, ignore_index=True)
        s = s[s.target.isin(TEST)]
        cols = [c for c in ('raw_ident_s34', 'raw_ident_s4R20', 'raw_ident_s4R4', 'val_r2', 'test_avg_r2', 'single_trial_r2', 'lag')
                if c in s]
        print('\nSIGNAL (raw image identification %, chance 0.5; R^2 of virtual vs real posterior-17)')
        print(s.groupby(['prep', 'kind', 'montage'])[cols].mean().round(3).to_string())
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(base, 'nb_round8', 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['prep', 'train', 'distill', 'eval', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--prep', default='z', choices=['z', 'mvnn', 'nb'])
    p.add_argument('--montages', nargs='+', default=ALL_MONT)
    p.add_argument('--kinds', nargs='+', default=['repaug'])
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--drive_base', default=None)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    p.add_argument('--reps_base', default='/content')
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    drive_base = a.drive_base or os.path.dirname(N.ENV['cache_dir'].rstrip('/'))
    base = N.ENV['results_dir']
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'report':
        return report(base)
    setup_prep(a.prep, drive_base, a.reps_base)
    print(f'prep {a.prep}: cache {N.ENV["cache_dir"]}, reps {N.ENV["reps_dir"]}, checkpoints {N.ENV["ckpt_dir"]}', flush=True)
    if a.cmd == 'prep':
        return prep_data(a.targets, a.prep)
    if a.prep != 'nb':
        miss = ensure_reps(a.targets)
        if miss:
            print(f'per-repetition caches missing for {miss} (fresh runtime): rebuilding', flush=True)
            for s in miss:
                for m in ALL_MONT:
                    f = os.path.join(N.ENV['cache_dir'], f'sub-{s:02d}', f'{m}.npz')
                    if os.path.exists(f):
                        os.remove(f)
            prep_data(miss, a.prep)
    for S in a.montages:
        for B in a.targets:
            if a.cmd == 'train':
                print(f'[train {a.prep}] {S} t{B}', flush=True)
                R5m.SEED = SEED
                R5m.train_rep('repaug', S, B, a.epochs)
                train_within(S, B, a.epochs)
            elif a.cmd == 'distill':
                print(f'[distill {a.prep}] {S} t{B}', flush=True)
                train_distill(S, B, epochs=a.epochs)
            else:
                for kind in a.kinds:
                    if kind in ('ve', 'veo', 'distill') and S == 'posterior17':
                        continue
                    eval_one(a.prep, kind, S, B)


if __name__ == '__main__':
    main()
