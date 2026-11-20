"""Layout transfer: can a decoder trained on dense (posterior-17) lab EEG teach a decoder that only has a sparse
headset's electrodes, for a new person on a later day? Three stages with stop points (notebooks/nb_layout.ipynb).

    python scripts/layout.py audit  --config C           # reference / session-pairing / channel-invariance checks
    python scripts/layout.py stage1 --config C           # leak-free data + direct baselines                -> STOP 1
    python scripts/layout.py legacy --config C --montages posterior17 crown_po2   # same on the old whitened data
    python scripts/layout.py stage2 --config C           # PO3/PO4: virtual electrodes, distillation, controls -> STOP 2
    python scripts/layout.py stage3 --config C           # Crown-8, occipital-8, montage-whitening sensitivity
    python scripts/layout.py report --config C --stage 1|2|3

Data. Everything is rebuilt from THINGS-EEG2's unwhitened release (nb_round8 prep z): channels selected FIRST,
then z-scored per channel per session with that session's training-trial statistics. No electrode a layout lacks
enters its input. The old 63-channel-whitened caches ("nb") are rerun through the same pipeline to measure the
preprocessing effect. `audit` checks the one thing this cannot fix: whether the "unwhitened" data were already
re-referenced to the average of all 63 channels (if so, every sparse channel still mixes in the others).

Arms (sparse montage S; decoders trained on the 9 other people; person 1 = selection only; people 2-10 = test):
  repaug       direct sparse decoder, 1/2/4-presentation sampling (round 5)             strongest existing recipe
  direct1      direct sparse decoder, ONE presentation per sample, image projector copied from the dense teacher
               and frozen                                                                  matched baseline (A)
  vs_          spatial ridge sparse -> posterior-17 (no lags), observed channels copied through, read by the frozen
               dense teacher                                                              virtual electrodes (B)
  vl_          same with lags of +-10 samples (+-40 ms)                                   virtual electrodes (C)
  vo_          lagged map fitted on the person's own sparse + dense data (privileged diagnostic, not deployable)
  dist_same    direct1 + KL to the dense teacher's distribution over population responses, teacher sees the SAME
               presentation                                                               same-trial distillation (D)
  dist_rep     same, teacher sees the OTHER presentation of the same photograph from the same session
                                                                                          independent-repeat (E)
  dist_sparse  same as dist_rep, teacher = the sparse repaug decoder (no extra electrodes)     sparse-teacher (F)
  dist_perm    same as dist_rep, teacher input = a different photograph of the same concept     negative control
Distillation loss: L = L_image + lambda T^2 KL(p_teacher,T || p_student,T), both scored against the teacher-space
population responses of the batch's photographs (mean teacher output over the 9 source people); T = 2; lambda from
{0.1, 0.3, 1.0} chosen on person 1 (dist_rep, zero-label pipeline, s4R4) and locked for D/E/F/perm.
For virtual electrodes, every source person's input is also passed through the population map, so the population
responses the readout uses are built from virtual channels, as at deployment.

Evaluation (each arm x target): round 6 at k = 500 (earlier-day calibration, 3 draws; zero-label pipeline,
plain decoder, rigid, fine-tuning) + round 4's gallery test (no warm-up / 100 earlier trials; image gallery /
population responses; cosine / batch CSLS) + raw identification + for virtual electrodes, R^2 on the MISSING
channels (raw and with the grand evoked response removed).

Primaries (prep z, PO3/PO4, s4R4, zero-label pipeline = 100 earlier unlabeled trials + population responses + batch
CSLS, Wilcoxon over people 2-10, Holm over the two): vl_ - direct1, dist_rep - direct1. Practical threshold: +1 point.
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
import nb_round8 as R8  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
SEED = 2025
DENSE = 'posterior17'
LAMBDAS = (0.1, 0.3, 1.0)
TEMP = 2.0
VIRTUAL = ('vs_', 'vl_', 'vo_')


def dev():
    return N.ENV['device']


# ---------------------------------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------------------------------
BASE = {}


def setup(prep, drive_base):
    R8.setup_prep(prep, drive_base)
    out = os.path.join(BASE['results'], 'layout', prep)
    os.makedirs(out, exist_ok=True)
    N.ENV['results_dir'] = out
    R5m.odir = lambda: out
    return out


def arm_dir(arm):
    d = os.path.join(N.ENV['results_dir'], arm)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------------------------------------------
def audit(subject=1, local='/content/ubp_audit'):
    """1) average reference? 2) are training presentations paired within sessions? 3) sparse inputs independent of
    unavailable channels?"""
    from huggingface_hub import hf_hub_download, list_repo_files
    from eegalign import ubp as U
    out = {}
    files = list_repo_files(R8.UBP_REPO, repo_type='dataset')
    hit = [f for f in files if f.endswith(f'Preprocessed_data_250Hz_no_whiten/sub-{subject:02d}/test.pt')]
    d = R8._tload(hf_hub_download(R8.UBP_REPO, hit[0], repo_type='dataset', local_dir=local))
    x = np.asarray(d['eeg'][:20, :10], np.float32)                     # (20 img, 10 rep, 63, T)
    names = list(d.get('ch_names', U.CHANNELS))
    cm = x.mean(2)                                                      # mean over all channels
    out['n_channels'] = len(names)
    out['avg_ref_ratio'] = float(cm.std() / x.std(-1).mean())           # ~0 => average-referenced
    out['channel_sum_corr'] = float(np.corrcoef(x[:, :, 0].ravel(), cm.ravel())[0, 1])
    out['reference_guess'] = ('AVERAGE REFERENCE over all channels: sparse channels still mix in every electrode'
                              if out['avg_ref_ratio'] < 0.02 else 'not average-referenced (channel mean is not ~0)')
    del d
    for f in glob.glob(os.path.join(local, '**', '*.pt'), recursive=True):
        os.remove(f)
    # 2) session pairing in the leak-free session table
    st = R4m.load_ses(subject)
    if st is not None:
        tr = st['train']
        out['rep_pairs_same_session'] = float(((tr[:, 0] == tr[:, 1]) & (tr[:, 2] == tr[:, 3])).mean())
        out['sessions'] = {int(k): int(v) for k, v in zip(*np.unique(tr, return_counts=True))}
    # 3) channel invariance of the z-scoring path: replace unavailable channels with noise, recompute
    rng = np.random.RandomState(0)
    raw = rng.randn(40, 4, 23, 50).astype(np.float32)
    ses = np.repeat([[1, 1, 3, 3]], 40, 0)
    union = list(dict.fromkeys(sum([U.MONTAGES[m] for m in R8.ALL_MONT], [])))
    ci = [union.index(c) for c in U.MONTAGES['crown_po2']]
    a, _ = R8.zscore_by_session(raw[:, :, ci], ses, raw[:5, :, ci], ses[:5])
    raw2 = raw.copy()
    other = [i for i in range(len(union)) if i not in ci]
    raw2[:, :, other] = rng.randn(*raw2[:, :, other].shape) * 100
    b, _ = R8.zscore_by_session(raw2[:, :, ci], ses, raw2[:5, :, ci], ses[:5])
    out['channel_invariance'] = bool(np.array_equal(a, b))
    pd.Series(out, dtype=object).to_json(os.path.join(N.ENV['results_dir'], 'audit.json'))
    print('AUDIT:', out, flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------
# students
# ---------------------------------------------------------------------------------------------------------------
def partner_reps(ses):
    """(n, 4) session ids -> (n, 4) index of the other presentation from the same session."""
    p = np.zeros(ses.shape, int)
    for r in range(4):
        same = (ses == ses[:, [r]])
        same[:, r] = False
        p[:, r] = np.where(same.any(1), same.argmax(1), (r + 1) % 4)
    return p


def population_anchors(teacher, mon, B):
    """mean teacher output over the 9 source people for every pool image (teacher space)."""
    acc, pool = 0, None
    for s in PEOPLE:
        if s == B:
            continue
        d = N.subject_data(s, mon, keys=('train', 'train_labels'))
        pool, _ = N.split(d['train_labels'])
        with torch.no_grad():
            z = torch.cat([teacher(torch.as_tensor(d['train'][pool[i:i + 4096]], device=dev()).float())
                           for i in range(0, len(pool), 4096)])
        acc = acc + z / (len(PEOPLE) - 1)
    return acc, pool


def train_student(arm, S, B, lam=0.0, epochs=50, batch=1024, lr=1e-4, wd=1e-4, tag=None):
    kind = tag or arm
    path = N.ckpt_path(kind, S, B, SEED, 'last')
    if os.path.exists(path):
        print(f'  [{kind}] {S} t{B}: done')
        return
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tmon = S if arm == 'dist_sparse' else DENSE
    tkind = 'repaug'
    tp = N.ckpt_path(tkind, tmon, B, SEED, 'last')
    assert os.path.exists(tp), f'teacher {tp} missing (stage 1 trains it)'
    teacher = N.load_model(tp)[0].to(dev()).eval().requires_grad_(False)
    use_t = arm != 'direct1' and lam > 0
    A_all, pool0 = population_anchors(teacher, tmon, B) if use_t else (None, None)
    Xs, Xt, IM, PR, CON, val = [], [], [], [], [], []
    for s in PEOPLE:
        if s == B:
            continue
        d = N.subject_data(s, S, keys=('train', 'train_labels'))
        pool, draws = N.split(d['train_labels'])
        Xs.append(torch.from_numpy(np.asarray(R.load_reps(s, S)[pool], np.float16)))
        if use_t:
            Xt.append(torch.from_numpy(np.asarray(R.load_reps(s, tmon)[pool], np.float16)))
        st = R4m.load_ses(s)
        PR.append(torch.from_numpy(partner_reps(st['train'][pool])))
        IM.append(torch.from_numpy(pool))
        CON.append(d['train_labels'][pool].astype(np.int64))
        val += [(d['train'][ix], ix) for ix in draws[:3]]
    XS = torch.cat(Xs).to(dev())
    XT = torch.cat(Xt).to(dev()) if use_t else None
    IMG, P, CON_np = torch.cat(IM).to(dev()), torch.cat(PR).to(dev()), CON
    del Xs, Xt
    n, _, C, T = XS.shape
    if use_t:
        pos = {int(i): j for j, i in enumerate(pool0)}
        apos = torch.as_tensor([pos[int(i)] for i in IMG.cpu().numpy()], device=dev())
        An = F.normalize(A_all, dim=-1)
        # dist_perm: the teacher sees ANOTHER photograph of the same concept from the same source person
        perm_np = np.empty(n, np.int64)
        off = 0
        for blk in CON_np:
            o = np.lexsort((np.arange(len(blk)), blk))
            g = blk[o]
            start = np.r_[0, np.nonzero(np.diff(g))[0] + 1]
            end = np.r_[start[1:], len(g)]
            nxt = np.arange(len(g)) + 1
            nxt[end - 1] = start
            perm_np[off + o] = off + o[nxt]
            off += len(blk)
        perm_src = torch.as_tensor(perm_np, device=dev())
    ftr = N.features('train', 'aug').to(dev())
    m = N.NBModel(C).to(dev())
    m.img_projector.load_state_dict(teacher.img_projector.state_dict())
    m.img_projector.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    hist = []
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n, device=dev())
        tot, totkl, nb = 0.0, 0.0, 0
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            b = len(idx)
            r = torch.randint(0, 4, (b,), device=dev())
            xs = XS[idx, r].float()
            sm = F.avg_pool1d(xs.reshape(-1, 1, T), 5, 1, 2, count_include_pad=False).reshape(b, C, T)
            xs = torch.where(torch.rand(b, C, 1, device=dev()) < 0.3, sm, xs)
            z = m(xs)
            with torch.no_grad():
                g = m.gallery(ftr[IMG[idx]])
            loss = N.nb_loss(z, g, m.scale())
            if use_t:
                if arm == 'dist_same':
                    ti, tr_ = idx, r
                elif arm == 'dist_perm':
                    ti = perm_src[idx]
                    tr_ = P[ti, r]
                else:                                                   # dist_rep, dist_sparse
                    ti, tr_ = idx, P[idx, r]
                with torch.no_grad():
                    zt = teacher(XT[ti, tr_].float())
                    Ab = An[apos[idx]]
                    lt = m.scale() * F.normalize(zt, dim=-1) @ Ab.T
                ls = m.scale() * F.normalize(z, dim=-1) @ Ab.T
                kl = F.kl_div(F.log_softmax(ls / TEMP, 1), F.softmax(lt / TEMP, 1), reduction='batchmean')
                loss = loss + lam * TEMP ** 2 * kl
                totkl += kl.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        rec = dict(arm=kind, montage=S, target=B, epoch=ep, loss=tot / max(nb, 1), kl=totkl / max(nb, 1))
        if ep % 10 == 0 or ep == epochs:
            m.eval()
            with torch.no_grad():
                rec['val_top1'] = float(np.mean([N.retrieval(m, x, ftr[ix])[0] for x, ix in val]))
            print(f'    ep {ep:3d} loss {rec["loss"]:.3f} kl {rec["kl"]:.3f} | val {rec["val_top1"]:5.1f}', flush=True)
        hist.append(rec)
    m.eval()
    cfg = dict(arm=arm, montage=S, target=B, lam=lam, temp=TEMP, teacher=tp, seed=SEED)
    for which in ('last', 'best'):
        N.save_model(m, N.ckpt_path(kind, S, B, SEED, which), cfg, hist, epochs)
    pd.DataFrame(hist).to_csv(os.path.join(N.ENV['results_dir'], 'student_curves.csv'), mode='a', index=False,
                              header=not os.path.exists(os.path.join(N.ENV['results_dir'], 'student_curves.csv')))
    print(f'  => [{kind}] {S} t{B} lam {lam} [{(time.time() - t0) / 60:.1f} min]', flush=True)
    del XS, XT
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------------------------------------------
# virtual electrodes
# ---------------------------------------------------------------------------------------------------------------
def fit_map(subjects, S, L, n_tr, n_va):
    """ridge sparse -> dense at a FIXED lag L; ridge strength chosen on held-out trials of the same subjects
    (R^2 on the missing channels). Observed channels are copied through afterwards."""
    from eegalign import ubp as U
    xs_tr, xd_tr, xs_va, xd_va = R8.sample_pairs(subjects, S, n_tr=n_tr, n_va=n_va)
    dense, sparse = U.MONTAGES[DENSE], U.MONTAGES[S]
    obs = [(dense.index(c), sparse.index(c)) for c in sparse if c in dense]
    miss = [i for i in range(len(dense)) if i not in [o for o, _ in obs]]
    A, Bm = 0, 0
    for i in range(0, len(xs_tr), 512):
        Fm = R8.lagged(torch.as_tensor(xs_tr[i:i + 512], device=dev()).float(), L).reshape(-1, xs_tr.shape[1] * (2 * L + 1) + 1)
        Y = torch.as_tensor(xd_tr[i:i + 512], device=dev()).float().permute(0, 2, 1).reshape(-1, xd_tr.shape[1])
        A = A + Fm.T.double() @ Fm.double()
        Bm = Bm + Fm.T.double() @ Y.double()
    Fv = R8.lagged(torch.as_tensor(xs_va, device=dev()).float(), L).double()
    Yv = torch.as_tensor(xd_va, device=dev()).double().permute(0, 2, 1)
    sst = ((Yv - Yv.mean((0, 1), keepdim=True)) ** 2).sum((0, 1))
    best = (-9, None, None)
    for rel in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        W = torch.linalg.solve(A + rel * A.trace() / len(A) * torch.eye(len(A), device=dev(), dtype=A.dtype), Bm)
        r2 = (1 - ((Fv @ W - Yv) ** 2).sum((0, 1)) / sst)[miss].mean().item()
        if r2 > best[0]:
            best = (r2, W.float(), rel)
    W = best[1]

    def apply(x):
        out = []
        for i in range(0, len(x), 1024):
            xb = torch.as_tensor(np.asarray(x[i:i + 1024], np.float32), device=dev())
            y = (R8.lagged(xb, L) @ W).permute(0, 2, 1)
            if N.ENV.get('_prep') == 'z':
                for di, si in obs:                                      # identical z-scored channels: copy through
                    y[:, di] = xb[:, si]
            out.append(y.cpu().numpy().astype(np.float16))
        return np.concatenate(out)
    return apply, dict(val_r2_missing=best[0], ridge=best[2], lag=L), miss


VSTATE = {}


def virtual_data(sub, mon):
    key = (sub, mon)
    if key in VSTATE.get('cache', {}):
        return VSTATE['cache'][key]
    S = mon.split('_', 1)[1]
    apply = VSTATE['apply_target'] if sub == VSTATE['B'] else VSTATE['apply_pop']
    d = R8._SD(sub, S)
    out = {k: d[k] for k in ('train_labels', 'train_paths', 'test_paths')}
    out['train'], out['test'] = apply(d['train']), apply(d['test'])
    tt = d['test_trials']
    out['test_trials'] = apply(tt.reshape(-1, *tt.shape[2:])).reshape(*tt.shape[:2], 17, tt.shape[-1])
    out['channels'] = np.array(['virtual'] * 17)
    VSTATE.setdefault('cache', {})[key] = out
    return out


def _sd(sub, mon, keys=None):
    if mon.startswith(VIRTUAL):
        d = virtual_data(sub, mon)
        return {k: v for k, v in d.items() if keys is None or k in keys}
    return R8._SD(sub, mon, keys)


def _lr(sub, mon):
    if mon.startswith(VIRTUAL):
        assert sub == VSTATE['B'], 'per-repetition virtual data exists for the target only'
        return np.load(VSTATE['reps'], mmap_mode='r')
    return R8._LR(sub, mon)


def _ck(kind, mon, sub, seed, which):
    return R8._CK(kind, DENSE if mon.startswith(VIRTUAL) else mon, sub, seed, which)


N.subject_data, R.load_reps, N.ckpt_path = _sd, _lr, _ck


def install_virtual(B, S, mode):
    """mode: vs (spatial, population), vl (lagged, population), vo (lagged, fitted on the person)."""
    src = [p for p in PEOPLE if p != B]
    L = 0 if mode == 'vs' else 10
    apply_pop, info_pop, miss = fit_map(src, S, L, 2500, 500)
    if mode == 'vo':
        apply_t, info_t, _ = fit_map([B], S, L, 6000, 1000)
    else:
        apply_t, info_t = apply_pop, info_pop
    mon = f'{mode}_{S}'
    VSTATE.clear()
    VSTATE.update(B=B, apply_pop=apply_pop, apply_target=apply_t, cache={})
    reps = R8._LR(B, S)
    rp = os.path.join(N.ENV['reps_dir'], f'sub-{B:02d}', f'{mon}.npy')
    mm = np.lib.format.open_memmap(rp, mode='w+', dtype=np.float16, shape=(reps.shape[0], 4, 17, reps.shape[-1]))
    for i in range(0, len(reps), 2048):
        blk = np.asarray(reps[i:i + 2048], np.float32)
        mm[i:i + len(blk)] = apply_t(blk.reshape(-1, *blk.shape[2:])).reshape(len(blk), 4, 17, blk.shape[-1])
    mm.flush()
    del mm
    VSTATE['reps'] = rp
    # reconstruction on the person's held-out test data, missing channels only
    v = virtual_data(B, mon)
    real = R8._SD(B, DENSE, keys=('test', 'test_trials'))

    def r2(a, b):
        a, b = a.astype(np.float32)[:, miss], b.astype(np.float32)[:, miss]
        return float(np.mean(1 - ((a - b) ** 2).sum((0, 2)) / ((b - b.mean((0, 2), keepdims=True)) ** 2).sum((0, 2))))
    va, ra = v['test'], real['test']
    info = dict(info_t, map_fit='person' if mode == 'vo' else 'population',
                r2_missing_avg=r2(va, ra),
                r2_missing_avg_stimulus=r2(va - va.mean(0, keepdims=True), ra - ra.mean(0, keepdims=True)),
                r2_missing_single=r2(v['test_trials'][:, :4].reshape(-1, 17, va.shape[-1]),
                                     real['test_trials'][:, :4].reshape(-1, 17, ra.shape[-1])))
    return mon, info


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------
def eval_arm(prep, arm, S, B):
    virt = arm in ('vs', 'vl', 'vo')
    kind = 'repaug' if virt else arm
    mon = f'{arm}_{S}' if virt else S
    out_dir = arm_dir(arm)
    done = os.path.join(out_dir, f'm2_t{B:02d}_{mon}.csv')
    if os.path.exists(done):
        print(f'  [eval {prep}/{arm}] {mon} t{B}: done')
        return
    if not os.path.exists(N.ckpt_path(kind, DENSE if virt else S, B, SEED, 'last')):
        print(f'  [eval {prep}/{arm}] {mon} t{B}: no checkpoint, skipping')
        return
    t0 = time.time()
    info = {}
    if virt:
        mon, info = install_virtual(B, S, arm)
    R6m.odir = lambda: out_dir
    R6m.DECODERS = [kind]
    R6m.run_target(B, mon)
    R5m.use_kind(kind)
    c = R4m.R4(B, mon, 'last')
    m2 = pd.DataFrame(R4m.exp_m2(c))
    sig = dict(R8.signal_rows(B, mon), **info, prep=prep, arm=arm, montage=S, target=B)
    pd.DataFrame([sig]).to_csv(os.path.join(out_dir, f'signal_t{B:02d}_{mon}.csv'), index=False)
    m2['arm'], m2['target'], m2['montage'] = arm, B, S
    m2.to_csv(done, index=False)
    if virt and os.path.exists(VSTATE.get('reps', '')):
        os.remove(VSTATE['reps'])
    VSTATE.clear()
    print(f'  [eval {prep}/{arm}] {mon} t{B}: {(time.time() - t0) / 60:.1f} min {info}', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------------------------
def train_direct(prep, mons, targets, epochs):
    for S in mons:
        for B in targets:
            print(f'[train {prep}] repaug + within {S} t{B}', flush=True)
            R5m.SEED = SEED
            R5m.train_rep('repaug', S, B, epochs)
            R8.train_within(S, B, epochs)


def chosen_lambda():
    f = os.path.join(N.ENV['results_dir'], 'lambda.txt')
    return float(open(f).read()) if os.path.exists(f) else None


def select_lambda(S, epochs):
    """train dist_rep at each lambda for person 1, evaluate, lock the best (zero-label pipeline, s4R4)."""
    if chosen_lambda() is not None:
        print(f'lambda locked at {chosen_lambda()}')
        return chosen_lambda()
    scores = {}
    for lam in LAMBDAS:
        tag = f'dist_rep_l{lam}'
        train_student('dist_rep', S, 1, lam=lam, epochs=epochs, tag=tag)
        eval_arm('select', tag, S, 1)
        m = pd.read_csv(os.path.join(arm_dir(tag), f'm2_t01_{S}.csv'))
        m = m[(m.exp == 'M2') & (m.offset == 'old_100') & (m.gallery == 'anchor') & (m.rule == 'csls_batch') & (m['query'] == 's4R4')]
        scores[lam] = float(m.top1.mean())
    lam = max(scores, key=scores.get)
    open(os.path.join(N.ENV['results_dir'], 'lambda.txt'), 'w').write(str(lam))
    print(f'lambda selection on person 1 (s4R4, zero-label pipeline): {scores} -> {lam}', flush=True)
    return lam


def stage1(targets, epochs):
    R8.prep_data(PEOPLE, 'z')
    audit()
    train_direct('z', [DENSE, 'crown_po2'], targets, epochs)
    for B in targets:
        for S in (DENSE, 'crown_po2'):
            eval_arm('z', 'repaug', S, B)


def stage_legacy(drive_base, mons, targets, epochs):
    setup('nb', drive_base)
    N.ENV['_prep'] = 'nb'
    train_direct('nb', mons, targets, epochs)
    for B in targets:
        for S in mons:
            eval_arm('nb', 'repaug', S, B)


def stage2(S, targets, epochs):
    lam = select_lambda(S, epochs)
    for B in targets:
        print(f'[stage 2] {S} t{B}', flush=True)
        train_student('direct1', S, B, epochs=epochs)
        for arm in ('dist_rep', 'dist_same', 'dist_sparse', 'dist_perm'):
            train_student(arm, S, B, lam=lam, epochs=epochs)
        for arm in ('direct1', 'dist_rep', 'dist_same', 'dist_sparse', 'dist_perm', 'vs', 'vl', 'vo'):
            eval_arm('z', arm, S, B)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
CELLS = {'zero-label pipeline': ('M2', 'old_100', 'anchor', 'csls_batch'),
         'no warm-up, cosine, population responses': ('M2', 'none', 'anchor', 'cos'),
         'plain decoder (image gallery, cosine)': ('M2', 'none', 'img', 'cos')}
CAL = {'rigid k=500': ('rigid', 'anchor', 'csls_batch'), 'fine-tune k=500': ('ft', 'img', 'csls_batch'),
       'within (ceiling)': ('within', 'img', 'cos')}


def load_all():
    m2, cal, sig = [], [], []
    for f in glob.glob(os.path.join(BASE['results'], 'layout', '*', '*', 'm2_t*.csv')):
        d = pd.read_csv(f)
        d['prep'] = f.split(os.sep)[-3]
        m2.append(d)
    for f in glob.glob(os.path.join(BASE['results'], 'layout', '*', '*', 'cal_t*.csv')):
        d = pd.read_csv(f)
        d['prep'], d['arm'] = f.split(os.sep)[-3], f.split(os.sep)[-2]
        d['montage'] = d['montage'].str.replace(r'^(vs|vl|vo)_', '', regex=True)
        cal.append(d)
    for f in glob.glob(os.path.join(BASE['results'], 'layout', '*', '*', 'signal_t*.csv')):
        sig.append(pd.read_csv(f))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else None  # noqa: E731
    return cat(m2), cat(cal), cat(sig)


def per_person(m2, cal):
    out = {}
    if m2 is not None:
        for name, (e, off, gal, rule) in CELLS.items():
            x = m2[(m2.exp == e) & (m2.offset == off) & (m2.gallery == gal) & (m2.rule == rule)]
            out[name] = x.groupby(['prep', 'arm', 'montage', 'query', 'target']).top1.mean()
    if cal is not None:
        for name, (meth, ro, rule) in CAL.items():
            x = cal[(cal.method == meth) & (cal.readout == ro) & (cal.rule == rule)]
            out[name] = x.groupby(['prep', 'arm', 'montage', 'query', 'target']).top1.mean()
    return out


def report(stage):
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 300)
    R4m.CONTRASTS.clear()
    for prep in ('z', 'nb'):
        a = os.path.join(BASE['results'], 'layout', prep, 'audit.json')
        if os.path.exists(a):
            print(f'AUDIT ({prep}):', open(a).read())
    lf = os.path.join(BASE['results'], 'layout', 'z', 'lambda.txt')
    if os.path.exists(lf):
        print('locked distillation lambda:', open(lf).read())
    m2, cal, sig = load_all()
    if m2 is None:
        raise SystemExit('nothing evaluated yet')
    per = per_person(m2, cal)
    tab = pd.concat({k: v[v.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3]).mean()
                     for k, v in per.items()}, axis=1)
    for q in ('s4R4', 's4R1', 's4R20', 'bench'):
        print(f'\nTOP-1 %, query {q} (people 2-10)')
        try:
            print(tab.xs(q, level='query').round(2).to_string())
        except KeyError:
            pass
    if sig is not None:
        s = sig[sig.target.isin(TEST)]
        cols = [c for c in ('raw_ident_s34', 'raw_ident_s4R20', 'raw_ident_s4R4', 'val_r2_missing', 'r2_missing_avg',
                            'r2_missing_avg_stimulus', 'r2_missing_single', 'ridge') if c in s]
        print('\nSIGNAL (raw image identification %, chance 0.5; virtual electrodes: R^2 on the missing channels)')
        print(s.groupby(['prep', 'arm', 'montage'])[cols].mean().round(3).to_string())

    def g(prep, arm, S, cell='zero-label pipeline', q='s4R4'):
        try:
            return per[cell].xs((prep, arm, S, q), level=('prep', 'arm', 'montage', 'query'))
        except KeyError:
            return None

    def con(a, b, label, **kw):
        if a is not None and b is not None and len(set(a.index) & set(b.index) & set(TEST)):
            return R4m.pstats(a, b, label, **kw)
        return None
    print('\n' + '=' * 100 + '\nSTAGE 1: leak-free baselines and the preprocessing effect\n' + '=' * 100)
    for S in (DENSE, 'crown_po2', 'crown8', 'occipital8'):
        for cell in ('zero-label pipeline', 'plain decoder (image gallery, cosine)', 'within (ceiling)'):
            for q in ('s4R4', 'bench'):
                con(g('z', 'repaug', S, cell, q), g('nb', 'repaug', S, cell, q), f'[{S}] {cell}, {q}: leak-free - 63-ch whitened')
    for S in ('crown_po2', 'crown8', 'occipital8'):
        a = g('z', 'repaug', S)
        if a is not None:
            print(f'  [{S}] leak-free zero-label pipeline s4R4: {a.reindex(TEST).mean():.2f} (chance 0.5); '
                  f'per person {np.round(a.reindex(TEST).values, 2).tolist()}')
    print('\nSTOP-1 RULE: continue to stage 2 only if (a) the audit shows no average reference (or we accept it '
          'and say so), (b) leak-free posterior-17 stays a usable teacher (zero-label s4R4 >= 6), and (c) leak-free '
          'PO3/PO4 is above chance for most people (zero-label s4R4 >= 1.0 mean, >= 6/9 above 0.5).')
    if stage >= 2:
        print('\n' + '=' * 100 + '\nSTAGE 2: PO3/PO4 layout transfer\n' + '=' * 100)
        prim = {}
        for key, arm in (('Q1', 'vl'), ('Q2', 'dist_rep')):
            p = con(g('z', arm, 'crown_po2'), g('z', 'direct1', 'crown_po2'), f'{key} {arm} - direct1 (matched)', primary=True,
                    margin=1.0)
            if p is not None:
                prim[key] = p
        if prim:
            from statsmodels.stats.multitest import multipletests
            ks = sorted(prim)
            print('HOLM:', dict(zip(ks, [round(float(v), 4) for v in multipletests([prim[k] for k in ks], method='holm')[1]])))
        for arm in ('vs', 'vl', 'vo', 'dist_rep', 'dist_same', 'dist_sparse', 'dist_perm', 'repaug'):
            for cell in ('zero-label pipeline', 'no warm-up, cosine, population responses', 'plain decoder (image gallery, cosine)', 'rigid k=500'):
                con(g('z', arm, 'crown_po2', cell), g('z', 'direct1', 'crown_po2', cell), f'  {arm} - direct1: {cell}')
        con(g('z', 'dist_rep', 'crown_po2'), g('z', 'dist_sparse', 'crown_po2'), 'KEY CONTROL dist_rep - dist_sparse (value of the extra electrodes)')
        con(g('z', 'dist_rep', 'crown_po2'), g('z', 'dist_same', 'crown_po2'), 'dist_rep - dist_same (independent vs same presentation)')
        con(g('z', 'dist_rep', 'crown_po2'), g('z', 'dist_perm', 'crown_po2'), 'dist_rep - dist_perm (image-level teacher signal)')
        con(g('z', 'dist_rep', 'crown_po2'), g('z', 'repaug', 'crown_po2'), 'dist_rep - repaug (vs the strongest direct recipe)')
        con(g('z', 'vl', 'crown_po2'), g('z', 'vs', 'crown_po2'), 'vl - vs (value of lags)')
        print('\nSTOP-2 RULE (prespecified): the stream stands if a primary (vl or dist_rep vs direct1) is >= +1 point with '
              'Holm p < .05, AND dist_rep beats dist_sparse (otherwise the gain is repeat consistency, not the dense '
              'electrodes), AND the gain survives at k = 500 (rigid). Otherwise: report the leak-free baselines and the '
              'preprocessing effect, and close the stream.')
    if stage >= 3:
        print('\n' + '=' * 100 + '\nSTAGE 3: other sparse layouts\n' + '=' * 100)
        for S in ('crown8', 'occipital8'):
            for arm in ('vl', 'vs', 'dist_rep', 'dist_sparse', 'repaug'):
                for cell in ('zero-label pipeline', 'rigid k=500'):
                    con(g('z', arm, S, cell), g('z', 'direct1', S, cell), f'[{S}] {arm} - direct1: {cell}')
        con(g('z', 'repaug', 'crown_po2'), g('z', 'repaug', 'crown8'), 'placement: PO3/PO4 alone - all 8 Crown sites (repaug)')
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(BASE['results'], 'layout', f'contrasts_stage{stage}.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['audit', 'stage1', 'legacy', 'stage2', 'stage3', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--stage', type=int, default=1)
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--drive_base', default=None)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    p.add_argument('--montages', nargs='+', default=['crown8', 'occipital8'])
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    BASE['results'] = N.ENV['results_dir']
    drive_base = a.drive_base or os.path.dirname(N.ENV['cache_dir'].rstrip('/'))
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'report':
        return report(a.stage)
    if a.cmd == 'legacy':
        return stage_legacy(drive_base, a.montages, a.targets, a.epochs)
    setup('z', drive_base)
    N.ENV['_prep'] = 'z'
    if a.cmd == 'audit':
        return audit()
    if a.cmd == 'stage1':
        return stage1(a.targets, a.epochs)
    miss = R8.ensure_reps(PEOPLE)
    if miss:
        print(f'per-repetition caches missing for {miss} (fresh runtime): rebuilding', flush=True)
        for s in miss:
            for m in R8.ALL_MONT:
                f = os.path.join(N.ENV['cache_dir'], f'sub-{s:02d}', f'{m}.npz')
                if os.path.exists(f):
                    os.remove(f)
        R8.prep_data(miss, 'z')
    if a.cmd == 'stage2':
        return stage2('crown_po2', a.targets, a.epochs)
    if a.cmd == 'stage3':
        train_direct('z', a.montages, a.targets, a.epochs)
        for S in a.montages:
            for B in a.targets:
                train_student('direct1', S, B, epochs=a.epochs)
                for arm in ('dist_rep', 'dist_sparse'):
                    train_student(arm, S, B, lam=chosen_lambda() or 0.3, epochs=a.epochs)
                for arm in ('repaug', 'direct1', 'dist_rep', 'dist_sparse', 'vs', 'vl'):
                    eval_arm('z', arm, S, B)


if __name__ == '__main__':
    main()
