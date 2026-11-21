"""Round 10: combine round 9's two wins and test them on 3 seeds.

    python scripts/nb_round10.py train  --config C [--seeds 2025 2026 2027]   # CNN/intermediate + EEGProject/aug decoders
    python scripts/nb_round10.py eval   --config C [--seeds ...]              # one pipeline for both decoders
    python scripts/nb_round10.py report --config C

Round 9: (1) a compact CNN trained on clean intermediate RN50 features beat the current EEGProject decoder on new
people (+4.6 with rigid calibration, 9/9, one seed, output-space offset only); (2) in the output edit, the rotation
does the work (rotation - shift +3.4, 8/9) and the global scale hurts (-1.1), so rotation-only beat rigid by ~1 point.

Decoders (target excluded from training, 10 folds, seeds 2025 / 2026 / 2027):
  eeg_aug   EEGProject on NeuroBridge's degraded-final CLIP features, repaug recipe (rounds 5-7 checkpoints; lr 1e-4)
  cnn_mid   compact CNN on clean intermediate RN50 features (round 9 recipe; lr 3e-4 chosen on the person-1 fold)
One pipeline for both, so differences come from the decoder:
  layer-1 offset  label-free shift of the first dense layer's output to the source mean (EEGProject: its first linear
                  layer, as in rounds 3-8; CNN: the first linear layer after the convolutions), from the calibration
                  trials, labels unused
  galleries       population templates (ridge from the decoder's own image features to the source people's mean
                  outputs on pool images) and the decoder's image gallery
  edits           zero (offset only); rigid (rounds 6-7, lambda 1); rot (rotation only, strength by 5-fold CV over the
                  calibration images, round 9 grid); ft (fine-tune 300 steps) and ft+rot, at k = 500 and 2000
  calibration     earlier day (session 3), k = 100 / 200 / 500 / 2000 images x 2 presentations, round 6 draws
  queries         session 4 at 1 / 4 / 20 presentations and the 80-presentation benchmark; cos (each query scored on its
                  own) and batch CSLS

PRIMARY (fixed before running): cnn_mid + rot - eeg_aug + rigid (the rounds 6-7 method), k = 500, population gallery,
s4R4, cos, per person averaged over 3 seeds, people 2-10, Wilcoxon.
Secondary (Holm over four): decoder effect (cnn_mid - eeg_aug, both rot); edit effect (rot - rigid, eeg_aug);
zero-label (cnn_mid - eeg_aug, zero); the primary with batch CSLS. Then: every k, s4R1 / s4R20 / bench, each seed alone,
fine-tuning at 500 / 2000, and the round-8 from-scratch decoder on the same images. Reproduction check: eeg_aug seed
2026 rigid - zero with batch CSLS should be close to round 7's +3.89. Development replication on people used before.
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
import nb_next as X  # noqa: E402
import nb_round3 as R  # noqa: E402
import nb_round4 as R4m  # noqa: E402
import nb_round5 as R5m  # noqa: E402
import nb_round6 as R6m  # noqa: E402
import nb_gallery as GL  # noqa: E402
import nb_round9 as R9  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
QK = ['s4R1', 's4R4', 's4R20', 'bench']
SEEDS = [2025, 2026, 2027]
KS = [100, 200, 500, 2000]
K_FT = (500, 2000)
CS = 3
DECODERS = {'eeg_aug': ('eegproject', 'aug'), 'cnn_mid': ('cnn', 'mid')}
CNN_LR = 3e-4


def dev():
    return N.ENV['device']


def odir(sub=''):
    d = os.path.join(N.ENV['results_dir'], 'nb_round10', sub)
    os.makedirs(d, exist_ok=True)
    return d


def nrm(x):
    return F.normalize(x.float(), dim=-1)


# ---------------------------------------------------------------------------------------------------------------
# decoders
# ---------------------------------------------------------------------------------------------------------------
def lr_cnn():
    p = os.path.join(N.ENV['results_dir'], 'nb_round9', 'encoders', 'lr_choice.csv')
    if os.path.exists(p):
        d = pd.read_csv(p)
        r = d[(d.encoder == 'cnn') & (d.target == 'mid')]
        if len(r):
            return float(r.lr.iloc[0])
    return CNN_LR


def load_decoder(dec, B, mon, seed, data=None, train=True):
    if dec == 'eeg_aug':
        p = N.ckpt_path('repaug', mon, B, seed, 'last')
        if not os.path.exists(p):
            if not train:
                return None
            R5m.SEED = seed
            R5m.train_rep('repaug', mon, B)
        m, _ = N.load_model(p)
        return m.to(dev()).eval().requires_grad_(False)
    enc, tgt = DECODERS[dec]
    p = R9._r9_ckpt(enc, tgt, mon, B, seed, lr_cnn())
    if not os.path.exists(p) and not train:
        return None
    m, _ = R9.train_cell(enc, tgt, mon, B, seed, lr_cnn(), data=data, verbose=False)
    return m.eval().requires_grad_(False)


def train(targets, seeds, mon='posterior17'):
    for seed in seeds:
        for B in targets:
            t0 = time.time()
            if not os.path.exists(N.ckpt_path('repaug', mon, B, seed, 'last')):
                R5m.SEED = seed
                R5m.train_rep('repaug', mon, B)
            enc, tgt = DECODERS['cnn_mid']
            if not os.path.exists(R9._r9_ckpt(enc, tgt, mon, B, seed, lr_cnn())):
                R9.train_cell(enc, tgt, mon, B, seed, lr_cnn(), verbose=False)
            print(f'[train] seed {seed} t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


# layer-1 access for both encoder types
def _h1(m, x):
    e = m.encoder
    if hasattr(e, 'conv'):
        return e.head[0](e.conv(x.unsqueeze(1)).flatten(1))
    return e.model[0](x.reshape(len(x), -1))


def _from_h1(m, h):
    e = m.encoder
    if hasattr(e, 'conv'):
        return m.eeg_projector(e.head[1](h))
    return m.eeg_projector(e.model[2](e.model[1](h)))


@torch.no_grad()
def z_of(m, x, d1=None, bs=1024):
    out = []
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(np.asarray(x[i:i + bs], np.float32), device=dev())
        h = _h1(m, xb)
        out.append(_from_h1(m, h if d1 is None else h + d1))
    return torch.cat(out)


@torch.no_grad()
def h1_mean(m, x, bs=1024):
    s, n = 0, 0
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(np.asarray(x[i:i + bs], np.float32), device=dev())
        s = s + _h1(m, xb).sum(0)
        n += len(xb)
    return s / n


# ---------------------------------------------------------------------------------------------------------------
# evaluation (one pipeline for both decoders)
# ---------------------------------------------------------------------------------------------------------------
def eval_one(dec, B, mon, seed, Q, ses, rows):
    m = load_decoder(dec, B, mon, seed, train=False)
    if m is None:
        print(f'   {dec} seed {seed} t{B}: no checkpoint (run train first), skipping')
        return
    t0 = time.time()
    tgt = DECODERS[dec][1]
    ftr, fte = R9.tgt_feats(tgt, 'train').to(dev()), R9.tgt_feats(tgt, 'test').to(dev())
    lab = N.subject_data(B, mon, keys=('train_labels',))['train_labels']
    pool, _ = N.split(lab)
    Zs, H = [], []
    for s in PEOPLE:
        if s == B:
            continue
        tr_s = N.subject_data(s, mon, keys=('train',))['train']
        Zs.append(z_of(m, tr_s[pool]))
        H.append(h1_mean(m, tr_s[np.random.RandomState(3).choice(pool, 300, replace=False)]))
        del tr_s
    pool_z = torch.stack(Zs).mean(0)
    mS = torch.stack(H).mean(0)
    Bm, xm, ym = X.ridge_svd(ftr[torch.as_tensor(pool, device=dev())], pool_z)
    Ganc, Gimg = nrm((fte - xm) @ Bm + ym), m.gallery(fte)
    pos_of = {int(i): j for j, i in enumerate(pool)}
    tr = ses['train']
    reps = R.load_reps(B, mon)
    cand = np.array([i for i in pool if (tr[i] == CS).sum() == 2])
    for k in KS:
        if k > len(cand):
            continue
        for draw in range(R6m.ndraw(k)):
            ims = GL.draw_images(cand, k, draw)
            rr = np.stack([np.nonzero(tr[i] == CS)[0] for i in ims])
            x2 = np.asarray(reps[ims[:, None], rr], np.float32)
            d1 = mS - h1_mean(m, x2.reshape(-1, *x2.shape[2:]))
            Xs = z_of(m, x2.mean(1), d1)
            Ys = pool_z[torch.as_tensor([pos_of[int(i)] for i in ims], device=dev())]
            Zq = {qk: [z_of(m, q, d1) for q in Q[qk]] for qk in QK}
            lam, _ = R9.cv_pick(lambda a, b, l: R9.e_procrustes(a, b, l, False), Xs, Ys, R9.REL, seed=draw)
            cells = [('zero', 'anchor', None, Ganc), ('zero', 'img', None, Gimg),
                     ('rigid', 'anchor', R9.e_procrustes(Xs, Ys, 1.0), Ganc),
                     ('rot', 'anchor', R9.e_procrustes(Xs, Ys, lam, False), Ganc)]
            base = dict(decoder=dec, seed=seed, k=k, draw=draw, rot_lam=lam)
            for meth, gal, mp, Gt in cells:
                for qk in QK:
                    for qi, z0 in enumerate(Zq[qk]):
                        z = z0 if mp is None else mp(z0)
                        for rule in ('cos', 'csls_batch'):
                            rows.append(dict(base, method=meth, gallery=gal, rule=rule, query=qk, qdraw=qi,
                                             top1=R6m.top1(R6m.smat(z, Gt, rule))))
            if k in K_FT:
                ft = R6m.finetune(m, x2, ims, ftr, seed=draw)
                with torch.no_grad():
                    Zft_cal = ft(torch.as_tensor(x2.mean(1), device=dev()))
                    Gft = ft.gallery(fte)
                lamf, _ = R9.cv_pick(lambda a, b, l: R9.e_procrustes(a, b, l, False), Zft_cal, Ys, R9.REL, seed=draw)
                rotf = R9.e_procrustes(Zft_cal, Ys, lamf, False)
                for qk in QK:
                    for qi, q in enumerate(Q[qk]):
                        with torch.no_grad():
                            zf = ft(torch.as_tensor(q, device=dev()))
                        for meth, gal, z, Gt in (('ft', 'img', zf, Gft), ('ft+rot', 'anchor', rotf(zf), Ganc)):
                            for rule in ('cos', 'csls_batch'):
                                rows.append(dict(base, method=meth, gallery=gal, rule=rule, query=qk, qdraw=qi,
                                                 top1=R6m.top1(R6m.smat(z, Gt, rule))))
                del ft
    print(f'   {dec} seed {seed} t{B}: {time.time() - t0:.0f}s', flush=True)


def evaluate(targets, seeds, mon='posterior17'):
    for seed in seeds:
        for B in targets:
            out = os.path.join(odir('eval'), f'r10_s{seed}_t{B:02d}_{mon}.csv')
            if os.path.exists(out):
                print(f'[eval] seed {seed} t{B}: done')
                continue
            Q, ses = R9._queries(B, mon)
            rows = []
            for dec in DECODERS:
                eval_one(dec, B, mon, seed, Q, ses, rows)
            if not rows:
                continue
            df = pd.DataFrame(rows).groupby(['decoder', 'seed', 'k', 'draw', 'method', 'gallery', 'rule', 'query']).agg(
                top1=('top1', 'mean'), rot_lam=('rot_lam', 'first')).reset_index()
            df['target'], df['montage'] = B, mon
            df.to_csv(out, index=False)
            print(f'[eval] seed {seed} t{B} written', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def holm(ps):
    ks = sorted(ps, key=lambda k: np.nan_to_num(ps[k], nan=1.0))
    out, run = {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (len(ks) - i) * np.nan_to_num(ps[k], nan=1.0)))
        out[k] = round(run, 4)
    return out


def report(mon='posterior17'):
    pd.set_option('display.width', 250)
    f = sorted(glob.glob(os.path.join(odir('eval'), f'r10_s*_t*_{mon}.csv')))
    if not f:
        print('no round 10 results yet')
        return
    d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
    seeds = sorted(d.seed.unique())
    print(f'ROUND 10 ({mon}; seeds {seeds}; folds {sorted(d.target.unique())}; people 2-10 tested; top-1 %)')
    perseed = d.groupby(['decoder', 'seed', 'k', 'method', 'gallery', 'rule', 'query', 'target']).top1.mean()
    per = perseed.groupby(level=['decoder', 'k', 'method', 'gallery', 'rule', 'query', 'target']).mean()   # mean over seeds
    T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3, 4, 5]).mean()
    for rule in ('cos', 'csls_batch'):
        for q in ('s4R4', 's4R1', 'bench'):
            print(f'\n{q}, {rule} (mean over seeds; rows = decoder x method x gallery, columns = calibration images)')
            print(T.xs((rule, q), level=('rule', 'query')).unstack('k').round(2).to_string())

    def g(dec, k, meth, gal='anchor', rule='cos', q='s4R4'):
        return per.xs((dec, k, meth, gal, rule, q), level=('decoder', 'k', 'method', 'gallery', 'rule', 'query'))
    print('\nREPRODUCTION CHECK')
    try:
        a = perseed.xs(('eeg_aug', 2026, 500, 'rigid', 'anchor', 'csls_batch', 's4R4'), level=('decoder', 'seed', 'k', 'method', 'gallery', 'rule', 'query'))
        z = perseed.xs(('eeg_aug', 2026, 500, 'zero', 'anchor', 'csls_batch', 's4R4'), level=('decoder', 'seed', 'k', 'method', 'gallery', 'rule', 'query'))
        R4m.pstats(a, z, 'eeg_aug seed 2026: rigid - zero, k = 500, csls_batch, s4R4 (round 7: +3.89)')
    except KeyError:
        print('  seed 2026 not evaluated')
    print('\nPRIMARY (cos = each query scored on its own; per person, mean over seeds)')
    p0 = R4m.pstats(g('cnn_mid', 500, 'rot'), g('eeg_aug', 500, 'rigid'),
                    'cnn_mid + rot - eeg_aug + rigid (current), k = 500, s4R4, cos', primary=True)
    ps = {'decoder (both rot)': R4m.pstats(g('cnn_mid', 500, 'rot'), g('eeg_aug', 500, 'rot'), 'decoder effect: cnn_mid - eeg_aug, both rot, k = 500'),
          'edit (eeg_aug)': R4m.pstats(g('eeg_aug', 500, 'rot'), g('eeg_aug', 500, 'rigid'), 'edit effect: rot - rigid, eeg_aug, k = 500'),
          'zero-label': R4m.pstats(g('cnn_mid', 500, 'zero'), g('eeg_aug', 500, 'zero'), 'zero labels: cnn_mid - eeg_aug'),
          'primary, csls': R4m.pstats(g('cnn_mid', 500, 'rot', rule='csls_batch'), g('eeg_aug', 500, 'rigid', rule='csls_batch'),
                                      'primary with batch CSLS')}
    print(f'  HOLM over the secondaries: {holm(ps)}')
    print('\nOTHER QUERIES AND BUDGETS (cnn_mid + rot - eeg_aug + rigid, cos)')
    for q in ('s4R1', 's4R20', 'bench'):
        R4m.pstats(g('cnn_mid', 500, 'rot', q=q), g('eeg_aug', 500, 'rigid', q=q), f'k = 500, {q}')
    for k in KS:
        if k != 500:
            R4m.pstats(g('cnn_mid', k, 'rot'), g('eeg_aug', k, 'rigid'), f'k = {k}, s4R4')
    print('\nEACH SEED ALONE (primary contrast)')
    for s in seeds:
        try:
            a = perseed.xs(('cnn_mid', s, 500, 'rot', 'anchor', 'cos', 's4R4'), level=('decoder', 'seed', 'k', 'method', 'gallery', 'rule', 'query'))
            b = perseed.xs(('eeg_aug', s, 500, 'rigid', 'anchor', 'cos', 's4R4'), level=('decoder', 'seed', 'k', 'method', 'gallery', 'rule', 'query'))
            R4m.pstats(a, b, f'seed {s}')
        except KeyError:
            pass
    print('\nCALIBRATION GAINS WITHIN EACH DECODER (rot - zero, cos, s4R4)')
    for dec in DECODERS:
        for k in KS:
            R4m.pstats(g(dec, k, 'rot'), g(dec, k, 'zero'), f'{dec}: rot - zero, k = {k}')
    print('\nFINE-TUNING')
    for k in K_FT:
        for dec in DECODERS:
            R4m.pstats(g(dec, k, 'ft+rot'), g(dec, k, 'ft', 'img'), f'{dec}: ft+rot - ft, k = {k}')
            R4m.pstats(g(dec, k, 'ft+rot'), g(dec, k, 'rot'), f'{dec}: ft+rot - rot, k = {k}')
        R4m.pstats(g('cnn_mid', k, 'ft+rot'), g('eeg_aug', k, 'ft+rot'), f'decoder effect with ft+rot, k = {k}')
    sf = sorted(glob.glob(os.path.join(N.ENV['results_dir'], 'nb_gallery', 'scratch', f'scratch_t*_{mon}.csv')))
    if sf:
        s = pd.concat([pd.read_csv(x) for x in sf], ignore_index=True)
        ep = int(s[s.target == 1].groupby('epochs').top1.mean().idxmax()) if 1 in set(s.target) else int(s.epochs.max())
        print(f'\nFROM-SCRATCH DECODER ON THE SAME IMAGES (round 8, {ep} epochs, cos, s4R4)')
        for k in (500, 2000):
            sc = s[(s.k == k) & (s.epochs == ep) & (s['query'] == 's4R4') & (s.rule == 'cos')].groupby('target').top1.mean()
            R4m.pstats(g('cnn_mid', k, 'rot'), sc, f'cnn_mid + rot - scratch, k = {k}')
    best = T.xs(('cos', 's4R4'), level=('rule', 'query'))
    try:
        print(f'\nLEVELS (s4R4, cos, mean over seeds): eeg_aug zero {best.loc[("eeg_aug", 500, "zero", "anchor")]:.2f}, '
              f'eeg_aug rigid {best.loc[("eeg_aug", 500, "rigid", "anchor")]:.2f}, cnn_mid zero {best.loc[("cnn_mid", 500, "zero", "anchor")]:.2f}, '
              f'cnn_mid rot {best.loc[("cnn_mid", 500, "rot", "anchor")]:.2f} (k = 500); '
              f'cnn_mid ft+rot at 2000 {best.loc[("cnn_mid", 2000, "ft+rot", "anchor")]:.2f}')
    except KeyError:
        pass
    print(f'chosen rotation strength (most common): {d.groupby("decoder").rot_lam.agg(lambda x: x.mode().iloc[0]).to_dict()}')
    if p0 is not None and not np.isnan(p0):
        dm = (g('cnn_mid', 500, 'rot') - g('eeg_aug', 500, 'rigid')).reindex(TEST).mean()
        print(f'\nDECISION: primary {"PASSES" if p0 < 0.05 and dm > 0 else "does not pass"} (mean {dm:+.2f}, Wilcoxon p = {p0:.4f})')
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'eval', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    p.add_argument('--montage', default='posterior17')
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'train':
        return train(a.targets, a.seeds, a.montage)
    if a.cmd == 'eval':
        return evaluate(a.targets, a.seeds, a.montage)
    report(a.montage)


if __name__ == '__main__':
    main()
