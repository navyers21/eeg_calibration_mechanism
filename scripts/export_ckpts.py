"""Export cross-subject decoders before / after calibration and within-person decoders for outside analysis (erasure).

    python scripts/export_ckpts.py within  --config C [--people 1..10]   # optional: recipe-matched within-person decoders
    python scripts/export_ckpts.py export  --config C --out DIR [--seed 2025] [--people 1..10]

Stages per person B (posterior17, the montage of every calibration result so far):
  f0          repaug EEGProject trained on the 9 other people (rounds 5-12 decoder), image gallery
  zero        f0 + label-free layer-1 offset from 500 calibration images' EEG (labels unused), population gallery
  rot100      zero's pipeline with 100 images + output rotation shrunk toward identity (strength by 5-fold CV), the
              round 10 'rot' arm, population gallery
  rot500      same with 500 images (~3 min of calibration EEG)
  within      nb_train 'within' decoder (B's own pool, averaged-repeat recipe; the one the interp experiments used)
  within_rep  repaug recipe on B's own pool, same optimizer steps as f0 (only if `within` was run), so recipe and
              architecture match f0 and only the training people differ
Calibration uses session 3 (round 6 draw 0), so session-4 queries and the benchmark stay unseen. Weights are fp16.
The export re-scores every stage (s4R4, bench) and checks the folded model against the pipeline's own outputs.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
from eegalign import nb as N  # noqa: E402
import nb_round3 as R  # noqa: E402
import nb_round5 as R5m  # noqa: E402
import nb_round6 as R6m  # noqa: E402
import nb_gallery as GL  # noqa: E402
import nb_round9 as R9  # noqa: E402
import nb_round10 as R10  # noqa: E402

MON = 'posterior17'
DELTA_KEYS = ('encoder.model.0.bias', 'out.weight', 'out.bias')
PEOPLE = list(range(1, 11))


README = """# Cross-subject and within-person checkpoints (THINGS-EEG2, posterior17)

Exported by `scripts/export_ckpts.py` on the main branch; scores and calibration images are in `manifest.json`.

```python
from ckpt_loader import load_stage            # torch + numpy only
m = load_stage('.', person=3, stage='rot500')
z = m(x)                                       # x: (n, 17, 250), our preprocessing; z: (n, 512)
top1 = (F.normalize(z, dim=-1) @ m.gallery_test.T).argmax(1)
```

| stage | what it is | scored against |
|---|---|---|
| f0 | repaug EEGProject trained on the other 9 people (the decoder every calibration round uses) | image gallery |
| zero | f0 + label-free layer-1 offset from 500 calibration images' EEG | population gallery |
| rot100 | offset + output rotation shrunk toward identity, 100 labeled images (~40 s) | population gallery |
| rot500 | same with 500 images (~3 min) | population gallery |
| within | trained on the person's own pool (nb_train recipe, used in the interp experiments) | image gallery |
| within_rep | f0's exact recipe and optimizer steps on the person's own pool, if present | image gallery |

All stages share layer names (encoder.model[0..2], eeg_projector, out), so hooks line up. In calibrated stages the
layer-1 offset is folded into encoder.model[0].bias, so activations from layer 1 on are the calibrated ones; the
rotation is the `out` layer (identity elsewhere). The rotation only acts after the projector, so anything measured
before `out` sees the offset but not the rotation.

Calibration is fit on session 3 (2 presentations per image); session-4 queries and the 80-presentation benchmark are
untouched. Weights are fp16. Seed 2025 for every cross-subject stage. People 2-10 are the evaluation cohort; person 1
was used for tuning choices in earlier rounds.

To rebuild the gallery for other images: population gallery = normalize((feats - pop_ridge.xm) @ pop_ridge.B + pop_ridge.ym),
image gallery = normalize(img_projector(feats)), with feats = NeuroBridge's 1024-d augmented CLIP features. Both are
stored in f0.pt (and within.pt has its own img_projector).
"""


def dev():
    return N.ENV['device']


def nrm(x):
    return F.normalize(x.float(), dim=-1)


def train_within(people, seed):
    """Recipe-matched within-person decoders: repaug recipe, B's own pool, 9x the epochs so optimizer steps match f0."""
    R5m.SEED = seed
    for B in people:
        R5m.train_rep('repaug', MON, B, epochs=450, people=[B], save_as='within_rep')


def stage_dict(m, d1=None, rot=None):
    """Plain state dict for ckpt_loader.Stage: d1 folded into the first Linear's bias, rotation as the `out` layer."""
    sd = {f'encoder.{k}': v.detach().clone() for k, v in m.encoder.state_dict().items()}
    sd.update({f'eeg_projector.{k}': v.detach().clone() for k, v in m.eeg_projector.state_dict().items()})
    if d1 is not None:
        sd['encoder.model.0.bias'] = sd['encoder.model.0.bias'] + d1.to(sd['encoder.model.0.bias'].device)
    if rot is None:
        W, b = torch.eye(N.PROJ_DIM), torch.zeros(N.PROJ_DIM)
    else:                                        # z -> (z - xm) @ R + ym  ==  z @ R + (ym - xm @ R)
        W, b = rot.R.T, (rot.ym - rot.xm @ rot.R).reshape(-1)
    sd['out.weight'], sd['out.bias'] = W, b
    return {k: v.detach().cpu().half() for k, v in sd.items()}


def score(z_list, G):
    return float(np.mean([R6m.top1(R6m.smat(z, G, 'cos')) for z in z_list]))


def check(sd, m_fn, x):
    """Max abs difference between the exported module (fp16 weights) and the pipeline's own computation."""
    import ckpt_loader as L
    st = L.Stage(sd['encoder.model.0.weight'].shape[1] // N.T_LEN)
    st.load_state_dict({k: v.float() for k, v in sd.items()})
    st = st.to(dev()).eval()
    with torch.no_grad():
        a = st(torch.as_tensor(x[:64], device=dev()))
    return float((nrm(a) - nrm(m_fn(x[:64]))).abs().max())


def export_person(B, seed, out, man):
    m = R10.load_decoder('eeg_aug', B, MON, seed, train=False)
    if m is None:
        print(f'  t{B}: no repaug seed {seed} checkpoint, skipping')
        return
    fte_raw = N.features('test', 'aug').to(dev())
    ftr = N.features('train', 'aug').to(dev())
    Q, ses = R9._queries(B, MON)
    lab = N.subject_data(B, MON, keys=('train_labels',))['train_labels']
    pool, _ = N.split(lab)
    Zs, H = [], []
    for s in PEOPLE:
        if s == B:
            continue
        tr_s = N.subject_data(s, MON, keys=('train',))['train']
        Zs.append(R10.z_of(m, tr_s[pool]))
        H.append(R10.h1_mean(m, tr_s[np.random.RandomState(3).choice(pool, 300, replace=False)]))
        del tr_s
    pool_z, mS = torch.stack(Zs).mean(0), torch.stack(H).mean(0)
    import nb_next as X
    Bm, rxm, rym = X.ridge_svd(ftr[torch.as_tensor(pool, device=dev())], pool_z)
    Ganc, Gimg = nrm((fte_raw - rxm) @ Bm + rym), m.gallery(fte_raw)
    pos_of = {int(i): j for j, i in enumerate(pool)}
    tr, reps = ses['train'], R.load_reps(B, MON)
    cand = np.array([i for i in pool if (tr[i] == R10.CS).sum() == 2])
    extras = dict(img_projector={k: v.cpu().half() for k, v in m.img_projector.state_dict().items()},
                  pop_ridge=dict(B=Bm.cpu().half(), xm=rxm.cpu(), ym=rym.cpu()))   # population gallery for any image
    P = man['people'].setdefault(str(B), dict(stages={}))
    xq = Q['s4R4'][0]

    def save(name, sd, G, info):
        """Calibrated stages share f0's weights, so they store only what changes (layer-1 bias, `out`)."""
        f = f'p{B:02d}/{name}.pt'
        os.makedirs(os.path.join(out, f'p{B:02d}'), exist_ok=True)
        body = dict(state_dict=sd)
        if info.get('base') == 'f0':
            body = dict(delta={k: sd[k] for k in DELTA_KEYS})
        torch.save(dict(body, gallery_test=G.cpu().half(), **info), os.path.join(out, f))
        P['stages'][name] = dict(file=f, **{k: v for k, v in info.items() if isinstance(v, (int, float, str))})
        print(f'  t{B} {name:10s} s4R4 {info["s4R4_top1"]:5.2f}  bench {info["bench_top1"]:5.2f}  '
              f'check {info["max_abs_diff"]:.1e}', flush=True)

    zq = {k: [R10.z_of(m, q) for q in Q[k]] for k in ('s4R4', 'bench')}
    sd = stage_dict(m)
    save('f0', sd, Gimg, dict(seed=seed, gallery='image', s4R4_top1=score(zq['s4R4'], Gimg),
                              bench_top1=score(zq['bench'], Gimg), max_abs_diff=check(sd, lambda x: R10.z_of(m, x), xq),
                              train_people=[s for s in PEOPLE if s != B], **extras))
    for k in (100, 500):
        ims = GL.draw_images(cand, k, 0)
        rr = np.stack([np.nonzero(tr[i] == R10.CS)[0] for i in ims])
        x2 = np.asarray(reps[ims[:, None], rr], np.float32)
        d1 = mS - R10.h1_mean(m, x2.reshape(-1, *x2.shape[2:]))
        Xs = R10.z_of(m, x2.mean(1), d1)
        Ys = pool_z[torch.as_tensor([pos_of[int(i)] for i in ims], device=dev())]
        zc = {q: [R10.z_of(m, x, d1) for x in Q[q]] for q in ('s4R4', 'bench')}
        base = dict(seed=seed, base='f0', gallery='population', calib_k=k, calib_session=3, calib_images=ims.tolist())
        if k == 500:
            sd = stage_dict(m, d1)
            save('zero', sd, Ganc, dict(base, s4R4_top1=score(zc['s4R4'], Ganc), bench_top1=score(zc['bench'], Ganc),
                                        max_abs_diff=check(sd, lambda x: R10.z_of(m, x, d1), xq)))
        lam, _ = R9.cv_pick(lambda a, b, l: R9.e_procrustes(a, b, l, False), Xs, Ys, R9.REL, seed=0)
        rot = R9.e_procrustes(Xs, Ys, lam, False)
        rot.xm, rot.ym = Xs.mean(0), Ys.mean(0)
        assert torch.allclose(rot(Xs[:8]), (Xs[:8] - rot.xm) @ rot.R + rot.ym, atol=1e-4), 'centering differs'
        sd = stage_dict(m, d1, rot)
        save(f'rot{k}', sd, Ganc, dict(base, rot_lambda=float(lam), s4R4_top1=score([rot(z) for z in zc['s4R4']], Ganc),
                                       bench_top1=score([rot(z) for z in zc['bench']], Ganc),
                                       max_abs_diff=check(sd, lambda x: rot(R10.z_of(m, x, d1)), xq)))
    for kind, which, seed_w in (('within', 'last', 2025), ('within_rep', 'last', seed)):
        p = N.ckpt_path(kind, MON, B, seed_w, which)
        if not os.path.exists(p):
            print(f'  t{B} {kind:10s} not found ({os.path.basename(p)}), skipping')
            continue
        w, ck = N.load_model(p)
        w = w.requires_grad_(False)
        Gw = w.gallery(fte_raw)
        zw = {q: [R10.z_of(w, x) for x in Q[q]] for q in ('s4R4', 'bench')}
        sd = stage_dict(w)
        save(kind, sd, Gw, dict(seed=seed_w, gallery='image', s4R4_top1=score(zw['s4R4'], Gw),
                                bench_top1=score(zw['bench'], Gw), max_abs_diff=check(sd, lambda x: R10.z_of(w, x), xq),
                                train_people=[B], recipe=str(ck.get('cfg', {}).get('recipe', 'nb_train within')),
                                img_projector={k: v.cpu().half() for k, v in w.img_projector.state_dict().items()}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['within', 'export'])
    p.add_argument('--config', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--people', type=int, nargs='+', default=PEOPLE)
    a = p.parse_args()
    N.setup(a.config)
    if a.cmd == 'within':
        return train_within(a.people, a.seed)
    out = a.out or os.path.join(N.ENV['results_dir'], 'export')
    os.makedirs(out, exist_ok=True)
    mp = os.path.join(out, 'manifest.json')
    man = json.load(open(mp)) if os.path.exists(mp) else dict(montage=MON, input_shape=[17, 250], people={})
    for B in a.people:
        export_person(B, a.seed, out, man)
        json.dump(man, open(mp, 'w'), indent=1)
    import shutil
    shutil.copy(os.path.join(ROOT, 'scripts', 'ckpt_loader.py'), out)
    open(os.path.join(out, 'README.md'), 'w').write(README)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
