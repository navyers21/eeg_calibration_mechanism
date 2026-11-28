"""Round 15: (A) does calibration add on top of the better gallery? (B) can a second image encoder predict what the
Round 14 gallery still misses? Frozen seed-2026 repaug NeuroBridge decoders; no EEG training.

    python scripts/nb_round15.py selftest
    python scripts/nb_round15.py dino   --config C                     # B: DINOv2 ViT-B/14 features, once
    python scripts/nb_round15.py run    --config C [--targets 2 3 ...]   # people 2-10 by default
    python scripts/nb_round15.py report --config C

Why. Round 14B: a gallery predicted from pooled RN50 layers 2-4 plus a layer-3 4x4 spatial residual (G2) beat the
current predicted gallery (G0) by +3.64 top-1 (9/9) with no target data; shuffling grid positions removed the spatial
part. Measured population templates still score 19.14% vs 12.85% for G2 on the same queries (a 6.29-point gap, part of
which may be out of reach from images). Round 8's rigid map added +0.09 on G2 at k = 500 vs +0.82 on G0.

Setup = Round 14B: R6 context, Round 8 evaluator, label-free layer-1 offset from the calibration trials, session-4
queries, k training-pool images x 2 session-3 viewings (Round 6 draws: 5 at k = 100, 3 at k = 500; no official test
image is ever used for calibration), each query scored alone by cosine (PRIMARY) or batch CSLS. G2 is LOCKED: rebuilt
from Round 14B's saved per-person settings and gated against its zero-label cells.

A. CALIBRATION ON THE BETTER GALLERY (2 x 2 and more, identical draws in every cell)
  Galleries G0, G2, G2w (G2's family with a wider search: layer weights in {0, 0.5, 1, 1.5, 2, 3} by coordinate ascent
  from Round 14's choice, then residual scale in {0.5, 1, 1.5, 2, 3, 4}; secondary only), Gstar (measured, headroom).
  Methods: zero (offset only); rot_pred (rotation fitted to THIS gallery's templates of the calibration images, shrinkage
  by 5-fold CV over the calibration images); rot (same rotation fitted to the measured source responses, Round 11);
  lf (label-free: Round 13 locked recovery, inferred pairs refit with CV shrinkage).
  PRIMARY A: on G2, rot_pred - zero, k = 500, s4R4, cos, people 2-10. Decision (fixed before running): continue the
  calibration branch if the gain is >= +1 point with a 95% paired interval above zero; if the interval excludes +1, report
  that better shared templates reduced what person-specific rotation adds; otherwise unresolved. Secondary: the same at
  k = 100; per-person interaction (rot_pred - zero on G2) - (rot_pred - zero on G0) at k = 500; rot; lf; G2w - G2.

B. A SECOND VISUAL REPRESENTATION FOR WHAT G2 MISSES
  Target: the measured source-mean response minus G2's prediction, with G2 refit out of fold by concept (5 folds; G2's
  own spatial head is cross-fitted inside each fold), in G2's coordinates before normalization. A ridge head (GCV) maps
  image features to that residual; the gallery is normalize(G2 + gamma * head), gamma in {0, 0.25, 0.5, 1, 1.5, 2}
  chosen on Round 14's 2 held-out source people (gamma = 0 lets validation discard the addition).
    G3d    DINOv2 ViT-B/14 (frozen): patch tokens of blocks 4, 8, 12, pooled globally and on a 4 x 4 grid (grid channels
           reduced to 64 per block by PCA on training-pool images) -> 2,304 + 3,072 = 5,376 dims
    G3dp   DINOv2 pooled only (2,304 dims): do the grids earn their place?
    G3c    matched CLIP control: the SAME procedure on CLIP features G2 already uses (final + pooled layers 2-4 + layer-3
           grid, 5,632 dims): a gain from a second fit, not a new representation
  Every column z-scored with training-pool statistics. PRIMARY B: G3d - G2, zero labels, s4R4, cos, people 2-10;
  practical threshold +2 points over G2 (set before running). Secondary (Holm over two): G3d - G3c, G3d - G3dp.
  Diagnostics: anchor cosine to Gstar, and the margin of the correct template over the strongest competitor (a better
  match to measured templates that leaves retrieval flat means stop expanding the feature bank).

Also written: example_queries_tXX.json (G2, zero labels, s4R4 draw 0): for each of the 200 queries the correct concept,
its rank and the top-5 concepts with cosine scores, for a real worked example on the poster.
Not run here (only if B falls short): retraining the EEG head toward G2 templates against matched continued training.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
import nb_round14 as R14  # noqa: E402

MON = 'posterior17'
SEED = 2026
TEST = list(range(2, 11))
KS = (100, 500)
W2 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
S2 = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SWEEPS = 2
QK = ['s4R1', 's4R4', 's4R20', 'bench']
GALS = ['G0', 'G2', 'G2w', 'G3d', 'G3dp', 'G3c', 'Gstar']
CAL_GALS = ['G0', 'G2', 'G2w', 'Gstar']
GAMMAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
DINO_BLOCKS = (4, 8, 12)
THRESH_A, THRESH_B = 1.0, 2.0
METHODS = ['zero', 'rot', 'rot_pred', 'lf']
l2n = R14.l2n


def parse_w(s):
    return {k: float(v) for k, v in re.findall(r'(l\d)([\d.]+)', s)}


def coord_ascent(score, start, grid=W2, keys=R14.G1_BLOCKS, sweeps=SWEEPS):
    """Maximize score(w) one block at a time over grid; ties -> smaller total weight. Returns (best w, all scores)."""
    seen = {}

    def ev(w):
        k = R14.wkey(w)
        if k not in seen:
            seen[k] = (score(w), w)
        return seen[k][0]
    best = dict(start)
    bv = ev(best)
    for _ in range(sweeps):
        moved = False
        for b in keys:
            for v in grid:
                w = dict(best, **{b: v})
                sv = ev(w)
                if (round(sv, 6), -sum(w.values())) > (round(bv, 6), -sum(best.values())):
                    best, bv, moved = w, sv, True
        if not moved:
            break
    return best, {k: v[0] for k, v in seen.items()}


def selftest():
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f'  [{"ok" if cond else "FAIL"}] {name} {detail}')
    check('parse_w', parse_w('l21|l30.5|l40') == {'l2': 1.0, 'l3': 0.5, 'l4': 0.0})
    target = {'l2': 2.0, 'l3': 0.5, 'l4': 3.0}
    f = lambda w: -sum((w[k] - target[k]) ** 2 for k in w)  # noqa: E731
    best, seen = coord_ascent(f, {'l2': 1.0, 'l3': 1.0, 'l4': 1.0})
    check('coordinate ascent finds a separable optimum', best == target, f'({best}, {len(seen)} evaluations)')
    flat = lambda w: 1.0  # noqa: E731
    best, _ = coord_ascent(flat, {'l2': 1.0, 'l3': 1.0, 'l4': 1.0})
    check('ties go to the smaller total weight', sum(best.values()) == 0.0, f'({best})')
    print('SELFTEST PASSED' if ok else 'SELFTEST FAILED')
    return ok


def _m():
    from eegalign import nb as N
    import nb_round3 as R
    import nb_round4 as R4m
    import nb_round5 as R5m
    import nb_round6 as R6m
    import nb_round9 as R9
    import nb_round11 as R11
    import nb_round13 as R13
    import nb_gallery as GL
    import nb_next as X
    import nb_interp as I
    return dict(N=N, R=R, R4m=R4m, R5m=R5m, R6m=R6m, R9=R9, R11=R11, R13=R13, GL=GL, X=X, I=I)


def odir(sub=''):
    d = os.path.join(_m()['N'].ENV['results_dir'], 'nb_round15', sub)
    os.makedirs(d, exist_ok=True)
    return d


def dino_paths():
    d = _m()['N'].ENV['feat_dir']
    return {s: {**{f'g{b}': os.path.join(d, f'image_{s}_dinov2b_b{b}_g4.npy') for b in DINO_BLOCKS},
                'pool': os.path.join(d, f'image_{s}_dinov2b_pool.npy')} for s in ('train', 'test')}


def image_root(local='/content/things_images', hf_repo='Haitao999/things-eeg'):
    root = next((os.path.join(r, d) for r, ds, _ in os.walk(local) for d in ds if d == 'Image_set_Resize'), None)
    if root is None:
        from huggingface_hub import list_repo_files, snapshot_download
        import zipfile
        import tarfile
        hit = [f for f in list_repo_files(hf_repo, repo_type='dataset') if 'Image_set_Resize' in f]
        assert hit, f'no Image_set_Resize in {hf_repo}'
        pre = hit[0][:hit[0].index('Image_set_Resize')]
        snapshot_download(hf_repo, repo_type='dataset', local_dir=local, allow_patterns=[f'{pre}Image_set_Resize*'],
                          max_workers=4)
        for arc in glob.glob(f'{local}/**/Image_set_Resize*.zip', recursive=True) + \
                glob.glob(f'{local}/**/Image_set_Resize*.tar*', recursive=True):
            (zipfile.ZipFile(arc) if arc.endswith('.zip') else tarfile.open(arc)).extractall(os.path.dirname(arc))
        root = next((os.path.join(r, d) for r, ds, _ in os.walk(local) for d in ds if d == 'Image_set_Resize'), None)
    assert root, 'Image_set_Resize folder not found'
    return root


def dino_feats(model_id='facebook/dinov2-base'):
    """DINOv2 ViT-B/14 patch tokens of blocks 4, 8, 12 for every THINGS-EEG2 image (NeuroBridge order, the listing that
    passed Round 9/14's order check): global mean (768 per block) and a 4 x 4 grid (float16)."""
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    paths = dino_paths()
    if all(os.path.exists(p) for v in paths.values() for p in v.values()):
        print('DINOv2 features exist')
        return
    dev = _m()['N'].ENV['device']
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()
    root = image_root()
    for split, sub, n_exp in (('test', 'test_images', 200), ('train', 'train_images', 16540)):
        d0 = os.path.join(root, sub)
        ps = [os.path.join(d0, cd, f) for cd in sorted(os.listdir(d0)) for f in sorted(os.listdir(os.path.join(d0, cd)))]
        assert len(ps) == n_exp, f'{split}: {len(ps)} images, expected {n_exp}'
        grids = {b: [] for b in DINO_BLOCKS}
        pooled = []
        with torch.no_grad():
            for i in range(0, len(ps), 64):
                x = proc(images=[Image.open(p).convert('RGB') for p in ps[i:i + 64]], return_tensors='pt')['pixel_values'].to(dev)
                hs = model(pixel_values=x, output_hidden_states=True).hidden_states
                pv = []
                for b in DINO_BLOCKS:
                    t = hs[b][:, 1:, :]                                            # patch tokens (no CLS)
                    n = int(round(t.shape[1] ** 0.5))
                    t = t.reshape(len(t), n, n, -1).permute(0, 3, 1, 2)
                    grids[b].append(F.adaptive_avg_pool2d(t, R14.GRID).flatten(2).transpose(1, 2).half().cpu())
                    pv.append(t.mean((2, 3)))
                pooled.append(torch.cat(pv, 1).float().cpu())
        np.save(paths[split]['pool'], torch.cat(pooled).numpy())
        for b in DINO_BLOCKS:
            np.save(paths[split][f'g{b}'], torch.cat(grids[b]).numpy())
        print(f'  {split}: {len(ps)} images', flush=True)
    print('saved', paths)


def zfit(Xtr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return lambda X: (X - mu) / sd  # noqa: E731


class ResFeats:
    """Feature sets for the residual head, every column z-scored with training-pool statistics."""

    def __init__(self, c, Fe):
        dev = c.ftr.device
        ip = torch.as_tensor(c.pool, device=dev)
        paths = dino_paths()
        for v in paths.values():
            for p in v.values():
                assert os.path.exists(p), f'{p} missing: run `nb_round15.py dino` first'
        dp = {s: torch.from_numpy(np.load(paths[s]['pool'])).to(dev) for s in ('train', 'test')}
        dg = {s: [] for s in ('train', 'test')}
        samp = np.sort(np.random.RandomState(1).choice(c.pool, min(3000, len(c.pool)), replace=False))
        for b in DINO_BLOCKS:
            g = {s: np.load(paths[s][f'g{b}'], mmap_mode='r') for s in ('train', 'test')}
            cells = torch.from_numpy(np.asarray(g['train'][samp], np.float32)).to(dev).reshape(-1, g['train'].shape[2])
            mu = cells.mean(0)
            _, V = torch.linalg.eigh(torch.cov((cells - mu).T))
            V = V[:, -R14.N_PC:]
            del cells
            for s in ('train', 'test'):
                dg[s].append(torch.cat([((torch.from_numpy(np.asarray(g[s][i:i + 2048], np.float32)).to(dev) - mu) @ V).flatten(1)
                                        for i in range(0, len(g[s]), 2048)]))
        raw = {'dino': {s: torch.cat([dp[s]] + dg[s], 1) for s in ('train', 'test')},
               'dino_pool': {s: dp[s] for s in ('train', 'test')},
               'clip': {s: torch.cat([Fe.fin[s]] + [Fe.blocks[s][k] for k in R14.G1_BLOCKS] + [Fe.sp[s]], 1)
                        for s in ('train', 'test')}}
        self.X = {}
        for k, v in raw.items():
            z = zfit(v['train'][ip])
            self.X[k] = {s: z(v[s]) for s in ('train', 'test')}
        self.dims = {k: int(v['train'].shape[1]) for k, v in self.X.items()}


def g2_raw(Fe, fit, w, s, split, idx=None):
    """G2's prediction before normalization."""
    x = Fe.g1x(split, w, idx)
    sp = Fe.sp[split] if idx is None else Fe.sp[split][idx]
    return R14.apply_ridge(fit['g1'], x) + s * R14.apply_ridge(fit['head']['real'], sp)


def residual_galleries(c, Fe, w14, s14, fits14, Yval, Yall, vq, labels):
    """G3 arms: ridge heads on concept-out-of-fold G2 residuals; gamma by source validation. fits14 = G2 fits for
    [Yval, Yall] on the whole pool. Returns {name: test gallery}, {name: fn(split, idx)}, info."""
    RF = ResFeats(c, Fe)
    dev = c.ftr.device
    fo = R14.concept_folds(labels)
    oof = [torch.zeros_like(Yval), torch.zeros_like(Yall)]
    for f in range(R14.N_FOLD):
        tr_, te_ = np.nonzero(fo != f)[0], np.nonzero(fo == f)[0]
        ft = R14.fit_gallery(Fe, c.pool[tr_], [Yval[tr_], Yall[tr_]], w14, labels[tr_], spatial=('real',))
        ite = torch.as_tensor(c.pool[te_], device=dev)
        for j in range(2):
            oof[j][torch.as_tensor(te_, device=dev)] = g2_raw(Fe, ft[j], w14, s14, 'train', ite)
    res = [Yval - oof[0], Yall - oof[1]]
    ip = torch.as_tensor(c.pool, device=dev)
    names = {'G3d': 'dino', 'G3dp': 'dino_pool', 'G3c': 'clip'}
    gals, fns, info = {}, {}, {'dims': RF.dims, 'resid_norm_ratio': float(res[1].norm() / Yall.norm())}
    for gname, fk in names.items():
        hv, ha = R14.ridge_fit(RF.X[fk]['train'][ip], res)
        val = {}
        for gm in GAMMAS:
            fn = (lambda h, gm_: (lambda split, idx: l2n(g2_raw(Fe, fits14[0], w14, s14, split, idx)
                                                         + gm_ * R14.apply_ridge(h, RF.X[fk][split][idx]))))(hv, gm)
            val[gm] = R14.val_score(fn, c, vq)
        gm = max(GAMMAS, key=lambda g_: (round(val[g_], 6), -g_))
        fns[gname] = (lambda h, gm_, fk_: (lambda split, idx=None: l2n(
            g2_raw(Fe, fits14[1], w14, s14, split, idx)
            + gm_ * R14.apply_ridge(h, RF.X[fk_][split] if idx is None else RF.X[fk_][split][idx]))))(ha, gm, fk)
        with torch.no_grad():
            gals[gname] = fns[gname]('test')
        info[f'gamma_{gname}'] = gm
        info[f'val_{gname}'] = {f'{k:g}': v for k, v in val.items()}
    return gals, fns, info


def build(c, meta14):
    """G2 from Round 14's saved settings; G2w from the wider search. Returns galleries, fns, info."""
    M = _m()
    I, N = M['I'], M['N']
    t0 = time.time()
    Fe = R14.Feats(c)
    Zp = []
    for s in c.src:
        x = N.subject_data(s, MON, keys=('train',))['train']
        Zp.append(torch.cat([I.acts(c.f0, x[c.pool[i:i + 4096]].astype(np.float32))['z']
                             for i in range(0, len(c.pool), 4096)]))
        del x
    Zp = torch.stack(Zp)
    vpos = np.sort(np.random.RandomState(c.B).choice(len(c.src), R14.N_VAL, replace=False))   # Round 14's people
    assert [c.src[i] for i in vpos] == meta14['val_people'], 'validation people differ from Round 14'
    rest = [i for i in range(len(c.src)) if i not in set(vpos)]
    Yval, Yall = Zp[rest].mean(0), c.pool_z
    vq = [c.Zsrc_held[i] for i in vpos]
    del Zp
    labels = c.labels[c.pool]
    w14, s14 = parse_w(meta14['w_G1']), float(meta14['s_G2'])
    # wider search: layer weights by coordinate ascent, then residual scale
    score = lambda w: R14.val_score(R14.gallery_fn(Fe, R14.fit_gallery(Fe, c.pool, [Yval], w)[0], w), c, vq)  # noqa: E731
    w2, seen = coord_ascent(score, w14)
    fv = R14.fit_gallery(Fe, c.pool, [Yval], w2, labels, spatial=('real',))[0]
    sv = {s_: R14.val_score(R14.gallery_fn(Fe, fv, w2, 'real', s_), c, vq) for s_ in S2}
    s2 = max(S2, key=lambda s_: (round(sv[s_], 6), -abs(s_ - 1.0)))
    # final fits on all 9 source people
    fits14 = R14.fit_gallery(Fe, c.pool, [Yval, Yall], w14, labels, spatial=('real',))
    fa14 = fits14[1]
    fa2 = fa14 if R14.wkey(w2) == R14.wkey(w14) else R14.fit_gallery(Fe, c.pool, [Yall], w2, labels, spatial=('real',))[0]
    fns = {'G0': lambda split, idx: c.galfn['anchor']((c.ftr if split == 'train' else c.fte)[idx]),
           'G2': R14.gallery_fn(Fe, fa14, w14, 'real', s14), 'G2w': R14.gallery_fn(Fe, fa2, w2, 'real', s2)}
    gals = {'G0': c.gal['anchor'][0], 'Gstar': c.gal['oracle'][0]}
    with torch.no_grad():
        gals['G2'], gals['G2w'] = fns['G2']('test'), fns['G2w']('test')
    g3, f3, info3 = residual_galleries(c, Fe, w14, s14, fits14, Yval, Yall, vq, labels)
    gals.update(g3)
    fns.update(f3)
    info = dict(w_G2=R14.wkey(w14), s_G2=s14, w_G2w=R14.wkey(w2), s_G2w=s2, val_G1w=seen[R14.wkey(w2)],
                val_G2w={f'{k:g}': v for k, v in sv.items()}, n_evals=len(seen),
                w_at_edge=[k for k, v in w2.items() if v == W2[-1]], s_at_edge=s2 == S2[-1],
                anchor_cos_to_Gstar={k: float((gals[k] * gals['Gstar']).sum(1).mean()) for k in gals},
                build_s=round(time.time() - t0), **info3)
    print(f'   galleries ({info["build_s"]}s): G2 [{info["w_G2"]}, s {s14:g}]; G2w [{info["w_G2w"]}, s {s2:g}], '
          f'val G1w {info["val_G1w"]:.2f}, G2w {sv[s2]:.2f} ({len(seen)} weight settings tried); gamma G3d '
          f'{info["gamma_G3d"]:g}, G3dp {info["gamma_G3dp"]:g}, G3c {info["gamma_G3c"]:g}', flush=True)
    return gals, fns, info


def run_target(B):
    out = os.path.join(odir('eval'), f'r15_t{B:02d}.csv')
    if os.path.exists(out):
        print(f'[15] t{B}: done')
        return
    M = _m()
    N, R, R5m, R6m, R9, R11, R13, GL, X, I = (M[k] for k in ('N', 'R', 'R5m', 'R6m', 'R9', 'R11', 'R13', 'GL', 'X', 'I'))
    m14 = os.path.join(N.ENV['results_dir'], 'nb_round14', 'b', f'r14b_t{B:02d}_meta.json')
    if not os.path.exists(m14):
        print(f'[15] t{B}: no Round 14B settings ({m14}), skipping')
        return
    meta14 = json.load(open(m14))
    lock = json.load(open(os.path.join(N.ENV['results_dir'], 'nb_round13', 'locked_settings.json')))
    v13 = lock['variant']
    R5m.SEED = SEED
    t0 = time.time()
    ctx = R6m.Ctx('repaug', B, MON)
    c = ctx.c
    gals, fns, info = build(c, meta14)
    tr = c.ses['train']
    reps = R.load_reps(B, MON)
    cand = np.array([i for i in c.pool if (tr[i] == GL.CS).sum() == 2])
    eye = torch.eye(512, device=c.pool_z.device)
    rows, example = [], None
    for k in KS:
        for draw in range(R6m.ndraw(k)):
            ims = GL.draw_images(cand, k, draw)
            rr = np.stack([np.nonzero(tr[i] == GL.CS)[0] for i in ims])
            x2 = np.asarray(reps[ims[:, None], rr], np.float32)
            pos = torch.as_tensor(np.array([ctx.pool_pos[int(i)] for i in ims]), device=eye.device)
            Ymeas = c.pool_z[pos]
            d1 = c.mS['h1'] - I.acts(c.f0, x2.reshape(-1, *x2.shape[2:]))['h1'].mean(0)
            Xs = X.acts2(c.f0, x2.mean(1), d1)['z']
            agree = (l2n(X.acts2(c.f0, x2[:, 0], d1)['z']) * l2n(X.acts2(c.f0, x2[:, 1], d1)['z'])).sum(1)
            Zq0 = c.zq(d1)
            iims = torch.as_tensor(ims, device=eye.device)

            def rot_to(Y):
                lam, _ = R9.cv_pick(lambda a, b, l: R11.e_rot_prior(a, b, l, eye), Xs, Y, R11.REL, seed=draw)
                return R11.e_rot_prior(Xs, Y, lam, eye)
            f_rot = rot_to(Ymeas)
            bank = R13.recover(Xs, Ymeas, torch.Generator().manual_seed(150000 + 1000 * B + 10 * k + draw), v13, agree)
            f_lf = None
            if len(bank['w']) >= 10:
                Xb, Yb = R13.prep(Xs[bank['eeg']], bank['mm']), Ymeas[bank['img']]
                lc, _ = R9.cv_pick(lambda a, b, l: R11.e_rot_prior(a, b, l, eye), Xb, Yb, R11.REL, seed=draw)
                g_lf = R11.e_rot_prior(Xb, Yb, lc, eye)
                f_lf = (lambda g, mm: (lambda Z: g(R13.prep(Z, mm))))(g_lf, bank['mm'])
            prec = float((bank['eeg'] == bank['img']).float().mean()) if len(bank['w']) else np.nan
            maps = {g: {'zero': None} for g in GALS}
            for g in CAL_GALS:
                with torch.no_grad():
                    Yp = Ymeas if g == 'Gstar' else fns[g]('train', iims)
                maps[g] = {'zero': None, 'rot': f_rot, 'rot_pred': f_rot if g == 'Gstar' else rot_to(Yp), 'lf': f_lf}
            for qk in QK:
                for qi, z0 in enumerate(Zq0[qk]):
                    for g in GALS:
                        for meth, mp in maps[g].items():
                            if meth == 'lf' and mp is None:
                                continue
                            z = z0 if mp is None else mp(z0)
                            for rule in ('cos', 'csls_batch'):
                                S = R6m.smat(z, gals[g], rule)
                                mt, _ = R14.score_rows(S, np.arange(200))
                                rows.append(dict(k=k, draw=draw, gallery=g, method=meth, rule=rule, query=qk, qdraw=qi,
                                                 precision=prec, **mt))
                            if example is None and k == KS[0] and draw == 0 and qk == 's4R4' and qi == 0 \
                                    and g == 'G2' and meth == 'zero':
                                S = R6m.smat(z, gals[g], 'cos')
                                top = S.topk(5, dim=1)
                                rank = (S > S.diagonal()[:, None]).sum(1)
                                names = [GL.concept(p) for p in c.test_paths]
                                example = [dict(query=i, correct=names[i], rank=int(rank[i]) + 1,
                                                top5=[[names[int(j)], round(float(v), 4)]
                                                      for j, v in zip(top.indices[i], top.values[i])])
                                           for i in range(200)]
            print(f'   k={k} draw {draw} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df['target'] = B
    # gate: G2 zero-label cells reproduce Round 14B
    gmsg = 'Round 14B file not found'
    r14 = os.path.join(N.ENV['results_dir'], 'nb_round14', 'b', f'r14b_t{B:02d}.csv')
    if os.path.exists(r14):
        r = pd.read_csv(r14)
        r = r[(r.gallery.isin(['G0', 'G2'])) & (r.method == 'zero')].groupby(['draw', 'gallery', 'rule', 'query']).top1.mean()
        mine = df[(df.k == 500) & df.gallery.isin(['G0', 'G2']) & (df.method == 'zero')].groupby(
            ['draw', 'gallery', 'rule', 'query']).top1.mean()
        j = mine.index.intersection(r.index)
        gmsg = f'max |diff| vs Round 14B (G0, G2; zero) over {len(j)} cells: {float((mine[j] - r[j]).abs().max()):.2e}'
    df.to_csv(out, index=False)
    if example is not None:
        json.dump(dict(target=B, gallery='G2', note='zero labels, s4R4 query draw 0, calibration draw 0 (k = 100 offset)',
                       queries=example), open(os.path.join(odir('eval'), f'example_queries_t{B:02d}.json'), 'w'), indent=1)
    info.update(target=B, gate=gmsg, minutes=round((time.time() - t0) / 60, 1))
    json.dump(info, open(out.replace('.csv', '_meta.json'), 'w'), indent=1, default=str)
    print(f'[15] t{B}: {info["minutes"]} min | gate: {gmsg}', flush=True)


def holm(ps):
    keys = sorted(ps, key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = round(run, 4)
    return out


def report():
    M = _m()
    R4m = M['R4m']
    R4m.CONTRASTS.clear()
    fs = sorted(f for f in glob.glob(os.path.join(odir('eval'), 'r15_t*.csv')))
    if not fs:
        print('Round 15: no results yet')
        return
    df = pd.concat(map(pd.read_csv, fs))
    print('ROUND 15: (A) calibration on the better gallery, (B) a second visual representation (seed 2026, people 2-10)')
    edges_w, edges_s = 0, 0
    for f in fs:
        m = json.load(open(f.replace('.csv', '_meta.json')))
        edges_w += bool(m['w_at_edge'])
        edges_s += bool(m['s_at_edge'])
        print(f'  t{m["target"]}: gate {m["gate"]}; G2 [{m["w_G2"]}, s {m["s_G2"]}] -> G2w [{m["w_G2w"]}, s {m["s_G2w"]}] '
              f'({m["n_evals"]} settings); anchor cos to G* ' + ', '.join(f'{k} {v:.3f}' for k, v in m['anchor_cos_to_Gstar'].items()))
    print(f'  wider grid: a layer weight at the new maximum ({W2[-1]:g}) for {edges_w}/{len(fs)} people, residual scale at '
          f'the new maximum ({S2[-1]:g}) for {edges_s}/{len(fs)}')
    pt = df.groupby(['target', 'k', 'gallery', 'method', 'rule', 'query']).agg(
        top1=('top1', 'mean'), precision=('precision', 'mean')).reset_index()
    te = pt[pt.target.isin(TEST)]
    for rule in ('cos', 'csls_batch'):
        for q in ('s4R4', 's4R1', 's4R20', 'bench'):
            x = te[(te.rule == rule) & (te['query'] == q)]
            print(f'\n{q}, {rule}: top-1 % (rows = gallery x method, columns = calibration images)')
            print(x.pivot_table(index=['gallery', 'method'], columns='k', values='top1').reindex(GALS, level=0).round(2).to_string())
    print('\nlabel-free landmark precision by k:', te.groupby('k').precision.mean().round(3).to_dict())

    def g(gal, meth='zero', k=500, rule='cos', q='s4R4'):
        x = pt[(pt.gallery == gal) & (pt.method == meth) & (pt.k == k) & (pt.rule == rule) & (pt['query'] == q)]
        return x.set_index('target').top1

    def ci(a, b):
        d = (a - b).reindex(TEST).dropna().values
        rng = np.random.RandomState(0)
        bt = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
        return d.mean(), np.percentile(bt, 2.5), np.percentile(bt, 97.5)
    print('\n' + '=' * 100 + '\nA. CALIBRATION ON THE BETTER GALLERY\nPRIMARY A')
    R4m.pstats(g('G2', 'rot_pred'), g('G2'), 'G2: rot_pred - zero, k = 500, s4R4, cos', primary=True)
    m, lo, hi = ci(g('G2', 'rot_pred'), g('G2'))
    verdict = ('continue the calibration branch' if (m >= THRESH_A and lo > 0) else
               'better shared templates reduced what person-specific rotation adds' if hi < THRESH_A else 'unresolved')
    print(f'  DECISION (>= +{THRESH_A:g} with CI above 0): {m:+.2f} (95% CI {lo:+.2f} to {hi:+.2f}) -> {verdict}')
    print('SECONDARY (Holm over four)')
    inter = (g('G2', 'rot_pred') - g('G2')) - (g('G0', 'rot_pred') - g('G0'))
    ps = {'k100': R4m.pstats(g('G2', 'rot_pred', 100), g('G2', k=100), 'G2: rot_pred - zero, k = 100'),
          'interaction': R4m.pstats(inter, inter * 0, 'interaction: (rot_pred - zero on G2) - (on G0), k = 500'),
          'lf': R4m.pstats(g('G2', 'lf'), g('G2'), 'G2: label-free - zero, k = 500'),
          'G2w - G2': R4m.pstats(g('G2w'), g('G2'), 'G2w - G2, zero labels (wider search)')}
    print('  HOLM:', holm(ps))
    print('OTHER')
    for k in KS:
        R4m.pstats(g('G0', 'rot_pred', k), g('G0', k=k), f'G0: rot_pred - zero, k = {k}')
        R4m.pstats(g('G0', 'rot', k), g('G0', k=k), f'G0: rot (to measured) - zero, k = {k} (the poster calibration)')
        R4m.pstats(g('G2', 'rot', k), g('G2', k=k), f'G2: rot (to measured) - zero, k = {k}')
        R4m.pstats(g('G2', 'rot_pred', k), g('G0', 'rot_pred', k), f'rot_pred on G2 - on G0, k = {k} (gallery gain after calibration)')
        R4m.pstats(g('G2w', 'rot_pred', k), g('G2w', k=k), f'G2w: rot_pred - zero, k = {k}')
    R4m.pstats(g('G2', 'lf', 100), g('G2', k=100), 'G2: label-free - zero, k = 100')
    R4m.pstats(g('G0', 'lf'), g('G0'), 'G0: label-free - zero, k = 500')

    print('\n' + '=' * 100 + '\nB. A SECOND VISUAL REPRESENTATION FOR WHAT G2 MISSES (zero labels)')
    metas = [json.load(open(f.replace('.csv', '_meta.json'))) for f in fs]
    for m_ in metas:
        print(f'  t{m_["target"]}: gamma G3d {m_["gamma_G3d"]:g}, G3dp {m_["gamma_G3dp"]:g}, G3c {m_["gamma_G3c"]:g}; '
              f'residual / response norm {m_["resid_norm_ratio"]:.3f}; dims {m_["dims"]}')
    zz = te[(te.method == 'zero') & (te.k == 500) & (te.rule == 'cos')]
    print(zz.pivot_table(index='gallery', columns='query', values='top1').reindex(GALS)[QK].round(2).to_string())
    mg = df[(df.method == 'zero') & (df.k == 500) & (df.rule == 'cos') & (df['query'] == 's4R4') & df.target.isin(TEST)]
    print('margin of the correct template over the strongest competitor (s4R4):',
          mg.groupby('gallery').margin.mean().reindex(GALS).round(4).to_dict())
    g2 = float(g('G2').reindex(TEST).mean())
    print(f'G2 {g2:.2f}; practical threshold +{THRESH_B:g} -> {g2 + THRESH_B:.2f}')
    print('PRIMARY B')
    R4m.pstats(g('G3d'), g('G2'), 'G3d - G2 (DINOv2 residual), zero, s4R4, cos', primary=True)
    print('SECONDARY (Holm over two)')
    ps = {'vs CLIP control': R4m.pstats(g('G3d'), g('G3c'), 'G3d - G3c (new representation vs a second CLIP fit)'),
          'vs pooled': R4m.pstats(g('G3d'), g('G3dp'), 'G3d - G3dp (DINOv2 grids vs pooled only)')}
    print('  HOLM:', holm(ps))
    print('OTHER')
    R4m.pstats(g('G3c'), g('G2'), 'G3c - G2 (second CLIP fit alone)')
    for q in ('s4R1', 's4R20', 'bench'):
        R4m.pstats(g('G3d', q=q), g('G2', q=q), f'G3d - G2, {q}')
    d = float((g('G3d') - g('G2')).reindex(TEST).mean())
    print(f'  DECISION: G3d - G2 {d:+.2f} vs threshold +{THRESH_B:g} -> '
          + ('worth carrying forward' if d >= THRESH_B else 'below threshold: consider the EEG-head retraining test next'))
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['selftest', 'dino', 'run', 'report'])
    p.add_argument('--config')
    p.add_argument('--targets', type=int, nargs='+', default=TEST)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    a = p.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    from eegalign import nb as N
    N.setup(a.config)
    N.ENV.setdefault('reps_dir', a.reps_dir)
    if a.cmd == 'report':
        report()
    elif a.cmd == 'dino':
        dino_feats()
    else:
        for t in a.targets:
            run_target(t)


if __name__ == '__main__':
    main()
