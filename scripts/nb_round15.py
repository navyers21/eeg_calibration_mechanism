"""Round 15: does calibration add on top of the better gallery, and was the Round 14 gallery limited by its grid?
Frozen seed-2026 repaug NeuroBridge decoders; no decoder training.

    python scripts/nb_round15.py selftest
    python scripts/nb_round15.py run    --config C [--targets 2 3 ...]   # people 2-10 by default
    python scripts/nb_round15.py report --config C

Why. Round 14B: a gallery predicted from pooled RN50 layers 2-4 plus a layer-3 4x4 spatial residual (G2) beat the
current predicted gallery (G0) by +3.64 top-1 (9/9) with no target data, about what a 500-image rotation adds on G0.
Two questions follow. (1) Does labelled or label-free calibration still add on top of G2, or do the two gains overlap?
A first look (Round 8's rigid map, k = 500) added only +0.09 on G2 vs +0.82 on G0. (2) The residual scale chose the top
of its grid (1.5) for 8 of 10 people and the layer weights mostly sat at their maximum (1): the grid may have been too
narrow.

Setup = Round 14B: R6 context, Round 8 evaluator, label-free layer-1 offset from the calibration trials, session-4
queries, k images x 2 session-3 viewings (Round 6 draws: 5 at k = 100, 3 at k = 500), cosine (PRIMARY) and batch CSLS.
Galleries
  G0      current predicted gallery (Round 8 'anchor')
  G2      Round 14B's gallery, rebuilt from its saved per-person settings (layer weights, residual scale) on all 9
          source people; GATE: zero-label top-1 must reproduce Round 14B's G2 cells
  G2w     the same family with a WIDER search, fixed before running: layer weights in {0, 0.5, 1, 1.5, 2, 3} by
          coordinate ascent from Round 14's choice (2 sweeps over layers 2, 3, 4), then residual scale s in
          {0.5, 1, 1.5, 2, 3, 4}; chosen on the same 2 held-out source people as Round 14 (no target data)
  Gstar   measured source-population responses to the test images (privileged headroom)
Methods on every gallery (same draws, offset and queries)
  zero      offset only
  rot       supervised rotation to the measured source responses of the calibration images, shrinkage by 5-fold CV
            (Round 11 'rot'; the calibration on the poster)
  rot_pred  the same rotation, target = this gallery's own prediction for the calibration images
  lf        label-free: Round 13 locked recovery (mm0|pop|rho1|q1|s2|ag1) infers pairs from the unlabeled calibration
            set, then the inferred pairs are refit with CV shrinkage (Round 14A's rec_cv)
PRIMARY (fixed before running): on G2, rot - zero, k = 100, s4R4, cos, people 2-10, Wilcoxon.
Secondary (Holm over four): rot - zero on G2 at k = 500; lf - zero on G2 at k = 100; G2w - G2 (zero labels);
  interaction at k = 100: (rot - zero on G2) - (rot - zero on G0).
Also written: example_queries_tXX.json (G2, zero labels, s4R4 draw 0, calibration draw 0): for every one of the 200
queries, the correct concept, its rank and the top-5 concepts with cosine scores, for a real worked example on the
poster.
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
GALS = ['G0', 'G2', 'G2w', 'Gstar']
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
    fa14 = R14.fit_gallery(Fe, c.pool, [Yall], w14, labels, spatial=('real',))[0]
    fa2 = fa14 if R14.wkey(w2) == R14.wkey(w14) else R14.fit_gallery(Fe, c.pool, [Yall], w2, labels, spatial=('real',))[0]
    fns = {'G0': lambda split, idx: c.galfn['anchor']((c.ftr if split == 'train' else c.fte)[idx]),
           'G2': R14.gallery_fn(Fe, fa14, w14, 'real', s14), 'G2w': R14.gallery_fn(Fe, fa2, w2, 'real', s2)}
    gals = {'G0': c.gal['anchor'][0], 'Gstar': c.gal['oracle'][0]}
    with torch.no_grad():
        gals['G2'], gals['G2w'] = fns['G2']('test'), fns['G2w']('test')
    info = dict(w_G2=R14.wkey(w14), s_G2=s14, w_G2w=R14.wkey(w2), s_G2w=s2, val_G1w=seen[R14.wkey(w2)],
                val_G2w={f'{k:g}': v for k, v in sv.items()}, n_evals=len(seen),
                w_at_edge=[k for k, v in w2.items() if v == W2[-1]], s_at_edge=s2 == S2[-1],
                anchor_cos_to_Gstar={k: float((gals[k] * gals['Gstar']).sum(1).mean()) for k in gals},
                build_s=round(time.time() - t0))
    print(f'   galleries ({info["build_s"]}s): G2 [{info["w_G2"]}, s {s14:g}]; G2w [{info["w_G2w"]}, s {s2:g}], '
          f'val G1w {info["val_G1w"]:.2f}, G2w {sv[s2]:.2f} ({len(seen)} weight settings tried)', flush=True)
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
            maps = {}
            for g in GALS:
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
    print('ROUND 15: calibration on top of the better gallery; wider gallery search (seed 2026, people 2-10)')
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

    def g(gal, meth, k=100, rule='cos', q='s4R4'):
        x = pt[(pt.gallery == gal) & (pt.method == meth) & (pt.k == k) & (pt.rule == rule) & (pt['query'] == q)]
        return x.set_index('target').top1
    print('\nPRIMARY')
    R4m.pstats(g('G2', 'rot'), g('G2', 'zero'), 'G2: rot - zero, k = 100, s4R4, cos', primary=True)
    print('SECONDARY (Holm over four)')
    inter = (g('G2', 'rot') - g('G2', 'zero')) - (g('G0', 'rot') - g('G0', 'zero'))
    ps = {'rot k500': R4m.pstats(g('G2', 'rot', 500), g('G2', 'zero', 500), 'G2: rot - zero, k = 500'),
          'lf k100': R4m.pstats(g('G2', 'lf'), g('G2', 'zero'), 'G2: label-free - zero, k = 100'),
          'G2w - G2': R4m.pstats(g('G2w', 'zero'), g('G2', 'zero'), 'G2w - G2, zero labels (wider search)'),
          'interaction': R4m.pstats(inter, inter * 0, 'interaction: calibration gain on G2 minus on G0, k = 100')}
    print('  HOLM:', holm(ps))
    print('OTHER')
    for k in KS:
        R4m.pstats(g('G0', 'rot', k), g('G0', 'zero', k), f'G0: rot - zero, k = {k} (reference: the poster calibration curve)')
        R4m.pstats(g('G2', 'rot', k), g('G0', 'rot', k), f'rot on G2 - rot on G0, k = {k} (gallery gain after calibration)')
        R4m.pstats(g('G2', 'rot_pred', k), g('G2', 'rot', k), f'G2: rot to own predictions - rot to measured, k = {k}')
        R4m.pstats(g('G2w', 'rot', k), g('G2w', 'zero', k), f'G2w: rot - zero, k = {k}')
    R4m.pstats(g('G2', 'lf', 500), g('G2', 'zero', 500), 'G2: label-free - zero, k = 500')
    R4m.pstats(g('G0', 'lf'), g('G0', 'zero'), 'G0: label-free - zero, k = 100')
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['selftest', 'run', 'report'])
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
    else:
        for t in a.targets:
            run_target(t)


if __name__ == '__main__':
    main()
