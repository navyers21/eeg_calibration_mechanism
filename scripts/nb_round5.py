"""Round 5: retrain the cross-subject decoder for the queries it actually gets.

    python scripts/nb_round5.py train     --config C --kinds repaug smooth4 --montages posterior17 occipital8
    python scripts/nb_round5.py trainbase --config C --montages crown_po2          # standard recipe, loso + within
    python scripts/nb_round5.py eval      --config C --kinds loso repaug smooth4 --montages posterior17 occipital8
    python scripts/nb_round5.py report    --config C

Why. f0 is trained on 4-presentation averages and every method loses most of its accuracy going from 80- to
4-presentation queries (anchors: 15.1 -> 7.0). The source people's single presentations are cached
(nb_round3 prep), so the decoder can be trained on what deployment looks like.

Kinds (LOSO: the 9 other people's pool images; NeuroBridge recipe otherwise: AdamW 1e-4, batch 1024, 50 epochs,
last epoch used, as everywhere since round 3):
  repaug   each sample is the average of k random presentations of that image, k in {1, 2, 4} uniformly, redrawn
           every time it is seen; random temporal smoothing (kernel 5, p 0.3) applied on the fly
  smooth4  control: always all 4 presentations, same on-the-fly smoothing (isolates the k part from the smoothing
           and seed change)
  loso     the existing f0 (round 3/4 baseline), evaluated with the same code
Crown: the raw-EEG test found PO3/PO4 alone carry 39% identification vs 5.3% for all 8 Crown sites, so crown_po2
models (loso + within, standard recipe; then repaug) test whether a Crown decoder should drop the anterior sites.

Eval (per kind x montage x target): the round-4 context rebuilt with that decoder (source anchors recomputed in its
own output space, same audited session-4 queries), galleries {image, anchor (pool ridge), anchor_held,
anchor_pool+held, oracle} x {cosine, batch CSLS} x offset {none, old_100}, queries s4R1 / s4R4 / s4R20 / bench.

Primary (posterior-17, people 2-10, Wilcoxon): repaug minus loso, anchor gallery, cosine, no offset, s4R4.
Secondary: repaug - smooth4 (same contrast); repaug - loso for the image gallery (the plain decoder); best
zero-label pipeline (anchor + old_100 + batch CSLS) repaug - loso; bench (no loss at 80 presentations); occipital-8
replication; crown_po2 vs crown8.
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
import nb_interp as I  # noqa: E402
import nb_round3 as R  # noqa: E402
import nb_round4 as R4m  # noqa: E402

PEOPLE = list(range(1, 11))
TEST = list(range(2, 11))
SEED = 2025
QK = ['s4R1', 's4R4', 's4R20', 'bench']


def dev():
    return N.ENV['device']


def odir():
    d = os.path.join(N.ENV['results_dir'], 'nb_round5')
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------------------------
# training on single presentations
# ---------------------------------------------------------------------------------------------------------------
def train_rep(kind, mon, B, epochs=50, batch=1024, lr=1e-4, wd=1e-4):
    path = N.ckpt_path(kind, mon, B, SEED, 'last')
    if os.path.exists(path):
        print(f'  [{kind}] {mon} t{B}: done')
        return
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Xs, IM, val = [], [], []
    for s in PEOPLE:
        if s == B:
            continue
        d = N.subject_data(s, mon, keys=('train', 'train_labels'))
        pool, draws = N.split(d['train_labels'])
        Xs.append(torch.from_numpy(np.asarray(R.load_reps(s, mon)[pool], np.float16)))
        IM.append(torch.from_numpy(pool))
        val += [(d['train'][ix], ix) for ix in draws[:3]]
        del d
    X = torch.cat(Xs).to(dev())                                         # (n, 4, C, T) fp16
    IMG = torch.cat(IM).to(dev())
    del Xs
    n, _, C, T = X.shape
    ftr = N.features('train', 'aug').to(dev())
    fte = N.features('test', 'aug')
    m = N.NBModel(C).to(dev())
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    kvals = torch.tensor([1., 2., 4.], device=dev())
    hist = []
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n, device=dev())
        tot, nb = 0.0, 0
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = X[idx].float()
            b = len(idx)
            k = kvals[torch.randint(0, 3, (b,), device=dev())] if kind == 'repaug' else torch.full((b,), 4., device=dev())
            rank = torch.rand(b, 4, device=dev()).argsort(1).argsort(1).float()
            mask = (rank < k[:, None]).float()
            xb = (xb * mask[..., None, None]).sum(1) / k[:, None, None]
            sm = F.avg_pool1d(xb.reshape(-1, 1, T), 5, 1, 2, count_include_pad=False).reshape(b, C, T)
            xb = torch.where(torch.rand(b, C, 1, device=dev()) < 0.3, sm, xb)
            loss = N.nb_loss(m(xb), m.gallery(ftr[IMG[idx]]), m.scale())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        rec = dict(kind=kind, montage=mon, target=B, epoch=ep, train_loss=tot / max(nb, 1))
        if ep % 10 == 0 or ep == epochs:
            m.eval()
            with torch.no_grad():
                rec['val_top1'] = float(np.mean([N.retrieval(m, x, ftr[ix])[0] for x, ix in val]))
            print(f'    ep {ep:3d} loss {rec["train_loss"]:.3f} | val {rec["val_top1"]:5.1f}', flush=True)
        hist.append(rec)
    m.eval()
    cfg = dict(kind=kind, montage=mon, target=B, seed=SEED, epochs=epochs, lr=lr, wd=wd, batch=batch, n_train=n,
               channels=C, recipe='neurobridge + single-presentation sampling (round 5)')
    N.save_model(m, path, cfg, hist, epochs)
    N.save_model(m, N.ckpt_path(kind, mon, B, SEED, 'best'), cfg, hist, epochs)
    d = N.subject_data(B, mon, keys=('test',))
    t1 = N.retrieval(m, d['test'], fte)[0]
    print(f'  => [{kind}] {mon} t{B}: 80-rep test top-1 {t1:.1f} [{(time.time() - t0) / 60:.1f} min]', flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(odir(), 'train_curves.csv'), mode='a', index=False,
                              header=not os.path.exists(os.path.join(odir(), 'train_curves.csv')))
    del X, IMG
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------------------------------------------
# evaluation with the round-4 context
# ---------------------------------------------------------------------------------------------------------------
def use_kind(kind):
    def lm(B, mon, which):
        f0, _ = N.load_model(N.ckpt_path(kind, mon, B, SEED, which))
        wp = N.ckpt_path('within', mon, B, SEED, which)
        w = N.load_model(wp)[0] if os.path.exists(wp) else f0
        return f0.to(dev()).requires_grad_(False), w.to(dev()).requires_grad_(False)
    I.load_models = lm


def eval_one(kind, mon, B, which='last'):
    out = os.path.join(odir(), f'e_{kind}_t{B:02d}_{mon}.csv')
    if os.path.exists(out):
        print(f'  [eval {kind}] {mon} t{B}: done')
        return
    if not os.path.exists(N.ckpt_path(kind, mon, B, SEED, which)):
        print(f'  [eval {kind}] {mon} t{B}: no checkpoint, skipping')
        return
    t0 = time.time()
    use_kind(kind)
    c = R4m.R4(B, mon, which)
    df = pd.DataFrame(R4m.exp_m2(c))
    df['kind'], df['target'], df['montage'] = kind, B, mon
    df.to_csv(out, index=False)
    print(f'  [eval {kind}] {mon} t{B}: {(time.time() - t0) / 60:.1f} min', flush=True)


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def report():
    pd.set_option('display.width', 250)
    f = sorted(glob.glob(os.path.join(odir(), 'e_*.csv')))
    if not f:
        raise SystemExit('nothing evaluated yet')
    e = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
    e = e[e.exp == 'M2']
    per = e.groupby(['montage', 'kind', 'offset', 'rule', 'gallery', 'query', 'target']).top1.mean()
    T = per[per.index.get_level_values('target').isin(TEST)].groupby(level=[0, 1, 2, 3, 4, 5]).mean().unstack('query')
    for mon in e.montage.unique():
        print(f'\n{"=" * 110}\n{mon}: top-1 % (people 2-10)\n{"=" * 110}')
        t = T.loc[mon]
        print(t[[q for q in QK if q in t]].round(2).to_string())

        def g(kind, q='s4R4', gal='anchor', rule='cos', off='none'):
            try:
                return per.loc[(mon, kind, off, rule, gal, q)]
            except KeyError:
                return None
        kinds = set(e[e.montage == mon].kind)
        print()
        for new, base in (('repaug', 'loso'), ('repaug', 'smooth4'), ('smooth4', 'loso')):
            if {new, base} <= kinds:
                prim = (mon == 'posterior17' and new == 'repaug' and base == 'loso')
                R4m.pstats(g(new), g(base), f'{new} - {base}: anchor, cos, no offset, s4R4', primary=prim)
                R4m.pstats(g(new, 's4R1'), g(base, 's4R1'), f'{new} - {base}: anchor, s4R1')
                R4m.pstats(g(new, 's4R20'), g(base, 's4R20'), f'{new} - {base}: anchor, s4R20')
                R4m.pstats(g(new, 'bench'), g(base, 'bench'), f'{new} - {base}: anchor, 80-rep benchmark')
                R4m.pstats(g(new, gal='img'), g(base, gal='img'), f'{new} - {base}: plain decoder (image gallery), s4R4')
                R4m.pstats(g(new, 'bench', gal='img'), g(base, 'bench', gal='img'), f'{new} - {base}: plain decoder, benchmark')
                R4m.pstats(g(new, rule='csls_batch', off='old_100'), g(base, rule='csls_batch', off='old_100'),
                           f'{new} - {base}: best pipeline (anchor + old_100 + batch CSLS), s4R4')
                R4m.pstats(g(new, 's4R20', rule='csls_batch', off='old_100'), g(base, 's4R20', rule='csls_batch', off='old_100'),
                           f'{new} - {base}: best pipeline, s4R20')
        for k in kinds:
            if g(k, gal='anchor_pool+held') is not None:
                R4m.pstats(g(k, gal='anchor_pool+held'), g(k), f'[{k}] anchor_pool+held - anchor (s4R4)')
                R4m.pstats(g(k, gal='anchor_held'), g(k), f'[{k}] anchor_held - anchor (s4R4)')
    # Crown: 2 posterior sites vs all 8
    for k in ('loso', 'repaug'):
        a = per.loc[('crown_po2', k, 'none', 'cos', 'anchor', 's4R4')] if ('crown_po2', k, 'none', 'cos', 'anchor', 's4R4') in per.index.droplevel('target') else None
        b = per.loc[('crown8', k, 'none', 'cos', 'anchor', 's4R4')] if ('crown8', k, 'none', 'cos', 'anchor', 's4R4') in per.index.droplevel('target') else None
        if a is not None and b is not None:
            R4m.pstats(a, b, f'crown_po2 - crown8 ({k}, anchor, s4R4)')
    pd.DataFrame(R4m.CONTRASTS).to_csv(os.path.join(odir(), 'contrasts_per_person.csv'), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'trainbase', 'eval', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--kinds', nargs='+', default=['repaug', 'smooth4'])
    p.add_argument('--montages', nargs='+', default=['posterior17'])
    p.add_argument('--targets', type=int, nargs='+', default=PEOPLE)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--reps_dir', default='/content/nb_reps')
    a = p.parse_args()
    N.setup(a.config)
    N.ENV['reps_dir'] = a.reps_dir
    print(f'device: {dev()}' + (' WARNING: no GPU' if dev() != 'cuda' else ''), flush=True)
    if a.cmd == 'report':
        return report()
    for mon in a.montages:
        for kind in a.kinds:
            for B in a.targets:
                if a.cmd == 'train':
                    print(f'[train {kind}] {mon} t{B}', flush=True)
                    train_rep(kind, mon, B, a.epochs)
                elif a.cmd == 'trainbase':
                    import nb_train as TR
                    for k in ('loso', 'within'):
                        if os.path.exists(N.ckpt_path(k, mon, B, SEED, 'last')):
                            print(f'  [{k}] {mon} t{B}: done')
                            continue
                        TR.train_one(k, mon, B, SEED, PEOPLE, a.epochs, 0)
                else:
                    eval_one(kind, mon, B)


if __name__ == '__main__':
    main()
