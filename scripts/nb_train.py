"""Train NeuroBridge decoders (official recipe, validation-based model selection) and save everything on Drive.

    python scripts/nb_train.py train  --config C --kind within --montages posterior17 occipital8 [--subjects ..]
    python scripts/nb_train.py train  --config C --kind loso   --montages posterior17 occipital8 [--subjects ..]
    python scripts/nb_train.py train  --config C --kind paper  --montages posterior17            # replication only
    python scripts/nb_train.py report --config C

Kinds (all: EEGProject + linear projectors, recipe in src/eegalign/nb.py, seed 2025 unless --seeds):
  within   one person's own training images minus the 200 held-out concepts (the calibration pool). Epoch picked
           by top-1 on that person's held-out concepts (10 draws x 200-way, averaged). The ceiling for calibration.
  loso     f0: the other 9 people's pool images (target never used). Epoch picked on the 9 people's held-out
           concepts. --patience stops after that many epochs without a new best (0 = run all epochs).
  paper    NeuroBridge's own protocol, for a replication check only: all 1654 training concepts, epoch with the
           lowest TEST loss (selection on the test set, as in the official train.py) and the last epoch.
Every epoch logs train loss, validation top-1 and test top-1/top-5/loss (test is never used for selection, except
by the 'paper' kind). Saved per model: <kind>_<montage>_tNN_s<seed>_{best,last}.pt (NeuroBridge checkpoint keys,
+ cfg + history), one row in results/nb_models.csv, per-epoch rows in results/nb_curves.csv. Resumable per model.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
from eegalign import nb as N  # noqa: E402

PAPER = {'posterior17': dict(top1=63.2, top5=89.9)}           # NeuroBridge, intra-subject, mean of subjects 1-10
PAPER_LOSO = dict(top1=19.0, top5=45.9)                       # inter-subject (TSConv, 63 channels)
NATHAN = {'posterior17': dict(within=62.4, loso=14.2)}        # our earlier NeuroBridge numbers (select=last)


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(kind, montage, target, subjects, seed):
    """Training arrays (smoothed once, as NeuroBridge's frozen EEG prior), validation sets, test set."""
    ftr_all = N.features('train', 'aug')
    train_x, train_img, val = [], [], []
    people = [target] if kind in ('within', 'paper') else [s for s in subjects if s != target]
    for s in people:
        d = N.subject_data(s, montage, keys=('train', 'train_labels'))
        pool, draws = N.split(d['train_labels'])
        idx = np.arange(len(d['train_labels'])) if kind == 'paper' else pool
        train_x.append(N.random_smooth(d['train'][idx], seed=seed * 100 + s).astype(np.float16))
        train_img.append(idx)
        if kind != 'paper':
            val += [(d['train'][ix], ix) for ix in draws]            # held-out concepts: never trained on
        del d
    t = N.subject_data(target, montage, keys=('test', 'test_trials'))
    return dict(x=np.concatenate(train_x), img=np.concatenate(train_img), val=val, test=t['test'],
                test_trials=t['test_trials'], ftr=ftr_all)


@torch.no_grad()
def val_top1(m, val, ftr):
    return float(np.mean([N.retrieval(m, x, ftr[ix])[0] for x, ix in val])) if val else float('nan')


def train_one(kind, montage, target, seed, subjects, epochs, patience, lr=1e-4, wd=1e-4, batch=1024):
    t0 = time.time()
    dev = N.ENV['device']
    seed_all(seed)
    D = build(kind, montage, target, subjects, seed)
    fte, fte_clean = N.features('test', 'aug'), N.features('test', 'clean')
    C = D['x'].shape[1]
    X = torch.from_numpy(D['x']).to(dev)                              # fp16 on the GPU, cast per batch
    IMG = torch.from_numpy(D['img']).to(dev)
    FTR = D['ftr'].to(dev)
    seed_all(seed)
    m = N.NBModel(C).to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    cfg = dict(kind=kind, montage=montage, target=target, seed=seed, subjects=subjects, epochs=epochs, lr=lr, wd=wd,
               batch=batch, n_train=len(X), channels=C, recipe='neurobridge-intra@4f59fbb', select='val' if
               kind != 'paper' else 'test_loss')
    hist, best, best_ep, wait = [], -float('inf'), 0, 0
    best_tl, best_tl_ep = float('inf'), 0
    paths = {w: N.ckpt_path(kind, montage, target, seed, w) for w in ('best', 'last')}
    n = len(X)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n, device=dev)
        tot, nb = 0.0, 0
        for i in range(0, n - batch + 1, batch):                    # drop_last, as NeuroBridge's DataLoader
            idx = perm[i:i + batch]
            z = m(X[idx])
            g = m.gallery(FTR[IMG[idx]])
            loss = N.nb_loss(z, g, m.scale())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        rec = dict(kind=kind, montage=montage, target=target, seed=seed, epoch=ep, train_loss=tot / max(nb, 1))
        rec['val_top1'] = val_top1(m, D['val'], FTR)
        rec['test_top1'], rec['test_top5'] = N.retrieval(m, D['test'], fte)
        rec['test_loss'] = N.test_loss(m, D['test'], fte)
        hist.append(rec)
        sel = -rec['test_loss'] if kind == 'paper' else rec['val_top1']
        if sel > best:
            best, best_ep, wait = sel, ep, 0
            N.save_model(m, paths['best'], cfg, hist, ep)
        else:
            wait += 1
        if rec['test_loss'] < best_tl:
            best_tl, best_tl_ep = rec['test_loss'], ep
        print(f'    ep {ep:3d} loss {rec["train_loss"]:.3f} | val {rec["val_top1"]:5.1f} | test {rec["test_top1"]:5.1f} '
              f'/ {rec["test_top5"]:5.1f} (loss {rec["test_loss"]:.3f}){" *" if best_ep == ep else ""}', flush=True)
        if patience and wait >= patience:
            print(f'    stopping: no better validation top-1 for {patience} epochs', flush=True)
            break
    N.save_model(m, paths['last'], cfg, hist, ep)
    # final numbers: picked epoch (validation; test loss for 'paper') and last epoch
    row = dict(kind=kind, montage=montage, target=target, seed=seed, epochs_run=ep, n_train=n,
               select=cfg['select'], best_epoch=best_ep, test_loss_best_epoch=best_tl_ep)
    for which in ('best', 'last'):
        mm, _ = N.load_model(paths[which], C)
        t1, t5 = N.retrieval(mm, D['test'], fte)
        tc1, _ = N.retrieval(mm, D['test'], fte_clean)
        tt = D['test_trials']
        st1, st5 = N.retrieval(mm, tt.reshape(-1, *tt.shape[2:]), fte, np.repeat(np.arange(len(tt)), tt.shape[1]))
        row.update({f'test_top1_{which}': t1, f'test_top5_{which}': t5, f'test_clean_top1_{which}': tc1,
                    f'single_trial_top1_{which}': st1, f'single_trial_top5_{which}': st5,
                    f'val_top1_{which}': hist[(best_ep if which == 'best' else ep) - 1]['val_top1']})
    row['test_top1_max_over_epochs'] = max(h['test_top1'] for h in hist)        # reference only (selects on test)
    if kind == 'paper':                     # replication check only: keep one checkpoint (Drive space)
        os.remove(paths['last'])
    row['minutes'] = round((time.time() - t0) / 60, 1)
    out = N.ENV['results_dir']
    pd.DataFrame(hist).to_csv(os.path.join(out, 'nb_curves.csv'), mode='a', index=False,
                              header=not os.path.exists(os.path.join(out, 'nb_curves.csv')))
    pd.DataFrame([row]).to_csv(os.path.join(out, 'nb_models.csv'), mode='a', index=False,
                               header=not os.path.exists(os.path.join(out, 'nb_models.csv')))
    print(f'  => {kind} {montage} t{target} s{seed}: test top-1 {row["test_top1_best"]:.1f} at epoch {best_ep} '
          f'(last epoch {row["test_top1_last"]:.1f}; single-trial {row["single_trial_top1_best"]:.1f}) '
          f'[{row["minutes"]} min]', flush=True)
    del X, IMG, D
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return row


def done_models():
    p = os.path.join(N.ENV['results_dir'], 'nb_models.csv')
    if not os.path.exists(p):
        return set()
    d = pd.read_csv(p)
    return set(zip(d.kind, d.montage, d.target, d.seed))


def report():
    p = os.path.join(N.ENV['results_dir'], 'nb_models.csv')
    if not os.path.exists(p):
        raise SystemExit('no models yet')
    d = pd.read_csv(p).drop_duplicates(['kind', 'montage', 'target', 'seed'], keep='last')
    pd.set_option('display.width', 200)
    for mon, g in d.groupby('montage'):
        print(f'\n{"=" * 90}\n{mon}\n{"=" * 90}')
        for kind, k in g.groupby('kind'):
            cols = ['target', 'seed', 'best_epoch', 'epochs_run', 'val_top1_best', 'test_top1_best', 'test_top5_best',
                    'test_top1_last', 'single_trial_top1_best', 'test_clean_top1_best']
            if kind == 'paper':
                cols = ['target', 'seed', 'best_epoch', 'test_top1_best', 'test_top5_best', 'test_top1_last',
                        'test_top5_last']
            print(f'\n{kind} ({k.target.nunique()} people)')
            print(k[cols].sort_values('target').round(2).to_string(index=False))
            mean = k[[c for c in cols if c.startswith(('test', 'single', 'val'))]].mean().round(2)
            print('   mean: ' + ', '.join(f'{c} {v}' for c, v in mean.items()))
        if mon in PAPER and 'paper' in set(g.kind):
            pk = g[g.kind == 'paper']
            print(f'\nreplication check: our "paper" protocol {pk.test_top1_best.mean():.1f} / {pk.test_top5_best.mean():.1f} '
                  f'vs NeuroBridge paper {PAPER[mon]["top1"]} / {PAPER[mon]["top5"]} (top-1 / top-5, mean of '
                  f'{pk.target.nunique()} subjects; the paper also averages 5 runs)')
        w, l = g[g.kind == 'within'], g[g.kind == 'loso']
        if len(w) and len(l):
            j = w.merge(l, on=['target', 'seed'], suffixes=('_w', '_f0'))
            print(f'\nwithin vs f0, same {len(j)} people (validation-picked epochs): within {j.test_top1_best_w.mean():.1f}, '
                  f'f0 {j.test_top1_best_f0.mean():.1f}, gap {(j.test_top1_best_w - j.test_top1_best_f0).mean():.1f} '
                  f'points; with last epochs: within {j.test_top1_last_w.mean():.1f}, f0 {j.test_top1_last_f0.mean():.1f}')
            if mon in NATHAN:
                print(f'   earlier NeuroBridge numbers (last epoch): within {NATHAN[mon]["within"]}, f0 {NATHAN[mon]["loso"]}')
        if len(l):
            print(f'   paper inter-subject (TSConv, 63 channels, test-picked): {PAPER_LOSO["top1"]} / {PAPER_LOSO["top5"]}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--kind', choices=['within', 'loso', 'paper'], default='within')
    p.add_argument('--montages', nargs='+', default=['posterior17', 'occipital8'])
    p.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--seeds', type=int, nargs='+', default=[2025])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=0)
    a = p.parse_args()
    N.setup(a.config)
    if a.cmd == 'train':
        dev = N.ENV['device']
        print(f'device: {dev}' + (f' ({torch.cuda.get_device_name(0)})' if dev == 'cuda' else
                                  '  WARNING: no GPU, training will be ~10x slower (Runtime > Change runtime type)'),
              flush=True)
    if a.cmd == 'report':
        return report()
    pool = N.ENV.get('subjects', list(range(1, 11)))
    done = done_models()
    for mon in a.montages:
        for seed in a.seeds:
            for t in a.subjects:
                if (a.kind, mon, t, seed) in done:
                    print(f'[{a.kind}] {mon} t{t} s{seed}: done, skipping')
                    continue
                print(f'[{a.kind}] {mon} t{t} s{seed}: training', flush=True)
                train_one(a.kind, mon, t, seed, pool, a.epochs, a.patience)


if __name__ == '__main__':
    main()
