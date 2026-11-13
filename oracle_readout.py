"""Is the lost accuracy recoverable from f0's representation? Linear readouts fitted on the target's own data.

    python oracle_readout.py --config C --targets 2 3 ... --montages posterior17 occipital8

For each target: features = f0's output (1024) or f0's first-layer activations (1024) on the target's calibration
pool (4-rep averages); a ridge readout to the blurred CLIP targets is fitted on n images (n = 50, 200, 1000, all;
alpha by efficient leave-one-out), then scored on the official test set (200-way top-1, %).
  plain     t_hat = ridge(features)
  residual  t_hat = s * z + ridge(t - s * z)  (output only; starts from f0's own prediction, s = norm ratio)
v2 (round 1 of this script was invalid): an MSE ridge readout predicts targets shrunk toward the MEAN image feature,
and cosine retrieval against uncentered CLIP features is then dominated by that shared mean (every prediction points
the same way), so plain ridge scored below f0 itself. v2 adds two valid readouts:
  centered     ridge on mean-centered targets; predictions and gallery both centered by the training target mean
  contrastive  z -> z + A (z - mu), A (1024 x 1024) from 0, trained with f0's own CLIP loss on the n images
               (precomputed features, 1,024 pool negatives), early-stopped on the held-out concepts
If the readout from f0's OUTPUT with all images approaches the within-subject decoder, the information is present
and only misaligned (an output-side fix can work); if it does not, it is missing and must be recovered at the input.
Rows append to <drive>/oracle_readout2.csv (resumable per target x montage).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from eegalign import env  # noqa: E402
from eegalign import ubp as U  # noqa: E402
from eegalign.data import Target  # noqa: E402
from eegalign.source import ensure_within  # noqa: E402

BUDGETS = [50, 200, 1000, 0]      # 0 = all


@torch.no_grad()
def feats(model, X, layer, bs=512):
    dev = U.ENV['device']
    out, store = [], []
    h = model.model[0].register_forward_hook(lambda m, i, o: store.append(o.detach().float().cpu())) \
        if layer == 'layer1' else None
    for i in range(0, len(X), bs):
        z = model(torch.as_tensor(X[i:i + bs]).to(dev).float()).float().cpu()
        out.append(store.pop() if h else z)
    if h:
        h.remove()
    return torch.cat(out).flatten(1).numpy()


def top1(pred, G):
    p = F.normalize(torch.as_tensor(pred).float(), dim=-1)
    g = F.normalize(torch.as_tensor(G).float(), dim=-1)
    return 100 * float(((p @ g.T).argmax(1) == torch.arange(len(g))).float().mean())


def ridge(Xtr, Ytr, Xte):
    from sklearn.linear_model import RidgeCV
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    m = RidgeCV(alphas=np.logspace(-1, 6, 15)).fit((Xtr - mu) / sd, Ytr)
    return m.predict((Xte - mu) / sd), float(m.alpha_)


def ridge_centered(Xtr, Ytr, Xte, Gte):
    ym = Ytr.mean(0)
    pred, alpha = ridge(Xtr, Ytr - ym, Xte)
    return top1(pred, np.asarray(Gte) - ym), alpha


def contrastive(Ztr, Ytr, Zte, Gte, Zval, Gval, scale, steps=1500, lr=1e-3, seed=0):
    """Full-rank residual linear readout trained with the CLIP loss; best step by held-out top-1."""
    dev = U.ENV['device']
    torch.manual_seed(seed)
    Ztr, Ytr = torch.as_tensor(Ztr).float().to(dev), torch.as_tensor(Ytr).float().to(dev)
    mu = Ztr.mean(0)
    A = torch.zeros(Ztr.shape[1], Ztr.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.AdamW([A], lr=lr, weight_decay=1e-4)
    def f(z):
        z = torch.as_tensor(z).float().to(dev)
        return z + (z - mu) @ A.T
    best, best_A = -1, A.detach().clone()
    gal_pool = F.normalize(Ytr, dim=-1)
    for step in range(steps):
        i = torch.randint(len(Ztr), (min(256, len(Ztr)),), device=dev)
        j = torch.randint(len(Ztr), (1024,), device=dev)
        gal = torch.cat([F.normalize(Ytr[i], dim=-1), gal_pool[j]])
        loss = F.cross_entropy(scale * f(Ztr[i]) @ gal.T, torch.arange(len(i), device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 99 or step == steps - 1:
            with torch.no_grad():
                v = np.mean([top1(f(z).cpu(), g) for z, g in zip(Zval, Gval)])
            if v > best:
                best, best_A = v, A.detach().clone()
    with torch.no_grad():
        A.copy_(best_A)
        return top1(f(Zte).cpu(), Gte)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument('--montages', nargs='+', default=['posterior17', 'occipital8'])
    a = p.parse_args()
    cfg = env.load(a.config)
    out = os.path.join(os.path.dirname(cfg['results_csv']), 'oracle_readout2.csv')
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out)
        done = set(zip(d.target, d.montage))
    for mon in a.montages:
        for t in a.targets:
            if (t, mon) in done:
                print(f'[oracle] t{t} {mon}: done, skipping', flush=True)
                continue
            T = Target(t, mon)
            Y = T.tgt.cpu().numpy()
            G = T.test_g
            within = ensure_within(t, mon, train=False)
            rows = [dict(target=t, montage=mon, features='within_decoder', readout='none', n=0,
                         test_top1=top1(feats(within, T.test_x, 'out'), G)),
                    dict(target=t, montage=mon, features='f0', readout='none', n=0,
                         test_top1=top1(feats(T.f0, T.test_x, 'out'), G))]
            order = np.random.RandomState(0).permutation(T.n)
            scale = float(T.f0.softplus(T.f0.logit_scale))
            Fval = [feats(T.f0, x, 'out') for x in T.held_x]
            Gval = T.held_g
            for layer in ('out', 'layer1'):
                Ftr, Fte = feats(T.f0, T.x, layer), feats(T.f0, T.test_x, layer)
                for n in BUDGETS:
                    idx = order[:n] if n else order
                    pred, alpha = ridge(Ftr[idx], Y[idx], Fte)
                    rows.append(dict(target=t, montage=mon, features=f'f0_{layer}', readout='plain', n=n or T.n,
                                     alpha=alpha, test_top1=top1(pred, G)))
                    acc, alpha = ridge_centered(Ftr[idx], Y[idx], Fte, G.numpy())
                    rows.append(dict(target=t, montage=mon, features=f'f0_{layer}', readout='centered', n=n or T.n,
                                     alpha=alpha, test_top1=acc))
                    if layer == 'out':
                        rows.append(dict(target=t, montage=mon, features='f0_out', readout='contrastive',
                                         n=n or T.n, test_top1=contrastive(Ftr[idx], Y[idx], Fte, G, Fval, Gval,
                                                                           scale)))
                    if layer == 'out':
                        s = np.linalg.norm(Y[idx], axis=1).mean() / np.linalg.norm(Ftr[idx], axis=1).mean()
                        res, alpha = ridge(Ftr[idx], Y[idx] - s * Ftr[idx], Fte)
                        rows.append(dict(target=t, montage=mon, features='f0_out', readout='residual', n=n or T.n,
                                         alpha=alpha, test_top1=top1(s * Fte + res, G)))
            df = pd.DataFrame(rows)
            print(df.to_string(index=False), flush=True)
            df.to_csv(out, mode='a', header=not os.path.exists(out), index=False)


if __name__ == '__main__':
    main()
