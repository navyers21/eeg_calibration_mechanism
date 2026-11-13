"""Is the lost accuracy recoverable from f0's representation? Linear readouts fitted on the target's own data.

    python oracle_readout.py --config C --targets 2 3 ... --montages posterior17 occipital8

For each target: features = f0's output (1024) or f0's first-layer activations (1024) on the target's calibration
pool (4-rep averages); a ridge readout to the blurred CLIP targets is fitted on n images (n = 50, 200, 1000, all;
alpha by efficient leave-one-out), then scored on the official test set (200-way top-1, %).
  plain     t_hat = ridge(features)
  residual  t_hat = s * z + ridge(t - s * z)  (output only; starts from f0's own prediction, s = norm ratio)
If the readout from f0's OUTPUT with all images approaches the within-subject decoder, the information is present
and only misaligned (an output-side fix can work); if it does not, it is missing and must be recovered at the input.
Rows append to <drive>/oracle_readout.csv (resumable per target x montage).
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--targets', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument('--montages', nargs='+', default=['posterior17', 'occipital8'])
    a = p.parse_args()
    cfg = env.load(a.config)
    out = os.path.join(os.path.dirname(cfg['results_csv']), 'oracle_readout.csv')
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
            for layer in ('out', 'layer1'):
                Ftr, Fte = feats(T.f0, T.x, layer), feats(T.f0, T.test_x, layer)
                for n in BUDGETS:
                    idx = order[:n] if n else order
                    pred, alpha = ridge(Ftr[idx], Y[idx], Fte)
                    rows.append(dict(target=t, montage=mon, features=f'f0_{layer}', readout='plain', n=n or T.n,
                                     alpha=alpha, test_top1=top1(pred, G)))
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
