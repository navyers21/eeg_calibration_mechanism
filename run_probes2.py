"""Probes v2 for one target subject: semantic labels that do not come from CLIP, and low-level probes with the
semantic category regressed out.

    python run_probes2.py --config C --target 2 --montage posterior17 [--eval_set held] [--n_images 50]

Per model mode (within, f0, f0_map, sulora_map), on the same images:
  retrieval  (images with a THINGS human-rated category only; gallery = those images)
      top1, category (retrieved image has the right THINGS category), fine (top-1 among same-category images)
  probes at input_linear and output (grouped CV by concept, as in mi/probes.py)
      low_r2       mean R^2 of the 6 low-level properties, all images
      lowres_r2    same, after regressing each property on the THINGS category (categorized images only):
                   low-level information that is NOT explained by what the object is
      cat_acc      THINGS-category accuracy (categorized images), with majority-class chance and a shuffled control
Output: <drive>/interp2/<eval>/<montage>/t<NN>.csv, one row per (mode, layer). Finished targets are skipped.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.eegalign import env, props  # noqa: E402
from src.eegalign import categories  # noqa: E402
from interp.ubp_adapter import UBPAdapter  # noqa: E402
from mi import probes, retrieval  # noqa: E402
from mi.pipeline import capture_all, encode_all  # noqa: E402

MODES = ['within', 'f0', 'f0_map', 'sulora_map']
LAYERS = ['input_linear', 'output']


def residualize(Y, labels):
    """Remove each column's category mean (= OLS on category one-hot)."""
    R = Y.astype(np.float64).copy()
    for c in np.unique(labels):
        R[labels == c] -= R[labels == c].mean(0)
    return R


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--target', type=int, required=True)
    p.add_argument('--montage', default='posterior17')
    p.add_argument('--eval_set', default='held', choices=['held', 'test'])
    p.add_argument('--n_images', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    cfg = env.load(a.config)
    out_dir = os.path.join(os.path.dirname(cfg['results_csv']), 'interp2', a.eval_set, a.montage)
    out = os.path.join(out_dir, f't{a.target:02d}.csv')
    if os.path.exists(out):
        print(f'[probes2] {a.montage} {a.eval_set} t{a.target}: done, skipping', flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev, rows = None, []
    for mode in MODES:
        ad = UBPAdapter(mode, a.montage, a.n_images, a.seed, a.eval_set)
        if ev is None:
            ev = ad.load_eval_set(a.target)
            groups = np.asarray(ev['groups'])
            low = props.props_for(ev['image_paths']).astype(np.float64)
            cat, names = categories.labels(ev['image_paths'])
            keep = cat >= 0
            lowres = residualize(low[keep], cat[keep])
            chance = np.bincount(cat[keep]).max() / keep.sum()
            G = torch.as_tensor(ev['gallery_feats']).float()
            print(f'[probes2] {a.montage} {a.eval_set} t{a.target}: {len(groups)} images, {keep.sum()} with a '
                  f'THINGS category ({len(np.unique(groups[keep]))} concepts, {len(names)} categories: {names})',
                  flush=True)
        model = ad.load_model(a.target, dev).eval()
        layers, axes = ad.get_layers(model)
        E = encode_all(ad, model, ev['eeg'], 64, dev)
        m = retrieval.summarize(retrieval.retrieval_metrics(E[keep], G[keep], cat[keep]))
        acts = capture_all(ad, model, ev['eeg'], {k: layers[k] for k in LAYERS}, axes, 64, dev)
        for lname in LAYERS:
            X = acts[lname].numpy()
            r = dict(target=a.target, montage=a.montage, eval_set=a.eval_set, mode=mode, layer=lname,
                     n_images=len(groups), n_cat=int(keep.sum()), n_categories=len(names),
                     top1=m['top1'], category=m['cluster_acc'], fine=m['within_cluster_top1'],
                     low_r2=np.mean([probes.probe_continuous(X, low[:, j], groups) for j in range(low.shape[1])]),
                     lowres_r2=np.mean([probes.probe_continuous(X[keep], lowres[:, j], groups[keep])
                                        for j in range(lowres.shape[1])]),
                     cat_acc=probes.probe_categorical(X[keep], cat[keep], groups[keep]),
                     cat_acc_shuffled=probes.probe_categorical(X[keep], cat[keep], groups[keep], shuffle_seed=0),
                     cat_chance=chance)
            rows.append(r)
            print(f'  {mode:10s} {lname:12s} top1 {r["top1"]:.3f} cat {r["category"]:.3f} fine {r["fine"]:.3f} | '
                  f'low {r["low_r2"]:.3f} lowres {r["lowres_r2"]:.3f} cat_probe {r["cat_acc"]:.3f} '
                  f'(chance {chance:.3f}, shuffled {r["cat_acc_shuffled"]:.3f})', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
