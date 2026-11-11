"""Run Karan's probes + erasures (mi/pipeline.py) on one target subject, for several model modes.

    python run_interp.py --config C --target 2 --montage posterior17 --modes within f0 f0_map sulora_map --n_images 50

Every mode is scored on the same target images. Results: <drive>/interp/<eval>/<montage>/t<target>/<mode>_n<n>_s<seed>/
(probes.csv, interventions.csv, summary.json, plots). Finished modes are skipped. Low-level image properties come
from the Drive cache (props phase), so no images are needed here.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from src.eegalign import env, props  # noqa: E402
from interp.ubp_adapter import UBPAdapter  # noqa: E402
from mi import pipeline, plots  # noqa: E402

CONCEPTS = ['low_level_all', 'edge_density', 'high_sf_ratio', 'luminance', 'contrast', 'semantic_cluster']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--target', type=int, required=True)
    p.add_argument('--montage', default='posterior17')
    p.add_argument('--modes', nargs='+', default=['within', 'f0', 'f0_map', 'sulora_map'])
    p.add_argument('--n_images', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval_set', default='held', choices=['held', 'test'])
    p.add_argument('--layers', nargs='+', default=['input_linear', 'output'])
    p.add_argument('--n_random', type=int, default=3)
    a = p.parse_args()
    cfg = env.load(a.config)
    base = os.path.join(os.path.dirname(cfg['results_csv']), 'interp', a.eval_set, a.montage, f't{a.target:02d}')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev = None
    for mode in a.modes:
        tag = mode if mode in ('within', 'f0') else f'{mode}_n{a.n_images}_s{a.seed}'
        out = os.path.join(base, tag)
        if os.path.exists(os.path.join(out, 'interventions.csv')):
            print(f'[interp] {a.montage} t{a.target} {tag}: done, skipping', flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        ad = UBPAdapter(mode, a.montage, a.n_images, a.seed, a.eval_set)
        if ev is None:                      # same images for every mode: load once, on the first mode that runs
            ev = ad.load_eval_set(a.target)
        np.save(os.path.join(out, 'low_level.npy'), props.props_for(ev['image_paths']))   # pipeline reuses this
        print(f'[interp] {a.montage} t{a.target} {tag}: running', flush=True)
        probe_df, int_df = pipeline.run(ad, a.target, out, device=dev, n_random=a.n_random,
                                        intervention_layers=a.layers, concept_names=CONCEPTS, norm_match=True)
        if len(probe_df):
            plots.plot_probes(probe_df, os.path.join(out, 'probes.png'))
        plots.plot_interventions(int_df, os.path.join(out, 'interventions.png'))
        json.dump(dict(mode=mode, target=a.target, montage=a.montage, n_images=a.n_images, seed=a.seed,
                       eval_set=a.eval_set), open(os.path.join(out, 'run.json'), 'w'))


if __name__ == '__main__':
    main()
