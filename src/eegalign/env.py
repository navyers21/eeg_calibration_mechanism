"""Paths and settings shared by every script: written once by scripts/download_data.py, read by everything else."""
import json
import os
import subprocess

from . import ubp as U


def load(config_path):
    """Read the JSON config and initialise the UBP backend. Returns the config dict."""
    with open(config_path) as f:
        cfg = json.load(f)
    commit = subprocess.run(['git', '-C', cfg['ubp_dir'], 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    U.setup(ubp_dir=cfg['ubp_dir'], whiten_dir=cfg['whiten_dir'], nowhiten_dir=cfg['nowhiten_dir'],
            image_root=cfg['image_root'], cache_dir=cfg['cache_dir'], feat_dir=cfg['feat_dir'],
            ckpt_dir=cfg['ckpt_dir'], results_csv=cfg['source_csv'], source=cfg['source'],
            pool_subjects=cfg['subjects'], num_workers=os.cpu_count(), ubp_commit=commit,
            local_montages=tuple(cfg.get('local_montages', ())), local_cache_dir=cfg.get('local_cache_dir'))
    for k in ('maps_dir',):
        os.makedirs(cfg[k], exist_ok=True)
    return cfg
