"""Phases of the "what does calibration fix?" study. Each calls one entry point per (target, montage).

    python scripts/sweep.py props    --config C                    # once: low-level image properties -> Drive
    python scripts/sweep.py within   --config C [--dry_run]        # within-subject decoders (the "before")
    python scripts/sweep.py interp   --config C [--eval_set test]  # probes + erasures on every model mode
    python scripts/sweep.py gap      --config C [--montage occipital8]
    python scripts/sweep.py subspace --config C                    # mechanism-guided calibration vs the maps
    python scripts/sweep.py summary  --config C --phase subspace

Fixed before running: targets 2-10 (subject 1 was used to tune the calibration recipe), posterior-17 first, then
occipital-8; maps for the interp phase fitted on 50 images, seed 0; subspace calibration k = 6 (36 numbers).
Go / no-go: (1) gap: low-level and semantic measures must differ clearly in the fraction the map restores;
(2) subspace: sub_low must match or beat the SuLoRA-style map at 25-50 images, and beat sub_rand.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
TARGETS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
MONTAGES = ['posterior17', 'occipital8']
MODES = ['within', 'f0', 'f0_map', 'sulora_map']
PHASES = {
    'subspace': dict(targets=TARGETS, montages=MONTAGES,
                     adapters=['f0', 'align', 'sulora2', 'sub_low', 'sub_sem', 'sub_rand'],
                     budgets=['25', '50', '100', '200'], seeds=[0, 1, 2]),
}


def call(script, args, dry):
    cmd = [sys.executable, os.path.join(ROOT, script), *args]
    print('$', ' '.join(cmd[1:]), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['props', 'within', 'interp', 'gap', 'subspace', 'summary'])
    p.add_argument('--config', required=True)
    p.add_argument('--montages', nargs='+', default=MONTAGES)
    p.add_argument('--montage', default='posterior17')
    p.add_argument('--targets', type=int, nargs='+', default=TARGETS)
    p.add_argument('--eval_set', default='held', choices=['held', 'test'])
    p.add_argument('--n_images', type=int, default=50)
    p.add_argument('--phase', default='subspace')
    p.add_argument('--out', default='')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    c = ['--config', a.config]
    if a.cmd == 'props':
        call('scripts/build_props.py', c, a.dry_run)
    elif a.cmd == 'within':
        call('train_within.py', c + ['--targets', *map(str, a.targets), '--montages', *a.montages], a.dry_run)
    elif a.cmd == 'interp':
        for mon in a.montages:
            for t in a.targets:
                call('run_interp.py', c + ['--target', str(t), '--montage', mon, '--modes', *MODES,
                                           '--n_images', str(a.n_images), '--eval_set', a.eval_set], a.dry_run)
    elif a.cmd == 'gap':
        call('gap_analysis.py', c + ['--montage', a.montage, '--eval_set', a.eval_set, '--n_images', str(a.n_images)]
             + (['--out', a.out] if a.out else []), a.dry_run)
    elif a.cmd == 'subspace':
        ph = PHASES['subspace']
        for t in ph['targets']:
            for mon in ph['montages']:
                call('calibrate.py', c + ['--phase', 'subspace', '--target', str(t), '--montage', mon,
                                          '--adapters', *ph['adapters'], '--budgets', *ph['budgets'],
                                          '--seeds', *map(str, ph['seeds'])], a.dry_run)
    else:
        import json
        import pandas as pd
        from eegalign.analysis import summary
        pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 300)
        out, curves = summary(json.load(open(a.config))['results_csv'], a.phase)
        print(out.round(3).to_string(index=False))
        print(curves.round(2).to_string())
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            out.round(3).to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
