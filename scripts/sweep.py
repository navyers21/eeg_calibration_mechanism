"""Phases of the "what does calibration fix?" study. Each calls one entry point per (target, montage).

    python scripts/sweep.py props    --config C                    # once: low-level image properties -> Drive
    python scripts/sweep.py within   --config C [--dry_run]        # within-subject decoders (the "before")
    python scripts/sweep.py interp   --config C [--eval_set test]  # probes + erasures on every model mode
    python scripts/sweep.py gap      --config C [--montage occipital8]
    python scripts/sweep.py subspace --config C                    # mechanism-guided calibration vs the maps
    python scripts/sweep.py summary  --config C --phase subspace
    python scripts/sweep.py subx_tune --config C                   # v2 subspace: lr per family, subject 1 only
    python scripts/sweep.py subx      --config C                   # v2 subspace on targets 2-10 with tuned lrs
    python scripts/sweep.py summary   --config C --phase subspace,subx
    python scripts/sweep.py probes2   --config C [--eval_set test] # THINGS-category probes, residualized low-level
    python scripts/sweep.py gap2      --config C --montage posterior17

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
# v2 subspace calibration. Families: the k=6 low-level / semantic / random comparison of the preregistered test, the
# largest semantic subspace (rank 19), and a random-capacity curve up to the full first-layer output (k=1024 = any
# linear edit after layer 1: the ceiling of editing at this point). lr per family chosen on subject 1 (excluded from
# every analysis), budget 50, by mean test top-1 over 3 seeds.
SUBX_FAMILIES = ['subx_low_k6', 'subx_sem_k6', 'subx_rand_k6', 'subx_sem_k19', 'subx_rand_k19', 'subx_rand_k64',
                 'subx_rand_k256', 'subx_rand_k1024']
SUBX_LRS = ['1e-3', '1e-2', '1e-1']


# Round 3 (mechanism2.py): calibrate where each kind of information breaks. Output-side edits for the misaligned
# category information, input-side pattern-restricted maps for the missing low-level information, map + output edit,
# and subx_low_k6 again with a wider lr grid (round 2 picked the lowest lr tried).
MECH2_FAMILIES = ['outtgt_k8', 'outtgt_k32', 'outpca_k32', 'outrand_k32', 'pmaplow_r4', 'pmaprand_r4',
                  'align+outtgt_k32', 'sulora2+outtgt_k32', 'subx_low_k6']
MECH2_LRS = ['1e-4', '3e-4', '1e-3', '3e-3', '1e-2']


def tuned_lrs(csv_path, montage, phase='subx_tune', families=SUBX_FAMILIES, lrs=SUBX_LRS):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[(df.phase == phase) & (df.montage == montage)]
    best = {}
    for fam in families:
        g = df[df.adapter.str.startswith(fam + '_lr')].groupby('adapter').test_top1.mean()
        if len(g) < len(lrs):
            raise SystemExit(f'{phase} not finished for {fam} {montage}: run the {phase} phase first')
        best[fam] = g.idxmax()
    return best


def call(script, args, dry):
    cmd = [sys.executable, os.path.join(ROOT, script), *args]
    print('$', ' '.join(cmd[1:]), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['props', 'within', 'interp', 'gap', 'subspace', 'summary',
                                      'subx_tune', 'subx', 'probes2', 'gap2', 'mech2_tune', 'mech2', 'oracle'])
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
    elif a.cmd == 'subx_tune':
        for mon in MONTAGES:
            call('calibrate.py', c + ['--phase', 'subx_tune', '--target', '1', '--montage', mon, '--adapters', 'f0',
                                      *[f'{f}_lr{lr}' for f in SUBX_FAMILIES for lr in SUBX_LRS],
                                      '--budgets', '50', '--seeds', '0', '1', '2'], a.dry_run)
    elif a.cmd == 'subx':
        import json
        csv_path = json.load(open(a.config))['results_csv']
        for mon in MONTAGES:
            best = tuned_lrs(csv_path, mon) if not a.dry_run else {f: f + '_lr1e-2' for f in SUBX_FAMILIES}
            print(f'{mon}: tuned on subject 1 ->', best, flush=True)
            for t in a.targets:
                call('calibrate.py', c + ['--phase', 'subx', '--target', str(t), '--montage', mon,
                                          '--adapters', *best.values(), '--budgets', '25', '50', '200',
                                          '--seeds', '0', '1', '2'], a.dry_run)
    elif a.cmd == 'mech2_tune':
        for mon in MONTAGES:
            call('calibrate.py', c + ['--phase', 'mech2_tune', '--target', '1', '--montage', mon, '--adapters', 'f0',
                                      *[f'{f}_lr{lr}' for f in MECH2_FAMILIES for lr in MECH2_LRS],
                                      '--budgets', '50', '--seeds', '0', '1', '2'], a.dry_run)
    elif a.cmd == 'mech2':
        import json
        csv_path = json.load(open(a.config))['results_csv']
        for mon in MONTAGES:
            best = (tuned_lrs(csv_path, mon, 'mech2_tune', MECH2_FAMILIES, MECH2_LRS) if not a.dry_run
                    else {f: f + '_lr1e-3' for f in MECH2_FAMILIES})
            print(f'{mon}: tuned on subject 1 ->', best, flush=True)
            for t in a.targets:
                call('calibrate.py', c + ['--phase', 'mech2', '--target', str(t), '--montage', mon,
                                          '--adapters', *best.values(), '--budgets', '25', '50', '200',
                                          '--seeds', '0', '1', '2'], a.dry_run)
    elif a.cmd == 'oracle':
        call('oracle_readout.py', c + ['--targets', *map(str, a.targets), '--montages', *a.montages], a.dry_run)
    elif a.cmd == 'probes2':
        for mon in a.montages:
            for t in a.targets:
                call('run_probes2.py', c + ['--target', str(t), '--montage', mon, '--eval_set', a.eval_set,
                                            '--n_images', str(a.n_images)], a.dry_run)
    elif a.cmd == 'gap2':
        call('gap2.py', c + ['--montage', a.montage, '--eval_set', a.eval_set] + (['--out', a.out] if a.out else []),
             a.dry_run)
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
