"""Build the NeuroBridge data cache on Drive (run once; resumable per subject).

    python scripts/nb_data.py --drive_base /content/drive/MyDrive/eegcal_nb --config /content/nb_config.json \
        [--subjects 1 ... 10] [--montages posterior17 crown8] [--ubp_cache <UBP montage_cache/hf_whiten>]

Source: NeuroBridge's preprocessed THINGS-EEG on Hugging Face (Feronl/NeuroBridge): MVNN-whitened over all 63
channels, 250 Hz, 0-1000 ms, train (1654 concepts, 10 images, 4 reps, 63, 250), test (200, 1, 80, 63, 250), plus
CLIP RN50 image features (clean and the fused augmented views). Per subject, the raw files are downloaded to the
runtime's local disk, reduced, written to Drive, and deleted:
  <drive_base>/data/sub-NN/<montage>.npz   (posterior17 and crown8; occipital8 is cut from posterior17 on load)
                                           train (16540, C, 250) fp16, 4 reps averaged
                                           test (200, C, 250) fp16, 80 reps averaged; test_trials (200, 80, C, 250)
                                           train_labels (concept 0..1653), train_paths / test_paths
  <drive_base>/features/                   image_{train,test}.npy (clean), <aug>/{train,test}.npy (fused)
Image paths: NeuroBridge orders images as sorted concept folders x sorted files (extract_feature.py), the same
order as the UBP pipeline's image list. With --ubp_cache, the UBP cache's (sorted) paths are attached and checked:
the per-image EEG of the two preprocessings must correlate better for matched than for shuffled images.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
REPO = 'Feronl/NeuroBridge'
AUG = 'GaussianBlur-GaussianNoise-LowResolution-Mosaic'


def dl(f, local):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, f, repo_type='dataset', local_dir=local)


def ubp_paths(ubp_cache, sub):
    """Sorted train / test image paths from the UBP cache (any montage file of any subject)."""
    for s in [sub] + list(range(1, 11)):
        for m in ('posterior17', 'occipital8'):
            p = os.path.join(ubp_cache, f'sub-{s:02d}', f'{m}.npz')
            if os.path.exists(p):
                d = np.load(p)
                return np.array(sorted(set(d['train_paths']))), np.array(sorted(set(d['test_paths']))), p
    return None, None, None


def check_order(nb_train, ch_nb, ubp_file, n=600, seed=0):
    """Correlate per-image EEG (posterior channels, all time points) of NB and UBP for matched vs shuffled images."""
    from eegalign import ubp as U
    d = np.load(ubp_file)
    order = np.argsort(d['train_paths'])                     # UBP rows -> sorted path order
    chans = U.MONTAGES['occipital8']
    xu = d['train'][order].astype(np.float32)
    mon = os.path.basename(ubp_file)[:-4]
    iu = [U.MONTAGES[mon].index(c) for c in chans]
    ib = [ch_nb.index(c) for c in chans]
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(xu), n, replace=False)
    a = xu[pick][:, iu].reshape(n, -1)
    b = nb_train[pick][:, ib].reshape(n, -1).astype(np.float32)
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    matched = float((a * b).mean(1).mean())
    shuf = float((a * b[rng.permutation(n)]).mean(1).mean())
    return matched, shuf


def build_subject(sub, montages, ch_names, out_dir, local, paths, ubp_file):
    from eegalign import ubp as U
    tr_f = dl(f'things_eeg/preprocessed_eeg/sub-{sub:02d}/train.npy', local)
    te_f = dl(f'things_eeg/preprocessed_eeg/sub-{sub:02d}/test.npy', local)
    tr = np.load(tr_f, mmap_mode='r')
    te = np.load(te_f, mmap_mode='r')
    assert tr.shape[:3] == (1654, 10, 4) and tr.shape[3] == len(ch_names), tr.shape
    assert te.shape[:3] == (200, 1, 80), te.shape
    print(f'  sub-{sub:02d}: train {tr.shape} {tr.dtype}, test {te.shape} {te.dtype}', flush=True)
    T = min(tr.shape[-1], 250)
    # average the 4 train repetitions once for all channels (chunked over concepts; ~250 MB reads)
    avg = np.empty((1654, 10, len(ch_names), T), np.float32)
    for i in range(0, 1654, 64):
        avg[i:i + 64] = np.asarray(tr[i:i + 64, :, :, :, :T], np.float32).mean(2)
    avg = avg.reshape(16540, len(ch_names), T)
    te_trials = np.asarray(te[:, 0, :, :, :T], np.float32)                    # (200, 80, 63, T)
    labels = np.repeat(np.arange(1654), 10)
    tr_paths, te_paths = paths
    os.makedirs(os.path.join(out_dir, f'sub-{sub:02d}'), exist_ok=True)
    for mon in montages:
        idx = [ch_names.index(c) for c in U.MONTAGES[mon]]
        tmp = os.path.join(local, f'sub-{sub:02d}_{mon}.npz')
        np.savez(tmp, train=avg[:, idx].astype(np.float16), test=te_trials[:, :, idx].mean(1).astype(np.float16),
                 test_trials=te_trials[:, :, idx].astype(np.float16), train_labels=labels,
                 train_paths=tr_paths, test_paths=te_paths, channels=np.array(U.MONTAGES[mon]))
        shutil.move(tmp, os.path.join(out_dir, f'sub-{sub:02d}', f'{mon}.npz'))   # write locally, then one copy
    info = {}
    if ubp_file:
        m, s = check_order(avg, ch_names, ubp_file)
        info = dict(order_check_matched_r=round(m, 3), order_check_shuffled_r=round(s, 3))
        print(f'  sub-{sub:02d}: NB vs UBP per-image EEG correlation, matched {m:.3f} vs shuffled {s:.3f}', flush=True)
        if not m > s + 0.05:
            print('  WARNING: image order may differ between NeuroBridge and UBP data; paths are NOT reliable',
                  flush=True)
    del tr, te, avg, te_trials
    for f in (tr_f, te_f):
        os.remove(f)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--drive_base', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--montages', nargs='+', default=['posterior17', 'crown8'],
                   help='stored montages; occipital8 is cut from posterior17 at load time')
    p.add_argument('--ubp_cache', default='', help='UBP montage_cache/<source> dir, for image paths + order check')
    p.add_argument('--local', default='/content/nb_raw')
    a = p.parse_args()
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
    data_dir, feat_dir = f'{a.drive_base}/data', f'{a.drive_base}/features'
    cfg = dict(drive_base=a.drive_base, cache_dir=data_dir, feat_dir=feat_dir, ckpt_dir=f'{a.drive_base}/checkpoints',
               results_dir=f'{a.drive_base}/results', subjects=a.subjects)
    for d in (data_dir, feat_dir, cfg['ckpt_dir'], cfg['results_dir'], a.local):
        os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(a.config, 'w'), indent=1)
    # channel names + image features (small; copied to Drive once)
    info_p = os.path.join(data_dir, 'info.json')
    if not os.path.exists(info_p):
        shutil.copy(dl('things_eeg/preprocessed_eeg/info.json', a.local), info_p)
    ch_names = json.load(open(info_p))['ch_names']
    for f in ('image_train.npy', 'image_test.npy', f'{AUG}/train.npy', f'{AUG}/test.npy'):
        dst = os.path.join(feat_dir, f)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(dl(f'things_eeg/image_feature/RN50/{f}', a.local), dst)
    ftr = np.load(os.path.join(feat_dir, 'image_train.npy'), mmap_mode='r')
    fau = np.load(os.path.join(feat_dir, AUG, 'train.npy'), mmap_mode='r')
    print(f'features: clean train {ftr.shape}, fused augmented train {fau.shape}; {len(ch_names)} channels')
    tr_paths, te_paths, ubp_file = ubp_paths(a.ubp_cache, a.subjects[0]) if a.ubp_cache else (None, None, None)
    if tr_paths is None or len(tr_paths) != 16540 or len(te_paths) != 200:
        print('no usable UBP cache: image paths are placeholders (train/OOOO_II, test/OOOO); low-level properties '
              'and THINGS categories will not be available for this data')
        tr_paths = np.array([f'train/{o:04d}_{i:02d}' for o in range(1654) for i in range(10)])
        te_paths = np.array([f'test/{o:04d}' for o in range(200)])
        ubp_file = None
    log_p = os.path.join(data_dir, 'build_log.json')
    log = json.load(open(log_p)) if os.path.exists(log_p) else {}
    for sub in a.subjects:
        done = all(os.path.exists(os.path.join(data_dir, f'sub-{sub:02d}', f'{m}.npz')) for m in a.montages)
        if done:
            print(f'  sub-{sub:02d}: cached')
            continue
        log[str(sub)] = build_subject(sub, a.montages, ch_names, data_dir, a.local, (tr_paths, te_paths), ubp_file)
        json.dump(log, open(log_p, 'w'), indent=1)
    print('done. config ->', a.config)


if __name__ == '__main__':
    main()
