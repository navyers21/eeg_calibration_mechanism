"""Download UBP's preprocessed THINGS-EEG (per subject) and the resized images from Hugging Face, then write the
JSON config every other script reads.

    python scripts/download_data.py --drive_base /content/drive/MyDrive/eegdistillation_ubp \
        --config /content/align_config.json

--drive_base should be the same Drive folder as the UBP / B runs so image features and source models are reused.
Set HF_TOKEN in the environment to avoid rate limits.
"""
import argparse
import collections
import glob
import json
import os
import tarfile
import time
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--drive_base', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--source', default='hf_whiten', choices=['hf_whiten', 'clean_mvnn'])
    p.add_argument('--hf_repo', default='Haitao999/things-eeg')
    p.add_argument('--local', default='/content/data_hf')
    p.add_argument('--ubp_dir', default='/content/ubp')
    p.add_argument('--images', action='store_true', help='also download the image set (needed once, for the props phase)')
    p.add_argument('--montages', nargs='+', default=['occipital8', 'posterior17'],
                   help='montages whose Drive caches must exist to skip the download (add crown8 crown_po2 for the crown phase)')
    a = p.parse_args()
    cache_dir = f'{a.drive_base}/montage_cache'           # on Drive: survives runtime resets
    # 63-channel caches are ~1 GB per subject: too big for a free Drive, so they stay on the runtime's local disk.
    # Source checkpoints and results still go to Drive, so only the data has to be rebuilt after a reset.
    local_montages, local_cache = ['all'], '/content/cache_local'
    cfg = dict(ubp_dir=a.ubp_dir, source=a.source, subjects=a.subjects, cache_dir=cache_dir,
               local_montages=local_montages, local_cache_dir=local_cache,
               whiten_dir=f'{a.local}/Preprocessed_data_250Hz_whiten',
               nowhiten_dir='/content/data_nowhiten/Preprocessed_data_250Hz_no_whiten',
               image_root=f'{a.local}/Image_set_Resize',
               feat_dir=f'{a.drive_base}/image_features', ckpt_dir=f'{a.drive_base}/checkpoints',
               source_csv=f'{a.drive_base}/results.csv', results_csv=f'{a.drive_base}/mechanism.csv',
               maps_dir=f'{a.drive_base}/mechanism_maps')
    feats = [f'{a.drive_base}/image_features/RN50__{x}.pt' for x in ('none', 'uniform-11')]
    where = lambda m: local_cache if m in local_montages else cache_dir
    cached = lambda sub: all(os.path.exists(f'{where(m)}/{a.source}/sub-{sub:02d}/{m}.npz') for m in a.montages)
    todo_subj = [sub for sub in a.subjects if not cached(sub)]          # only these need raw EEG
    need_images = a.images or not all(os.path.exists(f) for f in feats)  # images: CLIP features or the props phase
    with open(a.config, 'w') as f:
        json.dump(cfg, f, indent=1)
    print(f'Drive cache: {len(a.subjects) - len(todo_subj)}/{len(a.subjects)} subjects cached for {", ".join(a.montages)}; '
          f'image features {"missing" if need_images else "cached"}.')
    if not todo_subj and not need_images:
        print('Nothing to download. Config written to', a.config)
        return
    print(f'Downloading raw EEG for subjects {todo_subj or "none"}' + (' + the image set' if need_images else ''))
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
    from huggingface_hub import list_repo_files, snapshot_download

    files = list_repo_files(a.hf_repo, repo_type='dataset')

    def prefix_of(key):
        hits = [f for f in files if key in f]
        return hits[0][:hits[0].index(key)] if hits else None

    patterns = []
    keys = ['Preprocessed_data_250Hz_whiten'] + (['Preprocessed_data_250Hz_no_whiten'] if a.source != 'hf_whiten' else [])
    for key in keys:
        pre = prefix_of(key + '/')
        if pre is None:
            print(f'!! {key} not found on the Hub')
            continue
        patterns += [f'{pre}{key}/sub-{s:02d}/*' for s in todo_subj]
    if need_images:
        pre = prefix_of('Image_set_Resize')
        assert pre is not None, 'No Image_set_Resize on the Hub.'
        patterns.append(f'{pre}Image_set_Resize*')

    for attempt in range(8):
        try:
            snapshot_download(a.hf_repo, repo_type='dataset', local_dir=a.local, allow_patterns=patterns, max_workers=4)
            break
        except Exception as e:
            if '429' not in str(e) and 'Too Many Requests' not in str(e):
                raise
            print(f'Rate-limited by Hugging Face, retrying in {60 * (attempt + 1)}s ...')
            time.sleep(60 * (attempt + 1))
    else:
        raise RuntimeError('Still rate-limited after 8 attempts; wait a few minutes and re-run.')

    for arc in glob.glob(f'{a.local}/**/Image_set_Resize*.zip', recursive=True) + \
            glob.glob(f'{a.local}/**/Image_set_Resize*.tar*', recursive=True):
        print('extracting', arc)
        (zipfile.ZipFile(arc) if arc.endswith('.zip') else tarfile.open(arc)).extractall(os.path.dirname(arc))

    def find_dir(name):
        for root, dirs, _ in os.walk(a.local):
            if name in dirs:
                return os.path.join(root, name)

    image_root = find_dir('Image_set_Resize') or cfg['image_root']
    if need_images:
        assert os.path.isdir(f'{image_root}/train_images'), 'images not found after download'
    cfg.update(whiten_dir=find_dir('Preprocessed_data_250Hz_whiten'), image_root=image_root,
               nowhiten_dir=find_dir('Preprocessed_data_250Hz_no_whiten') or cfg['nowhiten_dir'])
    cfg['whiten_dir'] = cfg['whiten_dir'] or f'{a.local}/Preprocessed_data_250Hz_whiten'
    with open(a.config, 'w') as f:
        json.dump(cfg, f, indent=1)
    print(json.dumps(cfg, indent=1))
    # Build every subject/montage cache on Drive now, so the next runtime can skip this download entirely.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    from eegalign import env
    from eegalign import ubp as U
    env.load(a.config)
    for s in todo_subj:
        for m in a.montages:
            U.subject_montage(s, m)
        U.free_raw()
        print(f'  cached subject {s}: {", ".join(a.montages)}', flush=True)
    print('All caches built. Do not interrupt the next steps: source checkpoints are saved to Drive as each finishes.')


if __name__ == '__main__':
    main()
