"""Low-level image properties (Karan's mi.image_props) for every THINGS training image, computed ONCE from the
images and cached on Drive, so later runs need no image download. Semantic labels: k-means of the unblurred CLIP
features of pool (training) images, fit once per run on images the model was trained on."""
import os

import numpy as np

from . import ubp as U

LOW_LEVEL = ['luminance', 'contrast', 'saturation', 'colorfulness', 'high_sf_ratio', 'edge_density']


def props_path():
    return os.path.join(U.ENV['feat_dir'], 'lowlevel_props.npz')


def build_props_cache():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from mi.image_props import low_level_props
    path = props_path()
    if os.path.exists(path):
        return path
    paths = list(U.all_image_paths())      # train and test images
    rows = []
    for i, p in enumerate(paths):
        r = low_level_props(os.path.join(U.ENV['image_root'], p))
        rows.append([r[k] for k in LOW_LEVEL])
        if i % 2000 == 0:
            print(f'  low-level props {i}/{len(paths)}', flush=True)
    np.savez(path, paths=np.array(paths), props=np.array(rows, np.float32), names=np.array(LOW_LEVEL))
    print('saved', path)
    return path


_P = {}


def props_for(paths):
    if not _P:
        if not os.path.exists(props_path()):
            raise FileNotFoundError(f'{props_path()} missing: run the props phase once (needs the images)')
        d = np.load(props_path())
        _P['index'] = {str(p): i for i, p in enumerate(d['paths'])}
        _P['props'] = d['props']
    return _P['props'][[_P['index'][str(p)] for p in paths]]


_KM = {}


def semantic_kmeans(train_paths, k=20, seed=0):
    key = (len(train_paths), str(train_paths[0]), str(train_paths[-1]), k, seed)
    if key not in _KM:
        _KM[key] = _semantic_kmeans(train_paths, k, seed)
    return _KM[key]


def _semantic_kmeans(train_paths, k=20, seed=0):
    from sklearn.cluster import KMeans
    f = U.feats_for(train_paths, 'none', 'RN50').numpy()
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(f)


def semantic_labels(km, paths):
    f = U.feats_for(paths, 'none', 'RN50').numpy()
    return km.predict(f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8))
