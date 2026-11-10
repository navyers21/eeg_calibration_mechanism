"""Cross-validated linear probes, grouped so no image/concept appears in both train and test folds."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _prep(D, max_dim, n_train=None):
    steps = [StandardScaler()]
    k = min(max_dim, (n_train or max_dim + 1) - 1)
    if D > k:
        steps.append(PCA(n_components=k, random_state=0))
    return steps


def probe_continuous(X, y, groups, n_splits=5, max_dim=256, shuffle_seed=None):
    y = np.asarray(y, dtype=np.float64)
    if shuffle_seed is not None:
        y = np.random.default_rng(shuffle_seed).permutation(y)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = make_pipeline(*_prep(X.shape[1], max_dim, len(tr)), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return r2_score(y, pred)


def probe_categorical(X, y, groups, n_splits=5, max_dim=256, shuffle_seed=None):
    y = np.asarray(y)
    if shuffle_seed is not None:
        y = np.random.default_rng(shuffle_seed).permutation(y)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = make_pipeline(*_prep(X.shape[1], max_dim, len(tr)), LogisticRegression(C=0.1, max_iter=2000))
        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return accuracy_score(y, pred)


def probe_all_layers(acts, low_level, low_names, sem_labels, groups, n_splits=5):
    """acts: {layer: (n, D) array}. Returns list of result dicts (one per layer x target)."""
    rows = []
    chance = np.bincount(sem_labels).max() / len(sem_labels)
    for layer, X in acts.items():
        X = np.asarray(X)
        for j, name in enumerate(low_names):
            rows.append(dict(layer=layer, target=name, kind="low", metric="R2",
                             score=probe_continuous(X, low_level[:, j], groups, n_splits),
                             shuffled=probe_continuous(X, low_level[:, j], groups, n_splits, shuffle_seed=0)))
        rows.append(dict(layer=layer, target="semantic_cluster", kind="semantic", metric="acc",
                         score=probe_categorical(X, sem_labels, groups, n_splits),
                         shuffled=probe_categorical(X, sem_labels, groups, n_splits, shuffle_seed=0),
                         chance=chance))
    return rows
