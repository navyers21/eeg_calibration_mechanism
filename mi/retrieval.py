"""Retrieval metrics that separate 'gets the category right' from 'gets the exact image right'."""
import numpy as np
import torch


def _sim(E, G):
    E = torch.nn.functional.normalize(torch.as_tensor(E).float(), dim=1)
    G = torch.nn.functional.normalize(torch.as_tensor(G).float(), dim=1)
    return (E @ G.T).numpy()


def retrieval_metrics(E, G, clusters):
    """E: (n, d) predicted embeddings; G: (n, d) gallery, row i = true image for E[i]."""
    S = _sim(E, G)
    n = len(S)
    idx = np.arange(n)
    rank = (S > S[idx, idx][:, None]).sum(1)  # 0 = correct is top-1
    top1 = (rank == 0)
    top5 = (rank < 5)
    clusters = np.asarray(clusters)
    best = S.argmax(1)
    cluster_hit = clusters[best] == clusters
    # within-cluster: only compete against images from the same semantic cluster
    within = np.zeros(n, bool)
    for i in range(n):
        same = clusters == clusters[i]
        within[i] = S[i, same].argmax() == np.flatnonzero(same).tolist().index(i)
    return dict(top1=top1, top5=top5, cluster_acc=cluster_hit, within_cluster_top1=within)


def summarize(m):
    return {k: float(np.mean(v)) for k, v in m.items()}


def paired_bootstrap(a, b, groups, n_boot=2000, seed=0):
    """Mean(b - a) with a 95% CI, resampling concept groups (not individual images)."""
    a, b, groups = np.asarray(a, float), np.asarray(b, float), np.asarray(groups)
    ug = np.unique(groups)
    d_by_g = {g: (b - a)[groups == g] for g in ug}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug))
        stats.append(np.concatenate([d_by_g[g] for g in pick]).mean())
    return float((b - a).mean()), tuple(np.percentile(stats, [2.5, 97.5]))
