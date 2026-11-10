"""End-to-end mech interp run for one subject / one checkpoint.

Stages
  1. Image labels: low-level pixel properties + semantic clusters; confound table.
  2. Probes: which properties are linearly readable at each layer (grouped CV, shuffled control).
  3. Baseline retrieval on the SCORE half of the validation images.
  4. Interventions: fit LEACE on the FIT half, apply on the SCORE half, compare against
     random subspaces of the same rank. Report full, category-level and within-category retrieval.

Validation images are split into FIT / SCORE halves by concept, so the eraser is never fit on
the images it is evaluated on. The official 200-image test set is never touched.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from . import hooks, image_props, leace, probes, retrieval


def _batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


@torch.no_grad()
def encode_all(adapter, model, eeg, bs, device):
    return torch.cat([adapter.encode(model, eeg[s].to(device)).float().cpu() for s in _batches(len(eeg), bs)])


@torch.no_grad()
def capture_all(adapter, model, eeg, layers, feat_axes, bs, device, keep_tokens=False):
    with hooks.capture(model, layers, feat_axes, keep_tokens) as store:
        encode_all(adapter, model, eeg, bs, device)
    return {k: torch.cat(v) for k, v in store.items()}


def split_by_group(groups, frac=0.5, seed=0):
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    fit_g = set(rng.choice(ug, int(len(ug) * frac), replace=False))
    fit = np.array([g in fit_g for g in groups])
    return np.flatnonzero(fit), np.flatnonzero(~fit)


def confound_table(low, names, clusters):
    """Eta-squared: fraction of each low-level property's variance explained by semantic cluster."""
    out = {}
    for j, n in enumerate(names):
        y = low[:, j]
        ss_tot = ((y - y.mean()) ** 2).sum()
        ss_b = sum(((clusters == c).sum() * (y[clusters == c].mean() - y.mean()) ** 2) for c in np.unique(clusters))
        out[n] = float(ss_b / (ss_tot + 1e-12))
    return out


def run(adapter, subject, out_dir, device="cuda", bs=64, n_clusters=20, n_random=3,
        max_fit_tokens=200_000, intervention_layers=None, seed=0, skip_probes=False,
        concept_names=("low_level_all", "luminance", "contrast", "semantic_cluster"), norm_match=True):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

    model = adapter.load_model(subject, device).eval()
    layers, feat_axes = adapter.get_layers(model)
    data = adapter.load_eval_set(subject)
    eeg, groups, G = data["eeg"], np.asarray(data["groups"]), torch.as_tensor(data["gallery_feats"]).float()
    n = len(eeg)
    log(f"subject={subject} n_eval_images={n} n_groups={len(np.unique(groups))} layers={list(layers)}")

    # 1. labels -------------------------------------------------------------------
    cache = os.path.join(out_dir, "low_level.npy")
    if os.path.exists(cache):
        low = np.load(cache)
    else:
        low = image_props.compute_low_level(data["image_paths"])
        np.save(cache, low)
    # cluster in the raw image-encoder space if the adapter provides it (model-independent labels)
    clusters, _ = image_props.semantic_clusters(np.asarray(adapter.load_train_image_feats()),
                                                np.asarray(data.get("cluster_feats", G)), k=n_clusters, seed=seed)
    conf = confound_table(low, image_props.LOW_LEVEL, clusters)
    log("eta^2 of low-level props explained by semantic cluster:", {k: round(v, 3) for k, v in conf.items()})

    # 2. probes ---------------------------------------------------------------------
    pooled = capture_all(adapter, model, eeg, layers, feat_axes, bs, device)
    if skip_probes:
        probe_rows = []
        log("probes skipped")
    else:
        probe_rows = probes.probe_all_layers({k: v.numpy() for k, v in pooled.items()}, low,
                                             image_props.LOW_LEVEL, clusters, groups)
        pd.DataFrame(probe_rows).to_csv(os.path.join(out_dir, "probes.csv"), index=False)
        log("probes done")

    # 3. baseline -------------------------------------------------------------------
    fit_idx, score_idx = split_by_group(groups, seed=seed)
    Gs, cs, gs = G[score_idx], clusters[score_idx], groups[score_idx]
    base_E = encode_all(adapter, model, eeg[score_idx], bs, device)
    base = retrieval.retrieval_metrics(base_E, Gs, cs)
    log("baseline (score half, gallery =", len(score_idx), "):", retrieval.summarize(base))

    # 4. interventions --------------------------------------------------------------
    all_concepts = {"low_level_all": low, "semantic_cluster": clusters.astype(np.int64)}
    for j, nm in enumerate(image_props.LOW_LEVEL):
        all_concepts[nm] = low[:, j]
    concepts = {c: all_concepts[c] for c in concept_names}

    def delta_norm(lname, mod, ax, e):
        with hooks.edit(mod, e, ax):
            edited = capture_all(adapter, model, eeg[score_idx], {lname: mod}, {lname: ax}, bs, device)[lname]
        return float((edited - pooled[lname][score_idx]).norm(dim=1).mean())

    rows = []
    for lname in (intervention_layers or list(layers)):
        mod, ax = layers[lname], feat_axes.get(lname, -1)
        toks = capture_all(adapter, model, eeg[fit_idx], {lname: mod}, {lname: ax}, bs, device, keep_tokens=True)[lname]
        nt = toks.shape[1]
        X = toks.reshape(-1, toks.shape[-1])
        rep = lambda z: np.repeat(np.asarray(z)[fit_idx], nt, axis=0)
        sub = np.random.default_rng(seed).choice(len(X), min(len(X), max_fit_tokens), replace=False)
        for cname, z in concepts.items():
            er = leace.fit_leace(X[sub], torch.as_tensor(rep(z)[sub]))
            variants = [("leace", er)] + [(f"random{s}", er.random_control(s)) for s in range(n_random)]
            if norm_match:
                # scale each random edit so its average size on the scored images equals LEACE's
                target = delta_norm(lname, mod, ax, er)
                for _, e in variants[1:]:
                    e.scale = target / max(delta_norm(lname, mod, ax, e), 1e-8)
            for vname, e in variants:
                with hooks.edit(mod, e, ax):
                    E = encode_all(adapter, model, eeg[score_idx], bs, device)
                    edited = capture_all(adapter, model, eeg[score_idx], {lname: mod}, {lname: ax}, bs, device)[lname]
                m = retrieval.retrieval_metrics(E, Gs, cs)
                row = dict(layer=lname, concept=cname, variant=vname, rank=e.rank, edit_scale=e.scale,
                           delta_norm=float((edited - pooled[lname][score_idx]).norm(dim=1).mean()),
                           act_norm=float(pooled[lname][score_idx].norm(dim=1).mean()),
                           **retrieval.summarize(m))
                # did erasure remove the concept? re-probe edited pooled acts (score half, grouped CV)
                zs = np.asarray(z)[score_idx]
                if cname == "semantic_cluster":
                    row["residual_probe"] = probes.probe_categorical(edited.numpy(), zs, gs)
                elif zs.ndim == 1:
                    row["residual_probe"] = probes.probe_continuous(edited.numpy(), zs, gs)
                for k in ("top1", "cluster_acc", "within_cluster_top1"):
                    d, ci = retrieval.paired_bootstrap(base[k], m[k], gs)
                    row[f"d_{k}"], row[f"d_{k}_ci"] = d, ci
                rows.append(row)
            log(f"{lname} / {cname}: leace d_top1={rows[-n_random-1]['d_top1']:+.3f} "
                f"random d_top1={np.mean([r['d_top1'] for r in rows[-n_random:]]):+.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "interventions.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(subject=subject, n_eval=n, n_score=len(score_idx), confound_eta2=conf,
                       baseline=retrieval.summarize(base)), f, indent=2)
    log("done ->", out_dir)
    return pd.DataFrame(probe_rows), df
