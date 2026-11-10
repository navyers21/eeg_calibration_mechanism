import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_probes(probe_df, path=None):
    layers = list(dict.fromkeys(probe_df.layer))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    low = probe_df[probe_df.kind == "low"]
    for t, g in low.groupby("target", sort=False):
        axes[0].plot(range(len(g)), g.score, marker="o", label=t)
    axes[0].plot(range(len(layers)), low.groupby("layer", sort=False).shuffled.mean().reindex(layers),
                 "k--", label="shuffled (mean)")
    axes[0].set(title="Low-level properties: CV R²", xticks=range(len(layers)), ylabel="R²")
    axes[0].set_xticklabels(layers, rotation=45, ha="right")
    axes[0].legend(fontsize=7)
    sem = probe_df[probe_df.kind == "semantic"]
    axes[1].plot(range(len(sem)), sem.score, marker="o", label="semantic cluster")
    axes[1].plot(range(len(sem)), sem.shuffled, "k--", label="shuffled")
    axes[1].axhline(sem.chance.iloc[0], color="gray", lw=0.8, label="majority class")
    axes[1].set(title="Semantic cluster: CV accuracy", xticks=range(len(layers)), ylabel="accuracy")
    axes[1].set_xticklabels(layers, rotation=45, ha="right")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_interventions(df, path=None):
    metrics = ["d_top1", "d_cluster_acc", "d_within_cluster_top1"]
    df = df.copy()
    df["kind"] = np.where(df.variant == "leace", "LEACE", "random")
    agg = df.groupby(["layer", "concept", "kind"], sort=False)[metrics].mean().reset_index()
    combos = agg[["layer", "concept"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.45 * len(combos) + 2), sharey=True)
    y = np.arange(len(combos))
    for ax, m in zip(axes, metrics):
        for off, kind, c in [(-0.2, "LEACE", "C0"), (0.2, "random", "gray")]:
            vals = [agg[(agg.layer == l) & (agg.concept == cn) & (agg.kind == kind)][m].iloc[0] for l, cn in combos]
            ax.barh(y + off, vals, height=0.4, color=c, label=kind)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(m.replace("d_", "Δ "))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{l} | {cn}" for l, cn in combos], fontsize=8)
    axes[0].legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
