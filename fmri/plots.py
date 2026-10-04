from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"fMRI only": "#4c78a8", "with demographics": "#e45756",
          "demographics only": "#9d9d9d", "confounds removed": "#54a24b"}


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def timeseries(bold, meta, out, regions=(0, 1, 2, 3)):
    ids = [int(np.flatnonzero(meta["label"] == c)[0]) for c in (0, 1)]
    names = ["non-heavy drinker", "heavy drinker"]
    fig, axes = plt.subplots(2, 1, figsize=(7, 3.6), sharex=True, sharey=True)
    for ax, i, name in zip(axes, ids, names):
        for r in regions:
            ax.plot(np.arange(bold.shape[1]) * 2.2, bold[i, :, r], lw=0.7)
        ax.set_ylabel("BOLD (z)")
        ax.set_title(f"{meta['subject'][i]} ({name})", fontsize=9, loc="left")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    _save(fig, out, "timeseries.png")


def model_comparison(rows, out):
    """rows: list of (name, group, auc_mean, auc_sd)."""
    rows = sorted(rows, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, (name, group, m, s) in enumerate(rows):
        ax.barh(k, m, xerr=s, color=COLORS[group], capsize=2, height=0.7)
        ax.text(m + s + 0.01, k, f"{m:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlim(0.4, max(r[2] + r[3] for r in rows) + 0.08)
    ax.set_xlabel("ROC-AUC (mean and SD over outer folds)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, COLORS.keys(), fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, out, "model_comparison.png")


def age_distribution(meta, out):
    fig, ax = plt.subplots(figsize=(4.2, 3))
    bins = np.linspace(meta["age"].min(), meta["age"].max(), 18)
    ax.hist(meta.loc[meta["label"] == 0, "age"], bins, alpha=0.6, density=True,
            label="non-heavy")
    ax.hist(meta.loc[meta["label"] == 1, "age"], bins, alpha=0.6, density=True,
            label="heavy")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out, "age_distribution.png")


def permutation(score, perms, p, out):
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.hist(perms, bins=20, color="#9d9d9d")
    ax.axvline(score, color="#e45756", lw=2, label=f"observed {score:.2f} (p={p:.3f})")
    ax.set_xlabel("ROC-AUC with shuffled labels")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out, "permutation_test.png")
