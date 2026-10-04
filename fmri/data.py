from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def make_synthetic(m=715, n_heavy=122, T=269, N=25, seed=0):
    """Simulated stand-in for the NCANDA ICA time series (m subjects x T x N).

    Group differences are deliberately small. Heavy drinkers are slightly older,
    a few regions have a slightly larger amplitude, and a few region pairs are
    slightly more correlated. Age also shifts the amplitude of other regions,
    which is the kind of confounding the evaluation has to cope with.
    """
    rng = np.random.default_rng(seed)

    y = np.zeros(m, dtype=int)
    y[rng.choice(m, n_heavy, replace=False)] = 1
    age = np.clip(rng.normal(17.3, 0.8, m) + 0.6 * y, 16.0, 19.5)
    sex = rng.integers(0, 2, m)
    scanner = rng.integers(0, 2, m)

    load = rng.normal(size=(N, 4))
    cov = load @ load.T + 2.0 * np.eye(N)
    d = np.sqrt(np.diag(cov))
    corr0 = cov / np.outer(d, d)
    corr1 = corr0.copy()
    for i, j in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        corr1[i, j] = corr1[j, i] = corr1[i, j] + 0.03
    chol = [np.linalg.cholesky(c) for c in (corr0, corr1)]

    base_scale = rng.lognormal(0.0, 0.3, N)
    group_regions = np.zeros(N)
    group_regions[:6] = 1.0
    age_regions = np.zeros(N)
    age_regions[6:12] = 1.0

    bold = np.empty((m, T, N), dtype=np.float32)
    for k in range(m):
        e = rng.standard_normal((T, N)) @ chol[y[k]].T
        x = lfilter([1.0], [1.0, -0.5], e, axis=0)
        scale = base_scale * (1 + 0.04 * y[k] * group_regions)
        scale = scale * (1 + 0.05 * (age[k] - 17.3) * age_regions)
        scale = scale * (1.0 + 0.3 * scanner[k]) * rng.lognormal(0.0, 0.2)
        x = x * scale
        x += (rng.random((T, N)) < 0.003) * rng.normal(0, 4, (T, N))
        bold[k] = (x - x.mean()) / x.std()

    meta = pd.DataFrame({
        "subject": [f"S{k:04d}" for k in range(m)],
        "label": y, "age": age.round(2), "sex": sex, "scanner": scanner,
    })
    return bold, meta


def load_dataset(path):
    """Expects bold.npy with shape (subjects, timesteps, regions) and meta.csv
    with columns label, age, sex, scanner (one row per subject, same order)."""
    p = Path(path)
    bold = np.load(p / "bold.npy")
    meta = pd.read_csv(p / "meta.csv")
    missing = {"label", "age", "sex", "scanner"} - set(meta.columns)
    if missing:
        raise ValueError(f"meta.csv is missing columns: {sorted(missing)}")
    if len(meta) != len(bold):
        raise ValueError("bold.npy and meta.csv have different numbers of subjects")
    return bold, meta


def balanced_subset(meta, seed=0):
    """Downsample the majority class, as in the original paper."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(meta["label"] == 1)
    neg = np.flatnonzero(meta["label"] == 0)
    n = min(len(pos), len(neg))
    idx = np.r_[rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
    return np.sort(idx)
