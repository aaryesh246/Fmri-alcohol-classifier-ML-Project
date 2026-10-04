import numpy as np


def range_features(bold):
    """Dynamic range (max - min) of each region's time series."""
    return bold.max(axis=1) - bold.min(axis=1)


def connectivity_features(bold):
    """Fisher-z transformed correlation between every pair of regions."""
    x = bold - bold.mean(axis=1, keepdims=True)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    corr = np.einsum("mtn,mtk->mnk", x, x)
    iu = np.triu_indices(bold.shape[2], k=1)
    return np.arctanh(np.clip(corr[:, iu[0], iu[1]], -0.999, 0.999))


def binned_series(bold, size=10):
    """Average the series over non-overlapping windows and flatten."""
    m, T, N = bold.shape
    T2 = (T // size) * size
    x = bold[:, :T2].reshape(m, T2 // size, size, N).mean(axis=2)
    return x.reshape(m, -1)


def demographics(meta):
    return meta[["age", "sex", "scanner"]].to_numpy(dtype=float)
