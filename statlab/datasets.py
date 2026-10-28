"""Toy datasets generated with NumPy (no scikit-learn needed)."""
import numpy as np


def make_blobs(n=200, centers=((-2, -2), (2, 2)), std=1.0, weights=None, random_state=0):
    """Gaussian clouds around the given centres. weights = class proportions."""
    rng = np.random.default_rng(random_state)
    centers = np.asarray(centers, float)
    k = len(centers)
    weights = np.full(k, 1 / k) if weights is None else np.asarray(weights) / np.sum(weights)
    sizes = np.maximum((weights * n).round().astype(int), 2)
    X = np.vstack([rng.normal(c, std, (s, centers.shape[1])) for c, s in zip(centers, sizes)])
    y = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    return X, y


def make_moons(n=200, noise=0.2, weights=(0.5, 0.5), random_state=0):
    """Two interleaving half circles — not linearly separable."""
    rng = np.random.default_rng(random_state)
    n0 = int(n * weights[0] / sum(weights))
    n1 = n - n0
    t0, t1 = rng.uniform(0, np.pi, n0), rng.uniform(0, np.pi, n1)
    X = np.r_[np.c_[np.cos(t0), np.sin(t0)], np.c_[1 - np.cos(t1), 0.5 - np.sin(t1)]]
    X += rng.normal(0, noise, X.shape)
    return X, np.r_[np.zeros(n0, int), np.ones(n1, int)]


def make_imbalanced(n=300, minority=0.1, overlap=1.0, random_state=0):
    """Binary data with a small minority class; `overlap` controls how much the classes mix."""
    return make_blobs(n, centers=((0, 0), (2.2 / overlap, 2.2 / overlap)), std=1.0,
                      weights=(1 - minority, minority), random_state=random_state)


def make_regression_line(n=60, slope=2.0, intercept=1.0, noise=1.5, random_state=0):
    rng = np.random.default_rng(random_state)
    x = rng.uniform(-3, 3, n)
    return x.reshape(-1, 1), intercept + slope * x + rng.normal(0, noise, n)
