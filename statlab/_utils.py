"""Small helpers shared by the other modules."""
import numpy as np


def check_X_y(X, y=None):
    """Turn inputs into NumPy arrays and check their shapes match."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y is None:
        return X
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} values")
    return X, y


def pairwise_distances(A, B):
    """
    Euclidean distance between every row of A and every row of B.

    Uses the identity  |a - b|^2 = |a|^2 + |b|^2 - 2 a.b
    so it works with matrix products instead of slow Python loops.
    Result shape: (len(A), len(B))
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0.0))  # clip tiny negatives from rounding


def nearest_neighbors(X_query, X_ref, k, exclude_self=False):
    """
    Return indices (into X_ref) of the k nearest neighbours of each query row.

    exclude_self=True is used when X_query IS X_ref: each point's nearest
    neighbour would otherwise be itself (distance 0).
    """
    D = pairwise_distances(X_query, X_ref)
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    # stable argsort -> ties broken by index, deterministic results
    return np.argsort(D, axis=1, kind="stable")[:, :k]
