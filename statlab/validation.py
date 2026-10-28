"""
Splitting data and cross-validation.

Example
-------
>>> from statlab.validation import StratifiedKFold, cross_val_score
>>> from statlab.models import LogisticRegression
>>> scores = cross_val_score(LogisticRegression(), X, y, cv=StratifiedKFold(5))
"""
import copy
import numpy as np

from . import metrics


def train_test_split(*arrays, test_size=0.25, random_state=None, stratify=None,
                     shuffle=True):
    """
    Split arrays into train and test parts.

    stratify=y keeps the class proportions the same in both parts:
    each class is shuffled and split separately, then the pieces are joined.

    Returns: X_train, X_test, y_train, y_test (for two arrays), etc.
    """
    n = len(arrays[0])
    rng = np.random.default_rng(random_state)
    n_test = int(np.ceil(test_size * n)) if test_size < 1 else int(test_size)

    if stratify is None:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
    else:
        stratify = np.asarray(stratify)
        test_idx = []
        for cls in np.unique(stratify):
            cls_idx = rng.permutation(np.where(stratify == cls)[0])
            k = int(round(len(cls_idx) * n_test / n))  # this class's share of the test set
            test_idx.extend(cls_idx[:k])
        test_idx = np.array(sorted(test_idx))
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if shuffle:
            train_idx, test_idx = rng.permutation(train_idx), rng.permutation(test_idx)

    out = []
    for a in arrays:
        a = np.asarray(a)
        out.extend([a[train_idx], a[test_idx]])
    return out


class KFold:
    """
    Split indices 0..n-1 into k consecutive folds.
    Each fold is the test set once; the other k-1 folds form the training set.

    n=10, k=3  ->  fold sizes 4, 3, 3
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits, self.shuffle, self.random_state = n_splits, shuffle, random_state

    def split(self, X, y=None):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            idx = np.random.default_rng(self.random_state).permutation(n)
        sizes = np.full(self.n_splits, n // self.n_splits)
        sizes[: n % self.n_splits] += 1  # spread the remainder over the first folds
        start = 0
        for size in sizes:
            test = idx[start:start + size]
            train = np.concatenate([idx[:start], idx[start + size:]])
            yield train, test
            start += size


class StratifiedKFold:
    """
    Like KFold, but every fold has (almost) the same class proportions as y.

    How: deal the samples of each class round-robin into the k folds,
    the way you would deal cards to k players.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits, self.shuffle, self.random_state = n_splits, shuffle, random_state

    def split(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        fold_of = np.empty(len(y), dtype=int)
        offset = 0
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if self.shuffle:
                cls_idx = rng.permutation(cls_idx)
            # continue dealing where the previous class stopped, so fold sizes stay even
            fold_of[cls_idx] = (np.arange(len(cls_idx)) + offset) % self.n_splits
            offset += len(cls_idx)
        for k in range(self.n_splits):
            yield np.where(fold_of != k)[0], np.where(fold_of == k)[0]


class LeaveOneOut:
    """n folds, each with exactly one test sample."""

    def split(self, X, y=None):
        n = len(X)
        for i in range(n):
            yield np.r_[0:i, i + 1:n], np.array([i])


def cross_val_score(model, X, y, cv=5, scoring="accuracy"):
    """
    Fit a fresh copy of `model` on each training fold and score it on the test fold.

    scoring: 'accuracy', 'f1', 'balanced_accuracy', 'r2', 'roc_auc'
             or any function f(y_true, y_pred).
    Returns an array with one score per fold.
    """
    X, y = np.asarray(X, float), np.asarray(y)
    if isinstance(cv, int):
        cv = KFold(cv)
    score_fns = {
        "accuracy": metrics.accuracy_score,
        "f1": metrics.f1_score,
        "balanced_accuracy": metrics.balanced_accuracy_score,
        "r2": metrics.r2_score,
    }
    scores = []
    for train, test in cv.split(X, y):
        m = copy.deepcopy(model)
        m.fit(X[train], y[train])
        if scoring == "roc_auc":
            scores.append(metrics.roc_auc_score(y[test], m.predict_proba(X[test])[:, 1]))
        else:
            fn = score_fns[scoring] if isinstance(scoring, str) else scoring
            scores.append(fn(y[test], m.predict(X[test])))
    return np.array(scores)


def bootstrap(statistic, data, n_resamples=2000, confidence=0.95, random_state=None):
    """
    Bootstrap percentile confidence interval for any statistic.

    1. draw n samples WITH replacement from data
    2. compute the statistic
    3. repeat many times -> distribution of the statistic
    4. CI = the (alpha/2, 1 - alpha/2) percentiles of that distribution
    """
    data = np.asarray(data)
    rng = np.random.default_rng(random_state)
    stats = np.array([statistic(data[rng.integers(0, len(data), len(data))])
                      for _ in range(n_resamples)])
    alpha = 1 - confidence
    return {"estimate": statistic(data),
            "ci": (float(np.percentile(stats, 100 * alpha / 2)),
                   float(np.percentile(stats, 100 * (1 - alpha / 2)))),
            "distribution": stats}
