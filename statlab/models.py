"""
Classic machine-learning models written from scratch with NumPy.

All classifiers have:  fit(X, y), predict(X), score(X, y)
Probabilistic ones add: predict_proba(X)

Contents
--------
Perceptron               — Rosenblatt's online learning rule
LinearRegression         — ordinary least squares (closed form)
RidgeRegression          — least squares with L2 penalty
LogisticRegression       — gradient descent on cross-entropy (+ optional L2)
KNeighborsClassifier     — majority vote of k nearest points
KNeighborsRegressor      — mean of k nearest targets
GaussianNB               — Naive Bayes with normal likelihoods
DecisionTreeClassifier   — CART with Gini or entropy
StandardScaler, PCA      — preprocessing
"""
import numpy as np

from ._utils import check_X_y, pairwise_distances
from . import metrics


class _Classifier:
    def score(self, X, y):
        return metrics.accuracy_score(y, self.predict(X))


class _Regressor:
    def score(self, X, y):
        return metrics.r2_score(y, self.predict(X))


# ===========================================================================
# Perceptron
# ===========================================================================
class Perceptron(_Classifier):
    """
    Binary perceptron (Rosenblatt, 1958).

    Model:  y_hat = sign(w . x + b)        (labels internally -1 / +1)

    Learning rule — for each sample, in order:
        if it is misclassified  (y * (w.x + b) <= 0):
            w <- w + lr * y * x
            b <- b + lr * y
    Repeat for `n_epochs` passes, or stop early when an epoch has no mistakes.

    If the classes are linearly separable, this is guaranteed to converge.

    Attributes after fit:
        coef_, intercept_   learned weights
        errors_             number of mistakes in each epoch (learning curve)
        history_            (w, b) after every epoch, for animation
    """

    def __init__(self, lr=1.0, n_epochs=100, shuffle=True, random_state=0):
        self.lr, self.n_epochs, self.shuffle, self.random_state = lr, n_epochs, shuffle, random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Perceptron here is binary; got classes %s" % self.classes_)
        t = np.where(y == self.classes_[1], 1, -1)  # map labels to -1 / +1
        rng = np.random.default_rng(self.random_state)

        w, b = np.zeros(X.shape[1]), 0.0
        self.errors_, self.history_ = [], []
        for _ in range(self.n_epochs):
            order = rng.permutation(len(X)) if self.shuffle else range(len(X))
            mistakes = 0
            for i in order:
                if t[i] * (X[i] @ w + b) <= 0:   # wrong side (or on the line)
                    w += self.lr * t[i] * X[i]
                    b += self.lr * t[i]
                    mistakes += 1
            self.errors_.append(mistakes)
            self.history_.append((w.copy(), b))
            if mistakes == 0:
                break
        self.coef_, self.intercept_ = w, b
        return self

    def decision_function(self, X):
        return check_X_y(X) @ self.coef_ + self.intercept_

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, self.classes_[1], self.classes_[0])


# ===========================================================================
# Linear models for regression
# ===========================================================================
class LinearRegression(_Regressor):
    """
    Ordinary least squares.

    Minimise  sum (y - Xw - b)^2.
    Add a column of ones to X for the intercept, then solve
        X^T X w = X^T y      (normal equations)
    We use np.linalg.lstsq, which solves the same thing more stably.
    """

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        Xb = np.c_[np.ones(len(X)), X]
        beta, *_ = np.linalg.lstsq(Xb, y.astype(float), rcond=None)
        self.intercept_, self.coef_ = beta[0], beta[1:]
        return self

    def predict(self, X):
        return check_X_y(X) @ self.coef_ + self.intercept_


class RidgeRegression(_Regressor):
    """
    Least squares + L2 penalty:  minimise |y - Xw - b|^2 + alpha |w|^2

    Closed form (after centring X and y so the intercept is not penalised):
        w = (X^T X + alpha I)^-1 X^T y
        b = mean(y) - mean(X) . w
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = y.astype(float)
        x_mean, y_mean = X.mean(0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = y_mean - x_mean @ self.coef_
        return self

    def predict(self, X):
        return check_X_y(X) @ self.coef_ + self.intercept_


# ===========================================================================
# Logistic regression
# ===========================================================================
def sigmoid(z):
    """1 / (1 + e^-z), written to avoid overflow for large |z|."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # subtract max for numerical stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class LogisticRegression(_Classifier):
    """
    Logistic regression trained with full-batch gradient descent.

    Binary case
    -----------
        p = sigmoid(X w + b)
        loss = -mean( y log p + (1-y) log(1-p) ) + (C_inv/2) |w|^2
        gradient:  dL/dw = X^T (p - y) / n + C_inv * w
                   dL/db = mean(p - y)
        update:    w <- w - lr * dL/dw

    Multiclass case: softmax with one weight vector per class (same gradient form).

    Parameters
    ----------
    lr        learning rate
    n_iter    gradient steps
    l2        strength of the L2 penalty (0 = none)
    tol       stop when the loss changes less than this

    Tip: standardise features first (StandardScaler) — gradient descent
    converges much faster.
    """

    def __init__(self, lr=0.1, n_iter=2000, l2=0.0, tol=1e-8):
        self.lr, self.n_iter, self.l2, self.tol = lr, n_iter, l2, tol

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        K = len(self.classes_)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)  # one-hot
        if K == 2:
            Y = Y[:, 1:]  # binary: a single output column
        W = np.zeros((d, Y.shape[1]))
        b = np.zeros(Y.shape[1])

        self.loss_ = []
        for _ in range(self.n_iter):
            P = self._proba_raw(X, W, b)
            err = P - Y                                  # (n, outputs)
            W -= self.lr * (X.T @ err / n + self.l2 * W)
            b -= self.lr * err.mean(0)
            loss = self._loss(P, Y) + 0.5 * self.l2 * np.sum(W ** 2)
            self.loss_.append(loss)
            if len(self.loss_) > 1 and abs(self.loss_[-2] - loss) < self.tol:
                break
        self.coef_, self.intercept_ = W.T, b
        return self

    @staticmethod
    def _proba_raw(X, W, b):
        Z = X @ W + b
        return sigmoid(Z) if W.shape[1] == 1 else softmax(Z)

    @staticmethod
    def _loss(P, Y):
        P = np.clip(P, 1e-15, 1 - 1e-15)
        if Y.shape[1] == 1:
            return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))
        return float(-np.mean(np.sum(Y * np.log(P), axis=1)))

    def predict_proba(self, X):
        P = self._proba_raw(check_X_y(X), self.coef_.T, self.intercept_)
        return np.c_[1 - P, P] if P.shape[1] == 1 else P

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ===========================================================================
# k-nearest neighbours
# ===========================================================================
class KNeighborsClassifier(_Classifier):
    """
    Predict by majority vote among the k closest training points.
    No training step: fit() just stores the data ("lazy learner").

    weights='uniform'  -> each neighbour has one vote
    weights='distance' -> vote weight = 1 / distance (closer = stronger)
    """

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors, self.weights = n_neighbors, weights

    def fit(self, X, y):
        self.X_, self.y_ = check_X_y(X, y)
        self.classes_ = np.unique(self.y_)
        return self

    def predict_proba(self, X):
        D = pairwise_distances(check_X_y(X), self.X_)
        nn = np.argsort(D, axis=1, kind="stable")[:, :self.n_neighbors]
        if self.weights == "distance":
            w = 1 / np.maximum(np.take_along_axis(D, nn, 1), 1e-12)
        else:
            w = np.ones_like(nn, dtype=float)
        labels = self.y_[nn]
        proba = np.stack([(w * (labels == c)).sum(1) for c in self.classes_], axis=1)
        return proba / proba.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNeighborsRegressor(_Regressor):
    """Predict the mean target of the k nearest training points."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_, self.y_ = check_X_y(X, y)
        return self

    def predict(self, X):
        D = pairwise_distances(check_X_y(X), self.X_)
        nn = np.argsort(D, axis=1, kind="stable")[:, :self.n_neighbors]
        return self.y_[nn].astype(float).mean(1)


# ===========================================================================
# Naive Bayes
# ===========================================================================
class GaussianNB(_Classifier):
    """
    Gaussian Naive Bayes.

    Bayes' rule:   P(class | x) ∝ P(class) * P(x | class)
    "Naive":       features are independent given the class, so
                   P(x | class) = Π_j  N(x_j ; mean_cj, var_cj)

    fit: for each class store its prior, and the mean and variance of each feature.
    predict: pick the class with the largest log P(class) + Σ log N(...).
    (Logs turn the product into a sum and avoid underflow.)
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.theta_ = np.array([X[y == c].mean(0) for c in self.classes_])
        eps = self.var_smoothing * X.var(0).max()   # same smoothing as scikit-learn
        self.var_ = np.array([X[y == c].var(0) for c in self.classes_]) + eps
        self.class_prior_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def _joint_log_likelihood(self, X):
        X = check_X_y(X)
        jll = []
        for k in range(len(self.classes_)):
            log_like = -0.5 * np.sum(np.log(2 * np.pi * self.var_[k])
                                     + (X - self.theta_[k]) ** 2 / self.var_[k], axis=1)
            jll.append(np.log(self.class_prior_[k]) + log_like)
        return np.array(jll).T

    def predict_proba(self, X):
        return softmax(self._joint_log_likelihood(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


# ===========================================================================
# Decision tree (CART)
# ===========================================================================
def gini(counts):
    """Gini impurity = 1 - Σ p_k^2.   [5, 5] -> 0.5,  [10, 0] -> 0"""
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)


def entropy(counts):
    """Entropy = -Σ p_k log2 p_k.     [5, 5] -> 1 bit,  [10, 0] -> 0"""
    p = counts[counts > 0] / counts.sum()
    return -np.sum(p * np.log2(p))


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "impurity", "n")

    def __init__(self, value, impurity, n):
        self.feature = self.threshold = self.left = self.right = None
        self.value, self.impurity, self.n = value, impurity, n

    @property
    def is_leaf(self):
        return self.left is None


class DecisionTreeClassifier(_Classifier):
    """
    CART classification tree.

    Growing the tree (recursive):
      1. If the node is pure, too small, or at max_depth -> make a leaf.
      2. Otherwise try every feature and every midpoint between sorted
         distinct values as a threshold.
      3. For each candidate compute the weighted impurity of the children:
             (n_left/n) * I(left) + (n_right/n) * I(right)
      4. Keep the split with the largest impurity decrease and recurse.

    A leaf stores class counts -> predict_proba = counts / total.
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1):
        self.criterion, self.max_depth = criterion, max_depth
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_features_ = X.shape[1]
        self._impurity = gini if self.criterion == "gini" else entropy
        self.feature_importances_ = np.zeros(self.n_features_)
        self.tree_ = self._grow(X, y_enc, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _counts(self, y):
        return np.bincount(y, minlength=len(self.classes_)).astype(float)

    def _grow(self, X, y, depth):
        counts = self._counts(y)
        node = _Node(counts, self._impurity(counts), len(y))
        if (node.impurity == 0 or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        best = self._best_split(X, y, node.impurity)
        if best is None:
            return node
        f, thr, gain = best
        mask = X[:, f] <= thr
        node.feature, node.threshold = f, thr
        self.feature_importances_[f] += gain * len(y)
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X, y, parent_impurity):
        n, K = len(y), len(self.classes_)
        best_gain, best = 1e-12, None
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            # running class counts on the left side as the threshold moves right
            left = np.cumsum(np.eye(K)[ys], axis=0)
            total = left[-1]
            for i in range(self.min_samples_leaf - 1, n - self.min_samples_leaf):
                if xs[i] == xs[i + 1]:
                    continue  # cannot split between equal values
                nl, nr = i + 1, n - i - 1
                imp = (nl * self._impurity(left[i]) + nr * self._impurity(total - left[i])) / n
                gain = parent_impurity - imp
                if gain > best_gain:
                    best_gain, best = gain, (f, (xs[i] + xs[i + 1]) / 2, gain)
        return best

    def _leaf(self, x):
        node = self.tree_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X):
        X = check_X_y(X)
        return np.array([self._leaf(x).value / self._leaf(x).n for x in X])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def export_text(self, feature_names=None):
        """Human-readable tree, like sklearn.tree.export_text."""
        names = feature_names or [f"x{j}" for j in range(self.n_features_)]
        lines = []

        def walk(node, indent):
            pad = "|   " * indent
            if node.is_leaf:
                lines.append(f"{pad}|--- class: {self.classes_[np.argmax(node.value)]} "
                             f"(n={node.n}, counts={node.value.astype(int).tolist()})")
                return
            lines.append(f"{pad}|--- {names[node.feature]} <= {node.threshold:.3f}")
            walk(node.left, indent + 1)
            lines.append(f"{pad}|--- {names[node.feature]} >  {node.threshold:.3f}")
            walk(node.right, indent + 1)

        walk(self.tree_, 0)
        return "\n".join(lines)


# ===========================================================================
# Preprocessing
# ===========================================================================
class StandardScaler:
    """z = (x - mean) / std, per feature. Constant features are left unscaled."""

    def fit(self, X, y=None):
        X = check_X_y(X)
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (check_X_y(X) - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z) * self.scale_ + self.mean_


class PCA:
    """
    Principal component analysis via SVD.

    1. centre X  (subtract column means)
    2. SVD:  Xc = U S V^T
    3. rows of V^T are the principal directions
    4. explained variance of component k = S_k^2 / (n - 1)
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = check_X_y(X)
        self.mean_ = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        # sign convention: largest |loading| in each component is positive
        signs = np.sign(Vt[np.arange(len(Vt)), np.argmax(np.abs(Vt), axis=1)])
        Vt *= signs[:, None]
        var = S ** 2 / (len(X) - 1)
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = var[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / var.sum()
        return self

    def transform(self, X):
        return (check_X_y(X) - self.mean_) @ self.components_.T

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
