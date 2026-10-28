"""
Re-implementations of the main imbalanced-learn (imblearn) samplers.

API (same as imblearn):
    X_res, y_res = Sampler(...).fit_resample(X, y)

Over-sampling   : RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
Under-sampling  : RandomUnderSampler, TomekLinks, EditedNearestNeighbours,
                  NearMiss, ClusterCentroids
Combined        : SMOTETomek, SMOTEENN

sampling_strategy (same meaning as imblearn)
-----------------
'auto'          over-samplers -> every class except the majority is raised to the majority size
                under-samplers -> every class except the minority is lowered to the minority size
float r         binary only: target ratio  n_minority / n_majority = r after resampling
dict            {class: desired number of samples}
'minority', 'majority', 'not minority', 'not majority', 'all'   — which classes to touch

Worked example: y has 90 zeros and 10 ones.
    SMOTE()                       -> 90 zeros, 90 ones  (80 synthetic ones)
    SMOTE(sampling_strategy=0.5)  -> 90 zeros, 45 ones
    RandomUnderSampler()          -> 10 zeros, 10 ones
"""
import numpy as np

from ._utils import check_X_y, nearest_neighbors, pairwise_distances


# ---------------------------------------------------------------------------
# sampling_strategy -> {class: number of samples to add or remove}
# ---------------------------------------------------------------------------
def _target_counts(y, sampling_strategy, kind):
    """
    Translate sampling_strategy into how many samples to ADD (kind='over')
    or how many to KEEP (kind='under') for each class.
    """
    classes, counts = np.unique(y, return_counts=True)
    count = dict(zip(classes, counts))
    maj, mino = classes[np.argmax(counts)], classes[np.argmin(counts)]

    if isinstance(sampling_strategy, dict):
        if kind == "over":
            return {c: n - count[c] for c, n in sampling_strategy.items()}
        return dict(sampling_strategy)

    if isinstance(sampling_strategy, float):
        if len(classes) != 2:
            raise ValueError("a float sampling_strategy only works for binary problems")
        if kind == "over":   # raise minority to r * n_majority
            return {mino: int(sampling_strategy * count[maj]) - count[mino]}
        return {maj: int(count[mino] / sampling_strategy)}  # lower majority

    s = sampling_strategy
    if s == "auto":
        s = "not majority" if kind == "over" else "not minority"
    chosen = {
        "minority": [mino],
        "majority": [maj],
        "not minority": [c for c in classes if c != mino],
        "not majority": [c for c in classes if c != maj],
        "all": list(classes),
    }[s]
    if kind == "over":
        return {c: count[maj] - count[c] for c in chosen}
    return {c: count[mino] for c in chosen}


class _BaseSampler:
    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        return self._fit_resample(X, y)


# ===========================================================================
# Over-sampling
# ===========================================================================
class RandomOverSampler(_BaseSampler):
    """
    Duplicate randomly chosen samples of the smaller classes (with replacement)
    until the target counts are reached.

    shrinkage=None -> exact copies.
    shrinkage=h    -> "smoothed bootstrap": add Gaussian noise with std h * feature std
                      (this is what imblearn's ROSE-like option does).
    """

    def __init__(self, sampling_strategy="auto", shrinkage=None, random_state=None):
        self.sampling_strategy, self.shrinkage, self.random_state = sampling_strategy, shrinkage, random_state

    def _fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_new, y_new = [X], [y]
        for cls, n_add in _target_counts(y, self.sampling_strategy, "over").items():
            if n_add <= 0:
                continue
            idx = rng.choice(np.where(y == cls)[0], size=n_add, replace=True)
            Xc = X[idx].copy()
            if self.shrinkage is not None:
                std = X[y == cls].std(0, ddof=1)
                n_c, d = (y == cls).sum(), X.shape[1]
                bw = (4 / ((d + 2) * n_c)) ** (1 / (d + 4))  # Silverman's rule of thumb
                Xc += rng.standard_normal(Xc.shape) * self.shrinkage * bw * std
            X_new.append(Xc)
            y_new.append(np.full(n_add, cls))
        return np.vstack(X_new), np.concatenate(y_new)


def _interpolate(X_base, X_neigh, rng, step_max=1.0):
    """
    The SMOTE core: a random point on the line segment between two samples.
        x_new = x + u * (neighbour - x),   u ~ Uniform(0, step_max)
    """
    u = rng.uniform(0, step_max, size=(len(X_base), 1))
    return X_base + u * (X_neigh - X_base)


class SMOTE(_BaseSampler):
    """
    Synthetic Minority Over-sampling TEchnique (Chawla et al., 2002).

    For each class to be over-sampled:
      1. find the k nearest neighbours of every sample *within the same class*
      2. repeat n_add times:
           - pick a random sample x of that class
           - pick one of its k neighbours at random, x_nn
           - create  x + u * (x_nn - x),  u uniform in [0, 1]
    New points lie on segments between existing minority points, so the
    minority region is filled in rather than just duplicated.
    """

    def __init__(self, sampling_strategy="auto", k_neighbors=5, random_state=None):
        self.sampling_strategy, self.k_neighbors, self.random_state = sampling_strategy, k_neighbors, random_state

    def _fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_new, y_new = [X], [y]
        for cls, n_add in _target_counts(y, self.sampling_strategy, "over").items():
            if n_add <= 0:
                continue
            Xc = X[y == cls]
            k = min(self.k_neighbors, len(Xc) - 1)
            nn = nearest_neighbors(Xc, Xc, k, exclude_self=True)       # (n_c, k)
            base = rng.integers(0, len(Xc), n_add)                      # step 2a
            neigh = nn[base, rng.integers(0, k, n_add)]                 # step 2b
            X_new.append(_interpolate(Xc[base], Xc[neigh], rng))        # step 2c
            y_new.append(np.full(n_add, cls))
        return np.vstack(X_new), np.concatenate(y_new)


class BorderlineSMOTE(_BaseSampler):
    """
    Borderline-SMOTE (Han et al., 2005): only over-sample minority points near
    the class boundary.

    1. For every minority sample, look at its m nearest neighbours in the WHOLE data.
       Let n_maj = how many of them belong to other classes.
         n_maj == m          -> "noise"   (surrounded by others) — skipped
         m/2 <= n_maj < m    -> "danger"  (on the border)         — used
         n_maj < m/2         -> "safe"                            — skipped
    2. Run SMOTE using only the danger samples as starting points.
         kind='borderline-1': neighbours taken from the same class
         kind='borderline-2': neighbours from any class; if the neighbour is from
                              another class, u is drawn in [0, 0.5] so the new
                              point stays closer to the minority sample.
    """

    def __init__(self, sampling_strategy="auto", k_neighbors=5, m_neighbors=10,
                 kind="borderline-1", random_state=None):
        self.sampling_strategy, self.k_neighbors, self.m_neighbors = sampling_strategy, k_neighbors, m_neighbors
        self.kind, self.random_state = kind, random_state

    def _fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_new, y_new = [X], [y]
        self.danger_ = {}
        for cls, n_add in _target_counts(y, self.sampling_strategy, "over").items():
            if n_add <= 0:
                continue
            idx_c = np.where(y == cls)[0]
            Xc = X[idx_c]
            # step 1: classify each minority sample
            nn_all = nearest_neighbors(Xc, X, self.m_neighbors + 1)[:, 1:]  # drop itself
            n_maj = np.sum(y[nn_all] != cls, axis=1)
            danger = (n_maj >= self.m_neighbors / 2) & (n_maj < self.m_neighbors)
            self.danger_[cls] = idx_c[danger]
            if not danger.any():
                continue
            X_danger = Xc[danger]
            base = rng.integers(0, len(X_danger), n_add)

            if self.kind == "borderline-1":
                k = min(self.k_neighbors, len(Xc) - 1)
                nn = nearest_neighbors(X_danger, Xc, k + 1)[:, 1:]
                neigh = Xc[nn[base, rng.integers(0, k, n_add)]]
                X_syn = _interpolate(X_danger[base], neigh, rng)
            else:  # borderline-2
                nn = nearest_neighbors(X_danger, X, self.k_neighbors + 1)[:, 1:]
                pick = nn[base, rng.integers(0, self.k_neighbors, n_add)]
                same = y[pick] == cls
                u = np.where(same, rng.uniform(0, 1, n_add), rng.uniform(0, 0.5, n_add))[:, None]
                X_syn = X_danger[base] + u * (X[pick] - X_danger[base])
            X_new.append(X_syn)
            y_new.append(np.full(n_add, cls))
        return np.vstack(X_new), np.concatenate(y_new)


class ADASYN(_BaseSampler):
    """
    ADAptive SYNthetic sampling (He et al., 2008): create more synthetic points
    for minority samples that are harder to learn.

    1. For each minority sample x_i, find its k neighbours in the WHOLE data.
       r_i = (number of neighbours from other classes) / k       -> difficulty
    2. Normalise: r_i <- r_i / Σ r
    3. Sample i gets g_i = round(r_i * G) synthetic points, G = total to add.
    4. Each synthetic point: SMOTE interpolation towards a random
       same-class neighbour of x_i.
    """

    def __init__(self, sampling_strategy="auto", n_neighbors=5, random_state=None):
        self.sampling_strategy, self.n_neighbors, self.random_state = sampling_strategy, n_neighbors, random_state

    def _fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_new, y_new = [X], [y]
        for cls, G in _target_counts(y, self.sampling_strategy, "over").items():
            if G <= 0:
                continue
            Xc = X[y == cls]
            nn_all = nearest_neighbors(Xc, X, self.n_neighbors + 1)[:, 1:]
            r = np.sum(y[nn_all] != cls, axis=1) / self.n_neighbors      # step 1
            if r.sum() == 0:
                raise RuntimeError("ADASYN: no minority sample has neighbours from "
                                   "other classes; use SMOTE instead.")
            g = np.rint(r / r.sum() * G).astype(int)                     # steps 2-3
            k = min(self.n_neighbors, len(Xc) - 1)
            nn = nearest_neighbors(Xc, Xc, k, exclude_self=True)
            base = np.repeat(np.arange(len(Xc)), g)
            neigh = nn[base, rng.integers(0, k, len(base))]
            X_new.append(_interpolate(Xc[base], Xc[neigh], rng))          # step 4
            y_new.append(np.full(len(base), cls))
        return np.vstack(X_new), np.concatenate(y_new)


# ===========================================================================
# Under-sampling
# ===========================================================================
class RandomUnderSampler(_BaseSampler):
    """Keep a random subset (without replacement) of each larger class."""

    def __init__(self, sampling_strategy="auto", random_state=None):
        self.sampling_strategy, self.random_state = sampling_strategy, random_state

    def _fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        targets = _target_counts(y, self.sampling_strategy, "under")
        keep = []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if cls in targets:
                idx = rng.choice(idx, size=min(targets[cls], len(idx)), replace=False)
            keep.append(np.sort(idx))
        self.sample_indices_ = np.concatenate(keep)
        return X[self.sample_indices_], y[self.sample_indices_]


class TomekLinks(_BaseSampler):
    """
    Remove Tomek links (Tomek, 1976).

    Two samples a and b form a Tomek link when
      - they belong to different classes, and
      - each is the other's nearest neighbour.
    Such pairs sit on the class boundary (or one of them is noise).
    By default ('auto' = 'not minority') the non-minority member of each
    link is removed, which cleans the boundary.
    """

    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy

    @staticmethod
    def is_tomek(X, y):
        """Boolean mask: True for every sample that is part of a Tomek link."""
        nn = nearest_neighbors(X, X, 1, exclude_self=True)[:, 0]
        mutual = nn[nn] == np.arange(len(X))
        return mutual & (y != y[nn])

    def _fit_resample(self, X, y):
        targets = _target_counts(y, self.sampling_strategy, "under")
        in_link = self.is_tomek(X, y)
        remove = in_link & np.isin(y, list(targets))
        self.sample_indices_ = np.where(~remove)[0]
        return X[~remove], y[~remove]


class EditedNearestNeighbours(_BaseSampler):
    """
    Edited Nearest Neighbours (Wilson, 1972).

    For every sample of the classes being cleaned, look at its k nearest neighbours:
      kind_sel='all'  -> remove it if ANY neighbour has a different class
      kind_sel='mode' -> remove it if the MAJORITY of neighbours disagree
    Removes noisy and borderline points of the larger classes.
    """

    def __init__(self, sampling_strategy="auto", n_neighbors=3, kind_sel="all"):
        self.sampling_strategy, self.n_neighbors, self.kind_sel = sampling_strategy, n_neighbors, kind_sel

    def _fit_resample(self, X, y):
        targets = _target_counts(y, self.sampling_strategy, "under")
        nn = nearest_neighbors(X, X, self.n_neighbors, exclude_self=True)
        agree = y[nn] == y[:, None]
        if self.kind_sel == "all":
            ok = agree.all(1)
        else:  # mode: own class must be the most common among neighbours
            ok = np.array([
                np.sum(agree[i]) >= max(np.sum(y[nn[i]] == c) for c in np.unique(y[nn[i]]))
                for i in range(len(y))])
        keep = ok | ~np.isin(y, list(targets))
        self.sample_indices_ = np.where(keep)[0]
        return X[keep], y[keep]


class NearMiss(_BaseSampler):
    """
    NearMiss (Mani & Zhang, 2003): keep the majority samples that are "closest"
    to the minority class.

    version=1: keep majority samples with the smallest MEAN distance to their
               k nearest minority samples.
    version=2: same, but to their k FARTHEST minority samples.
    version=3: (two steps) first, for each minority sample keep its
               n_neighbors_ver3 nearest majority samples; then among those keep
               the ones with the LARGEST mean distance to their k nearest
               minority samples.
    """

    def __init__(self, sampling_strategy="auto", version=1, n_neighbors=3,
                 n_neighbors_ver3=3):
        self.sampling_strategy, self.version = sampling_strategy, version
        self.n_neighbors, self.n_neighbors_ver3 = n_neighbors, n_neighbors_ver3

    def _fit_resample(self, X, y):
        targets = _target_counts(y, self.sampling_strategy, "under")
        classes, counts = np.unique(y, return_counts=True)
        mino = classes[np.argmin(counts)]
        X_min = X[y == mino]
        keep = [np.where(~np.isin(y, list(targets)))[0]]

        for cls, n_keep in targets.items():
            idx = np.where(y == cls)[0]
            if self.version == 3:
                # step 1: nearest majority neighbours of each minority sample
                nn = nearest_neighbors(X_min, X[idx], min(self.n_neighbors_ver3, len(idx)))
                idx = idx[np.unique(nn)]
            D = np.sort(pairwise_distances(X[idx], X_min), axis=1)
            k = min(self.n_neighbors, D.shape[1])
            mean_d = D[:, -k:].mean(1) if self.version == 2 else D[:, :k].mean(1)
            order = np.argsort(-mean_d if self.version == 3 else mean_d, kind="stable")
            keep.append(np.sort(idx[order[:n_keep]]))
        self.sample_indices_ = np.sort(np.concatenate(keep))
        return X[self.sample_indices_], y[self.sample_indices_]


def kmeans(X, k, n_init=5, max_iter=300, random_state=None):
    """
    Lloyd's k-means with k-means++ initialisation.
      1. choose centres (k-means++: each new centre far from existing ones)
      2. assign every point to its nearest centre
      3. move each centre to the mean of its points
      4. repeat 2-3 until nothing changes
    Returns centres, labels, inertia (sum of squared distances).
    """
    rng = np.random.default_rng(random_state)
    best = None
    for _ in range(n_init):
        C = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(pairwise_distances(X, np.array(C)) ** 2, axis=1)
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(C)
        for _ in range(max_iter):
            labels = np.argmin(pairwise_distances(X, C), axis=1)
            C_new = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                              for j in range(k)])
            if np.allclose(C_new, C):
                break
            C = C_new
        inertia = np.sum((X - C[labels]) ** 2)
        if best is None or inertia < best[2]:
            best = (C, labels, inertia)
    return best


class ClusterCentroids(_BaseSampler):
    """
    Replace each larger class by the centroids of k-means clusters
    (k = target size). The new majority points are synthetic summaries.
    """

    def __init__(self, sampling_strategy="auto", random_state=None):
        self.sampling_strategy, self.random_state = sampling_strategy, random_state

    def _fit_resample(self, X, y):
        targets = _target_counts(y, self.sampling_strategy, "under")
        X_out, y_out = [], []
        for cls in np.unique(y):
            Xc = X[y == cls]
            if cls in targets and targets[cls] < len(Xc):
                Xc = kmeans(Xc, targets[cls], random_state=self.random_state)[0]
            X_out.append(Xc)
            y_out.append(np.full(len(Xc), cls))
        return np.vstack(X_out), np.concatenate(y_out)


# ===========================================================================
# Combinations
# ===========================================================================
class SMOTETomek(_BaseSampler):
    """SMOTE first, then remove Tomek links from ALL classes ('all')."""

    def __init__(self, sampling_strategy="auto", k_neighbors=5, random_state=None):
        self.smote = SMOTE(sampling_strategy, k_neighbors, random_state)
        self.tomek = TomekLinks(sampling_strategy="all")

    def _fit_resample(self, X, y):
        return self.tomek.fit_resample(*self.smote.fit_resample(X, y))


class SMOTEENN(_BaseSampler):
    """SMOTE first, then clean ALL classes with Edited Nearest Neighbours."""

    def __init__(self, sampling_strategy="auto", k_neighbors=5, random_state=None):
        self.smote = SMOTE(sampling_strategy, k_neighbors, random_state)
        self.enn = EditedNearestNeighbours(sampling_strategy="all")

    def _fit_resample(self, X, y):
        return self.enn.fit_resample(*self.smote.fit_resample(X, y))


ALL_SAMPLERS = {
    "RandomOverSampler": RandomOverSampler,
    "SMOTE": SMOTE,
    "BorderlineSMOTE": BorderlineSMOTE,
    "ADASYN": ADASYN,
    "RandomUnderSampler": RandomUnderSampler,
    "TomekLinks": TomekLinks,
    "EditedNearestNeighbours": EditedNearestNeighbours,
    "NearMiss": NearMiss,
    "ClusterCentroids": ClusterCentroids,
    "SMOTETomek": SMOTETomek,
    "SMOTEENN": SMOTEENN,
}
