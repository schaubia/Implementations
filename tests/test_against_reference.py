"""
Compare statlab against the reference libraries.

Run:  python -m pytest tests/   (needs scikit-learn, scipy, imbalanced-learn installed)
These libraries are ONLY needed for testing — statlab itself does not use them.
"""
import numpy as np
import pytest
from scipy import stats as ss

from statlab import metrics as M, models as ML, stats as S, imbalance as I, validation as V

sk = pytest.importorskip("sklearn")
from sklearn import metrics as skm, linear_model, neighbors, naive_bayes, tree, decomposition  # noqa
from sklearn.datasets import make_classification, make_regression, load_iris  # noqa

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- metrics
def test_classification_metrics():
    y = rng.integers(0, 2, 300)
    p = np.where(rng.random(300) < 0.8, y, 1 - y)
    s = y * 0.6 + rng.random(300)
    assert np.array_equal(M.confusion_matrix(y, p), skm.confusion_matrix(y, p))
    for ours, ref in [(M.precision_score, skm.precision_score),
                      (M.recall_score, skm.recall_score), (M.f1_score, skm.f1_score)]:
        assert np.isclose(ours(y, p), ref(y, p))
    assert np.isclose(M.matthews_corrcoef(y, p), skm.matthews_corrcoef(y, p))
    assert np.isclose(M.roc_auc_score(y, s), skm.roc_auc_score(y, s))
    assert np.isclose(M.average_precision_score(y, s), skm.average_precision_score(y, s))
    assert np.isclose(M.balanced_accuracy_score(y, p), skm.balanced_accuracy_score(y, p))
    s_tied = np.round(s, 1)  # ties in scores
    assert np.isclose(M.roc_auc_score(y, s_tied), skm.roc_auc_score(y, s_tied))


def test_multiclass_averages():
    y = rng.integers(0, 3, 200)
    p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 3, 200))
    for avg in ["macro", "micro", "weighted"]:
        assert np.isclose(M.f1_score(y, p, average=avg), skm.f1_score(y, p, average=avg))
        assert np.isclose(M.precision_score(y, p, average=avg),
                          skm.precision_score(y, p, average=avg))


def test_regression_metrics():
    y, p = rng.normal(size=100), rng.normal(size=100)
    assert np.isclose(M.r2_score(y, p), skm.r2_score(y, p))
    assert np.isclose(M.mean_absolute_error(y, p), skm.mean_absolute_error(y, p))


# ---------------------------------------------------------------- models
def test_linear_and_ridge():
    X, y = make_regression(200, 5, noise=5, random_state=0)
    a, b = ML.LinearRegression().fit(X, y), linear_model.LinearRegression().fit(X, y)
    assert np.allclose(a.coef_, b.coef_) and np.isclose(a.intercept_, b.intercept_)
    a, b = ML.RidgeRegression(3.0).fit(X, y), linear_model.Ridge(3.0).fit(X, y)
    assert np.allclose(a.coef_, b.coef_)


def test_logistic_matches_sklearn_unpenalised():
    X, y = make_classification(300, 4, random_state=1)
    X = ML.StandardScaler().fit_transform(X)
    ours = ML.LogisticRegression(lr=0.5, n_iter=20000, tol=1e-12).fit(X, y)
    ref = linear_model.LogisticRegression(C=1e10, max_iter=5000).fit(X, y)
    assert np.allclose(ours.coef_.ravel(), ref.coef_.ravel(), atol=1e-2)
    assert np.mean(ours.predict(X) == ref.predict(X)) > 0.99


def test_logistic_multiclass():
    X, y = load_iris(return_X_y=True)
    X = ML.StandardScaler().fit_transform(X)
    assert ML.LogisticRegression(lr=0.5, n_iter=3000).fit(X, y).score(X, y) > 0.95


def test_perceptron_separable():
    X = np.r_[rng.normal(-2, 0.5, (50, 2)), rng.normal(2, 0.5, (50, 2))]
    y = np.r_[np.zeros(50), np.ones(50)]
    m = ML.Perceptron().fit(X, y)
    assert m.score(X, y) == 1.0 and m.errors_[-1] == 0


def test_knn_and_nb():
    X, y = make_classification(300, 4, random_state=2)
    for w in ["uniform", "distance"]:
        a = ML.KNeighborsClassifier(5, w).fit(X, y).predict(X[:100])
        b = neighbors.KNeighborsClassifier(5, weights=w).fit(X, y).predict(X[:100])
        assert np.mean(a == b) >= 0.99
    a, b = ML.GaussianNB().fit(X, y), naive_bayes.GaussianNB().fit(X, y)
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_tree_matches_sklearn():
    X, y = load_iris(return_X_y=True)
    for crit in ["gini", "entropy"]:
        a = ML.DecisionTreeClassifier(crit, max_depth=3).fit(X, y)
        b = tree.DecisionTreeClassifier(criterion=crit, max_depth=3, random_state=0).fit(X, y)
        assert np.isclose(a.score(X, y), b.score(X, y))


def test_pca():
    X, _ = make_classification(200, 6, random_state=3)
    a, b = ML.PCA(3).fit(X), decomposition.PCA(3).fit(X)
    assert np.allclose(a.explained_variance_, b.explained_variance_)
    assert np.allclose(np.abs(a.transform(X)), np.abs(b.transform(X)))


# ---------------------------------------------------------------- validation
def test_folds():
    X, y = np.zeros((23, 1)), np.r_[np.zeros(17), np.ones(6)]
    from sklearn.model_selection import KFold, StratifiedKFold
    ours = [t.tolist() for _, t in V.KFold(5).split(X)]
    ref = [t.tolist() for _, t in KFold(5).split(X)]
    assert ours == ref
    for _, test in V.StratifiedKFold(3).split(X, y):
        assert y[test].sum() == 2  # 6 positives spread 2/2/2
    Xtr, Xte, ytr, yte = V.train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    assert len(Xte) + len(Xtr) == 23 and abs(yte.mean() - y.mean()) < 0.1


# ---------------------------------------------------------------- stats
def test_ttests():
    a, b = rng.normal(0, 1, 30), rng.normal(0.5, 2, 25)
    for eq in [True, False]:
        r, ref = S.ttest_ind(a, b, equal_var=eq), ss.ttest_ind(a, b, equal_var=eq)
        assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.pvalue, ref.pvalue)
    for alt in ["less", "greater"]:
        assert np.isclose(S.ttest_ind(a, b, alternative=alt).pvalue,
                          ss.ttest_ind(a, b, alternative=alt).pvalue)
    assert np.isclose(S.ttest_1samp(a, 0.2).pvalue, ss.ttest_1samp(a, 0.2).pvalue)
    assert np.isclose(S.ttest_rel(a[:25], b).pvalue, ss.ttest_rel(a[:25], b).pvalue)


def test_anova_chi2_fisher():
    g = [rng.normal(m, 1, 20) for m in (0, 0.3, 0.8)]
    assert np.isclose(S.f_oneway(*g).pvalue, ss.f_oneway(*g).pvalue)
    tbl = [[12, 5], [7, 14]]
    for corr in [True, False]:
        ref = ss.chi2_contingency(tbl, correction=corr)
        assert np.isclose(S.chi2_contingency(tbl, corr).pvalue, ref.pvalue)
    t3 = [[10, 20, 30], [15, 15, 25]]
    assert np.isclose(S.chi2_contingency(t3).statistic, ss.chi2_contingency(t3).statistic)
    assert np.isclose(S.fisher_exact(tbl).pvalue, ss.fisher_exact(tbl).pvalue)
    assert np.isclose(S.chisquare([18, 22, 30, 30]).pvalue, ss.chisquare([18, 22, 30, 30]).pvalue)
    th = S.tukey_hsd(*g)
    ref = ss.tukey_hsd(*g)
    for row in th:
        i, j = row["pair"]
        assert np.isclose(row["pvalue"], ref.pvalue[i, j], atol=1e-3)


def test_nonparametric():
    x, y = np.round(rng.normal(0, 1, 25), 1), np.round(rng.normal(0.4, 1, 30), 1)  # with ties
    for alt in ["two-sided", "less", "greater"]:
        r = S.mannwhitneyu(x, y, alternative=alt)
        ref = ss.mannwhitneyu(x, y, alternative=alt, method="asymptotic")
        assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.pvalue, ref.pvalue)
    a, b = np.round(rng.normal(0, 1, 30), 1), np.round(rng.normal(0.3, 1, 30), 1)
    r, ref = S.wilcoxon(a, b), ss.wilcoxon(a, b, method="approx")
    assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.pvalue, ref.pvalue)
    g = [np.round(rng.normal(m, 1, 15), 1) for m in (0, 0.5, 1)]
    assert np.isclose(S.kruskal(*g).pvalue, ss.kruskal(*g).pvalue)
    z = rng.normal(size=80)
    assert np.isclose(S.ks_1samp(z, ss.norm.cdf).statistic, ss.kstest(z, "norm").statistic)


def test_correlation_ci_power():
    x = rng.normal(size=50)
    y = x + rng.normal(size=50)
    assert np.isclose(S.pearsonr(x, y).pvalue, ss.pearsonr(x, y).pvalue)
    assert np.isclose(S.spearmanr(x, y).statistic, ss.spearmanr(x, y).statistic)
    lo, hi = S.ci_mean(x)
    ref = ss.t.interval(0.95, 49, x.mean(), ss.sem(x))
    assert np.allclose((lo, hi), ref)
    # classic table value: d=0.5, power 0.8, alpha .05 -> 64 per group
    assert S.sample_size_ttest_ind(0.5) == 64
    p = [0.01, 0.04, 0.03, 0.2]
    assert np.allclose(S.multipletests(p, method="holm")[1], [0.04, 0.09, 0.09, 0.2])
    assert np.allclose(S.multipletests(p, method="fdr_bh")[1], [0.04, 0.0533333, 0.0533333, 0.2])


# ---------------------------------------------------------------- imbalance
imb = pytest.importorskip("imblearn")
from imblearn import under_sampling as us, over_sampling as os_  # noqa


def _data():
    X, y = make_classification(400, 2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                               weights=[0.85], flip_y=0.03, random_state=4)
    return X, y


def test_oversampler_counts():
    X, y = _data()
    for ours, ref in [(I.RandomOverSampler(), os_.RandomOverSampler()),
                      (I.SMOTE(random_state=0), os_.SMOTE(random_state=0)),
                      (I.SMOTE(0.5, random_state=0), os_.SMOTE(sampling_strategy=0.5, random_state=0)),
                      (I.BorderlineSMOTE(random_state=0), os_.BorderlineSMOTE(random_state=0))]:
        _, a = ours.fit_resample(X, y)
        _, b = ref.fit_resample(X, y)
        assert np.array_equal(np.bincount(a), np.bincount(b)), type(ours).__name__
    _, a = I.ADASYN(random_state=0).fit_resample(X, y)
    _, b = os_.ADASYN(random_state=0).fit_resample(X, y)
    assert abs(len(a) - len(b)) <= 3  # rounding of g_i


def test_smote_points_on_segments():
    X, y = _data()
    Xr, yr = I.SMOTE(random_state=0).fit_resample(X, y)
    Xmin = X[y == 1]
    for p in Xr[len(X):][:50]:
        # each synthetic point must lie on a segment between two minority points
        cross2d = lambda u, v: u[0] * v[1] - u[1] * v[0]
        d = np.min([abs(cross2d(b - a, p - a)) / np.linalg.norm(b - a)
                    for a in Xmin for b in Xmin if not np.allclose(a, b)])
        assert d < 1e-8


def test_deterministic_undersamplers_match_exactly():
    X, y = _data()
    pairs = [(I.TomekLinks(), us.TomekLinks()),
             (I.EditedNearestNeighbours(), us.EditedNearestNeighbours()),
             (I.EditedNearestNeighbours(kind_sel="mode"), us.EditedNearestNeighbours(kind_sel="mode")),
             (I.NearMiss(version=1), us.NearMiss(version=1)),
             (I.NearMiss(version=2), us.NearMiss(version=2)),
             (I.NearMiss(version=3), us.NearMiss(version=3))]
    for ours, ref in pairs:
        Xa, ya = ours.fit_resample(X, y)
        Xb, yb = ref.fit_resample(X, y)
        key = lambda Z: Z[np.lexsort(Z.T)]  # compare as sets of rows
        assert len(ya) == len(yb), f"{type(ours).__name__} v{getattr(ours, 'version', '')}"
        assert np.allclose(key(Xa), key(Xb)), type(ours).__name__


def test_under_counts():
    X, y = _data()
    _, a = I.RandomUnderSampler(random_state=0).fit_resample(X, y)
    assert np.bincount(a).tolist() == [min(np.bincount(y))] * 2
    _, a = I.ClusterCentroids(random_state=0).fit_resample(X, y)
    assert np.bincount(a).tolist() == [min(np.bincount(y))] * 2
