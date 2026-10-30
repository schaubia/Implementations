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


# ---------------------------------------------------------------- regressions from the bug review
def test_linear_regression_constant_column():
    X = np.c_[rng.normal(size=50), np.ones(50)]
    m = ML.LinearRegression().fit(X, 3 * X[:, 0] + 1)
    assert np.allclose(m.coef_, [3, 0]) and np.isclose(m.intercept_, 1)


def test_enn_mode_ties_match_imblearn():
    X, y = make_classification(600, 4, n_informative=3, n_redundant=0, n_classes=3,
                               weights=[0.7, 0.2, 0.1], random_state=1)
    for k in (2, 3, 4, 6):
        _, a = I.EditedNearestNeighbours(n_neighbors=k, kind_sel="mode").fit_resample(X, y)
        _, b = us.EditedNearestNeighbours(n_neighbors=k, kind_sel="mode").fit_resample(X, y)
        assert np.array_equal(np.bincount(a), np.bincount(b)), k


def test_ratio_below_current_raises():
    X, y = _data()
    for s in (I.SMOTE(sampling_strategy=0.05), I.RandomUnderSampler(sampling_strategy=0.05)):
        with pytest.raises(ValueError):
            s.fit_resample(X, y)


def test_shrinkage_matches_imblearn_scale():
    X, y = _data()
    a = I.RandomOverSampler(shrinkage=1.0, random_state=0).fit_resample(X, y)[0][len(X):]
    b = os_.RandomOverSampler(shrinkage=1.0, random_state=0).fit_resample(X, y)[0][len(X):]
    assert np.allclose(a.std(0), b.std(0), rtol=0.1)


def test_cluster_centroids_with_duplicates():
    X = np.r_[np.zeros((20, 2)), np.ones((5, 2))]
    y = np.r_[np.zeros(20, int), np.ones(5, int)]
    _, yr = I.ClusterCentroids(random_state=0).fit_resample(X, y)
    assert np.bincount(yr).tolist() == [5, 5]


def test_balanced_accuracy_class_only_in_pred():
    y, p = [0, 0, 1, 1, 0], [0, 2, 1, 1, 2]
    assert np.isclose(M.balanced_accuracy_score(y, p), skm.balanced_accuracy_score(y, p))


def test_metric_guards():
    with pytest.raises(ValueError):
        M.f1_score(["a", "b", "a"], ["a", "b", "b"])          # pos_label=1 not a label
    assert np.isclose(M.f1_score(["a", "b", "a"], ["a", "b", "b"], pos_label="a"), 2 / 3)
    with pytest.raises(ValueError):
        M.roc_auc_score([1, 1, 1], [0.2, 0.5, 0.9])          # one class only
    assert M.r2_score([2, 2, 2], [2, 2, 2]) == 1.0


def test_cross_val_default_is_stratified_for_classifiers():
    X = np.r_[rng.normal(-1, 1, (100, 2)), rng.normal(1, 1, (100, 2))]
    y = np.r_[np.zeros(100, int), np.ones(100, int)]          # sorted by class
    s = V.cross_val_score(ML.LogisticRegression(), X, y, cv=5, scoring="roc_auc")
    assert len(s) == 5 and np.all(np.isfinite(s))            # every fold has both classes


def test_stratified_split_exact_size():
    from sklearn.model_selection import train_test_split as sk_split
    for n, ts, w in [(23, .3, .26), (100, .25, .1), (10, .5, .3), (57, .2, .05)]:
        y = np.r_[np.zeros(int(n * (1 - w))), np.ones(n - int(n * (1 - w)))]
        ours = V.train_test_split(np.zeros((n, 1)), y, test_size=ts, stratify=y, random_state=0)
        ref = sk_split(np.zeros((n, 1)), y, test_size=ts, stratify=y, random_state=0)
        assert len(ours[3]) == len(ref[3]) and ours[3].sum() == ref[3].sum()


def test_stats_input_errors():
    with pytest.raises(ValueError):
        S.ttest_rel([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        S.ttest_1samp([5.0], 1)
    with pytest.raises(ValueError):
        S.f_oneway([1, 2], [])
    with pytest.raises(ValueError):
        S.chisquare([10, 20], [10, 10, 10])
    with pytest.raises(ValueError):
        S.ci_proportion(30, 20)
    with pytest.raises(ValueError):
        S.wilcoxon([1, 2, 3], [1, 2, 3])
    assert np.isnan(S.fisher_exact([[0, 5], [0, 7]]).statistic)


def test_sample_size_fast_and_correct():
    import time
    t = time.time()
    for d, expected in [(0.2, 394), (0.5, 64), (0.8, 26)]:
        assert S.sample_size_ttest_ind(d) == expected
    n = S.sample_size_ttest_ind(0.05, 0.99)
    assert S.power_ttest_ind(0.05, n) >= 0.99 > S.power_ttest_ind(0.05, n - 1)
    assert time.time() - t < 2


# ---------------------------------------------------------------- boosting
from sklearn import ensemble  # noqa: E402
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes  # noqa: E402


def test_weighted_tree_matches_sklearn():
    X, y = load_breast_cancer(return_X_y=True)
    w = rng.uniform(0.1, 3, len(y))
    for d in (1, 3):
        a = ML.DecisionTreeClassifier(max_depth=d).fit(X, y, sample_weight=w)
        b = tree.DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y, sample_weight=w)
        assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_regression_tree_matches_sklearn():
    X, y = load_diabetes(return_X_y=True)
    for d in (2, 4, None):
        a = ML.DecisionTreeRegressor(d).fit(X, y)
        b = tree.DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y)
        assert np.allclose(a.predict(X), b.predict(X))


def test_adaboost_matches_sklearn():
    for X, y in [load_breast_cancer(return_X_y=True), load_wine(return_X_y=True)]:
        for lr in (1.0, 0.5):
            a = ML.AdaBoostClassifier(50, lr).fit(X, y)
            b = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=lr, random_state=0).fit(X, y)
            assert np.allclose(a.estimator_weights_, b.estimator_weights_[:len(a.estimator_weights_)])
            assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_gradient_boosting_matches_sklearn():
    X, y = load_diabetes(return_X_y=True)
    a = ML.GradientBoostingRegressor(100, 0.1, 3).fit(X, y)
    b = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                           random_state=0).fit(X, y)
    assert np.allclose(a.predict(X), b.predict(X))
    for X, y in [load_breast_cancer(return_X_y=True), load_iris(return_X_y=True)]:
        a = ML.GradientBoostingClassifier(50, 0.1, 3).fit(X, y)
        b = ensemble.GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3,
                                                random_state=0).fit(X, y)
        assert np.allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-8)


def test_boosting_staged_scores_improve():
    X, y = make_classification(300, 2, n_redundant=0, n_clusters_per_class=2, random_state=5)
    for m in (ML.AdaBoostClassifier(60), ML.GradientBoostingClassifier(60)):
        s = m.fit(X, y).staged_score(X, y)
        assert len(s) > 1 and s[-1] >= s[0] and np.isclose(s[-1], m.score(X, y))
