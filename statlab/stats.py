"""
Statistical tests, confidence intervals and effect sizes.

The test logic (statistics, degrees of freedom, corrections) is written out here;
scipy.stats is only used for the reference distributions (the CDF of t, chi^2, F,
normal), exactly as you would use a printed table.

Every test returns a TestResult with:
    statistic, pvalue, df (if any), and `details` — intermediate values so you
    can follow the calculation step by step.

Example
-------
>>> from statlab import stats
>>> r = stats.ttest_ind([5.1, 4.9, 5.6, 5.8], [6.2, 6.4, 5.9, 6.8])
>>> r.statistic, r.pvalue
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats as _dist  # distributions only: .t, .norm, .chi2, .f, .nct


@dataclass
class TestResult:
    name: str
    statistic: float
    pvalue: float
    df: object = None
    details: dict = field(default_factory=dict)

    def __repr__(self):
        df = f", df={self.df:.4g}" if isinstance(self.df, (int, float)) else (
            f", df={self.df}" if self.df is not None else "")
        return f"{self.name}: statistic={self.statistic:.4f}, p={self.pvalue:.4g}{df}"

    def conclusion(self, alpha=0.05):
        return (f"p = {self.pvalue:.4g} {'<' if self.pvalue < alpha else '>='} α = {alpha} → "
                + ("reject H0" if self.pvalue < alpha else "fail to reject H0"))


def _sample(x, name="sample", min_n=2):
    """Convert to a float array and refuse samples that are too small to test."""
    x = np.asarray(x, float).ravel()
    if len(x) < min_n:
        raise ValueError(f"{name} needs at least {min_n} values (got {len(x)})")
    return x


def _paired(a, b, min_n=2):
    a, b = _sample(a, "first sample", min_n), _sample(b, "second sample", min_n)
    if len(a) != len(b):
        raise ValueError(f"paired samples must have the same length (got {len(a)} and {len(b)})")
    return a, b


def _groups(groups):
    groups = [np.asarray(g, float).ravel() for g in groups]
    if len(groups) < 2:
        raise ValueError("need at least 2 groups")
    for i, g in enumerate(groups):
        if len(g) == 0:
            raise ValueError(f"group {i + 1} is empty")
    return groups


def _p_from_t(t, df, alternative):
    """Turn a t statistic into a p-value for the chosen alternative hypothesis."""
    if alternative == "two-sided":
        return float(2 * _dist.t.sf(abs(t), df))
    if alternative == "greater":
        return float(_dist.t.sf(t, df))
    return float(_dist.t.cdf(t, df))


def _p_from_z(z, alternative):
    if alternative == "two-sided":
        return float(2 * _dist.norm.sf(abs(z)))
    if alternative == "greater":
        return float(_dist.norm.sf(z))
    return float(_dist.norm.cdf(z))


# ===========================================================================
# Descriptive statistics
# ===========================================================================
def describe(x):
    """
    Common summary numbers.
      variance (sample) = Σ (x - mean)^2 / (n - 1)    — n-1: Bessel's correction
      SEM               = s / sqrt(n)                 — uncertainty of the mean
      skewness          = mean(z^3)      (0 for symmetric data)
      excess kurtosis   = mean(z^4) - 3  (0 for a normal distribution)
    """
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    z = (x - m) / x.std(ddof=0)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": n, "mean": m, "std": s, "var": s ** 2, "sem": s / np.sqrt(n),
            "min": x.min(), "q1": q1, "median": med, "q3": q3, "max": x.max(),
            "iqr": q3 - q1, "skewness": float(np.mean(z ** 3)),
            "kurtosis": float(np.mean(z ** 4) - 3)}


def zscore(x, ddof=0):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=ddof)


def rankdata(x):
    """Ranks 1..n; tied values get the average of their ranks.  [10, 20, 20, 30] -> [1, 2.5, 2.5, 4]"""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1  # average rank of the tie group
        i = j + 1
    return ranks


# ===========================================================================
# t-tests and z-test
# ===========================================================================
def ttest_1samp(x, popmean, alternative="two-sided"):
    """
    One-sample t-test.  H0: population mean = popmean
        t = (mean - popmean) / (s / sqrt(n)),   df = n - 1
    """
    x = _sample(x)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    se = s / np.sqrt(n)
    t = (m - popmean) / se
    return TestResult("One-sample t-test", float(t), _p_from_t(t, n - 1, alternative), n - 1,
                      {"n": n, "mean": m, "sd": s, "se": se})


def ttest_ind(a, b, equal_var=True, alternative="two-sided"):
    """
    Independent two-sample t-test.  H0: mean_a = mean_b

    Student (equal_var=True):
        pooled variance  s_p^2 = ((n1-1)s1^2 + (n2-1)s2^2) / (n1 + n2 - 2)
        SE = s_p * sqrt(1/n1 + 1/n2),             df = n1 + n2 - 2
    Welch (equal_var=False, safer when variances differ):
        SE = sqrt(s1^2/n1 + s2^2/n2)
        df = SE^4 / [ (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) ]   (Welch–Satterthwaite)
    t = (mean_a - mean_b) / SE
    """
    a, b = _sample(a, "first sample"), _sample(b, "second sample")
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
        name = "Student's t-test"
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = se ** 4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        name = "Welch's t-test"
    t = (a.mean() - b.mean()) / se
    return TestResult(name, float(t), _p_from_t(t, df, alternative), float(df),
                      {"mean_a": a.mean(), "mean_b": b.mean(), "var_a": v1, "var_b": v2,
                       "se": se, "diff": a.mean() - b.mean()})


def ttest_rel(a, b, alternative="two-sided"):
    """
    Paired t-test = one-sample t-test on the differences d = a - b against 0.
    Use when the same subjects are measured twice (before / after).
    """
    a, b = _paired(a, b)
    r = ttest_1samp(a - b, 0.0, alternative)
    r.name = "Paired t-test"
    r.details["mean_diff"] = r.details.pop("mean")
    return r


def ztest(x, popmean, sigma, alternative="two-sided"):
    """One-sample z-test when the population SD sigma is KNOWN:  z = (mean - mu0) / (sigma/sqrt(n))."""
    x = _sample(x, min_n=1)
    z = (x.mean() - popmean) / (sigma / np.sqrt(len(x)))
    return TestResult("One-sample z-test", float(z), _p_from_z(z, alternative))


def proportions_ztest(count, nobs, value=None, alternative="two-sided"):
    """
    z-test for proportions.
    One sample  (count, nobs scalars):  z = (p̂ - p0) / sqrt(p0(1-p0)/n)
    Two samples (length-2 arrays):      pooled p = (x1+x2)/(n1+n2)
                                        z = (p1 - p2) / sqrt(p(1-p)(1/n1 + 1/n2))
    """
    count, nobs = np.atleast_1d(count).astype(float), np.atleast_1d(nobs).astype(float)
    if len(count) == 1:
        p = count[0] / nobs[0]
        z = (p - value) / np.sqrt(value * (1 - value) / nobs[0])
        det = {"p_hat": p}
    else:
        p1, p2 = count / nobs
        pp = count.sum() / nobs.sum()
        z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / nobs[0] + 1 / nobs[1]))
        det = {"p1": p1, "p2": p2, "pooled_p": pp}
    return TestResult("Proportions z-test", float(z), _p_from_z(z, alternative), details=det)


# ===========================================================================
# ANOVA
# ===========================================================================
def f_oneway(*groups):
    """
    One-way ANOVA.  H0: all group means are equal.

    Split the total variation into two parts:
        SS_between = Σ_g n_g (mean_g - grand_mean)^2     df_b = k - 1
        SS_within  = Σ_g Σ_i (x_gi - mean_g)^2           df_w = N - k
    F = (SS_between / df_b) / (SS_within / df_w) = MS_between / MS_within
    Large F -> group means differ more than noise would explain.
    """
    groups = _groups(groups)
    k, N = len(groups), sum(len(g) for g in groups)
    if N <= k:
        raise ValueError("ANOVA needs more values than groups")
    grand = np.concatenate(groups).mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(np.sum((g - g.mean()) ** 2) for g in groups)
    df_b, df_w = k - 1, N - k
    F = (ss_b / df_b) / (ss_w / df_w)
    return TestResult("One-way ANOVA", float(F), float(_dist.f.sf(F, df_b, df_w)), (df_b, df_w),
                      {"SS_between": ss_b, "SS_within": ss_w, "SS_total": ss_b + ss_w,
                       "MS_between": ss_b / df_b, "MS_within": ss_w / df_w,
                       "eta_squared": ss_b / (ss_b + ss_w)})


def tukey_hsd(*groups):
    """
    Tukey's Honest Significant Difference: all pairwise comparisons after ANOVA.
        q = |mean_i - mean_j| / sqrt( MS_within/2 * (1/n_i + 1/n_j) )
    p-value from the studentized range distribution with k groups, N-k df.
    """
    groups = _groups(groups)
    anova = f_oneway(*groups)
    msw, dfw, k = anova.details["MS_within"], anova.df[1], len(groups)
    out = []
    for i, j in combinations(range(k), 2):
        diff = groups[i].mean() - groups[j].mean()
        se = np.sqrt(msw / 2 * (1 / len(groups[i]) + 1 / len(groups[j])))
        q = abs(diff) / se
        out.append({"pair": (i, j), "diff": diff, "q": q,
                    "pvalue": float(_dist.studentized_range.sf(q, k, dfw))})
    return out


# ===========================================================================
# Chi-square tests
# ===========================================================================
def chisquare(observed, expected=None):
    """
    Goodness of fit.  H0: the observed counts follow the expected distribution.
        χ² = Σ (O - E)^2 / E,      df = k - 1
    expected=None -> all categories equally likely.
    """
    O = _sample(observed, "observed counts")
    E = np.full_like(O, O.sum() / len(O)) if expected is None else np.asarray(expected, float)
    if len(E) != len(O):
        raise ValueError(f"expected has {len(E)} counts but observed has {len(O)}")
    if not np.isclose(E.sum(), O.sum()):
        raise ValueError(f"expected counts sum to {E.sum():g} but observed sum to {O.sum():g}; "
                         "they must be equal (rescale proportions to counts)")
    chi2 = np.sum((O - E) ** 2 / E)
    df = len(O) - 1
    return TestResult("Chi-square goodness of fit", float(chi2), float(_dist.chi2.sf(chi2, df)), df,
                      {"expected": E, "contributions": (O - E) ** 2 / E})


def chi2_contingency(table, correction=True):
    """
    Test of independence for a contingency table.  H0: rows and columns are independent.

    1. expected count of each cell  E_ij = row_total_i * col_total_j / grand_total
    2. χ² = Σ (O - E)^2 / E           df = (rows - 1)(cols - 1)
    3. For 2x2 tables, Yates' continuity correction (default) uses
           Σ (|O - E| - 0.5)^2 / E
    """
    O = np.asarray(table, float)
    if O.ndim != 2 or min(O.shape) < 2:
        raise ValueError("a contingency table needs at least 2 rows and 2 columns")
    if (O.sum(0) == 0).any() or (O.sum(1) == 0).any():
        raise ValueError("a row or column of the table is all zeros (expected count would be 0)")
    E = O.sum(1, keepdims=True) * O.sum(0, keepdims=True) / O.sum()
    df = (O.shape[0] - 1) * (O.shape[1] - 1)
    diff = np.abs(O - E)
    if correction and df == 1:
        diff = np.maximum(diff - 0.5, 0)
    chi2 = np.sum(diff ** 2 / E)
    return TestResult("Chi-square test of independence", float(chi2),
                      float(_dist.chi2.sf(chi2, df)), df, {"expected": E})


def fisher_exact(table):
    """
    Fisher's exact test for a 2x2 table [[a, b], [c, d]] (two-sided).

    With row and column totals fixed, `a` follows a hypergeometric distribution:
        P(a) = C(a+b, a) C(c+d, c) / C(n, a+c)
    p-value = sum of P over every table that is as or less likely than the observed one.
    """
    (a, b), (c, d) = np.asarray(table, int)
    r1, c1, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    support = np.arange(lo, hi + 1)
    probs = _dist.hypergeom.pmf(support, n, r1, c1)
    p_obs = _dist.hypergeom.pmf(a, n, r1, c1)
    p = probs[probs <= p_obs * (1 + 1e-7)].sum()
    odds = (a * d) / (b * c) if b * c else (np.inf if a * d else np.nan)  # 0/0 -> nan
    return TestResult("Fisher's exact test", float(odds), float(min(p, 1.0)),
                      details={"odds_ratio": odds})


# ===========================================================================
# Non-parametric tests
# ===========================================================================
def mannwhitneyu(x, y, alternative="two-sided", use_continuity=True):
    """
    Mann–Whitney U (Wilcoxon rank-sum) test — compares two independent samples
    without assuming normality.

    1. Rank all n1 + n2 values together (ties get average ranks).
    2. R1 = sum of ranks of sample x.
    3. U1 = R1 - n1(n1+1)/2   (number of (x, y) pairs where x wins; ties count 1/2)
    4. Normal approximation:
         mean_U = n1 n2 / 2
         sd_U   = sqrt( n1 n2 / 12 * ( (N+1) - Σ(t^3 - t) / (N(N-1)) ) )   (tie-corrected)
         z = (U1 - mean_U ∓ 0.5) / sd_U
    """
    x, y = _sample(x, "first sample", 1), _sample(y, "second sample", 1)
    n1, n2 = len(x), len(y)
    N = n1 + n2
    ranks = rankdata(np.concatenate([x, y]))
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    _, t = np.unique(ranks, return_counts=True)
    sd = np.sqrt(n1 * n2 / 12 * ((N + 1) - np.sum(t ** 3 - t) / (N * (N - 1))))
    mu = n1 * n2 / 2
    if alternative == "two-sided":
        U = max(U1, U2)
        z = (U - mu - 0.5 * use_continuity) / sd
        p = 2 * _dist.norm.sf(z)
    elif alternative == "greater":
        z = (U1 - mu - 0.5 * use_continuity) / sd
        p = _dist.norm.sf(z)
    else:
        z = (U1 - mu + 0.5 * use_continuity) / sd
        p = _dist.norm.cdf(z)
    return TestResult("Mann-Whitney U test", float(U1), float(min(p, 1.0)),
                      details={"U1": U1, "U2": U2, "R1": R1, "z": z,
                               "common_language_effect": U1 / (n1 * n2)})


def wilcoxon(x, y=None, alternative="two-sided", use_continuity=False):
    """
    Wilcoxon signed-rank test (paired, non-parametric).

    1. d = x - y; drop zeros.
    2. Rank |d|.
    3. W+ = sum of ranks where d > 0,  W- = where d < 0.
    4. Normal approximation:  mean = n(n+1)/4,
                              var  = n(n+1)(2n+1)/24 - Σ(t^3 - t)/48
    Statistic reported: min(W+, W-) for two-sided, like scipy.
    """
    d = _sample(x, min_n=1) if y is None else np.subtract(*_paired(x, y, 1))
    d = d[d != 0]
    n = len(d)
    if n == 0:
        raise ValueError("all differences are zero — nothing to test")
    r = rankdata(np.abs(d))
    w_plus, w_minus = r[d > 0].sum(), r[d < 0].sum()
    _, t = np.unique(r, return_counts=True)
    mu = n * (n + 1) / 4
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 - np.sum(t ** 3 - t) / 48)
    cc = 0.5 * use_continuity
    if alternative == "two-sided":
        T = min(w_plus, w_minus)
        z = (T - mu + cc) / sd
        p = 2 * _dist.norm.cdf(z)
    elif alternative == "greater":
        T = w_plus
        z = (T - mu - cc) / sd
        p = _dist.norm.sf(z)
    else:
        T = w_plus
        z = (T - mu + cc) / sd
        p = _dist.norm.cdf(z)
    return TestResult("Wilcoxon signed-rank test", float(T), float(min(p, 1.0)),
                      details={"W+": w_plus, "W-": w_minus, "n_nonzero": n, "z": z})


def kruskal(*groups):
    """
    Kruskal–Wallis H test — non-parametric one-way ANOVA on ranks.
        H = 12 / (N(N+1)) * Σ R_g^2 / n_g  -  3(N+1)
        tie correction: H / (1 - Σ(t^3 - t) / (N^3 - N))
        df = k - 1, compared to χ²
    """
    groups = _groups(groups)
    allx = np.concatenate(groups)
    N = len(allx)
    r = rankdata(allx)
    H, start = 0.0, 0
    for g in groups:
        H += r[start:start + len(g)].sum() ** 2 / len(g)
        start += len(g)
    H = 12 / (N * (N + 1)) * H - 3 * (N + 1)
    _, t = np.unique(allx, return_counts=True)
    H /= 1 - np.sum(t ** 3 - t) / (N ** 3 - N)
    df = len(groups) - 1
    return TestResult("Kruskal-Wallis H test", float(H), float(_dist.chi2.sf(H, df)), df)


def ks_1samp(x, cdf):
    """
    One-sample Kolmogorov–Smirnov test (asymptotic p-value).
    D = largest vertical gap between the empirical CDF and the theoretical cdf.
    `cdf` is a function, e.g. scipy.stats.norm(0, 1).cdf
    """
    x = np.sort(np.asarray(x, float))
    n = len(x)
    F = cdf(x)
    d_plus = np.max(np.arange(1, n + 1) / n - F)
    d_minus = np.max(F - np.arange(0, n) / n)
    D = max(d_plus, d_minus)
    p = _dist.kstwobign.sf(D * np.sqrt(n))
    return TestResult("Kolmogorov-Smirnov test", float(D), float(min(p, 1.0)))


# ===========================================================================
# Correlation
# ===========================================================================
def pearsonr(x, y):
    """
    Pearson correlation  r = cov(x, y) / (sd_x sd_y)
    Test of H0: rho = 0 uses  t = r sqrt((n-2) / (1-r^2)),  df = n - 2
    """
    x, y = _paired(x, y, 3)
    xc, yc = x - x.mean(), y - y.mean()
    r = np.sum(xc * yc) / np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    r = float(np.clip(r, -1, 1))
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-300))
    return TestResult("Pearson correlation", r, _p_from_t(t, n - 2, "two-sided"), n - 2, {"t": t})


def spearmanr(x, y):
    """Spearman rho = Pearson correlation of the ranks (robust to outliers, monotone trends)."""
    r = pearsonr(rankdata(x), rankdata(y))
    r.name = "Spearman correlation"
    return r


# ===========================================================================
# Confidence intervals
# ===========================================================================
def ci_mean(x, confidence=0.95):
    """mean ± t_{α/2, n-1} * s / sqrt(n)"""
    x = _sample(x)
    n, m, se = len(x), x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    tcrit = _dist.t.ppf(1 - (1 - confidence) / 2, n - 1)
    return (float(m - tcrit * se), float(m + tcrit * se))


def ci_diff_means(a, b, confidence=0.95):
    """Welch CI for mean_a - mean_b."""
    r = ttest_ind(a, b, equal_var=False)
    tcrit = _dist.t.ppf(1 - (1 - confidence) / 2, r.df)
    d, se = r.details["diff"], r.details["se"]
    return (float(d - tcrit * se), float(d + tcrit * se))


def ci_proportion(successes, n, confidence=0.95, method="wilson"):
    """
    Confidence interval for a proportion p̂ = successes / n.
      'wald'   : p̂ ± z sqrt(p̂(1-p̂)/n)               (simple, poor for small n or p near 0/1)
      'wilson' : (p̂ + z²/2n ± z sqrt(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)   (recommended)
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n (got {successes} of {n})")
    p = successes / n
    z = _dist.norm.ppf(1 - (1 - confidence) / 2)
    if method == "wald":
        h = z * np.sqrt(p * (1 - p) / n)
        return (float(p - h), float(p + h))
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return (float(centre - h), float(centre + h))


# ===========================================================================
# Effect sizes
# ===========================================================================
def cohens_d(a, b):
    """
    d = (mean_a - mean_b) / pooled SD
    Rough guide: 0.2 small, 0.5 medium, 0.8 large.
    """
    a, b = _sample(a, "first sample"), _sample(b, "second sample")
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / sp)


def hedges_g(a, b):
    """Cohen's d with small-sample bias correction  J = 1 - 3 / (4(n1+n2) - 9)."""
    n = len(a) + len(b)
    return cohens_d(a, b) * (1 - 3 / (4 * n - 9))


def eta_squared(*groups):
    """η² = SS_between / SS_total — share of variance explained by group membership."""
    return f_oneway(*groups).details["eta_squared"]


def cramers_v(table):
    """V = sqrt( χ² / (n * (min(rows, cols) - 1)) ), 0 = no association, 1 = perfect."""
    O = np.asarray(table, float)
    chi2 = chi2_contingency(O, correction=False).statistic
    return float(np.sqrt(chi2 / (O.sum() * (min(O.shape) - 1))))


def odds_ratio(table):
    """(a d) / (b c) for [[a, b], [c, d]]."""
    (a, b), (c, d) = np.asarray(table, float)
    return float(a * d / (b * c))


# ===========================================================================
# Power analysis
# ===========================================================================
def power_ttest_ind(effect_size, n_per_group, alpha=0.05, alternative="two-sided"):
    """
    Power of a two-sample t-test (equal n): probability of rejecting H0 when
    the true standardised difference is `effect_size` (Cohen's d).

    Under H1 the t statistic follows a NON-central t with
        df = 2n - 2,  noncentrality  λ = d * sqrt(n / 2)
    power = P(|T| > t_crit)
    """
    n = n_per_group
    df = 2 * n - 2
    nc = effect_size * np.sqrt(n / 2)
    if alternative == "two-sided":
        tcrit = _dist.t.ppf(1 - alpha / 2, df)
        return float(_dist.nct.sf(tcrit, df, nc) + _dist.nct.cdf(-tcrit, df, nc))
    tcrit = _dist.t.ppf(1 - alpha, df)
    return float(_dist.nct.sf(tcrit, df, nc))


def sample_size_ttest_ind(effect_size, power=0.8, alpha=0.05):
    """
    Smallest n per group reaching the requested power.
    Power grows with n, so: double n until it is enough, then bisect between
    the last too-small n and the first big-enough n (~2 log2(n) power evaluations
    instead of n).
    """
    if effect_size <= 0:
        raise ValueError("effect size must be positive")
    lo, hi = 1, 2                       # power(lo) < target is assumed, hi is the candidate
    while power_ttest_ind(effect_size, hi, alpha) < power:
        lo, hi = hi, hi * 2
        if hi > 100_000_000:
            raise ValueError("effect size too small")
    while hi - lo > 1:                  # invariant: power(lo) < target <= power(hi)
        mid = (lo + hi) // 2
        if power_ttest_ind(effect_size, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


# ===========================================================================
# Multiple testing
# ===========================================================================
def multipletests(pvalues, alpha=0.05, method="holm"):
    """
    Adjust p-values when running many tests.
      'bonferroni' : p * m
      'holm'       : step-down: sort p ascending, p_(i) * (m - i + 1), keep monotone
      'fdr_bh'     : Benjamini–Hochberg: p_(i) * m / i, monotone from the top
    Returns (reject mask, adjusted p-values) in the original order.
    """
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    if method == "bonferroni":
        adj_sorted = ps * m
    elif method == "holm":
        adj_sorted = np.maximum.accumulate(ps * (m - np.arange(m)))
    elif method == "fdr_bh":
        adj_sorted = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1]
    else:
        raise ValueError(method)
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1)
    return adj <= alpha, adj
