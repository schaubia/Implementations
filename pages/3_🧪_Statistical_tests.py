import numpy as np
import streamlit as st
from scipy import stats as dist

from app_common import fig_ax, show_source, explain, parse_numbers, COLORS
from statlab import stats as S

st.set_page_config(page_title="Tests · StatLab", page_icon="🧪", layout="wide")
st.title("🧪 Statistical tests & terms")

TESTS = {
    "One-sample t-test": S.ttest_1samp,
    "Two-sample t-test (Student / Welch)": S.ttest_ind,
    "Paired t-test": S.ttest_rel,
    "Mann-Whitney U": S.mannwhitneyu,
    "Wilcoxon signed-rank": S.wilcoxon,
    "One-way ANOVA + Tukey HSD": S.f_oneway,
    "Kruskal-Wallis": S.kruskal,
    "Chi-square goodness of fit": S.chisquare,
    "Chi-square independence / Fisher exact": S.chi2_contingency,
    "Correlation (Pearson / Spearman)": S.pearsonr,
    "Descriptive statistics & CI": S.describe,
    "Power & sample size": S.power_ttest_ind,
    "Multiple-testing correction": S.multipletests,
}
choice = st.sidebar.selectbox("Procedure", list(TESTS))
alpha = st.sidebar.select_slider("Significance level α", [0.001, 0.01, 0.05, 0.1], 0.05)

A_DEFAULT = "5.1 4.9 5.6 5.8 6.0 5.4 5.2 5.9 5.5 5.7"
B_DEFAULT = "6.2 6.4 5.9 6.8 6.1 6.6 5.8 6.9 6.3 6.0"
C_DEFAULT = "5.8 6.0 5.5 6.1 5.9 6.2 5.7 6.4 6.0 5.6"


def two_groups():
    a = parse_numbers(st.text_area("Group A", A_DEFAULT, height=70))
    b = parse_numbers(st.text_area("Group B", B_DEFAULT, height=70))
    return a, b


def report(r):
    c = st.columns(3)
    c[0].metric("Statistic", f"{r.statistic:.4f}")
    c[1].metric("p-value", f"{r.pvalue:.4g}")
    if r.df is not None:
        c[2].metric("df", f"{r.df:.4g}" if np.isscalar(r.df) else str(r.df))
    (st.success if r.pvalue < alpha else st.info)(r.conclusion(alpha))
    if r.details:
        with st.expander("Intermediate values"):
            st.write({k: (np.round(v, 5).tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in r.details.items()})


def t_plot(t, df, alternative="two-sided"):
    """Draw the null t distribution and shade the p-value area."""
    fig, ax = fig_ax(6, 2.8)
    lim = max(4, abs(t) + 1)
    xs = np.linspace(-lim, lim, 400)
    ax.plot(xs, dist.t.pdf(xs, df), color="black", lw=1)
    if alternative == "two-sided":
        mask = np.abs(xs) >= abs(t)
    elif alternative == "greater":
        mask = xs >= t
    else:
        mask = xs <= t
    ax.fill_between(xs, 0, dist.t.pdf(xs, df), where=mask, color=COLORS[1], alpha=0.5,
                    label="p-value area")
    ax.axvline(t, color=COLORS[1], ls="--")
    ax.set_title(f"t distribution with df = {df:.3g} under H0; observed t = {t:.3f}")
    ax.legend()
    st.pyplot(fig)


func = TESTS[choice]

# -----------------------------------------------------------------------
# Any problem with the typed-in data (letters, unequal lengths, empty groups...)
# raises ValueError with a readable message — show it instead of crashing.
try:
    if choice == "One-sample t-test":
        x = parse_numbers(st.text_area("Sample", A_DEFAULT, height=70))
        mu = st.number_input("Hypothesised mean μ0", value=5.0)
        alt = st.radio("Alternative", ["two-sided", "greater", "less"], horizontal=True)
        r = S.ttest_1samp(x, mu, alt)
        report(r)
        st.write(f"95 % CI for the mean: {np.round(S.ci_mean(x), 4)}")
        t_plot(r.statistic, r.df, alt)

    elif choice == "Two-sample t-test (Student / Welch)":
        a, b = two_groups()
        eq = st.radio("Variant", ["Welch (unequal variances)", "Student (equal variances)"],
                      horizontal=True).startswith("Student")
        alt = st.radio("Alternative", ["two-sided", "greater", "less"], horizontal=True)
        func = S.ttest_ind
        r = S.ttest_ind(a, b, equal_var=eq, alternative=alt)
        report(r)
        st.write(f"Cohen's d = {S.cohens_d(a, b):.3f}, Hedges' g = {S.hedges_g(a, b):.3f}, "
                 f"95 % CI for mean difference (Welch) = {np.round(S.ci_diff_means(a, b), 4)}")
        t_plot(r.statistic, r.df, alt)

    elif choice == "Paired t-test":
        st.caption("Same subjects measured twice — values must be in matching order.")
        a, b = two_groups()
        r = S.ttest_rel(a, b)
        report(r)
        t_plot(r.statistic, r.df)
        show_source(S.ttest_1samp, "It calls")

    elif choice == "Mann-Whitney U":
        a, b = two_groups()
        alt = st.radio("Alternative", ["two-sided", "greater", "less"], horizontal=True)
        report(S.mannwhitneyu(a, b, alt))
        show_source(S.rankdata, "Ranking with ties")

    elif choice == "Wilcoxon signed-rank":
        a, b = two_groups()
        report(S.wilcoxon(a, b))

    elif choice in ("One-way ANOVA + Tukey HSD", "Kruskal-Wallis"):
        groups = [parse_numbers(st.text_area(f"Group {g}", d, height=60))
                  for g, d in zip("ABC", [A_DEFAULT, B_DEFAULT, C_DEFAULT])]
        extra = st.text_area("More groups (one per line, optional)", "", height=60)
        groups += [parse_numbers(line) for line in extra.splitlines() if line.strip()]
        r = func(*groups)
        report(r)
        fig, ax = fig_ax(6, 3)
        ax.boxplot(groups, tick_labels=[chr(65 + i) for i in range(len(groups))])
        st.pyplot(fig)
        if func is S.f_oneway:
            st.write("**Tukey HSD — pairwise comparisons**")
            st.dataframe([{"pair": f"{chr(65 + p['pair'][0])} – {chr(65 + p['pair'][1])}",
                           "mean diff": round(p["diff"], 4), "q": round(p["q"], 3),
                           "p adj": round(p["pvalue"], 4)} for p in S.tukey_hsd(*groups)])
            show_source(S.tukey_hsd)

    elif choice == "Chi-square goodness of fit":
        O = parse_numbers(st.text_input("Observed counts", "18 22 30 30"))
        E = st.text_input("Expected counts (empty = uniform)", "")
        r = S.chisquare(O, parse_numbers(E) if E.strip() else None)
        report(r)

    elif choice == "Chi-square independence / Fisher exact":
        st.caption("One row of the table per line.")
        txt = st.text_area("Contingency table", "12 5\n7 14", height=90)
        rows = [parse_numbers(l) for l in txt.splitlines() if l.strip()]
        if len({len(r) for r in rows}) > 1:
            raise ValueError("every row of the table must have the same number of values")
        T = np.array(rows)
        corr = st.checkbox("Yates continuity correction (2×2 only)", True)
        r = S.chi2_contingency(T, corr)
        report(r)
        st.write(f"Cramér's V = {S.cramers_v(T):.3f}")
        if T.shape == (2, 2):
            f = S.fisher_exact(T)
            st.write(f"**Fisher exact**: odds ratio = {f.statistic:.3f}, p = {f.pvalue:.4g}")
            show_source(S.fisher_exact)

    elif choice == "Correlation (Pearson / Spearman)":
        st.caption("Paired values: the i-th x goes with the i-th y.")
        x = parse_numbers(st.text_area("x", A_DEFAULT, height=70))
        y = parse_numbers(st.text_area("y", B_DEFAULT, height=70))
        rp, rs = S.pearsonr(x, y), S.spearmanr(x, y)
        st.write(f"**Pearson** r = {rp.statistic:.4f}, p = {rp.pvalue:.4g}")
        st.write(f"**Spearman** ρ = {rs.statistic:.4f}, p = {rs.pvalue:.4g}")
        fig, ax = fig_ax(5, 3.5)
        ax.scatter(x, y, color=COLORS[0])
        st.pyplot(fig)
        show_source(S.spearmanr)

    elif choice == "Descriptive statistics & CI":
        x = parse_numbers(st.text_area("Data", A_DEFAULT, height=70))
        st.write({k: round(float(v), 5) for k, v in S.describe(x).items()})
        conf = st.slider("Confidence level", 0.80, 0.99, 0.95)
        st.write(f"CI for the mean: {np.round(S.ci_mean(x, conf), 4)}")
        st.markdown("**Proportion CI**")
        k = st.number_input("Successes", 0, 10_000, 7)
        nn = st.number_input("Trials", 1, 10_000, 20)
        st.write({"Wilson": np.round(S.ci_proportion(k, nn, conf), 4).tolist(),
                  "Wald": np.round(S.ci_proportion(k, nn, conf, "wald"), 4).tolist()})
        show_source(S.ci_mean)
        show_source(S.ci_proportion)

    elif choice == "Power & sample size":
        d = st.slider("Effect size (Cohen's d)", 0.05, 2.0, 0.5)
        pw = st.slider("Desired power", 0.5, 0.99, 0.8)
        n_req = S.sample_size_ttest_ind(d, pw, alpha)
        st.metric("Required n per group", n_req)
        ns = np.unique(np.linspace(2, max(3 * n_req, 30), 150).astype(int))
        fig, ax = fig_ax(6, 3)
        ax.plot(ns, [S.power_ttest_ind(d, n, alpha) for n in ns])
        ax.axhline(pw, ls="--", color="gray"); ax.axvline(n_req, ls="--", color=COLORS[1])
        ax.set_xlabel("n per group"); ax.set_ylabel("power")
        st.pyplot(fig)
        show_source(S.sample_size_ttest_ind)

    elif choice == "Multiple-testing correction":
        p = parse_numbers(st.text_input("p-values", "0.01 0.04 0.03 0.2 0.005"))
        rows = {"raw p": p}
        for m in ["bonferroni", "holm", "fdr_bh"]:
            rows[m] = np.round(S.multipletests(p, alpha, m)[1], 4)
        st.dataframe(rows)
except ValueError as e:
    st.error(f"⚠️ {e}")

explain(func)
show_source(func)
