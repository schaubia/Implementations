"""
StatLab — from-scratch statistics & ML.
Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="StatLab", page_icon="📐", layout="wide")

st.title("📐 StatLab")
st.markdown(
    "Readable, from-scratch implementations of important statistics and machine-learning "
    "procedures — built only on **NumPy** and **SciPy** (SciPy is used just for reference "
    "distributions like t, χ² and F)."
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("🧠 Classic ML models")
    st.markdown("Perceptron · Linear & Ridge regression · Logistic regression · k-NN · "
                "Gaussian Naive Bayes · Decision tree (CART) · StandardScaler · PCA")
    st.subheader("⚖️ Imbalanced learning")
    st.markdown("RandomOverSampler · SMOTE · Borderline-SMOTE · ADASYN · RandomUnderSampler · "
                "Tomek links · Edited Nearest Neighbours · NearMiss 1/2/3 · ClusterCentroids · "
                "SMOTETomek · SMOTEENN")
with c2:
    st.subheader("🧪 Statistical tests & terms")
    st.markdown("t-tests (one-sample, Student, Welch, paired) · z-tests · ANOVA + Tukey HSD · "
                "χ² (goodness of fit, independence) · Fisher exact · Mann-Whitney U · Wilcoxon · "
                "Kruskal-Wallis · KS · correlations · confidence intervals · effect sizes · "
                "power & sample size · multiple-testing corrections")
    st.subheader("📏 Metrics & validation")
    st.markdown("Confusion matrix · precision / recall / F1 / MCC · ROC & PR curves · AUC · "
                "train/test split · KFold · StratifiedKFold · cross-validation · bootstrap")

st.divider()
st.markdown("**Use the code without the app** — copy the `statlab/` folder next to your script:")
st.code(
    """from statlab.imbalance import SMOTE
from statlab.models import LogisticRegression, StandardScaler
from statlab import metrics, stats

X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
model = LogisticRegression().fit(StandardScaler().fit_transform(X_res), y_res)

stats.ttest_ind(group_a, group_b, equal_var=False)   # Welch's t-test
""", language="python")
st.caption("Every page has a '🧾 Show the implementation' panel with the exact source code used.")
