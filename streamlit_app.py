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

SECTIONS = [
    ("pages/1_🧠_Models.py", "🧠", "Models",
     "Train a model on toy data and watch its decision boundary — from the perceptron to boosting."),
    ("pages/2_⚖️_Imbalanced_learning.py", "⚖️", "Imbalanced learning",
     "See SMOTE & co. add or remove points, and whether it actually helps a classifier."),
    ("pages/3_🧪_Statistical_tests.py", "🧪", "Statistical tests",
     "Paste your own numbers and get the test, the p-value and every intermediate step."),
    ("pages/4_📏_Metrics_and_validation.py", "📏", "Metrics & validation",
     "Play with a confusion matrix, ROC curves, cross-validation folds and the bootstrap."),
]
cols = st.columns(2)
for i, (page, icon, title, blurb) in enumerate(SECTIONS):
    with cols[i % 2].container(border=True):
        st.page_link(page, label=f"**{title}**", icon=icon)
        st.caption(blurb)

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
