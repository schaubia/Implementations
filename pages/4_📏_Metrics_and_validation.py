import numpy as np
import streamlit as st

from app_common import fig_ax, show_source, explain, COLORS
from statlab import metrics as M, validation as V, datasets
from statlab.models import LogisticRegression, DecisionTreeClassifier

st.set_page_config(page_title="Metrics · StatLab", page_icon="📏", layout="wide")
st.title("📏 Metrics & validation")

tab1, tab2, tab3, tab4 = st.tabs(["Confusion matrix & scores", "ROC / PR curves",
                                  "Cross-validation", "Bootstrap"])

# ---------------------------------------------------------------------------
with tab1:
    st.caption("Set the four cells of a binary confusion matrix and see every score update.")
    c = st.columns(4)
    tp = c[0].number_input("TP", 0, 10_000, 40)
    fn = c[1].number_input("FN", 0, 10_000, 10)
    fp = c[2].number_input("FP", 0, 10_000, 20)
    tn = c[3].number_input("TN", 0, 10_000, 130)
    y_true = np.r_[np.ones(tp + fn), np.zeros(fp + tn)].astype(int)
    y_pred = np.r_[np.ones(tp), np.zeros(fn), np.ones(fp), np.zeros(tn)].astype(int)
    st.table({"": ["true 1", "true 0"], "pred 1": [tp, fp], "pred 0": [fn, tn]})
    scores = {
        "accuracy  (TP+TN)/all": M.accuracy_score(y_true, y_pred),
        "precision TP/(TP+FP)": M.precision_score(y_true, y_pred),
        "recall    TP/(TP+FN)": M.recall_score(y_true, y_pred),
        "specificity TN/(TN+FP)": M.specificity_score(y_true, y_pred),
        "F1": M.f1_score(y_true, y_pred),
        "F2 (recall-heavy)": M.fbeta_score(y_true, y_pred, beta=2),
        "balanced accuracy": M.balanced_accuracy_score(y_true, y_pred),
        "MCC": M.matthews_corrcoef(y_true, y_pred),
        "G-mean": M.geometric_mean_score(y_true, y_pred),
    }
    st.dataframe({"metric": list(scores), "value": np.round(list(scores.values()), 4)},
                 width="stretch")
    st.caption("Try TP=5, FN=45, FP=0, TN=950: accuracy looks great (0.955), recall and MCC "
               "show the model misses most positives.")
    for f in (M.confusion_matrix, M.precision_score, M.fbeta_score, M.matthews_corrcoef):
        show_source(f)

# ---------------------------------------------------------------------------
with tab2:
    sep = st.slider("Class separation", 0.0, 4.0, 1.5)
    n = st.slider("Samples ", 50, 2000, 400)
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.3).astype(int)
    score = rng.normal(0, 1, n) + sep * y
    fpr, tpr, thr = M.roc_curve(y, score)
    prec, rec, _ = M.precision_recall_curve(y, score)
    l, r = st.columns(2)
    with l:
        fig, ax = fig_ax(5, 4)
        ax.plot(fpr, tpr, color=COLORS[0], label=f"AUC = {M.auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(); ax.set_title("ROC")
        st.pyplot(fig)
    with r:
        fig, ax = fig_ax(5, 4)
        ax.step(np.r_[0, rec], np.r_[1, prec], color=COLORS[1],
                label=f"AP = {M.average_precision_score(y, score):.3f}")
        ax.axhline(y.mean(), ls="--", color="gray", label="baseline (prevalence)")
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(); ax.set_title("PR")
        st.pyplot(fig)
    show_source(M.roc_curve)
    show_source(M.average_precision_score)

# ---------------------------------------------------------------------------
with tab3:
    k = st.slider("Folds (k)", 2, 10, 5)
    strat = st.checkbox("Stratified", True)
    X, y = datasets.make_imbalanced(200, 0.15, 1.0)
    cv = V.StratifiedKFold(k, shuffle=True, random_state=0) if strat else V.KFold(k, shuffle=True, random_state=0)
    # visualise which samples are in each test fold
    order = np.argsort(y, kind="stable")
    grid = np.zeros((k, len(y)))
    for i, (_, test) in enumerate(cv.split(X, y)):
        grid[i, np.isin(order, test)] = 1
    fig, ax = fig_ax(8, 0.35 * k + 1)
    ax.imshow(grid, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.axvline((y == 0).sum() - 0.5, color=COLORS[1], lw=2)
    ax.set_yticks(range(k), [f"fold {i + 1}" for i in range(k)])
    ax.set_xlabel("samples sorted by class (red line = class boundary); blue = in test fold")
    st.pyplot(fig)
    st.write("Minority samples per test fold:",
             [int(y[t].sum()) for _, t in cv.split(X, y)])
    # No scaler here on purpose: fitting StandardScaler on all of X before CV would let
    # the test folds leak into training. (These features are already on a similar scale.)
    for label, model in {"Logistic regression": LogisticRegression(lr=0.5, n_iter=800),
                         "Decision tree (depth 3)": DecisionTreeClassifier(max_depth=3)}.items():
        s = V.cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
        st.write(f"**{label}** balanced accuracy per fold: {np.round(s, 3)} → "
                 f"mean {s.mean():.3f} ± {s.std():.3f}")
    show_source(V.KFold)
    show_source(V.StratifiedKFold)
    show_source(V.cross_val_score)

# ---------------------------------------------------------------------------
with tab4:
    stat_name = st.radio("Statistic", ["mean", "median", "std"], horizontal=True)
    rng = np.random.default_rng(1)
    data = rng.lognormal(0, 0.6, st.slider("Sample size", 10, 500, 40))
    fn = {"mean": np.mean, "median": np.median, "std": lambda a: np.std(a, ddof=1)}[stat_name]
    b = V.bootstrap(fn, data, 3000, 0.95, random_state=0)
    fig, ax = fig_ax(6, 3)
    ax.hist(b["distribution"], 50, color=COLORS[0], alpha=0.7)
    for v in b["ci"]:
        ax.axvline(v, color=COLORS[1], ls="--")
    ax.set_title(f"{stat_name} = {b['estimate']:.3f}, 95 % CI = "
                 f"({b['ci'][0]:.3f}, {b['ci'][1]:.3f})")
    st.pyplot(fig)
    explain(V.bootstrap)
    show_source(V.bootstrap)
