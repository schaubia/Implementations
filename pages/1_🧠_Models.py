import numpy as np
import streamlit as st

from app_common import fig_ax, scatter_classes, decision_regions, show_source, explain
from statlab import models as ML, datasets, metrics
from statlab.validation import train_test_split

st.set_page_config(page_title="Models · StatLab", page_icon="🧠", layout="wide")
st.title("🧠 Classic ML models")

MODEL = st.sidebar.selectbox("Model", [
    "Perceptron", "Logistic regression", "k-nearest neighbours", "Gaussian Naive Bayes",
    "Decision tree", "AdaBoost", "Gradient boosting", "Linear / Ridge regression",
    "Gradient boosting (regression)", "PCA"])


class _Staged:
    """Wraps a boosting model so decision_regions() draws it after only m rounds."""

    def __init__(self, model, m):
        self.model, self.m, self.classes_ = model, m, model.classes_

    def predict(self, X):
        if isinstance(self.model, ML.AdaBoostClassifier):
            return self.model.staged_predict(X, self.m)
        return self.classes_[np.argmax(self.model.predict_proba(X, n_stages=self.m), axis=1)]


# ------------------------------------------------------------------ gradient boosting, 1-D regression
if MODEL == "Gradient boosting (regression)":
    c = st.sidebar
    n = c.slider("Points", 30, 400, 120)
    noise = c.slider("Noise", 0.0, 1.0, 0.3)
    n_est = c.slider("Number of trees", 1, 300, 100)
    lr = c.slider("Learning rate", 0.01, 1.0, 0.1)
    depth = c.slider("Tree depth", 1, 6, 2)
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 6, n))
    y = np.sin(x) + 0.3 * x + rng.normal(0, noise, n)
    X = x[:, None]
    gb = ML.GradientBoostingRegressor(n_est, lr, depth).fit(X, y)
    xs = np.linspace(0, 6, 400)[:, None]
    stages_grid = [np.full(len(xs), gb.init_)] + list(gb.staged_predict(xs))
    stages_train = [np.full(n, gb.init_)] + list(gb.staged_predict(X))

    st.markdown("Each round fits a small tree to the **residuals** (what the model still gets "
                "wrong) and adds a scaled-down copy of it to the prediction.")
    m = st.slider("Show round m", 1, n_est, min(3, n_est))
    before = stages_train[m - 1]
    left, right = st.columns(2)
    with left:
        fig, ax = fig_ax(5.5, 4)
        ax.scatter(x, y, s=12, color="#2a6fdb", alpha=0.6, label="data")
        ax.plot(xs[:, 0], stages_grid[m - 1], "k--", lw=1, label=f"after {m - 1} trees")
        ax.plot(xs[:, 0], stages_grid[m], color="#e4572e", lw=2, label=f"after {m} trees")
        ax.legend(fontsize=8); ax.set_title("Prediction F(x)")
        st.pyplot(fig)
    with right:
        fig, ax = fig_ax(5.5, 4)
        ax.scatter(x, y - before, s=12, color="gray", alpha=0.7, label="residuals y − F")
        ax.plot(xs[:, 0], gb.estimators_[m - 1].predict(xs), color="#29a36a", lw=2,
                label=f"tree {m} fitted to them")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8); ax.set_title(f"Round {m}: F ← F + {lr} × tree {m}")
        st.pyplot(fig)
    fig, ax = fig_ax(6, 2.6)
    ax.plot(range(1, n_est + 1), gb.train_loss_)
    ax.axvline(m, color="#e4572e", ls="--", lw=1)
    ax.set_xlabel("number of trees"); ax.set_ylabel("training MSE")
    st.pyplot(fig)
    st.caption("Try learning rate 1.0 with deep trees: it fits the noise within a few rounds. "
               "A small rate with many trees takes smaller, safer steps.")
    explain(ML.GradientBoostingRegressor)
    show_source(ML.GradientBoostingRegressor)
    show_source(ML.DecisionTreeRegressor, "The tree fitted in every round")
    st.stop()

# ------------------------------------------------------------------ regression
if MODEL == "Linear / Ridge regression":
    c = st.sidebar
    noise = c.slider("Noise", 0.0, 5.0, 1.5)
    n = c.slider("Points", 10, 300, 60)
    alpha = c.slider("Ridge α (0 = plain least squares)", 0.0, 200.0, 0.0)
    X, y = datasets.make_regression_line(n, noise=noise)
    m = ML.RidgeRegression(alpha).fit(X, y) if alpha > 0 else ML.LinearRegression().fit(X, y)
    left, right = st.columns([3, 2])
    with left:
        fig, ax = fig_ax()
        ax.scatter(X[:, 0], y, s=20, color="#2a6fdb", alpha=0.8)
        xs = np.linspace(X.min(), X.max(), 50)
        ax.plot(xs, m.predict(xs[:, None]), color="#e4572e", lw=2,
                label=f"ŷ = {m.intercept_:.2f} + {m.coef_[0]:.2f}·x")
        ax.plot(xs, 1 + 2 * xs, "k--", lw=1, label="true: 1 + 2·x")
        for xi, yi in zip(X[:15, 0], y[:15]):  # residuals for a few points
            ax.plot([xi, xi], [yi, m.predict([[xi]])[0]], color="gray", lw=0.6)
        ax.legend()
        st.pyplot(fig)
    with right:
        st.metric("R²", f"{m.score(X, y):.3f}")
        st.metric("RMSE", f"{metrics.root_mean_squared_error(y, m.predict(X)):.3f}")
        st.caption("Grey lines are residuals — least squares minimises the sum of their squares. "
                   "Ridge shrinks the slope towards 0 as α grows.")
    explain(ML.RidgeRegression if alpha > 0 else ML.LinearRegression)
    show_source(ML.LinearRegression)
    show_source(ML.RidgeRegression)
    st.stop()

# ------------------------------------------------------------------ PCA
if MODEL == "PCA":
    rng = np.random.default_rng(0)
    corr = st.sidebar.slider("Correlation between x and y", 0.0, 0.99, 0.85)
    Z = rng.multivariate_normal([0, 0], [[1, corr], [corr, 1]], 300)
    p = ML.PCA(2).fit(Z)
    fig, ax = fig_ax(5, 5)
    ax.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.5, color="#2a6fdb")
    for comp, var, col in zip(p.components_, p.explained_variance_, ["#e4572e", "#29a36a"]):
        ax.arrow(0, 0, *(comp * 2 * np.sqrt(var)), width=0.03, color=col)
    ax.set_aspect("equal")
    st.pyplot(fig)
    st.write("Explained variance ratio:", np.round(p.explained_variance_ratio_, 3))
    explain(ML.PCA)
    show_source(ML.PCA)
    st.stop()

# ------------------------------------------------------------------ classifiers
c = st.sidebar
DATASETS = ["Two blobs", "Moons", "Three blobs"]
if MODEL == "Perceptron":            # binary model -> only two-class datasets
    DATASETS = DATASETS[:2]
data = c.selectbox("Dataset", DATASETS)
n = c.slider("Samples", 40, 600, 200)
noise = c.slider("Noise / spread", 0.1, 2.5, 0.8 if data != "Moons" else 0.25)
if data == "Two blobs":
    X, y = datasets.make_blobs(n, std=noise)
elif data == "Moons":
    X, y = datasets.make_moons(n, noise=noise)
else:
    X, y = datasets.make_blobs(n, centers=((-2, -1), (2, -1), (0, 2.5)), std=noise)

if MODEL == "Perceptron":
    lr = c.slider("Learning rate", 0.01, 2.0, 1.0)
    epochs = c.slider("Max epochs", 1, 200, 50)
    model = ML.Perceptron(lr=lr, n_epochs=epochs)
elif MODEL == "Logistic regression":
    lr = c.slider("Learning rate", 0.01, 2.0, 0.3)
    l2 = c.slider("L2 penalty", 0.0, 1.0, 0.0)
    model = ML.LogisticRegression(lr=lr, n_iter=2000, l2=l2)
elif MODEL == "k-nearest neighbours":
    model = ML.KNeighborsClassifier(c.slider("k", 1, 50, 5), c.radio("Weights", ["uniform", "distance"]))
elif MODEL == "Gaussian Naive Bayes":
    model = ML.GaussianNB()
elif MODEL == "AdaBoost":
    model = ML.AdaBoostClassifier(c.slider("Number of stumps", 1, 300, 50),
                                  c.slider("Learning rate", 0.05, 2.0, 1.0),
                                  max_depth=c.slider("Depth of each weak tree", 1, 4, 1))
elif MODEL == "Gradient boosting":
    model = ML.GradientBoostingClassifier(c.slider("Number of rounds", 1, 300, 100),
                                          c.slider("Learning rate", 0.01, 1.0, 0.1),
                                          max_depth=c.slider("Tree depth", 1, 6, 3))
else:
    model = ML.DecisionTreeClassifier(c.radio("Criterion", ["gini", "entropy"]),
                                      max_depth=c.slider("Max depth", 1, 12, 3),
                                      min_samples_leaf=c.slider("Min samples per leaf", 1, 30, 1))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
model.fit(Xtr, ytr)

left, right = st.columns([3, 2])
with left:
    fig, ax = fig_ax()
    decision_regions(ax, model, X)
    scatter_classes(ax, Xtr, ytr)
    scatter_classes(ax, Xte, yte, marker="X", s=40, label_prefix="test ", edge="black")
    ax.legend(fontsize=7, loc="best")
    ax.set_title("Decision regions (● train, ✕ test)")
    st.pyplot(fig)
with right:
    st.metric("Train accuracy", f"{model.score(Xtr, ytr):.3f}")
    st.metric("Test accuracy", f"{model.score(Xte, yte):.3f}")
    if MODEL == "Perceptron":
        fig2, ax2 = fig_ax(5, 2.6)
        ax2.plot(range(1, len(model.errors_) + 1), model.errors_, marker="o", ms=3)
        ax2.set_xlabel("epoch"); ax2.set_ylabel("mistakes")
        st.pyplot(fig2)
        st.caption("On separable data the mistakes drop to 0 and training stops. "
                   "On overlapping data they never do — the perceptron keeps oscillating.")
    elif MODEL == "Logistic regression":
        fig2, ax2 = fig_ax(5, 2.6)
        ax2.plot(model.loss_)
        ax2.set_xlabel("iteration"); ax2.set_ylabel("cross-entropy loss")
        st.pyplot(fig2)
    elif MODEL == "Decision tree":
        st.code(model.export_text(["x0", "x1"]), language="text")
    elif MODEL == "Gaussian Naive Bayes":
        st.write("Per-class feature means:", np.round(model.theta_, 2))
        st.write("Per-class feature variances:", np.round(model.var_, 2))
    elif MODEL in ("AdaBoost", "Gradient boosting"):
        tr, te = model.staged_score(Xtr, ytr), model.staged_score(Xte, yte)
        fig2, ax2 = fig_ax(5, 2.6)
        ax2.plot(range(1, len(tr) + 1), tr, label="train")
        ax2.plot(range(1, len(te) + 1), te, label="test")
        ax2.set_xlabel("boosting rounds"); ax2.set_ylabel("accuracy"); ax2.legend(fontsize=8)
        st.pyplot(fig2)
        st.caption("Train accuracy keeps climbing; when test accuracy stops following, "
                   "extra rounds only memorise the training data.")

    st.write("Confusion matrix (test):")
    st.dataframe(metrics.confusion_matrix(yte, model.predict(Xte)))

cls = {"Perceptron": ML.Perceptron, "Logistic regression": ML.LogisticRegression,
       "k-nearest neighbours": ML.KNeighborsClassifier, "Gaussian Naive Bayes": ML.GaussianNB,
       "Decision tree": ML.DecisionTreeClassifier, "AdaBoost": ML.AdaBoostClassifier,
       "Gradient boosting": ML.GradientBoostingClassifier}[MODEL]
explain(cls)
show_source(cls)
if MODEL == "Gradient boosting":
    show_source(ML.DecisionTreeRegressor, "The tree fitted in every round")

if MODEL in ("AdaBoost", "Gradient boosting"):
    st.subheader("Step through the boosting rounds")
    n_rounds = len(model.estimators_)
    m = st.slider("Round", 1, n_rounds, min(1, n_rounds)) if n_rounds > 1 else 1
    if MODEL == "AdaBoost":
        l3, r3 = st.columns(2)
        with r3:
            # the weak learner of round m and the weights it was trained on
            w = model.weight_history_[m - 1]
            fig3, ax3 = fig_ax(5.5, 4.5)
            decision_regions(ax3, model.estimators_[m - 1], X)
            for i, cl in enumerate(model.classes_):
                k = ytr == cl
                ax3.scatter(Xtr[k, 0], Xtr[k, 1], s=8 + 3000 * w[k], alpha=0.7,
                            color=["#2a6fdb", "#e4572e", "#29a36a"][i], edgecolor="white", linewidth=0.5)
            ax3.set_title(f"Stump {m}: error {model.estimator_errors_[m - 1]:.3f}, "
                          f"say α = {model.estimator_weights_[m - 1]:.2f}")
            st.pyplot(fig3)
            st.caption("Point size = the weight this stump was trained on. "
                       "Points the earlier stumps got wrong grow bigger.")
        target = l3
    else:
        target = st.container()
    with target:
        fig4, ax4 = fig_ax(5.5, 4.5)
        decision_regions(ax4, _Staged(model, m), X)
        scatter_classes(ax4, Xtr, ytr, s=14)
        ax4.set_title(f"Whole ensemble after {m} round{'s' if m > 1 else ''}")
        st.pyplot(fig4)

if MODEL == "Perceptron":
    st.subheader("Step through training")
    ep = st.slider("Epoch", 1, len(model.history_), len(model.history_))
    w, b = model.history_[ep - 1]
    fig3, ax3 = fig_ax()
    scatter_classes(ax3, Xtr, ytr)
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
    if abs(w[1]) > 1e-12:
        ax3.plot(xs, -(w[0] * xs + b) / w[1], "k-", lw=2, label="w·x + b = 0")
    ax3.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax3.set_title(f"after epoch {ep}: w = {np.round(w, 2)}, b = {b:.2f}")
    ax3.legend(fontsize=7)
    st.pyplot(fig3)
