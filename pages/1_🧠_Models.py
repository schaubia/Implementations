import numpy as np
import streamlit as st

from app_common import fig_ax, scatter_classes, decision_regions, show_source, explain
from statlab import models as ML, datasets, metrics
from statlab.validation import train_test_split

st.set_page_config(page_title="Models · StatLab", page_icon="🧠", layout="wide")
st.title("🧠 Classic ML models")

MODEL = st.sidebar.selectbox("Model", [
    "Perceptron", "Logistic regression", "k-nearest neighbours", "Gaussian Naive Bayes",
    "Decision tree", "Linear / Ridge regression", "PCA"])

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
data = c.selectbox("Dataset", ["Two blobs", "Moons", "Three blobs"])
n = c.slider("Samples", 40, 600, 200)
noise = c.slider("Noise / spread", 0.1, 2.5, 0.8 if data != "Moons" else 0.25)
if data == "Two blobs":
    X, y = datasets.make_blobs(n, std=noise)
elif data == "Moons":
    X, y = datasets.make_moons(n, noise=noise)
else:
    X, y = datasets.make_blobs(n, centers=((-2, -1), (2, -1), (0, 2.5)), std=noise)

if MODEL == "Perceptron":
    if len(np.unique(y)) > 2:
        st.warning("The perceptron here is binary — pick a two-class dataset.")
        st.stop()
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

    st.write("Confusion matrix (test):")
    st.dataframe(metrics.confusion_matrix(yte, model.predict(Xte)))

cls = {"Perceptron": ML.Perceptron, "Logistic regression": ML.LogisticRegression,
       "k-nearest neighbours": ML.KNeighborsClassifier, "Gaussian Naive Bayes": ML.GaussianNB,
       "Decision tree": ML.DecisionTreeClassifier}[MODEL]
explain(cls)
show_source(cls)

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
