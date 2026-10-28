import numpy as np
import streamlit as st

from app_common import fig_ax, scatter_classes, show_source, explain
from statlab import imbalance as I, datasets, metrics
from statlab.models import LogisticRegression, StandardScaler
from statlab.validation import train_test_split

st.set_page_config(page_title="Imbalanced · StatLab", page_icon="⚖️", layout="wide")
st.title("⚖️ Imbalanced learning (imblearn, from scratch)")

c = st.sidebar
name = c.selectbox("Sampler", list(I.ALL_SAMPLERS))
n = c.slider("Samples", 100, 1000, 300)
minority = c.slider("Minority share", 0.03, 0.4, 0.1)
overlap = c.slider("Class overlap", 0.5, 3.0, 1.2)
seed = c.number_input("Random seed", 0, 9999, 0)

X, y = datasets.make_imbalanced(n, minority, overlap, random_state=seed)

kwargs = {}
if name in ("SMOTE", "BorderlineSMOTE", "SMOTETomek", "SMOTEENN"):
    kwargs["k_neighbors"] = c.slider("k neighbours", 1, 15, 5)
if name == "ADASYN":
    kwargs["n_neighbors"] = c.slider("k neighbours", 1, 15, 5)
if name == "BorderlineSMOTE":
    kwargs["kind"] = c.radio("Kind", ["borderline-1", "borderline-2"])
    kwargs["m_neighbors"] = c.slider("m neighbours (danger test)", 3, 20, 10)
if name == "NearMiss":
    kwargs["version"] = c.radio("Version", [1, 2, 3])
if name == "EditedNearestNeighbours":
    kwargs["n_neighbors"] = c.slider("k neighbours", 1, 10, 3)
    kwargs["kind_sel"] = c.radio("Rule", ["all", "mode"])
if name not in ("TomekLinks", "EditedNearestNeighbours"):
    ratio = c.select_slider("sampling_strategy", ["auto", 0.25, 0.5, 0.75, 1.0], "auto")
    kwargs["sampling_strategy"] = ratio
if name not in ("TomekLinks", "EditedNearestNeighbours", "NearMiss"):
    kwargs["random_state"] = int(seed)

sampler = I.ALL_SAMPLERS[name](**kwargs)
try:
    Xr, yr = sampler.fit_resample(X, y)
except Exception as e:  # e.g. ADASYN with perfectly separated classes
    st.error(str(e))
    st.stop()

before, after = np.bincount(y, minlength=2), np.bincount(yr, minlength=2)
cols = st.columns(4)
cols[0].metric("Class 0 before", before[0])
cols[1].metric("Class 1 before", before[1])
cols[2].metric("Class 0 after", after[0], int(after[0] - before[0]))
cols[3].metric("Class 1 after", after[1], int(after[1] - before[1]))

left, right = st.columns(2)
with left:
    fig, ax = fig_ax(5.5, 4.5)
    scatter_classes(ax, X, y)
    if name == "BorderlineSMOTE":
        d = sampler.danger_.get(1, [])
        ax.scatter(X[d, 0], X[d, 1], s=90, facecolor="none", edgecolor="black",
                   label=f"danger (n={len(d)})")
    if name == "TomekLinks":
        nn = I.nearest_neighbors(X, X, 1, exclude_self=True)[:, 0]
        for i in np.where(I.TomekLinks.is_tomek(X, y))[0]:
            ax.plot(*zip(X[i], X[nn[i]]), color="black", lw=1)
    ax.legend(fontsize=7); ax.set_title("Original")
    st.pyplot(fig)
with right:
    fig, ax = fig_ax(5.5, 4.5)
    # mark which points are new (synthetic) and which were removed
    orig = {tuple(r) for r in np.round(X, 10)}
    res = {tuple(r) for r in np.round(Xr, 10)}
    new = np.array([tuple(r) not in orig for r in np.round(Xr, 10)])
    removed = np.array([tuple(r) not in res for r in np.round(X, 10)])
    scatter_classes(ax, Xr[~new], yr[~new])
    if new.any():
        ax.scatter(Xr[new, 0], Xr[new, 1], s=26, marker="^", color="#29a36a",
                   edgecolor="black", linewidth=0.3, label=f"synthetic (n={new.sum()})")
    if removed.any():
        ax.scatter(X[removed, 0], X[removed, 1], s=30, marker="x", color="gray",
                   label=f"removed (n={removed.sum()})")
    ax.set_xlim(ax.get_xlim()); ax.legend(fontsize=7); ax.set_title(f"After {name}")
    st.pyplot(fig)

# --- effect on a classifier
st.subheader("Does it help a classifier?")
st.caption("Logistic regression trained on the original vs. resampled TRAINING set, evaluated "
           "on the same untouched test set. Resample only the training data — never the test set.")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
rows = []
for label, (A, b) in {"original": (Xtr, ytr), name: sampler.fit_resample(Xtr, ytr)}.items():
    sc = StandardScaler().fit(A)
    m = LogisticRegression(lr=0.5, n_iter=1500).fit(sc.transform(A), b)
    p = m.predict(sc.transform(Xte))
    rows.append({"training data": label,
                 "accuracy": metrics.accuracy_score(yte, p),
                 "balanced acc.": metrics.balanced_accuracy_score(yte, p),
                 "recall (minority)": metrics.recall_score(yte, p),
                 "precision (minority)": metrics.precision_score(yte, p),
                 "F1": metrics.f1_score(yte, p),
                 "G-mean": metrics.geometric_mean_score(yte, p)})
st.dataframe(rows, width="stretch")

explain(I.ALL_SAMPLERS[name])
show_source(I.ALL_SAMPLERS[name])
if name in ("SMOTE", "BorderlineSMOTE", "ADASYN", "SMOTETomek", "SMOTEENN"):
    show_source(I._interpolate, "Core interpolation step")
show_source(I._target_counts, "How sampling_strategy is interpreted")
