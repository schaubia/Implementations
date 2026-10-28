"""Shared plotting and UI helpers for the Streamlit pages."""
import inspect

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

COLORS = ["#2a6fdb", "#e4572e", "#29a36a", "#9b5de5", "#f2a541"]


def fig_ax(w=6, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def scatter_classes(ax, X, y, alpha=0.8, s=22, marker="o", label_prefix="class ", edge=None):
    for i, c in enumerate(np.unique(y)):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=s, alpha=alpha, marker=marker, color=COLORS[i % 5],
                   edgecolor=edge or "white", linewidth=0.5, label=f"{label_prefix}{c} (n={m.sum()})")


def decision_regions(ax, model, X, steps=200):
    """Colour the plane by the class the model predicts there."""
    pad = 0.6
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, steps), np.linspace(y0, y1, steps))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Zi = np.searchsorted(model.classes_, Z).reshape(xx.shape)  # class label -> 0..k-1
    from matplotlib.colors import ListedColormap
    ax.contourf(xx, yy, Zi, alpha=0.15, cmap=ListedColormap(COLORS[:len(model.classes_)]),
                levels=np.arange(len(model.classes_) + 1) - 0.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def show_source(obj, label="Show the implementation"):
    """Expander with the source code of a class or function from statlab."""
    with st.expander(f"🧾 {label}: `{obj.__name__}`"):
        st.code(inspect.getsource(obj), language="python")


def explain(obj):
    """Render the object's docstring as the step-by-step explanation."""
    doc = inspect.getdoc(obj) or ""
    st.markdown("**How it works**")
    st.code(doc, language="text")


def parse_numbers(text):
    """'1, 2.5 3\n4' -> np.array([1, 2.5, 3, 4])"""
    parts = text.replace(",", " ").replace(";", " ").split()
    return np.array([float(p) for p in parts]) if parts else np.array([])
