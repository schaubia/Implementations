"""
Classification and regression metrics, implemented step by step.

Worked example used throughout the docstrings:
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1, 0, 1]
    -> TP=3, FN=1, FP=1, TN=2
"""
import numpy as np


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, labels=None):
    """
    Count how often each true class was predicted as each class.

    Rows = true class, columns = predicted class (same as scikit-learn).

    For binary labels [0, 1]:
                    pred 0   pred 1
        true 0   [[  TN  ,   FP  ],
        true 1    [  FN  ,   TP  ]]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm


def _binary_counts(y_true, y_pred, pos_label=1):
    """TP, FP, FN, TN treating `pos_label` as the positive class."""
    y_true = np.asarray(y_true) == pos_label
    y_pred = np.asarray(y_pred) == pos_label
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    return tp, fp, fn, tn


def _safe_div(a, b):
    return a / b if b else 0.0


def _check_pos_label(y_true, y_pred, pos_label):
    """With labels like 'yes'/'no', the default pos_label=1 would silently score 0."""
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    if len(labels) >= 2 and pos_label not in labels:
        raise ValueError(f"pos_label={pos_label!r} is not one of the labels {labels.tolist()}; "
                         f"pass pos_label=... or use average='macro'.")


# ---------------------------------------------------------------------------
# Basic scores
# ---------------------------------------------------------------------------
def accuracy_score(y_true, y_pred):
    """Fraction of correct predictions.  Example: 5 correct of 7 -> 0.714"""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _per_class(y_true, y_pred, func, average, pos_label):
    """
    Apply a binary metric per class and combine.

    average='binary'   -> only the positive class
    average='macro'    -> plain mean over classes (every class counts equally)
    average='weighted' -> mean weighted by class size (support)
    average='micro'    -> pool all TP/FP/FN first, then compute once
    average=None       -> return one value per class
    """
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    if average == "binary":
        _check_pos_label(y_true, y_pred, pos_label)
        return func(*_binary_counts(y_true, y_pred, pos_label))
    counts = [_binary_counts(y_true, y_pred, lab) for lab in labels]
    if average == "micro":
        tp, fp, fn, tn = (sum(c[i] for c in counts) for i in range(4))
        return func(tp, fp, fn, tn)
    scores = np.array([func(*c) for c in counts])
    if average is None:
        return scores
    if average == "macro":
        return float(scores.mean())
    if average == "weighted":
        support = np.array([np.sum(np.asarray(y_true) == lab) for lab in labels])
        return float(np.sum(scores * support) / support.sum())
    raise ValueError(f"unknown average={average!r}")


def precision_score(y_true, y_pred, average="binary", pos_label=1):
    """
    Precision = TP / (TP + FP)
    "Of everything I called positive, how much really was?"
    Example: 3 / (3 + 1) = 0.75
    """
    return _per_class(y_true, y_pred, lambda tp, fp, fn, tn: _safe_div(tp, tp + fp),
                      average, pos_label)


def recall_score(y_true, y_pred, average="binary", pos_label=1):
    """
    Recall (sensitivity, TPR) = TP / (TP + FN)
    "Of all real positives, how many did I find?"
    Example: 3 / (3 + 1) = 0.75
    """
    return _per_class(y_true, y_pred, lambda tp, fp, fn, tn: _safe_div(tp, tp + fn),
                      average, pos_label)


def specificity_score(y_true, y_pred, pos_label=1):
    """
    Specificity (TNR) = TN / (TN + FP)
    Example: 2 / (2 + 1) = 0.667
    """
    _check_pos_label(y_true, y_pred, pos_label)
    tp, fp, fn, tn = _binary_counts(y_true, y_pred, pos_label)
    return _safe_div(tn, tn + fp)


def fbeta_score(y_true, y_pred, beta=1.0, average="binary", pos_label=1):
    """
    F-beta = (1 + b^2) * P * R / (b^2 * P + R)
           = (1 + b^2) TP / ((1 + b^2) TP + b^2 FN + FP)
    beta > 1 weighs recall more, beta < 1 weighs precision more.
    """
    b2 = beta ** 2

    def f(tp, fp, fn, tn):
        return _safe_div((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp)

    return _per_class(y_true, y_pred, f, average, pos_label)


def f1_score(y_true, y_pred, average="binary", pos_label=1):
    """F1 = harmonic mean of precision and recall.  Example: 0.75"""
    return fbeta_score(y_true, y_pred, 1.0, average, pos_label)


def balanced_accuracy_score(y_true, y_pred):
    """
    Mean recall over the classes that occur in y_true — robust to class imbalance.
    (A class that appears only in y_pred has no recall, so it is left out, as in scikit-learn.)
    """
    y_true = np.asarray(y_true)
    return float(np.mean([np.mean(np.asarray(y_pred)[y_true == c] == c)
                          for c in np.unique(y_true)]))


def matthews_corrcoef(y_true, y_pred, pos_label=1):
    """
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    Ranges -1 .. +1; 0 = no better than chance. Good for imbalanced data.
    """
    _check_pos_label(y_true, y_pred, pos_label)
    tp, fp, fn, tn = _binary_counts(y_true, y_pred, pos_label)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return _safe_div(tp * tn - fp * fn, den)


def geometric_mean_score(y_true, y_pred):
    """G-mean = (product of per-class recalls)^(1/n_classes). Used in imbalanced-learn."""
    r = recall_score(y_true, y_pred, average=None)
    return float(np.prod(r) ** (1 / len(r)))


def classification_report(y_true, y_pred):
    """Return a dict {class: {precision, recall, f1, support}} plus averages."""
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    p = precision_score(y_true, y_pred, average=None)
    r = recall_score(y_true, y_pred, average=None)
    f = f1_score(y_true, y_pred, average=None)
    report = {}
    for i, lab in enumerate(labels):
        report[str(lab)] = {"precision": p[i], "recall": r[i], "f1": f[i],
                            "support": int(np.sum(np.asarray(y_true) == lab))}
    report["accuracy"] = accuracy_score(y_true, y_pred)
    report["macro avg"] = {"precision": p.mean(), "recall": r.mean(), "f1": f.mean()}
    return report


# ---------------------------------------------------------------------------
# Threshold-free metrics (use scores / probabilities, not hard labels)
# ---------------------------------------------------------------------------
def roc_curve(y_true, y_score, pos_label=1):
    """
    Receiver Operating Characteristic curve.

    Steps:
      1. sort samples by score, highest first
      2. walk down the list; at each distinct score, everything above
         the cut is predicted positive
      3. record FPR = FP / N_neg and TPR = TP / N_pos at each cut

    Returns fpr, tpr, thresholds (starting at (0, 0) with threshold +inf).
    """
    y_true = np.asarray(y_true) == pos_label
    if y_true.all() or not y_true.any():
        raise ValueError("ROC needs both positive and negative samples in y_true")
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted, s_sorted = y_true[order], y_score[order]

    # indices where the score changes -> one threshold per distinct score
    distinct = np.where(np.diff(s_sorted))[0]
    cut = np.r_[distinct, len(s_sorted) - 1]

    tps = np.cumsum(y_sorted)[cut]
    fps = (cut + 1) - tps
    tpr = np.r_[0, tps] / max(y_true.sum(), 1)
    fpr = np.r_[0, fps] / max((~y_true).sum(), 1)
    thresholds = np.r_[np.inf, s_sorted[cut]]
    return fpr, tpr, thresholds


def auc(x, y):
    """Area under any curve with the trapezoid rule: sum of (dx * mean height)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def roc_auc_score(y_true, y_score, pos_label=1):
    """
    Area under the ROC curve.
    Interpretation: probability that a random positive gets a higher score
    than a random negative (ties count 1/2).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label)
    return auc(fpr, tpr)


def precision_recall_curve(y_true, y_score, pos_label=1):
    """Precision and recall at every distinct score threshold (highest first)."""
    y_true = np.asarray(y_true) == pos_label
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted, s_sorted = y_true[order], y_score[order]
    cut = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[cut]
    fps = (cut + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / max(y_true.sum(), 1)
    return precision, recall, s_sorted[cut]


def average_precision_score(y_true, y_score, pos_label=1):
    """AP = sum over thresholds of (R_n - R_{n-1}) * P_n  (step-wise area under PR)."""
    p, r, _ = precision_recall_curve(y_true, y_score, pos_label)
    r_prev = np.r_[0, r[:-1]]
    return float(np.sum((r - r_prev) * p))


def log_loss(y_true, y_prob, eps=1e-15):
    """Binary cross-entropy: -mean( y log p + (1-y) log(1-p) )."""
    y = np.asarray(y_true, float)
    p = np.clip(np.asarray(y_prob, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y_true, y_prob):
    """Mean squared error of probabilities: mean((p - y)^2)."""
    return float(np.mean((np.asarray(y_prob, float) - np.asarray(y_true, float)) ** 2))


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------
def mean_squared_error(y_true, y_pred):
    return float(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))


def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def r2_score(y_true, y_pred):
    """
    R^2 = 1 - SS_res / SS_tot
    SS_res = sum (y - y_hat)^2   (error left after the model)
    SS_tot = sum (y - mean y)^2  (error of always predicting the mean)
    If y is constant, SS_tot = 0: return 1.0 for a perfect fit, else 0.0 (as scikit-learn).
    """
    y = np.asarray(y_true, float)
    ss_res = np.sum((y - np.asarray(y_pred, float)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)
