"""
statlab — from-scratch implementations of important statistics and ML procedures.

Only NumPy and SciPy are required. Every class follows the scikit-learn style:
    model.fit(X, y); model.predict(X)
    sampler.fit_resample(X, y)

Modules
-------
statlab.metrics      confusion matrix, precision/recall/F1, ROC-AUC, ...
statlab.validation   train_test_split, KFold, StratifiedKFold, cross_val_score
statlab.models       Perceptron, Linear/Logistic regression, kNN, Naive Bayes, Decision tree
statlab.imbalance    Random over/under sampling, SMOTE, Borderline-SMOTE, ADASYN,
                     Tomek links, Edited Nearest Neighbours, NearMiss
statlab.stats        t-tests, chi-square, ANOVA, Mann-Whitney, CIs, effect sizes
"""
from . import metrics, validation, models, imbalance, stats

__all__ = ["metrics", "validation", "models", "imbalance", "stats"]
__version__ = "0.1.0"
