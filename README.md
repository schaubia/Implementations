# StatLab

From-scratch implementations of important statistics and machine-learning procedures,
plus a Streamlit app that demonstrates each one. The library needs only **NumPy** and **SciPy**
(SciPy is used just for reference distributions — t, χ², F, normal — like a printed table).

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use the code without the app

Copy the `statlab/` folder next to your script — no install needed.

```python
from statlab.imbalance import SMOTE, TomekLinks
from statlab.models import Perceptron, LogisticRegression, StandardScaler
from statlab import metrics, stats

X_res, y_res = SMOTE(k_neighbors=5, random_state=0).fit_resample(X, y)

scaler = StandardScaler().fit(X_res)
model = LogisticRegression(lr=0.3, n_iter=2000).fit(scaler.transform(X_res), y_res)
print(metrics.f1_score(y_test, model.predict(scaler.transform(X_test))))

r = stats.ttest_ind(group_a, group_b, equal_var=False)   # Welch
print(r, r.details, r.conclusion(alpha=0.05))
```

## What's inside

| Module | Contents |
|---|---|
| `statlab.models` | Perceptron, LinearRegression, RidgeRegression, LogisticRegression (binary + softmax), KNeighborsClassifier/Regressor, GaussianNB, DecisionTreeClassifier (Gini/entropy), StandardScaler, PCA |
| `statlab.imbalance` | RandomOverSampler, SMOTE, BorderlineSMOTE (1 & 2), ADASYN, RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss (1, 2, 3), ClusterCentroids, SMOTETomek, SMOTEENN — same `fit_resample` API and `sampling_strategy` semantics as imbalanced-learn |
| `statlab.stats` | describe, t-tests (one-sample, Student, Welch, paired), z-tests, ANOVA + Tukey HSD, χ² goodness-of-fit / independence, Fisher exact, Mann-Whitney U, Wilcoxon, Kruskal-Wallis, KS, Pearson/Spearman, CIs (mean, difference, proportion Wald/Wilson), Cohen's d, Hedges' g, η², Cramér's V, power & sample size, Bonferroni/Holm/BH |
| `statlab.metrics` | confusion matrix, accuracy, precision, recall, specificity, F-beta, balanced accuracy, MCC, G-mean, ROC/PR curves, AUC, AP, log loss, Brier, MSE/RMSE/MAE/R² |
| `statlab.validation` | train_test_split (stratified), KFold, StratifiedKFold, LeaveOneOut, cross_val_score, bootstrap CI |
| `statlab.datasets` | NumPy-only toy data: blobs, moons, imbalanced, regression line |

Every docstring explains the method step by step; the app shows the docstring and source of
whatever you are looking at.

## Verified against the reference libraries

`tests/test_against_reference.py` compares results with scikit-learn, SciPy and imbalanced-learn
(those are needed only for the tests):

```bash
pip install pytest scikit-learn imbalanced-learn
python -m pytest tests -q        # 19 passed
```

- Metrics, linear/ridge, Gaussian NB, PCA, KFold, all statistical tests: match to floating-point precision.
- TomekLinks, EditedNearestNeighbours (all/mode), NearMiss 1/2/3: **identical** output to imbalanced-learn.
- Random samplers (SMOTE, ADASYN, …): same class counts; synthetic points verified to lie on segments
  between minority points. Exact points differ because the random-number streams differ.
- Logistic regression (gradient descent) matches sklearn's unpenalised solution to ~1e-2.

## Notes / limitations

- Mann-Whitney and Wilcoxon use the normal approximation (SciPy's `method="asymptotic"/"approx"`).
- The perceptron is binary; logistic regression handles any number of classes.
- Decision tree is classification only, without pruning.
