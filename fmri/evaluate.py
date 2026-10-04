import numpy as np
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate,
                                     permutation_test_score)

SCORING = {"bal_acc": "balanced_accuracy", "auc": "roc_auc", "f1": "f1"}


def run_cv(model, X, y, grid=None, n_splits=5, n_repeats=3, seed=0):
    """Repeated stratified CV. Hyperparameters are tuned in an inner loop that
    only sees the training folds, so no score is computed on tuning data."""
    if grid:
        inner = StratifiedKFold(5, shuffle=True, random_state=seed)
        model = GridSearchCV(model, grid, cv=inner, scoring="roc_auc")
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=seed)
    res = cross_validate(model, X, y, cv=outer, scoring=SCORING)
    return {k: res[f"test_{k}"] for k in SCORING}


def run_paper_cv(model, X, y, seed=0):
    """10-fold CV on the balanced subset, reporting train and test accuracy."""
    cv = StratifiedKFold(10, shuffle=True, random_state=seed)
    res = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1"],
                         return_train_score=True)
    return {k: res[k] for k in ("train_accuracy", "test_accuracy",
                                "train_f1", "test_f1")}


def permutation_test(model, X, y, n_permutations=100, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    score, perms, p = permutation_test_score(
        model, X, y, scoring="roc_auc", cv=cv,
        n_permutations=n_permutations, random_state=seed)
    return score, perms, p


def summarize(scores):
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}
