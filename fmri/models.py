import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_COV = 3  # age, sex, scanner are always the last three columns


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class NewtonLogisticRegression(ClassifierMixin, BaseEstimator):
    """Logistic regression fitted with Newton's method (the paper's baseline)."""

    def __init__(self, tol=1e-5, max_iter=25, l2=1e-4):
        self.tol = tol
        self.max_iter = max_iter
        self.l2 = l2

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X1 = np.c_[np.ones(len(X)), X]
        reg = self.l2 * np.eye(X1.shape[1])
        reg[0, 0] = 0.0
        theta = np.zeros(X1.shape[1])
        for self.n_iter_ in range(1, self.max_iter + 1):
            p = _sigmoid(X1 @ theta)
            grad = X1.T @ (y - p) - reg @ theta
            hess = -(X1.T * (p * (1 - p))) @ X1 - reg
            step = np.linalg.solve(hess, grad)
            theta = theta - step
            if step @ step < self.tol:
                break
        self.theta_ = theta
        return self

    def decision_function(self, X):
        return np.c_[np.ones(len(X)), X] @ self.theta_

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


class Residualizer(TransformerMixin, BaseEstimator):
    """Removes the linear effect of the covariates (last N_COV columns) from
    the features. Fitted on training data only, then applied to test data."""

    def fit(self, X, y=None):
        F, C = X[:, :-N_COV], X[:, -N_COV:]
        self.coef_ = np.linalg.lstsq(np.c_[np.ones(len(C)), C], F, rcond=None)[0]
        return self

    def transform(self, X):
        F, C = X[:, :-N_COV], X[:, -N_COV:]
        return F - np.c_[np.ones(len(C)), C] @ self.coef_


class Oversampled(ClassifierMixin, BaseEstimator):
    """Duplicates minority-class rows before fitting; for models without class_weight."""

    def __init__(self, estimator=None, seed=0):
        self.estimator = estimator
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
        extra = rng.choice(small, len(big) - len(small), replace=True)
        idx = np.r_[np.arange(len(y)), extra]
        self.est_ = clone(self.estimator).fit(X[idx], y[idx])
        self.classes_ = self.est_.classes_
        return self

    def predict(self, X):
        return self.est_.predict(X)

    def predict_proba(self, X):
        return self.est_.predict_proba(X)


def logreg(residualize=False):
    steps = [Residualizer()] if residualize else []
    steps += [StandardScaler(),
              LogisticRegression(class_weight="balanced", max_iter=2000)]
    return make_pipeline(*steps)


LOGREG_GRID = {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]}


def rbf_svm():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))


SVM_GRID = {"svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", 0.001]}


def mlp():
    net = MLPClassifier(hidden_layer_sizes=(64,), alpha=0.1, max_iter=200,
                        random_state=0)
    return make_pipeline(StandardScaler(), Oversampled(net))


def svm_kernel(kernel):
    return make_pipeline(StandardScaler(), SVC(kernel=kernel, degree=2))
