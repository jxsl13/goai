#!/usr/bin/env python3
"""Golden-value generator for GoAI's Gradient Boosting (classic/gbm.go, §V1).

Emits scikit-learn's GradientBoostingRegressor / GradientBoostingClassifier
reference predictions on fixed datasets and hyper-parameters so the Go
implementation can be checked for parity. Deterministic (fixed seeds).

Run from repo root inside the golden venv:

    .venv/bin/python classic/testdata/gen_gbm.py

Writes classic/testdata/gbm.json. Records the sklearn/numpy versions for
reproducibility (§V13). Exact per-prediction bit-match against sklearn's Cython
trees is unrealistic (float summation order + split tie-breaking), so the Go
tests assert R^2 parity (regressor) and accuracy parity (classifier) plus a
documented rtol on the regressor predictions.
"""
import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, accuracy_score

# --- shared hyper-parameters (match the Go defaults) ---
N_EST = 100
LR = 0.1
MAX_DEPTH = 3

flat = lambda a: [float(v) for v in np.asarray(a, dtype=float).reshape(-1)]


def friedman1(rng, n):
    """Friedman #1 nonlinear regression target (Friedman 1991); 5 features,
    uniform [0,1]. Strongly nonlinear -> GBM beats a linear model and a stump."""
    X = rng.uniform(0.0, 1.0, size=(n, 5))
    y = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20.0 * (X[:, 2] - 0.5) ** 2
         + 10.0 * X[:, 3]
         + 5.0 * X[:, 4]
         + rng.normal(scale=1.0, size=n))
    return X, y


def moons(rng, n):
    """Two interleaving half-moons: a nonlinear binary boundary (a linear
    classifier cannot separate them, GBM can)."""
    n_half = n // 2
    t_out = np.pi * rng.uniform(0.0, 1.0, size=n_half)
    x_out = np.stack([np.cos(t_out), np.sin(t_out)], axis=1)
    t_in = np.pi * rng.uniform(0.0, 1.0, size=n - n_half)
    x_in = np.stack([1.0 - np.cos(t_in), 1.0 - np.sin(t_in) - 0.5], axis=1)
    X = np.vstack([x_out, x_in]) + rng.normal(scale=0.15, size=(n, 2))
    y = np.concatenate([np.zeros(n_half, dtype=int), np.ones(n - n_half, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def build_regressor():
    rng = np.random.default_rng(20240711)
    Xtr, ytr = friedman1(rng, 160)
    Xte, yte = friedman1(rng, 40)
    m = GradientBoostingRegressor(
        n_estimators=N_EST, learning_rate=LR, max_depth=MAX_DEPTH,
        subsample=1.0, random_state=0)
    m.fit(Xtr, ytr)
    pred_te = m.predict(Xte)
    return {
        "n_train": Xtr.shape[0], "n_test": Xte.shape[0], "d": Xtr.shape[1],
        "n_estimators": N_EST, "learning_rate": LR, "max_depth": MAX_DEPTH,
        "X_train": flat(Xtr), "y_train": flat(ytr),
        "X_test": flat(Xte), "y_test": flat(yte),
        "init": float(m.init_.constant_.ravel()[0]),
        "pred_test": flat(pred_te),
        "r2_test": float(r2_score(yte, pred_te)),
    }


def build_classifier():
    rng = np.random.default_rng(20240712)
    Xtr, ytr = moons(rng, 200)
    Xte, yte = moons(rng, 60)
    m = GradientBoostingClassifier(
        n_estimators=N_EST, learning_rate=LR, max_depth=MAX_DEPTH,
        subsample=1.0, random_state=0)
    m.fit(Xtr, ytr)
    prob_te = m.predict_proba(Xte)[:, 1]
    pred_te = m.predict(Xte).astype(int)
    p1 = float(np.mean(ytr))
    return {
        "n_train": Xtr.shape[0], "n_test": Xte.shape[0], "d": Xtr.shape[1],
        "n_estimators": N_EST, "learning_rate": LR, "max_depth": MAX_DEPTH,
        "X_train": flat(Xtr), "y_train": [int(v) for v in ytr],
        "X_test": flat(Xte), "y_test": [int(v) for v in yte],
        "base_rate": p1, "init_logodds": float(np.log(p1 / (1.0 - p1))),
        "prob_test": flat(prob_te),
        "pred_test": [int(v) for v in pred_te],
        "accuracy_test": float(accuracy_score(yte, pred_te)),
    }


def main():
    data = {
        "regressor": build_regressor(),
        "classifier": build_classifier(),
        "sklearn": __import__("sklearn").__version__,
        "numpy": np.__version__,
    }
    dest = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dest, "gbm.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote classic/testdata/gbm.json (sklearn %s)" % data["sklearn"])


if __name__ == "__main__":
    main()
