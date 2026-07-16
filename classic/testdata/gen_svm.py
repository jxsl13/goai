#!/usr/bin/env python3
"""Generate SVM (SVC) goldens from REAL scikit-learn for classic/svm_test.go.

Reproduce with a scratch venv (do NOT commit .venv):

    python3 -m venv .venv-svm
    .venv-svm/bin/pip install numpy scikit-learn
    .venv-svm/bin/python classic/testdata/gen_svm.py

Writes classic/testdata/svm.json. The GoAI SVC solves the same convex
soft-margin dual by SMO, so its predictions match libsvm/sklearn exactly and
its decision-function values match to ~1e-3 (SMO iteration order differs).
"""
import json
import os

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_circles


def flat(a):
    return [float(v) for v in np.asarray(a, dtype=float).reshape(-1)]


def golden(X, y, **params):
    clf = SVC(**params).fit(X, y)
    n, d = X.shape
    return {
        "X": flat(X),
        "n": int(n),
        "d": int(d),
        "y": [int(v) for v in y],
        "C": float(params.get("C", 1.0)),
        "kernel": params.get("kernel", "rbf"),
        "gamma": float(clf._gamma) if params.get("kernel") != "linear" else 0.0,
        "degree": int(params.get("degree", 3)),
        "coef0": float(params.get("coef0", 0.0)),
        "pred": [int(v) for v in clf.predict(X)],
        "decision": flat(clf.decision_function(X)),
        "n_support": int(clf.support_.shape[0]),
        "dual_coef": flat(clf.dual_coef_),          # shape [1, n_SV], = y*alpha
        "intercept": float(clf.intercept_[0]),
        "train_acc": float((clf.predict(X) == y).mean()),
    }


def main():
    rng = np.random.default_rng(7)
    out = {"sklearn": __import__("sklearn").__version__}

    # --- rbf golden: two overlapping gaussian blobs (nonlinear-ish) ---
    Xa = rng.normal(loc=[0.0, 0.0], scale=0.8, size=(20, 2))
    Xb = rng.normal(loc=[2.2, 2.0], scale=0.8, size=(20, 2))
    X = np.vstack([Xa, Xb])
    y = np.array([0] * 20 + [1] * 20)
    out["rbf"] = golden(X, y, C=1.0, kernel="rbf", gamma="scale")

    # --- linear golden: linearly separable, hard margin (large C) ---
    Xa = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(15, 2))
    Xb = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(15, 2))
    Xl = np.vstack([Xa, Xb])
    yl = np.array([0] * 15 + [1] * 15)
    out["linear"] = golden(Xl, yl, C=10.0, kernel="linear")

    # --- poly golden ---
    Xa = rng.normal(loc=[0.0, 0.0], scale=0.7, size=(18, 2))
    Xb = rng.normal(loc=[1.8, 1.8], scale=0.7, size=(18, 2))
    Xp = np.vstack([Xa, Xb])
    yp = np.array([0] * 18 + [1] * 18)
    out["poly"] = golden(Xp, yp, C=1.0, kernel="poly", degree=3, gamma="scale", coef0=1.0)

    # --- concentric circles: RBF ≫ linear (the kernel-trick value) ---
    Xc, yc = make_circles(n_samples=120, factor=0.3, noise=0.08, random_state=3)
    lin = SVC(C=1.0, kernel="linear").fit(Xc, yc)
    rbf = SVC(C=1.0, kernel="rbf", gamma="scale").fit(Xc, yc)
    out["circles"] = {
        "X": flat(Xc),
        "n": int(Xc.shape[0]),
        "d": int(Xc.shape[1]),
        "y": [int(v) for v in yc],
        "gamma": float(rbf._gamma),
        "linear_acc": float((lin.predict(Xc) == yc).mean()),
        "rbf_acc": float((rbf.predict(Xc) == yc).mean()),
    }

    dest = os.path.join(os.path.dirname(__file__), "svm.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", dest, "(sklearn", out["sklearn"] + ")")


if __name__ == "__main__":
    main()
