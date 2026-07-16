#!/usr/bin/env python3
"""Generate k-NN and Gaussian Naive Bayes goldens from REAL scikit-learn for
classic/knn_test.go and classic/naivebayes_test.go (§V1 golden parity anchor).

Reproduce with a scratch venv (do NOT commit .venv):

    python3 -m venv .venv-knn
    .venv-knn/bin/pip install numpy scikit-learn
    .venv-knn/bin/python classic/testdata/gen_knn_nb.py

Writes classic/testdata/knn.json and classic/testdata/naivebayes.json.

Determinism / exact-match notes
-------------------------------
k-NN predictions are exact against scikit-learn. To make the k-nearest set of
each query unique (so neighbour-order tie-breaks never matter), the datasets use
continuous Gaussian features drawn from a fixed seed — no two pairwise distances
collide. Majority-vote ties then break to the lowest class label, which both
numpy.argmax (over sorted classes_) and the Go implementation do.

GaussianNB's log-posterior is a closed form, so predictions match exactly and
the joint log-likelihood matches to ~1e-6.
"""
import json
import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification, make_regression


def flat(a):
    return [float(v) for v in np.asarray(a, dtype=float).reshape(-1)]


def blobs_clf(rng, n_per, centers, scale=0.9):
    """Well-separated Gaussian blobs, one class per center."""
    Xs, ys = [], []
    for c, mu in enumerate(centers):
        Xs.append(rng.normal(loc=mu, scale=scale, size=(n_per, len(mu))))
        ys.append(np.full(n_per, c))
    return np.vstack(Xs), np.concatenate(ys)


def knn_clf_golden(Xtr, ytr, Xte, k, weights, metric):
    p = 2 if metric == "euclidean" else 1
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p).fit(Xtr, ytr)
    n_tr, d = Xtr.shape
    return {
        "n_tr": int(n_tr),
        "d": int(d),
        "Xtr": flat(Xtr),
        "ytr": [int(v) for v in ytr],
        "n_te": int(Xte.shape[0]),
        "Xte": flat(Xte),
        "k": int(k),
        "weights": weights,
        "metric": metric,
        "pred": [int(v) for v in clf.predict(Xte)],
        "proba": flat(clf.predict_proba(Xte)),
        "classes": [int(v) for v in clf.classes_],
        "train_acc": float((clf.predict(Xtr) == ytr).mean()),
        "test_pred_self_k1": [
            int(v) for v in KNeighborsClassifier(n_neighbors=1, p=p).fit(Xtr, ytr).predict(Xtr)
        ],
    }


def knn_reg_golden(Xtr, ytr, Xte, k, weights, metric):
    p = 2 if metric == "euclidean" else 1
    reg = KNeighborsRegressor(n_neighbors=k, weights=weights, p=p).fit(Xtr, ytr)
    n_tr, d = Xtr.shape
    return {
        "n_tr": int(n_tr),
        "d": int(d),
        "Xtr": flat(Xtr),
        "ytr": flat(ytr),
        "n_te": int(Xte.shape[0]),
        "Xte": flat(Xte),
        "k": int(k),
        "weights": weights,
        "metric": metric,
        "pred": flat(reg.predict(Xte)),
    }


def main_knn():
    rng = np.random.default_rng(11)
    out = {"sklearn": __import__("sklearn").__version__}

    # 3-class blobs for the classifier goldens (well separated ⇒ unique NN sets)
    centers = [(-3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)]
    Xtr, ytr = blobs_clf(rng, 25, centers, scale=0.9)
    Xte, _ = blobs_clf(rng, 8, centers, scale=0.9)

    out["clf_uniform_k5"] = knn_clf_golden(Xtr, ytr, Xte, 5, "uniform", "euclidean")
    out["clf_uniform_k1"] = knn_clf_golden(Xtr, ytr, Xte, 1, "uniform", "euclidean")
    out["clf_uniform_k9"] = knn_clf_golden(Xtr, ytr, Xte, 9, "uniform", "euclidean")
    out["clf_distance_k5"] = knn_clf_golden(Xtr, ytr, Xte, 5, "distance", "euclidean")
    out["clf_manhattan_k5"] = knn_clf_golden(Xtr, ytr, Xte, 5, "uniform", "manhattan")

    # a harder overlapping problem to report k-sensitivity (k=1 overfits)
    Xh, yh = make_classification(
        n_samples=120, n_features=6, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=1.2, random_state=5,
    )
    ntr = 90
    ks = [1, 3, 5, 9, 15]
    sens = {}
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xh[:ntr], yh[:ntr])
        sens[str(k)] = {
            "train_acc": float((clf.predict(Xh[:ntr]) == yh[:ntr]).mean()),
            "test_acc": float((clf.predict(Xh[ntr:]) == yh[ntr:]).mean()),
        }
    out["sensitivity"] = {
        "n_tr": ntr, "d": 6, "n_te": int(Xh.shape[0] - ntr),
        "Xtr": flat(Xh[:ntr]), "ytr": [int(v) for v in yh[:ntr]],
        "Xte": flat(Xh[ntr:]), "yte": [int(v) for v in yh[ntr:]],
        "ks": ks, "acc": sens,
    }

    # regressor goldens
    Xr, yr = make_regression(n_samples=80, n_features=4, noise=5.0, random_state=8)
    rtr = 60
    out["reg_uniform_k5"] = knn_reg_golden(Xr[:rtr], yr[:rtr], Xr[rtr:], 5, "uniform", "euclidean")
    out["reg_distance_k3"] = knn_reg_golden(Xr[:rtr], yr[:rtr], Xr[rtr:], 3, "distance", "euclidean")

    dest = os.path.join(os.path.dirname(__file__), "knn.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", dest, "(sklearn", out["sklearn"] + ")")


def nb_golden(Xtr, ytr, Xte, var_smoothing=1e-9):
    clf = GaussianNB(var_smoothing=var_smoothing).fit(Xtr, ytr)
    n_tr, d = Xtr.shape
    return {
        "n_tr": int(n_tr),
        "d": int(d),
        "Xtr": flat(Xtr),
        "ytr": [int(v) for v in ytr],
        "n_te": int(Xte.shape[0]),
        "Xte": flat(Xte),
        "var_smoothing": float(var_smoothing),
        "classes": [int(v) for v in clf.classes_],
        "theta": flat(clf.theta_),
        "var": flat(clf.var_),
        "epsilon": float(clf.epsilon_),
        "pred": [int(v) for v in clf.predict(Xte)],
        "proba": flat(clf.predict_proba(Xte)),
        "jll": flat(clf._joint_log_likelihood(Xte)),
        "train_acc": float((clf.predict(Xtr) == ytr).mean()),
        "test_pred": [int(v) for v in clf.predict(Xte)],
    }


def main_nb():
    out = {"sklearn": __import__("sklearn").__version__}

    # primary golden: 3-class, 5-feature classification problem
    X, y = make_classification(
        n_samples=150, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=1.5, random_state=13,
    )
    ntr = 110
    out["main"] = nb_golden(X[:ntr], y[:ntr], X[ntr:])
    out["main"]["yte"] = [int(v) for v in y[ntr:]]

    # degenerate: a constant (zero-variance) feature ⇒ var-smoothing must save us
    rng = np.random.default_rng(21)
    Xa = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(30, 2))
    Xb = rng.normal(loc=[3.0, 3.0], scale=1.0, size=(30, 2))
    Xd = np.vstack([Xa, Xb])
    Xd = np.hstack([Xd, np.full((60, 1), 7.0)])  # 3rd feature constant = 7
    yd = np.array([0] * 30 + [1] * 30)
    out["degenerate"] = nb_golden(Xd, yd, Xd)

    dest = os.path.join(os.path.dirname(__file__), "naivebayes.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", dest, "(sklearn", out["sklearn"] + ")")


if __name__ == "__main__":
    main_knn()
    main_nb()
