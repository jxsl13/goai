#!/usr/bin/env python3
"""Generate DBSCAN goldens from REAL scikit-learn for classic/dbscan_test.go
(§V1, sklearn-golden parity anchor).

Reproduce with a scratch venv (do NOT commit .venv):

    python3 -m venv .venv-dbscan
    .venv-dbscan/bin/pip install numpy scikit-learn
    .venv-dbscan/bin/python classic/testdata/gen_dbscan.py

Writes classic/testdata/dbscan.json.

Parity note
-----------
DBSCAN is deterministic given (eps, min_samples, the metric, and the point
order): a point is core iff at least min_samples points (itself included) lie
within eps, clusters are the connected components of the core points under
eps-reachability, and each border point joins the first cluster that reaches it
in index order. GoAI replicates scikit-learn's dbscan_inner traversal, so the
labels match EXACTLY up to a renumbering of cluster ids (noise is always -1).
"""
import json
import os

import numpy as np
from sklearn.cluster import DBSCAN


def flat(a):
    return [float(v) for v in np.asarray(a, dtype=float).reshape(-1)]


def dbscan_golden(X, eps, min_samples, metric="euclidean"):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X)
    n, d = X.shape
    core = np.zeros(n, dtype=bool)
    core[db.core_sample_indices_] = True
    n_clusters = len(set(int(l) for l in db.labels_) - {-1})
    return {
        "X": flat(X),
        "n": int(n),
        "d": int(d),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "metric": metric,
        "labels": [int(v) for v in db.labels_],
        "core": [bool(v) for v in core],
        "n_clusters": int(n_clusters),
        "n_noise": int((db.labels_ == -1).sum()),
    }


def main():
    out = {"sklearn": __import__("sklearn").__version__}
    rng = np.random.default_rng(11)

    # --- two well-separated dense blobs + explicit noise points ----------
    a = rng.normal(loc=[0.0, 0.0], scale=0.35, size=(40, 2))
    b = rng.normal(loc=[6.0, 6.0], scale=0.35, size=(40, 2))
    noise = np.array([[3.0, -3.0], [-4.0, 5.0], [10.0, 0.0], [3.0, 12.0]])
    X = np.vstack([a, b, noise])
    out["blobs"] = dbscan_golden(X, eps=0.7, min_samples=5)

    # --- eps monotonicity: same data, three eps values -------------------
    rng2 = np.random.default_rng(3)
    c = np.vstack([
        rng2.normal(loc=[0.0, 0.0], scale=0.5, size=(30, 2)),
        rng2.normal(loc=[3.0, 3.0], scale=0.5, size=(30, 2)),
        rng2.normal(loc=[6.0, 0.0], scale=0.5, size=(30, 2)),
    ])
    out["eps_small"] = dbscan_golden(c, eps=0.4, min_samples=4)
    out["eps_mid"] = dbscan_golden(c, eps=0.9, min_samples=4)
    out["eps_large"] = dbscan_golden(c, eps=5.0, min_samples=4)

    # --- manhattan metric variant ----------------------------------------
    out["manhattan"] = dbscan_golden(X, eps=1.0, min_samples=5, metric="manhattan")

    # --- interleaved half-moons: density clustering where k-means fails ---
    from sklearn.datasets import make_moons
    Xm, _ = make_moons(n_samples=150, noise=0.06, random_state=0)
    out["moons"] = dbscan_golden(Xm, eps=0.22, min_samples=5)

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dbscan.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", dest, "(sklearn", out["sklearn"] + ")")
    for key in ("blobs", "eps_small", "eps_mid", "eps_large", "manhattan", "moons"):
        g = out[key]
        print(key, "clusters", g["n_clusters"], "noise", g["n_noise"])


if __name__ == "__main__":
    main()
