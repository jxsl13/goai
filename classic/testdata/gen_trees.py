#!/usr/bin/env python3
"""GoAI decision-tree / random-forest golden generator (classic package, §V1).

Emits reference values from REAL scikit-learn so the Go CART tree and random
forest are validated against the canonical implementation (Breiman 1984 / 2001).

Reproduce:
    python3 -m venv .venv && .venv/bin/pip install scikit-learn numpy
    .venv/bin/python classic/testdata/gen_trees.py

Determinism / exact-match notes
-------------------------------
scikit-learn casts X to float32 (tree DTYPE) before building and computes split
thresholds as the float32 midpoint of consecutive sorted feature values. To make
the Go float64 tree reproduce sklearn's structure and predictions exactly we:

  * round every stored X value to float32 precision here, so sklearn's internal
    cast is a no-op and no two distinct float64 values collide under it, and
  * pick continuous random features (no exact ties), so every best split is
    unique and the feature-order tie-break in sklearn's BestSplitter is
    irrelevant — the same split is chosen regardless of visit order.

Under those conditions the greedy CART tie-break reduces to "lowest impurity,
then lowest feature index, then lowest threshold", which the Go tree implements
deterministically.
"""
import json
import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def f32(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def flat(a):
    return [float(v) for v in np.asarray(a).reshape(-1)]


def tree_arrays(est):
    t = est.tree_
    # argmax class / mean value per node (value is [n_nodes,1,n_out])
    val = t.value.reshape(t.node_count, -1)
    return {
        "feature": [int(v) for v in t.feature],
        "threshold": [float(v) for v in t.threshold],
        "children_left": [int(v) for v in t.children_left],
        "children_right": [int(v) for v in t.children_right],
        "value_argmax": [int(np.argmax(v)) for v in val],
        "value_mean": [float(v[0]) for v in val],
        "n_node_samples": [int(v) for v in t.n_node_samples],
    }


def main():
    data = {"sklearn": __import__("sklearn").__version__}

    # --- classification: 3 classes, continuous features, unconstrained -------
    Xc, yc = make_classification(
        n_samples=80, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=1.4, random_state=1,
    )
    Xc = f32(Xc)
    rng = np.random.default_rng(101)
    perm = rng.permutation(80)
    tr, te = perm[:60], perm[60:]
    Xtr, ytr, Xte, yte = Xc[tr], yc[tr], Xc[te], yc[te]

    gini = DecisionTreeClassifier(criterion="gini", random_state=0).fit(Xtr, ytr)
    ent = DecisionTreeClassifier(criterion="entropy", max_depth=3,
                                 random_state=0).fit(Xtr, ytr)

    data["clf_gini"] = {
        "n": 60, "d": 5, "k": 3, "criterion": "gini", "max_depth": 0,
        "Xtr": flat(Xtr), "ytr": [int(v) for v in ytr],
        "Xte": flat(Xte), "yte": [int(v) for v in yte],
        "pred_tr": [int(v) for v in gini.predict(Xtr)],
        "pred_te": [int(v) for v in gini.predict(Xte)],
        "depth": int(gini.get_depth()), "n_leaves": int(gini.get_n_leaves()),
        "tree": tree_arrays(gini),
    }
    data["clf_entropy"] = {
        "n": 60, "d": 5, "k": 3, "criterion": "entropy", "max_depth": 3,
        "Xtr": flat(Xtr), "ytr": [int(v) for v in ytr],
        "Xte": flat(Xte), "yte": [int(v) for v in yte],
        "pred_tr": [int(v) for v in ent.predict(Xtr)],
        "pred_te": [int(v) for v in ent.predict(Xte)],
        "depth": int(ent.get_depth()), "n_leaves": int(ent.get_n_leaves()),
        "tree": tree_arrays(ent),
    }

    # --- regression: continuous target, depth-limited ------------------------
    rng = np.random.default_rng(7)
    Xr = f32(rng.normal(size=(70, 4)))
    beta = np.array([1.5, -2.0, 0.0, 0.7])
    yr = f32(Xr @ beta + 0.3 * rng.normal(size=70))
    perm = rng.permutation(70)
    tr, te = perm[:55], perm[55:]
    Xrtr, yrtr, Xrte, yrte = Xr[tr], yr[tr], Xr[te], yr[te]
    reg = DecisionTreeRegressor(criterion="squared_error", max_depth=4,
                                random_state=0).fit(Xrtr, yrtr)
    data["reg"] = {
        "n": 55, "d": 4, "criterion": "squared_error", "max_depth": 4,
        "Xtr": flat(Xrtr), "ytr": flat(yrtr),
        "Xte": flat(Xrte), "yte": flat(yrte),
        "pred_tr": flat(reg.predict(Xrtr)),
        "pred_te": flat(reg.predict(Xrte)),
        "depth": int(reg.get_depth()), "n_leaves": int(reg.get_n_leaves()),
        "tree": tree_arrays(reg),
    }

    # --- random forest vs single deep tree on a NOISY dataset ----------------
    Xn, yn = make_classification(
        n_samples=300, n_features=12, n_informative=6, n_redundant=2,
        n_classes=2, flip_y=0.20, class_sep=0.8, random_state=3,
    )
    Xn = f32(Xn)
    rng = np.random.default_rng(2024)
    perm = rng.permutation(300)
    tr, te = perm[:200], perm[200:]
    Xntr, yntr, Xnte, ynte = Xn[tr], yn[tr], Xn[te], yn[te]
    sk_tree = DecisionTreeClassifier(random_state=0).fit(Xntr, yntr)
    sk_rf = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                   random_state=0).fit(Xntr, yntr)
    data["noisy"] = {
        "n_tr": 200, "n_te": 100, "d": 12, "k": 2,
        "Xtr": flat(Xntr), "ytr": [int(v) for v in yntr],
        "Xte": flat(Xnte), "yte": [int(v) for v in ynte],
        "sk_tree_test_acc": float(sk_tree.score(Xnte, ynte)),
        "sk_rf_test_acc": float(sk_rf.score(Xnte, ynte)),
    }

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trees.json")
    with open(dest, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote", dest, "(sklearn", data["sklearn"], ")")
    print("clf_gini depth", data["clf_gini"]["depth"],
          "leaves", data["clf_gini"]["n_leaves"])
    print("reg depth", data["reg"]["depth"], "leaves", data["reg"]["n_leaves"])
    print("noisy: single tree", data["noisy"]["sk_tree_test_acc"],
          "rf", data["noisy"]["sk_rf_test_acc"])


if __name__ == "__main__":
    main()
