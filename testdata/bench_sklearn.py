#!/usr/bin/env python3
"""Cross-language performance reference: time scikit-learn's fit for the six
classical methods in classic/perfcompare_test.go's scorecard, on the byte-for-byte
IDENTICAL synthetic dataset the Go side writes, so GoAI's pure-Go classical
learners can be compared head-to-head with scikit-learn in BENCHMARKS.md section 5
and docs/benchmarking.md.

Fairness contract:
  * identical data     -- reads $PERF_CSV_DIR/perf_X.csv (4000x20) and perf_y.csv
    (4000 labels, 3 classes) that TestPerfCompareVsSklearn writes; sklearn times
    the exact same rows GoAI timed.
  * identical labels    -- GradientBoosting and SVC are timed on the SAME binary
    derivation the Go side uses (y != 0 -> 1), the others on the 3-class y.
  * matched hyperparams -- max_depth=12 tree, 100-tree forest (random_state=42),
    100-estimator GBM, RBF SVC, k=5 KNN, matching the Go constructors.
  * best-of-N fit       -- the minimum fit wall-time over N repeats, same as a
    Go benchmark's steady state; sklearn/numpy versions are printed (V13).

n_jobs is reported at BOTH 1 and -1: only RandomForest and KNN honour it, so the
others show "-" in the parallel column. GoAI's forest is GOMAXPROCS-parallel, the
rest single-threaded -- compare GoAI to n_jobs=-1 for the forest, n_jobs=1 else.

Usage:
    PERF_CSV_DIR=/tmp/perf .venv/bin/python testdata/bench_sklearn.py
    PERF_CSV_DIR=/tmp/perf .venv/bin/python testdata/bench_sklearn.py --json
"""
import json
import os
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REPS = 5


def best_fit_ms(make, X, y) -> float:
    """Minimum fit wall-time (ms) over REPS fresh estimators."""
    best = float("inf")
    for _ in range(REPS):
        est = make()
        s = time.perf_counter()
        est.fit(X, y)
        best = min(best, time.perf_counter() - s)
    return best * 1e3


def main() -> None:
    d = os.environ.get("PERF_CSV_DIR")
    if not d:
        sys.exit("set PERF_CSV_DIR to the directory TestPerfCompareVsSklearn wrote")
    X = np.loadtxt(os.path.join(d, "perf_X.csv"), delimiter=",")
    y = np.loadtxt(os.path.join(d, "perf_y.csv"), dtype=int)
    y_bin = (y != 0).astype(int)  # matches the Go side's yb/ybi

    # (name, estimator factory taking n_jobs, target, honours_n_jobs)
    methods = [
        ("DecisionTree", lambda nj: DecisionTreeClassifier(max_depth=12), y, False),
        ("RandomForest100", lambda nj: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=nj), y, True),
        ("GradientBoosting100", lambda nj: GradientBoostingClassifier(n_estimators=100), y_bin, False),
        ("SVC_rbf", lambda nj: SVC(kernel="rbf"), y_bin, False),
        ("GaussianNB", lambda nj: GaussianNB(), y, False),
        ("KNN_fit", lambda nj: KNeighborsClassifier(n_neighbors=5, n_jobs=nj), y, True),
    ]

    rows = []
    for name, make, target, par in methods:
        one = best_fit_ms(lambda: make(1), X, target)
        allj = best_fit_ms(lambda: make(-1), X, target) if par else None
        rows.append((name, one, allj))

    versions = f"sklearn {sklearn.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}"
    if "--json" in sys.argv:
        print(json.dumps({
            "versions": versions,
            "n": int(X.shape[0]), "d": int(X.shape[1]),
            "fit_ms": {n: {"n_jobs=1": o, "n_jobs=-1": a} for n, o, a in rows},
        }, indent=2))
        return

    print(versions)
    print(f"dataset: {X.shape[0]}x{X.shape[1]}, best-of-{REPS} fit")
    print(f"{'method':<20} {'n_jobs=1 (ms)':>14} {'n_jobs=-1 (ms)':>15}")
    for name, one, allj in rows:
        aj = f"{allj:.2f}" if allj is not None else "-"
        print(f"{name:<20} {one:>14.2f} {aj:>15}")


if __name__ == "__main__":
    main()
