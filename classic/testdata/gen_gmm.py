#!/usr/bin/env python3
"""Generate Gaussian-mixture (EM) goldens from REAL scikit-learn for
classic/gmm_test.go (§V1, sklearn-golden parity anchor).

Reproduce with a scratch venv (do NOT commit .venv):

    python3 -m venv .venv-gmm
    .venv-gmm/bin/pip install numpy scikit-learn
    .venv-gmm/bin/python classic/testdata/gen_gmm.py

Writes classic/testdata/gmm.json.

Parity note
-----------
EM converges to a local optimum whose basin depends on the initialization and
the exact iteration order, so a byte-for-byte match with sklearn is neither
expected nor asserted. Instead the data here is drawn from *well separated*
Gaussians, for which the maximum-likelihood fit is unique: any reasonable
initialization (sklearn's k-means init, GoAI's seeded k-means++) lands in the
same basin. The Go test therefore checks

  * cluster assignments match sklearn up to a label permutation, and
  * per-sample log-likelihood (score_samples) matches to rtol 1e-3,

which is the standard, init-robust way to validate an EM implementation.
The K=1 collapse case is exact: a one-component GMM is just the single
maximum-likelihood Gaussian (population mean/covariance), matched to 1e-6.
"""
import json
import os

import numpy as np
from sklearn.mixture import GaussianMixture


def flat(a):
    return [float(v) for v in np.asarray(a, dtype=float).reshape(-1)]


def sample_blobs(rng, centers, scale, per):
    """Draw `per` isotropic points around each center; return X and truth."""
    xs, ys = [], []
    for k, c in enumerate(centers):
        xs.append(rng.normal(loc=c, scale=scale, size=(per, len(c))))
        ys.extend([k] * per)
    return np.vstack(xs), np.array(ys)


def gmm_golden(X, k, cov_type, reg, tol, max_iter):
    gm = GaussianMixture(
        n_components=k,
        covariance_type=cov_type,
        reg_covar=reg,
        tol=tol,
        max_iter=max_iter,
        init_params="kmeans",
        random_state=0,
    ).fit(X)
    n, d = X.shape
    return {
        "X": flat(X),
        "n": int(n),
        "d": int(d),
        "k": int(k),
        "covariance": cov_type,
        "reg_covar": float(reg),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "weights": flat(gm.weights_),
        "means": flat(gm.means_),           # [k, d]
        "covariances": flat(gm.covariances_),  # diag: [k,d]; full: [k,d,d]
        "predict": [int(v) for v in gm.predict(X)],
        "score_samples": flat(gm.score_samples(X)),
        "lower_bound": float(gm.lower_bound_),  # mean max-log-likelihood
        "n_iter": int(gm.n_iter_),
        "converged": bool(gm.converged_),
    }


def main():
    out = {"sklearn": __import__("sklearn").__version__}
    rng = np.random.default_rng(42)

    # --- diagonal covariance: 3 well-separated isotropic blobs -------------
    Xd, yd = sample_blobs(rng, [[0.0, 0.0], [8.0, 8.5], [0.5, 8.0]], 0.6, 60)
    gd = gmm_golden(Xd, k=3, cov_type="diag", reg=1e-6, tol=1e-3, max_iter=200)
    gd["truth"] = [int(v) for v in yd]
    gd["true_means"] = flat([[0.0, 0.0], [8.0, 8.5], [0.5, 8.0]])
    out["diag"] = gd

    # --- full covariance: 3 anisotropic, well-separated blobs -------------
    rng2 = np.random.default_rng(7)
    A = np.array([[1.2, 0.5], [0.0, 0.7]])  # shared shear -> correlated blobs
    base, yb = sample_blobs(rng2, [[-6.0, -6.0], [7.0, -5.0], [0.0, 8.0]], 1.0, 70)
    Xf = base.copy()
    for k in range(3):
        m = yb == k
        c = np.array([[-6.0, -6.0], [7.0, -5.0], [0.0, 8.0]])[k]
        Xf[m] = (base[m] - c) @ A.T + c
    gf = gmm_golden(Xf, k=3, cov_type="full", reg=1e-6, tol=1e-3, max_iter=300)
    gf["truth"] = [int(v) for v in yb]
    out["full"] = gf

    # --- K=1 collapse: one component == the single ML Gaussian ------------
    rng3 = np.random.default_rng(123)
    Xk = rng3.normal(loc=[3.0, -1.0], scale=[2.0, 0.5], size=(50, 2))
    gk = gmm_golden(Xk, k=1, cov_type="full", reg=0.0, tol=1e-6, max_iter=200)
    gk["data_mean"] = flat(Xk.mean(axis=0))
    # population (ddof=0) covariance = the K=1 ML estimate
    gk["data_cov"] = flat(np.cov(Xk.T, bias=True))
    out["k1"] = gk

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gmm.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote", dest, "(sklearn", out["sklearn"] + ")")
    print("diag: n_iter", out["diag"]["n_iter"], "lb", out["diag"]["lower_bound"])
    print("full: n_iter", out["full"]["n_iter"], "lb", out["full"]["lower_bound"])
    print("k1  : n_iter", out["k1"]["n_iter"], "lb", out["k1"]["lower_bound"])


if __name__ == "__main__":
    main()
