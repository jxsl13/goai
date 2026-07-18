#!/usr/bin/env python3
"""GoAI golden-value generator (§V1, §V13).

Emits reference values for parity tests. NumPy provides f32 rounding; transcendental
functions come from Python's C libm (math), an oracle independent of Go's pure-Go
math package — so agreement is a real cross-implementation check, not a tautology.

f32 goldens mirror the reference kernel's compute-in-f64-then-narrow path
(ADR-0001): round input to f32, compute in f64, round result to f32.

Run: `make golden` (uses .venv/bin/python). Deterministic — no RNG.
Outputs are committed; env metadata is embedded for reproducibility (§V13).
"""
import json
import math
import os
import platform
import sys

import numpy as np


def f32(v: float) -> float:
    return float(np.float32(v))


def f32_unary(fn, x):
    return [f32(fn(f32(xi))) for xi in x]


def f64_unary(fn, x):
    return [fn(xi) for xi in x]


def f32_binary(op, x, y):
    return [f32(op(f32(a), f32(b))) for a, b in zip(x, y)]


def f64_binary(op, x, y):
    return [op(a, b) for a, b in zip(x, y)]


def sigmoid(x):  # numerically stable, matches the reference kernel
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def gelu(x):  # exact erf form (ADR-0004)
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def relu(x):
    return x if x > 0.0 else 0.0


# Inputs: general domain (neg/zero/pos), positive domain (log), binary operand.
X = [-5.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
XPOS = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
Y = [1.0, -2.0, 3.0, -0.5, 2.0, -4.0, 1.5, -1.0, 0.25, -3.0, 2.0, -1.0, 0.5]

UNARY = {
    "neg": (lambda v: -v, X, "x"),
    "exp": (math.exp, X, "x"),
    "log": (math.log, XPOS, "xpos"),
    "tanh": (math.tanh, X, "x"),
    "relu": (relu, X, "x"),
    "gelu": (gelu, X, "x"),
    "sigmoid": (sigmoid, X, "x"),
}
BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}


# Reduction golden. §V10: f32 goldens accumulate in f64 then narrow (round input
# to f32, reduce in f64, round result to f32) — the reference-kernel semantics,
# NOT numpy-native f32 accumulation.
R = [
    [1.0, -2.0, 3.0, 0.5],
    [4.0, 5.0, -6.0, 2.0],
    [-1.0, 0.0, 7.0, -3.0],
]
RROWS, RCOLS = 3, 4

RED_FNS = {
    "sum": sum,
    "mean": lambda xs: sum(xs) / len(xs),
    "max": max,
    "min": min,
}


def _cols(mat):
    return [[mat[i][j] for i in range(len(mat))] for j in range(len(mat[0]))]


def _reduce(mat, axis, fn, cast):
    m = [[cast(v) for v in row] for row in mat]
    if axis == 0:  # over rows → per column
        groups = _cols(m)
        shape = [RCOLS]
    elif axis == 1:  # over columns → per row
        groups = m
        shape = [RROWS]
    else:  # all
        groups = [[v for row in m for v in row]]
        shape = []
    out = [cast(fn(g)) for g in groups]
    return out, shape


def _argmax(mat, axis, cast):
    m = [[cast(v) for v in row] for row in mat]
    if axis == 0:
        groups = _cols(m)
        shape = [RCOLS]
    elif axis == 1:
        groups = m
        shape = [RROWS]
    else:
        groups = [[v for row in m for v in row]]
        shape = []
    idx = [float(g.index(max(g))) for g in groups]
    return idx, shape


def build_reduce():
    ident = lambda v: v
    out = {"input": [v for row in R for v in row], "shape": [RROWS, RCOLS], "ops": {}}
    axkeys = {0: "axis0", 1: "axis1", None: "all"}
    for name, fn in RED_FNS.items():
        out["ops"][name] = {}
        for axis, key in axkeys.items():
            v64, shape = _reduce(R, axis, fn, ident)
            v32, _ = _reduce(R, axis, fn, f32)
            out["ops"][name][key] = {"f64": v64, "f32": [f32(v) for v in v32], "shape": shape}
    out["ops"]["argmax"] = {}
    for axis, key in {0: "axis0", 1: "axis1", None: "flat"}.items():
        idx, shape = _argmax(R, axis, ident)
        out["ops"]["argmax"][key] = {"idx": idx, "shape": shape}
    return out


# BLAS-1 golden. dot/nrm2 accumulate in f64 (§V10); f32 = round inputs to f32,
# accumulate in f64, round result to f32. axpy is elementwise.
BA = [1.0, -2.0, 0.5, 3.0, -1.5, 4.0, -0.25, 2.0]
BB = [0.5, 1.0, -3.0, 2.0, 4.0, -1.0, 0.25, -2.0]
ALPHA = 2.5


def build_blas():
    dot64 = sum(a * b for a, b in zip(BA, BB))
    dot32 = f32(sum(f32(a) * f32(b) for a, b in zip(BA, BB)))
    nrm64 = math.sqrt(sum(a * a for a in BA))
    nrm32 = f32(math.sqrt(sum(f32(a) * f32(a) for a in BA)))
    axpy64 = [ALPHA * a + b for a, b in zip(BA, BB)]
    axpy32 = [f32(ALPHA * f32(a) + f32(b)) for a, b in zip(BA, BB)]
    return {
        "a": BA,
        "b": BB,
        "alpha": ALPHA,
        "dot": {"f64": dot64, "f32": dot32},
        "nrm2": {"f64": nrm64, "f32": nrm32},
        "axpy": {"f64": axpy64, "f32": axpy32},
    }


# GEMM golden. Accumulate products in f64 (§V10); f32 = round inputs to f32,
# accumulate in f64, round result to f32.
GA = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]            # 2x3
GB = [[1.0, 0.0, -1.0, 2.0],
      [2.0, 1.0, 0.0, -1.0],
      [0.0, 3.0, 1.0, 1.0]]                         # 3x4


def matmul(A, B, cast):
    A = [[cast(v) for v in r] for r in A]
    B = [[cast(v) for v in r] for r in B]
    M, K, N = len(A), len(A[0]), len(B[0])
    C = []
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            C.append(cast(s))
    return C, [M, N]


def build_gemm():
    ident = lambda v: v
    c64, shape = matmul(GA, GB, ident)
    c32, _ = matmul(GA, GB, f32)
    return {
        "a": [v for r in GA for v in r], "a_shape": [len(GA), len(GA[0])],
        "b": [v for r in GB for v in r], "b_shape": [len(GB), len(GB[0])],
        "f64": c64, "f32": [f32(v) for v in c32], "shape": shape,
    }


# Loss golden (§T16). CE computed with the same stable per-row logsumexp the
# kernel uses (max-shift); MSE plain. f64 only — losses are scalars.
LP = [0.8, -1.2, 2.0, 0.3, -0.7, 1.5]   # pred 2x3
LT = [1.0, -1.0, 1.8, 0.0, -0.5, 2.0]   # target 2x3
LZ = [[2.0, 1.0, 0.1], [0.5, 2.5, -1.0], [-3.0, 0.0, 4.0]]  # logits 3x3
LY = [0, 1, 2]                            # class targets


def build_losses():
    mse = sum((p - t) ** 2 for p, t in zip(LP, LT)) / len(LP)
    ce = 0.0
    for zrow, y in zip(LZ, LY):
        m = max(zrow)
        lse = m + math.log(sum(math.exp(v - m) for v in zrow))
        ce += lse - zrow[y]
    ce /= len(LZ)
    return {
        "mse": {"pred": LP, "target": LT, "shape": [2, 3], "f64": mse},
        "ce": {"logits": [v for r in LZ for v in r], "shape": [3, 3],
               "targets": [float(y) for y in LY], "f64": ce},
    }


# Optimizer golden (§T17): 3-step trajectories in f64, exactly the documented
# torch/paper update rules (SGD w/ momentum: v=μv+g, p-=lr·v; Adam: Kingma & Ba
# 2015 with bias correction, eps outside the sqrt).
OP0 = [1.0, -2.0, 0.5]
OG = [[0.1, -0.2, 0.3], [0.05, 0.1, -0.1], [-0.2, 0.3, 0.02]]


def build_muon():
    """Muon optimizer golden trajectory — independent numpy (Jordan et al. 2024,
    §R78), current-master reference (lerp momentum, NS5, scale √max(1,R/C)). f64."""
    def ns5(G, steps=5, a=3.4445, b=-4.7750, c=2.0315):
        X = G.copy()
        transposed = False
        if X.shape[0] > X.shape[1]:
            X = X.T.copy()
            transposed = True
        X = X / (np.linalg.norm(X) + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return X.T if transposed else X

    rng = np.random.default_rng(808)
    R, C = 3, 4
    p0 = rng.standard_normal((R, C))
    grads = [rng.standard_normal((R, C)) for _ in range(3)]
    momentum, lr, wd, steps = 0.95, 0.02, 0.1, 5
    buf = np.zeros((R, C))
    p = p0.copy()
    traj = []
    for g in grads:
        buf = momentum * buf + (1 - momentum) * g
        upd = (1 - momentum) * g + momentum * buf  # nesterov
        O = ns5(upd, steps) * max(1.0, R / C) ** 0.5
        p = p * (1 - lr * wd) - lr * O
        traj.append([float(v) for v in p.reshape(-1)])
    flat = lambda t: [float(v) for v in np.asarray(t).reshape(-1)]
    return {
        "R": R, "C": C, "momentum": momentum, "lr": lr, "wd": wd, "ns_steps": steps,
        "p0": flat(p0), "grads": [flat(g) for g in grads], "traj": traj,
    }


def build_optim():
    lr = 0.1
    p = list(OP0)
    sgd = []
    for g in OG:
        p = [pi - lr * gi for pi, gi in zip(p, g)]
        sgd.append(list(p))

    mu = 0.9
    pm, v = list(OP0), [0.0] * 3
    sgdm = []
    for g in OG:
        v = [mu * vi + gi for vi, gi in zip(v, g)]
        pm = [pi - lr * vi for pi, vi in zip(pm, v)]
        sgdm.append(list(pm))

    alr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
    pa, m, vv = list(OP0), [0.0] * 3, [0.0] * 3
    adam = []
    for t, g in enumerate(OG, 1):
        m = [b1 * mi + (1 - b1) * gi for mi, gi in zip(m, g)]
        vv = [b2 * vi + (1 - b2) * gi * gi for vi, gi in zip(vv, g)]
        mh = [mi / (1 - b1**t) for mi in m]
        vh = [vi / (1 - b2**t) for vi in vv]
        pa = [pi - alr * mhi / (math.sqrt(vhi) + eps) for pi, mhi, vhi in zip(pa, mh, vh)]
        adam.append(list(pa))

    # AdamW: decoupled weight decay (Loshchilov & Hutter 2019 Alg.2 = torch)
    wd = 0.1
    pw, mw, vw = list(OP0), [0.0] * 3, [0.0] * 3
    adamw = []
    for t, g in enumerate(OG, 1):
        mw = [b1 * mi + (1 - b1) * gi for mi, gi in zip(mw, g)]
        vw = [b2 * vi + (1 - b2) * gi * gi for vi, gi in zip(vw, g)]
        mh = [mi / (1 - b1**t) for mi in mw]
        vh = [vi / (1 - b2**t) for vi in vw]
        pw = [pi * (1 - alr * wd) - alr * mhi / (math.sqrt(vhi) + eps)
              for pi, mhi, vhi in zip(pw, mh, vh)]
        adamw.append(list(pw))

    # Lion (Chen et al. 2023, Alg.2): c=β1·m+(1−β1)·g; θ−=lr·(sign(c)+λ·θ);
    # m=β2·m+(1−β2)·g (β2 ≠ β1, momentum updated AFTER the step).
    def sign(x):
        return (x > 0) - (x < 0)

    llr, lb1, lb2, lwd = 0.01, 0.9, 0.99, 0.1
    pl, ml = list(OP0), [0.0] * 3
    lion = []
    for g in OG:
        c = [lb1 * mi + (1 - lb1) * gi for mi, gi in zip(ml, g)]
        pl = [pi - llr * (sign(ci) + lwd * pi) for pi, ci in zip(pl, c)]
        ml = [lb2 * mi + (1 - lb2) * gi for mi, gi in zip(ml, g)]
        lion.append(list(pl))

    return {
        "p0": OP0, "grads": OG,
        "sgd": {"lr": lr, "steps": sgd},
        "sgd_momentum": {"lr": lr, "momentum": mu, "steps": sgdm},
        "adam": {"lr": alr, "beta1": b1, "beta2": b2, "eps": eps, "steps": adam},
        "adamw": {"lr": alr, "beta1": b1, "beta2": b2, "eps": eps, "wd": wd, "steps": adamw},
        "lion": {"lr": llr, "beta1": lb1, "beta2": lb2, "wd": lwd, "steps": lion},
    }


def build():
    out = {
        "meta": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "generator": "testdata/gen.py",
            "note": "f32 = round-in/compute-f64/round-out (ADR-0001)",
        },
        "x": X,
        "xpos": XPOS,
        "y": Y,
        "unary": {},
        "binary": {},
        "reduce": build_reduce(),
        "blas": build_blas(),
        "gemm": build_gemm(),
        "losses": build_losses(),
    }
    for name, (fn, xs, which) in UNARY.items():
        out["unary"][name] = {
            "in": which,
            "f64": f64_unary(fn, xs),
            "f32": f32_unary(fn, xs),
        }
    for name, op in BINARY.items():
        out["binary"][name] = {
            "f64": f64_binary(op, X, Y),
            "f32": f32_binary(op, X, Y),
        }
    return out


def build_transformer(root):
    """Transformer goldens from REAL torch in float64 (§T21, §V1)."""
    try:
        import torch
    except ImportError:
        print("torch missing — skipping transformer golden")
        return
    g = torch.Generator().manual_seed(42)

    def rnd(*sh):
        return torch.randn(*sh, generator=g, dtype=torch.float64)

    S = rnd(2, 4)
    sm = torch.softmax(S, dim=-1)

    L = rnd(2, 6)
    gamma, beta, eps = rnd(6), rnd(6), 1e-5
    ln = torch.nn.functional.layer_norm(L, (6,), weight=gamma, bias=beta, eps=eps)

    seq, dm, heads = 3, 8, 2
    dk = dm // heads
    x = rnd(seq, dm)
    Wq, Wk, Wv, Wo = rnd(dm, dm), rnd(dm, dm), rnd(dm, dm), rnd(dm, dm)
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    outs = []
    for h in range(heads):
        q = Q[:, h * dk:(h + 1) * dk]
        k = K[:, h * dk:(h + 1) * dk]
        v = V[:, h * dk:(h + 1) * dk]
        a = torch.softmax((q @ k.T) / math.sqrt(dk), dim=-1)
        outs.append(a @ v)
    mha_out = torch.cat(outs, dim=1) @ Wo

    flat = lambda t: [float(v) for v in t.reshape(-1)]
    data = {
        "softmax": {"x": flat(S), "shape": [2, 4], "y": flat(sm)},
        "layernorm": {"x": flat(L), "shape": [2, 6], "gamma": flat(gamma),
                      "beta": flat(beta), "eps": eps, "y": flat(ln)},
        "mha": {"seq": seq, "dmodel": dm, "heads": heads,
                "x": flat(x), "wq": flat(Wq), "wk": flat(Wk),
                "wv": flat(Wv), "wo": flat(Wo), "y": flat(mha_out),
                "torch": torch.__version__},
    }
    dest = os.path.join(root, "..", "nlp", "testdata")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "transformer.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nlp/testdata/transformer.json (torch", torch.__version__ + ")")


def build_classic(root):
    """classic-ML goldens from REAL sklearn (§T25, §V1)."""
    try:
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn missing — skipping classic golden")
        return
    rng = np.random.default_rng(2024)

    # linear regression: y = Xβ + ε
    X = rng.normal(size=(20, 3))
    beta = np.array([2.0, -1.5, 0.5])
    y = X @ beta + 3.0 + rng.normal(scale=0.1, size=20)
    lr = LinearRegression().fit(X, y)

    # 3-class logistic regression on OVERLAPPING blobs: with penalty=None the
    # MLE exists only if classes overlap (separable data → weights diverge and
    # any stopping point is arbitrary, §B25). Overlap → finite unique optimum.
    centers = np.array([[1.2, 0], [-1.2, 0.8], [0, -1.2]], dtype=float)
    Xc = np.vstack([rng.normal(loc=c, scale=1.3, size=(15, 2)) for c in centers])
    yc = np.repeat([0, 1, 2], 15)
    logr = LogisticRegression(penalty=None, max_iter=50000, tol=1e-14).fit(Xc, yc)
    probs = logr.predict_proba(Xc)

    # kmeans, deterministic init = first 3 points, pure Lloyd
    init = Xc[:3].copy()
    km = KMeans(n_clusters=3, init=init, n_init=1, algorithm="lloyd",
                max_iter=100, tol=0).fit(Xc)

    # PCA via covariance eigenstructure
    Xp = rng.normal(size=(15, 4)) @ np.diag([3.0, 2.0, 0.5, 0.1])
    pca = PCA(n_components=2).fit(Xp)

    flat = lambda a: [float(v) for v in np.asarray(a).reshape(-1)]
    data = {
        "linreg": {"X": flat(X), "n": 20, "d": 3, "y": flat(y),
                   "coef": flat(lr.coef_), "intercept": float(lr.intercept_),
                   "pred": flat(lr.predict(X))},
        "logreg": {"X": flat(Xc), "n": 45, "d": 2, "k": 3,
                   "y": [int(v) for v in yc], "probs": flat(probs)},
        "kmeans": {"X": flat(Xc), "n": 45, "d": 2, "k": 3, "init": flat(init),
                   "centers": flat(km.cluster_centers_),
                   "labels": [int(v) for v in km.labels_]},
        "pca": {"X": flat(Xp), "n": 15, "d": 4, "ncomp": 2,
                "explained_variance": flat(pca.explained_variance_),
                "components": flat(pca.components_)},
        "sklearn": __import__("sklearn").__version__,
    }
    dest = os.path.join(root, "..", "classic", "testdata")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "classic.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote classic/testdata/classic.json (sklearn)")


def build_dpo(root):
    """DPO loss golden (§T48). Independent numpy computation of Rafailov 2023 Eq.7."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    beta = 0.1
    pc = np.array([-2.0, -3.5, -1.0, -4.0])   # policy log-probs, chosen
    pl = np.array([-2.5, -3.0, -2.0, -3.5])   # policy log-probs, rejected
    rc = np.array([-2.2, -3.6, -1.5, -4.2])   # reference log-probs, chosen
    rl = np.array([-2.1, -2.9, -1.8, -3.0])   # reference log-probs, rejected
    delta = beta * ((pc - rc) - (pl - rl))
    # loss = mean(-log sigmoid(delta)) = mean(softplus(-delta)) = mean(log(1+exp(-delta)))
    loss = float(np.mean(np.logaddexp(0.0, -delta)))
    data = {
        "beta": beta,
        "pc": [float(v) for v in pc], "pl": [float(v) for v in pl],
        "rc": [float(v) for v in rc], "rl": [float(v) for v in rl],
        "loss": loss,
    }
    with open(os.path.join(dest, "dpo.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/dpo.json")


def build_labelsmooth(root):
    """Label-smoothing CE golden (§T52). Independent numpy = −Σ q'·logsoftmax(z),
    q'(k)=(1−ε)·onehot + ε/K (Szegedy 2016 / Vaswani 2017)."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    eps = 0.1
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, -1.0], [-3.0, 0.0, 4.0]])
    targets = [0, 1, 2]
    K = logits.shape[1]
    z = logits - logits.max(axis=1, keepdims=True)
    lsm = z - np.log(np.exp(z).sum(axis=1, keepdims=True))  # log-softmax
    loss = 0.0
    for i, ti in enumerate(targets):
        q = np.full(K, eps / K)
        q[ti] += (1 - eps)
        loss += float(-np.sum(q * lsm[i]))
    loss /= len(targets)
    data = {
        "epsilon": eps, "shape": list(logits.shape),
        "logits": [float(v) for v in logits.reshape(-1)],
        "targets": [float(t) for t in targets], "loss": loss,
    }
    with open(os.path.join(dest, "labelsmooth.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/labelsmooth.json")


def build_distill(root):
    """KD soft-target loss golden (§T51). Independent numpy = Hinton 2015 T²·KL(p‖q)."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    temp = 2.0
    student = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, -1.0]])
    teacher = np.array([[3.0, 0.5, -0.5], [0.2, 3.0, 0.1]])

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    p = softmax(teacher / temp)  # soft teacher targets
    q = softmax(student / temp)  # soft student dist
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    loss = float(np.mean(temp * temp * kl))
    data = {
        "temperature": temp, "shape": list(student.shape),
        "student": [float(v) for v in student.reshape(-1)],
        "teacher": [float(v) for v in teacher.reshape(-1)],
        "loss": loss,
    }
    with open(os.path.join(dest, "distill.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/distill.json")


def build_moe(root):
    """MoE load-balancing aux loss golden (§T61). Independent numpy = Switch eq.4."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    alpha = 0.01
    logits = np.array([[2.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, 0.0], [1.0, 1.0, 4.0, 0.0]])
    assignments = [0, 1, 2]  # top-1 (argmax) per token
    T, N = logits.shape
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    P = probs.mean(axis=0)  # mean router prob per expert
    f = np.zeros(N)
    for a in assignments:
        f[a] += 1
    f /= T
    loss = float(alpha * N * np.sum(f * P))
    data = {
        "alpha": alpha, "shape": [int(T), int(N)],
        "logits": [float(v) for v in logits.reshape(-1)],
        "assignments": [float(a) for a in assignments], "loss": loss,
    }
    with open(os.path.join(dest, "moe.json"), "w") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")
    print("wrote nn/testdata/moe.json")


def build_kto(root):
    """KTO loss golden (§T59). Independent numpy = Ethayarajh 2024 σ(±(r−z_ref))."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    beta, z_ref, lam_d, lam_u = 0.1, 0.5, 1.0, 1.0
    pl = np.array([-2.0, -3.5, -1.0, -4.0])  # policy logps
    rl = np.array([-2.2, -3.0, -1.5, -4.2])  # reference logps
    labels = np.array([1, 0, 1, 0])          # desirable / undesirable
    r = beta * (pl - rl)
    loss = 0.0
    for i in range(len(pl)):
        if labels[i]:
            loss += lam_d * sigmoid(z_ref - r[i])
        else:
            loss += lam_u * sigmoid(r[i] - z_ref)
    loss = float(loss / len(pl))
    data = {
        "beta": beta, "z_ref": z_ref, "lambda_d": lam_d, "lambda_u": lam_u,
        "pl": [float(v) for v in pl], "rl": [float(v) for v in rl],
        "labels": [float(v) for v in labels], "loss": loss,
    }
    with open(os.path.join(dest, "kto.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/kto.json")


def build_ipo(root):
    """IPO loss golden (§T58). Independent numpy = Azar 2023 (h − 1/(2β))²."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    beta = 0.1
    target = 1.0 / (2.0 * beta)
    pc = np.array([-2.0, -3.5, -1.0, -4.0])
    pl = np.array([-2.5, -3.0, -2.0, -3.5])
    rc = np.array([-2.2, -3.6, -1.5, -4.2])
    rl = np.array([-2.1, -2.9, -1.8, -3.0])
    h = (pc - rc) - (pl - rl)
    loss = float(np.mean((h - target) ** 2))
    data = {
        "beta": beta,
        "pc": [float(v) for v in pc], "pl": [float(v) for v in pl],
        "rc": [float(v) for v in rc], "rl": [float(v) for v in rl],
        "loss": loss,
    }
    with open(os.path.join(dest, "ipo.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/ipo.json")


def build_gae(root):
    """GAE golden (§T50). Independent FORWARD-SUM Â_t=Σ(γλ)^l δ_{t+l} vs the Go
    backward recursion; no mid-episode dones (nonterminal=1, last uses bootstrap)."""
    dest = os.path.join(root, "..", "rl", "testdata")
    os.makedirs(dest, exist_ok=True)
    gamma, lam = 0.99, 0.95
    rewards = np.array([1.0, 0.0, -0.5, 2.0, 0.0])
    values = np.array([0.5, 0.6, 0.4, 1.5, 0.3])
    next_value = 0.2
    T = len(rewards)
    delta = np.array([rewards[t] + gamma * (values[t + 1] if t + 1 < T else next_value) - values[t]
                      for t in range(T)])
    adv = np.array([sum((gamma * lam) ** l * delta[t + l] for l in range(T - t)) for t in range(T)])
    returns = adv + values
    data = {
        "gamma": gamma, "lam": lam, "next_value": next_value,
        "rewards": [float(v) for v in rewards], "values": [float(v) for v in values],
        "advantages": [float(v) for v in adv], "returns": [float(v) for v in returns],
    }
    with open(os.path.join(dest, "gae.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote rl/testdata/gae.json")


def build_ppo(root):
    """PPO clipped surrogate loss golden (§T49). Independent numpy = Schulman 2017 Eq.7."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    eps = 0.2
    lp_new = np.array([-1.0, -2.0, -1.5, -3.0, -0.5])
    lp_old = np.array([-1.2, -1.8, -1.5, -2.5, -1.0])
    adv = np.array([1.0, -1.0, 0.5, 2.0, -0.5])  # mix of ± advantages
    r = np.exp(lp_new - lp_old)
    surr1 = r * adv
    surr2 = np.clip(r, 1 - eps, 1 + eps) * adv
    loss = float(-np.mean(np.minimum(surr1, surr2)))
    data = {
        "epsilon": eps,
        "lp_new": [float(v) for v in lp_new], "lp_old": [float(v) for v in lp_old],
        "adv": [float(v) for v in adv], "loss": loss,
    }
    with open(os.path.join(dest, "ppo.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/ppo.json")


def build_half(root):
    """f16/bf16 conversion golden (§T27). numpy float16 + torch bfloat16, RNE."""
    dest = os.path.join(root, "..", "tensor", "testdata")
    os.makedirs(dest, exist_ok=True)
    vals = [0.0, 1.0, -1.0, 0.5, -0.5, 3.14159265, -2.71828, 65504.0, 1e-5,
            6.1035e-05, 100.0, -0.1, 1.0 / 3.0, 12345.678, 2.0 ** -14, 2.0 ** -24]
    xf = np.array(vals, dtype=np.float32)

    f16 = xf.astype(np.float16)
    f16_bits = [int(b) for b in f16.view(np.uint16)]
    f16_back = [float(v) for v in f16.astype(np.float32)]

    data = {"x": [float(v) for v in xf], "f16_bits": f16_bits, "f16_back": f16_back}
    try:
        import torch
        tb = torch.tensor(xf.tolist(), dtype=torch.float32).to(torch.bfloat16)
        bf16_bits = [int(b) & 0xFFFF for b in tb.view(torch.int16).tolist()]
        bf16_back = [float(v) for v in tb.to(torch.float32).tolist()]
        data["bf16_bits"] = bf16_bits
        data["bf16_back"] = bf16_back
    except ImportError:
        print("torch missing — bf16 golden skipped")

    with open(os.path.join(dest, "half.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote tensor/testdata/half.json")


def build_lora(root):
    """LoRA forward golden in torch f64 (§T40). Our [in,out] weight convention:
    y = x·W + (alpha/r)·(x·A)·B, A[in,r], B[r,out], W frozen."""
    try:
        import torch
    except ImportError:
        print("torch missing — skipping lora golden")
        return
    g = torch.Generator().manual_seed(2468)

    def rnd(*sh):
        return torch.randn(*sh, generator=g, dtype=torch.float64)

    M, IN, OUT, R = 3, 6, 5, 2
    alpha = 4.0
    x = rnd(M, IN)
    W = rnd(IN, OUT)
    A = rnd(IN, R)   # down
    B = rnd(R, OUT)  # up (non-zero here to exercise the delta path)
    y = x @ W + (alpha / R) * ((x @ A) @ B)

    flat = lambda t: [float(v) for v in t.reshape(-1)]
    data = {
        "m": M, "in": IN, "out": OUT, "r": R, "alpha": alpha,
        "x": flat(x), "w": flat(W), "a": flat(A), "b": flat(B), "y": flat(y),
    }
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "lora.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/lora.json (torch)")


def build_llama(root):
    """RMSNorm + RoPE goldens in torch f64, HF/Llama conventions (§T38)."""
    try:
        import torch
    except ImportError:
        print("torch missing — skipping llama golden")
        return
    g = torch.Generator().manual_seed(99)

    def rnd(*sh):
        return torch.randn(*sh, generator=g, dtype=torch.float64)

    # RMSNorm: y = x/sqrt(mean(x^2)+eps) * gamma  (last-axis, no mean-sub, no bias)
    D, eps = 6, 1e-5
    rx = rnd(3, D)
    gamma = rnd(D)
    rms = rx * torch.rsqrt(rx.pow(2).mean(-1, keepdim=True) + eps) * gamma

    # RoPE (HF rotate_half): head dim hd, positions 0..seq-1, base 10000
    seq, hd, base = 4, 8, 10000.0
    q = rnd(seq, hd)
    half = hd // 2
    inv_freq = base ** (-(torch.arange(0, half, dtype=torch.float64) * 2 / hd))
    pos = torch.arange(seq, dtype=torch.float64)
    freqs = torch.outer(pos, inv_freq)          # [seq, half]
    emb = torch.cat([freqs, freqs], dim=-1)      # [seq, hd]
    cos, sin = emb.cos(), emb.sin()

    def rotate_half(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    q_rot = q * cos + rotate_half(q) * sin

    flat = lambda t: [float(v) for v in t.reshape(-1)]
    data = {
        "rmsnorm": {"x": flat(rx), "shape": [3, D], "gamma": flat(gamma), "eps": eps, "y": flat(rms)},
        "rope": {"q": flat(q), "seq": seq, "hd": hd, "base": base, "y": flat(q_rot)},
    }
    dest = os.path.join(root, "..", "nlp", "testdata")
    with open(os.path.join(dest, "llama.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nlp/testdata/llama.json (torch)")


def build_llama2(root):
    """SiLU / SwiGLU / GQA goldens in torch f64, HF conventions (§T38b)."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("torch missing — skipping llama2 golden")
        return
    g = torch.Generator().manual_seed(321)

    def rnd(*sh):
        return torch.randn(*sh, generator=g, dtype=torch.float64)

    # SiLU
    sx = rnd(10)
    silu = F.silu(sx)

    # SwiGLU FFN: down( SiLU(x @ Wg) * (x @ Wu) )
    seq, D, H = 3, 8, 16
    sw_x = rnd(seq, D)
    Wg, Wu, Wd = rnd(D, H), rnd(D, H), rnd(H, D)
    swiglu = (F.silu(sw_x @ Wg) * (sw_x @ Wu)) @ Wd

    # GQA: n_heads=4 query heads, n_kv=2 kv heads, dk=3 → q[seq,12], k/v[seq,6]
    seqg, nh, nkv, dk = 4, 4, 2, 3
    q = rnd(seqg, nh * dk)
    k = rnd(seqg, nkv * dk)
    v = rnd(seqg, nkv * dk)
    rep = nh // nkv
    scale = 1.0 / math.sqrt(dk)
    outs = []
    for h in range(nh):
        kh = h // rep
        qh = q[:, h * dk:(h + 1) * dk]
        kk = k[:, kh * dk:(kh + 1) * dk]
        vv = v[:, kh * dk:(kh + 1) * dk]
        a = torch.softmax((qh @ kk.T) * scale, dim=-1)  # non-causal
        outs.append(a @ vv)
    gqa = torch.cat(outs, dim=1)  # [seq, nh*dk]

    flat = lambda t: [float(v) for v in t.reshape(-1)]
    data = {
        "silu": {"x": flat(sx), "y": flat(silu)},
        "swiglu": {"x": flat(sw_x), "shape": [seq, D], "h": H,
                   "wg": flat(Wg), "wu": flat(Wu), "wd": flat(Wd), "y": flat(swiglu)},
        "gqa": {"q": flat(q), "k": flat(k), "v": flat(v),
                "seq": seqg, "nh": nh, "nkv": nkv, "dk": dk, "y": flat(gqa)},
    }
    dest = os.path.join(root, "..", "nlp", "testdata")
    with open(os.path.join(dest, "llama2.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nlp/testdata/llama2.json (torch)")


def build_tokenizer(root):
    """Export gpt2 BPE ranks + encode goldens from the real tiktoken (§T37)."""
    try:
        import tiktoken
    except ImportError:
        print("tiktoken missing — skipping tokenizer golden")
        return
    import base64
    enc = tiktoken.get_encoding("gpt2")
    dest = os.path.join(root, "..", "nlp", "testdata")
    os.makedirs(dest, exist_ok=True)

    ranks = enc._mergeable_ranks
    # sanity: all 256 single bytes must be base tokens (byte-level base vocab)
    missing = [b for b in range(256) if bytes([b]) not in ranks]
    with open(os.path.join(dest, "gpt2_ranks.txt"), "w") as f:
        for b, r in ranks.items():
            f.write(base64.b64encode(b).decode() + " " + str(r) + "\n")

    samples = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "  leading and trailing spaces  ",
        "numbers 123 and 4567 mixed",
        "don't can't won't it's",
        "Café résumé naïve",
        "tabs\tand\nnewlines here",
        "PascalCase snake_case kebab-case",
        "punctuation!?; (parens) [brackets]",
        "",
        "a",
        "   ",
    ]
    golden = [{"text": s, "ids": enc.encode(s)} for s in samples]
    with open(os.path.join(dest, "tokenizer.json"), "w") as f:
        json.dump({"vocab": enc.n_vocab, "missing_single_bytes": missing, "samples": golden,
                   "tiktoken": True}, f, indent=1)
        f.write("\n")
    print("wrote nlp/testdata/{gpt2_ranks.txt, tokenizer.json} (missing bytes:", len(missing), ")")


def build_cv(root):
    """conv2d / pooling goldens from REAL torch f64 (§T24, §V1)."""
    try:
        import torch
    except ImportError:
        print("torch missing — skipping cv golden")
        return
    g = torch.Generator().manual_seed(777)

    def rnd(*sh):
        return torch.randn(*sh, generator=g, dtype=torch.float64)

    x = rnd(2, 3, 5, 6)
    w = rnd(4, 3, 3, 3)
    b = rnd(4)
    F = torch.nn.functional
    flat = lambda t: [float(v) for v in t.reshape(-1)]
    cases = {
        "x": {"shape": list(x.shape), "values": flat(x)},
        "w": {"shape": list(w.shape), "values": flat(w)},
        "b": flat(b),
        "conv_s1p0": {"out": flat(F.conv2d(x, w, bias=b)), "shape": list(F.conv2d(x, w, bias=b).shape)},
        "conv_s2p1": {"out": flat(F.conv2d(x, w, bias=b, stride=2, padding=1)),
                      "shape": list(F.conv2d(x, w, bias=b, stride=2, padding=1).shape)},
        "conv_nobias": {"out": flat(F.conv2d(x, w)), "shape": list(F.conv2d(x, w).shape)},
        "maxpool_k2": {"out": flat(F.max_pool2d(x, 2)), "shape": list(F.max_pool2d(x, 2).shape)},
        "avgpool_k2s1": {"out": flat(F.avg_pool2d(x, 2, stride=1)),
                         "shape": list(F.avg_pool2d(x, 2, stride=1).shape)},
    }
    dest = os.path.join(root, "..", "backend", "ref", "testdata")
    with open(os.path.join(dest, "cv.json"), "w") as f:
        json.dump(cases, f, indent=1)
        f.write("\n")
    print("wrote backend/ref/testdata/cv.json (torch)")


def build_gpt(root):
    """Tiny GPT (pre-LN, causal, exact GELU) in REAL torch f64 → weights via
    safetensors + expected logits (§T23, §V1)."""
    try:
        import torch
        from safetensors.numpy import save_file
    except ImportError:
        print("torch/safetensors missing — skipping gpt golden")
        return
    g = torch.Generator().manual_seed(1234)
    V, CTX, D, H, L = 17, 8, 8, 2, 2
    dk = D // H
    eps = 1e-5

    def rnd(*sh):
        return torch.randn(*sh, generator=g, dtype=torch.float64) * 0.5

    W = {"tok_emb": rnd(V, D), "pos_emb": rnd(CTX, D)}
    for l in range(L):
        p = f"blocks.{l}."
        W[p + "ln1.gamma"] = rnd(D)
        W[p + "ln1.beta"] = rnd(D) * 0.1
        for nm in ("wq", "wk", "wv", "wo"):
            W[p + "attn." + nm] = rnd(D, D)
        W[p + "ln2.gamma"] = rnd(D)
        W[p + "ln2.beta"] = rnd(D) * 0.1
        W[p + "ffn.w1"] = rnd(D, 4 * D)
        W[p + "ffn.b1"] = rnd(4 * D) * 0.1
        W[p + "ffn.w2"] = rnd(4 * D, D)
        W[p + "ffn.b2"] = rnd(D) * 0.1
    W["lnf.gamma"] = rnd(D)
    W["lnf.beta"] = rnd(D) * 0.1
    W["head"] = rnd(D, V)

    def ln(x, gm, bt):
        return torch.nn.functional.layer_norm(x, (D,), weight=gm, bias=bt, eps=eps)

    def attn(x, l):
        p = f"blocks.{l}."
        Q, K, Vv = x @ W[p + "attn.wq"], x @ W[p + "attn.wk"], x @ W[p + "attn.wv"]
        seq = x.shape[0]
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1)
        outs = []
        for h in range(H):
            q = Q[:, h * dk:(h + 1) * dk]
            k = K[:, h * dk:(h + 1) * dk]
            v = Vv[:, h * dk:(h + 1) * dk]
            s = (q @ k.T) / math.sqrt(dk)
            s = s.masked_fill(mask, float("-inf"))
            outs.append(torch.softmax(s, dim=-1) @ v)
        return torch.cat(outs, 1) @ W[p + "attn.wo"]

    def forward(tokens):
        x = W["tok_emb"][tokens] + W["pos_emb"][: len(tokens)]
        for l in range(L):
            p = f"blocks.{l}."
            x = x + attn(ln(x, W[p + "ln1.gamma"], W[p + "ln1.beta"]), l)
            h = ln(x, W[p + "ln2.gamma"], W[p + "ln2.beta"])
            h = torch.nn.functional.gelu(h @ W[p + "ffn.w1"] + W[p + "ffn.b1"], approximate="none")
            x = x + (h @ W[p + "ffn.w2"] + W[p + "ffn.b2"])
        x = ln(x, W["lnf.gamma"], W["lnf.beta"])
        return x @ W["head"]

    tokens = [3, 1, 4, 1, 5]
    targets = [1, 2, 0, 3, 2]  # next-token-style labels for the training loss

    # full backward: enable grads on all weights, loss = mean CE, backward
    for v in W.values():
        v.requires_grad_(True)
    logits = forward(tokens)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor(targets))
    loss.backward()
    grads = {k: [float(x) for x in v.grad.reshape(-1)] for k, v in W.items()}

    dest = os.path.join(root, "..", "nlp", "testdata")
    os.makedirs(dest, exist_ok=True)
    save_file({k: v.detach().numpy() for k, v in W.items()}, os.path.join(dest, "gpt.safetensors"))
    with open(os.path.join(dest, "gpt.json"), "w") as f:
        json.dump({
            "config": {"vocab": V, "ctx": CTX, "dim": D, "heads": H, "layers": L, "eps": eps},
            "tokens": tokens,
            "targets": targets,
            "logits": [float(v) for v in logits.detach().reshape(-1)],
            "loss": float(loss.detach()),
            "grads": grads,
            "torch": torch.__version__,
        }, f, indent=1)
        f.write("\n")
    print("wrote nlp/testdata/{gpt.safetensors, gpt.json} (+ torch grads)")


def build_qmatmul(root):
    """Quantized-weight matmul golden (§T39): y = x @ dequant(W)ᵀ, W stored Q8_0
    and Q4_0 via the official gguf lib. Go dequantizes on-the-fly and must match."""
    try:
        import gguf
        from gguf import GGMLQuantizationType as Q
        from gguf.quants import quantize, dequantize
    except ImportError:
        print("gguf lib missing — skipping qmatmul golden")
        return
    import base64
    rng = np.random.default_rng(555)
    M, N, K = 3, 4, 64  # K multiple of 32
    x = rng.normal(size=(M, K)).astype(np.float32)
    W = rng.normal(size=(N, K)).astype(np.float32)  # [out, in]

    q8 = quantize(W, Q.Q8_0)
    q4 = quantize(W, Q.Q4_0)
    # f64 accumulation then narrow to f32 — matches the Go kernel (§V10)
    dq8 = dequantize(q8, Q.Q8_0).astype(np.float64)
    dq4 = dequantize(q4, Q.Q4_0).astype(np.float64)
    xf = x.astype(np.float64)
    yq8 = (xf @ dq8.T).astype(np.float32)
    yq4 = (xf @ dq4.T).astype(np.float32)

    flat = lambda a: [float(v) for v in np.asarray(a).reshape(-1)]
    dest = os.path.join(root, "..", "format", "gguf", "testdata")
    data = {
        "m": M, "n": N, "k": K,
        "x": flat(x),
        "q8_bytes": base64.b64encode(q8.tobytes()).decode(),
        "q4_bytes": base64.b64encode(q4.tobytes()).decode(),
        "y_q8": flat(yq8),
        "y_q4": flat(yq4),
    }
    with open(os.path.join(dest, "qmatmul.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote format/gguf/testdata/qmatmul.json")


def build_gguf(root):
    """Sample GGUF + expected dequantized values via the official gguf lib
    (llama.cpp project) — §T22, §V1."""
    try:
        import gguf
        from gguf import GGMLQuantizationType as Q
        from gguf.quants import quantize, dequantize
    except ImportError:
        print("gguf lib missing — skipping gguf golden")
        return
    dest = os.path.join(root, "..", "format", "gguf", "testdata")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "sample.gguf")

    dense = np.array([[1.5, -2.25, 3.0], [0.5, -0.125, 4.75]], dtype=np.float32)
    qsrc = np.linspace(-4, 4, 64, dtype=np.float32)
    half = np.array([0.5, -1.5, 2.25, -3.0, 65504.0, -0.0625, 1.0, -1.0], dtype=np.float16)

    q8 = quantize(qsrc, Q.Q8_0)
    q4 = quantize(qsrc, Q.Q4_0)

    w = gguf.GGUFWriter(path, arch="goai-test")
    w.add_uint32("goai.test_u32", 7)
    w.add_string("goai.note", "golden")
    w.add_tensor("dense_f32", dense)
    w.add_tensor("half_f16", half)  # float16 array → F16 automatically
    # raw_shape for pre-quantized tensors is the BYTE shape; gguf-py converts
    # it back to logical dims via the block geometry.
    w.add_tensor("q8", q8, raw_shape=[len(q8)], raw_dtype=Q.Q8_0)
    w.add_tensor("q4", q4, raw_shape=[len(q4)], raw_dtype=Q.Q4_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    expected = {
        "dense_f32": {"shape": [2, 3], "values": [float(v) for v in dense.reshape(-1)]},
        "half_f16": {"shape": [8], "values": [float(v) for v in half.astype(np.float32)]},
        "q8": {"shape": [64], "values": [float(v) for v in dequantize(q8, Q.Q8_0)]},
        "q4": {"shape": [64], "values": [float(v) for v in dequantize(q4, Q.Q4_0)]},
    }
    with open(os.path.join(dest, "expected.json"), "w") as f:
        json.dump(expected, f, indent=1)
        f.write("\n")
    print("wrote format/gguf/testdata/{sample.gguf, expected.json}")


def write_safetensors_sample(root):
    """Emit a reference .safetensors file via the official lib (§T19, §V1)."""
    try:
        from safetensors.numpy import save_file
    except ImportError:
        print("safetensors lib missing — skipping sample")
        return
    dest = os.path.join(root, "..", "format", "safetensors", "testdata")
    os.makedirs(dest, exist_ok=True)
    tensors = {
        "w": np.array([[1.5, -2.0, 3.25], [4.0, 5.5, -6.75]], dtype=np.float64),
        "b": np.array([0.5, -1.5, 2.5], dtype=np.float32),
        "scalar": np.array(42.0, dtype=np.float64),
    }
    save_file(tensors, os.path.join(dest, "ref.safetensors"),
              metadata={"format": "goai-golden", "gen": "testdata/gen.py"})
    print("wrote format/safetensors/testdata/ref.safetensors")


def write_npy_samples(root):
    """Emit real numpy .npy files to validate the Pure-Go reader (§T10).

    These are the §V16 foreign-convention anchors for format/npy: written by the
    reference numpy, not by our writer, so a symmetric round-trip cannot hide a
    header or byte-order error (§B67).

    The values below reproduce the committed fixtures BYTE-FOR-BYTE (verified
    against numpy 2.5.1), so regenerating is a no-op unless numpy itself changes
    what it emits — which is exactly the drift this generator exists to surface.
    Do not "tidy" the values: format/npy/npy_test.go asserts them literally.

    Historical note worth keeping: this function used to write into
    internal/npy/testdata. That package has ZERO importers, while format/npy has
    35 — so the only reproducible numpy-produced fixtures in the tree were feeding
    a package nothing used, and the live reader's fixtures had no generator at
    all. It now targets the live package.
    """
    dest = os.path.join(root, "..", "format", "npy", "testdata")
    os.makedirs(dest, exist_ok=True)
    a64 = np.array([[1.5, -2.0, 3.25], [4.0, 5.5, -6.75]], dtype=np.float64)
    a32 = np.array([0.5, 1.5, 2.5, -3.5], dtype=np.float32)
    np.save(os.path.join(dest, "ref_f64.npy"), a64)
    np.save(os.path.join(dest, "ref_f32.npy"), a32)
    print("wrote", os.path.relpath(dest, os.path.join(root, "..")), "ref_*.npy")


def build_grpo(root):
    """GRPO loss + group advantage golden — independent numpy from DeepSeekMath
    (Shao et al. 2024 arXiv:2402.03300 eqs 3-4, §R69). Population std + eps=1e-4
    (verl/TRL), k3 KL penalty, ε=0.2, β=0.04."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    eps_clip, beta, adv_eps = 0.2, 0.04, 1e-4

    # group advantage: two groups of rewards → normalized advantages
    rewards = np.array([1.0, 0.0, 0.5, -0.5])
    mean = rewards.mean()
    std = rewards.std(ddof=0)  # population
    adv_group = (rewards - mean) / (std + adv_eps)

    # GRPO loss on a batch of per-token log-probs (advantage broadcast per token)
    lp_new = np.array([-1.0, -2.0, -1.5, -3.0, -0.5])
    lp_old = np.array([-1.2, -1.8, -1.5, -2.5, -1.0])
    lp_ref = np.array([-1.1, -2.1, -1.4, -2.9, -0.7])
    adv = np.array([1.0, -1.0, 0.5, 2.0, -0.5])
    r = np.exp(lp_new - lp_old)
    surr = np.minimum(r * adv, np.clip(r, 1 - eps_clip, 1 + eps_clip) * adv)
    delta = lp_ref - lp_new
    kl = np.exp(delta) - delta - 1.0  # Schulman k3
    loss = float(-np.mean(surr - beta * kl))

    data = {
        "epsilon": eps_clip, "beta": beta, "adv_eps": adv_eps,
        "rewards": [float(v) for v in rewards],
        "adv_group": [float(v) for v in adv_group],
        "lp_new": [float(v) for v in lp_new], "lp_old": [float(v) for v in lp_old],
        "lp_ref": [float(v) for v in lp_ref], "adv": [float(v) for v in adv],
        "loss": loss,
    }
    with open(os.path.join(dest, "grpo.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/grpo.json")


def build_conv1d(root):
    """Causal depthwise 1-D conv golden — independent numpy (Mamba mixing conv,
    §R77). out[t,c] = Σ_k w[c,k]·x_pad[t+k,c] + b[c], left-pad K-1 zeros."""
    dest = os.path.join(root, "..", "autograd", "testdata")
    os.makedirs(dest, exist_ok=True)
    rng = np.random.default_rng(707)
    L, D, K = 5, 3, 4
    x = rng.standard_normal((L, D))
    w = rng.standard_normal((D, K))
    b = rng.standard_normal(D)
    xpad = np.concatenate([np.zeros((K - 1, D)), x], axis=0)  # left-pad K-1
    y = np.zeros((L, D))
    for t in range(L):
        for c in range(D):
            for k in range(K):
                y[t, c] += w[c, k] * xpad[t + k, c]
            y[t, c] += b[c]

    flat = lambda t: [float(v) for v in np.asarray(t).reshape(-1)]
    data = {"L": L, "D": D, "K": K, "x": flat(x), "w": flat(w), "b": flat(b), "y": flat(y)}
    with open(os.path.join(dest, "conv1d.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote autograd/testdata/conv1d.json")


def build_ssm(root):
    """Mamba selective-scan (S6) golden — independent numpy from the official
    selective_scan_ref (Gu & Dao 2023 arXiv:2312.00752 §3.2, §R76). With skip."""
    dest = os.path.join(root, "..", "autograd", "testdata")
    os.makedirs(dest, exist_ok=True)
    rng = np.random.default_rng(606)
    L, D, N = 4, 3, 2
    u = rng.standard_normal((L, D))
    delta = np.log1p(np.exp(rng.standard_normal((L, D))))  # softplus → Δ>0
    A = -np.exp(rng.standard_normal((D, N)))               # negative (stable Ā∈(0,1))
    B = rng.standard_normal((L, N))
    C = rng.standard_normal((L, N))
    Dskip = rng.standard_normal(D)

    h = np.zeros((D, N))
    y = np.zeros((L, D))
    for t in range(L):
        for d in range(D):
            for n in range(N):
                abar = np.exp(delta[t, d] * A[d, n])
                h[d, n] = abar * h[d, n] + delta[t, d] * B[t, n] * u[t, d]
                y[t, d] += C[t, n] * h[d, n]
            y[t, d] += Dskip[d] * u[t, d]

    flat = lambda t: [float(v) for v in np.asarray(t).reshape(-1)]
    data = {
        "L": L, "D": D, "N": N,
        "u": flat(u), "delta": flat(delta), "A": flat(A), "B": flat(B), "C": flat(C),
        "Dskip": flat(Dskip), "y": flat(y),
    }
    with open(os.path.join(dest, "ssm.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote autograd/testdata/ssm.json")


def build_mla(root):
    """Multi-head Latent Attention golden — independent numpy from DeepSeek-V2
    (Liu et al. 2024 arXiv:2405.04434 §2.1, §R74). Op-level (attention core) +
    full-layer (all projections). Internal per-head decoupled RoPE, causal."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    rng = np.random.default_rng(505)
    seq, d, heads, dh, dR, dc, dcq, base = 5, 8, 2, 4, 4, 6, 6, 10000.0
    hdh = heads * dh

    def rope(src, nheads):  # src [seq, nheads*dR] → roped [seq, nheads*dR]
        half = dR // 2
        out = np.zeros_like(src)
        for p in range(seq):
            for h in range(nheads):
                for e in range(half):
                    th = base ** (-(2.0 * e) / dR)
                    c, s = np.cos(p * th), np.sin(p * th)
                    x0, x1 = src[p, h * dR + e], src[p, h * dR + e + half]
                    out[p, h * dR + e] = x0 * c - x1 * s
                    out[p, h * dR + e + half] = x1 * c + x0 * s
        return out

    h = rng.standard_normal((seq, d))
    W_DKV = rng.standard_normal((d, dc)) * 0.3
    W_UK = rng.standard_normal((dc, hdh)) * 0.3
    W_UV = rng.standard_normal((dc, hdh)) * 0.3
    W_KR = rng.standard_normal((d, dR)) * 0.3
    W_DQ = rng.standard_normal((d, dcq)) * 0.3
    W_UQ = rng.standard_normal((dcq, hdh)) * 0.3
    W_QR = rng.standard_normal((dcq, heads * dR)) * 0.3
    W_O = rng.standard_normal((hdh, d)) * 0.3

    cKV = h @ W_DKV
    kC, vC = cKV @ W_UK, cKV @ W_UV
    kRpre = h @ W_KR
    cQ = h @ W_DQ
    qC = cQ @ W_UQ
    qRpre = cQ @ W_QR

    qRrot = rope(qRpre, heads)
    kRrot = rope(kRpre, 1)
    scale = 1.0 / np.sqrt(dh + dR)
    O = np.zeros((seq, hdh))
    for hh in range(heads):
        hc = hh * dh
        for i in range(seq):
            score = np.full(i + 1, -np.inf)
            for j in range(i + 1):  # causal
                sc = qC[i, hc:hc + dh] @ kC[j, hc:hc + dh]
                sc += qRrot[i, hh * dR:(hh + 1) * dR] @ kRrot[j]
                score[j] = sc * scale
            score = np.exp(score - score.max())
            score /= score.sum()
            for j in range(i + 1):
                O[i, hc:hc + dh] += score[j] * vC[j, hc:hc + dh]
    u = O @ W_O

    flat = lambda t: [float(v) for v in np.asarray(t).reshape(-1)]
    data = {
        "seq": seq, "d": d, "heads": heads, "dh": dh, "dR": dR, "dc": dc, "dcq": dcq, "base": base,
        "h": flat(h), "W_DKV": flat(W_DKV), "W_UK": flat(W_UK), "W_UV": flat(W_UV),
        "W_KR": flat(W_KR), "W_DQ": flat(W_DQ), "W_UQ": flat(W_UQ), "W_QR": flat(W_QR), "W_O": flat(W_O),
        "qC": flat(qC), "kC": flat(kC), "vC": flat(vC), "qRpre": flat(qRpre), "kRpre": flat(kRpre),
        "O": flat(O), "u": flat(u),
    }
    with open(os.path.join(dest, "mla.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/mla.json")


def build_simpo_orpo(root):
    """SimPO + ORPO reference-free preference-loss goldens — independent numpy
    (Meng 2024 arXiv:2405.14734 eq5-7; Hong 2024 arXiv:2403.07691 eq6-7, §R71)."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)

    def softplus(u):
        return np.maximum(u, 0) + np.log1p(np.exp(-np.abs(u)))

    # length-averaged log-probs (negative: probabilities < 1)
    avg_w = np.array([-0.5, -1.2, -0.8, -2.0])
    avg_l = np.array([-1.0, -0.9, -1.5, -1.8])

    beta, gamma = 2.0, 1.0
    z = beta * (avg_w - avg_l) - gamma
    simpo = float(np.mean(softplus(-z)))

    lam = 0.1
    pw, pl = np.exp(avg_w), np.exp(avg_l)
    log_or = (avg_w - avg_l) - (np.log1p(-pw) - np.log1p(-pl))
    orpo = float(np.mean(-avg_w + lam * softplus(-log_or)))

    data = {
        "avg_w": [float(v) for v in avg_w], "avg_l": [float(v) for v in avg_l],
        "beta": beta, "gamma": gamma, "lambda": lam,
        "simpo": simpo, "orpo": orpo,
    }
    with open(os.path.join(dest, "simpo_orpo.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/simpo_orpo.json")


def build_cpo(root):
    """CPO reference-free preference + NLL/BC golden — independent numpy
    (Xu et al. 2024 arXiv:2401.08417 eq3-5, §R121). Log-probs are SUMMED over
    tokens (DPO-style, not length-normalized); loss = mean(softplus(-beta*(lw-ll))
    + alpha*(-lw)). Defaults beta=0.1, alpha (TRL cpo_alpha)=1."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)

    def softplus(u):
        return np.maximum(u, 0) + np.log1p(np.exp(-np.abs(u)))

    # SUMMED sequence log-probs (larger magnitude than per-token averages)
    sum_w = np.array([-8.0, -15.0, -6.5, -20.0])
    sum_l = np.array([-10.0, -12.0, -9.0, -18.0])

    beta, alpha = 0.1, 1.0
    z = beta * (sum_w - sum_l)
    cpo = float(np.mean(softplus(-z) + alpha * (-sum_w)))

    data = {
        "sum_w": [float(v) for v in sum_w], "sum_l": [float(v) for v in sum_l],
        "beta": beta, "alpha": alpha, "cpo": cpo,
    }
    with open(os.path.join(dest, "cpo.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/cpo.json")


def build_dora(root):
    """DoRA weight-decomposition golden (Liu et al. 2024 arXiv:2402.09353 eq5,
    §R70). Independent numpy, [in,out] convention, per-output-column norm."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    rng = np.random.default_rng(404)
    din, dout, r, alpha = 4, 3, 2, 2.0
    W = rng.standard_normal((din, dout))
    A = rng.standard_normal((din, r))
    B = rng.standard_normal((r, dout))       # nonzero → general (non-init) case
    m = np.abs(rng.standard_normal(dout)) + 0.5  # test magnitude (positive)
    x = rng.standard_normal((2, din))

    V = W + (alpha / r) * (A @ B)
    n = np.sqrt((V**2).sum(axis=0))          # per-output-column L2 norm [out]
    Wp = m[None, :] * V / n[None, :]         # W' = m·V/‖V‖_col
    y = x @ Wp
    m_init = np.sqrt((W**2).sum(axis=0))     # ‖W‖_col → init magnitude (B=0 ⇒ W'=W)

    flat = lambda t: [float(v) for v in np.asarray(t).reshape(-1)]
    data = {
        "in": din, "out": dout, "r": r, "alpha": alpha,
        "W": flat(W), "A": flat(A), "B": flat(B), "m": flat(m), "x": flat(x),
        "V": flat(V), "Wp": flat(Wp), "y": flat(y), "m_init": flat(m_init),
    }
    with open(os.path.join(dest, "dora.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/dora.json")


def build_moecombine(root):
    """MoE renormalize-and-combine golden (Mixtral §R61, §T68). Independent numpy:
    softmax gates → top-2 mask → renormalize over selected → mix expert outputs."""
    dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(dest, exist_ok=True)
    rng = np.random.default_rng(303)
    T, E, D, k = 3, 4, 5, 2
    logits = rng.standard_normal((T, E))
    gates = np.exp(logits - logits.max(-1, keepdims=True))
    gates /= gates.sum(-1, keepdims=True)
    mask = np.zeros((T, E))
    for t in range(T):
        mask[t, np.argsort(-logits[t])[:k]] = 1  # top-k
    w = gates * mask
    experts = rng.standard_normal((E, T, D))
    wn = w / w.sum(-1, keepdims=True)          # renormalize over selected
    out = np.einsum("te,etd->td", wn, experts)  # y[t,d] = Σ_e ŵ[t,e]·E_e[t,d]
    data = {
        "T": T, "E": E, "D": D,
        "w": [float(v) for v in w.reshape(-1)],
        "experts": [[float(v) for v in experts[i].reshape(-1)] for i in range(E)],
        "out": [float(v) for v in out.reshape(-1)],
    }
    with open(os.path.join(dest, "moecombine.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/moecombine.json")


# ---------------------------------------------------------------------------
# §B68 non-vacuous HF goldens — nlp/testdata/<arch>_hf.{safetensors,_logits.npy}
# ---------------------------------------------------------------------------


def _b68_seed(name):
    """Per-tensor phase in [0, 2π), from an FNV-1a hash of the tensor NAME.

    Sibling layers (…layers.0.… vs …layers.1.…) and sibling roles (q_norm vs
    k_norm, conv1d.bias vs D) therefore receive DIFFERENT values, so a
    cross-layer or cross-role swap in a loader moves the logits.

    FNV-1a rather than the byte-sum seedOf the Go-side §B68 fixtures use
    (nlp/mamba2_gguf_test.go): a byte sum is PERMUTATION-INVARIANT over the
    name, and the collision guard below caught it aliasing
    rwkv.blocks.0.ln2.bias with rwkv.blocks.1.ln1.bias ("0"+"2" == "1"+"1") —
    which would have made that very cross-layer swap invisible again."""
    h = 2166136261
    for b in name.encode():
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return (h % 1000003) / 1000003.0 * 2.0 * math.pi


def _b68_fill(name, n, base, amp, step):
    """Deterministic non-constant vector: base + amp*sin(seed + step*i).

    Non-constant is the point (§B68): a CONSTANT tensor is invariant under
    every reshape/broadcast/permute a loader can get wrong, and a ZERO tensor
    is additionally invariant under being dropped entirely."""
    return np.array([base + amp * math.sin(_b68_seed(name) + step * i)
                     for i in range(n)], dtype=np.float32)


# Patch kinds. Ranges are chosen to stay numerically tame (norm gains straddle
# 1, the multiplicative skips stay positive) while being far enough from the
# vacuous default that a dropped or mis-indexed term moves the logits by O(1).
_B68_KINDS = {
    "gain":       (1.00, 0.25, 2.3),   # RMS/LayerNorm gamma — golden default 1
    "qk_gain":    (1.00, 0.60, 2.3),   # QK-norm gamma — see below
    "ln_bias":    (0.02, 0.05, 1.3),   # LayerNorm beta      — golden default 0
    "bias":       (0.01, 0.05, 1.0),   # additive projection bias — default 0
    "skip":       (0.80, 0.20, 1.7),   # multiplicative skip D — default 1
    "time_first": (0.50, 1.00, 1.9),   # RWKV WKV bonus u     — default 1
}

# Why qk_gain is spread WIDER than a plain gain: q/k-side perturbations only
# reach the logits through the attention SCORES, and softmax is invariant to a
# per-row constant shift, so much of a q-side change cancels. Measured on the
# qwen3 golden, a q_norm↔k_norm swap moves the logits by 4.2e-3 at ±0.25 (a
# thin 2.1x over the test's 2e-3 gate) but 1.1e-2 at ±0.60 (5.4x). Same reason
# the Qwen2 q/k projection biases below cannot be gated by logits at any sane
# amplitude — see the structural assertion in nlp/llama_hf_test.go.

# Per-architecture: the tensor-name suffixes whose committed value is vacuous
# (all-zero or all-constant) yet convention-critical, i.e. some load transform
# — a drop, a q/k swap, a per-head vs full-width read, a channel mis-index —
# is invisible while they stay that way.
_B68_PATCH = {
    "qwen2": [
        (".self_attn.q_proj.bias", "bias"),
        (".self_attn.k_proj.bias", "bias"),
        (".self_attn.v_proj.bias", "bias"),
        (".input_layernorm.weight", "gain"),
        (".post_attention_layernorm.weight", "gain"),
        ("model.norm.weight", "gain"),
    ],
    "qwen3": [
        (".self_attn.q_norm.weight", "qk_gain"),  # per-head QK-norm: THE Qwen3 feature
        (".self_attn.k_norm.weight", "qk_gain"),
        (".input_layernorm.weight", "gain"),
        (".post_attention_layernorm.weight", "gain"),
        ("model.norm.weight", "gain"),
    ],
    "olmo2": [
        (".self_attn.q_norm.weight", "qk_gain"),  # full-width QK-norm
        (".self_attn.k_norm.weight", "qk_gain"),
        (".post_attention_layernorm.weight", "gain"),
        (".post_feedforward_layernorm.weight", "gain"),
        ("model.norm.weight", "gain"),
    ],
    "mamba": [
        (".mixer.D", "skip"),                   # y += D ⊙ x — D≡1 reads as y += x
        (".mixer.conv1d.bias", "bias"),
        (".norm.weight", "gain"),
        ("backbone.norm_f.weight", "gain"),
    ],
    "mamba2": [
        (".mixer.D", "skip"),                   # per-HEAD skip: broadcast-critical
        (".mixer.conv1d.bias", "bias"),
        (".norm.weight", "gain"),               # covers .mixer.norm too (gated norm)
        ("backbone.norm_f.weight", "gain"),
    ],
    "rwkv": [
        (".attention.time_first", "time_first"),
        (".ln1.weight", "gain"), (".ln1.bias", "ln_bias"),
        (".ln2.weight", "gain"), (".ln2.bias", "ln_bias"),
        (".pre_ln.weight", "gain"), (".pre_ln.bias", "ln_bias"),
        ("rwkv.ln_out.weight", "gain"), ("rwkv.ln_out.bias", "ln_bias"),
    ],
}

# The prompt every nlp/*_hf_test.go golden test feeds the model.
_B68_TOKENS = [3, 7, 1, 9, 4, 2, 8]


def _b68_configs():
    """Reference configs, reverse-engineered from the committed fixtures and
    VERIFIED: with the committed weights each one reproduces the committed
    *_hf_logits.npy BIT-FOR-BYTE (max abs diff exactly 0.0 under transformers
    5.14.1 / torch 2.12.1). That equality is what licenses regenerating the
    goldens here — the generator is a faithful stand-in for whatever ad-hoc
    script originally produced them, which was never committed."""
    from transformers.models.mamba import MambaConfig, MambaForCausalLM
    from transformers.models.mamba2 import Mamba2Config, Mamba2ForCausalLM
    from transformers.models.olmo2 import Olmo2Config, Olmo2ForCausalLM
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.rwkv import RwkvConfig, RwkvForCausalLM

    return {
        "qwen2": (Qwen2ForCausalLM, Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
            rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False)),
        "qwen3": (Qwen3ForCausalLM, Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=6,
            max_position_embeddings=32, rms_norm_eps=1e-6, rope_theta=10000.0,
            tie_word_embeddings=False, attention_bias=False)),
        "olmo2": (Olmo2ForCausalLM, Olmo2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
            rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False)),
        "mamba": (MambaForCausalLM, MambaConfig(
            vocab_size=32, hidden_size=16, state_size=8, num_hidden_layers=2,
            conv_kernel=4, expand=2, time_step_rank=4, layer_norm_epsilon=1e-5,
            use_mambapy=False)),
        "mamba2": (Mamba2ForCausalLM, Mamba2Config(
            vocab_size=32, hidden_size=32, state_size=8, num_hidden_layers=2,
            conv_kernel=4, expand=1, n_groups=2, num_heads=4, head_dim=8,
            layer_norm_epsilon=1e-5)),
        "rwkv": (RwkvForCausalLM, RwkvConfig(
            vocab_size=32, hidden_size=32, num_hidden_layers=2, intermediate_size=64,
            layer_norm_epsilon=1e-5, rescale_every=0)),
    }


def build_hf_b68(root):
    """Make the nlp HF goldens DISCRIMINATING, then re-derive their logits (§B68).

    The committed fixtures leave every convention-critical *gain*, *skip* and
    *bias* tensor at its vacuous default — norm gammas all 1, Qwen2's q/k/v
    projection biases all 0, Mamba's D and RWKV's time_first all 1. A zero
    vector is invariant under being dropped or permuted; an all-ones
    multiplicative skip makes `y += D ⊙ x` indistinguishable from `y += x`;
    an all-ones gamma makes the input_layernorm→attn_norm /
    post_attention_layernorm→ffn_norm mapping (and q_norm↔k_norm) unobservable.
    Such a golden passes with the code under test deleted.

    This rewrites ONLY those tensors — every projection/embedding weight is
    carried through byte-identically — and then recomputes *_hf_logits.npy with
    the real transformers model, so the goldens stay §V16 foreign-convention
    anchors rather than GoAI-vs-GoAI tautologies.

    Idempotent: the fills are pure functions of the tensor name, so re-running
    reproduces the committed bytes. The base (non-patched) weights come from
    the committed file — their original RNG stream was never committed, so
    preserving them is the only way to keep the unrelated goldens stable."""
    try:
        import torch
        from safetensors.numpy import load_file, save_file
    except ImportError:
        print("torch/safetensors/transformers missing — skipping nlp HF §B68 goldens")
        return
    try:
        cfgs = _b68_configs()
    except ImportError:
        print("transformers missing — skipping nlp HF §B68 goldens")
        return

    dest = os.path.join(root, "..", "nlp", "testdata")
    for arch, rules in sorted(_B68_PATCH.items()):
        path = os.path.join(dest, arch + "_hf.safetensors")
        if not os.path.exists(path):
            print("missing", path, "— skipping")
            continue
        sd = load_file(path)
        patched = {}
        for name, w in sd.items():
            for suffix, kind in rules:
                if not name.endswith(suffix):
                    continue
                base, amp, step = _B68_KINDS[kind]
                v = _b68_fill(name, int(np.prod(w.shape)), base, amp, step)
                sd[name] = v.reshape(w.shape).astype(w.dtype)
                patched[name] = sd[name]
                break
        if not patched:
            raise SystemExit(f"{arch}: no tensor matched the §B68 patch rules")

        # A patched tensor that is constant, zero, or equal to a sibling would
        # reintroduce exactly the invariance this exists to destroy.
        for name, v in patched.items():
            flat = v.reshape(-1)
            if flat.min() == flat.max():
                raise SystemExit(f"{arch}/{name}: patched value is constant")
            if not np.abs(flat).max() > 0:
                raise SystemExit(f"{arch}/{name}: patched value is all-zero")
        names = sorted(patched)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if patched[a].shape == patched[b].shape and np.array_equal(patched[a], patched[b]):
                    raise SystemExit(f"{arch}: {a} and {b} got identical values")

        save_file(sd, path)
        cls, cfg = cfgs[arch]
        model = cls(cfg)
        model.load_state_dict({k: torch.tensor(v) for k, v in sd.items()}, strict=True)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor([_B68_TOKENS])).logits[0]
        out = os.path.join(dest, arch + "_hf_logits.npy")
        np.save(out, logits.numpy().astype(np.float32))
        print(f"wrote nlp/testdata/{arch}_hf.safetensors (+{len(patched)} §B68 tensors) "
              f"and {arch}_hf_logits.npy")


def build_yarn(root):
    """YaRN NTK-by-parts RoPE golden — independent numpy from the paper formula
    (Peng et al. 2023 arXiv:2309.00071 §3.2, §R66). Params chosen so the ramp band
    exercises extrapolated, fractional-ramp and interpolated dimensions."""
    seq, hd, base = 6, 16, 10000.0
    s = 4.0      # scale factor = new_ctx / orig_ctx
    L = 2048.0   # original trained context length
    beta_fast, beta_slow = 32.0, 1.0
    half = hd // 2
    rng = np.random.default_rng(202)
    q = rng.standard_normal((seq, hd))

    i = np.arange(half, dtype=np.float64)
    theta = base ** (-(2.0 * i) / hd)     # base inverse frequencies θ_i

    def corr_dim(rot):
        return hd * np.log(L / (rot * 2.0 * np.pi)) / (2.0 * np.log(base))
    low = max(np.floor(corr_dim(beta_fast)), 0.0)
    high = min(np.ceil(corr_dim(beta_slow)), half - 1.0)
    if low == high:
        high += 0.001
    gamma = np.clip((i - low) / (high - low), 0.0, 1.0)
    inv = (1.0 - gamma) * theta + gamma * (theta / s)   # NTK-by-parts blend

    pos = np.arange(seq, dtype=np.float64)
    freqs = np.outer(pos, inv)                            # [seq, half]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)

    def rotate_half(x):
        return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    y = q * cos + rotate_half(q) * sin
    m_scale = 0.1 * np.log(s) + 1.0

    flat = lambda t: [float(v) for v in np.asarray(t).reshape(-1)]
    data = {
        "q": flat(q), "seq": seq, "hd": hd, "base": base,
        "scale": s, "orig_ctx": L, "beta_fast": beta_fast, "beta_slow": beta_slow,
        "low": float(low), "high": float(high),
        "theta": flat(theta), "inv_freq": flat(inv),
        "y": flat(y), "attn_scale": float(m_scale),
    }
    dest = os.path.join(root, "..", "autograd", "testdata")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "yarn.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print("wrote autograd/testdata/yarn.json")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(root, "..", "backend", "ref", "testdata")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "elementwise.json")
    with open(path, "w") as f:
        json.dump(build(), f, indent=1)
        f.write("\n")
    print("wrote", os.path.relpath(path, os.path.join(root, "..")))
    # nn package consumes just the losses section
    nn_dest = os.path.join(root, "..", "nn", "testdata")
    os.makedirs(nn_dest, exist_ok=True)
    with open(os.path.join(nn_dest, "losses.json"), "w") as f:
        json.dump({"losses": build_losses()}, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/losses.json")
    with open(os.path.join(nn_dest, "optim.json"), "w") as f:
        json.dump({"optim": build_optim()}, f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/optim.json")
    with open(os.path.join(nn_dest, "muon.json"), "w") as f:
        json.dump(build_muon(), f, indent=1)
        f.write("\n")
    print("wrote nn/testdata/muon.json")
    write_npy_samples(root)
    write_safetensors_sample(root)
    build_transformer(root)
    build_qmatmul(root)
    build_gguf(root)
    build_gpt(root)
    build_cv(root)
    build_classic(root)
    build_half(root)
    build_lora(root)
    build_dpo(root)
    build_ppo(root)
    build_moe(root)
    build_kto(root)
    build_ipo(root)
    build_gae(root)
    build_grpo(root)
    build_moecombine(root)
    build_dora(root)
    build_simpo_orpo(root)
    build_cpo(root)
    build_conv1d(root)
    build_ssm(root)
    build_mla(root)
    build_labelsmooth(root)
    build_distill(root)
    build_llama(root)
    build_llama2(root)
    build_hf_b68(root)
    build_yarn(root)
    build_tokenizer(root)


if __name__ == "__main__":
    sys.exit(main())
