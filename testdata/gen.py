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

    return {
        "p0": OP0, "grads": OG,
        "sgd": {"lr": lr, "steps": sgd},
        "sgd_momentum": {"lr": lr, "momentum": mu, "steps": sgdm},
        "adam": {"lr": alr, "beta1": b1, "beta2": b2, "eps": eps, "steps": adam},
        "adamw": {"lr": alr, "beta1": b1, "beta2": b2, "eps": eps, "wd": wd, "steps": adamw},
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
    """Emit real numpy .npy files to validate the Pure-Go reader (§T10)."""
    dest = os.path.join(root, "..", "internal", "npy", "testdata")
    os.makedirs(dest, exist_ok=True)
    a64 = np.array([[1.5, -2.0, 3.25], [4.0, 5.5, -6.75]], dtype=np.float64)
    a32 = np.arange(12, dtype=np.float32).reshape(3, 4) * 0.5
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # 1-D shape (3,)
    np.save(os.path.join(dest, "mat_f64.npy"), a64)
    np.save(os.path.join(dest, "mat_f32.npy"), a32)
    np.save(os.path.join(dest, "vec_f64.npy"), v)
    print("wrote", os.path.relpath(dest, os.path.join(root, "..")), "*.npy")


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
    build_llama(root)
    build_llama2(root)
    build_tokenizer(root)


if __name__ == "__main__":
    sys.exit(main())
