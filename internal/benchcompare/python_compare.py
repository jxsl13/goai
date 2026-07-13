#!/usr/bin/env python3
"""Python-library comparison micro-benchmarks (§T93).

Times the SAME ops and shapes as the Go cross-backend benchmarks
(internal/benchcompare/compare_test.go) using the common Python stacks — NumPy and
PyTorch (CPU + the Apple-GPU MPS backend) — so GoAI's cpu/metal/vulkan numbers can
be read next to torch-cpu / torch-mps / numpy. Output rows mirror the Go format:

    <Op>/<shape>/<backend>   <value> <unit>

GFLOP/s for matmul and conv2d (higher = faster), ms/op for attention. Run via
`make bench-python`. This is a measurement harness, not an algorithm (no §V16).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

ITERS = 30
WARMUP = 5
torch.manual_seed(0)

MPS = torch.backends.mps.is_available()


def bench(fn, sync=None):
    """Return seconds/op for fn, with warmup and optional device sync."""
    for _ in range(WARMUP):
        fn()
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    if sync:
        sync()
    return (time.perf_counter() - t0) / ITERS


def row(op, shape, backend, value, unit):
    print(f"{op}/{shape}/{backend:11s} {value:12.2f} {unit}")


def gflops(flops, sec):
    return flops / sec / 1e9


def bench_matmul():
    for sz in (256, 512, 1024):
        flops = 2.0 * sz ** 3
        a = np.random.randn(sz, sz).astype(np.float32)
        b = np.random.randn(sz, sz).astype(np.float32)
        row("MatMul", sz, "numpy-cpu", gflops(flops, bench(lambda: a @ b)), "GFLOP/s")

        ta, tb = torch.from_numpy(a), torch.from_numpy(b)
        row("MatMul", sz, "torch-cpu", gflops(flops, bench(lambda: torch.matmul(ta, tb))), "GFLOP/s")

        if MPS:
            ma, mb = ta.to("mps"), tb.to("mps")
            sec = bench(lambda: torch.matmul(ma, mb), sync=torch.mps.synchronize)
            row("MatMul", sz, "torch-mps", gflops(flops, sec), "GFLOP/s")


def bench_mha():
    seq, heads, dk = 512, 8, 64
    shape = f"{seq}x{heads}x{dk}"
    q = torch.randn(1, heads, seq, dk)
    k = torch.randn(1, heads, seq, dk)
    v = torch.randn(1, heads, seq, dk)

    def fwd(qq, kk, vv):
        return F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)

    row("MHAForward", shape, "torch-cpu", bench(lambda: fwd(q, k, v)) * 1e3, "ms/op")
    if MPS:
        mq, mk, mv = q.to("mps"), k.to("mps"), v.to("mps")
        row("MHAForward", shape, "torch-mps",
            bench(lambda: fwd(mq, mk, mv), sync=torch.mps.synchronize) * 1e3, "ms/op")

    # forward+backward (our OpMHABackward recomputes the forward internally, so
    # torch's fwd+bwd is the fair analog).
    def fwdbwd(qq, kk, vv):
        for t in (qq, kk, vv):
            t.grad = None
        fwd(qq, kk, vv).sum().backward()

    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    row("MHAFwdBwd", shape, "torch-cpu", bench(lambda: fwdbwd(qg, kg, vg)) * 1e3, "ms/op")
    if MPS:
        qm = q.to("mps").clone().requires_grad_(True)
        km = k.to("mps").clone().requires_grad_(True)
        vm = v.to("mps").clone().requires_grad_(True)
        row("MHAFwdBwd", shape, "torch-mps",
            bench(lambda: fwdbwd(qm, km, vm), sync=torch.mps.synchronize) * 1e3, "ms/op")


def bench_conv2d():
    n, c, hw, f, k = 8, 16, 32, 32, 3
    shape = f"{n}x{c}x{hw}sq_k{k}"
    ho = wo = hw  # stride 1, pad 1, k 3 → same spatial size
    flops = 2.0 * n * f * ho * wo * c * k * k
    x = torch.randn(n, c, hw, hw)
    w = torch.randn(f, c, k, k)
    row("Conv2D", shape, "torch-cpu", gflops(flops, bench(lambda: F.conv2d(x, w, padding=1))), "GFLOP/s")
    if MPS:
        mx, mw = x.to("mps"), w.to("mps")
        sec = bench(lambda: F.conv2d(mx, mw, padding=1), sync=torch.mps.synchronize)
        row("Conv2D", shape, "torch-mps", gflops(flops, sec), "GFLOP/s")


if __name__ == "__main__":
    print(f"# python compare: numpy {np.__version__}, torch {torch.__version__}, "
          f"mps={MPS}, threads={torch.get_num_threads()}, iters={ITERS}")
    bench_matmul()
    bench_mha()
    bench_conv2d()
