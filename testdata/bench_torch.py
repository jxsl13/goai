#!/usr/bin/env python3
"""Cross-language performance reference: time PyTorch dense GEMM at the same
square sizes and dtypes as backend/cpu/gflops_bench_test.go, so GoAI's optimized
pure-Go cpu backend can be compared head-to-head with torch (Accelerate/MKL) in
docs/benchmarking.md.

A dense N×N · N×N matmul is 2·N³ flops. We warm up, then time the steady state
and report GFLOP/s. Threads are left at the torch default; set the env var
OMP_NUM_THREADS / torch.set_num_threads to match GOMAXPROCS for a fair core count.

Usage:
    .venv/bin/python testdata/bench_torch.py           # print table
    .venv/bin/python testdata/bench_torch.py --json     # emit JSON
"""
import json
import sys
import time

import torch

SIZES = [512, 1024]
DTYPES = [("f64", torch.float64), ("f32", torch.float32)]


def bench_matmul(n: int, dtype) -> tuple[float, float]:
    a = torch.randn(n, n, dtype=dtype)
    b = torch.randn(n, n, dtype=dtype)
    for _ in range(3):  # warmup
        c = a @ b
    reps = 30
    t0 = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    dt = (time.perf_counter() - t0) / reps
    gflops = 2.0 * n**3 / dt / 1e9
    return dt, gflops


def main() -> int:
    results = {"threads": torch.get_num_threads(), "torch": torch.__version__}
    rows = []
    for n in SIZES:
        for name, dt in DTYPES:
            secs, gflops = bench_matmul(n, dt)
            results[f"matmul_{name}_{n}"] = {"s_per_op": secs, "gflops": gflops}
            rows.append((name, n, gflops, secs * 1e3))

    if "--json" in sys.argv:
        json.dump(results, sys.stdout, indent=1)
        print()
    else:
        print(f"# torch {torch.__version__}, {torch.get_num_threads()} threads")
        print(f"{'dtype':>6} {'N':>6} {'GFLOP/s':>10} {'ms/op':>10}")
        for name, n, gflops, ms in rows:
            print(f"{name:>6} {n:>6} {gflops:>10.1f} {ms:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
