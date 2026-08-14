#!/usr/bin/env python3
"""Emit one Go-benchmark-format PyTorch SiLU out= sample for benchstat."""

import argparse
import gc
import math
import os
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--name", default="BenchmarkSiLUF64Into-12")
    args = parser.parse_args()
    if args.seconds <= 0 or args.threads <= 0:
        parser.error("seconds and threads must be positive")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    count = 256 * 1408
    values = torch.arange(count, dtype=torch.float64)
    values = -4.0 + 8.0 * torch.remainder(values, 1000) / 1000.0
    out = torch.empty_like(values)

    def op() -> None:
        torch.ops.aten.silu.out(values, out=out)

    for _ in range(50):
        op()
    expected = values * torch.sigmoid(values)
    denom = torch.maximum(torch.abs(expected), torch.tensor(1e-300, dtype=torch.float64))
    max_rel = torch.max(torch.abs(out - expected) / denom).item()
    if not math.isfinite(max_rel) or max_rel > 1e-15:
        raise RuntimeError(f"SiLU out quality mismatch: max relative error {max_rel:.3e}")

    iterations = 1
    while True:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            op()
        elapsed = time.perf_counter_ns() - start
        if elapsed >= 50_000_000:
            break
        iterations *= 2
    iterations = max(iterations, math.ceil(iterations * args.seconds * 1e9 / elapsed))
    gc.disable()
    try:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            op()
        elapsed = time.perf_counter_ns() - start
    finally:
        gc.enable()
    ns_per_op = elapsed / iterations

    print("goos: darwin")
    print("goarch: arm64")
    print("pkg: github.com/jxsl13/goai/backend/cpu")
    print(f"cpu: {os.environ.get('GOAI_CPU', 'Apple M2 Pro')}")
    print(f"# pytorch={torch.__version__} threads={args.threads} max_relative_error={max_rel:.3e}")
    print(f"{args.name}\t{iterations}\t{ns_per_op:.0f} ns/op")


if __name__ == "__main__":
    main()
