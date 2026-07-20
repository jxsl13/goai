#!/usr/bin/env python3
"""Cross-language performance reference: time safetensors model-file loading in the
Python reference implementation (Rust-cored safetensors + numpy), on a fixture that
GoAI's format/safetensors reader loads too, so GoAI's pure-Go hostile-gated reader
(SPEC V15/V29) can be compared head-to-head in BENCHMARKS.md.

It (1) generates a deterministic fixture at $ST_BENCH_FILE if absent -- N f32
tensors t0..t{N-1} of shape [1024,1024] (4 MiB each), value t_k[i,j] = f32((i*1024
+ j + k) % 1000), exactly f32-representable so the Go side can bit-check it; (2)
times FULL load (safetensors.numpy.load_file -- materializes every tensor) and
PARTIAL load (safe_open + get_tensor of ONE tensor -- the seek-to-one-tensor path
GoAI's LoadTensor also serves), best-of-N; (3) prints MB/s and versions (V13).

The Go companion (format/safetensors load bench, gated on ST_BENCH_FILE) reads the
SAME file and verifies the same values -- that equality is the fairness anchor.

Usage:
    ST_BENCH_FILE=/tmp/st_bench.safetensors .venv/bin/python testdata/bench_safetensors_load.py
"""
import os
import sys
import time

import numpy as np
import safetensors
from safetensors.numpy import load_file, save_file

N_TENSORS = 16  # 16 x 4 MiB = 64 MiB fixture
ROWS = COLS = 1024


def gen(path: str) -> None:
    tensors = {}
    base = (np.arange(ROWS * COLS, dtype=np.float64).reshape(ROWS, COLS)) % 1000
    for k in range(N_TENSORS):
        tensors[f"t{k}"] = ((base + k) % 1000).astype(np.float32)
    save_file(tensors, path)


def best_of(fn, reps: int = 7) -> float:
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - s)
    return best


def main() -> None:
    path = os.environ.get("ST_BENCH_FILE")
    if not path:
        sys.exit("set ST_BENCH_FILE to the shared fixture path")
    if not os.path.exists(path):
        gen(path)
    nbytes = os.path.getsize(path)
    one_bytes = ROWS * COLS * 4

    def full():
        d = load_file(path)
        # touch one element so a lazy backend can't skip materialization
        _ = d["t0"][0, 0]

    def partial():
        with safetensors.safe_open(path, framework="numpy") as f:
            _ = f.get_tensor(f"t{N_TENSORS - 1}")[0, 0]

    full_best = best_of(full)
    part_best = best_of(partial)

    print(f"safetensors {safetensors.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}")
    print(f"fixture: {nbytes} bytes ({N_TENSORS} x [{ROWS},{COLS}] f32)")
    print(f"full  load best-of-7: {full_best*1e3:7.2f} ms  =>  {nbytes/full_best/1e6:7.1f} MB/s (all {N_TENSORS} tensors)")
    print(f"one-tensor  best-of-7: {part_best*1e3:7.3f} ms  =>  {one_bytes/part_best/1e6:7.1f} MB/s (1 tensor from the {nbytes//1_000_000} MB file)")


if __name__ == "__main__":
    main()
