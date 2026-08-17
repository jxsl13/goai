#!/usr/bin/env python3
"""Cross-language performance reference: time GGUF model-file loading in the Python
reference (gguf-py's mmap-based GGUFReader) on a fixture GoAI's format/gguf reader
loads too, so GoAI's pure-Go hostile-gated GGUF reader can be compared head-to-head
in BENCHMARKS.md. The GGUF companion to bench_safetensors_load.py (T885).

It (1) writes a deterministic F32 fixture at $GGUF_BENCH_FILE if absent -- N tensors
t0..t{N-1} of shape [1024,1024], value t_k[i,j] = f32((i*1024+j+k) % 1000), exactly
f32-representable so the Go side can bit-check it (F32 = no dequant on either side,
a pure load comparison); (2) times a FULL load -- GGUFReader(path) then a copy of
every tensor out of the mmap, matching GoAI's eager ReadFile that materializes all
tensors into File.Tensors -- best-of-N; (3) prints MB/s and versions (V13).

Usage:
    GGUF_BENCH_FILE=/tmp/gguf_bench.gguf .venv/bin/python testdata/bench_gguf_load.py
"""
import os
import sys
import time
from importlib import metadata

import gguf
import numpy as np

N_TENSORS = 16  # 16 x 4 MiB = 64 MiB, matching the safetensors fixture
ROWS = COLS = 1024


def gen(path: str) -> None:
    w = gguf.GGUFWriter(path, "bench")
    base = (np.arange(ROWS * COLS, dtype=np.float64).reshape(ROWS, COLS)) % 1000
    for k in range(N_TENSORS):
        w.add_tensor(f"t{k}", ((base + k) % 1000).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def best_of(fn, reps: int = 7) -> float:
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - s)
    return best


def main() -> None:
    path = os.environ.get("GGUF_BENCH_FILE")
    if not path:
        sys.exit("set GGUF_BENCH_FILE to the shared fixture path")
    if not os.path.exists(path):
        gen(path)
    nbytes = os.path.getsize(path)

    def full():
        r = gguf.GGUFReader(path)
        # copy every tensor out of the mmap to match GoAI's eager materialization
        _ = {t.name: np.array(t.data) for t in r.tensors}

    full_best = best_of(full)
    ver = metadata.version("gguf")
    print(f"gguf-py {ver}, numpy {np.__version__}, python {sys.version.split()[0]}")
    print(f"fixture: {nbytes} bytes ({N_TENSORS} x [{ROWS},{COLS}] f32)")
    print(f"full load best-of-7: {full_best*1e3:7.2f} ms  =>  {nbytes/full_best/1e6:7.1f} MB/s (all {N_TENSORS} tensors)")


if __name__ == "__main__":
    main()
