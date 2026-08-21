#!/usr/bin/env python3
"""Matched raw-GGUF open benchmark for gguf-py.

The Go arm is format/gguf.BenchmarkOpenRawModelPath (or ModelCopy). Both arms
parse one regular GGUF, retain mapped encoded tensor views, and materialize no
f32 weights. Set GGUF_RAW_COPY=1 to additionally copy every encoded tensor into
independent storage, matching BenchmarkOpenRawModelCopy.
"""

import gc
import json
import os
import sys
import time
from importlib import metadata

import gguf
import numpy as np


def load(path: str, copy_payload: bool):
    reader = gguf.GGUFReader(path)
    copies = None
    if copy_payload:
        copies = [np.array(t.data, copy=True) for t in reader.tensors]
    return reader, copies


def main() -> None:
    path = os.environ.get("GGUF_BENCH_FILE")
    if not path:
        sys.exit("set GGUF_BENCH_FILE to a production GGUF model")
    copy_payload = os.environ.get("GGUF_RAW_COPY") == "1"

    warm_reader, warm_copies = load(path, copy_payload)
    if not warm_reader.tensors:
        sys.exit("GGUF contains no tensors")
    del warm_reader, warm_copies
    gc.collect()

    start = time.perf_counter_ns()
    reader, copies = load(path, copy_payload)
    elapsed = time.perf_counter_ns() - start
    tensor_bytes = sum(t.data.nbytes for t in reader.tensors)
    if copies is not None and sum(a.nbytes for a in copies) != tensor_bytes:
        sys.exit("copied tensor byte count differs from mapped views")

    print(json.dumps({
        "elapsed_ns": elapsed,
        "fields": len(reader.fields),
        "file_bytes": os.path.getsize(path),
        "gguf": metadata.version("gguf"),
        "mode": "copy" if copy_payload else "open",
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "tensor_bytes": tensor_bytes,
        "tensors": len(reader.tensors),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
