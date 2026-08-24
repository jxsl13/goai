#!/usr/bin/env python3
"""Matched TinyLlama serving cells for the RTX 3060 SGLang comparison.

The four cells mirror the vLLM rows in docs/benchmarking.md: batch-1 tg128,
batch-1 pp128, and n=16/n=64 aggregate output throughput. The engine is warm,
the radix cache is flushed before every timed repetition, and repetitions run
sequentially so prefill state cannot leak between samples.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time

import sglang as sgl
import torch


def package_version(distribution, module):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return getattr(module, "__version__", "unknown")


def command_output(*args):
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def require_exclusive_gpu(allow_shared):
    processes = command_output(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader",
    )
    if not allow_shared and processes not in ("", "unavailable"):
        raise RuntimeError(
            "GPU has an existing compute process; rerun GPU-exclusive or pass "
            f"--allow-shared-gpu explicitly:\n{processes}"
        )


def token_prompts(batch, prompt_len):
    prompt = [1] + [500 + (i % 2000) for i in range(prompt_len - 1)]
    return [prompt[:] for _ in range(batch)]


def output_ids(result):
    rows = result if isinstance(result, list) else [result]
    ids = []
    for row in rows:
        if not isinstance(row, dict) or "output_ids" not in row:
            raise RuntimeError(f"unexpected SGLang result shape: {type(row)!r}")
        ids.append(row["output_ids"])
    return ids


def timed_generate(engine, batch, prompt_len, output_len):
    engine.flush_cache()
    prompts = token_prompts(batch, prompt_len)
    params = {
        "temperature": 0.0,
        "max_new_tokens": output_len,
        "ignore_eos": True,
    }
    start = time.perf_counter()
    result = engine.generate(input_ids=prompts, sampling_params=params)
    elapsed = time.perf_counter() - start
    rows = output_ids(result)
    if len(rows) != batch:
        raise RuntimeError(f"SGLang returned {len(rows)} rows, want {batch}")
    for i, row in enumerate(rows):
        if len(row) != output_len:
            raise RuntimeError(
                f"SGLang row {i} returned {len(row)} tokens, want {output_len}"
            )
    return elapsed


def benchmark_case(engine, case, reps):
    samples = []
    for _ in range(reps):
        elapsed = timed_generate(
            engine, case["batch"], case["prompt_len"], case["output_len"]
        )
        tokens = (
            case["batch"] * case["prompt_len"]
            if case["basis"] == "input"
            else case["batch"] * case["output_len"]
        )
        samples.append(tokens / elapsed)
    return {
        **case,
        "samples_tok_s": samples,
        "median_tok_s": statistics.median(samples),
        "min_tok_s": min(samples),
        "max_tok_s": max(samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/.local/share/goai-vllm/tinyllama-hf"),
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.reps < 3:
        parser.error("--reps must be at least 3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    require_exclusive_gpu(args.allow_shared_gpu)
    engine = sgl.Engine(
        model_path=args.model,
        dtype="float16",
        mem_fraction_static=args.mem_fraction_static,
        disable_radix_cache=args.disable_radix_cache,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    cases = [
        {
            "name": "decode_tg128_b1",
            "batch": 1,
            "prompt_len": 1,
            "output_len": 128,
            "basis": "output",
        },
        {
            "name": "prefill_pp128_b1",
            "batch": 1,
            "prompt_len": 128,
            "output_len": 1,
            "basis": "input",
        },
        {
            "name": "serving_ctx128_tg128_n16",
            "batch": 16,
            "prompt_len": 128,
            "output_len": 128,
            "basis": "output",
        },
        {
            "name": "serving_ctx128_tg128_n64",
            "batch": 64,
            "prompt_len": 128,
            "output_len": 128,
            "basis": "output",
        },
    ]
    try:
        timed_generate(engine, 1, 128, 4)
        results = [benchmark_case(engine, case, args.reps) for case in cases]
    finally:
        engine.shutdown()

    record = {
        "protocol": "TinyLlama fp16, warm engine, sequential GPU-exclusive runs",
        "model": os.path.realpath(args.model),
        "repetitions": args.reps,
        "radix_cache": not args.disable_radix_cache,
        "cuda_graph": not args.disable_cuda_graph,
        "host": platform.node(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
        "sglang": package_version("sglang", sgl),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver": command_output(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "goai_commit": command_output("git", "rev-parse", "HEAD"),
        "results": results,
    }
    print(json.dumps(record, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
