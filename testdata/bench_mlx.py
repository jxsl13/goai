#!/usr/bin/env python3
"""MLX (Apple's own ML framework) datapoint for the T887 Apple-silicon LLM
head-to-head. Times mlx-lm decoding a native-4-bit TinyLlama-1.1B on the M2 Metal
GPU, so GoAI's llamagpu and llama.cpp can be compared against Apple's framework too.

Fairness note: MLX loads its OWN 4-bit affine-quantized TinyLlama (group size 64),
converted from the same TinyLlama-1.1B-Chat-v1.0 the others use via
`mlx_lm.convert -q --q-bits 4`, NOT the Q4_K_M GGUF the other two engines share --
different 4-bit quant scheme, same base model + same ~4-bit precision. So this is a
roughly-precision-matched ENGINE comparison (which Apple-Metal runtime is fastest
on a TinyLlama-class model), not a byte-identical-weights one.

Convert first (once): .venv/bin/python -m mlx_lm convert --hf-path
TinyLlama/TinyLlama-1.1B-Chat-v1.0 -q --q-bits 4 --mlx-path models/tinyllama-mlx-4bit
Then run:             .venv/bin/python testdata/bench_mlx.py
"""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate

MODEL = "models/tinyllama-mlx-4bit"
N_GEN = 64


def main() -> None:
    model, tok = load(MODEL)
    # ~64-token prompt so prefill is pp64, matching the llama-bench / GoAI harness.
    text = ("The quick brown fox jumps over the lazy dog. Tokenization throughput "
            "matters for language model serving, and prefill processes the whole "
            "prompt in one batched forward pass before autoregressive decoding begins "
            "one token at a time from the key value cache. ")
    prompt = tok.encode(text)[:64]

    # Warm (kernel compile + weights resident), then best-of-3 on generation t/s.
    best_tg, best_pp = 0.0, 0.0
    ttft = []
    for r in range(4):
        gen_toks, gen_t, prompt_t, n_prompt = 0, 0.0, 0.0, 0
        for resp in stream_generate(model, tok, prompt, max_tokens=N_GEN):
            # each response carries running prompt_tps / generation_tps
            if resp.generation_tokens == 1:
                prompt_t = resp.prompt_tps
                n_prompt = resp.prompt_tokens
            gen_toks = resp.generation_tokens
            gen_t = resp.generation_tps
        if r == 0:
            continue  # discard warm-up run
        best_tg = max(best_tg, gen_t)
        best_pp = max(best_pp, prompt_t)

    print(f"mlx {mx.__version__} | model {MODEL}")
    print(f"prefill (pp{n_prompt}) best-of-3: {best_pp:7.1f} tok/s")
    print(f"decode  (tg{N_GEN}) best-of-3: {best_tg:7.1f} tok/s")


if __name__ == "__main__":
    main()
