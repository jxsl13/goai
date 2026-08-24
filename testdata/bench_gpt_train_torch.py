#!/usr/bin/env python3
"""Cross-language performance reference: time a full GPT TRAINING OBJECTIVE
(forward + cross-entropy + backward), optionally followed by AdamW, in PyTorch at the
IDENTICAL geometry as internal/benchcompare/compare_test.go's
BenchmarkGPTTrainingStep, so GoAI's cpu-simd / Metal / Vulkan training step can be
compared head-to-head with torch-cpu and torch-mps on the same box (BENCHMARKS.md).

Geometry (matches nlp.GPTConfig{Vocab:4096, Ctx:256, Dim:512, Heads:8, Layers:6}):
GPT-2 pre-LN -- token + learned-positional embedding, per block
LN -> causal MHA -> residual -> LN -> GELU MLP(4x) -> residual, final LN, LM head.
batch=1, seq=256, f32. The default measures only the objective; --adamw adds the
same F32 AdamW update as GPT.NewAdamWSession (lr=1e-3, betas=(0.9,0.999),
eps=1e-8, weight_decay=0.1). Throughput is tok/s = seq / step_time. Warm-up is
excluded; we report the median and the min.

Fairness notes: single sequence (batch 1) exactly as the Go side; MPS is
synchronized before the timer stops (dispatch is async); CPU thread count is
printed (torch's default) so the core budget is on the record.

Usage:
    .venv/bin/python testdata/bench_gpt_train_torch.py
    .venv/bin/python testdata/bench_gpt_train_torch.py --adamw
    .venv/bin/python testdata/bench_gpt_train_torch.py --json
"""
import json
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, CTX, DIM, HEADS, LAYERS, SEQ = 4096, 256, 512, 8, 6, 256


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DIM, eps=1e-5)
        self.attn = nn.MultiheadAttention(DIM, HEADS, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(DIM, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))

    def forward(self, x, mask):
        normalized = self.ln1(x)
        a, _ = self.attn(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, DIM)
        self.wpe = nn.Embedding(CTX, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(DIM, eps=1e-5)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, idx):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        mask = torch.triu(torch.full((idx.size(1), idx.size(1)), float("-inf"), device=idx.device), 1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.lnf(x))


def bench(device: str, adamw: bool, reps: int = 12) -> tuple[float, float]:
    torch.manual_seed(0)
    model = GPT().to(device).train()
    idx = (torch.arange(SEQ, device=device) % VOCAB).unsqueeze(0)
    tgt = torch.arange(SEQ, device=device) % VOCAB
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.1) if adamw else None

    def step():
        if optimizer is None:
            model.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(idx).view(-1, VOCAB), tgt)
        loss.backward()
        if optimizer is not None:
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()

    for _ in range(3):  # warm-up (excluded)
        step()
    times = []
    for _ in range(reps):
        s = time.perf_counter()
        step()
        times.append(time.perf_counter() - s)
    return statistics.median(times), min(times)


def main() -> None:
    adamw = "--adamw" in sys.argv
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    out = {}
    for dev in devices:
        med, best = bench(dev, adamw)
        out[dev] = {"median_ms": med * 1e3, "min_ms": best * 1e3,
                    "tok_s_median": SEQ / med, "tok_s_best": SEQ / best}

    meta = {
        "torch": torch.__version__, "python": sys.version.split()[0],
        "cpu_threads": torch.get_num_threads(),
        "geometry": f"vocab={VOCAB} ctx={CTX} dim={DIM} heads={HEADS} layers={LAYERS} seq={SEQ} batch=1 f32",
        "attention_bias": False,
        "shared_pre_attention_norm": True,
        "timed": "forward + cross_entropy + backward + AdamW" if adamw
                 else "forward + cross_entropy + backward (no optimizer step)",
    }
    if "--json" in sys.argv:
        print(json.dumps({"meta": meta, "results": out}, indent=2))
        return
    print(f"torch {meta['torch']}, python {meta['python']}, cpu_threads={meta['cpu_threads']}")
    print(f"geometry: {meta['geometry']}")
    print(f"timed: {meta['timed']}")
    for dev, r in out.items():
        print(f"torch-{dev:4}: median {r['median_ms']:7.2f} ms  =>  {r['tok_s_median']:6.0f} tok/s"
              f"   (best {r['tok_s_best']:.0f})")


if __name__ == "__main__":
    main()
