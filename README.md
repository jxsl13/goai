# GoAI

[![ci](https://github.com/jxsl13/goai/actions/workflows/ci.yml/badge.svg)](https://github.com/jxsl13/goai/actions/workflows/ci.yml)

Go-native, full-spectrum AI library — **Pure-Go-first, cgo-last**.

Every operation ships as a validated Pure-Go reference first (cross-checked
against PyTorch/tiktoken/ggml goldens), then optimized as a separate step
against that reference. GPU backends are optional, benchmark-gated, and always
fall back to Pure Go: `CGO_ENABLED=0` builds and tests green with no C
toolchain on any platform.

## What works today

- **Transformer LLMs end-to-end**: GPT and Llama (GQA/SwiGLU/RoPE) — build,
  train (full backward validated against real torch gradients at f64 rtol
  ~1e-9), checkpoint (safetensors/GGUF round-trips), tokenize (BPE bit-exact
  vs tiktoken, SentencePiece-Unigram, WordPiece), and generate.
- **GPU inference (Metal + Vulkan/MoltenVK)**: the `llamagpu` batched decoders
  record a whole decode step into one command buffer over device-resident
  weights and KV cache — measured **24× (Metal) / 21× (Vulkan)** over per-op
  decode, **41×** prompt prefill, cooperative long-context attention on every
  surface, and quantized (ggml Q-block) decode at 4-8× less weight memory.
- **Accelerated decoding, measured on real trained models**: Medusa multi-head
  drafting (**1.81×**, 97% acceptance), lossless prompt-lookup (**1.80×**),
  draft-model speculative decoding (lossless; pays off on compute-bound
  targets), tree-Medusa (candidate trees verified in ONE masked forward),
  Jacobi parallel decoding, Self-Extend length extrapolation (4× the training
  length with no fine-tuning) — plus beam/diverse-beam, contrastive
  search & decoding, DoLa, classifier-free guidance, Mirostat, top-k/p/min-p/
  typical/eta sampling, repetition penalties, regex-constrained decoding, and
  watermarking. Speed numbers in [`docs/benchmarking.md`](docs/benchmarking.md);
  what these features deliver on real trained models — measured — in
  [`docs/inference.md`](docs/inference.md).
- **Training toolbox** (`nn`, 100+ files): optimizers from SGD to Muon/SOAP/
  Sophia/Schedule-Free with composable wrappers (SAM, Lookahead, GaLore, …),
  the PEFT family (LoRA/DoRA/PiSSA/VeRA/IA³/prefix/prompt, QLoRA proven
  end-to-end), quantization &
  pruning (AWQ/GPTQ/HQQ/NF4/SparseGPT/Wanda), post-transformer blocks
  (MoE/Mamba/RWKV/RetNet/MLA — every family proven end-to-end as a trained
  char-LM with structural causality checks; RWKV also in O(1)-state recurrent
  inference mode) — the optimizers and wrappers real-workload-verified
  with a measured comparison in [`docs/training.md`](docs/training.md) —
  diffusion (DDPM/DDIM/EDM/flow matching), SSL,
  RLHF/alignment (GRPO, GSPO and the DPO family proven end-to-end on a real model,
  including a reproduced-and-mitigated reward-hacking case study — see
  [`docs/alignment.md`](docs/alignment.md)), continual learning, and model
  merging.
- **Vision, classic ML & RL**: a reference CNN image classifier over the
  NCHW conv/pool stack (`vision`); linear/softmax regression, K-Means, PCA
  (`classic`); environments, advantage estimation, and canonical agents (`rl`).

Every algorithm carries its paper citation (§R in [`SPEC.md`](SPEC.md)) and is
validated on the §V16 ladder: tier-1 parity against an official reference
where one exists, tier-2 the defining paper.

## Layout (§I)

| Layer | Package | Role |
|------|---------|------|
| L0 | `tensor` | Tensor, Dtype, strides/views |
| L1 | `backend`, `backend/ref` | op dispatch + Pure-Go reference (numerical truth) |
| L1b | `backend/cpu`, `backend/metal`, `backend/vulkan`, `backend/cuda` | optimized + GPU backends, auto-selected, always with fallback |
| L2 | `autograd` | reverse-mode tape + VJPs |
| L3 | `nn`, `ops`, `linalg` | layers, optimizers, losses; eager op API |
| L4 | `nlp`, `vision`, `classic`, `rl` | domain packages |
| L5 | `format` | safetensors, GGUF (every ggml quant reads: K-quants, all 8 i-quants, MXFP4), npy/npz |
| — | `llamagpu` | batched GPU decoding for GPT/Llama (Metal/Vulkan) |

## Build & verify

```sh
make build        # CGO_ENABLED=0 pure-Go build (§V7)
make vet test     # pure-Go gate: compiles tests too (§V23)
make metal-test   # Metal/MPS backend suite (darwin, cgo)
make vulkan-test  # Vulkan backend suite (MoltenVK on macOS)
make bench-compare  # cross-backend benchmark harness (§C3)
```

Requires Go 1.26+. No C toolchain needed for the default build. Architecture
and task history live in [`SPEC.md`](SPEC.md) (caveman-encoded, see
[`FORMAT.md`](FORMAT.md)); design rationale in [`docs/`](docs/); performance
numbers and measurement policy in [`docs/benchmarking.md`](docs/benchmarking.md).

## License

TBD.

## License

GoAI is licensed under the **Mozilla Public License 2.0** (see [LICENSE.md](LICENSE.md)):
file-level copyleft. In practice:

- **Using GoAI in your product** — statically or dynamically linked, open or
  closed source — imposes **no obligations on your own code**.
- **Modifying GoAI's files** — those modified files (and only those) must remain
  available under the MPL, with source offered to recipients of your binaries.

