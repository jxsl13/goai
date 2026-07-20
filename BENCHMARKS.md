# Benchmarks — GoAI vs the incumbents

> **In plain terms:** this page answers one question — *how fast is GoAI
> compared to the software people actually use today?* Every table pits GoAI
> against an industry-standard incumbent (llama.cpp, vLLM, PyTorch, NumPy,
> scikit-learn) on the **same machine, same data, same model weights**, and
> reports wins and losses with equal prominence. Numbers are point-in-time
> measurements on the named hardware and date; the authoritative, running log —
> with full methodology, variance and history — is
> [`docs/benchmarking.md`](docs/benchmarking.md).

**How to read the terms** (each explained once, here): **tok/s** = generated
tokens per second (higher is better). **GFLOP/s** = billions of floating-point
operations per second (higher is better). **GEMM** = general matrix multiply,
the core operation of deep learning. **Decode** = generating one token at a
time from a language model; **prefill** = processing the whole prompt in one
batch before decoding starts. **KV cache** = the attention key/value memory a
model keeps per generated token. **Quantization** = storing weights in fewer
bits (Q8_0 ≈ 8 bits, Q4_K ≈ 4.5 bits per weight) to cut memory and bandwidth.
**GGUF** = llama.cpp's single-file model format. **BLAS** = the decades-old
optimized linear-algebra libraries (Accelerate, OpenBLAS) behind NumPy/PyTorch.
**GQA** = grouped-query attention (several query heads share one key/value
head). **SIMD** = the CPU's vector math unit; **AMX** = Apple's matrix
coprocessor; **MPS** = Metal Performance Shaders, Apple's GPU compute library.

Two machines appear below:

- **M2 Pro** — Apple M2 Pro (darwin/arm64), macOS, Go 1.26; Python side
  numpy 2.5.1, torch 2.12.1 in the repo `.venv`.
- **RTX 3060** — NVIDIA GeForce RTX 3060 12 GB on a Linux/amd64 host (Zen 3
  CPU); GoAI's `-tags cuda` backend vs llama.cpp (prebuilt Vulkan,
  b9960–b10012) and vLLM 0.25.1 (torch 2.11+cu130).

---

## Scoreboard at a glance

| Category | Benchmark | GoAI | Best incumbent | Verdict |
|---|---|---|---|---|
| GPU LLM decode (single stream) | Mistral-7B Q4_K, RTX 3060 | 49.6 tok/s | llama.cpp Q8 41.6 tok/s | **GoAI +19%** (0.84× of their Q4_K_M) |
| GPU LLM decode (single stream) | TinyLlama-1.1B, RTX 3060 | 258.5 tok/s | llama.cpp Q8 245.4; vLLM fp16 103 | **GoAI leads both** |
| GPU LLM prefill | TinyLlama pp128, RTX 3060 | 5,600 tok/s | vLLM 10,729 | vLLM 1.9× ahead (open front) |
| Multi-request serving throughput | 64 concurrent, RTX 3060 | ≈5,503 tok/s† | vLLM 4,655 | **GoAI 1.18×** — continuous batching landed (†full-step, §1) |
| CPU GEMM f32 1024³ | M2 Pro, `GOEXPERIMENT=simd` | ≈2,590 GFLOP/s | torch-cpu ≈2,584 | **parity** (pure-Go path ≈2,100) |
| CPU GEMM f32, >L2 shapes | 512×2048×4096, M2 Pro | 1,695 GFLOP/s (raw AMX) | Accelerate 1,294 | **GoAI +31%, in pure Go** |
| Classical ML fit | 6 methods vs scikit-learn 1.9.0 | see §5 | scikit-learn | **wins the heavy ensembles** (GBM 9.2×, RF 3.4×, past sklearn's own parallel fit); C cores win tree/SVC |
| CPU tokenizer encode | GPT-2 BPE 1 MB, M2 Pro | ≈28.2 MB/s (6.7M tok/s) | tiktoken Rust 18.8 | **GoAI 1.50×** (237,208-token parity) |
| GPU matmul (Apple) | f32 1024³, M2 Pro | 1,376 GFLOP/s (Metal) | torch-mps 4,171 | 3.0× behind (MPS-kernel ceiling) |
| Apple-GPU LLM decode | 17.7 M-param toy, M2 Pro | 236 tok/s | llama.cpp Metal 723 | ≈3.1× behind at toy size (see caveat) |
| Apple-GPU LLM decode (production) | TinyLlama-1.1B ~4-bit, M2 | 9.9 tok/s | llama.cpp 197, MLX 231 | ≈20–23× behind BOTH Apple engines (§2) |
| Training step (fwd+bwd) | GPT dim512/6L seq256, M2 Pro | Metal 3,263 / cpu 2,257 tok/s | torch-mps 12,904; torch-cpu 5,058 | 2–4× behind (fusion + MPS ceiling, §6) |
| Vision fwd/train (toy) | ViT+CNN 32²/batch-8, M2 Pro | see §7 | torch-mps | 2.4–40× behind (ViT per-image loop → T908) |

Losses are listed with the same care as wins — each has a diagnosed cause and,
where one exists, a booked lever (see
[Where GoAI loses today](#where-goai-loses-today) and
[Not yet measured](#not-yet-measured--booked-benchmark-tasks)).

---

## 1. Single-stream LLM decode on an NVIDIA GPU — vs llama.cpp and vLLM

*RTX 3060, real GGUF checkpoints, greedy decode, coherence-gated output;
llama.cpp prebuilt Vulkan via `llama-bench -ngl 99`; measured 2026-07-16/18,
re-validated on current main. Source: docs/benchmarking.md, sections "The
decode scoreboard", "Q4 across scales", "Three-way head-to-head".*

GoAI's CUDA decoder (built from scratch on nvrtc + cuBLAS, CUDA-graph
capture, Q4_K weights resident in ggml's own super-block format) vs the
incumbents on identical model files:

| Model | GoAI Q4_K graph | llama.cpp Q8 | llama.cpp Q4_K_M | vs their Q8 | vs their Q4_K_M |
|---|---:|---:|---:|---:|---:|
| TinyLlama-1.1B | 258.4 tok/s | 244 | 328.0 | **1.06×** | 0.79× |
| Qwen2.5-1.5B | 175.0 tok/s | 166 | 214.9 | **1.05×** | 0.81× |
| Qwen2.5-3B | 98.4 tok/s | 87 | 121.9 | **1.13×** | 0.81× |
| Mistral-7B | 49.6 tok/s | 41.6 | 59.1 | **1.19×** | 0.84× |

- **GoAI leads llama.cpp-Q8 at every scale, and the lead grows with model
  size** (1.06× → 1.19×) — the weight-bandwidth story: the bigger the model,
  the more decode is bound by weight bytes, the more the 0.5625-bytes/weight
  Q4_K format and a near-ceiling GEMV (matrix-vector multiply) pay.
- **The same-precision-class comparison is honestly open**: llama.cpp's
  Q4_K_M stays 1.19–1.27× ahead at similar weight bytes — their iterative
  quantization encoder plus fused-attention margin. At Qwen2.5-0.5B (Q4-
  ineligible) GoAI's Q8 graph path still leads their Q8, 316 vs 306 tok/s.
- **Quality is gated, not assumed**: each run must produce the verified
  answer (e.g. "Paris"), a gate that caught a real tokenizer bug parity
  tests could not see; at 7B, Q4_K decode is token-for-token identical to
  Q8 over the whole measured run.

Against the professional serving engines (TinyLlama-1.1B, batch = 1, same
GPU, 2026-07-18 — vLLM 0.25.1 runs fp16 with eager mode + Triton attention,
its documented pip-only setup):

| Metric | GoAI | llama.cpp | vLLM | Winner |
|---|---:|---:|---:|---|
| Decode tg128, batch=1 | **257** (Q4_K) | 244 (Q8) | 103 (fp16) | **GoAI** |
| Prefill pp128, batch=1 | 5,600 (f16) | 8,474 (Q8) | **10,729** (fp16) | vLLM |
| Batched decode, 64 concurrent | **5,503**† | ≈244* | 4,655 | **GoAI 1.18×** |
| Batched decode, saturated (b768) | **≈8,790**† | ≈244* | — | continuous-batching peak |

\* llama.cpp is a single-stream engine here — no continuous batching, so its
aggregate equals the single-stream rate.
† GoAI now has a PagedAttention-style paged KV pool + continuous batching
(admit/evict, validated bit-identical to eager decode). These are
graph-captured **full-step** f16-KV GQA decode numbers (22 layers + final
RMSNorm + logits GEMM [batch,dim]×[dim,32000] — the head vLLM's figure also
pays) on the synthetic layer stack; throughput is weight-value-independent so
these are representative of the real model. Only the argmax reduction (one
memory-bound pass over the logits, <1%) is omitted. What's still booked: a
real-model end-to-end run with variable sequence lengths + real scheduling,
for a fully like-for-like serving figure.

**Verdict by regime:** single-user chat (batch-1 decode) — GoAI wins all
three engines. Prompt processing — vLLM's fused FlashAttention leads 1.9×
(GoAI's attention is cuBLAS-batched, not fused; the GEMMs themselves already
run at ≈51% of the card's f16 peak, near the cuBLAS ceiling). Many
concurrent users — GoAI now HAS the paged KV + continuous-batching
capability (PagedAttention-style paged pool + block tables, admit/evict
scheduling, CUDA-graph-captured f16-KV GQA decode), all validated
bit-identical to eager decode. On full-step synthetic throughput (layers +
final-norm + logits GEMM) it clears vLLM's published 64-concurrent number
(5,503 vs 4,655 = 1.18×) and saturates near 8,790 tok/s. Still booked: a
real-model end-to-end serving head-to-head (variable-length scheduling +
sampling) for a fully like-for-like figure
(`SPEC-worker-linux-amd64-cuda.md` §ROADMAP). Prefill remains the open front.

## 2. LLM inference on Apple silicon — vs llama.cpp Metal

*M2 Pro. Identical 17.7 M-parameter F32 GGUF exported by
`internal/benchcompare/exportgguf`, llama.cpp `llama-bench`, 2026-07-14,
independently reproduced 2026-07-20. Source: docs/benchmarking.md §T607.*

| Engine | Prefill (pp64) | Decode (tg64) |
|---|---:|---:|
| llama.cpp Metal (2026-07-20 session) | 17,397 ± 11,973 tok/s | 723 ± 36 tok/s |
| GoAI Metal batched decoder (same session) | 8,613 tok/s | 236 tok/s |

Honest reading: at this **toy size** decode is ≈3.1× behind (4.2× in the
2026-07-14 session — llama.cpp's own number moves between sessions);
llama.cpp's prefill error bar spans ±11,973 tok/s, so only decode is a
stable comparison. 17.7 M parameters is far below production size, and
Apple's Accelerate BLAS alone nearly saturates a model this small.

**Production scale (T887, 2026-07-20) — the caveat discharged, and the gap
widens.** A real TinyLlama-1.1B Q4_K_M GGUF (669 MB, downloaded), the same file
timed by both engines on M2 Metal (llama.cpp `llama-bench` b9960, 3 reps; GoAI's
batched quant decoder via `llamagpu.NewQuant`, best-of-3):

Three Apple-Metal engines on TinyLlama-1.1B at ~4-bit — GoAI and llama.cpp share
the Q4_K_M GGUF; MLX (Apple's own framework) runs its native 4-bit quant of the
same base model (converted via `mlx_lm.convert -q --q-bits 4`):

| Engine (TinyLlama-1.1B ~4-bit, M2 Metal) | Prefill | Decode (tg64) |
|---|---:|---:|
| MLX (Apple, native 4-bit) | 953 tok/s (pp56) | **230.6 tok/s** |
| llama.cpp Metal (Q4_K_M) | 1,754 tok/s (pp64) | 197.2 tok/s |
| GoAI Metal (Q4_K_M, batched quant) | 82 tok/s (pp64) | 9.9 tok/s |

The toy-size hope — that the gap narrows at scale where memory bandwidth dominates
— does **not** hold: it *widens*, from ≈3× at 17.7 M to ≈**20–23×** at 1.1 B. And it
is **not llama.cpp-specific**: Apple's own MLX decodes even faster than llama.cpp
(231 vs 197 tok/s), so GoAI trails *both* mature Apple-Metal engines ~20×. That
pins the cause on GoAI's kernel maturity, not one incumbent's tricks: GoAI's Q4_K
dequant kernels are one-thread-per-output, not MPS-class (§T416), so at 1.1 B the
dequant dominates every decode step, where llama.cpp's hand-tuned Metal Q4_K
kernels and MLX's fused Metal graph are years of decode-path engineering. It is not
a broken path — GoAI's prefill/decode ratio (8.3×) matches llama.cpp's (8.9×), it
loads the model at the correct config (vocab 32000, dim 2048, 22 layers, GQA 32:4)
and its quantized decode is f32-exact against gguf-py (the parity gates). The lever
is production-grade Q4_K Metal kernels + attention fusion — a large effort and the
core competency of both incumbents. Harnesses:
`internal/benchcompare/prod_decode_external_test.go` (GoAI, gated on
TINYLLAMA_GGUF), `llama-bench` (llama.cpp), `testdata/bench_mlx.py` (MLX).

What the batched recorder architecture is worth *within* GoAI on this machine is
measured precisely:

| GoAI path (M2 Pro) | Rate | vs per-op dispatch |
|---|---:|---|
| Llama decode, batched Metal | 263 tok/s | **24×** (per-op Metal: 7.5–160 tok/s by config) |
| Llama decode, batched Vulkan/MoltenVK | 245 tok/s | **21×** |
| Llama prefill, one batched `StepN` | 11,068 tok/s | **36–41×** over sequential steps |
| Long context @1920 KV | 72 tok/s | 17.6× via cooperative online-softmax attention |

## 3. Pure-Go CPU inference — the no-toolchain path

*M2 Pro, `GOEXPERIMENT=simd` f32 fast path (the default build stays
bit-exact scalar). Source: docs/benchmarking.md, f32 campaign §T656–T664 and
CPU serving arc T762/T777–T793.*

The same GPT forward (512 dim, 6 layers, 256 tokens) through the pure-Go CPU
campaign — AMX GEMM, NEON vectorized exp/GELU/softmax, profile-driven:

| CPU f32 GPT | Before | After campaign | Speedup |
|---|---:|---:|---:|
| Forward | ≈1,250 tok/s | **≈13,600 tok/s** | ≈10.9× |
| Training step | ≈1,325 tok/s | **≈1,930 tok/s** | 1.48× (now matmul-bound at the AMX ceiling) |

Serving-path levers on the CPU decode/prefill path (all output-exact,
bit-identical or machine-epsilon-gated): sparse MoE decode 4.0–7.8× per
token, batched prompt prefill up to 6.7× across all 31 architectures,
DeepSeek's absorbed-latent attention cache at 6.7× less KV memory, O(1)
recurrent decode for RWKV/Mamba/Jamba, and a latency-aware thread pool for
1.68× end-to-end decode.

**Honest open gap:** CPU *quantized* decode runs ≈8.8× slower than f32
decode (Q8_0, dim-256 model) because the CPU quantized matmul dequantizes
ggml blocks on the fly — quantization currently buys memory (4–8× less), not
CPU speed. The fix (a block-native quantized GEMV kernel) is flagged in the
log; on GPU the quantized decoders already run block-native.

### Tokenizer throughput — pure-Go BPE vs tiktoken

*M2 Pro, GPT-2 / r50k_base vocab, one 1,000,116-byte corpus, single-threaded
both sides. Both emit the **identical 237,208 tokens** (bit-exact parity — the
fairness anchor). GoAI = `go test -bench` average; tiktoken = best-of-9, its most
flattering measure, so the ratios if anything under-sell GoAI. Source:
`nlp/bpe_throughput_test.go`, companion
`internal/benchcompare/tokenizer_compare.py` (tiktoken 0.13.0). T882.*

| GPT-2 BPE, 1 MB | GoAI pure-Go | tiktoken (Rust core) | GoAI |
|---|---:|---:|---:|
| Encode | **≈28.2 MB/s** (6.7M tok/s) | 18.8 MB/s (4.5M tok/s) | 1.50× faster |
| Decode | **≈470 MB/s** (111M tok/s) | 392 MB/s (93M tok/s) | 1.20× faster |

GoAI's allocation-free byte-pair merge (§T625, tiktoken's own `byte_pair_merge`
algorithm in pure Go) tokenizes faster than tiktoken *as delivered to the host
language*: the tiktoken figure is end-to-end through its Python binding, which
materializes the 237k-token list a Python application consumes, whereas a Go
application gets a native `[]int32` with no cross-language marshalling. tiktoken's
`encode_ordinary` fast path measures the same ≈18.9 MB/s, so the gap is not a
special-token-scan artifact. The honest framing: this is a library-delivery win,
not a claim that pure Go out-computes the Rust core in isolation.

## 4. GEMM and core kernels — vs Accelerate, PyTorch, NumPy

*M2 Pro, f32, `GOEXPERIMENT=simd`; paired A/B medians. Source:
docs/benchmarking.md ADR-0026/ADR-0027 sections and the full comparison
matrix §T606.*

| Shape | GoAI NEON | GoAI raw AMX (pure Go) | Accelerate (cgo) | Winner on this box |
|---|---:|---:|---:|---|
| 256³ | 428 | 537 | 1,116 | Accelerate |
| 1024³ | 758 | ≈2,100 | ≈2,590 | Accelerate — **≈1.0× torch-cpu** |
| 2048³ | — | **2,325** | 2,100 | raw AMX (+11%) |
| 512×2048×4096 | — | **1,695** | 1,294 | raw AMX (+31%) |

- GoAI dispatches per shape to the measured winner. With a C toolchain it
  **matches PyTorch's CPU GEMM** (which uses the same Apple AMX hardware via
  Accelerate); with `CGO_ENABLED=0` — no C toolchain at all — the hand-written
  Plan9-assembly AMX kernel still reaches ≈2,100 GFLOP/s at 1024³, and on the
  large, cache-exceeding shapes it **beats Accelerate (and therefore the
  PyTorch/NumPy stack) outright, in pure Go**.
- The **default build** (no `GOEXPERIMENT=simd`) deliberately stays on the
  bit-exact f64-accumulating kernel (≈68 GFLOP/s at 1024³): correctness-first
  is the contract (§V10); speed is opt-in and parity-gated (rtol 2e-3,
  ADR-0021).
- Journey for context: 42× behind torch-cpu (scalar, 2026-07-06) → parity →
  ahead on >L2 shapes (2026-07-15).

Other kernels vs the same-box incumbents:

| Kernel | GoAI best | Incumbent | Gap |
|---|---:|---:|---|
| Conv2D (same per-call transfer+sync contract) | 253 GFLOP/s (Metal) | torch-mps 133 | **GoAI 1.9×** |
| Multi-head attention forward (CPU, simd) | 1.9 ms | torch-cpu fused SDPA 0.71 ms | 2.6× behind (fusion gap) |
| MatMul f32 1024³ on GPU | 1,376 GFLOP/s (Metal/MPS) | torch-mps 4,171 | 3.0× behind — Apple's closed MPS kernels; parked as a silicon/API ceiling |

## 5. Classical ML — vs scikit-learn

*M2 Pro, identical synthetic dataset (n=4000, d=20, 3 classes) written to a
shared CSV so both sides fit the exact same rows. Fit time, ms, minimum over 5
runs each side. scikit-learn 1.9.0 / numpy 2.5.1 (versions recorded per §V13).
Source: SPEC §T713–§T716/§T881, harness `classic/perfcompare_test.go` + committed
companion `testdata/bench_sklearn.py`.*

| Method | GoAI | scikit-learn | Verdict |
|---|---:|---:|---|
| Gradient boosting (100 trees) | 134 | 1,232 | **GoAI 9.2× faster** |
| Random forest (100 trees) | 80.8 | 271 (1 job) / 96 (all cores) | **GoAI faster than both** — GOMAXPROCS goroutines beat scikit-learn's own `n_jobs=-1` here |
| Gaussian naive Bayes | 0.42 | 0.66 | **GoAI 1.6× faster** |
| Decision tree (max_depth 12) | 18.1 | 13.9 | scikit-learn 1.3× (mature Cython splitter) |
| SVC (RBF kernel) | 6.8 | 3.4 | scikit-learn 2.0× (libsvm, decades-tuned C) |
| k-NN fit (k=5) | 4.5 | 0.27 | scikit-learn faster **at fit** — GoAI eager-builds the query index (note) |

GoAI wins the compute-heavy ensembles decisively — gradient boosting **9.2×**
(presort-once) and random forest even past scikit-learn's *own* parallel fit
(GOMAXPROCS goroutines vs `n_jobs=-1`) — plus Gaussian naive Bayes. scikit-learn
1.9.0's mature C cores are faster on the single decision tree (1.3×) and the RBF
SVC (2×); GoAI runs a pure-Go CART and SMO within 2× of those C floors, no
toolchain required. The k-NN row is a **fit-only artifact**: GoAI eagerly builds a
ball tree at `Fit` (moving cost off the query path), whereas
`KNeighborsClassifier.fit` builds its tree in optimized C — a fit+query
comparison, not yet harnessed, is the fair k-NN measure.

Honesty note (§B103): an earlier revision claimed "beats or matches every
method." It was measured against an **unrecorded** scikit-learn via an uncommitted
script and does not reproduce against 1.9.0 — GoAI's own numbers reproduce
exactly, but scikit-learn sped up on the tree and SVC (flipping parity/1.2× to
1.3×/2.0× behind) and the old 0.06 ms k-NN figure predated GoAI's eager ball tree.
Recording the incumbent version (§V13) and committing the companion is what makes
the scorecard rot-proof from here.

## 6. End-to-end training step — vs PyTorch

*M2 Pro, one GPT training step = forward + cross-entropy + backward (no optimizer
update, matching the Go benchmark), identical geometry (vocab 4096, ctx 256, dim
512, 8 heads, 6 layers, seq 256, batch 1, f32). tok/s = seq / step time. GoAI:
`internal/benchcompare` BenchmarkGPTTrainingStep (`GOEXPERIMENT=simd` cpu, Metal,
Vulkan). torch 2.12.1, companion `testdata/bench_gpt_train_torch.py` (torch-cpu 8
threads, torch-mps), median of 12. T883.*

| Training step | GoAI | PyTorch | Gap |
|---|---:|---:|---|
| CPU | 2,257 tok/s (simd) | torch-cpu 5,058 | torch 2.24× ahead |
| Apple GPU | 3,263 tok/s (Metal) | torch-mps 12,904 | torch-mps 3.95× ahead |
| Vulkan | 1,966 tok/s | — | MoltenVK; no torch Vulkan path |

A loss, and an expected one. Two diagnosed causes, both already on the
[losses table](#where-goai-loses-today):

- **Apple GPU (3.95×):** GoAI's Metal training runs the autograd tape op-by-op —
  each op is its own command-buffer commit + wait (≈0.27 ms dispatch floor), and a
  6-layer forward+backward is hundreds of ops. PyTorch dispatches one fused MPS
  graph and synchronizes once. GoAI's matmuls already call the same MPS kernels
  (§4), so the gap is dispatch + fusion, not raw GEMM: the batched recorder that
  gives 24× on *decode* (ADR-0019) buys only ≈1.4× on this training shape (§T411,
  matmul-dominated at seq 256), leaving the ≈3× MPS-tuning ceiling.
- **CPU (2.24×):** GoAI's f32 GEMM matches torch's Accelerate/AMX (§4), but a
  training step is not all GEMM — PyTorch fuses scaled-dot-product attention and
  its autograd backward, where GoAI runs separate NEON kernels (its CPU attention
  is 2.6× behind torch's fused SDPA on its own). "No interpreter overhead" does not
  beat a decade of fused-kernel engineering here.

Honest read: GoAI's *inference* decode is competitive-to-ahead (§1–§3), but the
*training* step trails PyTorch 2–4× on this box — fusion and MPS-kernel tuning, not
a pure-Go penalty (the pure-Go f32 GEMM is at parity). The lever is graph/kernel
fusion, tracked for both backends.

## 7. Vision models — forward and training vs PyTorch

*M2 Pro, a ViT (807,306 params: patch 4, dim 128, depth 4, heads 4) and a small
CNN (1,562 params: two 3×3 conv stages → global-avg-pool → head) at 32×32×3,
batch 8, f32. img/s = batch / step time. Forward-only and a full training step
(forward + cross-entropy + backward, no optimizer). GoAI: `internal/benchcompare`
BenchmarkViT*/CNN* (`GOEXPERIMENT=simd` cpu, Metal, Vulkan). torch 2.12.1,
companion `testdata/bench_vision_torch.py`; both models carry the identical
807,306 / 1,562 params — the fairness anchor. T884.*

| img/s | GoAI cpu | GoAI Metal | torch-cpu | torch-mps |
|---|---:|---:|---:|---:|
| ViT forward | 775 | 111 | 2,034 | **4,352** |
| ViT train | 155 | 39 | 652 | **1,592** |
| CNN forward | 8,701 | 7,375 | 25,744 | 17,832 |
| CNN train | 2,618 | 1,083 | 9,453 | 6,017 |

torch is ahead everywhere, but the two models fail differently — and one gap is a
GoAI inefficiency worth calling out, not a platform ceiling:

- **ViT on the GPU is catastrophic (≈40× behind torch-mps), and it is fixable.**
  `vision.ViT.Forward` loops over the batch INTERNALLY — it slices each of the 8
  images and runs a separate length-65 encoder forward+backward instead of one
  batched [8,65,128] attention. Every per-image op pays the Metal dispatch floor
  (≈0.27 ms), ×8 images ×hundreds of ops. torch batches the attention in one pass.
  On CPU (no dispatch floor) the same defect is only 2.6–4.2×. Booked as **T908**:
  batch the ViT encoder (the GPT/MHA path already does). The single biggest vision
  lever.
- **CNN is a normal fusion gap (2.4× fwd / 5.6× train on the GPU).** The CNN is
  natively batched on both sides, so this is the same fused-conv + fused-backward
  story as the training-step section (§6): torch's fused kernels vs GoAI's separate
  conv/pool/backward ops.

Note the inversion: for these toy models GoAI's CPU beats its own Metal/Vulkan
(dispatch-bound at 32²/batch-8) — the GPU pays off only at larger shapes.

## 8. Speculative and assisted decoding — measured on real trained models

*M2 Pro, in-repo-trained char-level models (the schemes' value depends on
acceptance rates, so they are measured on genuinely trained models, not
random weights). Source: docs/benchmarking.md §T434–T455.*

| Scheme | Acceptance | Lossless | Speedup |
|---|---:|:---:|---:|
| Medusa chain (3 trained heads, drafting folded into the verify pass) | 97% | no (typical acceptance) | **3.08×** |
| Prompt-lookup (n-gram, no draft model) | 15% | yes | **1.80×** |
| Draft-model speculative (1-layer draft) | 81% | yes | 1.12× (dispatch-bound; pays on large targets) |
| Self-Extend (4× the trained context, no fine-tuning) | — | — | holds CE 0.52 where plain attention degrades to 1.49 |
| Prompt-prefix reuse (`PrefillAppend`, 96 shared + 8 new tokens) | — | bit-exact | **7.13×** prefill latency |

These are GoAI-internal A/B numbers (scheme on vs off, same model, same
hardware) — the cross-engine comparison of speculative stacks is a separate,
unmeasured axis. Lesson worth the table: on a dispatch-bound decoder the
*round cost*, not the acceptance rate, decides the win — free-drafting
schemes (Medusa, prompt-lookup) pay off where a draft *model* does not.

## 9. Correctness parity — the benchmark behind every benchmark

Speed claims are only meaningful if both sides compute the same thing.
Every GoAI number above sits on mechanically enforced parity gates
(§V1/§V3/§V16): forward and backward validated against PyTorch goldens (f64
rtol ≈1e-12), 31 Hugging Face architectures anchored at ≈1e-8 against the
transformers reference, BPE tokenization bit-exact against tiktoken's own
test vectors, GGUF quantized decode f32-exact against gguf-py for **every**
ggml quant type, and llama.cpp itself loading and byte-identically decoding
GoAI-written GGUF files. A speed win that changes outputs is a bug, not a
result — the quantized-decode quality gates (teacher-forced agreement ≥97%)
are recorded next to the speed tables in the log.

---

## Where GoAI loses today

Kept current and prominent on purpose (measurement discipline §V22 — an
honestly documented deficit with a root cause is a deliverable):

| Deficit | Size | Diagnosed cause | Lever |
|---|---|---|---|
| GPU prefill vs vLLM (RTX 3060) | 1.9× | fused FlashAttention + zero-overhead scheduling on their side; ours is cuBLAS-batched attention | fused-attention prefill (worker roadmap FRONT A) |
| ~~Multi-request serving throughput vs vLLM~~ | **CLOSED** | was ≈18× behind (no continuous batching); FRONT B built the paged KV pool + admit/evict continuous batching + graph-captured f16-KV GQA decode → now **1.18× ahead at 64-concurrent** (full-step 5,503 vs 4,655, §1), saturating ≈8,790 tok/s | remaining: real-model end-to-end run (variable-length scheduling + sampling) |
| Same-class Q4 decode vs llama.cpp Q4_K_M | 1.19–1.27× | their iterative quant encoder + fused attention | Q4_K encoder quality + attention fusion |
| Apple-GPU matmul vs torch-mps | 3.0× | Apple's closed MPS kernel tuning; measured as the platform ceiling | parked (§B39/§T410) — revisit only with new evidence |
| Training step vs torch-mps (Apple GPU) | 3.95× | op-by-op autograd dispatch (≈0.27 ms/op × hundreds) + MPS-kernel ceiling; torch dispatches one fused graph | tape recorder (≈1.4× at seq 256, §T411) + fusion |
| Training step vs torch-cpu | 2.24× | GEMM is at AMX parity, but torch fuses SDPA attention + autograd backward; GoAI runs separate NEON kernels | fused-attention/backward CPU kernels |
| safetensors full load vs safetensors-python | 1.45× | Rust core + mmap + zero-copy numpy views; ours is a pure-Go hostile-gated read+parse (8.4 vs 12.2 GB/s) | mmap the file to skip the read copy |
| safetensors one-tensor load vs `safe_open` | 2.69× | their mmap+memcpy vs our read()+frame double-copy; both read only that tensor's bytes | mmap-based partial load, no intermediate buffer |
| GGUF full load vs gguf-py | 5.4× | `decodeTensor`'s F32/F16 path is a per-element decode loop (`Float32frombits` per element), not a bulk copy — a fixable inefficiency, not a ceiling (GoAI's own safetensors reader is already bulk at 8.4 GB/s vs GGUF's 2.2) | bulk F32/F16 decode fast path → **T907** (format/gguf) |
| ViT training vs torch-mps (Apple GPU) | ≈40× | `vision.ViT.Forward` runs the batch as 8 separate per-image encoders → each op pays the Metal dispatch floor ×8; torch batches attention in one pass (on CPU the same defect is only 2.6–4.2×) | batch the ViT encoder → **T908** (vision) |
| CPU attention vs torch fused SDPA | 2.6× | operator fusion | candidate fused-attention CPU kernel |
| CPU quantized decode vs own f32 | 8.8× | on-the-fly block dequantize in the hot loop | block-native quantized GEMV (flagged) |
| Apple decode vs llama.cpp AND MLX | ≈3.1× toy → **≈20–23× at 1.1B ~4-bit** (T887) | one-thread-per-output Q4_K dequant (§T416); trails BOTH mature Apple-Metal engines (llama.cpp 197, MLX 231 vs GoAI 9.9 tok/s) so it is kernel maturity, not one incumbent | production-grade Q4_K Metal kernels + attention fusion |

## Not yet measured — booked benchmark tasks

Comparisons this file deliberately does **not** claim yet; each is booked as
a spec task with an id you can grep in [`SPEC.md`](SPEC.md):

| Axis | Incumbent | Task |
|---|---|---|
| Committed, versioned sklearn timing script (today the sklearn side of §5 is reproducible only by an ad-hoc script) | scikit-learn | T881 |
| SGLang datapoint beside vLLM (installs on the worker box, never measured) | SGLang | T888 |

Reinforcement learning and probabilistic methods are currently out of perf
scope: GoAI's RL agents are canonical reference implementations validated on
convergence, and no incumbent comparison is claimed for them.

## Method — how these numbers are made

The full rules live in [`docs/benchmarking.md`](docs/benchmarking.md); the
non-negotiables (§V22, §C3):

1. **Same machine, same data, same weights** — incumbents run on the exact
   hardware and inputs GoAI runs on; identical GGUF files where applicable.
2. **Warm-up excluded, repeats reported** — medians of repeated runs;
   variance shown where it is material (error bars in the tables above).
3. **Real workloads over micro-ops** — end-to-end forwards, decode loops and
   fits; isolated kernel wins that do not move the real workload are
   recorded as non-levers.
4. **A/B on the same session** — before/after comparisons never cross a
   machine, thermal state or build config silently.
5. **Losses are published** with root causes, and "X is the bottleneck" is
   itself a measured claim.
6. **Correctness gates every speed number** — parity suites, golden tests
   and end-to-end generation checks run beside the timers.

## Reproduce

| Table | Command(s) |
|---|---|
| Python bench venv (incumbents) | `python3 -m venv .venv && .venv/bin/pip install -r testdata/requirements-bench.txt` (versions pinned per §V13) |
| Comparison matrix + Python side (M2) | `make bench-compare bench-python`, render with `go run ./internal/benchcompare/rendertables` |
| CPU f32 campaign | `GOEXPERIMENT=simd make bench-compare` |
| CUDA decode scoreboard (worker) | `go test -tags cuda -run TestCUDAQ4KGraphDecodeSweep ./backend/cuda/` |
| llama.cpp side (worker) | `scripts/bench-llamacpp.sh <model.gguf>` |
| vLLM side (worker) | uv venv + env flags as documented in docs/benchmarking.md ("Three-way head-to-head") |
| Apple toy head-to-head | `go run ./internal/benchcompare/exportgguf` + `llama-bench`, per docs/benchmarking.md §T607 |
| Classical ML vs scikit-learn | `make bench-classic-python` (GoAI + scikit-learn 1.9.0 on the same CSV; §5, T881) |
| BPE tokenizer vs tiktoken | `go test ./nlp -run '^$' -bench BenchmarkGPT2` + `.venv/bin/python internal/benchcompare/tokenizer_compare.py` (§3, T882) |
| GPT training step vs torch | `make bench-gpt-train-python` + `GOEXPERIMENT=simd VK_ICD_FILENAMES=$VK_MOLTENVK_ICD go test -tags vulkan ./internal/benchcompare -run '^$' -bench BenchmarkGPTTrainingStep` (§6, T883) |
| GEMM AMX head-to-head | `GOEXPERIMENT=simd go test ./backend/cpu -run '^$' -bench GEMM -benchtime 10x` |

## Further reading

- [`docs/benchmarking.md`](docs/benchmarking.md) — the running log: every
  optimization rung, regression policy, and the measurement lessons.
- Hoefler & Belli, *Scientific Benchmarking of Parallel Computing Systems*
  (SC '15) — the variance/warm-up/honest-reporting canon these rules follow.
- Georges, Buytaert & Eeckhout, *Statistically Rigorous Java Performance
  Evaluation* (OOPSLA '07) — why repeated runs with variance beat single
  numbers.
- [`SPEC.md`](SPEC.md) §V22/§C3 — the in-repo law behind the method, and the
  §T ids cited throughout this page.
