# Changelog

All notable changes per §T task. Dates ISO. Pre-1.0: API unstable (§V8).

## [Unreleased]

### T44 — NPU: documented honest non-goal (2026-07-06)
- `backend/npu`: pure-Go documentation package with `Available() bool` → **false**,
  so feature-detection probes NPU support uniformly and gets an explicit "no",
  never a silent gap (§V4). Dual-audience godoc (professional + newcomer, §V17)
  with a runnable example (`go test` verifies it).
- **Decision (ADR-0011, §R45):** op-level NPU dispatch from Go is impractical —
  Apple ANE and Intel NPU are whole-model, inference-only (no per-op or gradient
  API); DirectML is op-granular and trainable but only via native C++/COM on
  Direct3D12 with no Go binding. NPU therefore belongs at the **model level**
  (future L5 CoreML/ONNX/OpenVINO/DirectML runner), not GoAI's L1b op backend.
  Confirmed vs Apple/Microsoft/Intel docs via research-lite.
- Closes the accelerator arc: CPU-SIMD, Metal, CUDA, Vulkan shipped; NPU honestly
  scoped out. Full suite green ×15 (new npu pkg), cross-compiled.

### T43 — portable Vulkan compute GPU backend (2026-07-06)
- `backend/vulkan`: optional **vendor-neutral** backend behind `-tags vulkan`
  (cgo, `-lvulkan`). Runs on AMD/Intel/NVIDIA and Apple (via MoltenVK). f32 MatMul
  via a hand-written GLSL compute shader (Vulkan has no BLAS, §R39); all other ops
  fall back to Pure-Go (§I4). Serves inference **and** training (matmul VJP GEMMs
  dispatch here via `autograd.NewTapeOn(vulkan)`).
- `shaders/matmul.comp`: row-major f32 GEMM, one invocation per output element,
  16×16 workgroup, dims via push constants. Compiled to `matmul.spv` by
  `make vulkan-spv` (glslc) and embedded — a build artifact, like `-tags cuda`
  needing the CUDA toolkit (ADR-0010).
- Mirrors Metal/CUDA (tag-free `doc.go` keeps the default `CGO_ENABLED=0` build
  green — package = doc only without the tag).
- **§V16:** the full Vulkan compute orchestration (compute-queue setup,
  host-coherent storage buffers, 3-binding descriptor set, push constants,
  compute pipeline, dispatch ceil(N/16)×ceil(M/16), GID→row/col mapping)
  **confirmed vs the Vulkan spec + compute tutorials** (§R44; API convention =
  definitional, no paper). CI (§B36): §V3 cross-ref vs ref within rtol(K)=1e-6·√K,
  an unaligned-rectangular guard on the dispatch-overhang mask, an fwd+bwd
  GPU-training test, §C3 gate benchmarks.
- **HOST-VERIFIED on Apple M2 Pro via MoltenVK** (Vulkan toolchain installed via
  brew: shaderc/glslc + vulkan-loader + vulkan-tools). The shader compiles to
  `matmul.spv` (committed); the cgo directive uses `pkg-config: vulkan` for
  portability. Runtime portability fix: MoltenVK is a portability driver, so the
  instance enables `VK_KHR_portability_enumeration` + the ENUMERATE_PORTABILITY
  flag and the device enables `VK_KHR_portability_subset` — detected at runtime,
  skipped on native Linux/Windows (one portable code path). All `-tags vulkan`
  tests pass on-GPU: §V3 cross-ref green ∀ shapes, **GPU training converges
  identically to CPU (loss 0.0371 == 0.0371)**, unaligned + fallback green.
- Pure-Go suite green ×15, cross-compiled (win/amd64, linux/amd64+arm64).

### T42 — CUDA/cuBLAS GPU backend (2026-07-06)
- `backend/cuda`: optional NVIDIA backend behind `-tags cuda` (cgo, linux/windows).
  f32 MatMul via `cublasSgemm`; every other op falls back to Pure-Go (§I4). Serves
  **both inference and training** — `autograd.NewTapeOn(cuda)` dispatches the
  forward and both backward GEMMs of the matmul VJP to the GPU (user GPU priority).
- Mirrors the Metal backend exactly (ADR-0009): a tag-free `doc.go` keeps the
  default `CGO_ENABLED=0` build green (package = doc only without the tag); the
  cgo/cuBLAS files are excluded. Row-major C=A·B via the column-major idiom
  `cublasSgemm(OP_N,OP_N,N,M,K,B,N,A,K,C,N)`.
- **§V16:** cuBLAS SGEMM row-major↔column-major mapping + `cudaGetDeviceCount`
  detection **confirmed vs NVIDIA cuBLAS docs** (§R43; API convention =
  definitional, no paper). CI (Linux/Windows CUDA runners, §B35): §V3 cross-ref
  vs ref within rtol(K)=1e-6·√K, a rectangular-shape guard on the idiom, an
  fwd+bwd GPU-training test, and §C3 gate benchmarks vs the ceiling Pure-Go cpu.
- **Not host-verifiable** on this arm64/macOS host (no CUDA) — like archsimd
  (§B13); Go side compiles under the metal-identical pattern, gofmt-clean.
  Verified here: pure-Go suite green ×14, cross-compiled, cuda pkg builds doc-only.

### T41 — mixed-precision training (2026-07-06)
- `nn.MixedPrecision`: keeps an **fp32 master** copy of each weight while the model
  computes with an f16/bf16-rounded copy; `Sync()` rounds masters → half before
  each forward, `UnscaleGrads()` maps half-weight gradients back to the fp32
  masters (÷ loss scale) and flags inf/nan. The optimizer runs over the masters.
- `nn.LossScaler`: **dynamic loss scaling** with the torch/Apex defaults — initial
  2¹⁶, growth ×2 after 2000 clean steps, backoff ×0.5 on overflow, and the step is
  **skipped** (not clamped) when a gradient is inf/nan; scale floored at 1.
- `autograd.Tape.BackwardScaled(loss, S)`: seeds the output cotangent with S — the
  loss-scaling primitive (every gradient ×S), unscaled before the optimizer step.
- **§V16 both tiers:** tier-1 the recipe trains end-to-end (MSE 2.80→0.00 for f16
  **and** bf16 compute). tier-2 all three techniques + exact GradScaler constants
  **confirmed vs Micikevicius et al. 2018 §3.1-3.3 AND PyTorch grad_scaler.py**
  via research-lite (§R42).
- **Verification:** master-weights recover sub-ULP updates a direct-fp16 weight
  loses (100×5e-4 at w=3.0: fp16 frozen, master→2.95); loss scaling rescues a
  2e-8 gradient that underflows to 0 in fp16; scaler grow/backoff/skip/floor
  logic; `BackwardScaled` scales grads exactly. fp32 accumulation already holds
  (§V10). Full suite green ×14, cross-compiled.

### T27 — f16 + bf16 dtypes (2026-07-06)
- L0 gains two 16-bit float dtypes (`tensor.F16` IEEE-754 binary16, `tensor.BF16`
  bfloat16), stored as raw `[]uint16` bits. Storage/allocator/`atF64`/`setF64`/
  `Cast` all handle them; `tensor.U16()` exposes the raw bits. Unblocks
  mixed-precision (T41) and native quant kernels.
- `tensor/half.go`: `f16↔f32`, `bf16↔f32`. f32→16 uses **round-to-nearest-even**;
  bf16→f32 is a pure left-shift; NaN is preserved (never turned into Inf).
- **§V16 both tiers:** tier-1 golden parity — f32→f16 bit-exact vs `numpy.float16`,
  f32→bf16 bit-exact vs `torch.bfloat16` (§V1). tier-2 the RNE/shift/NaN
  conventions **confirmed vs IEEE 754-2019, numpy docs, and PyTorch BFloat16.h +
  Eigen** via research-lite (§R41).
- **Verification:** bit-parity over 16 edge values (zero, ±1, subnormals 2⁻¹⁴/2⁻²⁴,
  f16-max 65504, 1/3); explicit RNE tie-to-even (0x3F808000→0x3F80 not truncation);
  NaN-preservation; tensor `Cast` round-trip. Full suite green ×14, cross-compiled.

### T40 — LoRA low-rank adapters (2026-07-06)
- `nn.LoRALinear`: y = x·W + (α/r)·(x·A)·B over a **frozen** base W (Hu et al.
  2021). A Kaiming-initialized, B zero → the adapter starts as a no-op. `Params()`
  returns only A,B, so optimizers never touch the base.
- **§V16 both tiers:** tier-1 forward parity vs torch (f64 rtol 1e-12); tier-2
  formula (BA decomposition, Gaussian-A/zero-B init, α/r scaling, frozen base)
  **confirmed vs the paper §4.1 AND microsoft/LoRA + HF PEFT** via research-lite
  (§R27).
- **Verification:** golden parity; B=0 ⇒ output equals the frozen base (§R27
  no-op start); **fine-tunes MSE 2.99→0.00 while the base W stays byte-for-byte
  frozen** and only A,B receive gradients. Full suite green ×14, cross-compiled.

### T39 — quantized-weight inference (2026-07-06)
- `gguf.QMatMul(x, weight, Q8_0|Q4_0, n, k)`: y = x · dequant(W[N,K])ᵀ with the
  quantized weight **dequantized one row at a time** — a quantized linear layer
  that never materializes the full-precision matrix (the memory point of
  quantized inference). f64 accumulation (§V10); the per-block dequant is the
  ggml-verified path (§R19/§R21, no new algorithm).
- **Verification (§V1):** parity vs gguf-py (dequant + f64 matmul) at rtol 1e-5
  for both Q8_0 and Q4_0; memory footprint **f32 → 3.8× smaller (Q8_0), 7.1×
  smaller (Q4_0)**; shape/byte-count error cases. Full suite green ×14,
  cross-compiled.
- Closes the quantization story: T22 dequant-on-load + T39 quantized matmul →
  a GGUF-quantized model can run without full-precision weights.

### T38b — Llama-family pt2: SwiGLU + GQA (2026-07-06)
- New `OpSiLU` (z·σ(z)) + VJP (ref + cpu). `nn.SwiGLU` FFN
  (SiLU(x·Wgate)⊙(x·Wup))·Wdown — fully trainable composition.
- **GQA**: generalized `OpMHA` with a `kv_heads` attr — query head h shares KV
  head h/(heads/kv_heads) (contiguous groups); `kv_heads==heads` = MHA (default,
  backward-compatible), `kv_heads==1` = MQA. Forward + VJP (grouped dK/dV
  accumulation across query heads sharing a KV head).
- **§V16 both tiers:** tier-1 parity vs torch (SiLU, SwiGLU FFN, GQA 4q/2kv) at
  f64 rtol 1e-12; tier-2 formulas **confirmed vs papers AND HF source** via
  research-lite (§R30 SwiGLU, §R31 GQA).
- **Verification:** SiLU/SwiGLU/GQA golden parity; SwiGLU trains (grads reach all
  3 matrices); §V2 gradchecks for SiLU and GQA (validating the grouped KV
  gradient); MQA shape/error cases. Full suite green ×14, cross-compiled, metal.
- **Milestone: modern Llama-class building blocks complete** — RMSNorm, RoPE,
  SwiGLU, GQA/MQA, all paper-verified and (where applicable) trainable.

### T38 — Llama-family pt1: RMSNorm + RoPE (2026-07-06)
- New ops **`OpRMSNorm`** (x/√(mean(x²)+eps)·γ, no mean-sub/bias) and **`OpRoPE`**
  (rotary position embedding, HF rotate_half half-split pairing (i,i+d/2)), each
  with a hand-derived VJP; `ops.RMSNorm`/`ops.RoPE`, `nn.RMSNorm` layer.
- **§V16 both tiers:** tier-1 forward parity vs torch (f64 rtol 1e-12); tier-2
  formulas **confirmed vs the papers AND HF source** via research-lite (§R28
  RoPE, §R29 RMSNorm) — including that HF uses the half-split (not interleaved)
  RoPE convention.
- **Verification:** RMSNorm & RoPE golden parity (op + layer); §V2 gradchecks for
  both (RMSNorm x/γ, RoPE q); §V6 RoPE preserves each position's L2 norm
  (orthogonal rotation). Full suite green ×14, cross-compiled.
- SwiGLU + GQA split to §T38b.

### T37 — byte-level BPE tokenizer (gpt2) (2026-07-06)
- `nlp.Tokenizer`: tiktoken-gpt2-compatible byte-level BPE. Rank table (from real
  tiktoken, 50256 entries) doubles as merge priority (`byte_pair_merge`). Manual
  GPT-2 pre-tokenizer replicating the regex
  (`'s|'t|…| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`) since Go's RE2
  has no lookahead. Sennrich et al. 2016 (§R33).
- **Verification (§V1, §V15):** **encode bit-exact vs real tiktoken** across 12
  samples (ascii/unicode/digits/punct/contractions/whitespace); **decode∘encode
  byte-exact for ANY input** (all 256 bytes are base tokens) — round-trip unit
  tests + **fuzz 3.6M execs clean**.
- **backprop §B34:** fuzz caught the pre-tokenizer corrupting invalid UTF-8 via
  `[]rune` (→ U+FFFD); fixed to scan the original bytes with
  `utf8.DecodeRuneInString`, invalid bytes → byte tokens → round-trip byte-exact.
- Completes the text-in-text-out inference pipeline. Full suite green ×14,
  cross-compiled.

### T36 — sampling + generation (2026-07-06)
- `nlp.Sampler`: temperature → top-k → top-p (nucleus) → softmax → multinomial,
  with `Greedy()` (argmax) and explicit-seed determinism. `GPT.Generate` drives
  autoregressive decoding through the KV-cache.
- top-p follows the **paper definition** (Holtzman et al. 2019, §R34): nucleus =
  smallest desc-prob set with cumulative prob ≥ p (crossing token included,
  renormalized) — matches HF except at exact float-equality boundaries (noted).
- **Verification (§V1):** greedy = argmax deterministically; top-k restricts
  support to the k highest; top-p keeps only the nucleus ({0} at p=0.5, {0,1} at
  p=0.95 for a peaked dist); same-seed reproducibility; cold temp ≈ greedy, hot
  temp broadens; end-to-end greedy Generate is deterministic and correct length.
  Full suite green ×14, cross-compiled.
- **§V16 both tiers:** tier-1 deterministic unit tests; tier-2 **confirmed vs the
  paper** (Holtzman §3.1–3.3, eqs 2–4) AND HF logits_process.py via research-lite
  (§R34) — top-p nucleus (crossing token incl + renorm), top-k mask, and
  temperature-before-filtering all match.

### T35 — KV-cache + incremental autoregressive decoding (2026-07-06)
- Generalized `OpMHA` to `sq ≤ sk` (a short query batch against a longer cached
  K/V), with absolute-position causal masking — the full-attention path
  (sq==sk) is unchanged, so all existing goldens still pass.
- `nlp`: `KVCache` (per-layer K/V), `MHA.StepKV` (append token's k/v, attend all
  cached keys), `GPT.DecodeStep` (one-token incremental forward). Inference-only;
  the MHA VJP now guards sq==sk (training never uses the cache).
- **Verification (§V1):** incremental KV-cache decoding **reproduces the full
  forward logits at every position** (rtol 1e-11) — the standard decode
  optimization with zero accuracy loss; single-query-attends-all unit check.
  Full suite green ×14, cross-compiled, metal builds.

### T30b — full transformer on GPU: LLM inference AND training (2026-07-06)
- `tensor.Cast(dtype)` (f64↔f32, reusable for mixed-precision §T41).
- **A full GPT runs on the GPU** (Metal), both directions, validated vs CPU:
  - **Inference:** f32 GPT forward with a metal context — its projection/FFN/
    head GEMMs execute on the GPU; per-position argmax matches CPU and logits
    stay within the documented cross-tolerance.
  - **Training:** f32 GPT trained through a metal-backed tape (forward + backward
    GEMMs on GPU) — **CE 3.29 → 0.046, tracking the CPU run** (3.29 → 0.046).
- Delivers the user's #1 priority at the LLM level: **GPU acceleration for LLM
  inference AND training**, correctness-verified against the CPU truth. Default
  `CGO_ENABLED=0` build untouched (`-tags metal` isolation).
- Full default suite green ×14, cross-compiled.

### T34 — GPT end-to-end training, validated vs torch grads (2026-07-06)
- New op **`OpEmbed`** (row gather; VJP scatter-adds to the table) → token &
  position embeddings are trainable; `GPT.Embed` now runs through the dispatch,
  `GPT.Params()` exposes all 29 weights.
- **Full GPT backward matches REAL torch gradients** for every one of the 29
  weights (embed, LayerNorm, fused attention, FFN, residuals, LM head) at f64
  **rtol 1e-9** — torch computes the golden grads (`gen.py`: loss.backward()),
  we compute ours analytically; loss also bit-matches (1e-12). The whole
  training path is validated against the oracle.
- **GPT trains end to end: CE 3.29 → 0.015** under AdamW + global-norm grad
  clipping (§G5).
- Full suite green ×14, cross-compiled, metal builds.
- **Milestone: complete transformer TRAINING stack** — differentiable embed/LN/
  attention/FFN, AdamW/clip/schedule, GPT converging with torch-exact gradients.
  Unblocks GPU transformer training (metal-tape, §T30 pattern).

### T33 — AdamW + grad clipping + LR schedule (2026-07-06)
- `nn`: **AdamW** (`NewAdamW`, decoupled weight decay p←p·(1−lr·wd)−lr·m̂/(√v̂+ε),
  wd=0 ⇒ plain Adam); **ClipGradNorm** (global-L2-norm gradient clipping);
  **WarmupCosine** LR schedule (linear warmup → cosine decay to minLR) — the
  standard LLM training trio.
- **§V16 both tiers:** tier-1 AdamW golden **bit-exact vs real torch 2.12.1**
  (`verify_torch.py`, err 0.0); tier-2 formula **confirmed vs the defining paper**
  (Loshchilov & Hutter 2019 Alg. 2 line 12) AND torch source via research-lite
  (§R26, schema-free — no StructuredOutput crash).
- **Verification:** 3-step AdamW trajectory parity (rtol 1e-12); grad-clip scales
  a norm-5 gradient set to exactly norm 1 (each ×1/5), no-op below threshold;
  warmup+cosine hits 0.5/1.0/1.0/0.5/minLR at the key steps. Full suite green
  ×14, cross-compiled.

### T32 — trainable attention (fused MHA + SDPA backward) (2026-07-06)
- New op **`OpMHA`**: fused multi-head scaled-dot-product attention
  (Q,K,V)[seq,dmodel]→[seq,dmodel], heads + causal internal. Making it ONE op
  keeps head split/concat/mask off the tape → sidesteps §B15 (view gradients)
  entirely, unlike composing through Slice/Transpose views.
- Hand-derived fused SDPA VJP (`autograd/vjp_transformer.go`): dV=Aᵀ·dO,
  dA=dO·Vᵀ, dS=A⊙(dA−rowsum(dA⊙A)), dQ=scale·dS·K, dK=scale·dSᵀ·Q; causal drops
  j>i (A=0 there). f64 accumulation. Paper: Vaswani 2017 / FlashAttention §R32.
- `nlp.MHA.Forward` rewritten to projections + OpMHA + Wo — fully differentiable,
  forward numerically identical (T21/GPT goldens unchanged at rtol 1e-12).
- **Verification (§V2):** gradcheck of OpMHA vs finite differences, causal AND
  non-causal; **full MHA block trains end-to-end** (MSE 1.207→0.000, gradients
  reach all of Wq/Wk/Wv/Wo). Full suite green ×14, cross-compiled, metal builds.
- Unblocks transformer training (§T34) — the CPU reference for GPU transformer
  training.

### T30 — GPU-accelerated training (Metal), first delivery (2026-07-06)
- **ADR-0008 GPU strategy:** offload compute-bound ops (GEMM/conv) only;
  memory-bound per-op offload (elementwise/addbias) fails the §C3 gate
  (transfer≫compute) → §T28/§T29 dropped. Durable high-throughput path is graph
  residency (MPSGraph/CUDA graphs).
- `autograd.NewTapeOn(backend)`: forward AND backward ops dispatch to the chosen
  backend; ops it lacks fall back to cpu (§I4). ~15 lines, no new kernels.
- **GPU training works & is validated:** an f32 MLP trained with a metal-backed
  tape runs its Linear GEMMs (forward + both backward GEMMs of the matmul VJP)
  on the GPU and **converges identically to the CPU reference** (gpu loss 0.037
  == cpu loss 0.037). `-tags metal` test; default `CGO_ENABLED=0` untouched.
- Delivers the user's #1 priority (GPU for training) using the existing Metal
  GEMM + existing autograd — correctness-first, host-verified.
- Full default suite green ×14, cross-compiled.

### Scope expansion: GPU/accel-first + full LLM (2026-07-06)
- **§T backlog expanded (T27–T44):** dtypes f16/bf16; GPU op coverage +
  **GPU training via MPSGraph autodiff**; transformer training (LayerNorm/
  attention VJPs, AdamW, GPT end-to-end); LLM inference (KV-cache, sampling,
  BPE tokenizer, RMSNorm/RoPE/SwiGLU/GQA, quantized inference); LoRA;
  mixed-precision; CUDA + Vulkan + NPU backends. Priority: accel > LLM > rest.
- **§V16 VALIDATION-LADDER** now mandatory: ref-lib parity (tier 1) + defining
  scientific paper (tier 2, final). §R26–R35 seed the paper pointers.
- **GPU/NPU landscape validated** (research-lite, primary sources, §R36–R40):
  macOS MPSGraph exposes GPU autodiff → GPU *training* is viable on Metal;
  ANE inference-only; cuDNN cgo-only; Vulkan/WebGPU no built-in autodiff;
  DirectML/oneDNN trainable. Reshapes T30 (use MPSGraph gradient API).
- **T31 — LayerNorm VJP** (closes §B22): dL/dx=(1/σ)(a−mean a−x̂·mean(a·x̂)),
  a=g⊙γ; dγ=Σg⊙x̂; dβ=Σg. Gradcheck green vs finite differences →
  **transformer training unblocked**. nn.LayerNorm now trainable.
- **/loop restarted at 1-minute cadence** (LOOP.md, cron d168c8b7); research
  uses the schema-free research-lite workflow only (never the crashing built-in).
- Full suite green ×14, cross-compiled.

### Deep bug analysis: fuzz + round-trips + manual review (2026-07-06)
- **Native Go fuzz tests for all file formats** (§V15): safetensors
  (FuzzLoad + FuzzRoundTrip), npy (FuzzRead + FuzzRoundTrip), gguf (FuzzRead) —
  round-trips assert bit-exact recovery incl. NaN/±0/±Inf; robustness fuzz
  asserts no input can panic. Seeded with the real golden files.
- **3 real robustness bugs found & fixed** (all DoS-class, hostile-input only):
  - §B31 (fuzz-found): npy uint32 header length uncapped → ~4 GB `make` OOM;
    and data buffer pre-allocated from the *claimed* shape (npy streams, so no
    offset cross-check like safetensors/gguf). Fixed: 64 MiB header cap +
    1<<25 numel cap.
  - §B32 (manual analysis): gguf metadata arrays may nest (element type ARRAY)
    → deep nesting stack-overflow; and `make([]any, n)` pre-alloc up to 256 MB
    from a claimed length. Fixed: depth cap 64 + grow-not-pre-alloc.
  - §B28 (earlier /review): hostile tensor dims (2^63) → int overflow slips
    past size check → panic. Fixed: dim + running-product caps.
- **Verification:** race detector clean across parallel/stateful packages
  (cpu/nn/nlp/rl/autograd); 40 s+ fuzz per target with no crash after fixes;
  full suite green ×14, cross-compiled. New invariant **§V15 UNTRUSTED-IO** now
  fuzz-backed.
- **Deep-research web harness failed twice** (StructuredOutput/session limit) —
  correctness verified instead by: primary-source extractions (pass 1),
  bit-exact parity vs the *official* reference libs (torch/sklearn/gguf-py/
  safetensors in §T22/§T25/§T21/§T23), fuzzing, and manual review.

### Loop end state on arm64 host (2026-07-06)
- **28/29 §T tasks complete.** T11b remains, structurally gated: archsimd needs
  an amd64 runtime for V-CROSS (§B13); the NEON/arm64 half was **parked by
  measurement** (§B27) — the portable loop already sustains 55 GB/s (~1
  elem/cycle, flat L1→L2; `internal/simd` benchmarks added), hand-NEON would
  net ≤7% end-to-end and is obsoleted by upcoming archsimd-arm64.
- Resume conditions documented in `LOOP.md` (amd64 runner, archsimd-arm64
  release, or new §T tasks drawn from the §B candidate list).

### T24b — CV optimization + training gradients (2026-07-06)
- `backend/cpu/conv.go`: conv2d via **im2col + blocked GEMM**. Columns laid out
  in (c,ky,kx) order and zero-padding materialized as 0·w terms → accumulation
  order identical to the reference → **bit-identical** (§V3/§V11 tol 0). Ref
  conv now adds bias last (same linear order; torch golden unaffected).
- `autograd/vjp_cv.go`: conv2d VJP (gx/gw/gb), maxpool VJP (routes to first
  max, §B16 rule), avgpool VJP (uniform g/k²) — **§B24 resolved**: CV training
  unblocked.
- **Verification:** §V3 cross bit-exact over shapes/stride/pad/bias/dtypes;
  §V2 gradchecks for conv2d (both geometries, incl. bias), maxpool, avgpool
  against central finite differences; ref torch-golden still green after the
  bias-order change. **§V5: conv 8×8×28×28·16f 40.3ms → 1.11ms (~36×).**
  Full suite green, cross-compiled (§V4/§V7). `go vet` clean.

### T26 — RL basics: REINFORCE + DQN (2026-07-06)
- `rl` package: `Env` interface + deterministic chain MDP; **REINFORCE**
  (Williams 1992) with cross-episode EMA baseline, loss composed purely from
  existing differentiable ops (Sum(W ⊙ log softmax)); **DQN** (Mnih et al.
  2015) with replay buffer, target-network sync, ε-greedy decay, and the
  replaced-entry MSE trick (non-taken actions get zero gradient).
- **Verification:** discounted returns exact vs hand-computed; chain-env
  semantics; **REINFORCE reaches optimal 1.000** (early 0.982 → late 1.000);
  **DQN reaches 1.000** with greedy policy = right from every interior state
  and Q(s,right) > Q(s,left) ∀s (Bellman-consistent ordering). Deterministic
  seeds. Full suite green, cross-compiled (§V4/§V7). `go vet` clean.
- **backprop §B26:** REINFORCE initially learned LEFT — the within-episode
  mean-of-Gₜ baseline subtracts exactly the between-episode signal (2-step
  episodes → advantage ≈ noise → the 0.1 left-attractor wins). Baseline must
  be cross-episode (EMA of G₀): 0.100 → 1.000. Threshold untouched.
- **Milestone: all §T domain tasks complete** — linalg, autograd, NN training,
  transformers/LLM, CV, classic ML, and RL all live with verified instances.

### T25 — classic ML vs sklearn (2026-07-06)
- `classic` package: `LinearRegression` (normal equations + Cholesky, Golub &
  Van Loan §4.2), `SoftmaxRegression` (penalty-free multinomial logistic via
  our autograd: Adam warmup + full-batch GD polish to the unique convex
  optimum), `KMeans` (Lloyd, deterministic init, sklearn tie rule), `PCA`
  (cyclic Jacobi eigendecomposition of the n−1 covariance, G&VL §8.5).
- Golden from real sklearn 1.9.0 (`classic/testdata/classic.json`).
- **Verification (§V1):** OLS coef/intercept/predictions at 1e-8
  (condition-number-justified vs SVD lstsq); softmax-regression probabilities
  **1.3e-9** vs sklearn (bound 1e-5); kmeans labels exact + centers 1e-10; PCA
  eigenvalues 1e-9 + components sign-invariant (|cos|≈1). Edge cases error.
- **backprop §B25 (two root causes fixed, no tolerance weakening):**
  (1) separable blobs make the penalty-free MLE non-existent → golden switched
  to overlapping classes (well-posed unique optimum); (2) Adam oscillates near
  the optimum at constant LR → monotone GD polish after warmup.
- Full suite green, cross-compiled (§V4/§V7). `go vet` clean.

### T24 — CV: conv2d + pooling (2026-07-06)
- New ops `OpConv2D` (NCHW cross-correlation — torch convention, zero-padding,
  stride, optional bias input), `OpMaxPool2D`, `OpAvgPool2D` (stride defaults
  to kernel). Direct reference kernels with f64 accumulation (§V10);
  `ops.Conv2D`/`MaxPool2D`/`AvgPool2D`.
- Golden from real torch f64 (`backend/ref/testdata/cv.json`): stride/pad
  variants, with/without bias, both pool types.
- **Verification (§V1):** all cases match `F.conv2d`/`F.max_pool2d`/
  `F.avg_pool2d` at f64 rtol 1e-12; geometry/attr error cases. Full suite
  green, cross-compiled (§V4/§V7). `go vet` clean.
- §T24b tracks the im2col+GEMM fast path and conv/pool VJPs (§B24: CV training
  blocked until then, inference fine).

### T23 — LLM inference end-to-end (2026-07-06)
- `nlp.GPT`: decoder-only transformer inference (GPT-2-style pre-LN) — token+
  position embedding, L×[LN1→**causal** MHA→residual, LN2→FFN(GELU exact)→
  residual], final LN, LM head. `nlp.FromSafetensors` assembles the model from
  a tensor map (naming convention documented; §B19 torch-import transposes).
- `nlp.MHA` gains `Causal`: future positions masked to −∞ before softmax
  (weight exactly 0).
- Golden: tiny GPT (V=17, D=8, H=2, L=2) built in **real torch f64**, weights
  through our own safetensors reader (§T19 in anger), expected logits in
  `nlp/testdata/gpt.json`.
- **Verification (§V1/§G2):** full-forward **logits match torch at f64 rtol
  1e-12**; **causality bit-exact** (mutating the last token leaves all earlier
  positions identical, and does change its own position); prompt/vocab/ctx
  error cases; missing-tensor loader errors. Full suite green, cross-compiled
  (§V4/§V7). §B23: reference is torch (ggml binary unavailable), revisit.
- **Milestone: every §G "vollwertig" criterion now has a working instance** —
  L0–L3 green w/ parity, optimized CPU backend w/ V-BENCH numbers, safetensors
  IO, converging trained model, gated GPU backend, LLM inference path.

### T22 — GGUF reader + dequantization (2026-07-06)
- `format/gguf`: parser for GGUF v2/v3 — magic/version, full metadata KV
  taxonomy (ints, floats, bool, string, nested arrays), tensor infos with
  **dims reversed** into row-major, `general.alignment`-aware data section;
  sanity caps against malformed counts/lengths.
- Dequantization to F32 on load: **Q8_0** (f16 scale + 32×int8), **Q4_0**
  (f16 scale + 32 nibbles, −8 offset, ggml pairing i/i+16), **F16→F32**
  (full IEEE binary16 incl. subnormals/inf/NaN), raw F32.
- Golden: `gen.py` writes `sample.gguf` via the **official gguf lib**
  (llama.cpp project) + expected dequantized values from its `dequantize`.
- **Verification (§V1):** all tensor types match the reference dequantization
  at f32 exactness; metadata round-trips (u32/string/arch); f16 edge table
  (max-finite, subnormal, ±inf, NaN); malformed inputs error. Full suite green,
  cross-compiled (§V4/§V7). `go vet` clean.

### T21 — transformer inference blocks (2026-07-06)
- New ops: `OpSoftmax` (stable max-shift, last axis) + `OpLayerNorm` (torch
  semantics: biased variance, eps in sqrt, γ/β inputs) — ref kernels with f64
  accumulation (§V10); `ops.Softmax`/`ops.LayerNorm`; softmax VJP
  (y⊙(g−Σgy)). LayerNorm VJP deferred (§B22 — inference unaffected).
- `nn.LayerNorm` layer (Ba, Kiros & Hinton 2016; γ=1/β=0 init).
- `nlp.MHA`: multi-head scaled dot-product attention (Vaswani et al. 2017) —
  zero-copy head splits via Slice views, Kᵀ via Transpose view, everything
  through backend dispatch (metal GEMM applies automatically when enabled).
- Golden: `gen.py` now generates from **real torch 2.12.1** in f64
  (`nlp/testdata/transformer.json`).
- **Verification (§V1):** softmax, layernorm (op + layer) and **full MHA
  forward match torch at f64 rtol 1e-12**; softmax rows sum to 1 (§V6) and
  shift-invariance under +1000 (§V12); weight/shape error cases. Full suite
  green, cross-compiled (§V4/§V7). `go vet` clean.

### T20 — Metal/MPS GPU backend, first cgo gate PASSED (2026-07-05)
- `backend/metal` (strictly behind `-tags metal && darwin && cgo`): f32 GEMM via
  `MPSMatrixMultiplication` (ObjC bridge, ARC, shared UMA buffers, synchronous
  per §V14 — async batching later without API break). Registers only when an
  MPS device exists; all other ops fall back to Pure-Go (§I4). Without the tag
  the package is a doc file — the `CGO_ENABLED=0` default build is untouched
  (§V7/§I5).
- **cgo gate (§C2/§C3) passed with all three conditions:** Pure-Go ceiling
  documented (T12/T12b) → benchmark: **512³ 4.6× (272 GFLOP/s), 1024³ 12.7×
  (906 GFLOP/s)** vs the optimized cpu backend → threshold ≥1.5× clearly beaten.
- **Verification:** §V3/§V11 cross vs ref with documented K-scaled tolerance
  rtol(K)=1e-6·√K, shapes incl. odd/rectangular; §I4 fallback test; §V4
  skip-with-log when no GPU; default build + vet green. `make metal-test` /
  `make metal-bench`.

### T12b — GEMM 4-row register blocking (2026-07-05)
- `backend/cpu/gemm.go`: band kernels process 4 C-rows per pass — each B row is
  loaded once and reused ×4 (quarters B memory traffic). Per-element k-order
  stays ascending → results remain **bit-identical to ref** (§V3/§V11 tol 0);
  no tolerance trade. F32 twin keeps the f64 scratch accumulator (§V10).
- **§V5 delta (darwin/arm64):** 256³ f64 1.20ms→0.92ms (**+31%, 36.4 GFLOP/s**);
  512³ **50.6 GFLOP/s**; 128³ unchanged (cache-resident). Ref-vs-cpu now ~110×
  at 256³. archsimd FMA microkernel folded into §T11b (amd64 CI, §B13).
- **Policy note:** T20 (GPU/cgo) was deliberately deferred — §C2 requires the
  Pure-Go ceiling first; this task raises it. Full suite green, cross-compiled,
  `GOEXPERIMENT=simd` amd64 build green (§V4/§V7).

### T19 — safetensors reader/writer (2026-07-05)
- `format/safetensors`: `Save`/`Load` (+File variants) for the HuggingFace
  format — u64-LE header length, JSON header, raw LE C-order data. Writer is
  byte-deterministic (sorted names, contiguous offsets, 8-byte space padding);
  reader validates strictly: 100 MB header cap, size = shape·dtype, offsets
  tile the data section exactly (gaps/overlaps/trailing bytes error). F32+F64;
  other dtypes error clearly (§C4 pending).
- **Verification (§V1):** reads a golden written by the official Python lib
  (safetensors 0.8.0, via `make golden`); **bidirectional interop** — a
  Go-written file loads correctly in the Python lib; round-trip incl.
  non-contiguous views, scalars, empty tensors; byte-determinism (§V13); six
  malformed-file cases all rejected. All packages green ×5 runs (§V4/§V7).
- **backprop §B21:** zero-size tensors share their begin offset with the next
  tensor; unstable sort order made validation flaky → sort by (begin, end).
  Flaky test treated as a real bug, not rerun noise.

### T18 — end-to-end training convergence (2026-07-05)
- `nn/e2e_test.go`: deterministic 4-class Gaussian-blob dataset; MLP
  Linear(2,16)→ReLU→Linear(16,4) trained with fused CrossEntropy + Adam through
  the full stack (dispatch → tape → VJPs → optimizer).
- **Verification (§G5):** initial loss sanity-checked near chance ln(4);
  **converges 1.061 → 0.024, training accuracy 1.000**; ≥5× improvement gate;
  f32 variant also converges (the §V10 f64-accumulation/master-state guards pay
  off). All packages green, cross-compiled (§V4/§V7).
- **§B20 resolved:** torch 2.12.1 installed in `.venv` (user-approved);
  `testdata/verify_torch.py` replays the golden gradient sequences through real
  `torch.optim.SGD/Adam` — **all optimizer goldens bit-exact (max err 0.0)**.
- **Milestone: L3 complete (T15–T18)** — layers, losses, optimizers, and a
  converging end-to-end training loop, all parity- and grad-checked.

### T17 — optimizers SGD + Adam (2026-07-05)
- `nn/optim.go`: `SGD` (optional classical momentum, torch formulation
  v←μv+g, p←p−lr·v) and `Adam` (Kingma & Ba 2015: bias correction, eps outside
  the sqrt). Optimizer state kept in **f64 regardless of param dtype** (§V10
  master-state analog). `Step(grad GradFn)` — `tape.Grad` fits directly as a
  method value, keeping nn decoupled from autograd.
- Golden: `gen.py` emits 3-step f64 reference trajectories
  (`nn/testdata/optim.json`); provenance documented in §B20 (torch not in venv
  → exact reimplementation of the documented update rules).
- **Verification (§V1):** step-by-step trajectory parity at f64 rtol 1e-12 for
  SGD, SGD+momentum, Adam; end-to-end Linear+MSE training strictly decreases
  loss under both optimizers (T18 preview); nil-grad skip + shape-mismatch
  error. All packages green, cross-compiled (§V4/§V7). `go vet` clean.

### T16 — activations, Sequential, losses (2026-07-05)
- **ADR-0007**: CrossEntropy as a fused op — stable per-row max-shift
  logsumexp, VJP = (softmax−onehot)/b, targets non-differentiable. Ref-only
  kernel; cpu falls back via §I4 (fallback path exercised in production).
- `nn`: `Activation` layers (ReLU/Tanh/GELU/Sigmoid) + `Sequential`;
  `MSE` (pure composition — gradient entirely from existing VJPs) and
  `CrossEntropy` (`OpCrossEntropy`, `ops` not needed: nn wraps Execute).
- Golden: `gen.py` emits a `losses` section (stable logsumexp reference) +
  `nn/testdata/losses.json`.
- **Verification:** §V1 golden parity (MSE, CE at f64 rtol 1e-12); §V12
  stability — CE finite and shift-invariant under +1000 logit offset; §V2 CE
  gradient vs finite differences + targets nil-grad + MSE analytic 2(p−t)/n;
  error cases (out-of-range target, batch mismatch); Sequential end-to-end
  (Linear→ReLU→Linear, 4 params). All packages green (§V4/§V7). `go vet` clean.

### T15 — L3 Linear layer + initializers (2026-07-05)
- `nn`: `Layer` interface (Forward through a context → same code paths for
  eager and taped execution), `Linear` (y = x·W + b, W stored [in,out] — no
  transpose in forward; torch-layout divergence documented §B19), `Params()`
  for optimizers.
- `nn/init.go`: `XavierUniform` (Glorot & Bengio 2010), `KaimingNormal` /
  `KaimingUniform` (He et al. 2015), `Zeros` — all deterministic via explicit
  seed (§V13).
- New op `OpAddBias` ([m,n]+[n] row-broadcast): ref + cpu kernels (bit-identical)
  + VJP (gx=g, gb=column-sums in f64 §V10) + `ops.AddBias`. General
  broadcasting deferred (§B18).
- **Verification:** forward parity vs explicit loops; §V3 cpu/ref addbias
  bit-identical; **§V2 gradcheck through the layer** for x, W, and b (rel 1e-5);
  init statistics (bounds, mean, std ±15%) + same-seed determinism; shape-error
  cases. All packages green, cross-compiled (§V4/§V7). `go vet` clean.

### T14 — VJP rules for all L1 ops + systematic grad checks (2026-07-05)
- `autograd/vjp_elementwise.go`: div, exp, log, tanh, relu, gelu (exact erf
  derivative Φ(x)+x·φ(x)), sigmoid — scalar-loop VJPs reusing forward outputs.
- `autograd/vjp_linalg.go`: matmul (gA=g·Bᵀ, gB=Aᵀ·g via transpose views on the
  optimized GEMM), dot, nrm2 (subgradient 0 at 0), axpy (α from attrs).
- `autograd/vjp_reduce.go`: sum/mean broadcast VJPs honoring axes/keepdims;
  max/min route grad to the first extremum (tie → lowest index, §B16); argmax →
  nil (non-differentiable).
- **Verification (§V2):** systematic gradcheck harness — 18 op/axes cases, every
  element of every input checked against central finite differences at rel 1e-5
  (spec bound 1e-4); Sum VJP exercised in every composite loss. Argmax backward
  is a clean no-op. **backprop B17**: stale negative test (asserted exp had no
  VJP) → negative tests now use synthetic op codes. All packages green, all
  platforms (§V4/§V7). `go vet` clean.
- **Milestone: L2 autograd complete** — every L1 op differentiable and verified.

### T13 — L2 autograd engine (2026-07-05)
- **ADR-0006**: tape-based reverse-mode AD. `Tape` implements
  `backend.Recorder` — the §T5 seam captures every forward op with **zero L1
  changes**. Nodes in execution order → Backward is one reverse walk.
- `autograd`: `Tape` (`Context()`, `Backward(out)` seeded with ones,
  `Grad(x)`), gradient **accumulation at fan-out**, `Variable` wrapper, VJP
  registry (`RegisterVJP`) mirroring the kernel registry. Backward runs in a
  non-recording context (tape never grows). Engine-proof VJPs: add/sub/mul/neg
  — full table + systematic grad checks land with §T14.
- **Verification:** analytic gradients bit-exact (mul, fan-out y+1, chain
  2xy²); §V2 preview — central finite differences match analytic grads
  (rel < 1e-7); Backward-not-recorded; missing-VJP errors cleanly; unrelated
  tensors get nil grad. All platforms green (§V4/§V7). `go vet` clean.
- §B15: view↔base gradient flow deferred (pointer-identity design).

### T12 — optimized cpu GEMM (2026-07-05)
- `backend/cpu/gemm.go`: raw-slice **ikj**-order matmul. ikj preserves the
  reference's per-element k accumulation order → **bit-identical to ref** (§V3,
  §V11 tol 0), while giving row-wise B/C access (vectorizable SAXPY inner loop)
  and clean parallelism over disjoint C row-bands. f32 accumulates in an f64
  scratch matrix (§V10).
- `backend/cpu`: `parallelWork(n, workPerItem, …)` — parallelize by estimated
  total work (GEMM parallelizes at M·K·N ≥ threshold; elementwise unchanged).
- **Verification:** §V3/§V11 — cpu GEMM bit-identical to ref across square/
  non-square/large-M shapes, both dtypes, and transposed-view operands. §V5:
  **MatMulF64 128³ 9.1ms→294µs (~31×), 0.46→14.3 GFLOP/s; 256³ 27.9 GFLOP/s.**
  All platforms + `GOEXPERIMENT=simd` amd64 build green (§V4/§V7). `go vet` clean.
- §T12b tracks cache-blocking + archsimd FMA microkernel for higher GFLOP/s.

### T11 — optimized cpu backend (elementwise) (2026-07-05)
- **ADR-0005**: optimized kernels live in a separate `backend/cpu` (preferred
  Default); `ref` stays the truth (§V9). archsimd amd64 intrinsics split to §T11b
  (not host-verifiable on arm64, §B13).
- `backend/cpu`: elementwise kernels — contiguous typed-slice loops (no
  per-element alloc), goroutine parallelism above 32K elements, binary ops via
  `internal/simd` primitives (auto-vectorizable; amd64 archsimd override in T11b).
- `internal/simd`: portable Add/Sub/Mul/Div f32/f64 primitives.
- `backend`: `RegisterDefault` + preferred-Default selection; root imports cpu.
- **Verification:** §V3/§V11 CROSS — cpu is **bit-identical to ref** (tolerance 0)
  across all elementwise ops, both dtypes, incl. the parallel path (256K) and
  non-contiguous inputs. §V5 delta: **AddF64 4K 127µs/4104 allocs → 12µs/9 allocs
  (~10×, 456× fewer allocs)**. All platforms + `GOEXPERIMENT=simd` amd64 build
  green (§V4/§V7). `go vet` clean.

### T10 — golden/npy tooling + benchmark harness (2026-07-05)
- `internal/npy`: Pure-Go NumPy `.npy` reader + writer (v1/2 headers, `<f4`/`<f8`,
  C-order; fortran_order/other dtypes rejected). For exchanging large golden
  tensors with the Python reference.
- `internal/bench`: deterministic seeded data generators (`RandF64`/`RandF32`).
- `backend/ref/bench_test.go`: baseline benchmarks (Add, Sum, MatMul f32/f64) —
  the numbers §T11/§T12 must beat (§V5). `docs/benchmarking.md` documents the
  regression + cgo-gate policy.
- `testdata/gen.py` also emits real `.npy` samples for the reader tests.
- **Verification:** npy reads real numpy files + Go round-trip (incl. scalar,
  non-contiguous, bad-magic); bench harness runs. Baselines recorded: AddF64 4K
  ~100µs/4.1k allocs, MatMulF64 128³ ~9.4ms — targets for §T11/§T12.
  Cross-compiled win/darwin (§V4). `go vet` clean.
- **§T10 complete** (JSON goldens landed early with §T6).

### T9 — L1 GEMM reference kernel (2026-07-05)
- `backend/ref/gemm.go`: scalar row-major `matmul` C[M,N]=A[M,K]·B[K,N], inner
  product accumulated in float64 (§V10). Reads via AtF64 → transposed/sliced
  views work with no trans flags. The truth for the future SIMD GEMM (§T12).
- `ops`: `MatMul`.
- Golden: `testdata/gen.py` gains a `gemm` section (2×3·3×4, f64+f32).
- **Verification (§V1, §V9, §V10):** parity vs golden (f64 rtol 1e-12, f32 1e-5);
  A·I identity (§V6); transposed-view operand; f64-accum accurate at K=1e5; errors
  on inner-dim/rank mismatch; K=0 → zero matrix. Cross-compiled win/darwin/linux
  (§V4). `go vet` clean.
- **Milestone: L1 reference (§T5–§T9) complete** — the Pure-Go numeric truth for
  elementwise, reductions, BLAS-1, and GEMM. Optimization (§T11/§T12) can begin.

### T8 — L1 BLAS-1 reference kernels (2026-07-05)
- `backend/ref/blas1.go`: `dot` (full inner product), `nrm2` (Euclidean norm),
  `axpy` (alpha*x+y). dot/nrm2 accumulate in float64 (§V10); nrm2 uses the scaled
  LAPACK dnrm2 algorithm — overflow/underflow-safe (§G4, §V9).
- `backend`: `Attrs.Float` accessor (axpy alpha).
- `ops`: `Dot`/`Nrm2`/`Axpy`.
- Golden: `testdata/gen.py` gains a `blas` section (f32 = f64-accumulate-narrow).
- **Verification (§V1, §V9, §V10):** dot/nrm2/axpy match golden (f64 rtol 1e-12,
  f32 rtol 1e-5). nrm2 overflow-safe: `‖[1e200,1e200]‖ = 1e200·√2` finite (naive
  would be +Inf); underflow-safe for 1e-200. dot f64-accum stays accurate at
  K=1e5. Edge: empty dot/nrm2→0, shape-mismatch error, default alpha=1.
  Cross-compiled win/darwin (§V4). `go vet` clean.

### T7 — L1 reduction reference kernels (2026-07-05)
- `backend/ref/reduce.go`: axis-aware sum/mean/max/min (multi-axis via `axes`
  attr, `keepdims`) + argmax (single-axis or flat). **§V10 ACCUM**: all
  reductions accumulate in float64, narrowing only the final result.
- `ops`: `Sum`/`Mean`/`Max`/`Min`/`ArgMax`/`ArgMaxFlat`. argmax returns a float
  index for now (no int dtype until §C4, §B12).
- Golden tooling: `testdata/gen.py` extended with a `reduce` section; f32 goldens
  use the f64-accumulate-then-narrow semantics (matches §V10, not numpy-native).
- **Verification (§V1, §V10):** parity vs golden over axis0/axis1/all for
  sum/mean/max/min (f64 rtol 1e-12, f32 rtol 1e-5) + argmax indices, with output
  shapes checked. §V10 demonstrated: naive f32 sum of 1e5×0.1 drifts by 1.44
  while the f64-accum kernel stays exact. Edge: keepdims shape, empty-reduction
  identities (sum→0, mean→NaN). Cross-compiled win/darwin (§V4). `go vet` clean.

### T6 — L1 elementwise reference kernels + first golden parity (2026-07-05)
- **ADR-0004**: GELU uses the exact erf definition (PyTorch default, §V9 truth).
- `backend/ref/elementwise.go`: dtype-agnostic scalar kernels for add/sub/mul/div,
  neg/exp/log/tanh/relu/gelu/sigmoid (one impl serves F32+F64; f32 narrows on
  store, ADR-0001). Stable sigmoid; arbitrary-layout inputs.
- `ops` package: public eager API (`Add`…`Sigmoid`) over the dispatch path.
- Golden tooling (partial §T10): `testdata/gen.py` emits env-pinned JSON goldens
  from NumPy + Python libm — an oracle independent of Go's pure-Go math.
  `make golden` uses a local `.venv`.
- **Verification (§V1 PARITY):** Go kernels match golden within f64 rtol 1e-12 /
  f32 rtol 1e-5 for all ops. Edge cases (§V12): div-by-zero ±Inf/NaN, log domain,
  exp overflow, sigmoid saturation, empty tensor, non-contiguous input; binary
  shape-mismatch errors. Cross-compiled win/darwin/linux (§V4). `go vet` clean.

### T5 — L1 Backend/Kernel interface + reference backend (2026-07-05)
- **ADR-0003**: opcode dispatch through a single `Execute` choke-point + a
  `Recorder` hook — closes review BLOCK B7 (§V14 sync model) and the T5↔T13
  autograd-interception gate.
- `backend` (L1): `Op` opcodes + `Attrs`; `Backend` interface (`Kernel` lookup,
  `Synchronize` sync model §V14); `Kernel` signature; `Recorder` interception
  seam (§T13); `Context` (backend + recorder); registry
  (`Register`/`RegisterReference`/`Get`/`Available`/`Default`, §I4 feature
  detection); `Execute` — resolves kernel, **falls back to reference** when the
  active backend lacks it (§I4), then records (§T13).
- `backend/ref`: Pure-Go reference backend = source of truth (§V9); self-registers
  as reference; synchronous (`Synchronize` no-op §V14); kernel table filled by
  §T6–§T9. `register.go` blank-imports it so it is always available.
- **Verification:** all packages green — registry/Default/Available, Execute happy
  path, **fallback to reference** (§I4), **Recorder sees every forward op** (§T13
  gate), unserved-op errors (not panics), Attrs accessors, ref self-registration +
  Synchronize no-op. Cross-compiled win/darwin/linux (§V4). `go vet` clean.

### T4 — L0 Device + Allocator (2026-07-05)
- **ADR-0002**: allocator interface; alignment advisory in L0 (guaranteed
  over-alignment deferred to §T11, §B11) — keeps L0 unsafe-free (ADR-0001).
- `Device` (`tensor/device.go`): `DeviceKind` (CPU + accel placeholders), `CPU()`
  default, `NewCPUDevice(alloc)` for pooling. Tensors carry a device; views
  inherit it.
- `Allocator` (`tensor/allocator.go`): `Heap` (make-based) + `Pool` (sync-pooled
  by power-of-two size class, concurrency-safe, zeroed on Alloc, advisory
  `WithAlignment`).
- `Storage.Release()` returns the buffer to its allocator; `NewOn(dev,…)`
  allocates through the device allocator.
- **Verification:** 33 tensor tests green (incl. `-race`) — pool reuse recycles
  the same backing array and re-zeroes (§V6), size classes, empty/foreign-free
  no-ops (§V12), device inheritance, Release round-trip. Cross-compiled
  windows/amd64 + darwin/arm64 (§V4). `go vet` clean.

### T3 — L0 Tensor + zero-copy views (2026-07-05)
- **ADR-0001**: type-erased Storage with a runtime Dtype (F32→[]float32,
  F64→[]float64 behind `any`) — matches ATen/ggml, supports runtime-dtype model
  loading and one Backend interface over all dtypes.
- `Storage` (`tensor/storage.go`): allocation, typed accessors `F32`/`F64`,
  dtype-agnostic `atF64`/`setF64` reference accessors.
- `Tensor` (`tensor/tensor.go`): `New`/`FromFloat64`/`FromFloat32`, shape/stride/
  offset accessors, `AtF64`/`SetF64`, `Contiguous` (explicit copy).
- Views (`tensor/view.go`): `Reshape` (contiguous, zero-copy), `Slice`,
  `Transpose`, `Permute` — all pure stride/offset ops sharing storage (§C5).
- **Verification:** tensor suite green — zero-copy proven (write via
  transpose/slice/reshape visible in parent, §V6/§C5); transpose/permute index
  mapping checked; edge cases scalar/len-mismatch/bounds/dtype-guard (§V12).
  Cross-compiled windows/amd64 + linux/arm64 (§V4). `go vet` clean.

### T2 — L0 dtype / shape / strides (2026-07-05)
- `Dtype` (`tensor/dtype.go`): F32/F64 with `Size`/`String`/`IsValid`/`IsFloat`;
  zero value `Invalid`, out-of-range safe. Extensible for f16/bf16/int later (§C4).
- `Shape` (`tensor/shape.go`): `Ndim`/`Numel`/`Equal`/`Clone`/`String` +
  `IsScalar`/`IsEmpty`/`IsValid`. Scalar (`()`) numel 1; zero-dim → empty (§V12).
- `Strides` (`tensor/strides.go`): `RowMajorStrides`, `Offset`, `Unravel`,
  `IsContiguous` — row-major, element-based, view-ready (§C5, prep §T3).
- **Verification:** 13 tests green incl. property tests (§V6) — Offset↔Unravel
  round-trip and contiguous-offset bijection over 500+300 random shapes; edge
  cases (scalar, zero-dim, rank-mismatch panic) covered (§V12). `go vet` clean.

### T1 — repo scaffold (2026-07-05)
- `go.mod` at Go 1.26, module `github.com/jxsl13/goai`.
- Layer packages per §I: `tensor` (L0), `backend` + `backend/ref` (L1),
  `autograd` (L2), `nn` (L3), `format` (L5), `internal/simd` (SIMD wrapper, §B3).
- `Makefile` with `CGO_ENABLED=0` default targets (build/vet/test/bench/simd-build/golden).
- CI skeleton `.github/workflows/ci.yml`: pure-go matrix on ubuntu/macos/windows
  (CGO_ENABLED=0) + soft `GOEXPERIMENT=simd` job.
- **Verification:** `CGO_ENABLED=0 go build/vet/test ./...` green; cross-compiled
  green for darwin/{arm64,amd64}, linux/{amd64,arm64}, windows/amd64 (§V4, §V7).
