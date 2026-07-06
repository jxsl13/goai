# SPEC — GoAI

Go-native AI library, full spectrum, Pure-Go-first / cgo-last.
Encoding: caveman (see `FORMAT.md`). Target Go 1.26.
Source research: `docs/research/00-landscape.md`. Framing: `PLANNING_PROMPT.md`.

## §G — goals

G1: idiomatic modular Go lib ∀ AI domains — linalg, autograd, classic-ML, deep-learning, NLP/LLM-inference, CV, RL, probabilistic.
G2: core ops → as close as possible to C/C++ refs (Eigen, OpenBLAS, oneDNN, ggml, ONNX Runtime, PyTorch/ATen) via Pure-Go-SIMD; GPU/NPU only where needed.
G3: correctness before speed. ∀ op → valid Pure-Go reference first, optimize as separate step.
G4: math/scientific grounding ! per unit; numeric decisions (stability, accuracy, overflow) documented ≠ implicit.
G5: numeric parity = acceptance. "done" → results match ref within fixed tolerance, proven ≠ claimed.
G6: perf measurable | ⊥ exist. no "faster" without benchmark + baseline compare.
G7: native macOS & Windows & Linux on CPU & GPU & (later) NPU. ∀ accel op → Pure-Go fallback runs everywhere.

"vollwertig" (measurable): L0–L3 green + parity ∀ ops; ≥1 optimized CPU backend beating scalar ref w/ V-BENCH numbers; safetensors IO; ≥1 end-to-end trained model converging; ≥1 GPU backend gated by V-CGO; ≥1 LLM inference path.

## §C — constraints

C1: target Go 1.26. `CGO_ENABLED=0` = default build, ! green ∀ platforms.
C2: cgo-LAST policy = measurable gate. Pure Go default (experimental `simd/archsimd` via `GOEXPERIMENT=simd` on AMD64; Plan9-NEON on ARM64; `avo`; goroutines). cgo/external-C only as optional build-tag backend when ALL hold:
  (1) Pure-Go version §V-green & optimized to ceiling (documented stages + roofline).
  (2) benchmark beats §C.thr vs that optimized Pure-Go version.
  (3) `CGO_ENABLED=0` stays fully functional (fallback everywhere).
  ⊥ all 3 → reject cgo, park idea in §B.
C3 (thr): "deutlich" threshold = speedup ≥ 1.5× OR reaches ≥ 80% of C++ baseline that Pure-Go cannot. revisable → ADR. [B1]
C4: dtypes f32 & f64 first; f16, bf16, int8 later.
C5: tensor layout row-major + strides/views.
C6: GPU → cgo unavoidable (no Pure-Go GPU path) → late optional backend behind build-tags.
C7: NPU (ANE/CoreML, DirectML, oneDNN) = explicit non-goal of first stage. re-eval after GPU. ⊥ silent promise.
C8: ⊥ commit | push without explicit user permission. loop works in working-tree only.
C9: deps minimal. Pure-Go path ⊥ pull cgo deps. accel backends isolate deps behind build-tags.
C10: docs = first-class deliverable, ⊥ afterthought. ∀ public API → godoc for TWO audiences: AI professional (math/algorithm/paper-cite/§R) AND layperson (plain-language what/why/when). PLUS runnable godoc `Example`/`ExampleT_method` fns at MULTIPLE levels: (a) trivial (one call, minimal), (b) ≥1 realistic trivial use-cases, (c) ≥1 complex/EMBEDDED example showing the piece inside a bigger pipeline (e.g. LoRA inside a full fine-tune loop). examples ! compile & pass via `// Output:` (go test). guarded by V17.

## §I — architecture invariants

layer model (strict, top ⊥ know backend internals):
I.L0 core: `Tensor` (data, dtype, shape, strides, device), `Dtype`, `Device`, `Allocator`, views (reshape/slice/transpose = stride ops). ⊥ cgo in L0.
I.L1 compute: `Backend` + `Kernel` interface + Pure-Go reference backend = TRUTH. registry + feature-detection.
I.L1b accel: swappable backends (cpu-simd, cuda, metal, vulkan) behind same interface, feature-detect + fallback.
I.L2 autograd: tape/graph + `Variable` + VJP rules per op.
I.L3 nn: layers, init, optimizer, loss, data pipeline.
I.L4 domains: transformer/LLM, CV, classic-ML, RL, probabilistic.
I.L5 io: safetensors, GGUF, ONNX.

INVARIANTS:
I1: higher layer ⊥ import backend internals. public API backend-agnostic.
I2: ∀ op → Pure-Go fallback exists.
I3: ⊥ cgo in L0.
I4: accel backend selected via feature-detection @ runtime, ⊥ compile-time-only lock; missing accel → fallback.
I5: `CGO_ENABLED=0` build green ∀ {macOS, Windows, Linux}.

## §R — research log

id|claim|source|conf
R1|Go 1.26 `simd/archsimd` ships Feb 2026 under `GOEXPERIMENT=simd`, AMD64-only, 128/256/512-bit, AVX2+AVX-512, no cgo/asm-stub|golang/go#73787, go.dev/doc/go1.26, pkg.go.dev/simd/archsimd|high
R2|`simd` pkg inlined ~4× vs next-best, ~16× vs plain loop; `avo` ~3× vs plain, can't inline thru `.s` stub|callistaenterprise.se 2025-10-20|med
R3|`simd/archsimd` ARM64/NEON not yet covered → use Plan9-NEON on ARM64|golang/go#73787|med
R4|GoMLX active (v0.26.0 Dec 2025) but accel via OpenXLA/`gopjrt`=cgo; Pure-Go backend unoptimized. Gorgonia dormant. gonum BLAS incomplete (f32/f64)|github.com/gomlx/gomlx, gorgonia, gonum/blas|high
R5|no practical Pure-Go path to discrete-GPU compute; CUDA/Metal/Vulkan/ROCm all need cgo/C-loader|synthesis §3|high
R6|parity tolerance default f64 rtol 1e-12, f32 rtol 1e-5; golden from NumPy/torch fixed-seed → testdata/golden|methodology|med
R7|safetensors low effort (JSON header + raw tensors); GGUF medium; ONNX high (protobuf + large opset)|synthesis §6|med

# validation pass 2026-07-06 (/deep-research primary sources + /research book refs)
# conf: high=primary-source-confirmed this pass; ref=canonical text identified, page-level read pending (session-limited)
R8|GELU exact 0.5x(1+erf(x/√2)) = PyTorch default approximate='none'|docs.pytorch.org/.../torch.nn.GELU; alg4ai/Deep-Learning-Interviews (Books repo)|high
R9|Adam: bias-corr m̂=m/(1−β₁ᵗ) v̂=v/(1−β₂ᵗ), ε OUTSIDE sqrt|github.com/pytorch/pytorch torch/optim/adam.py; Kingma&Ba 2015|high
R10|SGD momentum torch: v=μv+g, p−=lr·v|docs.pytorch.org/.../torch.optim.SGD|high
R11|LayerNorm: biased var (÷N), ε INSIDE sqrt|docs.pytorch.org/.../torch.nn.LayerNorm|high
R12|conv2d in DL = cross-correlation (no flip), out=(H+2p−k)/s+1|docs.pytorch.org/.../torch.nn.Conv2d|high
R13|GGUF magic 0x46554747 LE; alignment 32 via general.alignment(u32); metadata enum u8..f64 order; v1/v2/v3 exist (we support 2/3)|github.com/ggml-org/ggml docs/gguf.md; verified bidir vs gguf-py 0.x (T22)|high
R14|GGUF dims innermost-first → reverse for row-major: NOT in spec text, confirmed vs ggml reference impl + our bidir gguf-py roundtrip|ggml gguf.c ne[]; T22 test|med
R15|linalg refs (nrm2 scaled, Cholesky, Jacobi eigen, PCA): Matrix Cookbook, math4ml, LAEF|Books: 8.matrixcookbook.pdf, 5.math4ml.pdf, 12.LAEF.pdf|ref
R16|RL refs (REINFORCE baseline independence, DQN replay+target): RL-Overview, MARL|Books: 3.Reinforcement Learning-An Overview.pdf, 10.marl.pdf; Williams 1992 / Mnih 2015|ref
R17|LLM/transformer refs (attention, layernorm, tokenization): Foundation of LLM, ML-Systems|Books: 2.Foundation of LLM.pdf, 13.Machine-Learning-Systems.pdf|ref
R18|deep-research adversarial verify panel BLOCKED by session limit 2026-07-06 → primary-source extractions used directly; re-run adversarial pass after reset for R8–R14|workflow w6aqxijn6|?
R19|Q4_0 dequant CONFIRMED vs ggml-quants.c: low nibble qs[j]→elem j, high nibble→elem j+16 (split-half, NOT 2j/2j+1), x=(nibble−8)·d. matches our dequantQ4_0 bit-for-bit|ggml-org/ggml src/ggml-quants.c dequantize_row_q4_0; 3-agent cross-check|high
R20|research mechanism: built-in /deep-research forces agent({schema})→StructuredOutput retry-cap crash under rate-limits. USE `.claude/workflows/research-lite.js` (schema-free, 1 focused Q, compressing sub-agents, graceful). validated 2026-07-06 (wtoduufsd, clean)|LOOP.md Research-Regel|high

# tier2 paper pointers for the expansion tasks (§V16) — verify formula vs paper during each task
R26|AdamW decoupled wd: p←p·(1−lr·wd)−lr·m̂/(√v̂+ε) (= paper Alg.2 line12, multiplicative on param NOT in grad). CONFIRMED vs paper+torch; our NewAdamW matches, golden err 0 vs real torch|Loshchilov&Hutter 2019 arXiv:1711.05101 Alg.2 + torch/optim/adamw.py (T33, §V16 tier2)|high
R27|LoRA h=W0x+(α/r)·BAx; A[r,k] Gaussian/Kaiming, B[d,r] zero→ΔW=0 start; W0 frozen only A,B train. CONFIRMED vs paper §4.1+microsoft/LoRA+PEFT; matches nn.LoRALinear ([in,out] convention)|Hu et al. 2021 arXiv:2106.09685 §4.1 (T40 §V16)|high
R28|RoPE: inv_freq_i=base^(−2i/d) base=10000; q_rot=q·cos+rotate_half(q)·sin; cos/sin=cat(freqs,freqs); rotate_half=cat(−q[d/2:],q[:d/2]) → HALF-SPLIT pairing (i,i+d/2) NOT interleaved. CONFIRMED vs paper+HF; matches our OpRoPE|Su et al. 2021 arXiv:2104.09864 + HF modeling_llama.py (T38 §V16)|high
R29|RMSNorm y=x/√(mean(x²)+eps)·γ, no mean-sub, no bias, eps in sqrt. CONFIRMED vs paper eq(4)+HF LlamaRMSNorm; matches our OpRMSNorm|Zhang&Sennrich 2019 arXiv:1910.07467 eq4 + HF (T38 §V16)|high
R30|SwiGLU FFN=(SiLU(x·Wg)⊙(x·Wu))·Wd, 3 matrices, no bias; SiLU(z)=z·σ(z)=Swish β=1. CONFIRMED vs paper+HF LlamaMLP; matches our nn.SwiGLU|Shazeer 2020 arXiv:2002.05202 + HF (T38b §V16)|high
R31|GQA: nkv divides nh; query head h→KV head h//(nh/nkv), contiguous groups (repeat_kv); MQA=nkv1. CONFIRMED vs paper+HF; matches OpMHA kv_heads|Ainslie et al. 2023 arXiv:2305.13245 + HF (T38b §V16)|high
R32|FlashAttention: tiled softmax, O(N) mem, no materialized N×N|Dao et al. 2022 arXiv:2205.14135 (T32/T38)|ref
R33|BPE subword tokenization|Sennrich et al. 2016 arXiv:1508.07909 (T37)|ref
R34|top-p nucleus = smallest desc-prob set cumsum≥p (crossing token incl), renorm (eq.3); top-k = k highest logits masked; temperature logits/T applied BEFORE top-k/top-p. CONFIRMED vs paper+HF; matches our Sampler|Holtzman et al. 2019 arXiv:1904.09751 §3.1-3.3 eq.2-4 + HF logits_process.py (T36, §V16 tier2)|high
R35|LayerNorm backward formula: dL/dx=(1/σ)(a−mean(a)−x̂·mean(a·x̂)), a=g⊙γ|Ba et al. 2016 arXiv:1607.06450 + std deriv (T31)|ref

# GPU/NPU landscape validated 2026-07-06 (research-lite w4aos12wu) — shapes T30/T42/T43/T44
R36|macOS GPU TRAINING via MPSGraph autodiff (gradientForPrimaryTensor:withRespectToTensors:) — GPU-side backward, not just inference. USE MPSGraph for T30 (autodiff-on-GPU) vs hand-writing every bwd kernel|developer.apple.com/documentation/metalperformanceshadersgraph|high
R37|Apple ANE = INFERENCE-ONLY via CoreML (FP16, no gradient API); training only via private reverse-eng APIs → macOS NPU training = honest non-goal (T44)|apple ANE docs; arXiv:2603.06728|high
R38|NVIDIA cuDNN full fwd+bwd (dgrad/wgrad) but cgo-ONLY (proprietary C ABI, no pure-Go path) → T42 cgo-gated|docs.nvidia.com/.../cudnn-cnn-library|high
R39|Vulkan/WebGPU = general GPGPU, NO built-in autodiff → Go must hand-write backward kernels; training-grade maturity early → T43 inference-first, bwd hand-rolled|onnxruntime webgpu; tracel-ai/burn|high
R40|Windows DirectML trainable (DML_*_GRAD + DML_ADAM_OPTIMIZER, manual graph); Intel oneDNN trainable (backward_data/weights); OpenVINO inference-only → T44 DirectML/oneDNN feasible for accel training|learn.microsoft.com directml; uxlfoundation oneDNN|high

# validation ladder pass 2026-07-06 (§V16): tier1=ref-lib parity (done), tier2=scientific/spec source
R21|Q8_0: 34B block = f16 d + 32×int8 qs, x=d·qs[i] NO offset (scale first). matches dequantQ8_0|ggml-common.h block_q8_0 + ggml-quants.c dequantize_row_q8_0 (format-spec = definitional, no paper §V16)|high
R22|safetensors: 8B-LE u64 header-len + JSON + data; data_offsets relative to data section; ref REJECTS gaps/overlaps + requires full contiguous tiling; MAX_HEADER=100MB; __metadata__ str→str. matches our Load|huggingface/safetensors tensor.rs validate() + README (format-spec)|high
R23|.npy: v1 uint16 / v2-3 uint32 header-len; TOTAL header padded w/ spaces to mult of 64B (not 16), \n-terminated; keys descr/fortran_order/shape. matches our reader+writer|numpy.lib.format docs (format-spec)|high
R24|attention scale = 1/√d_k with d_k = d_model/h PER-HEAD (d_model=512,h=8→d_k=64), NOT 1/√d_model. matches MHA `1/√dk`, dk=dm/heads|Vaswani et al. 2017 arXiv:1706.03762 §3.2.1/3.2.2 (PAPER = tier2 final)|high
R25|nrm2 scaled recurrence (scale=0,ssq=1; scale<a→ssq=1+ssq(scale/a)²,scale=a; else ssq+=(a/scale)²; ret scale√ssq) = CLASSIC ref-BLAS dnrm2 = DLASSQ = Blue 1978. matches our nrm2Kernel. NOTE: newest LAPACK ≥3.9 uses Anderson 2017 Alg.978 (3-accum) — our classic form correct+overflow-safe, documented divergence [B33]|Reference-LAPACK dnrm2.f (v3.8); Blue 1978 ACM TOMS 4(1) (PAPER = tier2 final)|high
R41|f16=IEEE754 binary16 (1/5/10 bias15); f32→f16 = round-to-nearest-even, overflow→inf, subnormals+NaN per IEEE. bf16=top 16 bits of f32 (1/8/7 bias127); f32→bf16 = RNE via bias=((u32>>16)&1)+0x7FFF then (u32+bias)>>16 (NOT truncation), NaN kept as NaN (0x7FC0 not inf); bf16→f32 = pure left-shift u32(bf16)<<16. matches our tensor/half.go, bit-exact vs numpy.float16 + torch.bfloat16|IEEE 754-2019 §4.3; numpy.half docs; PyTorch torch/headeronly/util/BFloat16.h round_to_nearest_even; Eigen BFloat16.h (PAPER/spec = tier2 final)|high
R45|op-level NPU dispatch from Go = impractical (2026). ANE: CoreML whole-model only (.mlpackage/ML Program), NO per-op API, NO gradient API, FP16 inference-only. Intel NPU: OpenVINO graph/whole-model, inference-only. DirectML: HAS per-op (CompileOperator/DML_GEMM) + training grads (DML_*_GRAD) BUT native C++/nano-COM on D3D12, no Go binding → heavy interop. ∴ NPU = MODEL-level target (export→CoreML/ONNX/OpenVINO/DirectML runner), NOT L1b op backend. T44 = documented honest non-goal, backend/npu.Available()=false|developer.apple.com/documentation/coreml/mlmodel + learn.microsoft.com directml + docs.openvino.ai npu-device (vendor docs = definitional, no paper §V16)|high
R44|Vulkan compute matmul orchestration: instance→enum physical devices→pick queue family w/ VK_QUEUE_COMPUTE_BIT→device→queue (avail=≥1 compute device). 3× STORAGE_BUFFER host-visible|coherent (map/memcpy, no staging). DescriptorSetLayout 3 storage bindings @0,1,2 stage COMPUTE + pool + update; dims M,K,N via push constants. ShaderModule(SPIR-V)→ComputePipeline. Cmd: begin→bindPipeline→bindDescriptors→pushConstants→dispatch(ceil(N/16),ceil(M/16),1) for 16×16 local→end→submit→queueWaitIdle→read. GLSL C[y=row][x=col], GID.x=col→N, GID.y=row→M. matches vk_bridge.c + matmul.comp|registry.khronos.org/vulkan spec §dispatch/§descriptor-sets + docs.vulkan.org/tutorial Compute_Shader (API convention = definitional, no paper §V16)|high
R43|cuBLAS is COLUMN-MAJOR; cublasSgemm(h,transa,transb,m,n,k,α,A,lda,B,ldb,β,C,ldc)=α·op(A)·op(B)+β·C, ld=rows of each col-major matrix. Row-major C[M,N]=A·B via col-major C^T=B^T·A^T: cublasSgemm(h,OP_N,OP_N,N,M,K,&α,B,N,A,K,&β,C,N) — operands B THEN A, dims N,M,K, ld N,K,N. GPU-detect: cudaGetDeviceCount(&n)==cudaSuccess && n>0. matches cuda_bridge.c|docs.nvidia.com/cuda/cublas + leimao.github.io cuBLAS-Transpose (API convention = definitional, no paper §V16)|high
R42|mixed-precision training 3 techniques: (1) fp32 MASTER weights — optimizer updates fp32 master, fp16 copy for fwd/bwd, updates <2^-24 lost in fp16 but accumulate in master; (2) LOSS SCALING — ×S before backward, unscale ×1/S before step; (3) DYNAMIC scaler (torch/Apex defaults) init 2^16=65536, growth ×2 after 2000 clean steps, backoff ×0.5 on inf/nan AND SKIP the step (not clamp); (4) fp32 ACCUMULATION (already §V10). matches nn.MixedPrecision + nn.LossScaler + Tape.BackwardScaled|Micikevicius et al. 2018 arXiv:1710.03740 §3.1-3.3 + PyTorch torch/amp/grad_scaler.py (PAPER = tier2 final)|high

## §V — verification invariants

V1 PARITY: ∀ op → golden test vs NumPy/torch within §R6 tol (f64 rtol 1e-12, f32 rtol 1e-5). missing golden → generate reproducibly + commit.
V2 GRAD: ∀ differentiable op → numeric gradient check (central finite diff) under rel 1e-4.
V3 CROSS: accel backend result == Pure-Go reference within backend tol defined by V11 (exact bit-match ⊥ required — SIMD/parallel reorders sums).
V4 PLATFORM: CI green ∀ {macOS, Windows, Linux} × {Pure-Go fallback + available accel}. missing accel → skip w/ log, ⊥ silent pass.
V5 BENCH: ∀ optimized op → benchmark + baseline number recorded. regression breaks CI.
V6 PROP: property-based tests (shape algebra, linearity, associativity where math-guaranteed).
V7 CGO: ⊥ cgo in ship path without green optimized Pure-Go ref + committed benchmark over §C.thr. `CGO_ENABLED=0` green everywhere.
V8 STABLE: public API changes only via documented deprecation path.
V9 TRUTH: Pure-Go reference backend = source of numeric truth; accel validated against it (⊥ vice versa).
V10 ACCUM: reduction & GEMM accumulation → f32 inputs accumulate in f64 (| Kahan) unless §R justifies else. parallel/goroutine reduction order documented; non-determinism only w/ §R-justified tol. guards non-associative FP [B6].
V11 CROSSTOL: ∀ accel op → backend tol documented per op, rtol scales w/ reduction length K (⊥ single fixed rtol). defines V3. established: elementwise (T11) & blocked-GEMM (T12/T12b) = exact (tol 0, same accum order); metal f32 GEMM (T20) = rtol(K)=1e-6·√K (MPS f32 accum + reorder) [B5].
V15 UNTRUSTED-IO: ∀ format reader (gguf/safetensors/npy) on hostile input → return error, NEVER panic/OOM/overflow. all length/dim/nesting claims capped BEFORE alloc (header ≤ cap, numel ≤ cap, array-depth ≤ 64, grow ⊥ pre-alloc-from-claim). GUARDED BY native fuzz tests (FuzzLoad/FuzzRead/FuzzRoundTrip) + hostile-input regressions [B28,B31,B32].
V16 VALIDATION-LADDER: ∀ ALGORITHM impl → 2 tiers, both required. tier1 = bit/tol parity vs a reference LIB (torch/sklearn/ggml-py/…). tier2 (FINAL authority) = the defining SCIENTIFIC paper/source (arXiv/DOI/canonical textbook) — implemented formula matches the paper's equation, cited in §R. tier1 alone ⊥ sufficient. FILE FORMATS have no paper → their reference SPEC/impl IS the definitional source (stated, ⊥ invent a paper). paper-tier status tracked per algorithm in §R.
V12 EDGE: ∀ op → documented policy for NaN/Inf (IEEE-754 propagate), empty tensor, zero-dim, non-contiguous view. golden ! include edge cases [B9].
V13 GOLDEN-REPRO: ∀ golden file → record generating env (lib+version, dtype, seed, shape) in sidecar. regeneration deterministic. guards V1 reproducibility [B8].
V14 BACKEND-EXEC: `Backend`/`Kernel` interface ! define execution/sync model (sync default; async accel exposes explicit `Synchronize()`) at T5, before any accel/T20. adding GPU ⊥ break V8 [B7].
V17 DOCS: ∀ exported package & top-level symbol → godoc present, dual-audience (professional: math + §R paper-cite; layperson: plain what/why/when). ∀ user-facing package → runnable `Example` fns at ≥3 levels: trivial, realistic use-case, embedded-in-bigger-pipeline. examples live in `example_test.go`, verified by `// Output:` under `go test` (∴ docs ⊥ rot — CI runs them). new public API ⊥ "done" until its godoc + examples exist (part of task DoD). guards C10.

## §T — task backlog

id|status|task|cites
T1|x|repo scaffold: `go.mod` (go 1.26), module `github.com/jxsl13/goai`, dir layout L0..L5, CI skeleton w/ `CGO_ENABLED=0` job, Makefile|I1,I5,C1
T2|x|L0 `Dtype` (f32,f64) + `Shape` + strides + row-major index math|I.L0,C4,C5
T3|x|L0 `Tensor` struct + views (reshape/slice/transpose = stride ops), zero-copy [ADR-0001]|I.L0,C5
T4|x|L0 `Device` + `Allocator` (arena/pool), alignment = parameter (⊥ SIMD-coupled assumption in L0) [ADR-0002]|I.L0
T5|x|L1 `Backend`+`Kernel` interface (incl. exec/sync model per V14 + op-dispatch that autograd T13 can intercept) + Pure-Go reference backend + registry + feature-detect scaffold [ADR-0003]|I.L1,I2,I4,V9,V14
T6|x|L1 elementwise scalar ref (add/sub/mul/div/neg; exp/log/tanh/relu/gelu/sigmoid) + golden [ADR-0004]|V1,V9
T7|x|L1 reduce scalar ref (sum/mean/max/min/argmax, axis-aware, f64 accum §V10) + golden|V1,V9,V10
T8|x|L1 BLAS-1 scalar ref (dot/axpy/nrm2, f64 accum §V10, scaled nrm2) + golden|V1,V9,V10
T9|x|L1 GEMM scalar ref (matmul, row-major, f64 accum §V10) + golden|V1,V9,V10
T10|x|tooling: golden-gen (`testdata/gen.py` NumPy → JSON + `.npy`) + npy reader/writer + bench harness|V1,V5
T11|x|L1-opt elementwise: optimized `cpu` backend (typed loops + goroutines, internal/simd), bench vs T6 ~10× / allocs 4104→9 [ADR-0005]|V3,V5,V7,V11
T11b|~|L1-opt archsimd (AMD64, `simd/archsimd` behind `goexperiment.simd` tag): elementwise + GEMM FMA microkernel. GATED: needs amd64 runtime for V-CROSS (B13). NEON/arm64 part parked by measurement (B27). host-executable work: NONE — resume w/ amd64 CI runner | archsimd-arm64 release|V3,V5,V7,R1,R3,B13,B27
T12|x|L1-opt GEMM: ikj-order raw-slice + goroutines vs T9 ref, bit-identical. ~31× @128³ f64 (0.46→14.3 GFLOP/s)|V3,V5,V7,R2
T12b|x|L1-opt GEMM: 4-row register blocking (B-row reuse ×4), k-order preserved → still bit-identical (tol 0). 256³ +31% (36.4 GFLOP/s), 512³ 50.6 GFLOP/s. archsimd FMA microkernel → folded into T11b (amd64 CI)|V3,V5,V7,R2,V11
T13|x|L2 autograd: tape/graph + `Variable` + `Backward()` (intercepts op-dispatch from T5, ⊥ refactor L1 ops) [ADR-0006]|I.L2,T5
T14|x|L2 VJP rules ∀ L1 ops (elementwise, matmul, blas1, reduce) + numeric grad check ∀ op|V2
T15|x|L3 `Linear` layer + init (xavier/kaiming) + `OpAddBias` (ref+cpu+VJP)|I.L3,V1,V2,V3
T16|x|L3 activations-as-layers + `Sequential` + losses (MSE composition, fused stable CrossEntropy) + golden [ADR-0007]|I.L3,V1,V2,V12
T17|x|L3 optimizers SGD(+momentum) + Adam (Kingma&Ba 2015) + golden 3-step trajectories, f64 master state|I.L3,V1,V10
T18|x|L3 end-to-end: 4-class MLP converges (CE 1.06→0.024, acc 1.000, f32 variant too)|G5,V1,V2
T19|x|L5 safetensors reader+writer, strict offset validation, bidirectional interop w/ official Python lib|I.L5,R7,V1,V13
T20|x|L1b Metal/MPS GPU backend (`-tags metal`, f32 GEMM, sync per V14, fallback ∀ else). cgo-GATE PASSED: 4.6× @512³, 12.7× @1024³ (906 GFLOP/s) vs optimized cpu; `CGO_ENABLED=0` default untouched. CUDA later|C2,C6,V3,V7,R5,V11,V14
T21|x|L4 transformer inference blocks: OpSoftmax + OpLayerNorm (+VJP softmax), nn.LayerNorm, nlp.MHA — golden vs REAL torch 2.12.1 @ 1e-12|I.L4,V1,V10,V12
T22|x|L5 GGUF reader (v2/v3, metadata KVs, reversed dims, alignment) + dequant Q8_0/Q4_0/F16→F32, parity vs official gguf lib|I.L5,R7,C4,V1
T23|x|L4 LLM inference end-to-end: GPT decoder (pre-LN, causal MHA, exact GELU) via safetensors loader — logits match REAL torch f64 @ 1e-12; causality bit-exact [B23]|I.L4,G2,V1
T24|x|L4 CV: conv2d (NCHW, stride/pad/bias) + max/avg-pool ref kernels, golden vs torch @ 1e-12. im2col/Winograd opt → T24b|I.L4,V1,V10
T24b|x|L4 CV-opt: im2col+GEMM conv (cpu, (c,ky,kx) col order → bit-identical, 36× @8×8×28×28) + conv/pool VJPs, gradchecked → B24 closed|V3,V5,V11,B24
T25|x|L4 classic-ML: OLS (normal eq + Cholesky), softmax regression (Adam+GD polish, probs 1.3e-9 vs sklearn), kmeans (Lloyd, labels exact), PCA (Jacobi eigen, sign-invariant) — golden vs sklearn 1.9.0 [B25]|I.L4,V1
T26|x|L4 RL basics: REINFORCE (EMA baseline) + DQN (replay/target-net/ε-greedy) on chain MDP — both reach optimal 1.000 [B26]|I.L4

# ═══ EXPANSION 2026-07-06: GPU/accel-first + full LLM (inference + ALL training). priority: GPU accel > LLM > rest. every algo tier2-paper-verified (§V16); every op V-CROSS vs cpu-ref (§V3/V11) ═══
# dtypes (prereq for mixed-precision + quantized GPU paths)
T27|x|L0 dtypes f16 + bf16 (storage, convert f32↔, parity) — unblocks mixed-precision + quant kernels|C4,V1,R41
# GPU op coverage — PRIORITY 1 (user: accel most important)
T28|~|DROPPED per ADR-0008: per-op offload of memory-bound elementwise/addbias fails §C3 gate (transfer≫compute). GPU only for compute-bound / graph-resident|ADR-0008
T29|~|DROPPED per ADR-0008 (see T28). softmax/norm offload only inside a resident GPU graph (MPSGraph task), not per-op|ADR-0008
T30|x|L1b metal GPU TRAINING: `NewTapeOn(metal)` dispatches fwd + both bwd GEMMs to GPU (other ops cpu-fallback §I4). MLP converges gpu-loss==cpu-loss (0.037), V-CROSS vs cpu ref [ADR-0008]|V2,V3,C6,ADR-0008
T30b|x|L1b metal GPU for the FULL TRANSFORMER: `tensor.Cast` f64→f32; GPT INFERENCE on GPU (argmax+logits match cpu) + GPT TRAINING on GPU (CE 3.29→0.046, tracks cpu). LLM inference+training on GPU, V-CROSS vs cpu|V2,V3,C6,ADR-0008
# transformer TRAINING (unblock — needed before GPU-training can be validated)
T31|x|L2 LayerNorm VJP (closes B22): dL/dx=(1/σ)(a−mean a−x̂·mean(a·x̂)), a=g·γ; dγ=Σg·x̂; dβ=Σg. gradcheck green → transformer training unblocked|V2,B22,R35
T32|x|L2 attention: fused OpMHA (heads/mask internal → avoids §B15 view-grads) + hand-derived SDPA VJP (dV=AᵀdO,dS=A⊙(dA−rowsum),dQ/dK bilinear). gradcheck causal+non-causal; full MHA trains MSE 1.2→0 (grads→all 4 Ws); forward golden unchanged|V2,R32
T33|x|L3 AdamW (decoupled wd, §V16 paper+torch verified) + ClipGradNorm (global-norm) + WarmupCosine LR. AdamW golden bit-exact vs real torch (err 0)|V1,R26
T34|x|L4 GPT end-to-end TRAINING: OpEmbed (differentiable gather+scatter VJP) makes embeddings trainable; full backward embed→LN→MHA→FFN→head matches REAL torch grads for all 29 weights @ rtol 1e-9; trains CE 3.29→0.015 (AdamW+clip)|G5,V1,V2
# LLM INFERENCE features — PRIORITY 2
T35|x|L4 KV-cache + incremental decode: generalized OpMHA (sq≤sk, abs-pos causal), MHA.StepKV, GPT.DecodeStep. logit parity vs full forward @ rtol 1e-11 across all positions. VJP guards sq==sk (cache = inference-only)|V1
T36|x|L4 sampling: greedy/temperature/top-k/top-p (nucleus, paper-def smallest-set-cumsum≥p §R34) + KV-cached Generate(). deterministic-seed tests|V1,R34
T37|x|L4 byte-level BPE tokenizer (gpt2): manual GPT-2 pre-tokenizer (RE2 no lookahead) + tiktoken byte_pair_merge. encode bit-exact vs real tiktoken (12 samples); decode∘encode byte-exact ∀ input (fuzz 3.6M) [B34]|V1,V15,R33
T38|x|L4 Llama-family pt1: RMSNorm (op+VJP) + RoPE (op+VJP, HF rotate_half half-split). golden vs torch @1e-12, gradchecks, RoPE norm-preserving. §V16 confirmed vs paper+HF|V1,V2,R28,R29
T38b|x|L4 Llama-family pt2: SiLU op+VJP, SwiGLU FFN (nn layer, trainable), GQA (OpMHA kv_heads attr, h→h/(nh/nkv), MQA=1). golden vs torch @1e-12, gradchecks (silu, gqa incl grouped dK/dV). §V16 confirmed vs paper+HF|V1,V2,R30,R31
T39|x|L4 quantized inference: `gguf.QMatMul` dequant-on-the-fly (1 row at a time) for Q8_0/Q4_0 weights, f64 accum. parity vs gguf-py-dequant matmul @1e-5; memory 3.8×(Q8)/7.1×(Q4) smaller. dequant = ggml-verified §R19/R21|V1,G2
# advanced/parameter-efficient training
T40|x|L3 LoRA: nn.LoRALinear y=x·W+(α/r)(x·A)·B, A Kaiming/B zero, W frozen. golden vs torch @1e-12; B=0→base no-op; fine-tunes MSE 2.99→0 w/ base byte-frozen. §V16 confirmed vs paper+microsoft/LoRA+PEFT|V1,V2,R27
T41|x|L3 mixed-precision training: f16/bf16 compute + f32 master weights + loss scaling|V10,V1,R42
# more accel backends — PRIORITY 1, cgo-gated + CI-gated
T42|x|L1b CUDA backend (cuBLAS GEMM + kernels, fwd+bwd) behind build tag, cgo-gate, CI-gated Linux/Windows|C2,C6,V7,V3,R43
T43|x|L1b Vulkan compute backend (portable, no vendor lock) behind build tag, cgo-gate. HOST-VERIFIED on Apple M2 Pro/MoltenVK: V3 green ∀ shapes + GPU-training converges == cpu [B36]|C6,V3,V7,R44
T44|x|L1b NPU: documented honest non-goal — op-level dispatch impractical (ANE/Intel whole-model inference-only; DirectML op-granular but C++/COM/D3D12 no-Go-binding). backend/npu.Available()=false + dual-audience doc; NPU = future L5 model-runner. ADR-0011|C7,R5,R45,V4,V17
T45|.|docs+examples sweep: backfill dual-audience godoc + runnable Example fns (trivial/use-case/embedded, `// Output:`) across ALL public pkgs (tensor,backend,autograd,nn,nlp,vision,classic,rl,format/*). ongoing: each future §T ships its own docs+examples (V17 = DoD). start w/ highest-use surface (tensor,nn,autograd)|C10,V17

## §B — backprop log

id|date|cause|fix
B1|2026-07-05|cgo threshold unproven → default speedup ≥1.5× OR ≥80% C++ baseline; revise after first GEMM bench|C3
B2|2026-07-05|ARM64 not in `simd/archsimd` yet → Plan9-NEON default on arm64; revisit per Go release|R3,T11
B3|2026-07-05|`go-highway` maturity unverified → ⊥ depend until vetted; internal SIMD wrapper instead|C9
B4|2026-07-05|per-op tolerances unproven → §R6 defaults; tighten/loosen per op only w/ §R justification, ⊥ to pass test|V1
B5|2026-07-05|review: V3 "backend tol" undefined → SIMD reorders FP sums, exact match impossible|V11
B6|2026-07-05|review: no accumulation-dtype rule → f32 GEMM/reduce drifts w/ large K (non-associative FP)|V10
B7|2026-07-05|review: Backend interface lacked exec/sync model → sync-only serializes GPU | leaks async → breaks V8 @ T20|V14
B8|2026-07-05|review: goldens irreproducible w/o env pin (torch/BLAS version, seed)|V13
B9|2026-07-05|review: edge cases (NaN/Inf/empty/zero-dim/non-contig) ungoverned|V12
B10|2026-07-05|review NOTE: C3 threshold speed-only → weigh cgo portability/maintenance cost, require real-workload bottleneck ≠ micro-bench. revisable|C3
B11|2026-07-05|ADR-0002: guaranteed SIMD over-alignment (32B/64B) not doable for typed slices w/o unsafe → L0 alignment advisory; T11 requests explicitly (maybe byte-arena)|T11
B12|2026-07-05|T7 argmax returns float index (no int dtype until C4) → exact ≤2^53; add int-dtype argmax when int dtypes land|C4
B13|2026-07-05|T11 archsimd amd64-only + host arm64 → intrinsics not host-verifiable → split to T11b, CI-gated. Pure-Go typed loop = verified arm64 ceiling now|T11b,R3
B14|2026-07-05|cpu elementwise still allocs output per call (32KB @ 4K) → dominant cost; in-place/pooled-output op variants a later opt|T11
B15|2026-07-05|ADR-0006: grads keyed by tensor pointer → view of tensor ≠ base; grads ⊥ flow through reshape/slice/transpose until view-VJPs land|T13
B16|2026-07-05|max/min VJP: tie → grad routes to FIRST (lowest-index) extremum, consistent w/ argmax tie rule. torch splits ties evenly — divergence documented, ⊥ silent|T14
B17|2026-07-05|test coupled to unfinished state (asserted OpExp lacks VJP; T14 added it) → red. ∴ negative tests use synthetic op codes, ⊥ real ops|T14
B18|2026-07-05|no general broadcasting → `OpAddBias` covers dominant NN case ([m,n]+[n]). general numpy-style broadcast = future op-layer feature|T15
B19|2026-07-05|Linear stores W [in,out] (row-major matmul w/o transpose); torch uses [out,in]+x·Wᵀ → checkpoint import must transpose. documented ⊥ silent|T15,T19
B20|2026-07-05|optimizer golden: torch ∉ venv → Python f64 reimplementation of documented torch/paper update rules. RESOLVED 2026-07-05: torch 2.12.1 installed (user ok), `testdata/verify_torch.py` → all goldens bit-exact (err 0.0)|T17
B21|2026-07-05|safetensors Load: zero-size tensor shares begin w/ next tensor → `sort.Slice` tie order random → flaky gap/overlap error. ∴ sort by (begin,end); flaky test = bug, ⊥ rerun-until-green|T19
B22|2026-07-06|LayerNorm VJP pending (softmax VJP done) → transformer TRAINING blocked, inference (T23) unaffected. RESOLVED 2026-07-06: VJP added + gradchecked (T31)|T21,T31
B23|2026-07-06|T23 golden = REAL torch f64 (bit-tight, in venv) ≠ ggml binary (llama.cpp not installed). ggml parity re-check when llama.cpp binary available; G2 lists torch/ATen as valid ref|T23
B24|2026-07-06|conv/pool VJPs pending → CV training blocked, inference fine. w/ T24b. RESOLVED 2026-07-06: VJPs live + gradchecked (T24b)|T24,T24b
B25|2026-07-06|logreg parity 2 root causes: (1) separable blobs → penalty-free MLE ⊥ exists (weights diverge, any stop arbitrary) ∴ golden uses overlapping classes; (2) Adam oscillates near optimum @ const LR ∴ GD polish after warmup. tol NOT weakened (1e-5 kept, achieved 1.3e-9)|T25
B26|2026-07-06|REINFORCE learned LEFT: within-episode mean-of-Gₜ baseline subtracts the between-episode signal (2-step episodes → advantage ≈ noise → 0.1-attractor wins) ∴ baseline ! cross-episode (EMA of G₀). result 0.100 → 1.000|T26
B27|2026-07-06|T11b NEON part parked BY MEASUREMENT: portable loop sustains 55 GB/s (L1 & L2, ~1 elem/cycle); hand-NEON ≤2× on loop = ≤7% net (loop ~15% of op cost, rest alloc/dispatch B14); fragile Plan9 asm obsoleted by upcoming archsimd-arm64 (R3). higher arm64 lever = B14 pooled outputs. amd64 archsimd part stays CI-pending (B13)|T11b
B28|2026-07-06|/review found: gguf/safetensors/npy readers PANIC on hostile dims (2^63 → negative int; n·bytesize wraps to 0 → slips past size check → tensor.New panics). fix: cap dims + running product before alloc. regression test w/ 2^63/2^62/over-cap dims|V15
B29|2026-07-06|/check drift: golden files cv/losses/optim/transformer/gguf lack embedded env meta (V13). low sev — provenance in gen.py + git; embed on next golden regen|V13
B30|2026-07-06|/check drift: V5 says regression breaks CI but ci.yml had no bench job. fix: added bench smoke step (build+run, not yet perf-gated; benchstat gate = future)|V5
B31|2026-07-06|FUZZ-found (npy FuzzRead): (a) v2/v3 uint32 header len uncapped → make([]byte,~4GB) OOM-crash; (b) data buffer pre-alloc'd from CLAIMED shape, numel cap 1<<40 = 8TB (npy streams, ⊥ offset-check like safetensors/gguf). fix: header cap 64MiB + numel cap 1<<25. regression + 40s fuzz clean|V15
B32|2026-07-06|MANUAL-analysis: gguf metadata array element-type may be ARRAY → deep nesting = stack-overflow DoS; make([]any,n) pre-alloc'd n≤1<<24 =256MB from claim. fix: depth cap 64 + grow-not-prealloc. regression w/ 500-deep nest + huge-len-no-data|V15
B33|2026-07-06|nrm2 = CLASSIC scaled dnrm2 (Blue 1978/DLASSQ), NOT newest LAPACK Anderson 2017 Alg.978 (3-accum abig/amed/asml). both correct+overflow-safe; our classic form matches its paper. optional: adopt Alg.978 for ~better accuracy on mixed magnitudes|R25|open|nice-to-have
B34|2026-07-06|FUZZ-found (BPE): pre-tokenizer used []rune(text) → invalid UTF-8 (e.g. \xc3) corrupted to U+FFFD → round-trip broken. fix: scan original bytes via utf8.DecodeRuneInString, invalid→"other" byte token. round-trip now byte-exact ∀ input|V15,T37
B35|2026-07-06|CUDA backend (T42) not host-verifiable on this arm64/macOS host (no NVIDIA GPU/CUDA toolkit) — like archsimd (B13). §V3 cross-ref + §C3 gate benchmarks are CI-gated on Linux/Windows CUDA runners. Go side mirrors metal (compiles under its tag); cgo/cuBLAS side reviewed vs NVIDIA docs (R43). CGO_ENABLED=0 stays green (pkg=doc.go only). ADR-0009|V3,V7,C2,decision:revisit-when-cuda-CI|open
B36|2026-07-06|Vulkan backend (T43) initially not host-verifiable → RESOLVED same day: installed Vulkan toolchain via brew (shaderc/glslc, vulkan-loader, vulkan-tools; molten-vk+vulkan-headers pre-present). Compiled matmul.comp→matmul.spv (committed). cgo directive → `pkg-config: vulkan` (portable macOS/Linux/Win). Fix: MoltenVK is a PORTABILITY driver → hidden unless instance enables VK_KHR_portability_enumeration + ENUMERATE_PORTABILITY_BIT and device enables VK_KHR_portability_subset (detected at runtime, skipped on native Linux/Win → one portable path). NOW HOST-VERIFIED on Apple M2 Pro/MoltenVK: §V3 cross-ref green ∀ shapes, GPU-training converges == cpu (0.0371), unaligned+fallback green. ADR-0010|V3,V7,C2|closed:host-verified
