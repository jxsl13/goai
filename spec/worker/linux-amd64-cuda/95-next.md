## §NEXT — open levers

**Tw-COOPMAT (booked 2026-07-18, research-lite CONFIRMED 3/3 + host probes): the prefill
toolchain wall MOVES via Vulkan.** How llama.cpp actually gets its pp lead on this GPU:
ggml-vulkan runs tensor cores FROM GLSL — VK_KHR_cooperative_matrix (mul_mm.comp COOPMAT) and
on NVIDIA VK_NV_cooperative_matrix2 (mul_mm_cm2.comp, coopMatLoadTensorNV with dequant-decode
callbacks = dequant-on-the-fly inside the tile, BM/BN=64 BK=16-32), plus an int8 DP4A MMQ
path (VK_KHR_shader_integer_dot_product + runtime q8_1 activation quant, PR #12135). ALL
CUDA-free (glslc -> SPIR-V). HOST PROBES PASS: the RTX 3060 exposes KHR_cooperative_matrix
rev2 + NV_cooperative_matrix2 + integer_dot_product (driver 610.43), and our glslc
(shaderc 2026.2) compiles both a coopMatMulAdd f16 shader and a dotPacked4x8AccSatEXT shader
clean. => Arc: implement a coopmat GEMM (then dequant-in-tile, then DP4A MMQ) in OUR vulkan
backend and route prefill through it; beats the cuBLAS-f16 CUDA ceiling (5600 tok/s pp128 vs
llama.cpp 8474) without nvcc. The crashed session's vulkan-sdk/matmul.comp scratch was this
same trail. Sources: ggml/src/ggml-vulkan/vulkan-shaders/{mul_mm.comp,mul_mm_cm2.comp,
dequant_funcs_cm2.glsl,mul_mmq.comp}, PRs #10206/#10721/#12135/#10713.


**GAP RESEARCH 2026-07-15 (research-lite, 3 angles + synth, all confirmed with primary
sources): the remaining single-GPU decode lever class is llama.cpp's 2025 FUSION STACK
(github.com/ggml-org/llama.cpp/discussions/17621): GEMV+gated-act epilogue fusion,
RMSNorm+MUL+ADD fusion, concurrent Q/K/V streams — each <10%, cumulative +17-43% on
RTX 4090/5090. REPO CROSS-CHECK: epilogue fusion = REAL GAP (we run gate-GEMV, up-GEMV,
separate SwiGLU kernel; NOT the Tw39 rejection — that combined the two GEMVs, this
fuses the activation into the GEMV epilogue); concurrent QKV streams = REAL GAP (we run
the 3 projections sequentially; K/V GEMVs are small under GQA); residual-add fusion =
ALREADY HAVE (QMatMulAccInto beta=1); TopK-MoE fusion = N/A (no MoE on the CUDA path);
vLLM-V1/SGLang CPU-overhead levers = N/A (graph replay is already ~zero host overhead);
KV-quant-for-speed = externally CONFIRMED DEAD (matches Tw53); Marlin W4A16 = PARKED
(GEMV-parity at batch 1, pays at batch 16+ only). Booked as Tw55.**



**STATE 2026-07-15 (post-Tw53): the decode-attention arc is CLOSED — Tw52 flash
(+3.5%/+26%), Tw53 f16 KV (flat speed, −50% KV VRAM, opt-in). (b2) flash-PREFILL
is PARKED BY PROFILE (attention = 13.8% of prefill, FFN GEMM 53.8% — ≤14% ceiling;
the llcpp prefill gap is GEMM utilization, cublas already near its rate). ALL LISTED
LEVERS CLOSED: (c2) native Q6_K GEMV DONE (Tw54 — flat speed after the
transactions-not-bytes fix, bit-native, −23% minority VRAM); (d) was RESOLVED-STALE
(the CI hygiene rider closed by PR#100; verified against the latest main run log —
only the accepted init.defaultBranch hints remain). The worker is now on the
empty-backlog rule: gap research / beat-the-incumbents on the next hot path.**

**STATE 2026-07-15 (post-Tw40/41, historical): Q4 arc closed with the fair compare; goai-Q4 leads
llama.cpp-Q8 at every scale (max +13% at 7B); a real 7B (Mistral) runs coherently on
the 12GB card. Remaining decode gap is same-class Q4 (0.74-0.80× vs Q4_K_M) = the
super-block-quant + fused-attention kernel margin. Tokenizer §B59 fixed via nlp.SPM (additive).**

**STATE 2026-07-15 (pre-Tw40, historical): the CUDA inference arc is COMPLETE + documented + validated.**
Decode 26→164.7 tok/s (6.3×), within 1.48× of llama.cpp Vulkan; greedy+sampled
generation writes real text; graceful −14% to 2048 ctx (PERF-LONGCTX). All
primitives public in backend/cuda. Recent probes were REJECTIONS (Q4, Q8-prefill)
→ the easy optimization space is exhausted.

**HIGHEST-VALUE NEXT (public API — usable library): a llamagpu CUDA adapter.**
llamagpu.Decoder is already a backend-agnostic, single-command-buffer,
resident-quantized GPU decoder with New(Metal)/NewVulkan/NewQuant adapters + a
clean public Generate(prompt,maxNew,sampler). Adding NewCUDA/NewQuantCUDA =
implement its recorder/linear/buffer interfaces over my backend/cuda primitives
(the Decoder core then gives Step/StepN/Generate/sampler FOR FREE). My primitives
already cover most of `recorder`: RMSNorm✓ MatMul/MatMulAcc✓ RoPE✓ MHA/GQA✓
Unary/Binary✓ QMatMulResident✓(Q8). GAPS to add in backend/cuda first (low-collision,
mine): LayerNorm✓ AddBias✓ (PR#63), Copy2D✓ Blit✓ (PR#73),
RoPEAt/RoPEPair✓(fused-QKV bands, strided RoPE — cu_rope_f32_band + RoPEAtBand/RoPEPairBand, PR#74; validated == host ref, v band untouched) — ALL RECORDER GAP PRIMITIVES NOW COVERED; Commit/Wait ← my CUDA graph/stream.
ADAPTER SLICE 1 (PR#75): cuda.Recorder — full f32 recorder (RMSNorm/LayerNorm/AddBias/MatMul/MatMulAcc/RoPE·RoPEAt·RoPEPair/Blit/Copy2D/MHA·GQA-causal/Unary·SiLU·GELU/Binary·add·mul·swiglu/Commit·Wait·Finish·Free) + buffer iface on DeviceF32 (UploadF32/DownloadF32/Release, cu_upload_into H2D-into). Eager-submit model (Commit no-op, Wait=stream sync). Validated == host ref (TestCUDARecorder: MatMul/MatMulAcc/Unary/Binary/MHA). QMatMulResident stubbed → SLICE 2 (Q8 UploadQuant format conv).
ADAPTER SLICE 3 DONE (PR#76): llamagpu/cuda.go — cBuf/cRec + NewCUDA(f32) wiring the cuda.Recorder into the backend-agnostic Decoder. cuda.NewDeviceBufferF32 (raw-slice newBuffer); UploadF32/DownloadF32 relaxed to len≤capacity (fixed-capacity scratch, active-prefix fill, matches metal/vulkan). END-TO-END VALIDATED: llamagpu.NewCUDA(m).Generate == nlp.Llama.Generate greedy token-for-token, 2 models incl GQA prefill+decode (TestCUDAGenerateMatchesReference, TestCUDAPrefillStepMatchesReference). **NVIDIA IS NOW A FIRST-CLASS llamagpu BACKEND** (Generate/Step/StepN/samplers free, on par with metal+vulkan).
ADAPTER SLICE 2 DONE (PR#77): quantized path — NewQuantCUDA + cudaUploadQWeight (any ggml qt → Dequantize → transpose → uniform resident Q8; CUDA has one Q8 kernel unlike metal/vulkan's per-type) + recorder.QMatMulResident (cu_qmatmul_q8, beta=0) + ResidentBQ8.Close() (qweight). Validated: TestCUDARecorderQ8 (Q8 GEMV vs host f32, max rel 2.7% — genuine Q8 acc) + TestCUDAQuantGenerateValid (end-to-end Q8 decode, valid/prefix/length contract). **ADAPTER COMPLETE: f32 + Q8 both live via unified Decoder.**
THROUGHPUT BENCH DONE (PR#78, TestCUDAUnifiedVsGraphDecodeThroughput): SAME TinyLlama-1.1B Q8 weights, ONE window — UNIFIED llamagpu (recorder, eager per-op submit, no graph) 44.4 tok/s vs BESPOKE graph (fixed buffers + CUDA graph + dev argmax) 160.6 tok/s → **graph is 3.61× the unified path**. The unified path is launch-bound: per-op kernel launches (no graph collapse) + full-logit [1,V] D2H + host argmax every token, vs one graph replay + on-device argmax. VERDICT: unified path = correctness + Generate/samplers for free but NOT peak; graph capture is the big lever (3.6×). NEXT: graph-capture mode in the unified path needs a capture hook in llamagpu/decoder.go (MAIN-MACHINE-OWNED shared core) → COORDINATE/DEFER, don't unilaterally refactor. Meanwhile: on-device argmax + logit-download elision are the launch-bound wins I CAN land in backend/cuda without touching the shared decoder. MAIN MACHINE IDLE ≈12 fires → adapter collision risk LOW, building it incrementally: gap primitives first (backend/cuda, mine), then llamagpu/cuda.go (NewCUDA + cudaBuf/cudaRec via backendOps struct — mirrors llamagpu.go/vulkan.go ≈150L). No import cycle (nlp/llamagpu
don't import backend/cuda).
**BLOCKER — DEFER:** llamagpu is a MAIN-MACHINE HOTSPOT right now (T613 QKV/RoPE/
SwiGLU fusions, T614 encode-overlap, T644 in-flight) → editing its interfaces now
COLLIDES. Wait for llamagpu to settle, then add the CUDA adapter (coordinated),
OR the main machine adds it on my public primitives. Until then my primitives are
the enablement; the assembled decoder stays in backend/cuda tests.

Other deferred levers: flash-style attention + tiled quant GEMM (prefill gap
2.5-3.6×, modest EV at short ctx); f16 KV cache (only ≈14% context-dependent, low
EV); Qwen (needs nlp QKV-bias fields, main-machine); T631 VRAM-offload (needs a
>12GB model + shared-executor infra).

--- historical (pre-completion) ---
Nx1: ◐ CUDA activation residency — Phase-2 device-matmul CHAIN DONE (GPU-5/Tw11, `DeviceF32`, 29× MLP). remaining = full recorder (arbitrary op chains in one submit, async, non-matmul ops on device) + integrate into an nn/llamagpu decode path (currently a standalone primitive, orphan until wired). BIG. USER PRIORITY.
Nx2: ◐ f32-native nr=16 DONE (Tw10, 153 GFLOP/s). remaining → cache blocking (ADR-0017 re-open this large-cache x86) + FMA-saturation microkernel to close §GAP F32 3.8×/F64 2.7×.
Nx3: ✅ FULL Llama-block kernel set on-device (GPU-7): matmul + GELU/SiLU + residual-add + RMSNorm + Softmax + RoPE. attention (RoPE→QKᵀ→softmax→·V) AND FFN (RMSNorm→matmul→SiLU→matmul→residual) run fully resident. ◐ Nx1: SwiGLU FFN block composed+verified e2e (Tw18). ✅ FULL single-head decoder LAYER composed+verified e2e (Tw21, max rel 1.4e-5): pre-norm attn (Q/K/V/O proj + RoPE + causal attn + residual) + pre-norm SwiGLU FFN + residual, all resident. ✅ FULL-MODEL FORWARD-PASS OP SET COMPLETE on-device: embed-gather → decoder layers (MHA/GQA + SwiGLU FFN + norms + residuals) → final RMSNorm → output matmul→logits. All verified vs ref. ONLY REMAINING for real inference: real GGUF weights (needs USER download permission) + tokenizer + the glue (multi-layer loop, load weights into ResidentB). NEXT candidates: multi-layer stack test (synthetic); perf-tune resident reduction kernels (warp-shuffle); XPos RoPE; OR real weights if user approves. then a full decoder layer; real GGUF weights; batched/GQA Sgemm; expose via backend Kernel interface for nn/llamagpu. + XPos RoPE.
