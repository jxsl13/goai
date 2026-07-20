## §GOAL — beat ALL professional incumbents, target = cuBLAS parity (user directive 2026-07-18)

The performance bar is NOT llama.cpp. User directive: "wir wollen an cuBLAS drankommen, da
llama.cpp scheinbar noch nicht gut genug ist" + "es gibt bessere Konkurrenten zu llama.cpp,
die professionell eingesetzt werden. vergleiche auch mit diesen. wir wollen besser als alle
drei sein." So the standing targets, in order of ceiling:
1. **cuBLAS-f16 GEMM = 26.5 TFLOP/s** (the RTX 3060 tensor-core kernel ceiling; our CUDA side
   already hits it — it's the number every engine below bottoms out at). Coopmat GEMM must
   reach it (currently 9.7 TFLOP/s f16-acc; B-streaming-bound, see Tw-COOPMAT below).
2. **The pro serving engines: vLLM, TensorRT-LLM, SGLang** (+ HF TGI) — not just llama.cpp.
   Benchmark against these on the SAME 3060 where installable (§RUN6 host-constraint caveat:
   no nvcc/system-CUDA/TensorRT SDK — research-lite is scoping which ship runnable pip wheels).
   Beat all of them on tokens/s (decode) and prompt-processing (prefill) at single-GPU scale.
The decode side already LEADS llama.cpp-Q8 (1.05-1.19x); prefill (0.66x) is the frontier the
Tw-COOPMAT arc attacks.

### §ROADMAP — beat-vLLM plan (research-lite 2x adversarial-verified, cited; 2026-07-18)

MEASURED 3-way (RTX 3060, TinyLlama-1.1B): decode batch=1 goai 257 > llcpp 244 > vLLM 103
(WE WIN); prefill pp128 vLLM 10729 > llcpp 8474 > goai 5600; batched-serving vLLM dominates
(a CAPABILITY goai lacks). Two fronts, different physics — prefill is COMPUTE-bound, decode
MEMORY-bound (confirmed):

**FRONT A — prefill (compute-bound), ranked levers (all buildable on CUDA now that nvcc works):**
A1. fp16 tensor-core GEMM + **fp16 activations END-TO-END** (kill the per-GEMM f32<->f16
    conversions ResidentBF16 does each call) — THE #1 research lever; our GEMM already hits
    cuBLAS 26.5 TFLOP/s so the conversion overhead is the waste.
A2. fused FlashAttention — DONE (#168, mma.h kernel, 3.0x the cuBLAS-batched path). NEXT: wire
    e2e into the resident prefill forward, measure pp128.
A3. fused epilogues: SwiGLU into the FFN GEMM epilogue; RMSNorm+RoPE+residual folded in.
A4. piecewise CUDA-graph for prefill (SGLang does; lower leverage — prefill kernels are long).
A5. chunked prefill (scheduling, not raw throughput).
REFUTED as levers: full-prefill-graph-capture, megakernels (those are decode-phase).

**FRONT B — serving throughput (the missing capability), minimal build order (PagedAttention
SOSP'23 / RadixAttention / FlashInfer, all cited):**
B1. paged KV-cache pool + block tables (16-token blocks, logical->physical).
B2. paged+ragged batched decode kernel (block-sparse/BSR, ragged qo_indptr) — needs mma.h /
    custom kernel = nvcc, now available.
B3. iteration-level continuous-batch scheduler (admit/evict per decode step).
B4. chunked prefill (interleave prefill/decode).
B5. RadixAttention prefix caching + copy-on-write.
(NOTE: the "18x" throughput figure is NOT a primary-source constant — papers state 2-4x vs
SOTA, up to 24x vs HF; don't cite 18x.)

Execution order: A2-e2e -> A1 (fp16 e2e) -> A3, in parallel B1->B2 when a serving arc opens.
The nvcc unlock carries BOTH fronts (fused-attn/epilogue kernels AND paged/ragged attention).

**PIVOT (2026-07-18, data-driven): FRONT A prefill is near its CUDA ceiling — go FRONT B.**
A2 landed e2e (pp128 5578->6807, +22%, gap to vLLM 0.52->0.63x, #171). Re-profiled WITH fused
attention (46.6ms, was 52.9): ffn-gemm 61.1% + qkv 12.6 + o 8.0 + head 4.7 = **86% cuBLAS
GEMM**; attention now 5.5% (was 14). A1 measured only ≈6%; A3 (rmsnorm 3.5 + rope 2.8 + swiglu
1.9) ≈8%. So the whole remaining CUDA-prefill upside is ≈14% (6807 -> ≈7800), and vLLM's
1.58x lead is NOT closable on the CUDA side — both run the SAME cuBLAS-f16 GEMM at M=128; vLLM's
extra comes from bigger effective batch (chunked prefill), not a faster kernel. DECISION: stop
grinding prefill; the biggest "beat all three" lever is FRONT B serving throughput (18x-class
CAPABILITY gap, where a from-scratch engine wins because it's a missing feature not a kernel
race). Start B1 (paged KV pool + block tables) -> B2 (paged/ragged batched decode kernel,
nvcc-buildable) -> B3 (continuous-batch scheduler). A3 is a fill-in only.

**STATUS 2026-07-20 — FRONT B decode perf EXHAUSTIVELY COMPLETE (see GPU-8):** B1✓ B2✓ B3✓ done + merged (graph-capture + GQA-shared + 32-key-tile paged decode, #177/#178). A1 fp16-activations (PR #181): token-identical to f32 on real TinyLlama, DeviceF16 clean API, prefill 1.13×. ⚠️ decode-vs-vLLM CORRECTED to ≈PARITY at matched context (the "1.33-1.37×" was a context-length mismatch — see GPU-8: our ctx-128 peak vs vLLM's ctx≈320 average; at matched 128→512 range ≈ 6452 vs vLLM ≈6988 ≈ 0.92×). The A1 engine win (kills per-GEMM converts, halves activation traffic) is real vs OUR f32 path (1.47×); the vLLM decode LEAD is not established at matched context. Decode perf at the f16 HARDWARE CEILING: GEMMs cuBLAS-f16-peak, attention 34%/latency-bound resisted 8 approaches (f16-KV, int8×2, multi-accum, p-in-shared, f16-in-shared, split-K, WMMA tensor-core decode — all measured, all lose to warp GQA kernel; 2.1× A1-noattn ceiling needs production-grade FlashDecoding beyond bounded effort). REMAINING = DEPLOYMENT not perf: wire A1 (DeviceF16) into a REAL batched serving loop — real prefill → PER-LAYER paged KV (22 caches, not the synthetic shared pool) → A1 continuous-batch decode (append K/V per step, fixed-buffer graph over growing KV like vLLM) → argmax/sample. Large multi-component engineering phase; the A1 kernel+accuracy foundation is done & validated. B4 (chunked prefill) + B5 (RadixAttention prefix cache) are further serving features.

**STATUS 2026-07-20 ≈12:25 — serving loop ASSEMBLED e2e + device-side append (PR #181):** the full graph-captured A1 decode step now runs as ONE cuda graph replayed over GROWING paged KV (RMSNorm_f16→QKV f16-GEMM→RoPE device-pos `RoPEF16DposRaw`→paged attn `BatchedDecodeAttnViewInto` over a persistent view→Wo→SwiGLU FFN; between replays `PagedBatchView.Update`/`UpdateLens` refresh + `DevicePos` advance). Make-or-break graph-over-growing-KV validated (`TestGraphDecodeGrowingKV`, rel-RMS 1.8e-7 vs eager each step). NEW real-serving primitive `cu_paged_append_batched` (+ `SeqKV.Reserve1`, `PagedKVPool.AppendBatched`, `PagedBatchView.UpdateLens`): one kernel scatters the step's own device-resident dk/dv into each seq's next paged slot — NO host round-trip (`TestPagedAppendBatched`: 6 seqs 14→20 across the block boundary, bit-exact). HONEST head-to-head (batch 512, 22 layers, IDENTICAL KV growth 128→328, graph-captured, matched f16/f16, 200x×2): device-side append **≈7470 tok/s** vs host per-seq append ≈6789 = **+10%** from the primitive. ⚠️ Iw8 CORRECTION: an intermediate commit (a534bf1) claimed "9425 clean decode = 1.35× vLLM vs 6832 host-append" attributing the gap to the harness — NOT apples-to-apples (the 9425 bench keeps KV FIXED at 128, so most of that gap is cheaper short-context attention, not append overhead); corrected in 9eab0e9. The ≈9310 tok/s is ONLY the fixed-128-KV decode-step ceiling; no new serving-vs-vLLM multiplier is claimed from the growing-KV bench (no matched-context vLLM baseline run). The isolated-step **≈1.33-1.37× fair vLLM** headline stands. Net: serving loop assembled + correct + last host round-trip removed. REMAINING = productization (wire DeviceF16 into `residentLlamaF16` real decode w/ per-layer paged KV; matched-context vLLM serving re-bench) — engineering + measurement, not new decode perf.
