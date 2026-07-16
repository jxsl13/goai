---
name: inference-benchmark-hyperoptimize
description: User directive (2026-07-15) — benchmark GPU inference throughput vs industry-standard (llama.cpp) then hyper-optimize everything
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 491e34f2-de40-467f-89a0-f918efb8a6b3
---

Benchmark goai's GPU inference **throughput** (tokens/s) against **industry-standard inference implementations** (llama.cpp is the direct GGUF/CUDA comparison; also relevant: vLLM, HF TGI/transformers), then **hyper-optimize everything** to close/beat the gap. User directive 2026-07-15, EMPHATIC ("definitely").

**Why:** the CUDA backend now runs a real LLM (TinyLlama-1.1B) end-to-end on the RTX 3060, token-for-token vs CPU (§Tw30). The user wants it FAST vs the incumbents, not just correct.

**MEASURED baseline (PR#33/#34, RTX 3060, TinyLlama-1.1B, sub-spec §PERF):**
| test | llama.cpp Vulkan Q8_0 | goai CUDA f32 | goai/llcpp |
|---|---|---|---|
| prefill pp32 | 3298 tok/s | 1204 | 0.37× |
| prefill pp128 | 8389 tok/s | 2002 | 0.24× |
| decode tg128 | 243 tok/s | 12.6 (KV-cache, PR#36) | 0.052× (19×) |
goai from-scratch prefill is within 2.7–4.2× of llama.cpp — solid start. Reproduce industry side: `scripts/bench-llamacpp.sh` (downloads prebuilt llama.cpp Vulkan; CUDA build impossible — no nvcc/no Linux CUDA prebuilt). goai side: `go test -tags cuda -bench 'TinyLlamaPrefill|TinyLlamaDecode' ./backend/cuda`.
**KV-cache DONE (PR#35 attn kernel, PR#36 KVCache+decode):** decode == full re-forward token-for-token (correct); 12.6 tok/s FLAT across context (cache is O(1)/step). Gap to 243 = 19×, **LAUNCH/SYNC-BOUND not bandwidth** (55 GB/s effective ≪ 360 peak; single-token forward = ~330 op launches, same fixed cost as prefill, 1/128th compute; GQA pointer-array does cudaMalloc+cudaMemcpy+sync EVERY call, 22×/token).
**QUIET-box TRUE decode = 25.7 tok/s** (PR#46; the 12.6→16.35 logged during the 4-subagent campaign were contention-depressed — launch-bound decode is very host-CPU sensitive; paired-A/B relative gains still valid). Gap to llama.cpp 243 = 9.5×. Fused gate|up REJECTED (−4%, strided swiglu coalescing).
**Q8 arc STARTED (PR#47):** ResidentBQ8 (transposed [N,K] int8 + per-32-block scales) + warp-per-output GEMV kernel — 0.4% err vs f32, decode-GEMV 1.88× faster than cuBLAS f32. Q8 full decode wired + CORRECT (5/5 tokens==f32) but end-to-end SLOWER 20.6 vs f32 25.8 (PR#48): Q8 decode LAUNCH-BOUND-MASKED — GPU-bandwidth win useless while host-dispatch bound; driver-API kernel adds launch cost. ⇒ CUDA GRAPHS is the PREREQUISITE to unlock Q8 (+ suspect small-N GQA warp underutil). BREAKTHROUGH (PR#51): decode was ALLOC-BOUND (≈250 cudaMallocAsync/FreeAsync/token, esp. GROWING scores buf) as much as launch-bound. Fixed persistent buffers + device-pos + fixed-size padded attention (graphDecoder) = 66.4 tok/s vs 25.9 regular = 2.56×, correct 5/5. Gap to llama.cpp 243 now 3.7× (was 9.5×). Fixed-buffer decode now 68.6 (persistent RoPE inv, PR#52). GRAPH CAPTURE BLOCKED: nvrtc kernels capture fine (GELU POC), but full decode graph DIVERGES deterministically (cuBLAS suspect, unconfirmed; tried ThreadLocal/Global/workspace/device-ptr-mode). cuBLAS now capture-ready (device α/β + workspace). GRAPH DECODE WORKS (PR#53): root cause was GQA host-built cublasBatched pointer arrays memcpy'd to device — shared host source overwritten during capture → all layers read last layer's ptrs. Fixed with cu_build_batch_ptrs (device-built). Graph 70.6 vs fixed-buffer 68.98 = +2.4% only — because fixed-buffer already killed alloc-bound → decode now GPU-MEMORY-bound (4.4GB f32/token). So Q8 (4× bandwidth) is the BIG remaining win, now UNMASKED. ★ PAYOFF (PR#54): Q8-on-fixed-buffer/graph decode = 161.6 tok/s (2.29× over f32 graph 70.6), correct 5/5. Decode arc 26→70→161.6 = 6.2×. GAP TO llama.cpp 243 = 1.5× (was 9.5×). Q8 finally shone because decode was memory-bound. Q4 REJECTED (PR#55): Q4_0 correct (L1 6.5%=16×Q8) but decode 156 ≤ Q8 161.6 (NO win — decode no longer weight-bandwidth-bound after Q8) + breaks TinyLlama 0/5. New bottleneck post-Q8 = Out proj[dim×32000] + f32 KV reads + per-token overhead (128KB logit D2H+sync). On-device argmax DONE (PR#56, +2%→164.7): 128KB logit download was NOT the bottleneck. Decode now GPU-BOUND (Q8 weight bw + attention), gap to llama.cpp 243 = 1.48×. CUDA decode arc largely MAXED for TinyLlama — further needs Q4_K (Q4_0 too lossy) or flash attention (diminishing returns). goai competitive. GENERATE DEMO DONE (PR#57): TinyLlama on GPU writes 'Paris, which is the capital of France.' from 'The capital of France is' — full pipeline tokenizer→Q8 GPU decode→argmax→detokenize, 164 tok/s, 1.48× off llama.cpp. CUDA inference arc COMPLETE + demonstrated. NEXT DIRECTIONS: flash-style attention; larger models (Qwen needs QKV bias); sampling/temperature; or pivot to other goai work per LOOP.md.
**NEXT lever = cut per-op overhead** (persistent GQA scratch, fewer host syncs, kernel fusion) — the decode bottleneck; THEN quant matmul (bandwidth, once compute-bound); THEN fusion for prefill scaling.

**How to apply:**
- Metrics: prefill tok/s AND decode tok/s, same model/hardware, warm-up excluded, medians (§V22). Report vs llama.cpp `llama-bench` on the same GGUF + RTX 3060. NB llama-bench pp variance is high on first sample — run -r 3+ and take stable run.
- **Honest gap FIRST**: goai runs F32 (dequantized), NO KV cache (re-forwards each step, O(N²) decode) → currently FAR slower than llama.cpp's quantized+KV-cache+fused kernels. Measure it honestly, then optimize.
- **Hyper-optimize levers (biggest first):** (1) KV-cache decode on-device (RoPE PosOffset; the O(N²)→O(N) decode win) — HUGE. (2) quantized matmul on device (Q8/Q4 resident weights → 4× memory, quantized GEMM) — mirrors llama.cpp. (3) kernel fusion (RMSNorm+matmul, attention flash-style) + fewer launches. (4) pooled intermediates. (5) larger batch/prefill.
- **llama.cpp CUDA build needs nvcc** (pip wheels have only ptxas) — investigate: NVIDIA runfile user-install, or a prebuilt CUDA binary + the pip CUDA runtime libs. CPU llama.cpp is the easy fallback baseline.

See [[user-directives-cuda-bottomup]], [[linux-amd64-worker-role]].
