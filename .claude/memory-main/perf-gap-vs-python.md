---
name: perf-gap-vs-python
description: GoAI GPU-perf state after the 2026-07-12 session (T335–T343) — remaining gaps vs torch, which levers worked, which are parked.
metadata: 
  node_type: memory
  type: project
  originSessionId: 56de00c6-6c80-4734-a46b-f5e03083b2b4
---

As of 2026-07-12 (session T335–T343) GoAI's GPU ops are correct AND the
structural per-call overhead is fixed. Current state vs Python stacks (M2 Pro):

- matmul@1024³: GoAI-metal ~1186 GFLOP/s (torch-mps 4200, ~3.5×); GoAI-vulkan ~594.
- attention fwd 512×8×64: GoAI metal ~11 ms via flash kernel (torch-mps 0.45 ms, ~24×).
  **§T393 §V22 SURPRISE: the gap is NOT structural / does NOT need cooperative tiling.** Measured
  the MATMUL-ONLY floor (the 16 MPS matmuls attention needs: 8× Q·Kᵀ + 8× P·V, no softmax) at
  **0.71 ms** — the hand-written flash kernel is 15× SLOWER than just the two matmuls via MPS. So the
  lever is REFORMULATE attention as MPS(Q·Kᵀ)→softmax→MPS(P·V) (unfused, materializes the [seq,seq]
  score matrix — fine on UMA), NOT more flash-kernel tiling. Expected ~7-10× on attn fwd, landing
  near torch. The old "needs cooperative tiling" note here was WRONG — a §V22 measure-the-floor caught
  it before a wasted rewrite.
  **§T394 IMPLEMENTED IT: 6.9× win.** mtl_mha_mps: per-head S=scale·Q_h·K_hᵀ (MPS matmul,
  transposeRight, strided MPSMatrix views — column offset h·dk, rowBytes=full row) → causal
  softmax_causal kernel (256-thread/row, masks j>i, zeros the tail so S·V ignores it) → O_h=S·V_h
  (MPS), one [seq,seq] scratch reused across heads (hazard tracking orders it). MEASURED 512×8×64
  causal: flash 10.77ms → MPS 1.55ms = 6.9×, now only ~3.4× vs torch (was ~24×). Cross-validated
  @1e-4. Supports GQA.
  **§T395 WIRED into OpMHA forward (the sq==sk && Window==0 prefill/training branch, replacing the
  flash kernel) → real GPT forward 1.87×.** Same-session A/B (BenchmarkGPTForward/metal): flash
  4241 tok/s (60.4ms) → MPS 7916 tok/s (32.3ms). So attention was ~28ms of the 60ms GPT-fwd wall
  time; the 6.9× attn win → 1.87× whole-forward. All OpMHA cross-ref tests (mha/causal/gqa/mqa/
  kvcache/attnscale) pass through the new path. Decode (sq<sk) + sliding-window stay on mtl_mha_f32.
  This CORRECTS the old "attention is ~1% of GPT fwd" note — post-GELU/AddBias-fix, the SLOW flash
  kernel made attention ~half the wall time.
  **§T399 did the same for the BACKWARD → metal training step 2.04×.** §V22 measured: metal mha
  BACKWARD was 70-75ms (54× the 1.3ms fwd!) — the training-step bottleneck (bwd atomic-bound, old
  §B44). mtl_mha_backward_mps reformulates it as MPS matmuls (dV=Pᵀ·dO, dP=dO·Vᵀ, dS=softmax_jacobian
  (P,dP), dQ=scale·dS·K, dK=scale·dSᵀ·Q; recompute S/P; strided MPSMatrix head views + softmax_causal +
  new softmax_jacobian kernel, one cmd buffer) → 27.3× (75→2.75ms @512×8×64), cross-validated vs the
  ref-validated flash-backward @1e-3. Wired into OpMHABackward (kvHeads==heads && Window==0). Training
  step (BenchmarkGPTTrainingStep/metal): 1133→2315 tok/s = 2.04×. GQA + sliding-window stay on the
  atomic flash-backward. THE MPS-matmul reformulation is the KEY metal attention lever: 6.9× fwd /
  1.87× GPT-fwd (§T394/5) AND 27.3× bwd / 2.04× training (§T399).
- conv2d ResNet-shape: GoAI-metal 593 GFLOP/s / vulkan 367 (torch-mps 1180, ~2-3×).
  **§T417 PARKED with numbers:** floor-measured (conv N8C16HW32F32K3: total 0.855ms, GEMM-only 0.283ms
  of which ~0.27ms is the dispatch floor) — even a perfect fused MPSCNN/MPSGraph conv can't beat
  dispatch+transfer at small shapes (ceiling ~1.7×), large shapes ~2×; an MPSCNN rewrite = MPSImage
  NCHW conversion + big API surface; and conv is NOT on the LLM critical path (GPT/Llama use none).
  Don't chase unless the library pivots to CV.
- CPU GEMM unchanged (~70 GFLOP/s, amd64-SIMD host-blocked, §T11b/§T74).

What WORKED (all A/B-measured, §T rows have numbers): buffer pools both backends
(T335/T336); Metal threadgroup sizing — never `maxTotalThreadsPerThreadgroup` for
per-row or register-heavy kernels, use 64/TG (T337/T339, 2-3× each); OpMHA→flash
routing (T340, 1.59×); conv im2col-GEMM lowering (T341/T342); single cmd-buffer +
barriers for multi-stage vulkan ops (T343). What did NOT work (don't retry on this
host): Vulkan GEMM register/cache blocking (§B39/§B41 — MoltenVK/M-series is
bandwidth-bound).

**BIGGEST LESSON (§T352, §V22): profile the REAL workload before optimizing.** The whole
session's per-kernel GPU work (cooperative norms/softmax T335-347, flash tiling T349) gave the
real GPT forward ~nothing — those ops are ~1% of it. The actual bottleneck was TWO ops silently
falling back to the reference CPU backend: exact-erf GELU (11ms/op!) and AddBias (4.4ms/op), which
had no metal/vulkan case. Fixing them (GELU via Abramowitz-Stegun erf approx since MSL/GLSL lack
erf; AddBias kernel; both backends) gave GPT forward 1264→4227 tok/s (3.3×), metal now 23× vs cpu.
ALWAYS run the §T350/§T351 GPT throughput benchmark + per-op probes to find the real bottleneck.
AUDIT for more silent ref-fallbacks (check every op the GPT fwd/bwd touches).
**§T401 FOUND ANOTHER BIG ONE: OpCrossEntropy FORWARD fell back to CPU = 20ms (metal had the
BACKWARD kernel but NOT the forward).** §V22 audit: timed the ops the GPT forward touches on a metal
ctx — OpEmbed 1ms + OpCrossEntropy 20ms fell to ref (unregistered). Implemented mtl_crossentropy_f32
(cooperative lse+NLL, one 256-thread tg per [seq,vocab] row; basic case, label-smoothing/z-loss stay
on ref) + registered OpCrossEntropy on metal → 20.05→0.85ms (23.6×), cross-validated @1e-4. NOTE: the
loss is in the TRAINING step, NOT GPT.Forward (which only makes logits) — so it sped the TRAINING
step, not the fwd bench. Training step 2315→2924 tok/s (1.26×). ATTENTION+CE LEVERS COMBINED: metal
training step 1133 (pre-§T399) → 2924 tok/s = 2.58× (mha_bwd §T399/400 + CE fwd §T401). OpEmbed fwd
(1ms) left on ref — marginal. HOW TO AUDIT (refined §T402): standalone op-timing MISLEADS — OpSum [256,4096]→ax0 timed 19.57ms
standalone but is NOT on the GPT training path (§V22 caught it before a wasted kernel). The DEFINITIVE
audit is INSTRUMENTING the real workload's dispatch: add an env-gated log in backend/execute.go's
fallback branch printing `ctx.Backend.Name()` + op + shape, run BenchmarkGPTTrainingStep, filter to
FALLBACK[metal]. Result: the metal training step's ONLY fallback was OpEmbed (fwd, ~1-2ms) — the
layernorm/mha/crossentropy "fallbacks" in the unfiltered log were the CPU pure-Go backend (the bench
runs both). §T402 implemented mtl_embed_f32 (row gather, one thread/elem) + registered OpEmbed →
the metal GPT training step is now FULLY FALLBACK-FREE. So the fallback-hunting lever is EXHAUSTED
for metal training: its ~87ms is now legitimate GPU compute (matmuls/norms/attention). Remaining gap
vs torch is the MPS matmul rate (~3.5×), which is Apple's best — hard to beat.
§T403: brought the SAME two forward kernels to VULKAN (parity, [[gpu-ops-all-backends]]) — vulkan
OpCrossEntropy + OpEmbed were also silent CPU fallbacks (backwards registered, forwards not).
NEW crossentropy.comp + embed.comp (GLSL mirrors of the MSL) + vk_crossentropy_f32/vk_embed_f32
+ Go crossentropyF32/embedF32 registered. Cross-validated @1e-4/exact. Vulkan training step
865.7→937.5 tok/s = 1.08× (smaller than metal's 1.26× because vulkan's overall step is slower, so the
fixed ~22ms fallback is a smaller fraction). Both backends' GPT training are now fallback-clean for
the fwd loss+embed. AUDIT PATTERN for any backend: forwards can be missing while backwards exist —
check both directions per op.
§T404 DECODE audit + §C3 proof: instrumented BenchmarkGPTDecode → metal decode is FALLBACK-CLEAN but
DISPATCH-BOUND: metal 52 tok/s < cpu 125 tok/s (per-op commit/wait dominates a 1-token step). THE FIX
is ADR-0019's batched recorder, and §T404 measured its REAL win: internal/gpudecode
TestLlamaDecodeBatchedVsPerOpThroughput drives a REAL nlp.Llama (D512/GQA8:2/6-layer/V1024) via the
batched recorder vs the library's own m.DecodeStep on the SAME metal backend → batched 179.3 tok/s
vs per-op 7.5 tok/s = **23.97×** (far > the T389 3.45× batched-vs-per-op-RECORDER microbench, because
nlp's per-op DecodeStep ALSO pays tensor-alloc + full Execute dispatch per op). The batched 179 tok/s
also BEATS cpu decode — fixing the metal<cpu problem. Kept a permanent env-gated
fallback logger in backend/execute.go (GOAI_LOG_FALLBACK=1) for future audits.
PRODUCTIONIZED (§T405-409): internal/llamagpu — a BACKEND-AGNOSTIC batched Decoder (core over small
buffer/recorder interfaces in decoder.go; thin metal/vulkan adapters assert back to the concrete
types; both backends' exported Recorder/DeviceBuffer APIs are identical by construction §T391/T408).
API: New(m)/NewVulkan(m) → Decoder; Step(token,pos)→logits; Generate(prompt,maxNew,sampler)→tokens;
Release(). PROVEN: greedy Generate == nlp.Llama.Generate token-for-token on BOTH backends; GGUF(F32)
models work (§T407); real-model decode D512/GQA8:2/6-layer: metal batched 179 vs per-op 7.5 tok/s =
23.97×; vulkan batched 155 vs 7.5 = 20.7×. GOTCHA: apicheck §C15 bans magic backend-name strings —
use string(backend.Metal) etc. Remaining polish: public promotion (Example+docs §V19), on-device
sampling/argmax to skip the logits download per step.
§T410 OP-PROFILE of the metal training step (NEW permanent GOAI_TIME_OPS=1 timer in
backend/execute.go, sibling of GOAI_LOG_FALLBACK; aggregate on $NF — shapes contain spaces):
matmul 49% (MPS-bound), small elementwise ops at the ~0.27ms dispatch floor (~20%), attention 30%.
REGRESSION caught+fixed: the §T402/T403 GPU embed uploaded the whole 8MB table PER CALL (2.47ms) —
slower than the ref fallback it replaced. Fixed to a direct f32 HOST gather on both backends (~50µs;
a resident-table cache would be UNSAFE — training mutates the table). Training 2924→2997 tok/s.
LESSON: never GPU-ify tiny memory-bound ops (gather/scatter of KBs); re-profile the REAL workload
after each kernel lands.
§T411 CLOSED the tape-batching question: measured the batching ceiling AT THE TRAINING SHAPE
(TestLayerBatchBench "train D512 S256"): only **1.40×** (vs decode-S1 5.94×) — at seq=256 matmuls are
compute-heavy so the dispatch floor is minor, and the backward is heavier still. Recorder-izing the
autograd tape (big project) buys ≤~1.4× → PARKED. Metal training is now FULLY characterized: 49%
matmul @MPS rate (Apple's best, don't retry), ~30% attention (already MPS-reformulated), ~20%
dispatch floor (parked 1.4×). Training levers left: beat MPS GEMM (hard) or bigger batches. The
HIGH-ROI direction remains INFERENCE (decode batching gave 24×) — e.g. quantized decode or larger
models. §T412: llamagpu is now PUBLIC (top-level package, moved from internal/): New/NewVulkan from
any *nlp.Llama (incl. LlamaFromGGUF) → Generate/Step/Release; tag-free core builds under CGO0;
Example guarded by metal.Available(). The whole ADR-0019 line (design→primitives→proof→public API)
is COMPLETE. §T413-415 DID the quantized batched decode: recorder QMatMulResident on both backends (record-mode
qmatmul over §T156 resident qweights, all ggml types) + llamagpu.NewQuant/NewQuantVulkan(m
*nlp.QuantLlama). KEY discovery: nlp ALREADY had the whole quantized-model stack (QuantLlama,
nn.QuantLinear with exported ggml bytes, QuantLlamaFromGGUF) — always scope-check before building.
The Decoder core got a `linear` interface (f32Linear=MatMul / quantLinear=QMatMulResident) so ONE
core serves both. Verified: quant decoder logits == QuantLlama.DecodeStep @3e-2 + greedy Generate
token-for-token. §T416 MEASURED (D512/GQA8:2/6-layer Q8_0): batched-quant 160.7 tok/s vs QuantLlama
per-op(metal) 45.1 = 3.6× (NOT ~20× — the per-op quant path was already 6× faster than per-op f32
thanks to §T154 lazy resident weights skipping re-uploads); batched-f32 191.4 — quant is ~16% SLOWER
per token (dequant kernels are one-thread-per-output, not MPS-class). SO: Q8's value = 4× smaller
weights (bigger models fit), NOT throughput. On-device sampling SKIPPED by arithmetic (§T418: the
logits download is 4-512KB on UMA, <4% of a step).
§T418 StepN: multi-token batched step → PREFILL 40.9× (64-tok prompt D512/6-layer: 12.7ms vs 519ms;
5040 vs 123 tok/s). Generate prefills via ONE StepN now. Enablers: scratch sized to maxLen rows,
recorder guards relaxed ==→>= (over-allocated operands OK), DownloadF32 = prefix download (both
backends). MHA sq=k/sk=pos+k causal gives exact prefill/verify semantics (off=sk-sq). §T419 COMPOSED speculative: llamagpu.SpeculativeGenerate(target, draft, prompt, maxNew, lookahead,
s, seed) — draft proposes k-1 via Step (q_i via s.Dist), target verifies the window in ONE StepN,
nlp.SpeculativeRun accepts/rejects (lossless). Key design: lastTok-lead-window (carry token leads the
next verify window → ps[0] comes from the StepN itself); KV rollback FREE (position-addressed caches
— next window overwrites rejected rows); one extra draft.Step keeps draft KV complete on full accept.
Verified at BOTH acceptance extremes token-for-token == plain Generate (different draft → 0%, self
draft → 100%). Batched-decode line COMPLETE: 24× decode, 41× prefill, quant memory, speculative.
For a REAL speculative speedup measurement use a related draft; cost structure measured (§T420):
t_draft/t_target=0.18 → 1.95×@50% / 2.65×@80% acceptance.
§T434 MEASURED IT with REAL trained models (no downloads — trained IN-REPO: char-grammar corpus,
target dim96/3L + draft dim48/1L, 240 AdamW steps each on metal, ~11s total;
llamagpu/speculative_trained_test.go). Result: acceptance 81%, losslessness token-for-token — but
end-to-end speedup only 1.12× (1150→1293 tok/s). WHY: at tiny scale both decoders are
DISPATCH-BOUND (step cost ∝ #recorded ops, not compute) → 1-layer draft only ~3× cheaper than
3-layer target → k·t_draft+t_verify eats the win. §V22 lesson: high acceptance alone is NOT
sufficient; speculative pays off only when the target is compute-bound (large) so t_draft/t_target
is small. The §T420 cost model stays valid for that regime. Trained-in-repo models are the pattern
for any future "needs model files" experiment.
§T446 CLOSED that gap with MEDUSA on the batched decoder (llamagpu.MedusaGenerateGPT +
GPTDecoder.StepHidden): heads draft HOST-side for free (no draft decoder steps — the exact cost
§T434 diagnosed), one StepN verifies the window → measured 97% acceptance, 1120→2025 tok/s =
1.81× on the SAME dispatch-bound setup where draft-model speculative got 1.12×. Lesson pair:
dispatch-bound decode → speculative only pays when drafting is ~free (Medusa/prompt-lookup);
draft-model speculative needs compute-bound targets.
§T452 completed the trio: prompt-lookup on the trained base = 1.80× LOSSLESS at only 15%
acceptance — its round is ONE StepN (vs Medusa's ~2 steps), so a cheap round beats high
acceptance; needs repetitive output. MEASUREMENT GOTCHA: sequential A/B blocks hand the
first scheme any cold/thermal outlier (a spurious 4.57× until interleaved A,B,A,B medians).
§T455 then applied that round-cost insight BACK to Medusa: StepNHidden (verify pass also
returns hidden rows) lets heads draft the NEXT window from the CURRENT verification
(lastTok-lead-window §T419) → 1 step/round → 1.81× became 3.08× (1152→3546 tok/s, 97% acc).
Round cost is EVERYTHING on dispatch-bound decoders; fold work into existing passes.
§T422-424 added GPT: GPTDecoder (NewGPT/NewGPTVulkan; LayerNorm+learned-pos-emb+biased-GELU;
recorder gained LayerNorm+AddBias record ops on both backends), GPT StepN (prefill parity),
SpeculativeGenerate generalized over the exported Stepper interface {Step,StepN,Vocab,Ctx} → works
for both architectures. FINAL MATRIX: {Llama,GPT}×{Step,StepN,Generate,Speculative}+Llama×Quant,
all public+documented+cross-validated on metal+vulkan. GOTCHA: apicheck demands an Example per
exported type — parameter-constraint interfaces go in typeExampleExempt instead.
§T428 LONG-CONTEXT decode was ~SERIAL: the two-pass mha kernel = ONE THREAD per (head,query) — at
sq=1 that's `heads` threads total, step @L=1920 took 242ms (vs <1ms floor). FIX: mha_decode_f32 —
one SIMDGROUP/head, 32 lanes partition keys, online-softmax partials (m,l,acc[dk]) in registers
merged via simd_shuffle_down tree; routed in mtl_recorder_mha at sq==1 → 242→13.8ms (17.6×).
TWO BUG LESSONS: (1) merging two EMPTY lanes gives exp(-INF−(-INF))=NaN — guard M==-INF in any
online-softmax simd merge; (2) logits tests were NaN-BLIND (NaN>tol==false → silently passes) —
always `math.IsNaN(got) || diff > tol`; token-equality tests are the NaN-safe backstop. Q8-KV-cache
PARKED: the kernel was the real lever. §T429 did the VULKAN parity: mha_decode.comp — one 32-lane subgroup/head, subgroupShuffleDown merge
(needs SPIR-V 1.3: glslc --target-env=vulkan1.1; the instance already requests VK 1.1), NaN guard
built in; routed in Go inside Recorder.MHA (vulkan passes spv as params). @pos1920: 15.2ms (≈metal's
13.8). GOTCHA: the make vulkan-spv rule silently missed 5 newer shaders (committed .spv artifacts
masked it) — keep the rule in sync when adding shaders. Long-context decode healthy on BOTH backends.
§T431: PREFILL windows (sq>1 vs long cache) had the same serial cliff (291ms for a 128-window
@pos1792) — mha_decode GENERALIZED to sq>1 (grid (heads,sq), per-row causal jmax=sk-sq+i+1), routing
widened to (causal||sq==1)&&window==0&&dk≤128, both backends → 104/109ms (2.8×); @pos0 windows also
2.2× (the recorder never had a flash route). RESIDUAL: sq-fold redundant K/V reads (~800% growth
with depth) — the flash-tiling problem; candidates: MPS sq≠sk reformulation (off-aware causal
softmax) or threadgroup K/V tiles. Build only if long-prompt prefill measures as a pain point
(full 1920-prefill now ~0.9s, was 2.4s).
§T432 completed the fix family: the PER-OP path (nlp DecodeStep etc.) also routed sq<sk through the
serial two-pass kernel — added mtl_mha_decode_host / vk_mha_decode_f32 (host-slice wrappers of the
cooperative kernel) + routing in both backends' mhaF32 at (causal||sq==1)&&window==0&&dk≤128.
Per-op OpMHA sq=1 sk=1920: 2.18ms (~18× vs the serial profile). HONEST: short-context per-op decode
UNCHANGED (51.9 tok/s — dispatch-bound, attention wasn't its bottleneck there). Cooperative attention
now covers ALL surfaces: recorder+per-op × metal+vulkan.

**BATCHING CEILING MEASURED (§T367–T374, ADR-0019 Phase 2):** the #1 lever (fewer per-op
round-trips) is now de-risked AND measured. Built device-resident buffers (T367) + a batch
"recorder" (one open command buffer) with record-mode variants of ALL four decode op-classes:
elementwise/binary (T370/T374), matmul via MPS (T371), rmsnorm (T372), flash-attention (T373).
Then A/B'd a FULL 12-op GPT layer per-op vs one-command-buffer (TestLayerBatchBench): **decode
sizes win ~2.8–3.5×** (D512·S1 3.24→0.92ms), prefill (compute-bound) only 1.55×. CRITICAL
CORRECTION: the earlier §T368 "41.8×" was a trivial unary kernel with ~0 compute — the HONEST
decode ceiling is ~3×, not 41×. Per-op ≈270µs/op (commit/wait round-trip dominates).

**INTEGRATION DONE + VALIDATED (§T375–T379):** built the whole batched decode over persistent
device KV caches and PROVED it. Missing primitives added: decode-attention sq≠sk (T375 — reused
the existing gMHA kernel which already separates query/key len), record-mode RoPE (T376), blit for
KV-append (T377), DeviceBuffer.UploadF32 for per-step buffer reuse (T378). Assembled a single
attn-block step (T378) then a FULL multi-layer decoder — per-layer attn+MLP over per-layer caches
+ final-norm + vocab head, 98 ops/step at 6 layers, ONE recorder/step (T379). CORRECTNESS: GPU
recorder logits == host float64 (full stack: growing-cache attention, MLP+GELU, vocab) @2e-3 across
multiple steps. MEASURED at REALISTIC size D512·H8·6-layer·V1024: batched 7.73ms vs per-op 23.98ms
= **3.10×**, matching the T374 ceiling exactly (the T378 9.36× was a tiny D=32 dispatch-bound model
— win narrows with compute density, §V22 confirmed). All recorder ops UNEXPORTED in package metal.

**VULKAN PARITY DONE (§T380–T385):** the whole recorder is now ported to Vulkan and the batched
decode PROVEN on both backends. Vulkan needed platform-specific handling Metal didn't: EXPLICIT
barriers (no auto hazard tracking — one broad compute+transfer VkMemoryBarrier between ops) and a
DEDICATED descriptor pool giving each recorded op its OWN set (a reused set binds only its LAST
write at submit). C: vk_devbuf_* (host-visible VkBuffer, §T381) + vk_recorder_begin/unary/binary/
matmul/rmsnorm/rope/mha/blit/finish (§T382-384; matmul uses the tiled matmul.spv — Vulkan has NO
MPS; blit = vkCmdCopyBuffer for KV-append). Go DeviceBuffer + recorder mirror the Metal API exactly.
Full multi-layer decode (§T385): recorder logits == host float64 @2e-3 (D=64, incl MLP/gelu/vocab);
A/B at D512·H8·6-layer·V1024: per-op 47.17ms vs batched 8.34ms = **5.66×** — HIGHER than Metal's
3.10× because Vulkan/MoltenVK's per-op submit+waitIdle+descriptor-rewrite floor (47ms) is worse than
Metal's (24ms) while batched is comparable (8.3 vs 7.7ms) → batching recovers MORE on Vulkan.
Recorder OP+ARCH coverage COMPLETE (§T386–T390): embed-via-blit closes the on-device
token→logits→token loop (embed = blit a row of the resident embedding table; NO new kernel, real
autoregressive loop verified). LayerNorm added (§T387) so BOTH norm styles record. SwiGLU FFN
(§T388, matmul+SiLU+mul+matmul, no new op). FULL both-ARCHITECTURE assemblies verified+measured on
BOTH backends: GPT-2 (LayerNorm+GELU-MLP) metal 3.10× / vulkan 5.66×; Llama (RMSNorm+RoPE+GQA+SwiGLU,
kvHeads<heads with kvDim-wide caches, query head h→kv head h/rep) metal 3.45× / vulkan 6.15×. Vulkan
speedups are HIGHER because MoltenVK's per-op floor is worse; ALL these are CONSERVATIVE lower bounds
— the per-op baseline is device-resident, but the real library's per-op path ALSO pays host up/down
memcpy per op, so the true end-to-end win is larger. nlp.Llama maps onto the recorder DIRECTLY
(project=x·W weight [in,out] → no transpose; RMSNorm; RoPE; SwiGLU) — verified by inspection.
§C3 WIRE-IN DONE (§T391 export, §T392 proof): the Metal recorder is now EXPORTED (Recorder,
NewRecorder, MatMul/RMSNorm/LayerNorm/RoPE/MHA/Unary/Binary/Blit/Finish/Free + DeviceBuffer;
apicheck-exempt backend subpkg §V19). internal/gpudecode test drives a REAL nlp.Llama (GQA 8:2,
SwiGLU, 2 layers) decode through the recorder over its F64→F32 weights and confirms the logits ==
the model's own per-op DecodeStep on the reference backend across 5 tokens @3e-2 (f32 vs f64). KEY:
RoPE freqs come from backend.RoPEFreqs(dk, RoPEAttrs{Base}) — the SAME the per-op OpRoPE uses — so
conventions match; weight orientation is x·W [in,out] (no transpose); this validated the hand-written
host references were correct. So the batched decode BOTH reproduces the real model AND is 3.45×
(metal) / 6.15× (vulkan) faster — the ADR-0019 payoff is proven end-to-end on a real model.
REMAINING (optional polish): export the vulkan recorder too, on-device logit sampling, a public
DecodeSession convenience wrapper. Recorder is NOT concurrency-safe (one per goroutine).

**How to apply:** next levers: (1) FEWER per-op GPU round-trips — batch ops into one
command buffer + barriers (§T343 conv pattern) or a persistent encoder/graph. **VULKAN attention (§T396/T397): isolated win, real-workload LOSS → reverted.** §T396 measured the
matmul-only floor (1.66ms) « vulkan flash (12ms) → looked promising. §T397 built it: matmul_strided.comp
(strided/offset GEMM over packed Q/K/V windows) + softmax_causal.comp + recorder ops matmulStrided/
softmaxCausal + mhaMatmul (per head: strided QKᵀ→causal softmax→strided PV in one recorder cmd buffer).
Cross-validated mha/gqa/mqa @1e-4; ISOLATED 512×8×64: flash 12→matmul 2.33ms = 5.2×. BUT the real
GPT forward got SLOWER (§V22 A/B: flash 3191→matmul 2785 tok/s) → REVERTED the forward wiring (flash
stays; mhaMatmul kept+tested for later). WHY: mhaMatmul does Go-side per-CALL device-buffer alloc
(5× NewDeviceBufferF32 = vkCreateBuffer+vkAllocateMemory + download) + per-head multi-dispatch — that
overhead beats the matmul win at the forward's seq. Metal WON (§T395 1.87×) because mtl_mha_mps is a
self-contained C fn over its own MTLBuffers (cheap on UMA), one call. §T398 tried the fix: a package-level buffer CACHE (reuse the 5 device buffers across calls, keyed by
shape) → mhaMatmul isolated 8.2× @512, 6.0× @256 (the forward's seq). But the REAL forward STILL
didn't improve (3191→2957 tok/s). ROOT CAUSE REFINED: on VULKAN, attention is NOT the GPT-forward
bottleneck (unlike metal, where the fast MPS FFN matmuls left attention ~half the wall time). Vulkan's
tiled-GEMM FFN matmuls (~300-600 GFLOP/s vs metal MPS ~1186) DOMINATE the forward, so a 6× faster
attention saves little and the per-head multi-dispatch/up-download overhead eats it. REVERTED the
wiring (flash stays); mhaMatmul+cache+shaders kept+tested (ready for large-seq or after the FFN matmul
is sped up). So the real VULKAN lever is the MATMUL (~3.5× vs torch), NOT attention. LESSON
(again, like §T348/B42): an isolated micro-bench win ≠ a real-workload win; A/B the REAL forward AND
identify the actual bottleneck (attention was metal's, matmul is vulkan's).
Also fixed a latent bug: adding 2 shaders pushed the embedded count to 33 > VK_MAX_PIPELINES=32 →
pipeline_for -4 → a random later shader (embed_bwd) failed ONLY in the full suite; bumped to 48.

This is
the REAL floor for the norm/softmax family: §T348/§B42 built zero-copy UMA end-to-end,
measured it (same-session A/B @2048²: copy 1.75ms vs zero-copy 1.73ms = NO delta), and
REVERTED it — the memcpy is NOT the bottleneck; per-op dispatch/waitUntilCompleted
latency is. Do NOT re-attempt zero-copy without evidence an op is copy-bound. (2)
cooperative tiled attention (threadgroup-shared K/V, simdgroup ops, both backends). (3)
MPSGraph conv. LESSON (§V22): a "floor is X" claim must be A/B-measured (force the
suspected cost off) before recording it or building on it — the old "memcpy floor" note
here was unverified and wrong. MoltenVK micro-benchmarks are THERMALLY NOISY →
same-session A/B medians only. A/B baselines: NO `git stash` (repo = 1 commit, reverts to
scaffold); temp-swap old code from a scratchpad backup (§T336/§T341). Measure with
`make bench-compare` / `make bench-python`. See [[gpu-ops-all-backends]].

**Update 2026-07-13 (§T511/§T524):** cpu parallelWork hat jetzt einen persistenten
Worker-Pool (Allocs −70–75%/Op, Zeit ±0–2% — der Gewinn ist GC-Druck, nicht Latenz).
Bench-Regressions-Check §T524: alle §T335-Snapshot-Zeilen halten oder besser
(FlashAttn metal 10,8ms, MatMul@1024 metal 1645 GFLOP/s, vulkan 582). A/B-Baselines
per Datei-Toggle aus Scratchpad-Backup — NIE git stash (Regel oben, §T515 erneut
bestätigt).


**Update 2026-07-13 (§T528–§T532):** Vulkan-Attention beidseitig matmul-dekomponiert
(Backward-Kette 71,5→4,7ms = 15×; Forward-Kette +18% am Trainingsschritt, Default für
sq==sk/no-window — §T398s Ablehnung galt der teureren Alt-Struktur und ist für diese
Form überholt). Vulkan-GPT-Training 935→1882 tok/s = 2,01× (metal-Klasse). Profil
danach: matmul 51% — aber §B39: Register-Blocking bereits versucht, bandbreiten-
gebunden auf MoltenVK → GEMM-Decke, Arc GESCHLOSSEN. Neuer stehender Bench:
BenchmarkGPTTrainingStepVK (backend/vulkan, vulkan-tagged).

**At-Scale-Zahlen 2026-07-13 (§T543–§T546, synthetische 124M-Modelle, Batched-Decode):**
GPT-2-Form: metal 51 / vulkan 59 tok/s (Toy-Ordnung invertiert bei d=768!).
Llama-Klasse: metal f32 76 / Q8 57 / Q4_K 66; vulkan f32 62 / Q8 71 / **Q4_K 72**.
ZWEI Regeln: (1) Q4_K > Q8_0 auf beiden Backends bei d≥768 (Gewichtsverkehr dominiert);
(2) Quant-vs-f32 INVERTIERT je Backend — metal MPS-f32 schlägt Quant, vulkan
(bandbreiten-gebunden) belohnt Quant. Praxis: metal → f32, vulkan → Q4_K.
Harness bereit für echte GPT-2-Gewichte (synthGPT2HF austauschen, §T543).

**Update 2026-07-14 — der große Perf-Sprint (§T620-T623, alle same-machine-A/B, CI-grün):**
- CONV war NICHT am Ceiling (§T417 war zu pessimistisch): **native MPSGraph conv2D** (nicht MPSCNN)
  schlägt im2col+MPSMatrixMultiplication **2,35×** @n8c64hw56 → nur noch ~1,3× hinter torch-mps.
  Vulkan hat kein MPS → fused Implicit-GEMM `conv_igemm.comp` (X-Gather im Tile-Load, +20%) +
  vec4-Coalescing (+7%). Metal-hand-tiled-fused verlor 1,7× (MPS-GEMM unschlagbar von Hand) → nur
  Apples getunte Primitive gewinnt bei COMPUTE-bound Ops.
- MPSGraph-Cache muss SHAPE-KEYED sein (§T622): single-last-shape-Cache thrasht bei Multi-Layer-CNNs
  (jede Schicht recompiliert, 15-30ms) → 16-Slot-FIFO, Multi-Shape 7,2×.
- ATTENTION: MPSGraph-batched-prefill schlägt den per-head mtl_mha_mps **+26%** (§T621) — aber
  15-30ms-Compile/Shape → OPT-IN (SetMHAUseMPSGraph), Default bleibt custom für arbitrary-length.
- MATMUL DEFINITIV am MPS-Ceiling (§T623, nicht wieder versuchen): MPSGraph-matmul == MPSMatrixMult
  BIT-IDENTISCH (gleicher Apple-Kernel); fp16 verliert (un-offloaded Host-f32↔f16-Konversion). Der
  standalone-Bench-„3×-Gap" ist der HOST-ROUND-TRIP (~12MB memcpy/Call ≈ halbe Zeit), NICHT der
  Kernel — reale Workloads meiden ihn via Recorder (§T369/T404-432, GPT-fwd 8396 tok/s kompetitiv).
- CPU-Devirt-Class-Audit (§T602-Muster) auf ALLE heißen Kernel: f64at/f64set-Closures → generische
  []T-Cores. RMSNorm 3,5× / LayerNorm 3,7× / Softmax 1,33× / Retention fwd 3,85× / bwd 4,25×.
  Größter Win wo die inneren Schleifen closure-lastig waren (retention). CPU-Closure-Audit KOMPLETT.
- METHODIK-Bestätigung: Same-Machine-INTERLEAVED-A/B (nicht cold/warm) gegen Thermaldrift; B55 drückt
  Absolutwerte ~2× → nur relativer Delta zählt. Sub-Delegation an worktree-Agenten funktioniert MIT
  `git merge main` sync-first (sonst stale Basis, [[worktree-agent-stale-base]]).

**Update 2026-07-15 — „besser als pytorch"-Kampagne, CPU + GPU beide GESCHLOSSEN (§T656-659):**
- **CPU-GEMM war Apple AMX** (nicht Code): torch-cpu/numpy erreichen ~2584 GFLOP/s @1024 via AMX
  (Matrix-Coprozessor, undokumentiert, ≠ ARM SME). Wir erreichen AMX ZWEI Wege (ADR-0027, §T658):
  (A) Accelerate cblas_sgemm via cgo = 2506 = **torch GEMATCHT** (war 42× hinter mit Skalar, 3.25×
  hinter mit NEON T656); (B) hand-geschriebener PURE-GO raw-AMX Plan9-asm-Kernel (kein cgo, 32×32-Tile
  über alle 4 Z-Banks, dual-AMX pipelined, prepacked, M1/M2/M3-gated) = **SCHLÄGT Accelerate 6-31%**
  auf >L2-Shapes (2048³ +6%, 512×2048×4096 +24%) = schneller als pytorch, in pure Go. Per-Shape-
  Dispatch (raw-AMX wenn k·n·4≥16MiB && m≥256, sonst Accelerate, sonst NEON); CGO0 fällt auf raw-AMX
  (2082) dann NEON. NEON-Ceiling T656: Go 1.26 arm64 auto-vektorisiert f32 NICHT (objdump: skalares
  FMADDS) → Plan9-NEON 4×16-Tile nötig. [[base-perf-sweep]].
- **Metal-„Gaps" waren BENCH-SEMANTIK, nicht Kernel** (§T659, wichtigste Erkenntnis): der Python-Bench
  hält Tensoren GPU-resident + synct 1×/30 Iter (pipelined); unser `backend.Execute` up/downloadet +
  synct JEDEN Call. Zwingt man torch-mps auf UNSERE Semantik (per-Call sync+transfer, same-machine):
  Conv/n8c16 133 GFLOP/s (**wir 1.9× SCHNELLER** @253), MatMul/1024 2491, MHA 1.28ms → bei gleicher
  Semantik matchen/schlagen wir torch-mps auf allen dreien schon. Unsere Kernel SIND bereits
  MPSMatrixMultiplication / MPSGraph conv2D / MPSGraph batched-attn (= was torch aufruft). Echte Wins
  liegen am Go/Dispatch/Alloc-Overhead DRUMHERUM: MatMul zero-copy operand-wrap (newBufferWithBytes-
  NoCopy + MPSMatrix-Offset + mincore-guard + 64-Slot wrap-cache — Cache load-bearing, uncached 0.8×)
  = 2115→2977 @1024 = **1.41×** (re-verifiziert); MHA MPSGraph-Rework (scale als [1]-Graph-Input statt
  1MB-malloc/Call, pooled buffers, encodeToCommandBuffer) = 1.00→0.64ms = **1.58×**. Verbleibende
  torch-mps-Distanz = per-Op-Sync-Floor (GPU idle-clock zwischen synchronen Ops, torch zahlt ihn
  identisch), NICHT Kernel. REGEL: bevor man einen „N× hinter torch"-GPU-Gap jagt, MISS torch unter
  DEINER sync/transfer-Semantik — der Gap ist oft Pipelining, kein Kernel-Defizit.
- **CPU f32-forward am FLOOR nach T660-T663 (der shared vexp-NEON-Kernel + Profil-getrieben):** der
  T660 NEON-`exp`-Kernel (vexp_arm64.s, Cephes-Reduktion) ist eine VERIFIZIERTE LEAF (§C26), auf der
  MHA-Softmax (T660 2×), standalone OpSoftmax (T661 2-3.2×) UND GELU (T663: erf via Abramowitz-Stegun
  reuse-t e^(−u²), 5.1× OpGELU) alle aufbauen — Muster: ein transzendenter NEON-Primitiv, viele
  f32-native-tolerante Ops assemblieren drauf. Decode-GEMV: cblas_sgemv füllt ein m=2-sgemm-Loch
  (T662 1.6×; m=1 wash — sgemm-m=1 IST schon Apples GEMV; ADR-0027 „m<4 scalar" gilt nur CGO0).
  END-TO-END: f32 CPU GPTForward 1250→13600 tok/s = **~10.9×** über die ganze T656-T663-Kampagne
  (benchmarking.md). GOAI_TIME_OPS-Profil des f32-forward DANACH: matmul 54% (Apple-AMX-Decke, aus —
  nicht wieder versuchen), mha 15% + gelu (vektorisiert), layernorm 7.6% + addbias 6.5% + add 2.9%.
  KEY-REGEL (bestätigt Profil-vor-Optimieren): die verbleibenden Ops (layernorm/addbias/add) sind
  BANDBREITEN-gebunden — KEIN Transzendenter im Hot-Loop (layernorm = reduce+affine, das eine
  math.Sqrt ist per-ROW, vernachlässigbar), also KEIN vexp-artiger Win dort möglich; NICHT grinden
  (Null-Risiko, gegen die „nur compute-bound Ops vektorisieren"-Disziplin). f32-forward = matmul-
  bound am Silicon-Ceiling. Der f32-native Pfad ist opt-in (GOEXPERIMENT=simd); Default bleibt
  bit-exakt. Objdump-Regel bestätigt 3× (T656/T660/T663): Go 1.26 arm64 auto-vektorisiert f32 NICHT
  → Plan9-NEON-WORD-encoding nötig für jeden f32-SIMD-Kernel.
- **TRAINING-Step (T664) — Backward-Pfad war die größte Beute, via SILENT-REF-FALLBACK-Audit:**
  GOAI_TIME_OPS auf dem f32-TrainingStep zeigte gelu_backward 18.9% + crossentropy fwd+bwd 10.7% —
  BEIDE hatten GAR KEINEN cpu-Kernel, fielen auf backend/ref (seriell, skalar f64 math.Erf/math.Exp;
  CE-bwd zahlte exp 2×/Logit). NEON-cpu-Kernel drauf (vgeluGradQuadsNeonF32: GELU-Ableitung
  dx=g·(Φ+x·φ) — das e^(−a²) der AS-erf mit a=abs(x)/√2 IST e^(−x²/2), EIN exp speist erf UND pdf;
  crossentropy.go via vexpRowF32): gelu_bwd 45×, CE 25×/32× vs seriellem ref, **TrainingStep 1.48×
  (1325→1930 tok/s)**, danach matmul 81% = matmul-bound am AMX-Ceiling wie der forward. Gradcheck
  grün, default bit-exakt. LEHRE (bestätigt die stehende silent-ref-fallback-Regel): GOAI_LOG_FALLBACK=1
  über den ECHTEN Workload findet fehlende cpu-Kernel (Format `FALLBACK[cpu→ref]` — MIT Pfeil, nicht
  `[cpu]`); standalone-Profiling tarnt sie als „langsame Op". NACH T664 sind fwd+train VOLL cpu-native
  bis auf addbias_backward (3.3%, BW-gebundene Reduktion) + embed/embed_backward (~0.5%, winzige
  Host-Gather, absichtlich ref §T410) — alle BW-bound/winzig → NICHT grinden. CPU f32 fwd+train
  Kampagne KOMPLETT: alle compute-bound Ops vektorisiert, Rest matmul-bound am Silicon-Ceiling.

**CLASSICAL ML vs sklearn — the "grind perf to beat all incumbents" campaign (2026-07-16, T713-T716).** After the compute kernels (matmul/conv/GPU) were confirmed at BLAS/MPS ceilings (f32 matmul 81% of torch-cpu USING Apple's own cblas → the gap is per-call overhead not the GEMM), the WINNABLE frontier was where GoAI has a STRUCTURAL edge: classical ML vs sklearn (no Python/Cython interpreter overhead). MEASURED head-to-head (n=4000,d=20, fit ms, identical CSV data via classic/perfcompare_test.go + a sklearn script) BEFORE grinding: GoAI beat sklearn on the LIGHT methods (GaussianNB 3.2×, kNN-fit 5.6× — Python overhead dominates), but LOST on the heavy ones: DecisionTree 2.0×, RandomForest 2.3×, GBM 1.9×, SVC **38×**. Root causes + fixes (all §V22 A/B, correctness-preserving, sklearn goldens bit-identical):
- **SVC 38×→1.2× (T713):** Fit EAGERLY MATERIALIZED the full n² Gram (274ms alone @4000) though SMO only needs working-set columns → libsvm approach: lazy bounded kernel-COLUMN cache + incremental gradient + 2nd-order working-set selection (Fan-Chen-Lin). 217→7ms, at the libsvm floor.
- **Tree 2×→parity, GBM 1.9×→9.3× FASTER (T715):** builders RE-SORTED every feature at every node (reflective sort.Slice) → PRESORT each feature once/fit + in-place partition (children inherit sorted ranges); GBM presorts once, reused across boosting rounds. §C3 CATCH: presort-partition maintains ALL d cols/node but a FOREST samples √d/node → it REGRESSED forest 665→965 → made strategy CONDITIONAL on feature-subsampling (presort for full-feature tree+GBM, bit-exact original per-node path for subsampled forest).
- **RandomForest 2.3×→beats-3.4× (T716):** forest = N independent trees = embarrassingly parallel but SINGLE-THREADED → parallelize over GOMAXPROCS with per-tree seeds PRE-DERIVED sequentially from the forest RNG (so completion order can't change the result → BIT-IDENTICAL). 665→84ms, beats sklearn-n_jobs=1 (286).
FINAL: GoAI BEATS-OR-MATCHES sklearn on EVERY classical method (GBM 9.3×, forest 3.4×, kNN 6.5×, NB 2.1×, DecisionTree parity, SVC 1.2× libsvm-floor). REUSABLE LESSONS: (1) beat incumbents where you have a STRUCTURAL edge (no-interpreter-overhead), not where they hit silicon/BLAS ceilings; (2) eager O(n²) precompute is a classic trap — lazy+cached is the libsvm-parity fix; (3) presort-once beats per-node-resort for full-feature trees but REGRESSES subsampled forests — condition on subsampling + always A/B the DEPENDENT variants (forest), not just the primary (tree); (4) embarrassingly-parallel classical fit is free multi-core speedup IF you pre-derive per-unit seeds sequentially (never consume a shared RNG in goroutine order); (5) harness = classic/perfcompare_test.go (gated PERF_CSV_DIR, writes shared CSV) + a sklearn timing script. The measure-first classical-vs-sklearn probe is the template for "beat incumbents" on any domain with a Python-stack incumbent.

**CORRECTION 2026-07-20 (T881/B103) — the "beats EVERY method" claim did NOT reproduce against a RECORDED sklearn.** The T713-718 scorecard's sklearn side ran through an UNCOMMITTED script with NO recorded version (a §V13 gap), so it silently ROTTED as the incumbent improved. T881 committed a reproducible companion (testdata/bench_sklearn.py, best-of-5 fit, n_jobs 1 and -1, prints versions) + `make bench-classic-python` and re-measured against RECORDED sklearn 1.9.0 / numpy 2.5.1 on M2 Pro. GoAI's OWN numbers reproduce exactly (GBM 137→134, forest 83.8→80.8, tree 18.0→18.1, SVC 6.9→6.8 ms — GoAI did NOT regress), but the honest verdict is now a SPLIT, not a clean sweep: GoAI WINS gradient-boosting 9.2×, random-forest (80.8 vs sklearn's own n_jobs=-1 96 — beats even sklearn parallel), naive-Bayes 1.6×; sklearn 1.9.0 WINS decision-tree 1.3× (its Cython splitter got faster → old "parity" flipped) and RBF-SVC 2.0× (libsvm got faster → old "1.2×" flipped); KNN-fit is now a fit-only ARTIFACT (GoAI's KNNClassifier.Fit EAGER-builds the ball tree → 4.5 ms, the old 0.06 ms predated that — a fit+query comparison is the fair KNN measure, not yet harnessed). LESSON (the whole point of §V13, now §V38): a cross-library perf number WITHOUT a recorded incumbent version is NOT reproducible and decays silently — always commit the companion + record the version. So when recalling any "GoAI beats sklearn on X" claim below, treat it as "beat that ERA's sklearn"; the durable wins are the compute-heavy ensembles (GBM/forest) + NB where the structural no-interpreter edge is largest. [[t650-topic-discovery-round]]

**CAMPAIGN COMPLETION (2026-07-16, T717-T718 + full sweep).** Extended the classical-vs-sklearn measurement to EVERY method + predict + clustering. MORE WINS found unmeasured: OLS 1.4×, PCA 2×, KMeans 29×, GMM 1.6× (GoAI beats). MORE GAPS found + fixed: (a) **SoftmaxRegression/logistic 47×→2.5× (T718):** was gradient-descent (352ms/200 steps) vs sklearn lbfgs → replaced with multinomial NEWTON/IRLS (reference-class param for a PD Hessian, Cholesky solve via linalg, ridged Armijo step) = 17.7ms/~9 iters, SAME optimum (accuracy/log-loss identical, sklearn golden TIGHTENED 1e-6→1.3e-9), API stable (steps→max-iters). Lesson: a slow GD/Adam solver is a solver-QUALITY gap — 2nd-order Newton/IRLS closes it (convex loss → same optimum). (b) **kNN-predict/DBSCAN neighbor-search (T717 ball-tree):** brute O(n²)→ball-tree, kNN-predict 15.6×/DBSCAN 1.9× self-speedup but still trail sklearn 7.2×/6.8× at d=20 (CURSE OF DIMENSIONALITY weakens ball pruning + Go-vs-C-f64 distance) = the honest irreducible-ish gap; a distance-loop micro-opt gave nothing (kNN was sort-bound not distance-bound in brute force; ball-tree's heap already fixed that). FINAL SCORECARD — GoAI beats-or-matches-or-competes with sklearn on ~ALL classical: OLS 1.4×/PCA 2×/KMeans 29×/GMM 1.6×/GBM 9.3×/RF 3.4×/NB 2.1×/kNN-fit 6.5×/tree parity/SVC 1.2×(libsvm-floor)/softmax 2.5×; only kNN-PREDICT 7.2× + DBSCAN 6.8× trail (high-d neighbor search). STALE-BASE TRAP (reinforced [[worktree-agent-stale-base]]): 2 perf grinds launched after a LOCAL-only commit (T717, not yet pushed) branched from a base WITHOUT it → the one touching T717's files (knn/dbscan) was unmergeable+discarded; the one on an untouched file (models.go) merged fine. RULE: push (or note file-overlap) before spawning worktree grinds on recently-locally-changed files. Compute side (matmul/conv/GPU) reconfirmed at BLAS/MPS ceilings — the winnable frontier was 100% the classical structural gaps.

**INDUSTRY-COMPARISON SWEEP 2026-07-20 (T882/T881/T883/T885, torch 2.12.1 + sklearn 1.9.0 + tiktoken 0.13.0 in the repo .venv — pin: testdata/requirements-bench.txt; run: make bench-classic-python / bench-gpt-train-python / bench-safetensors-load + nlp/bpe_throughput_test.go + internal/benchcompare tokenizer_compare.py).** The honest CURRENT scorecard vs the Python stacks on this M2 Pro, all cross-validated (identical inputs, output-equality anchor):**WINS** — BPE tokenizer (T882): pure-Go GPT-2 BPE beats tiktoken's Rust end-to-end, encode 28.2 vs 18.8 MB/s (1.50×), decode 470 vs 392 (1.20×), 237208-token parity — a library-delivery win (tiktoken pays Python-binding marshalling; the Rust CORE is not slower). Classical GBM 9.2× / forest (beats sklearn n_jobs=-1) / NB 1.6× (B103 split, see above). **LOSSES (honest, root-caused, on the BENCHMARKS.md losses table)** — (1) END-TO-END TRAINING STEP (T883, fwd+CE+bwd, GPT dim512/6L/seq256): GoAI cpu-simd 2257 / metal 3263 / vulkan 1966 tok/s vs torch-cpu 5058 (2.24×) / torch-mps 12904 (3.95×). NOT a pure-Go penalty — GoAI's f32 GEMM is at AMX parity and metal calls the same MPS kernels; the gap is op-by-op autograd DISPATCH (~0.27ms/op × hundreds) + torch's fused SDPA/graph. Lever: recorder-ize the training tape (only ~1.4× at seq256 per §T411, matmul-dominated) + fusion — the ~3× MPS ceiling is Apple's. (2) safetensors LOAD (T885): full-load GoAI 8.4 vs safetensors-python 12.2 GB/s (1.45×, pure-Go read+parse vs Rust mmap+zero-copy-numpy, AND GoAI is hostile-gated), one-tensor 4.0 vs 10.8 (2.69×, read()+frame double-copy vs mmap+memcpy — lever: mmap partial read). KEY takeaways: GoAI wins where the structural edge is real (no-interpreter classical + tokenizer library-delivery) and trails 1.5–4× where the incumbent has FUSION (torch training) or MMAP+RUST-IO (safetensors) — both diagnosed, both with levers, neither a silicon-ceiling loss. (3) MODEL-FILE LOAD (T885, both halves done): safetensors full-load GoAI 8.4 vs safetensors-python 12.2 GB/s (1.45×, Rust+mmap+zero-copy vs pure-Go read+parse, GoAI also hostile-gated); GGUF load GoAI 2.2 vs gguf-py 12.2 GB/s (5.4×) — the GGUF gap is a FIXABLE per-element decode: gguf.decodeTensor's F32/F16 branch does math.Float32frombits per element where GoAI's OWN safetensors reader is bulk → booked T907 (format/gguf, worker zone, chip spawned). (4) VISION (T884, subagent-built + reverified): ViT+CNN fwd+train vs torch, param-count anchor 807306/1562 verified both sides; torch ahead everywhere but ViT ~40× on GPU is a FIXABLE defect not a ceiling — vision.ViT.Forward LOOPS over the batch (N separate per-image encoder passes) → each op pays the ~0.27ms dispatch floor ×N; on CPU only 2.6-4.2× → booked T908 (batch the ViT encoder, vision zone, MINE). CNN 2.4-5.6× (ordinary fusion gap). Toy-shape inversion: GoAI cpu > its own metal/vulkan (dispatch-bound). META-LESSON (reinforced 2×): measuring against an incumbent SURFACES concrete pure-Go optimizations (T907 per-element decode, T908 per-image ViT loop) — the honest-loss-with-a-lever pattern is a discovery tool, not just a scorecard. Whole 2026-07-20 sweep = 8 CI-green commits (T906 classic bug + T882/T881/T883/T885/T884 comparisons + reproducibility capstone). Bench venv: testdata/requirements-bench.txt; targets make bench-{classic,gpt-train,safetensors-load,gguf-load,vision}-python. **T887 PRODUCTION APPLE LLM DECODE vs llama.cpp (user granted model download): real TinyLlama-1.1B Q4_K_M GGUF (669MB, gitignored models/), same file both engines on M2 Metal → decode GoAI 9.9 vs llama.cpp 197.2 tok/s (20×), prefill 82 vs 1754 (21×). Toy caveat DISCHARGED and the gap WIDENS at scale (~3× toy → ~20× 1.1B) — GoAI's BIGGEST GPU-decode gap. Cause: Q4_K dequant one-thread-per-output (§T416) dominates at 1.1B + llama.cpp hand-tuned Metal kernels/fused-attn; NOT broken (pp/tg 8.3×≈8.9×, correct-config load, f32-exact quant decode). Lever = production-grade Q4_K Metal kernels + attention fusion. Harness internal/benchcompare/prod_decode_external_test.go (gated TINYLLAMA_GGUF) + llama-bench; MLX optional untaken. FULL industry-comparison mandate now EXHAUSTED for pip/download-able incumbents (tiktoken/sklearn/torch-train+vision/safetensors/gguf/llama.cpp). Session infra (non-perf): T893 cichange always-run MECHANISM built (default EMPTY — enabling apicheck+mdlint gating blocked until both green: T892 apicheck-V19 + T889 mdlint-.claude/SPEC-worker); T892 MY-ZONE DONE (nlp/vision/classic/safetensors godoc via reverified opus subagent, apicheck undoc 140→18, the 18 = worker's cuda-tagged llamagpu New*CUDA — chip-flagged); worker CUDA PRs #180/#181 merged. DELEGATION PATTERN confirmed twice (T884 vision-bench + T892 doc-debt): opus worktree subagent builds, main agent independently reverifies (param-count/apicheck/quality/build) before commit — subagent output is DATA not truth.
