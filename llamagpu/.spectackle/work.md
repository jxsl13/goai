---
schema: v1
---

## T-01KYJPSEWVFRSRSMX2VEV7AY95 Port the dispatch fusion and encode-overlap from Decoder onto GPTDecoder
kind: task
state: draft
created: 2026-07-27

HIGHEST-LEVERAGE ITEM FOUND SO FAR, and it is a replay rather than an estimate: the identical arc on the sibling Decoder measured 270.0 -> 335.5 (fusion) -> 375 tok/s (overlap), +38.9% cumulative on Metal. docs/history/tasks.md:634-635 records that GPTDecoder was deliberately left out at the time ("GPT decoder (own gptBlock) untouched", "its overlap = follow-up"). This is that follow-up.

SITE: llamagpu/gpt.go:112 (*GPTDecoder).Step, record loop :132-155, r.Finish() at :159. Constructor llamagpu/llamagpu.go:131 NewGPT — its backendOps literal lacks asyncEncode:true, unlike New (:111) and NewQuant (:172).

DEFECT, two independent regressions against Decoder:
(a) Dispatch count. gpt.go:133-150 issues 16 dispatches per layer (LayerNorm, wq, wk, wv, Blit k, Blit v, MHA, wo, Binary-add, LayerNorm, w1, AddBias, GELU, w2, AddBias, Binary-add). Decoder is at 12 after fused QKV, RoPE-pair, MatMulAcc residual epilogues and fused SwiGLU. Two of the four extra dispatches are free to remove today: wo.record + Binary(dx, ao, dx, add) and w2.record + Binary(dx, mo, dx, add) are exactly what linear.recordAdd / MatMulAcc exists for (decoder.go:101), and Metal implements MatMulAcc (llamagpu.go:35). QKV fusion is also free upstream — HF GPT-2 ships attn.c_attn.weight as a fused [d,3d] block that nlp.GPT splits apart.
(b) Synchronization. r.Finish() (gpt.go:159) is commit AND wait; GPTDecoder has no pending/pendingPos field at all, so host encode of token N+1 cannot overlap GPU execution of N. Prior measurement put per-submitted-buffer commit/drain at about 0.22ms and recorder create-plus-empty-commit-wait at 23.7us on this hardware class — exposed every token.
Minor, same loop: d.ffnDim(b) (gpt.go:245) does an interface type assertion per layer per token to recover a construction-time constant; hoist to an int field on gptBlock.

FIX: port the three landed slices. (a) add ffn int to gptBlock, drop ffnDim; (b) replace both matmul+Binary-add pairs with recordAdd; (c) add a fused wqkv column-concatenated weight plus Copy2D band extraction, mirroring decoder.go:2814-2822; (d) factor Step into encodeStep(pos), add pending/pendingPos, switch Finish -> Commit ... pre-encode pos+1 ... Wait, and set asyncEncode:true in NewGPT. GPT's positional embedding is summed host-side into dx before commit (gpt.go:119-124), so the recorded chain depends only on pos — the property that made the overlap legal for Decoder.

VALIDATION GATE (benchmark only): no benchmark covers this. Write BenchmarkGPTDecodeStepMetal in the existing //go:build darwin && cgo harness of gpt2_scale_test.go, which already builds a real-geometry model (vocab 50257, ctx 1024, d 768, 12 layers, 12 heads via synthGPT2HF + nlp.GPT2FromHF): prefill a 16-token prompt once, then time dec.Step(tok, pos) with pos wrapping below Ctx, reporting tok/s. LAND THE SUB-SLICES SEPARATELY — single-dispatch removals sit inside +/-2% noise, so measure fusion and overlap as two distinct A/Bs on the same binary and session, f32 on both sides.

EXPECTED: +25-40% tok/s, high confidence.

BIT-IDENTITY BAR: overlap (d) is zero-risk — the recorded chain is unchanged, only submit/wait timing moves. Residual fusion (b) and QKV fusion (c) MUST be checked: MatMulAcc is MPS beta=1, folding the add into the GEMM epilogue so the sum is formed at a different point; and one [d,3d] GEMM need not reduce over K in the same order as three [d,d] GEMMs. Both were accepted on the Decoder side against a 2e-3 tolerance parity test, NOT bit-equality. Bar for this port: extend TestGPT2ScalePipeline's greedy-vs-full-forward comparison (gpt2_scale_test.go:50-73) to at least 256 generated tokens and assert TOKEN-FOR-TOKEN equality before and after.

PERFSCAN RULE REQUIRED, two: (i) sibling divergence — two methods of the same name on sibling types in one package whose bodies call the same recorder interface, where one calls Commit+Wait and the other Finish, or one's call-set includes MatMulAcc while the other has MatMul immediately followed by Binary(..., add) on the same destination ident. Pair methods on types sharing an interface and diff their recorder call multisets. (ii) fusible pair — an X.record(r, src, tmp, m) whose tmp result is consumed only by a following r.Binary(dst, tmp, dst, add) in the same block; an AST use-def pass matches this directly.

## T-01KYJPSFC1FABA9F2ADEX2BPQT Fix the column-major stride in Medusa headArgmax
kind: task
state: draft
created: 2026-07-27

SITE: llamagpu/medusa.go:186 headArgmax.

WHY HOT: called at medusa.go:177 once per Medusa head per drafting round, inside MedusaGenerate's main loop (:135-181). A round emits up to K+1 tokens, so this is roughly K/(m+1) full [dim,vocab] projections PER GENERATED TOKEN. MedusaGenerate takes a HiddenStepper which *Decoder satisfies (medusa.go:82), so this runs on Metal on this host.

DEFECT: wd is a contiguous row-major [dim, vocab] f32 (nlp/medusa.go:125). The inner loop indexes wd[i*vocab+v] with i as the INNER variable, so the stride is vocab floats — 201KB for GPT-2's 50257 vocab. Every inner iteration touches a fresh cache line and a fresh page: no spatial locality, no prefetch, TLB thrash. For d=768/vocab=50257 that is 38.6M useful floats fetched as roughly 2.5GB of cache-line traffic, per head, per round. The doc comment at medusa.go:92 calls host projections negligible; this contradicts it, and it means the previously reported 1.81x headline was measured WITH this cost included, so the ceiling is higher than recorded.

FIX: loop interchange with tiling over v. Hold a reusable []float32 accumulator tile (width about 8192 so 32KB stays in L1); for each tile stream the dim weight rows contiguously, accumulating acc[j] += h*w, then scan acc ascending with strict > against the running best.

VALIDATION GATE (benchmark only): none exists. headArgmax is pure host code needing no build tag. Write BenchmarkHeadArgmax with sub-benchmarks dim=768/vocab=50257 (GPT-2 small) and dim=4096/vocab=32000 (Llama-7B), building the f32 [dim,vocab] tensor once outside the timer, b.SetBytes(dim*vocab*4), reporting ns/op and MB/s for both arms in f32. Then confirm end to end with TestMedusaGenerateGPTTrainedThroughput (medusa_test.go:69) at larger Dim and vocab.

EXPECTED: 20-100x on this function; high confidence on the function, medium on the end-to-end multiple (depends on m, K and model size).

BIT-IDENTITY BAR: bit-identical, and provably so IF done exactly as described. For fixed v both forms accumulate hidden[i]*wd[i*vocab+v] into a float32 accumulator with i strictly ascending from zero — same operation sequence, same rounding, same FMA contraction on arm64. The argmax tie-break must also be preserved: scan v ascending with strict >, so the first maximum wins. EXPLICIT PROHIBITION: do NOT widen the accumulator to float64, do NOT delegate to a tensor/nn matvec (which may reduce pairwise or in f64), and do NOT hand-vectorize with independent partial sums. Any of those changes the reduction order, and this function's output is a token id, so a flipped near-tie changes the emitted sequence.

PERFSCAN RULE REQUIRED: a strong general rule. Node shape: a nested for where the INNER induction variable appears in an index expression A[i*S + j] with S loop-invariant and j the OUTER variable — the inner variable carries the large stride. Detect by parsing index expressions of the form BinaryExpr{+, BinaryExpr{*, ident_a, invariant}, ident_b} and comparing ident_a/ident_b against the enclosing loop nest order. Restrict to loops whose bound is a shape or len expression to avoid noise. This is the classic column-major-over-row-major GEMV/argmax.

## T-01KYJPSFVPE1HBM80MHZE5EJRY Stop computing and downloading discarded prefill logits in StepN
kind: task
state: draft
created: 2026-07-27

SITE: llamagpu/decoder.go:3037 (d.recordLogits(r, k)) and :3044 (out := make([]float32, k*d.v) plus DownloadF32); identically llamagpu/gpt.go:230 and :237.

WHY HOT: once per request, but it is the whole of time-to-first-token and it is the path advertised as the 41x prefill win. Of five callers, FOUR discard the result entirely: SpeculativeGenerate (speculative.go:45 and :48, both bound to _), PromptLookupGenerate (promptlookup.go:36), MedusaGenerate (medusa.go:125). Only Decoder.Generate (decoder.go:3066) and GPTDecoder.Generate (gpt.go:264) use it, and they use ROW k-1 ONLY. The per-round verify calls (speculative.go:93, promptlookup.go:73, medusa.go:145) genuinely need all rows and must keep the current path.

DEFECT: for a 512-token prompt on GPT-2-124M geometry the discarded LM head is 512*768*50257 = about 19.8 GFLOP against roughly 43.5 GFLOP for the entire 12-layer prefill — about 31% of prefill compute spent on rows nobody reads. On top of that, out := make([]float32, k*d.v) is a 103MB Go allocation and DownloadF32 a 103MB copy per prefill, both fully wasted at three call sites. Cost scales O(k * vocab * dim), so it worsens exactly where prefill matters.

FIX, two separable changes:
1. BIT-SAFE. Add an internal prefill(tokens []int, pos int) error — StepN's body with recordLogits and the download omitted — and point speculative.go:45, speculative.go:48, promptlookup.go:36 and medusa.go:125 at it. No logits are read at those sites, so output cannot change by construction.
2. NEEDS VALIDATION. Add StepNLast for Generate: after the final norm, r.Blit(d.xn.b, (k-1)*d.d, d.xn.b, 0, d.d) to move the last hidden row to offset 0, then recordLogits(r, 1) and download d.v floats. recorder.MatMul takes no source offset, so the Blit is how to express this with existing primitives — one tiny extra dispatch against a whole [k,vocab] GEMM removed.

VALIDATION GATE (benchmark only): TestPrefillThroughput (llamagpu_test.go:358) already times StepN on Metal but is a Test that logs rather than a Benchmark. Write BenchmarkPrefillMetal (//go:build darwin && cgo) over the existing D=512/h8/kv2/6L/vocab=32000 bench model plus GPT-2-124M geometry, prompt lengths 64/256/512, sub-benchmarks StepN vs prefill vs StepNLast, reporting ms/prefill, prompt-tok/s and B/op. Same model instance and f32 on both arms.

EXPECTED: 1.3-1.6x on prefill wall time at 512-token prompts (larger at bigger vocab or shallower depth), plus elimination of a k*vocab*4-byte allocation and copy per request. High confidence for change 1, medium-high for change 2.

BIT-IDENTITY BAR: change 1 has NONE — the removed work has no consumer. Change 2 is REAL and must be gated: the last row's logits would come from a GEMM with M=1 instead of M=k, MPS may select a different kernel and K-reduction order, and the package's own TestStepNMatchesSequentialSteps (llamagpu_test.go:350) asserts only 2e-3 relative tolerance between M=1 and M=k — it already knows they are not bit-equal. A 1e-3 logit shift can flip a near-tie and change the emitted token. Gate change 2 on a new test: greedy Generate(prompt, 256) before and after must return TOKEN-FOR-TOKEN identical ids across several prompt lengths and both decoders. If that does not hold, ship change 1 alone.

PERFSCAN RULE REQUIRED: a function returns a slice of n*stride elements and at one or more call sites the result is bound to the blank identifier, or the only subsequent index expression is a constant-offset suffix such as all[(n-1)*stride:]. AST pass over call sites of package-local functions with slice results: classify each site as discarded, single-row, or full-use, and flag any producer whose expensive tail (a record*/Download* call guarded by the row count) is unconditional while at least one site is discarded.
