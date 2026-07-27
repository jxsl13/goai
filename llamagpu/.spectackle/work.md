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
