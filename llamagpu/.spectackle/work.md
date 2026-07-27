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
