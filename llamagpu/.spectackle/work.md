---
schema: v1
---

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

## P-01M0SN14HSEJ2VWB9HT1S7W01F Right-size decoder logits residency and lazily materialize full StepN output
kind: proposal
state: done
created: 2026-08-24
grilled: 2026-08-24 open=0
targets: llamagpu/decoder.go, llamagpu/gpt.go, llamagpu/gpt_storage_test.go

Remove maximum-context logits residency from both GPTDecoder and the shared Llama-style Decoder. They currently allocate Ctx times Vocab F32 output storage for their lifetime although Step, Generate, and StepNLast project and consume exactly one vocab row. GPT-2-small therefore retains 205,852,672 logits bytes for a 201,028-byte common-path requirement, a 1,024-fold amplification. Keep one resident vocab row by default and lazily allocate an exact reusable high-water buffer only when full StepN requests multiple rows; release or replace that buffer safely with the decoder. Retain an internal eager-capacity control in the same binary. Preserve every public output shape and bit pattern across Step, StepNLast, full StepN, recurrent fallbacks, Metal, CUDA, and Vulkan. Promote only after structural tests prove one-row default residency and bounded lazy growth/reuse/release, the GPT-2-small constructor benchmark removes at least 200,000,000 B/op and improves latency by at least 10 times versus eager control, public Step and StepNLast retain at least 0.97 times throughput with unchanged allocations, and the existing exact generation and StepN parity suites pass.

## ADR-01M0SN1SY1EF1B4H7TW75ERAH2 How should decoders serve rare multi-row StepN logits without retaining Ctx times Vocab storage?
kind: adr
state: done
created: 2026-08-24
context: Step and StepNLast need one vocab row, while full StepN needs k rows and may repeat at a stable verification width. Allocation must remain backend-agnostic and release-safe.
decision: One-row resident plus lazy reusable high-water buffer
consequences: Cuts dominant lifetime residency to one vocab row, amortizes repeated full-StepN allocation, and preserves public output semantics. Growth replaces and releases the previous overflow buffer before allocating a larger exact capacity; decoder Release owns the remaining buffer. An internal eager flag retains the incumbent same-binary benchmark control.
status: accepted

kind: radio
option: One-row resident plus lazy reusable high-water buffer
option: Allocate and free an exact temporary buffer on every full StepN
option: Retain the eager maximum-context buffer
option: Redesign StepN around caller-owned output buffers
blocks: P-01M0SN14HSEJ2VWB9HT1S7W01F
choice: One-row resident plus lazy reusable high-water buffer

## T-01M0SN2SMDFK09NPC0WFVNGWSR Implement and gate lazy full-StepN logits buffers
kind: task
state: done
created: 2026-08-24
parent: P-01M0SN14HSEJ2VWB9HT1S7W01F
grilled: 2026-08-24 open=0
targets: llamagpu/decoder.go, llamagpu/gpt.go, llamagpu/gpt_storage_test.go

Change GPTDecoder and Decoder default logits storage from Ctx times Vocab to exactly one Vocab row. Add a backend-agnostic reusable overflow buffer that full multi-row StepN grows to exactly its required rows times Vocab capacity, reuses for smaller requests, replaces safely for larger requests, and releases with the decoder. Keep an unexported eagerFullLogits backendOps switch as the same-binary incumbent control. Route final projections and downloads to the selected buffer while preserving recurrent sequential fallbacks. Extend storage tests for both decoder cores, growth/reuse/release, and add a GPT-2-small-geometry constructor benchmark. Gate on at least 200 MB/op removed, at least 10x constructor speedup, Step and StepNLast throughput at least 0.97x, unchanged hot-path allocations, and all parity suites.

## P-01M0SPBR6NFCJAE77W5JZZ4YA8 Right-size GPT activation residency with lazy reusable prefill workspace
kind: proposal
state: active
created: 2026-08-24
refs: ADR-01M0SPCGWTFB08X22KNMW0DDV6
grilled: 2026-08-24 open=0
targets: llamagpu/gpt.go, llamagpu/gpt_storage_test.go

GPTDecoder currently retains every activation scratch tensor at maximum context although dominant Step uses one row. Retain one row by default, lazily allocate an exact reusable high-water workspace for StepN and StepNLast, preserve a same-binary eager control, and require exact parity plus M2 throughput non-regression. GPT-2-small should remove 35140608 resident bytes without changing public semantics.

## ADR-01M0SPCGWTFB08X22KNMW0DDV6 Use one-row resident GPT workspace plus lazy grouped high-water storage
kind: adr
state: done
created: 2026-08-24
parent: P-01M0SPBR6NFCJAE77W5JZZ4YA8
decision: One-row resident workspace plus lazy grouped high-water storage
consequences: Removes 35140608 GPT-2-small resident bytes while keeping Step allocation-free; prefill growth is atomic across activation buffers, reuse is bounded by the largest prompt, and growth avoids simultaneous old-plus-new workspace residency.
status: accepted
targets: llamagpu/gpt.go, llamagpu/gpt_storage_test.go

Choose a grouped workspace owner: keep one row resident for Step, allocate all prefill activation buffers together at exact requested rows, reuse the group for smaller requests, release the old group before growth, and release the final group with the decoder. An eager max-context control remains internal for same-binary comparison. Rejected alternatives: per-field independent growth risks mixed generations after partial failure; permanent max-context storage wastes 35140608 bytes at GPT-2-small geometry; per-call allocation churn adds latency.
choice: One-row resident workspace plus lazy grouped high-water storage

## T-01M0SPF62VEYQRE53F1G5G11X3 Implement and gate lazy GPT activation workspaces
kind: task
state: active
created: 2026-08-24
parent: P-01M0SPBR6NFCJAE77W5JZZ4YA8
refs: ADR-01M0SPCGWTFB08X22KNMW0DDV6
grilled: 2026-08-24 open=0
targets: llamagpu/decoder.go, llamagpu/gpt.go, llamagpu/gpt_storage_test.go

Add a one-row resident GPT activation workspace, grouped exact-row lazy high-water allocation for StepN and StepNLast, release-safe growth and teardown, and an internal eager max-context control. Test fused and portable shapes, reuse, growth, failure cleanup, final release, and full StepN parity. Benchmark GPT-2-small activation residency and require at least 34000000 fewer B/op, at least 10x lower constructor ns/op, and at least 0.97x M2 Step and StepNLast throughput with unchanged allocations.

## T-01M0SPRH8NEV49Q636WM7Y1PQA Route GPT hidden-state readback through selected activation workspace
kind: task
state: draft
created: 2026-08-24
parent: P-01M0SPBR6NFCJAE77W5JZZ4YA8
refs: ADR-01M0SPCGWTFB08X22KNMW0DDV6
grilled: 2026-08-24 open=0
targets: llamagpu/medusa.go, llamagpu/medusa_test.go, llamagpu/example_test.go

Update GPT StepHidden and StepNHidden so hidden-state downloads read the resident or lazy workspace selected by the completed Step call. Preserve Llama hidden readback unchanged and validate Medusa all-reject, accept-all, and GPT examples.
