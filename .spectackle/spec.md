---
schema: v1
prefix: PERF
---

## PERF-TOOL-001
IF perfscan runs locally or in CI, THEN the GoAI performance workflow SHALL execute github.com/jxsl13/perfscan/perfscan@v0.89.0 with GOPROXY=direct and perfscan.yaml.

Rationale: Keeps the public library module dependency-free, avoids the slow Go proxy, and makes local and CI analysis reproducible.

## intent
- T-01M00HJ1XZE328Z83KRRC34619 Migrate GoAI to external perfscan v0.89.0: Replaced the duplicate in-repository scanner with github.com/jxsl13/perfscan/perfscan@v0.89.0, installed through GOPROXY=direct. GoAI now owns only perfscan.yaml plus Make/CI wiring; its go.mod remains dependency-free. Full analyzer execution and targeted PS2002/PS4002/PS5001 parity passed. Spectackle diff/commit automation was unavailable because the supplied workspace is not a Git worktree; unre [body truncated at tombstone retention cap]
- T-01M00N1614E0JTWGQP5ERH3D1X Implement allocation-free ExecuteInto for CPU SiLU: validated pass by l0-output-validator no attributed diff (46558ea70fc3 binds the target list, not code) :: PASS. No Git diff was available because /Users/john/Desktop/goai is not a Git work tree, so validation used the rendered task, direct source inspection, named tests, race run, cross-compiles, external perfscan, and measured benchmark evidence. ExecuteInto preserves the shared routing and fall [body truncated at tombstone retention cap]
- T-01M00R6G6EFJKSMZB5MJ55NC8Y Build the publishable M2 leadership matrix: Archived after validated delivery; source pin promotion remains blocked solely by the operator-owned bare Git configuration.
- T-01M00R71HNET0A8NR0AS6HNA6G Profile native Metal against pinned MLX and llama.cpp: Profile consumed by PERF-M2-Q4K-COOPERATIVE-001, T989 implementation, benchmark evidence, and perfscan #565.
- T-01M00R7KE5F688F41NDC5G8K21 Implement the measured M2 Metal winner: Q4_K cooperative implementation and all measured learnings landed in code, leadership matrix, docs, BM60, evidence bundle, and perfscan #565. The remaining M2 gap continues at Q6_K.
- T-01M00X1S9RF9K8Z56XWXMQ41GB Measure and implement cooperative M2 Q6_K decode: Q6_K M=1 cooperative Metal decode landed with 2.692x-11.788x leaf wins, 1.677x isolated full-model gain, capability and scalar fallbacks, complete evidence, BM61, and perfscan issue 565. Remaining external gap is retained as roadmap input rather than overstated as leadership.
- T-01M00YJZTREKTRP5ZQQEM0NBTS Profile the next M2 decode bottleneck after Q4_K and Q6_K: Archived profile decision: TinyLlama is 77.06% Q4_K and 22.89% Q6_K by encoded bytes, with no Q2_K/Q3_K/Q5_K. Metal System Trace shows short-context decode is GPU-kernel-bound (57.437583 ms median GPU duration per command buffer versus normally 0.24-0.65 ms CPU encoding); typical buffers contain 23 grouped compute intervals and 22 tiny blits. No per-shader attribution was claimed because shader ti [body truncated at tombstone retention cap]

## PERF-TOOL-002
WHEN the ARM64 SIMD F64 SiLU kernel changes, the GoAI CPU backend SHALL retain non-SIMD exact behavior, prove vector-tail bit identity and at most 1e-13 relative error, verify NEON instructions, and demonstrate at least 1.5x M2 speedup over scalar.

Rationale: The two-lane assembly is valuable only when measured and numerically interchangeable within the established F64 model tolerance.

## PERF-TOOL-003
WHEN a GoAI performance investigation identifies a reusable optimization pattern, the optimizing agent SHALL create a GitHub issue in jxsl13/perfscan containing measured reproduction evidence and false-positive boundaries before the local task reaches done.

Rationale: Generalizable performance discoveries should improve the reusable analyzer rather than remain project-local.

## PERF-TOOL-004 {applies: go:backend.ExecuteInto,go:cpu.siluIntoKernelCPU}
WHEN backend.ExecuteInto selects a backend whose IntoBackend capability supports the operation, the GoAI backend SHALL write into the exact caller outputs with 0 output-storage allocations and without invoking ordinary allocating execution.

Rationale: Caller-owned outputs remove dominant eager allocation without unsafe implicit pooling or lifetime inference.

## PERF-LEADERSHIP-MATRIX-001
WHEN a GoAI performance comparison is published, the GoAI benchmark suite SHALL write a BENCHMARKS.md matrix row with hardware, workload, dtype or quantization, batch and context, warm or cold state, workspace and transfer boundaries, quality, and immutable baseline versions.

## PERF-INTERLEAVED-EVIDENCE-001
WHEN a repeatable performance optimization is validated, the optimizing agent SHALL collect at least 10 interleaved prebuilt samples and report the median and confidence interval through benchstat, unless the external harness mandates a stricter protocol.

## PERF-M2-NATIVE-METAL-001
WHEN an M2 performance gap is targeted, the optimizing agent SHALL profile native Metal before MoltenVK, force-disable the suspected bottleneck, and select only a leaf whose same-session evidence clears the applicable performance gate.

## PERF-ARCHITECTURE-GATE-001
WHEN a graph, IR, JIT, or autotuner redesign is proposed, the optimizing agent SHALL attach an end-to-end profile plus a forced-off A/B showing dispatch, fusion, or specialization is the bottleneck before implementation.

## PERF-EXECUTE-INTO-VALIDATION-001 {applies: go:backend.ExecuteInto,go:backend.validateOutputBase,go:backend.ValidateOutput}
WHEN backend.ExecuteInto begins execution, the GoAI backend SHALL return an error from backend.ExecuteInto before arithmetic for recorder contexts, released or aliased storage, or layout, dtype, shape, count, and device mismatches.

## PERF-M2-Q4K-COOPERATIVE-001 {applies: go:metal.QMatMulQ4_K,c:mtl_recorder_qmatmul}
WHEN native Metal executes a resident Q4_K decode matvec with M=1, the GoAI Metal backend SHALL use a simdgroup-cooperative kernel, retain a scalar forced-off path, pass model-scale cross-reference tests, and prove at least 1.5x same-session leaf speedup before default enablement.

## PERF-M2-Q6K-COOPERATIVE-001 {applies: go:metal.QMatMulQ6_K,c:mtl_qmatmul_q6k,c:mtl_recorder_qmatmul}
WHEN native Metal executes resident Q6_K decode with M=1, the GoAI Metal backend SHALL enable a cooperative kernel only after 1.5x same-session speedup on every selected shape, retain scalar forced-off control, and pass model-scale cross-reference tests.

## PERF-M2-KQUANT-SUCCESSOR-001 {applies: backend/metal/metal_bridge.m,backend/metal/qmatmul_test.go,internal/benchcompare/leadership}
WHEN selecting a successor Q4_K or Q6_K M=1 specialization, the GoAI Metal backend SHALL retain the current cooperative same-binary control and require 1.10x paired speedup on every selected leaf plus 1.05x full-model median speedup before promotion.

Rationale: T994 isolates GPU kernel work; existing Q4_K/Q6_K rules retain correctness and fallback requirements.

## PERF-FUSION-BOUNDARY-001 {applies: backend/metal/q4k_bench_test.go,internal/benchcompare/leadership,docs/benchmarking.md}
WHEN estimating GPU graph-fusion leverage for operations already recorded together, the GoAI benchmark suite SHALL compare the complete fused seam against a control using exactly 1 command buffer per sample instead of summing isolated synchronous leaf times.

Rationale: T997 showed that summing two standalone Q4_K leaf times predicted a false gate/up win; the correct batched control was already as fast as concatenation plus extraction.
