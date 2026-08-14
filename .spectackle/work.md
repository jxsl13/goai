---
schema: v1
---

## P-01M00H6T2VEKQ9SP46JABMCAEN Externalize GoAI perfscan without Go proxy
kind: proposal
state: approved
created: 2026-08-14
refs: R-01M00H953TFSFBB0ENZW1E97N8
grilled: 2026-08-14 open=1
targets: Makefile, .github/workflows/ci.yml, internal/cichange/config.go, internal/cichange/impact_alwaysrun_test.go, internal/perfscan, perfscan.yaml, docs/perf-notes-training.md, spec/20-constraints.md, spec/70-tasks.md

Goal: replace GoAI's duplicated internal/perfscan engine with the released github.com/jxsl13/perfscan/perfscan CLI while preserving GoAI-specific vocabulary and CI gating. Pin perfscan/v0.89.0; fetch/install with GOPROXY=direct only so no Go proxy is used. Keep perfscan dependencies out of GoAI's go.mod. Move internal/perfscan/perfscan.json to root perfscan.yaml, expanding current vocabulary to the external schema where required. Makefile shall provide deterministic install, scan, and verification targets; CI shall install directly from GitHub and run the external scanner for every non-documentation source change. Remove internal/perfscan from cichange alwaysRun because it will no longer be a Go package, and update selector tests. Delete the duplicate engine/tests only after external command parity is measured on GoAI, including representative PS1001, PS2002, PS4002, and PS5001 findings. Update human-facing docs and the generated legacy spec through docgraph. Verification: direct install succeeds with GOPROXY=direct; make perfscan lists expected domain findings; Makefile/cichange tests and spec checks pass. Rejected alternative: add the tool module to GoAI go.mod; this would pollute the zero-dependency library module with x/tools/yaml and may trigger proxy-backed module resolution.

## R-01M00H953TFSFBB0ENZW1E97N8 External perfscan compatibility and CI migration
kind: research
state: draft
created: 2026-08-14
targets: Makefile, .github/workflows/ci.yml, internal/cichange/config.go, internal/perfscan, perfscan.yaml

Finding: github.com/jxsl13/perfscan/perfscan is a nested Go module (Go 1.25) with stable release tag perfscan/v0.89.0 and current main c42afc2. The released CLI provides approximately 75 staticcheck-style go/analysis checks, stable PS IDs, YAML or JSON project vocabulary, graded fixes, baselines, JSON, and SARIF. GoAI's current domain IDs PS1001, PS1002, PS2001, PS2002, PS3002, PS4002, and PS5001 are present upstream. Direct resolution works through GOPROXY=direct; a go.mod tool dependency is unnecessary and rejected because it would add x/tools and yaml to the public library module. Migration risks: external go/analysis follows active Go build constraints rather than parsing every tagged source file, the CLI uses -exit-zero instead of the internal -strict polarity, and current cichange alwaysRun can schedule only Go packages. Mitigations: pin v0.89.0, keep external installation in Make/CI, retain the GoAI vocabulary at root perfscan.yaml, remove internal/perfscan from alwaysRun, add a dedicated CI perfscan step for code changes, and compare representative finding classes before deleting the duplicate scanner. Verification must cover direct install, vocabulary activation, cichange selection, and spec rendering.

## T-01M00HJ1XZE328Z83KRRC34619 Migrate GoAI to external perfscan v0.89.0
kind: task
state: done
created: 2026-08-14
parent: P-01M00H6T2VEKQ9SP46JABMCAEN
grilled: 2026-08-14 open=1
targets: Makefile, .github/workflows/ci.yml, internal/cichange/config.go, internal/cichange/impact_alwaysrun_test.go, internal/perfscan, perfscan.yaml, docs/perf-notes-training.md, spec/20-constraints.md, spec/70-tasks.md

Goal: migrate GoAI from internal/perfscan to github.com/jxsl13/perfscan/perfscan@v0.89.0 with direct VCS installation and preserve advisory CI analysis. EDIT Makefile: add PERFSCAN_VERSION=v0.89.0 and configurable PERFSCAN binary, add perfscan-install with GOPROXY=direct, make perfscan invoke the external binary with -config perfscan.yaml, make perfscan-check scan ./... with -exit-zero, and run it from preflight. EDIT .github/workflows/ci.yml: add a code-change-conditioned perfscan job using Go 1.26 and make perfscan-check. CREATE perfscan.yaml from internal/perfscan/perfscan.json without the unsupported _comment key; include shapeMethods, fanOutHelpers, and dtypeMethods needed by upstream domain checks. EDIT internal/cichange/config.go: remove internal/perfscan from alwaysRun and explain that the external CLI is gated by its dedicated CI job. EDIT internal/cichange/impact_alwaysrun_test.go or internal/cichange/config_test.go with a regression proving the removed directory is not scheduled as a Go package. DELETE internal/perfscan/PATTERNS.md, perfscan-bench.sh, perfscan.go, perfscan.json, and perfscan_test.go only after upstream parity is observed. EDIT docs/perf-notes-training.md to name the external tool and make target. EDIT spec/20-constraints.md and spec/70-tasks.md only through go run ./internal/docgraph, then render SPEC.md. Existing public library APIs and go.mod are OUT OF SCOPE and must not change. VERIFY: GOPROXY=direct GOBIN=/tmp/goai-perfscan-bin go install github.com/jxsl13/perfscan/perfscan@v0.89.0; PERFSCAN=/tmp/goai-perfscan-bin/perfscan make perfscan ARGS='-exit-zero ./tensor/... ./backend/cpu/...'; output must include PS2002, PS4002, and PS5001; make perfscan-check; go test ./internal/cichange ./internal/speccheck ./internal/docgraph; go run ./internal/docgraph spec verify; /opt/homebrew/bin/spectackle lint .; /opt/homebrew/bin/spectackle call -root . check '{}'.

## R-01M00JC6F4EM0VYVXNJ0B1T8NJ ARM64 F64 activation SIMD gap
kind: research
state: draft
created: 2026-08-14
targets: backend/cpu/elementwise.go, backend/cpu/vexp_arm64.go, backend/cpu/vexp_arm64.s, backend/cpu/vsilu_f64_bench_test.go, backend/cpu/vsilu_f64_test.go

Bottom-up layer: L1 CPU kernels on the primary M2. The arm64 GOEXPERIMENT=simd build already vectorizes F32 activation families but declares vexpF64Fast=false, leaving F64 SiLU on scalar math.Exp. The amd64 SIMD build already proves the reusable design: a range-reduced degree-13 FMA exp approximation, vector SiLU, a scalar polynomial tail, sub-1e-13 relative accuracy, and lane/tail bit identity. Candidate: port only the highest-leverage F64 SiLU leaf to two-lane NEON first, split the shared F64 capability gate so softplus and softcap remain unchanged until independently measured, and broaden the existing benchmark/accuracy tests to arm64. Risks: Plan 9 encodings, FMA sequencing, NaN/Inf and signed-zero semantics, tail mismatch, and a 2-lane vector whose speedup may not clear the threshold. Required evidence: default-build exact cross-reference remains unchanged; arm64 SIMD accuracy and body-versus-tail bit tests; edge values; objdump verification; interleaved count>=6 benchmark against the scalar default on the same M2; retain only if >=1.5x with no allocation regression. No similar Spectackle rejection was found. Existing unrelated nlp documentation and nn bit-identity failures are outside this kernel task.

## P-01M00JCWVTETERW5JY19ME28PB Vectorize F64 SiLU with ARM64 NEON
kind: proposal
state: approved
created: 2026-08-14
refs: R-01M00JC6F4EM0VYVXNJ0B1T8NJ
grilled: 2026-08-14 open=1
targets: backend/cpu/elementwise.go, backend/cpu/vexp_default.go, backend/cpu/vexp_amd64.go, backend/cpu/vexp_arm64.go, backend/cpu/vexp_arm64.s, backend/cpu/vsilu_f64_bench_test.go, backend/cpu/vsilu_f64_test.go, docs/perf-notes-cpu.md, spec/70-tasks.md

Port the already-proven amd64 F64 SiLU polynomial pipeline to two-lane ARM64 NEON for GOEXPERIMENT=simd. Replace the overly broad vexpF64Fast switch with operation-specific F64 fast-path constants so arm64 enables only SiLU; amd64 keeps SiLU, softplus, and softcap enabled; default builds enable none. Implement a noescape NEON vector body and an FMA-identical scalar tail, preserving the current non-SIMD CPU/reference behavior. Expand the existing SiLU benchmark and validation files to arm64. Acceptance: arm64 SIMD body and scalar tail are bit-identical over a dense sweep; relative error versus stable math.Exp SiLU is <=1e-13 plus explicit finite/zero/Inf/NaN edge checks; generated instructions are objdump-confirmed; default and SIMD package tests pass; interleaved M2 count>=6 benchmark demonstrates >=1.5x versus default with equal allocations. Reject and record if the two-lane kernel does not clear the measured threshold.

## T-01M00JM4ZYE0AB0DZBZRKJG9VF Implement ARM64 NEON F64 SiLU
kind: task
state: done
created: 2026-08-14
parent: P-01M00JCWVTETERW5JY19ME28PB
refs: R-01M00JC6F4EM0VYVXNJ0B1T8NJ
grilled: 2026-08-14 open=1
targets: backend/cpu/elementwise.go, backend/cpu/vexp_default.go, backend/cpu/vexp_amd64.go, backend/cpu/vexp_arm64.go, backend/cpu/vexp_arm64.s, backend/cpu/vsilu_f64_bench_test.go, backend/cpu/vsilu_f64_test.go, docs/perf-notes-cpu.md

Implement the approved L1 kernel change. In backend/cpu, split the F64 feature gates by operation, keep every default/non-SIMD gate false, retain all amd64 SIMD gates, and enable only SiLU on arm64 SIMD. Add a noescape two-lane NEON SiLU body using the existing degree-13 FMA exp design and a scalar FMA twin for the remainder. Expand vsilu_f64_bench_test.go and vsilu_f64_test.go to arm64, including finite sweep, signed zero, infinities, NaN, and vector-versus-tail bit checks. Verify with gofmt, default go test ./backend/cpu, GOEXPERIMENT=simd CGO_ENABLED=0 go test ./backend/cpu, go tool objdump instruction inspection, and interleaved benchstat-style count>=6 default-versus-SIMD BenchmarkSiLUF64Kernel on this M2. Keep only a speedup of at least 1.5x with equal allocations, then record exact numbers in docs/perf-notes-cpu.md and legacy SPEC. Scope excludes standalone F64 Exp/Log/Tanh, F64 softplus/softcap on arm64, non-ARM assembly, public APIs, and GPU backends. RESTORE: if accuracy, edge semantics, instruction verification, or speed threshold fails, restore arm64 SiLU gate false and remove the unused assembly/test additions; no data migration or public compatibility surface is involved.

## R-01M00MRVYDFVGVQ6GTM2R2MBJS Caller-owned output reuse at the Execute choke point
kind: research
state: done
created: 2026-08-14
targets: go:backend.Execute, go:tensor.NewOn, go:cpu.siluKernelCPU, go:tensor.Storage.Release

Finding: the current F64 SiLU eager path allocates 2,883,884 B and 9 objects per call at shape [256,1408]; alloc-space profiling attributes effectively all bytes to tensor.heapAllocator.Alloc. The SIMD kernel itself is already 1.91x faster than PyTorch aten.silu.out at one thread, but the 12-thread eager operation is 225.2 us versus the best PyTorch result of 196.35 us because fresh outputs induce allocation, GC, and scheduler work. Existing rule PERF-OP-OUTPUT-ALLOCATION-001 requires this optimization to stop at the op API instead of hiding pooling beneath tensor ownership. The current PyTorch out contract requires compatible output shape, device, and dtype, rejects grad-requiring outputs, and permits kernels to reuse destination storage (https://docs.pytorch.org/docs/stable/notes/out.html). Decision: add an explicit caller-owned ExecuteInto path and a checked Context output accessor; do not add a global pool, finalizers, implicit Release, or refcount redesign. Safety requirements: reject recorder and autograd contexts; reject output and input storage aliasing initially; require exact dtype, shape, contiguous layout, live capacity, and execution device; preserve Execute behavior when unused; preserve fallback and routing; return the exact provided tensor; unsupported kernels must fail without silently allocating. Initial proof slice: CPU F64 SiLU, because its arithmetic is already validated and isolates the allocation leverage. Benchmark both Execute and ExecuteInto with alternating prebuilt binaries, require zero output bytes and allocations in the reused path, and require performance faster than local PyTorch out on M2.

## P-01M00MX92BE1ERNRWY3VPJZB80 Add allocation-free ExecuteInto for caller-owned outputs
kind: proposal
state: done
created: 2026-08-14
parent: R-01M00MRVYDFVGVQ6GTM2R2MBJS
grilled: 2026-08-14 open=0
targets: go:backend.Execute, go:tensor.NewOn, go:cpu.siluKernelCPU

Implement a zero-buffer-allocation inference API: ExecuteInto(ctx, op, inputs, outputs, attrs) error. Keep Kernel and Backend source compatibility by adding optional IntoBackend and IntoKernel capability interfaces instead of changing every kernel ABI. Refactor Execute dispatch resolution into one shared resolver so routing, optimized-CPU fallback, reference fallback, diagnostics, and dtype selection cannot drift. ExecuteInto rejects Recorder contexts, nil or released outputs, output-input or output-output Storage aliasing, non-contiguous or offset outputs, and device-kind mismatch before mutation. Each IntoKernel validates exact output count, dtype, shape, and storage length; an unsupported backend and op capability returns a clear error and never falls back to ordinary allocating execution. Implement CPU SiLU F32 and F64 through one shared arithmetic body used by both allocating Kernel and IntoKernel; keep existing Execute results and autograd behavior unchanged. Correctness tests cover parity, exact destination identity, dtype, shape, device, layout, released storage, alias, duplicate outputs, recorder rejection, unsupported operation, routing and fallback behavior, and concurrent calls with independent outputs. Performance gate on Apple M2 Pro at [256,1408] F64: ExecuteInto reports zero bytes and zero allocations per operation, beats ordinary Execute by at least 1.5x, and beats the local PyTorch aten.silu.out median. Preserve the already proven SIMD accuracy and vector-tail identity. Research parent: R-01M00MRVYDFVG. Standing constraint: PERF-OP-OUTPUT-ALLOCATION-001.

## T-01M00N1614E0JTWGQP5ERH3D1X Implement allocation-free ExecuteInto for CPU SiLU
kind: task
state: done
created: 2026-08-14
parent: P-01M00MX92BE1ERNRWY3VPJZB80
grilled: 2026-08-14 open=1
targets: backend/backend.go, go:backend.Execute, backend/execute_test.go, backend/cpu/cpu.go, go:cpu.siluKernelCPU, backend/cpu/vsilu_f64_bench_test.go, docs/perf-notes-cpu.md, spec/70-tasks.md

Implement proposal P-01M00MX92BE1E and consume research R-01M00MRVYDFVG. Scope: backend/backend.go, backend/execute.go, backend execution tests, backend/cpu/cpu.go, backend/cpu/elementwise.go, backend/cpu SiLU tests and benchmarks, docs/perf-notes-cpu.md, and legacy SPEC task records only; do not introduce global pooling, finalizers, implicit Release, tensor refcounts, or unrelated kernel migrations. Add public IntoKernel and IntoBackend capability contracts plus ExecuteInto(ctx, op, inputs, outputs, attrs) error. Extract shared kernel resolution from Execute so regular and into execution preserve op routing, CPU then reference fallback, attrs gates, dtype selection, and diagnostics. ExecuteInto must reject recorder contexts, nil and released outputs, non-contiguous or offset outputs, device-kind mismatch, output-input storage aliases, duplicate output storage, and unsupported into kernels before invoking arithmetic. Implement CPU SiLU F32 and F64 into capability using one shared apply function so Execute and ExecuteInto run identical arithmetic. Validate exact output count, dtype, shape, and storage length before writes. Tests: go test ./backend ./backend/cpu; GOEXPERIMENT=simd CGO_ENABLED=0 go test ./backend ./backend/cpu; CGO_ENABLED=0 go test ./... with known unrelated failures recorded separately; race-test concurrent independent outputs when practical. Benchmarks: at F64 [256,1408] on Apple M2 Pro use prebuilt input and output slices, ReportAllocs, alternating repeated runs, and GOMAXPROCS values matching the PyTorch comparison. Acceptance is 0 B/op and 0 allocs/op for ExecuteInto, at least 1.5x median speedup over Execute, and faster than the measured local PyTorch aten.silu.out median 196.35 us. Update the performance note and add a legacy task entry through internal/docgraph; run spectackle check and external perfscan before completion.

## R-01M00QYSTCFM09R66AZK79ZCPN Prioritize publishable leadership and M2-native Metal
kind: research
state: done
created: 2026-08-14
targets: BENCHMARKS.md, docs/benchmarking.md, backend/metal, backend/cuda, backend/vulkan, go:backend.Execute

Primary-source synthesis, 2026-08-14. Benchmark leadership must be a matrix over hardware, model/shape, dtype/quantization, batch/context, and warm/cold state; each cell holds semantics, quality, workspace, transfer boundaries, and immutable baseline versions. Go benchstat requires repeated samples and recommends interleaved prebuilt runs; MLPerf separates scenarios and accuracy constraints. M2 is Apple GPU family 8 and supports SIMD-scoped matrix multiply because that capability starts at Apple7. The native Metal path is the M2 peak path; MLX and llama.cpp expose concrete Metal JIT/kernel, quantized matmul, resident-resource, and attention implementations worth auditing. Local history changes the action: GoAI already has Metal/Vulkan Q4_K, Flash/cooperative attention, resident resources, shape-keyed MPSGraph caches, CUDA paged attention, and batched decode. Do not rebuild these features. First profile current end-to-end paths against pinned MLX and llama.cpp, then force-disable the suspected cost before selecting one leaf. CPU work remains profile-driven zero-allocation, dispatch/fusion, and fixed-shape or quantized kernels; use layered packing/microkernel/cache-blocking only where a measured shape exposes a gap. RTX 3060 work is later and hardware-bound: compare standard GEMM against reusable cuBLASLt heuristics, then target measured small-M quantized decode or whole-step graph/fusion gaps. Ampere compute capability 8.6 has distinct occupancy/shared-memory constraints, so Blackwell kernels are not a baseline. Vulkan is a portability backend with enumerated capability tiers and cache identity including device, driver, pipeline UUID, shape, dtype, and shader hash; native Metal remains the M2 winner path. Graph IR, fusion groups, shape/dtype/device kernel registry, AOT/JIT specialization, autotuning, persistent cache, and portable fallback are a future architecture only after an end-to-end profile proves dispatch or fusion overhead is the floor. Generalizable measured single-last-shape compiled-cache thrashing was filed as perfscan issue 559 (GoAI evidence: 7.2x Conv2D and 4.4x attention wins after a bounded shape-keyed cache). Sources: https://pkg.go.dev/golang.org/x/perf/cmd/benchstat ; https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc ; https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf ; https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal ; https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal ; https://docs.nvidia.com/cuda/ampere-tuning-guide/ ; https://docs.nvidia.com/cuda/cublas/ ; https://docs.vulkan.org/guide/latest/subgroups.html ; https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html ; https://iree.dev/reference/tuning/ ; https://arxiv.org/abs/2307.08691

## P-01M00R3TH1EW78X2Z8GDP4Q879 Run an M2-first measured accelerator frontier
kind: proposal
state: approved
created: 2026-08-14
parent: R-01M00QYSTCFM09R66AZK79ZCPN
grilled: 2026-08-14 open=0
targets: BENCHMARKS.md, docs/benchmarking.md, backend/metal, backend/cuda, backend/vulkan, go:backend.Execute

Sequence the accelerator frontier without reopening the completed ExecuteInto slice. Phase 1 establishes a publishable leadership matrix and benchmark protocol on M2: immutable competitor versions, identical model/shape and semantics, dtype/quantization, batch/context, warm/cold state, workspace/transfers, quality, and at least 10 interleaved prebuilt samples analyzed by benchstat. Phase 2 profiles existing native Metal end to end against pinned MLX and llama.cpp. Audit before building because GoAI already contains Q4_K, Flash/cooperative attention, resident buffers, and shape-keyed graph caches. Use forced-off A/B to choose one leaf among quantized decode, attention IO, command/graph persistence, cache behavior, or ViT batching. Phase 3 implements only the measured M2 winner with parity and regression benchmarks; if current GoAI leads every comparable cell, open the next missing AI area instead. Later hardware tracks remain ordered behind M2 evidence: RTX 3060 first benchmarks standard GEMM against reusable cuBLASLt heuristics, then considers small-M quantized decode and full-step CUDA Graph/fusion; Vulkan receives enumerated capability tiers and complete persistent-cache identity; graph/IR/JIT/autotuning starts only after a profile proves dispatch, fusion, or specialization is the floor. Explicit exclusions: MoltenVK as the M2 peak path, M5/A19 neural-accelerator assumptions on M2, direct Blackwell FA4 transplantation to Ampere, a universal Vulkan shader, and speculative repository-wide compiler redesign. Sources and measured local delta are captured by R-01M00QYSTCFM0 and perfscan issue 559.

## T-01M00R8FRKEDYTV3RNK6R9D0AK Profile the RTX 3060 Ampere winner zones
kind: task
state: draft
created: 2026-08-14
parent: P-01M00R3TH1EW78X2Z8GDP4Q879
targets: backend/cuda, llamagpu, BENCHMARKS.md, docs/benchmarking.md

Priority: medium and hardware-bound; queue behind the M2 winner. On the RTX 3060 worker, pin CUDA, driver, cuBLASLt, compiler, llama.cpp, and GoAI revisions. Benchmark standard GEMM first through reusable cuBLASLt heuristic results and retain it whenever custom code does not win. Profile existing GoAI paged attention and batched decode before adding kernels. Candidate custom zones are measured small-M quantized decode, fused dequantization plus matmul, RMSNorm/SwiGLU/RoPE plus KV fusion, and a CUDA Graph over the complete decode step. Tune for compute capability 8.6 limits and do not transplant Blackwell-only FA4 assumptions. Acceptance: a leadership-matrix row and forced-off A/B select one leaf; implementation requires parity, Nsight evidence, and interleaved incumbent comparison.

## T-01M00R9319FTR8T8Z0NQ6VAGXK Define measured Vulkan capability and cache tiers
kind: task
state: draft
created: 2026-08-14
parent: P-01M00R3TH1EW78X2Z8GDP4Q879
targets: backend/vulkan, llamagpu, BENCHMARKS.md, docs/benchmarking.md

Priority: medium; portability track after the M2 winner. Audit current Vulkan capability checks, cooperative-matrix path, shader variants, pipeline caches, and device lifecycle. Define enumerated tiers: portable Vulkan 1.3; subgroup, FP16, and integer-dot support; KHR cooperative-matrix properties; then vendor-specific paths. Keep offline SPIR-V with runtime specialization and autotuning only where measured. Persistent cache identity must include GPU, driver, pipelineCacheUUID, shape, dtype, and shader hash. Do not build one universal shader and do not use MoltenVK as the M2 peak path. Acceptance: capability tests choose a correct fallback on every tier, cache invalidation is deterministic, and each specialized path owns a same-device benchmark plus parity evidence.

## T-01M00R9M8XF03TPPZ3HKQWCSM1 Gate graph IR and autotuning on a proven floor
kind: task
state: draft
created: 2026-08-14
parent: P-01M00R3TH1EW78X2Z8GDP4Q879
targets: backend, BENCHMARKS.md, docs/benchmarking.md, docs/decisions

Priority: conditional; no implementation until a prior end-to-end profile satisfies PERF-ARCHITECTURE-GATE-001. If dispatch, fusion, or specialization is a proven floor, design the smallest staged architecture: Go frontend to graph or IR and fusion groups; shape, dtype, and device keyed kernel registry; AOT hot paths plus bounded JIT specializations; offline or runtime tuner; persistent cache; correct portable fallback. Use MLX, PyTorch Inductor/Triton, cuBLASLt, CUTLASS, and IREE as pinned design references rather than dependencies by analogy. Start with one measured operation chain and keep existing eager APIs and backends working. Acceptance: an ADR records the profile, forced-off A/B, cache identity, invalidation, fallback, and rollback; a prototype must beat eager execution under the leadership protocol before broader migration.

## T-01M00R6G6EFJKSMZB5MJ55NC8Y Build the publishable M2 leadership matrix
kind: task
state: done
created: 2026-08-14
parent: P-01M00R3TH1EW78X2Z8GDP4Q879
grilled: 2026-08-14 open=1
targets: BENCHMARKS.md, docs/benchmarking.md, internal/benchcompare

Priority: highest, first after completed T986. Turn BENCHMARKS.md and the companion harness into a publishable M2-first leadership matrix. Each cell records hardware and OS, immutable GoAI and incumbent commit or release, compiler/runtime, model and exact shape, dtype or quantization, batch and context, warm or cold state, workspace and transfer boundaries, semantic and quality parity, latency or throughput, bytes, and allocation or memory metrics where applicable. Use identical inputs and output checks. Compile before timing, collect at least 10 interleaved samples, and report benchstat median/confidence evidence. Start with the existing M2 CPU SiLU and llama.cpp decode rows; do not claim global leadership from uncovered cells. Acceptance: committed reproducible commands or scripts can regenerate every populated row and unmeasured cells remain explicitly open.

## T-01M00R71HNET0A8NR0AS6HNA6G Profile native Metal against pinned MLX and llama.cpp
kind: task
state: done
created: 2026-08-14
parent: P-01M00R3TH1EW78X2Z8GDP4Q879
targets: backend/metal, llamagpu, BENCHMARKS.md, docs/benchmarking.md, internal/benchcompare

Priority: high; begins only after T-01M00R6G6EFJK establishes the comparison protocol. Profile the current end-to-end M2 native Metal paths against immutable MLX and llama.cpp baselines for identical model, quantization, prompt, and decode geometry. Audit existing GoAI Q4_K and other quantized matvec kernels, Flash/cooperative attention, resident resources, command reuse, graph and pipeline caches, and ViT batching before proposing code. Measure prefill and decode separately, include warm and cold states, and use Metal profiling plus same-session forced-off A/B to distinguish arithmetic, memory traffic, command submission, compilation, and host dispatch. MoltenVK is not a peak baseline. Acceptance: a ranked bottleneck table identifies one highest-leverage bottom-up leaf with reproducible evidence, or proves every comparable cell leads and names the next uncovered AI area.

## T-01M00R7KE5F688F41NDC5G8K21 Implement the measured M2 Metal winner
kind: task
state: done
created: 2026-08-14
parent: P-01M00R3TH1EW78X2Z8GDP4Q879
targets: backend/metal, llamagpu, BENCHMARKS.md, docs/benchmarking.md

Priority: high; begins only after T-01M00R71HNET0 selects the measured leaf. Implement exactly one M2-native Metal winner from the profile, with a portable correct fallback and the smallest capability boundary. Candidate areas are fused Q4_K dequantization plus matvec, IO-aware attention, persistent command or graph execution, pipeline or buffer cache behavior, or ViT batching and fusion, but the profile decides and existing features must not be rebuilt. Preserve result quality and ownership semantics, add isolated parity and edge tests, and benchmark at least 10 interleaved prebuilt samples. Report every newly identified reusable optimization pattern to perfscan with measurements and false-positive boundaries. Acceptance: the selected leadership-matrix cell beats the pinned incumbent or records a measured rejection and advances to the next ranked leaf; if all comparable cells already lead, create the next-area proposal instead.

## T-01M00X1S9RF9K8Z56XWXMQ41GB Measure and implement cooperative M2 Q6_K decode
kind: task
state: done
created: 2026-08-14
parent: P-01M00R3TH1EW78X2Z8GDP4Q879
targets: backend/metal/metal_bridge.m, backend/metal/qmatmul_test.go, internal/benchcompare/leadership

Measure resident M=1 Q6_K exact TinyLlama projection shapes against the scalar forced-off path with at least ten paired alternating samples. Implement a SIMD-group cooperative kernel only in measured winning cells, retain M>1 and unsupported-device fallback, cross-reference scalar and f64 GGUF truth, update leadership evidence and docs, and report reusable perfscan findings.

## T-01M00YJZTREKTRP5ZQQEM0NBTS Profile the next M2 decode bottleneck after Q4_K and Q6_K
kind: task
state: done
created: 2026-08-14
parent: P-01M00R3TH1EW78X2Z8GDP4Q879
targets: objc:metal_bridge.mtl_recorder_qmatmul, backend/metal/metal_bridge.m, internal/benchcompare/prod_decode_external_test.go, internal/benchcompare/leadership

Measure quant-type incidence and full-step Metal execution after the cooperative Q4_K/Q6_K wins. Separate GPU kernel time, command encoding/submission, host synchronization, attention, norms, and model-scale throughput. Select one implementation only if the measured leaf has model incidence and an operation/end-to-end leverage path; do not add Q2_K/Q3_K/Q5_K work for the TinyLlama Q4_K_M baseline because the file contains no such tensors. Preserve pinned llama.cpp/MLX baselines and leadership-matrix boundaries.
