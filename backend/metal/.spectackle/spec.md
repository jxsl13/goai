---
schema: v1
---

## intent
- R-01M08T14CKES181P11RG303W1H M2 Q4_K half-MMA improves the old prototype but is 0.4018x of cached-f16 MPS: Consumed by rejection of T-01M08SPZPNEG6 and P-01M08SN9PFECD. The half-input float-accumulate Q4_K simdgroup candidate preserved finite trained-like parity (max relative error 2.947e-4) but measured 960.125 us versus 385.750 us for the production cached-f16 MPS path at M64 K2048 N5632, only 0.4018x baseline throughput and far below the 1.10x promotion gate. All executable and test changes were ful [body truncated at tombstone retention cap]
- R-01M08VQWMFEB9S9JCJ8NVSTYC2 Reject the M2 legacy Q4_K 64x32 tile against the production cache: Closed no-action after the leaf gate failed. The complete pinned llama.cpp legacy tile was finite but missed the declared cached-f16 numerical gate and reached only 0.741x/0.746x production throughput at M64 in two fresh processes. The old uncached-f32 comparator would have reported a false win. The executable experiment was fully reverted; the generalized second-reuse-axis detector is tracked in [body truncated at tombstone retention cap]
- R-01M08XC0VNF9RVA1AG1HF28714 Reject M2 MPSGraph quant projection fusion despite broad-shape speedups: Closed no-action and consumed by the rejection of T-01M08WVXA2F6W and P-01M08WKNGEF49. MPSGraph delivered 1.152x-1.381x on broad M64 projection leaves but changed reduction results far beyond the 5e-4 gate; two explicit rounding/accumulation repairs failed. All executable changes were reverted. The generalized parity-before-timing guard is jxsl13/perfscan#761.
- T-01M0FNQVFTFQ0ABX96T9QM18R4 Gate deterministic host Metal embedding backward: Implemented deterministic host-resident F32 scatter with reference-order rounding; preserved the old Metal atomic route as a mutation-proven benchmark control. Three independent seven-sample M2 campaigns cleared all five frozen shapes, worst 3.931x and best 30.762x. Cross-reference, preflight, full Metal lane, focused perfscan, and all 15 PR #1105 CI checks passed. Evidence: internal/benchcompare/ [body truncated at tombstone retention cap]
- T-01M0FVGM88EWMRQCHFN4B748AV Gate Metal bias-gradient routing against the optimized CPU kernel: Add a mutation-proven direct-Metal benchmark control and production selector over the frozen F32 shape matrix. Run three independent count-7 campaigns, reject unstable timing before routing, pin both selector arms, strengthen contiguous and noncontiguous reference parity, run an end-to-end GPT training-step no-regression gate, and retain only a measured winner zone. Record reproducible evidence an [body truncated at tombstone retention cap]
- T-01M0GZFS4DFY08YRAND3TRHB5N Implement and gate Q2_K scalar-word loads: Implemented and validated default-on M2 Metal Q2_K scalar-word quant loads for eligible M=1 decode shapes. Replaced eight indexed uchar reads per lane with two aligned scalar uint loads while preserving byte order and accumulation semantics; uint2/ulong were excluded because the 84-byte block stride guarantees only 4-byte alignment. Production routing requires K*N >= 6291456 and preserves the hist [body truncated at tombstone retention cap]
- P-01M0GZESBAFGS8J03YXK0YS5YF Pack Q2_K quant bytes into aligned uint loads: Delivered the M2-first Q2_K word-load optimization through task T-01M0GZFS4DFY0. The aligned scalar-uint design is numerically equivalent within 2.068e-6 max scalar relative error, guarded by an explicit K*N >= 6291456 decode threshold, and measured 1.300x-1.788x faster across five eligible production shapes in each of three independent AB/BA campaigns; fallback routing stayed within 0.983x-1.016x [body truncated at tombstone retention cap]
- T-01M0M9BPF2FW68JAGG2XZ8F917 Implement and benchmark native Metal Q4_1: Implemented exact GGUF type-3 Q4_1 Metal support with scalar and two-SIMD-group cooperative kernels, resident recorder dispatch, and explicit llamagpu upload. Three fresh-process campaigns across four production shapes retained the cooperative route at minimum 2.745x GPU and 1.462x Metal/CPU wall; whole-model decode improved 72.00 to 182.52 token/s (2.535x). Generic host-bound dispatch deliberatel [body truncated at tombstone retention cap]
- P-01M0M9B6FRFCZA18408PMM2WGH Add native M2 Metal Q4_1 quantized matmul: Delivered native resident Q4_1 as the next M2 bottom-up quantized decode tranche. The exact affine wire format and cooperative decoder path are production-reachable through llamagpu, while generic host-I/O execution remains on the measured-faster ARM64 kernel. Minimum campaign medians were 2.745x cooperative/scalar GPU and 1.462x Metal/CPU wall across four production shapes; TinyLlama-shaped end-t [body truncated at tombstone retention cap]
- R-01M0N4TXJQE4NTD97E585SHFTP Measure production-shape M2 Q6_K roofline and geometry leverage: Consumed by P-01M0N4ZCJ1FRN and T-01M0N51GZ1FC1. The permanent GPU-timestamp roofline covers Q6_K K2048 N2048, K2048 N5632, K5632 N2048, and cache-busting K2048 N131072. Initial measurements were 191.3, 214.6, 208.0, and 157.0 GB/s, but repeated campaigns corrected the final cell to 184-190 GB/s and showed no stable rows-per-SIMD leverage. One-row specialization remained 0.986x to 1.054x control; [body truncated at tombstone retention cap]
- P-01M0NSPZ36FZKRC1N1CFER4XCG Stabilize the fused RoPE F16-KV performance gate: Archive retry after the child task completed and Git index access was granted. The revised performance contract, deterministic threshold coverage, and 12-of-12 M2 stability evidence are complete.
- R-01M0Q4QAP0ESS9GBF2ERVRBHAM Compare pinned upstream M2 quant dot instruction shapes: Consumed by P-01M0Q4TE2ZE52 and T-01M0Q4V43GEB3. The pinned comparison isolated explicit full unrolling as the only untested Q4_K instruction-shape delta; MLX affine quantization was not transferable.
- R-01M0Q59EV4EKSTBJACTNCBQK42 Compare pinned upstream M2 Q6_K instruction shapes: Consumed by P-01M0Q5CKW6FH8 and T-01M0Q5DGB6FMB. At pinned llama.cpp b0539c43ed13b16bf0d8a0840646faea65469702, Q6_K matches GoAI lane ownership, loads, four-plane reconstruction, scaling, row stepping, and SIMD reduction; only forced full unrolling differs. Its int8 cast is value-equivalent for codes 0 through 63. Pinned MLX d9077d8316ad7305497a3ecf2296bd0e0e99a627 has no transferable GGUF Q6_K ke [body truncated at tombstone retention cap]
- R-01M0Q69J1CEDCBKG8VDVB580GS Attribute Metal Recorder.Profile event-label allocations: Consumed by proposal P-01M0Q6B7A3E1S and task T-01M0Q6C2RAF6E. External perfscan PS2004 attributed 340 per-event 96-byte scratch allocations in Recorder.Profile; benchmark gates separate the scratch-hoist experiment from any later bulk-cgo or label-interning work.
- R-01M0Q6SJ8CF6R9984KYGJ8AQ8M Attribute bulk Metal profile extraction leverage: Consumed by proposal P-01M0Q6T9RZEV9. The scratch-only result isolated repeated cgo crossings as the remaining warm extraction cost; the successor must benchmark native snapshot construction on first extraction as well as cached repeat extraction.
- T-01M0Q6VRADE45AJWQRNNN3JZNC Implement and gate bulk Metal profile snapshots: Implemented additive mtl_recorder_profile_snapshot with a recorder-owned immutable event snapshot, inline one-event storage, compact unboxed valid indices, and one cgo extraction in Recorder.Profile. Existing profile semantics and legacy C entry points pass. Three order-alternated count-seven M2 campaigns measured warm speedups of 2.44-2.55x at 1 event and 17.91-18.23x at 340 events; first extract [body truncated at tombstone retention cap]
- P-01M0Q6T9RZEV9RWA9ZS5V8KGYC Bulk-extract Metal recorder profile events in one cgo call: Promoted bulk Metal profile extraction after every frozen semantic, warm, cold, and allocation gate passed in three order-alternated Apple M2 Pro campaigns. The final design combines one cgo snapshot call, immutable recorder-lifetime native event storage, an inline single-event fast path, and zero-NSNumber compact valid indices while preserving legacy C ABI functions. Durable contracts are RECORDE [body truncated at tombstone retention cap]
- T-01M0Q9H5G2FP38JV8VMPSFP297 Implement and gate extraction-scoped profile label deduplication: Implemented a contiguous native label-token sidecar, 16-entry native label-copy reuse, extraction-scoped Go ownership with content fallback, and an explicit count-one fast path. Three final independent-process count-seven AB/BA-alternated M2 campaigns measured warm repeated-label speedups of 2.23x, 2.06x, and 2.53x; mixed-ten-label speedups of 1.86x, 1.67x, and 1.75x; cold repeated-label speedups [body truncated at tombstone retention cap]
- R-01M0Q913MDFQSTD3ME8MTRCGRZ Recorder profile label ownership and allocation scaling: Consumed by P-01M0Q9EZJ8ENC. The initial recorder-scoped cache was rejected because it enlarged the production Recorder; extraction-scoped deduplication plus a native identity-token sidecar preserved default and one-event paths while eliminating per-event Go string ownership.
- P-01M0Q9EZJ8ENCSJ4YR1J41MDGD Deduplicate labels within each Recorder.Profile extraction: Promoted after T-01M0Q9H5G2FP3 passed every frozen M2 gate. Multi-event labels now use native identity tokens plus exact-content fallback and one Go-owned clone per distinct label; native snapshot materialization reuses up to 16 labels. The one-event ABI remains unchanged. Final campaign and allocation results are recorded on the archived task and perfscan issue 855.

## METAL-RESIDENT-TOPK-001
WHEN TopKN is called with valid n and k on a live f32 DeviceBuffer, the Metal resident selection boundary SHALL return k distinct first-n index/value pairs matching the host top k, ordered by descending value then ascending index.

Rationale: Exact deterministic selection preserves the existing sampler draw.

## METAL-RESIDENT-TOPK-TRANSFER-001
WHEN TopKN returns k candidates from n resident logits, the Metal sampling boundary SHALL copy and allocate O(k) result data in Go without materializing the n logits.

Rationale: The optimization exists to eliminate the measured full-vocabulary host boundary.

## HOST-RESIDENT-EMBED-BACKWARD-001
WHEN receives a valid host-resident F32 table, index vector, and upstream gradient, the Metal OpEmbedBackward SHALL execute exactly 0 Metal command submissions and return the deterministic reference-order scatter-add gradient.

## MEASURED-METAL-BIAS-GRAD-ROUTE-001 {applies: go:metal.addBiasBackwardF32}
WHEN a synchronous host-resident F32 bias gradient is requested, the Metal add-bias backward SHALL route through CPU only where 3 count-7 campaigns each prove at least 1.10x median speedup, and preserve direct Metal elsewhere.

Rationale: A later exact CPU reduction can invalidate an older synchronous GPU route, but the winner zone must remain measurement-bounded.

## MEASURED-METAL-BIAS-ROUTE-001
WHEN a synchronous host-resident F32 bias add is requested, the Metal host wrapper SHALL route through CPU only where 3 count-7 campaigns each prove at least 1.10x median speedup and preserve direct Metal elsewhere.

Rationale: A later optimized CPU broadcast kernel can invalidate an older synchronous GPU route, but the winner zone must remain measurement-bounded.

## METAL-Q5K-WIDE-LOAD-PERF-001
WHEN a resident M=1 Q5_K vector-load kernel is considered for production on Apple M2, the Metal Q5_K selector SHALL retain it only when every representative shape in three independent count-7 same-binary campaigns reaches at least 1.10x the current cooperative kernel.

## METAL-Q5K-WIDE-LOAD-NUMERIC-001
WHEN it executes a valid resident M=1 projection, the Metal Q5_K vector-load kernel SHALL match the current cooperative output within 2e-5 relative error across K=256 through K=4096, preserve NaN class, and leave activation and weight inputs unchanged.

## METAL-Q5K-WIDE-LOAD-SCOPE-001
WHEN M is greater than 1 or the device lacks a 32-lane SIMD group with 64-thread threadgroups, the Metal Q5_K selector SHALL execute the existing Q5_K path with zero vector-load-candidate dispatches.

## METAL-Q5K-WIDE-LOAD-THRESHOLD-001
WHEN a Q5_K cooperative pipeline is selected, the Metal Q5_K selector SHALL dispatch the aligned per-lane uint2 vector-load candidate only when M equals 1 and K times N is at least 6291456; otherwise dispatch the historical cooperative pipeline.

Rationale: The steady-state M2 boundary sweep measured 1.117x at 2048x3072 but only 1.055x at 2048x2048; per-lane coalesced uint2 loads also beat the simd_shuffle sharing variant on eligible FFN cells.

## METAL-Q4K-WIDE-LOAD-PERF-001
WHEN a resident M=1 Q4_K vector-load candidate is considered for production on Apple M2, the Metal Q4_K selector SHALL retain it only when every eligible shape in three independent count-7 same-binary campaigns reaches at least 1.10x the cooperative kernel.

Rationale: A packed-load microbenchmark is promotable only as a repeatable shape-bounded production win.

## METAL-Q4K-WIDE-LOAD-NUMERIC-001
WHEN it executes a valid resident M=1 projection, the Metal Q4_K vector-load kernel SHALL match the current cooperative output within 2e-5 relative error, preserve finite Inf and NaN classification, and leave activation and weight inputs unchanged.

Rationale: Changing load width must not change Q4_K bytes, arithmetic semantics, exceptional-value behavior, or ownership.

## METAL-Q4K-WIDE-LOAD-SCOPE-001
WHEN M is greater than 1 or the device lacks a 32-lane SIMD group with 64-thread threadgroups, the Metal Q4_K selector SHALL execute the existing Q4_K path with zero vector-load-candidate dispatches.

Rationale: The experiment is an M2 M=1 specialization and must retain the portable historical fallback.

## METAL-Q6K-PACKED-LOAD-PERF-001-001
WHEN three independent count-seven same-binary M2 campaigns cover every representative shape, the Metal Q6_K packed-load promotion SHALL the system shall retain the candidate only when every eligible shape is at least 1.10x control.

Rationale: A narrow or noisy speedup cannot justify another permanent kernel route.

## METAL-Q6K-PACKED-LOAD-NUMERIC-001-001
WHEN the candidate processes finite or nonfinite inputs, the Metal Q6_K packed-load candidate SHALL the system shall preserve value error within 2e-5, floating-point class, and input immutability.

Rationale: Load packing must not change observable numerical semantics.

## METAL-Q6K-PACKED-LOAD-ALIGNMENT-001-001
WHEN a packed byte-plane load is issued across 210-byte blocks or rows, the Metal Q6_K packed-load candidate SHALL the system shall use only ushort pointer loads and never require wider alignment.

Rationale: Q6_K guarantees two-byte but not four-byte alignment across consecutive blocks.

## METAL-Q6K-PACKED-LOAD-SCOPE-001-001
WHEN M exceeds one or packed-load support is unavailable, the Metal Q6_K dispatch SHALL the system shall use the historical route and issue zero candidate dispatches.

Rationale: The experiment is scoped to supported single-token M2 decode.

## METAL-Q3K-SCALE-BROADCAST-PERF-001
WHEN three independent count-seven M2 campaigns cover every representative shape, the Metal Q3_K scale-broadcast selector SHALL retain the candidate only when every eligible shape reaches at least 1.10x control.

Rationale: Uniform-header elimination must produce broad end-to-end leverage.

## METAL-Q3K-SCALE-BROADCAST-NUMERIC-001
WHEN the candidate processes finite or nonfinite inputs, the Metal Q3_K scale-broadcast kernel SHALL match control within 2e-5, preserve finite Inf and NaN class, and mutate zero input bytes.

Rationale: Broadcasting scale metadata must preserve observable quant semantics.

## METAL-Q3K-SCALE-BROADCAST-ALIGNMENT-001
WHEN it reads a 12-byte scale header across 110-byte blocks or rows, the Metal Q3_K scale-header loader SHALL issue exactly six ushort loads and zero uint-or-wider device-pointer loads.

Rationale: Q3_K guarantees two-byte but not four-byte alignment.

## METAL-Q3K-SCALE-BROADCAST-SCOPE-001
WHEN M exceeds one or scale-broadcast support is unavailable, the Metal Q3_K dispatch SHALL select the historical pipeline and issue zero candidate dispatches.

Rationale: The experiment is limited to supported single-token decode.

## METAL-Q2K-WORD-LOAD-PERF-001
WHEN three independent count-seven M2 campaigns cover every representative shape, the Metal Q2_K word-load selector SHALL retain the candidate only when every eligible shape reaches at least 1.10x control.

Rationale: Packed lane-unique loads must produce broad end-to-end leverage.

## METAL-Q2K-WORD-LOAD-NUMERIC-001
WHEN the candidate processes finite or nonfinite inputs, the Metal Q2_K word-load kernel SHALL match control within 2e-5, preserve finite Inf and NaN class, and mutate zero input bytes.

Rationale: Word extraction must preserve observable quant semantics.

## METAL-Q2K-WORD-LOAD-ALIGNMENT-001
WHEN it reads eight lane-unique bytes across 84-byte blocks or rows, the Metal Q2_K quant-plane loader SHALL issue exactly two uint loads and zero uint2-or-wider device-pointer loads.

Rationale: Q2_K guarantees four-byte but not eight-byte alignment.

## METAL-Q2K-WORD-LOAD-SCOPE-001
WHEN M exceeds one or word-load support is unavailable, the Metal Q2_K dispatch SHALL select the historical pipeline and issue zero candidate dispatches.

Rationale: The experiment is limited to supported single-token decode.

## METAL-Q2K-WORD-LOAD-THRESHOLD-001
WHEN it evaluates an M=1 decode shape, the Metal Q2_K cooperative selector SHALL select scalar-word loads only when K times N is at least 6291456; otherwise retain control.

Rationale: The pilot showed broad gains beginning at K2048,N3072 while smaller cells were unstable.

## METAL-Q4-0-PAIR-LOAD-PERF-001
WHEN three independent count-seven M2 campaigns cover every eligible and fallback shape, the Metal Q4_0 pair-load selector SHALL retain the candidate only when every eligible median is at least 1.10x control and every fallback ratio is between 0.97x and 1.03x.

## METAL-Q4-0-PAIR-LOAD-NUMERIC-001
WHEN the pair-load candidate processes finite or nonfinite inputs, the Metal Q4_0 pair-load kernel SHALL match control within 2e-5 relative error, preserve finite, Inf, and NaN class, and mutate exactly zero input bytes.

## METAL-Q4-0-PAIR-LOAD-ALIGNMENT-001
WHEN the candidate reads one 16-byte Q4_0 quant plane across 18-byte blocks or rows, the Metal Q4_0 quant-plane loader SHALL issue exactly eight aligned ushort device loads and zero uint-or-wider device-pointer loads per SIMD group.

## METAL-Q4-0-PAIR-LOAD-SCOPE-001
WHEN M exceeds one, support is unavailable, or the pair-load toggle is disabled, the Metal Q4_0 dispatch SHALL select the historical pipeline and issue exactly zero pair-load candidate dispatches.

## METAL-Q4-0-PAIR-LOAD-THRESHOLD-001
WHEN it evaluates an M=1 Q4_0 decode shape, the Metal Q4_0 cooperative selector SHALL select pair loads only when K times N is at least 6291456; otherwise retain control.

## METAL-Q4-1-BLOCK-001
WHEN a GGUF type-3 Q4_1 block is decoded, the Metal kernel SHALL decode each 20-byte block as f16 d and m plus sixteen split-half nibble bytes, reconstructing d times q plus m for q from zero through fifteen.

Rationale: This is the exact GGUF Q4_1 wire contract.

## METAL-Q4-1-DISPATCH-001
WHEN host-bound QuantMatMul or UploadQuant receives type-3 Q4_1, the Metal backend SHALL return ErrQuantUnsupported while explicit Q4_1 and llamagpu resident recorder APIs provide native dispatch.

Rationale: M2 measurements show standalone Metal submissions lose to the ARM64 fused CPU kernel; recorder residency amortizes submission boundaries.

## METAL-Q4-1-NUMERIC-001
WHEN a valid Q4_1 matmul executes, the Metal kernel SHALL match gguf.QMatMul within 2e-5 relative error, preserve floating-point class, and mutate zero activation or weight bytes.

## METAL-Q4-1-FALLBACK-001
WHEN M exceeds the cooperative limit or SIMD-group requirements are unavailable, the Metal Q4_1 selector SHALL dispatch the scalar Q4_1 pipeline and issue zero cooperative Q4_1 threadgroups.

## METAL-Q4-1-PERF-001
WHEN three count-seven M2 campaigns compare cooperative and scalar resident single-token Q4_1 across representative shapes, the cooperative Metal route SHALL remain enabled only if every eligible median is at least 1.10 times faster with identical allocation semantics.

Rationale: The retained leverage is SIMD-group occupancy inside a resident recorder, not host-bound dispatch.

## METAL-Q4-1-HOST-ROUTE-001
WHEN M2 host input and output benchmarks do not beat ARM64 Q4_1 by at least 1.10 times, the generic Metal Q4_1 dispatch SHALL return ErrQuantUnsupported so QuantLinear executes the faster CPU path.

## METAL-IQ4NL-BLOCK-001
WHEN a GGUF type-20 block is decoded, the Metal IQ4_NL kernel SHALL read one f16 scale and sixteen split-half nibble bytes, applying the verified 16-entry nonlinear codebook to exactly 32 values.

## METAL-IQ4NL-DISPATCH-001
WHEN host-bound QuantMatMul or UploadQuant receives type-20 IQ4_NL, the Metal backend SHALL return ErrQuantUnsupported while explicit IQ4_NL and llamagpu resident recorder APIs provide native dispatch.

## METAL-IQ4NL-NUMERIC-001
WHEN a valid IQ4_NL matmul executes, the Metal IQ4_NL kernel SHALL match gguf.QMatMul within 2e-5 relative error, preserve floating-point class, and mutate zero activation or weight bytes.

## METAL-IQ4NL-FALLBACK-001
WHEN M exceeds the cooperative limit or SIMD-group requirements are unavailable, the Metal IQ4_NL selector SHALL dispatch the scalar pipeline and issue zero cooperative IQ4_NL threadgroups.

## METAL-IQ4NL-PERF-001
WHEN three count-seven M2 campaigns cover representative resident single-token IQ4_NL shapes, the cooperative Metal route SHALL remain enabled only when every eligible median is at least 1.10 times scalar control with identical allocation semantics.

## METAL-IQ4NL-HOST-ROUTE-001
WHEN M2 equal host-boundary benchmarks do not beat ARM64 IQ4_NL by at least 1.10 times, the generic Metal IQ4_NL dispatch SHALL return ErrQuantUnsupported so QuantLinear executes the faster CPU path.

## METAL-IQ4XS-BLOCK-001
WHEN a GGUF type-23 IQ4_XS super-block is decoded, the Metal IQ4_XS kernel SHALL read one f16 super-scale, eight packed signed six-bit sub-scales, and eight split-half nonlinear nibble groups from exactly 136 bytes.

## METAL-IQ4XS-DISPATCH-001
WHEN host-bound QuantMatMul or UploadQuant receives type-23 IQ4_XS, the Metal backend SHALL return backend.ErrQuantUnsupported while explicit IQ4_XS and llamagpu resident recorder APIs provide native dispatch.

## METAL-IQ4XS-NUMERIC-001
WHEN a valid IQ4_XS matmul executes, the Metal IQ4_XS kernel SHALL match gguf.QMatMul within 2e-5 relative error, preserve floating-point class, and mutate zero activation or weight bytes.

## METAL-IQ4XS-FALLBACK-001
WHEN M exceeds the cooperative limit or SIMD-group requirements are unavailable, the Metal IQ4_XS selector SHALL dispatch the scalar IQ4_XS pipeline and issue zero cooperative IQ4_XS threadgroups.

## METAL-IQ4XS-PERF-001
WHEN three count-seven M2 campaigns cover representative resident single-token IQ4_XS shapes, the cooperative Metal route SHALL remain enabled only when every eligible median is at least 1.10 times scalar control with identical allocation semantics.

## METAL-IQ4XS-HOST-ROUTE-001
WHEN M2 equal host-boundary benchmarks do not beat ARM64 IQ4_XS by at least 1.10 times, the generic Metal IQ4_XS dispatch SHALL return backend.ErrQuantUnsupported so QuantLinear executes the faster CPU path.

## METAL-IQ3S-BLOCK-001
WHEN a GGUF type-21 IQ3_S super-block is decoded, the Metal IQ3_S kernel SHALL read one f16 scale, 64 low grid indices, eight high-bit bytes, 32 direct-sign bytes, and four packed sub-scale bytes from exactly 110 bytes.

## METAL-IQ3XXS-BLOCK-001
WHEN a GGUF type-18 IQ3_XXS super-block is decoded, the Metal IQ3_XXS kernel SHALL read one f16 scale, 64 grid indices, and eight packed sign-and-scale words from exactly 98 bytes.

## METAL-IQ3-GRID-RESIDENCY-001
WHEN an IQ3 Metal pipeline initializes, the Metal backend SHALL reconstruct the matching 256-by-4 or 512-by-4 grid once from gguf.Dequantize and retain exactly one immutable Metal buffer.

## METAL-IQ3-NUMERIC-001
WHEN valid IQ3_S or IQ3_XXS matrix multiplication executes, the Metal backend SHALL match gguf.QMatMul within 1e-4 relative error and mutate zero activation or compressed-weight bytes.

## METAL-IQ3-DISPATCH-001
WHEN direct, resident, or recorder IQ3 dispatch selects a pipeline, the Metal backend SHALL use one shared cooperative predicate per wire type and bind the matching persistent grid at buffer index 4.

## METAL-IQ3-PERF-001
WHEN three count-seven M2 campaigns cover four resident single-token geometries for one IQ3 type, the cooperative Metal route SHALL remain enabled only when every median is at least 1.10 times scalar control at GPU and host-command boundaries.

## METAL-IQ3-HOST-ROUTE-001
WHEN equal host-boundary M2 benchmarks do not beat fused ARM64 IQ3 by at least 1.10 times, the generic Metal IQ3 dispatch SHALL return backend.ErrQuantUnsupported so QuantLinear executes the faster CPU route.

## METAL-IQ2-XXS-BLOCK-001
The Metal IQ2_XXS decoder SHALL decode every 66-byte type-16 block as one f16 d plus eight 8-byte pairs, applying four 8-value grid indices, four seven-bit sign indices, and the high-nibble scale to exactly 256 values.

## METAL-IQ2-XS-BLOCK-001
The Metal IQ2_XS decoder SHALL decode every 74-byte type-17 block as one f16 d, thirty-two little-endian grid/sign words, and sixteen four-bit scales to exactly 256 values.

## METAL-IQ2-GRID-RESIDENCY-001
WHEN type 16 or 17 is first used, the Metal IQ2 runtime SHALL reconstruct its 8-value codebook through the public GGUF decoder exactly once and retain one immutable process-lifetime Metal buffer reused by direct, resident, and recorder paths.

## METAL-IQ2-DISPATCH-001
WHEN an M=1 supported IQ2 projection is encoded, the Metal IQ2 dispatcher SHALL select scalar or cooperative execution with one per-type predicate shared by direct, resident, and recorder paths, with mtl_iq2_*_cooperative_set proving distinct toggle arms.

## METAL-IQ2-PERF-001
WHEN the route is promoted on M2, the Metal IQ2 cooperative route SHALL exceed scalar by 1.10x for GPU and recorder wall time in every required cell across three fresh-process count-seven AB/BA campaigns.

## METAL-IQ2-HOST-ROUTE-001
IF either IQ2 format fails to beat the fused ARM64 CPU route by at least 1.10x in every required host cell and campaign, THEN the generic synchronous Metal quant dispatcher SHALL retain CPU fallback for that format.

## METAL-IQ1S-BLOCK-001
WHEN a GGUF wire type 19 IQ1_S super-block is decoded, the Metal IQ1_S decoder SHALL decode each 50-byte block as one f16 scale, thirty-two 11-bit indices, thirty-two sign or delta bits, and eight packed multipliers, producing exactly 256 values.

Rationale: Exact wire compatibility is required before performance promotion.

## METAL-IQ1M-BLOCK-001
WHEN a GGUF wire type 29 IQ1_M super-block is decoded, the Metal IQ1_M decoder SHALL decode each 56-byte block as four split-f16 scale words, thirty-two 11-bit indices, four high-bit bytes, and sixteen packed sub-scales, producing exactly 256 values.

## METAL-IQ1-GRID-RESIDENCY-001
WHEN wire type 19 or 29 is first used, the Metal IQ1 runtime SHALL reconstruct the shared 2048-by-8 ternary grid through gguf.Dequantize exactly once and retain one immutable process-lifetime Metal buffer.

## METAL-IQ1-NUMERIC-001
WHEN valid IQ1_S or IQ1_M matrix multiplication executes, the Metal IQ1 backend SHALL match gguf.QMatMul within 1e-4 relative error, preserve floating-point class, and mutate zero activation or compressed-weight bytes.

## METAL-IQ1-DISPATCH-001
WHEN direct, resident, or recorder IQ1 dispatch selects a pipeline, the Metal IQ1 backend SHALL select exactly one scalar or cooperative pipeline through a format-specific predicate and bind the persistent grid at buffer index 4.

## METAL-IQ1-PERF-001
WHEN three fresh-process count-seven M2 campaigns cover every representative resident single-token IQ1 shape, the cooperative route SHALL exceed scalar control by at least 1.10 times for GPU and recorder wall time in every required cell.

## METAL-IQ1-HOST-ROUTE-001
IF either IQ1 format fails to beat fused ARM64 CPU by 1.10 times in any required host cell or campaign, THEN the generic synchronous Metal quant dispatcher SHALL retain CPU fallback for that format.

## METAL-IQ2S-BLOCK-001
WHEN a GGUF wire type 22 IQ2_S super-block is decoded, the Metal IQ2_S decoder SHALL decode each 82-byte block as one f16 scale, thirty-two 10-bit grid indices, thirty-two direct sign bytes, and sixteen four-bit sub-scales, producing exactly 256 values.

## METAL-IQ2S-GRID-RESIDENCY-001
WHEN wire type 22 is first used, the Metal IQ2 runtime SHALL reconstruct the exact 1024-by-8 grid once through gguf.Dequantize and retain one immutable 2 KiB process-lifetime buffer.

## METAL-IQ2S-NUMERIC-001
WHEN valid IQ2_S matrix multiplication executes, the Metal IQ2_S backend SHALL match gguf.QMatMul within 1e-4 relative error, preserve floating-point class, and mutate zero activation or compressed-weight bytes.

## METAL-IQ2S-DISPATCH-001
WHEN direct, resident, or recorder IQ2_S dispatch selects a pipeline, the Metal IQ2_S backend SHALL select exactly one scalar or cooperative pipeline through one shared predicate and bind the persistent grid at buffer index 4.

## METAL-IQ2S-FALLBACK-001
WHEN M exceeds the cooperative limit or 32-lane SIMD groups are unavailable, the Metal IQ2_S selector SHALL dispatch the scalar pipeline and issue zero cooperative IQ2_S threadgroups.

## METAL-IQ2S-PERF-001
WHEN three fresh-process count-seven M2 campaigns cover every representative resident single-token IQ2_S shape, the cooperative IQ2_S route SHALL exceed scalar control by at least 1.10 times for GPU and recorder wall time in every required cell.

## METAL-IQ2S-HOST-ROUTE-001
IF direct host-bound IQ2_S fails to beat fused ARM64 CPU by 1.10 times in any required cell or campaign, THEN the generic Metal quant dispatcher SHALL return backend.ErrQuantUnsupported and preserve the fused ARM64 CPU route.

## METAL-TQ1-BLOCK-001
WHEN a GGUF wire type 34 TQ1_0 block is decoded, the Metal TQ1_0 kernel SHALL decode each 54-byte block as 48 five-trit base-243 bytes, 4 four-trit tail bytes, and 1 trailing f16 scale in the pinned 256-element order.

## METAL-TQ2-BLOCK-001
WHEN a GGUF wire type 35 TQ2_0 block is decoded, the Metal TQ2_0 kernel SHALL decode each 66-byte block as 64 two-bit code bytes in 32-lane plane order plus 1 trailing f16 scale, producing exactly 256 values.

## METAL-TQ2-CODES-001
WHEN arbitrary raw TQ2_0 codes execute on Metal, the Metal TQ2_0 kernel SHALL map codes 0, 1, 2, and 3 to minus 1, 0, plus 1, and plus 2 times the block scale.

## METAL-TQ-NUMERIC-001
WHEN valid TQ1_0 or TQ2_0 matrix multiplication executes, the Metal TQ backend SHALL match gguf.QMatMul within 1e-4 relative error, preserve floating-point class, and mutate 0 activation or compressed-weight bytes for both wire types.

## METAL-TQ-DISPATCH-001
WHEN direct, resident, or recorder TQ dispatch selects a pipeline, the Metal TQ backend SHALL select exactly 1 scalar or cooperative format-specific pipeline through 1 shared per-format predicate.

## METAL-TQ-FALLBACK-001
WHEN M exceeds the cooperative limit or 32-lane SIMD groups are unavailable, the Metal TQ selector SHALL dispatch the matching scalar TQ pipeline and issue exactly 0 cooperative TQ threadgroups.

## METAL-TQ-PERF-001
WHEN three fresh-process count-seven M2 campaigns cover every representative resident single-token cell for one TQ format, the cooperative TQ route SHALL exceed scalar control by at least 1.10 times for GPU and recorder wall time in every required cell.

## METAL-TQ-HOST-ROUTE-001
IF direct host-bound TQ1_0 or TQ2_0 loses any required M2 cell or campaign, THEN the generic Metal quant dispatcher SHALL return backend.ErrQuantUnsupported and preserve ARM64 CPU for that wire type.

## METAL-Q1-BLOCK-001
WHEN a GGUF wire type 41 Q1_0 block is decoded, the Metal Q1_0 kernel SHALL decode 18 bytes into 128 values using one f16 scale and sixteen LSB-first sign bytes where set selects positive scale and clear selects negative scale.

Rationale: Pin the Q1_0 wire layout independently of the implementation.

## METAL-MXFP4-BLOCK-001
WHEN a GGUF wire type 39 MXFP4 block is decoded, the Metal MXFP4 kernel SHALL decode each 17-byte block as one leading E8M0 exponent byte plus sixteen split-half nibble bytes, producing exactly 32 values.

## METAL-MXFP4-CODES-001
WHEN an MXFP4 nibble is decoded, the Metal MXFP4 kernel SHALL map codes 0 through 7 to 0,1,2,3,4,6,8,12 and codes 8 through 15 to 0,-1,-2,-3,-4,-6,-8,-12 respectively.

## METAL-MXFP4-SCALE-001
WHEN an MXFP4 exponent byte e is decoded, the Metal MXFP4 kernel SHALL construct scale bits as 0x00200000 shifted left by e when e is below 2, and as (e minus 1) shifted left by 23 otherwise.

## METAL-Q1-MXFP4-NUMERIC-001
WHEN valid Q1_0 or MXFP4 matrix multiplication executes, the Metal backend SHALL match gguf.QMatMul within 1e-4 relative error, preserve floating-point class, and leave every activation and compressed-weight byte unchanged.

## METAL-Q1-MXFP4-DISPATCH-001
WHEN direct, resident, or recorder Q1_0 or MXFP4 dispatch selects a pipeline, the Metal backend SHALL select exactly 1 of qmatmul_q1_0_* or qmatmul_mxfp4_* with 0 wire-type branches inside its decode loop.

## METAL-Q1-MXFP4-FALLBACK-001
WHEN M exceeds the cooperative limit or 32-lane SIMD groups are unavailable, the Metal selector SHALL dispatch the matching scalar Q1_0 or MXFP4 pipeline and issue exactly zero cooperative threadgroups for that format.

## METAL-Q1-MXFP4-PERF-001
WHEN three fresh-process count-seven M2 campaigns cover every representative resident single-token cell for one format, the cooperative Q1_0 or MXFP4 route SHALL exceed scalar control by at least 1.10 times for GPU and recorder wall time in every required cell.

## METAL-Q1-MXFP4-HOST-ROUTE-001
IF Q1_0 or MXFP4 direct host execution loses any required M2 cell or campaign, THEN the generic Metal quant dispatcher SHALL return backend.ErrQuantUnsupported for that wire type and preserve the fused ARM64 CPU route.

## METAL-ROPE-F16KV-NUMERIC-001
WHEN the fused single-token RoPE and f16 KV append executes, the Metal fusion SHALL match control Q/K float32 and cache binary16 bits, preserve nonfinite class, and mutate zero V, inverse-frequency, or unrelated cache bytes.

## METAL-ROPE-F16KV-PERF-001
WHEN three order-alternated count-seven M2 campaigns compare fused and control paths, the promotion gate SHALL retain fusion only when aggregate 21-sample median boundary speedup reaches 1.20 times and every TinyLlama decode campaign reaches 1.01 times.

## METAL-ROPE-PAIR-F16KV-NUMERIC-001
WHEN grouped-QKV RoPE and f16 KV append fusion executes, the Metal fusion SHALL match control QKV float32 and cache binary16 bits while mutating zero inverse-frequency or unrelated cache bytes.

## RECORDER-PROFILE-SNAPSHOT-PARITY-001
WHEN a completed profile is extracted, the Recorder.Profile bulk snapshot SHALL make TestRecorderProfileLabelsDurationsAndParity pass with identical RecorderProfile fields and errors.

## RECORDER-PROFILE-SNAPSHOT-OWNERSHIP-001
WHEN a completed recorder is resolved, the native profile snapshot SHALL retain exactly one immutable 96-byte-label event array until Recorder.Free.

## RECORDER-PROFILE-SNAPSHOT-WARM-PERF-001
WHEN three order-alternated count-seven campaigns compare control and bulk extraction, the warm Recorder.Profile events340 on Apple M2 SHALL run at least 1.50 times faster by median.

## RECORDER-PROFILE-SNAPSHOT-WARM-ALLOC-001
WHEN the bulk extraction candidate is measured, the warm Recorder.Profile events340 on Apple M2 SHALL allocate at least 1000 fewer objects and 40000 fewer Go bytes per operation.

## RECORDER-PROFILE-SNAPSHOT-SMALL-PERF-001
WHEN each frozen campaign compares control and bulk extraction, the warm Recorder.Profile events1 on Apple M2 SHALL run at least 1.10 times faster by median.

## RECORDER-PROFILE-SNAPSHOT-COLD-PERF-001
WHEN each frozen campaign measures events340 and events1, the first Recorder.Profile extraction on Apple M2 SHALL make events340 at least 1.10 times faster and retain 0.97 times events1 throughput.

## RECORDER-PROFILE-SNAPSHOT-ABI-001
WHEN the additive snapshot entry point is used, the Metal profile bulk snapshot ABI SHALL keep mtl_recorder_profile_summary and mtl_recorder_profile_event compatible while NewRecorder performs 0 snapshot allocations.

## RECORDER-PROFILE-VALID-INDEX-STORAGE-001
WHEN valid event indices are materialized, the native profile resolver SHALL store at most 1 index inline and all larger sets contiguously with 0 NSNumber boxes.

## RECORDER-PROFILE-LABEL-OWNERSHIP-001
WHEN a Recorder.Profile result outlives Recorder.Free and garbage collection churn, the Metal recorder profile boundary SHALL preserve 100 percent of returned event-label bytes in Go-owned memory.

## RECORDER-PROFILE-LABEL-ALLOCATION-001
WHEN one completed profile extraction contains repeated native event labels, the Recorder.Profile SHALL clone each distinct label at most once in that extraction and allocate 0 Go strings for repeated-label cache hits.

Rationale: Per-extraction deduplication avoids increasing the production Recorder footprint while preserving returned-string ownership.

## RECORDER-PROFILE-LABEL-PERF-001
WHEN three order-alternated count-seven M2 campaigns compare warm repeated-label events340 extraction, the Recorder.Profile label-cache promotion gate SHALL require at least 1.25 times median speedup, 300 fewer allocations per operation, and 4000 fewer bytes per operation.

## RECORDER-PROFILE-LABEL-NONREGRESSION-001
WHEN each frozen M2 campaign measures warm events1, first events1, and disabled recorder construction, the Recorder.Profile label-cache promotion gate SHALL retain at least 0.97 times control throughput in every cell with unchanged disabled-recorder allocations.

## RECORDER-PROFILE-LABEL-COLD-PERF-001
WHEN each frozen M2 campaign compares the first events340 profile extraction, the Recorder.Profile label-cache promotion gate SHALL require at least 1.10 times control throughput while preserving exact profile fields.

## RECORDER-PROFILE-LABEL-TOKEN-SIDECAR-001
WHEN a multi-event native profile snapshot materializes labels, the Metal recorder profile bridge SHALL return one recorder-owned uintptr_t token per event and reuse up to 16 full 96-byte labels by identity until Recorder.Free.

Rationale: Keep the token sidecar and native label-copy reuse durable without changing the one-event ABI.

## RECORDER-PROFILE-INTO-API-001
WHEN ProfileInto succeeds for a nonnil destination, the Recorder.ProfileInto SHALL set len(dst.Events) to native eventCount, reuse cap(dst.Events) when sufficient, and overwrite all 7 RecorderProfile scalar fields with Profile-equivalent values.

Rationale: Make capacity and complete overwrite behavior directly verifiable.

## RECORDER-PROFILE-INTO-ATOMIC-OWNERSHIP-001
WHEN ProfileInto receives nil, an invalid recorder, an incomplete recorder, or native extraction failure, or its successful result outlives Recorder.Free, the Metal recorder profile boundary SHALL return an explicit error without mutating a nonnil destination on failure and preserve all successful event-label bytes in Go-owned memory after Free.

## RECORDER-PROFILE-INTO-PERF-001
WHEN three independent order-alternated count-seven Apple M2 campaigns compare warmed capacity-sufficient events340 ProfileInto with Profile, the ProfileInto promotion gate SHALL require at least 1.25 times throughput in every campaign with at least 10000 fewer Go bytes and 2 fewer allocations per operation.

## RECORDER-PROFILE-INTO-LABEL-REUSE-001
WHEN capacity-sufficient ProfileInto repeatedly extracts unchanged repeated-label or mixed-label snapshots, the ProfileInto label owner SHALL reuse exact matching Go-owned destination strings and allocate zero new label strings after the first successful extraction.

## RECORDER-PROFILE-INTO-NONREGRESSION-001
WHEN each frozen Apple M2 campaign measures Profile events1 and events340 plus disabled recorder construction after adding ProfileInto, the profile extraction change SHALL retain at least 0.97 times baseline Profile throughput in every cell, preserve disabled-recorder allocations, and keep existing native profile entry points ABI-compatible.
