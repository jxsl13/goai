---
schema: v1
prefix: SIZE
---

## SIZE-THE-FANOUT-TO-THE-WORK-001
WHEN a fan-out helper splits into GOMAXPROCS workers with only an on/off work gate, the that helper SHALL scale the WORKER COUNT to the work as well, one worker per fixed grain of element-work capped by GOMAXPROCS, and pick the grain against both the clock and the CPU.

Rationale: Measured on the quantized decode matmul, which is called thousands of times per generation with one activation row. A profile of a 500-token quantized Llama generation at twelve cores spent 88 percent of its samples in pthread_cond_signal and pthread_cond_wait and 1.5 percent in the kernel itself, yet the fan-out was still worth having: forcing the same call serial cost 8 percent of the clock, and leaving it at twelve workers burned 42 percent more user CPU to buy that 8 percent back. THE ON/OFF GATE CANNOT EXPRESS THE ANSWER because the answer is a worker count, not a yes or no. One worker per 1<<15 of element-work, capped by GOMAXPROCS, beat both: BenchmarkQuantLlamaGenerate500 549.7 to 527.4 ms, minus 4.1 percent, with system CPU 2.03 to 1.44 s, minus 27 percent, and the prefill cell flat as a control since it takes the m greater than 1 path. Bit-identical at any chunk count - each output row is an independent dot with its own accumulator over ascending k. THE GRAIN IS A TRADE AND THE TWO AXES DISAGREE: a coarser 1<<16 gave 536.1 ms and 1.19 s of system time, so it costs 4 percent of the clock to save another 44 percent of the system time, and which side is right depends on whether the machine serves one request or many. Encoded as perfscan PS3061, 10 candidates tree-wide - every other fan-out helper in the repo.

## HANG-A-DECODE-SCRATCH-ON-THE-RECEIVER-001
WHEN a decode primitive declares a local fixed-size byte array and hands a slice of it to a call that takes an interface, the that primitive SHALL hang the buffer on the receiver instead, because the slice escapes and the array is heap-allocated on every invocation.

Rationale: Measured on the GGUF header reader. u32 and u64 each declared var b [N]byte locally and passed b[:] to a read helper that calls io.ReadFull - an interface method - so the slice escapes and the array escapes with it: 1.34M objects of a 4.0M allocation profile, one per scalar decoded. The string reader allocated twice per value, a byte slice and the string built from it, when only the string survives. Moving the numeric scratch onto the reader and reusing one buffer for string bodies: BenchmarkReadFileSynth/header-heavy 223892 to 95804 allocations, minus 57.2 percent, and 5.45 to 4.50 ms, minus 17.3 percent; the tensor-heavy cell 4446 to 1917 allocations; the skewed cell 201 to 110. Consistent across three interleaved rounds, race detector clean, and trivially bit-identical - the same bytes are parsed into the same values. SAFE ONLY BECAUSE NOTHING KEEPS THE BUFFER: the string case works because string(b) COPIES, and the numeric buffer is consumed before the next read. Check every caller before sharing one scratch, and check the type is not used concurrently - a parser reading one file sequentially is, a shared decoder would not be. Encoded as perfscan PS3071.

## AN-ALLOCATION-CUT-NEED-NOT-MOVE-THE-CLOCK-001
WHEN reporting a change that removes per-call allocations, the report SHALL state the allocation delta and the clock delta separately, because they routinely disagree by an order of magnitude.

Rationale: Two applications of the same transform in one package, measured the same way. Hanging the READER scratch on its receiver took BenchmarkReadFileSynth/header-heavy from 223892 to 95804 allocations, minus 57.2 percent, AND from 5.45 to 4.50 ms, minus 17.3 percent. Hanging the WRITER scratch on its receiver took BenchmarkWriteQuantizedModel from 429 to 138 allocations, minus 67.8 percent, and left the clock FLAT - 3.85 to 4.00 ms on the minimum of three interleaved rounds, inside the run-to-run spread. The difference is the absolute count, not the proportion: the reader allocated a hundred thousand times per call and the writer four hundred. A proportional cut on a small count is a resource improvement and nothing more. Two further edits in the same pass - the single-byte metadata arms of both the reader and the writer - cut nothing measurable at all, because the synthetic model carries no u8 or bool metadata; they were kept because they are strictly better and they clear the check, and they are reported as flat rather than folded into the headline.

## intent
- P-01M0JBW0SVETY8QC1HZV6G561V Open quantized GGUF files through retained read-only mappings: Consumed by archived task T-01M0JBX2XYFNR. OpenRaw ships explicit retained-mapping ownership, passed all gates, delivered 8.90x raw-open and 1.57x full-consumer-copy speedups, and leads matched gguf-py by 89.17x/11.86x. Evidence and perfscan #798 are committed.
- P-01M0JE35SGFV0BNHQA5AQC2GDA Fuse ARM64 Q4_K decode unpack and dot on M2: Merged in PR #1132 at 89795e4a after all 15 CI lanes passed. The retained ARM64 Q4_K fused dot delivered 4.73x at the largest measured production QMatMul shape and 2.94x in Mamba2 while preserving all non-ARM64 and M>1 routes. Generalizable fast-path bypass risk was reported as perfscan issue #799. The Q6_K negative control was statistically flat and is intentionally left for a separate measured p [body truncated at tombstone retention cap]
- T-01M0JWPKA3E13AFRYG012RQ706 Implement and benchmark IQ4_NL QMatMul with ARM64 fused lookup dot: validated pass by codex-root-validator diff b53186d69cc4 :: Committed source implements portable IQ4_NL F32/F64 QMatMul, caller-owned decode, scoped M1 selector, and a row-level noescape ARM64 fused lookup dot. Direct gates prove exact layout, scalar bit identity, selector boundaries, input immutability, zero leaf allocations, <=1e-4 cancellation/random error (observed max 4.55e-16), and allocatio [body truncated at tombstone retention cap]
- P-01M0JWHMHZEDQVCFCT3GJVPME7 Add IQ4_NL QMatMul and fuse its ARM64 decode dot: Implemented and merged by PR #1138 after all CI checks passed; merge commit 61a34fed57c179222b895a74bdc21a4599b8d17d.
- T-01M0JZ5WKSFH0SGA8PSCGEDYFR Implement and benchmark IQ4_XS QMatMul with ARM64 fused super-block dot: validated pass by codex-root-post-validator diff 0e6626269f33 :: Committed source implements portable IQ4_XS F32/F64 QMatMul and a scoped ARM64 F32-M1 fused super-block dot. Exact decode identity, packed golden -29568, cancellation, immutability, selector scope, zero leaf allocations, race, three cross-builds, full preflight, benchmark smoke, and external perfscan all pass. Retained n=10 alternati [body truncated at tombstone retention cap]
- P-01M0JZ33ZVEMKS1K6FYAEFJFK6 Add IQ4_XS QMatMul and fuse its ARM64 super-block dot: Merged by PR #1139 after all 15 CI checks passed; evidence retained under m2-arm64-iq4xs-fused-dot-20260821.
- T-01M0K1CEY0FNEBG4A4TWD9VKMN Implement and benchmark MXFP4 QMatMul with ARM64 fused row dot: validated pass by codex-root-post-validator no attributed diff (c193f3ddce65 binds the target list, not code) :: Post-commit source review confirms portable MXFP4 QMatMul dispatch, caller-owned exact decode, and an ARM64-only row selector across every declared target. Numerical gates pass: exact 48 golden, all 256 E8M0 entries bit-exact, caller-owned decode and scalar fused dot bit-exact, maximum [body truncated at tombstone retention cap]
- P-01M0K1A86JFY3RHNM84CR28WKW Add MXFP4 QMatMul and fuse its ARM64 row dot: Merged as goai PR #1140 at d1daf49033de16b26711aca26fca849200f15345 with all 15 CI lanes green. Portable MXFP4 QMatMul and the Apple ARM64 fused row dot clear every retained 2x gate; evidence and perfscan #799 report are published. IQ3, IQ2, and IQ1 remain explicit next families.
- T-01M0K3PD38FVWSMA8JTVVD2ERE Implement and statistically gate exact IQ3_S QMatMul and M2 ARM64 fused row dot: Merged in goai PR #1141 at merge commit 1daaf7163700f288337f36d17ed0fa4a2b71d910 after all 15 CI lanes passed. Implemented exact caller-owned IQ3_S decode, portable F32/F64 QMatMul, constant per-worker scratch, and an Apple ARM64 zero-allocation fused row dot. Retained n=10 fresh-process 500 ms alternating-order results: K4096 leaf 5.254 us to 1.080 us (4.86x, p=0.000), M1/N64/K1024 84.83 us to 17 [body truncated at tombstone retention cap]
- P-01M0K3NE11FSRAY6V751A2PC1K M2-first exact IQ3_S fused row dot and portable QMatMul: Delivered and merged by goai PR #1141 at 1daaf7163700f288337f36d17ed0fa4a2b71d910 with all 15 CI lanes green. The direct-F32 IQ3_S boundary is exact and materially faster on M2: 4.86x K4096 leaf, 4.73x M1/N64/K1024, and 3.64x M1/N4096/K1024, all p=0.000, with neutral IQ4_XS and tensor-dequant controls, flat allocation/byte profiles, and 2.223032812824975e-15 maximum scalar-relative row error. Exte [body truncated at tombstone retention cap]
- T-01M0K6D56YFAYSY4QQPYVXCE4A Implement and statistically gate exact IQ3_XXS QMatMul and M2 ARM64 fused row dot: Shipped in goai PR #1142. All 15 CI lanes passed; merge commit 6667d30f0aaa7e4214aa2a68de4f1402a527e1b0 contains validated head d5026cafd73689315c9da7789fd8bde74350e4f7, and the remote feature branch was deleted after ancestry verification. Added portable IQ3_XXS F32/F64 QMatMul and scratch reuse plus an Apple ARM64 exact fused M=1 F32 row dot. M2 gains: leaf K4096 6.32x, M1/N64/K1024 6.13x, M1/N4 [body truncated at tombstone retention cap]
- P-01M0K6A4A6F0SAGEMT1937ZQQN M2-first exact IQ3_XXS fused row dot and portable QMatMul: Completed by goai PR #1142 and merge commit 6667d30f0aaa7e4214aa2a68de4f1402a527e1b0 after all 15 CI lanes passed. The tranche established portable IQ3_XXS F32/F64 QMatMul semantics and scratch reuse, then added an exact Apple ARM64 M=1 F32 fused row dot while retaining the portable oracle and fallback. Statistically significant M2 improvements were 6.32x at leaf K4096, 6.13x at M1/N64/K1024, and [body truncated at tombstone retention cap]
- T-01M0K90SRGFGRSGF2MZZ4V2R4Q Implement and statistically gate exact IQ2_XXS QMatMul and M2 ARM64 fused row dot: Shipped exact portable IQ2_XXS F32/F64 QMatMul plus the zero-allocation ARM64 fused row dot in PR #1143. Ten alternating fresh-process samples measured 6.27x at K4096, 5.98x at M1/N64/K1024, and 4.18x at M1/N4096/K1024, all p=0.000 with flat allocations; IQ3_XXS and dequant controls remained neutral. Maximum scalar-relative error was 8.731189028573922e-15. All 15 CI lanes passed. GitHub merged val [body truncated at tombstone retention cap]
- P-01M0K8Z3S1ER2BYJ5H815QYNM8 M2-first exact IQ2_XXS fused row dot and portable QMatMul: Completed through archived task T-01M0K90SRGFGR and merged PR #1143. The portable IQ2_XXS QMatMul contract and ARM64 fused leaf shipped with 4.18x to 6.27x statistically significant same-semantics speedups, neutral controls, flat allocations, complete evidence/source pins, all 15 CI lanes green, and verified merge ancestry. No llama.cpp leadership claim was made because activation semantics differ [body truncated at tombstone retention cap]
- T-01M0KBJMSPENETC1VE2QPSM91C Implement and statistically gate exact IQ2_XS QMatMul and M2 ARM64 fused row dot: Implemented exact portable F32/F64 IQ2_XS QMatMul and a zero-allocation M2 ARM64 fused M1 row dot. Ten retained alternating fresh-process samples measured 5.75x at K4096, 5.48x at M1/N64/K1024, and 3.95x at M1/N4096/K1024, all p=0.000, with flat allocation counts, neutral IQ2_XXS and decoder controls, and maximum scalar-relative error 4.1954810760924744e-15. Full package, race, Linux ARM64/AMD64 c [body truncated at tombstone retention cap]
- P-01M0KBF17FE9Z83X9MEJ23S1VM M2-first exact IQ2_XS fused row dot and portable QMatMul: Closed the IQ2_XS CPU execution gap with exact portable direct-F32/F64 semantics and statistically validated M2 ARM64 leverage. PR #1144 merged exact head d51304c4244d4a365fe5b6d6ed96e716344a3d7c as f15eeb522e4d87af8f537e8b9568e9945eacf9c6 after all 15 CI lanes succeeded; the remote feature branch was verified deleted. The retained evidence records 5.75x leaf, 5.48x N64, and 3.95x N4096 gains with [body truncated at tombstone retention cap]
- T-01M0KEGBWFFG08YYQMEMZMQSXR Implement and statistically gate exact IQ2_S QMatMul and M2 ARM64 fused row dot: Shipped by PR #1145 and merged as 46cf0883280379fa025e95b37f469870c9ca1784. Exact portable IQ2_S F32/F64 QMatMul and the M2-first ARM64 fused row dot deliver 5.44x leaf, 5.20x M1/N64, and 4.27x M1/N4096 speedups across ten retained alternating fresh-process samples; allocation counts are flat. IQ2_S tensor decode and IQ2_XS negative controls remain neutral. Final source reproduces benchmark binary [body truncated at tombstone retention cap]
- P-01M0KEEGADEWBRSZJ2KTWCT4FX M2-first exact IQ2_S fused row dot and portable QMatMul: Completed and shipped through PR #1145, merge 46cf0883280379fa025e95b37f469870c9ca1784, after all 15 CI lanes succeeded. The proposal added exact portable IQ2_S QMatMul and a scoped M2 ARM64 fused F32 M1 row dot, with 4.27x to 5.44x retained speedups, flat allocations, neutral dequant and IQ2_XS controls, and maximum scalar-relative error 1.9996704013343956e-15. Four contracts now preserve portabl [body truncated at tombstone retention cap]
- T-01M0KGWNMSF72T3J11M86A89G3 Implement and statistically gate exact IQ1_S QMatMul and M2 ARM64 fused row dot: Completed implementation and validation merged through PR #1146. Durable behavior is captured by the four IQ1_S rules, CHANGELOG entry, evidence bundle, governing ADR, and perfscan issues #808 through #811.
- T-01M0KM5H42FBTR5N0G6SMNMB6V Implement and statistically gate exact IQ1_M QMatMul and M2 ARM64 fused row dot: Archived after verified PR 1147 merge. Standing semantics and selector constraints remain in IQ1M-PORTABLE-QMATMUL-001, IQ1M-PORTABLE-SCRATCH-001, ARM64-IQ1M-FUSED-DOT-001, and ARM64-IQ1M-FUSED-DOT-SCOPE-001; reproducible evidence is committed under m2-arm64-iq1m-fused-dot-20260822.
- T-01M0M30EFGEN49ZG22VFTE4JSE Route every supported IQ and MXFP4 wire type through decodeTensor: Archived after implementation commit 36c2456a and complete local validation. The committed evidence directory contains raw benchmark streams, benchstat, manifest, correctness gates, and external perfscan null-delta results.
- P-01M0M2XZRGFSWVT5G94ZXT61S8 Complete GGUF IQ and MXFP4 wire decode dispatch: Implemented by archived task T-01M0M30EFGEN4 under ADR-01M0M2ZMEDF86. Commit 36c2456a completes eager Read dispatch for every existing IQ/MXFP4 decoder, preserves unsupported-type behavior, and commits exhaustive tests plus reproducible neutral-overhead and external perfscan evidence.
- T-01M0PCYV52EF48DMJCJ79D1WEN Bulk-unpack paired Q4_K coefficient headers: Bulk-decoded paired Q4_K headers while preserving coefficient and reduction order. Go 1.26.6 M2 Pro measurements: K=2048 paired row 571.4 ns to 536.4 ns (1.065x, 7/7 wins, U=49, p=0.0005828); TinyLlama FFN pair-apply 555.135 us to 528.617 us (1.050x, 7/7 wins); production digest exact with 3/5 wins, 1.017x median paired ratio, and neutral aggregate median. Full preflight, Go 1.27 GGUF tests, and D [body truncated at tombstone retention cap]
- T-01M0PEP367F7WBD4RTC87J8VG9 Bulk-unpack independent Q4_K coefficient headers: Bulk-decoded independent ARM64 Q4_K headers while preserving coefficient values, block-dot operands, and F64 row reduction. M2 Pro Go 1.26.6 K=2048 row improved 312.0 ns to 288.6 ns median (1.081x, 7/7 wins, U=49, p=0.0005828, zero allocations). Mixed TinyLlama QKV improved 190.126 us to 182.325 us median (1.043x, 5/7 wins; supporting only because p=0.1649 under late host contention). Clean 8-thre [body truncated at tombstone retention cap]

## ARM64-Q4K-FUSED-DOT-001
WHEN QMatMul receives contiguous F32 M1 activations with Q4_K weights, the ARM64 Q4_K selector SHALL dispatch to fused NEON unpack-affine-dot with zero leaf allocations and scalar-relative error at most 1e-4.

Rationale: M2 evidence shows 8.00x leaf, 4.73x production-shaped QMatMul, and 2.94x recurrent decode gains; materialize-then-dot regressed and allocated.

## ARM64-Q4K-FUSED-DOT-SCOPE-001
The non-ARM64 and M>1 Q4_K QMatMul paths SHALL remain on their portable or prefill implementations without dispatching the ARM64 M1 kernel.

Rationale: The measured gain and tolerance contract cover only ARM64 single-token decode.

## ARM64-Q6K-FUSED-DOT-001
WHEN QMatMul receives contiguous F32 M1 activations with Q6_K weights, the ARM64 Q6_K single-token QMatMul selector SHALL dispatch to fused NEON unpack-scale-dot with zero leaf allocations and scalar-relative error at most 1e-4.

## ARM64-Q6K-FUSED-DOT-SCOPE-001
The non-ARM64 and M greater than one Q6_K QMatMul paths SHALL remain on their portable or prefill implementations without dispatching the ARM64 M1 kernel.

## ARM64-Q8-FUSED-DOT-001
WHEN QMatMul receives contiguous F32 M1 activations with Q8_0 weights, the ARM64 Q8_0 single-token QMatMul selector SHALL dispatch to a fused NEON signed-int8 scale dot with zero leaf allocations and scalar-relative error at most 1e-4.

## ARM64-Q8-FUSED-DOT-SCOPE-001
The non-ARM64, M greater than one, and non-Q8_0 QMatMul paths SHALL remain on their portable or prefill implementations without dispatching the ARM64 Q8_0 M1 kernel.

## ARM64-Q8-FUSED-DOT-SCOPE-002 {applies: go:gguf.QMatMul,go:gguf.dotQ5_KRow,go:gguf.dotQ5_KRowASM,go:gguf.dotQ5KBlockNeon}
WHEN QMatMul receives contiguous F32 M1 activations with Q5_K weights, the ARM64 Q5_K single-token QMatMul selector SHALL dispatch to fused NEON nibble-plus-high-bit unpack-affine-dot with zero leaf allocations and scalar-relative error at most 1e-4.

Rationale: Q5_K decode currently bypasses the architecture selector used by adjacent K-quants and repeatedly decodes byte planes in scalar Go.

## ARM64-Q8-FUSED-DOT-SCOPE-003 {applies: go:gguf.QMatMul}
The non-ARM64 and M greater than one Q5_K QMatMul paths SHALL remain on their portable or prefill implementations without dispatching the ARM64 M1 kernel.

Rationale: The optimization is an ARM64 single-token kernel and must not alter portable builds or the general matrix path.

## ARM64-Q8-FUSED-DOT-SCOPE-004 {applies: go:gguf.QMatMul,go:gguf.dotQ3_KRow}
WHEN QMatMul receives contiguous F32 M1 activations with Q3_K weights, the ARM64 Q3_K single-token QMatMul selector SHALL dispatch to fused NEON two-bit-plus-inverted-mask unpack-scale-dot with zero leaf allocations and scalar-relative error at most 1e-4.

Rationale: Q3_K is the slowest current recurrent quant path and bypasses the architecture selectors used by adjacent K-quants.

## ARM64-Q8-FUSED-DOT-SCOPE-005 {applies: go:gguf.QMatMul}
The non-ARM64 and M greater than one Q3_K QMatMul paths SHALL remain on their portable or prefill implementations without dispatching the ARM64 M1 kernel.

Rationale: The optimization is an ARM64 single-token kernel and must not alter portable builds or the general matrix path.

## ARM64-Q2K-FUSED-DOT-SCOPE-001
The Q2_K QMatMul dispatcher SHALL keep non-ARM64 and M greater than one paths on their portable or prefill implementations without dispatching the ARM64 M1 kernel.

## ARM64-Q2K-FUSED-DOT-001
WHEN QMatMul receives contiguous F32 M1 activations with Q2_K weights, the ARM64 Q2_K single-token selector SHALL dispatch to fused NEON two-bit affine unpack-dot with zero leaf allocations and scalar-relative error at most 1e-4.

## IQ4NL-PORTABLE-QMATMUL-001
WHEN IQ4_NL weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve the 16-entry nonlinear codebook, low-half then high-half order, f16 block scaling, and f64 accumulation.

## IQ4NL-PORTABLE-SCRATCH-001
The portable IQ4_NL QMatMul SHALL use exactly 1 reusable scratch per worker for M greater than one and perform 0 per-output-row tensor allocations.

## ARM64-IQ4NL-FUSED-DOT-001
WHEN contiguous F32 M1 activations use IQ4_NL weights, the ARM64 IQ4_NL selector SHALL dispatch one row-level fused NEON nonlinear-lookup dot with zero leaf allocations and scalar-relative error at most 1e-4.

## ARM64-IQ4NL-FUSED-DOT-SCOPE-001
The IQ4_NL QMatMul dispatcher SHALL keep non-ARM64 and M greater than one paths portable and dispatch 0 ARM64 M1 kernel calls.

## IQ4XS-PORTABLE-QMATMUL-001
WHEN IQ4_XS weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve f16 super-scaling, 8 signed six-bit sub-scales per 256 weights, low-half then high-half nonlinear lookup order, and f64 accumulation.

## ARM64-IQ4XS-FUSED-DOT-001
WHEN contiguous F32 M1 activations use IQ4_XS weights, the ARM64 IQ4_XS selector SHALL dispatch a zero-allocation fused 256-weight NEON nonlinear-lookup dot with scalar-relative error at most 1e-4.

## ARM64-IQ4XS-FUSED-DOT-SCOPE-001
The IQ4_XS QMatMul dispatcher SHALL keep non-ARM64 and M greater than one paths portable and dispatch 0 ARM64 M1 kernel calls.

## MXFP4-PORTABLE-QMATMUL-001
WHEN MXFP4 weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve E8M0-half scale conversion, low-half then high-half signed E2M1 codebook order, float32 scale multiplication, and float64 accumulation.

## ARM64-MXFP4-FUSED-DOT-001
WHEN contiguous F32 M1 activations use MXFP4 weights, the ARM64 MXFP4 selector SHALL dispatch one row-level fused NEON E8M0-scale and signed-codebook dot with 0 leaf allocations and scalar-relative error at most 1e-4.

## ARM64-MXFP4-FUSED-DOT-SCOPE-001
The MXFP4 QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 ARM64 M1 kernel calls.

## IQ3S-PORTABLE-QMATMUL-001
WHEN IQ3_S weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve the 512-entry grid, 9-bit indices, direct signs, eight four-bit scales expanded as 1+2*s, float32 block scaling, and float64 accumulation with exactly 1 scratch-set allocation per worker.

## ARM64-IQ3S-FUSED-DOT-001
WHEN contiguous F32 M1 activations use IQ3_S weights, the Apple ARM64 IQ3_S selector SHALL dispatch 1 row-level fused NEON 9-bit-grid and direct-sign dot with 0 leaf allocations and scalar-relative error at most 1e-4.

## ARM64-IQ3S-FUSED-DOT-SCOPE-001
The IQ3_S QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

## IQ3XXS-PORTABLE-QMATMUL-001 {applies: go:gguf.dequantIQ3_XXSInto,go:gguf.dotIQ3XXSRow,go:gguf.QMatMul,go:gguf.TestDotIQ3XXSRowMatchesMaterializedReferenceExactly,go:gguf.TestQMatMulIQ3XXSMatchesDequantizedReference}
WHEN IQ3_XXS weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve the 256-entry grid, four 7-bit ksigns indices per 32 weights, float32 d*(0.5+s)*0.5 scaling, ascending mapping, and float64 accumulation.

## IQ3XXS-PORTABLE-SCRATCH-001 {applies: go:gguf.QMatMul,go:gguf.TestQMatMulIQ3XXSScratchAllocationsDoNotScaleWithOutputRows}
The portable IQ3_XXS QMatMul SHALL use exactly 1 scratch-set allocation per worker and perform 0 per-output-row tensor allocations.

## ARM64-IQ3XXS-FUSED-DOT-001 {applies: go:gguf.dotIQ3XXSRowASM,go:gguf.TestDotIQ3XXSBlockNeonKnownValue,go:gguf.TestDotIQ3XXSAsmRandomRaw,go:gguf.TestDotIQ3XXSAsmCancellationHeavy,go:gguf.TestDotIQ3XXSAsmAllocs}
WHEN contiguous F32 M1 activations use IQ3_XXS weights, the Apple ARM64 IQ3_XXS selector SHALL dispatch 1 row-level fused NEON grid, ksigns, and scale dot with 0 leaf allocations and scalar-relative error at most 1e-4.

## ARM64-IQ3XXS-FUSED-DOT-SCOPE-001 {applies: go:gguf.dotIQ3XXSRowFn,go:gguf.QMatMul,go:gguf.TestQMatMulIQ3XXSSelectorScope}
The IQ3_XXS QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

## IQ2XXS-PORTABLE-QMATMUL-001 {applies: go:gguf.dequantIQ2_XXSInto,go:gguf.dotIQ2XXSRow,go:gguf.QMatMul,go:gguf.TestDotIQ2XXSRowMatchesMaterializedReferenceExactly,go:gguf.TestQMatMulIQ2XXSMatchesDequantizedReference}
WHEN IQ2_XXS weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve the 256-entry eight-wide grid, four 7-bit ksigns indices per 32 weights, float32 d*(0.5+s)*0.25 scaling, ascending element mapping, and float64 accumulation.

Rationale: The portable path is the semantic oracle and direct-F32/F64 boundary for every architecture-specific optimization.

## IQ2XXS-PORTABLE-SCRATCH-001 {applies: go:gguf.QMatMul,go:gguf.TestQMatMulIQ2XXSScratchAllocationsDoNotScaleWithOutputRows}
The portable IQ2_XXS QMatMul SHALL use exactly 1 scratch-set allocation per worker and perform 0 per-output-row tensor allocations.

Rationale: Reusable worker-local scratch removes allocation growth with N while preserving the all-M portable path.

## ARM64-IQ2XXS-FUSED-DOT-001 {applies: go:gguf.dotIQ2XXSRowASM,asm:gguf.dotIQ2XXSBlockNeon,go:gguf.TestDotIQ2XXSBlockNeonKnownValue,go:gguf.TestDotIQ2XXSAsmRandomRaw,go:gguf.TestDotIQ2XXSAsmAllocs}
WHEN contiguous F32 M1 activations use IQ2_XXS weights, the Apple ARM64 IQ2_XXS selector SHALL dispatch 1 row-level fused NEON eight-wide grid, ksigns, and scale dot with 0 leaf allocations and scalar-relative error at most 1e-4.

Rationale: The M2 decode hot path must eliminate materialized weights while retaining a measurable numerical gate.

## ARM64-IQ2XXS-FUSED-DOT-SCOPE-001 {applies: go:gguf.QMatMul,go:gguf.TestQMatMulIQ2XXSSelectorScope}
The IQ2_XXS QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

Rationale: Architecture and dtype specialization must not leak into portable or prefill semantics.

## IQ2XS-PORTABLE-QMATMUL-001 {applies: go:gguf.QMatMul,go:gguf.dequantIQ2_XSInto,go:gguf.dotIQ2XSRow,go:gguf.TestDotIQ2XSRowMatchesMaterializedReferenceExactly,go:gguf.TestQMatMulIQ2XSMatchesDequantizedReference,go:gguf.TestIQ2XSUnalignedCodePlaneMatchesAlignedExactly}
WHEN IQ2_XS weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve the 512-entry eight-wide grid, 9-bit grid and 7-bit ksign indices, one 4-bit scale per 16 weights, float32 d*(0.5+s)*0.25 scaling, ascending mapping, and float64 accumulation.

Rationale: The portable path is the semantic oracle and direct-F32/F64 boundary for every architecture-specific IQ2_XS optimization.

## IQ2XS-PORTABLE-SCRATCH-001 {applies: go:gguf.QMatMul,go:gguf.dequantIQ2_XSInto,go:gguf.TestQMatMulIQ2XSScratchAllocationsDoNotScaleWithOutputRows}
The portable IQ2_XS QMatMul SHALL use exactly 1 scratch-set allocation per worker and perform 0 per-output-row tensor allocations.

Rationale: The allocation gate proves output-row growth reuses worker scratch instead of recreating materialized tensors.

## ARM64-IQ2XS-FUSED-DOT-001 {applies: go:gguf.QMatMul,go:gguf.dotIQ2XSRowASM,go:gguf.dotIQ2XSBlockNeon,go:gguf.TestDotIQ2XSBlockNeonKnownValue,go:gguf.TestDotIQ2XSAsmRandomRaw,go:gguf.TestDotIQ2XSAsmCancellationHeavy,go:gguf.TestDotIQ2XSAsmAllocs}
WHEN contiguous F32 M1 activations use IQ2_XS weights, the Apple ARM64 IQ2_XS selector SHALL dispatch 1 row-level fused NEON grid, ksign, explicit-scale dot with 0 leaf allocations and scalar-relative error at most 1e-4.

Rationale: The assembly leaf, numerical gates, cancellation case, known block, and allocation gate jointly prove the ARM64 contract.

## ARM64-IQ2XS-FUSED-DOT-SCOPE-001 {applies: go:gguf.QMatMul,go:gguf.dotIQ2XSRowASM,go:gguf.TestQMatMulIQ2XSSelectorScope}
The IQ2_XS QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

Rationale: The selector test injects a counting oracle and proves only contiguous F32 M1 reaches the row leaf.

## IQ2S-PORTABLE-QMATMUL-001 {applies: go:gguf.QMatMul,go:gguf.dequantIQ2_S,go:gguf.dequantIQ2_SInto,go:gguf.dotIQ2SRow,go:gguf.TestDequantIQ2SIntoMatchesTensorDecoderExactly,go:gguf.TestDotIQ2SRowMatchesMaterializedReferenceExactly,go:gguf.TestQMatMulIQ2SMatchesDequantizedReference,go:gguf.TestQMatMulIQ2SRejectsInvalidInputs}
WHEN IQ2_S weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve 1024 eight-wide grid rows, 10-bit byte-plus-qh indices, direct 8-weight sign bytes, 16-weight four-bit scales, float32 d*(0.5+s)*0.25, ascending mapping, and float64 accumulation.

Rationale: The portable path is the semantic oracle and direct-F32/F64 boundary for every architecture-specific IQ2_S optimization.

## IQ2S-PORTABLE-SCRATCH-001 {applies: go:gguf.QMatMul,go:gguf.TestQMatMulIQ2SScratchAllocationsDoNotScaleWithOutputRows}
The portable IQ2_S QMatMul SHALL use exactly 1 scratch-set allocation per worker and perform 0 per-output-row tensor allocations.

Rationale: Worker-owned scratch prevents output-row fanout from turning decoding into allocation traffic.

## ARM64-IQ2S-FUSED-DOT-001 {applies: go:gguf.QMatMul,go:gguf.dotIQ2SRowASM,go:gguf.dotIQ2SBlockNeon,go:gguf.TestDotIQ2SBlockNeonKnownValue,go:gguf.TestDotIQ2SAsmRandomRaw,go:gguf.TestDotIQ2SAsmCancellationHeavy,go:gguf.TestDotIQ2SAsmAllocs}
WHEN contiguous F32 M1 activations use IQ2_S weights, the Apple ARM64 IQ2_S selector SHALL dispatch 1 row-level fused NEON 10-bit-grid, direct-sign, explicit-scale dot with 0 leaf allocations and scalar-relative error at most 1e-4.

Rationale: Single-token decode is the M2 CPU latency cell where fused unpack and dot has measured leverage across the aggressive-quant family.

## ARM64-IQ2S-FUSED-DOT-SCOPE-001 {applies: go:gguf.QMatMul,go:gguf.TestQMatMulIQ2SSelectorScope}
The IQ2_S QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

Rationale: The ARM64 leaf must not narrow portable dtype, shape, or architecture semantics.

## IQ1S-PORTABLE-QMATMUL-001 {applies: go:gguf.QMatMul,go:gguf.dequantIQ1_S,go:gguf.dequantIQ1_SInto,go:gguf.dotIQ1SRow,go:gguf.TestDequantIQ1SIntoMatchesTensorDecoderExactly,go:gguf.TestDotIQ1SRowMatchesMaterializedReferenceExactly,go:gguf.TestQMatMulIQ1SMatchesDequantizedReference,go:gguf.TestQMatMulIQ1SRejectsInvalidInputs}
WHEN IQ1_S weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve 2048 eight-wide ternary grid rows, packed 11-bit indices, odd qh multipliers, signed 0.125 deltas, float32 scaling, ascending mapping, and float64 accumulation.

Rationale: The portable path is the semantic oracle and direct-F32/F64 boundary for every architecture-specific IQ1_S optimization.

## IQ1S-PORTABLE-SCRATCH-001 {applies: go:gguf.QMatMul,go:gguf.TestQMatMulIQ1SScratchAllocationsDoNotScaleWithOutputRows}
The portable IQ1_S QMatMul SHALL use exactly 1 scratch-set allocation per worker and perform 0 per-output-row tensor allocations.

Rationale: M greater than one must reuse decoded-row storage instead of rebuilding a full tensor or allocating per output row.

## ARM64-IQ1S-FUSED-DOT-001 {applies: go:gguf.QMatMul,go:gguf.dotIQ1SRowASM,go:gguf.dotIQ1SRowNeon,go:gguf.TestDotIQ1SBlockNeonKnownValue,go:gguf.TestDotIQ1SAsmRandomRaw,go:gguf.TestDotIQ1SAsmCancellationHeavy,go:gguf.TestDotIQ1SAsmAllocs}
WHEN contiguous F32 M1 activations use IQ1_S weights, the Apple ARM64 IQ1_S selector SHALL dispatch 1 row-level fused NEON 11-bit-grid, odd-scale, signed-delta dot with 0 leaf allocations and scalar-relative error at most 1e-4.

Rationale: Single-token direct-F32 decode is the M2 CPU hot path; the portable scalar row dot remains its semantic oracle.

## ARM64-IQ1S-FUSED-DOT-SCOPE-001 {applies: go:gguf.QMatMul,go:gguf.TestQMatMulIQ1SSelectorScope}
The IQ1_S QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

Rationale: The specialized leaf assumes contiguous F32 activations and must not alter portable or prefill semantics.

## IQ1M-PORTABLE-QMATMUL-001 {applies: go:gguf.QMatMul,go:gguf.dequantIQ1_M,go:gguf.dequantIQ1_MInto,go:gguf.dotIQ1MRow,go:gguf.TestDequantIQ1MIntoMatchesTensorDecoderExactly,go:gguf.TestDotIQ1MRowMatchesMaterializedReferenceExactly,go:gguf.TestQMatMulIQ1MMatchesDequantizedReference,go:gguf.TestQMatMulIQ1MRejectsInvalidInputs}
WHEN IQ1_M weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve split-f16 super-scale reconstruction, 2048 ternary grid rows, packed 11-bit indices, paired odd multipliers, signed 0.125 deltas, float32 scaling, ascending mapping, and float64 accumulation.

Rationale: The portable path is the semantic oracle and direct-F32/F64 boundary for every architecture-specific IQ1_M optimization.

## IQ1M-PORTABLE-SCRATCH-001 {applies: go:gguf.QMatMul,go:gguf.TestQMatMulIQ1MScratchAllocationsDoNotScaleWithOutputRows}
The portable IQ1_M QMatMul SHALL use exactly 1 scratch-set allocation per worker and perform 0 per-output-row tensor allocations.

Rationale: Caller-owned decode scratch prevents output-row count from multiplying temporary allocations.

## ARM64-IQ1M-FUSED-DOT-001 {applies: go:gguf.QMatMul,go:gguf.dotIQ1MRowASM,go:gguf.dotIQ1MRowNeon,asm:gguf.dotIQ1MRowNeon,go:gguf.TestDotIQ1MRowNeonKnownValue,go:gguf.TestDotIQ1MAsmRandomRaw,go:gguf.TestDotIQ1MAsmCancellationHeavy,go:gguf.TestDotIQ1MAsmAllocs}
WHEN contiguous F32 M1 activations use IQ1_M weights, the Apple ARM64 IQ1_M selector SHALL dispatch 1 whole-row fused NEON split-scale, 11-bit-grid, paired-odd-scale, signed-delta dot with 0 leaf allocations and scalar-relative error at most 1e-4.

Rationale: Single-token direct-F32 decode is the M2 CPU hot path; the portable scalar row dot remains its semantic oracle.

## ARM64-IQ1M-FUSED-DOT-SCOPE-001 {applies: go:gguf.QMatMul,go:gguf.TestQMatMulIQ1MSelectorScope}
The IQ1_M QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

Rationale: The native leaf assumes Apple ARM64 F32 row layout and must not leak into portable or prefill semantics.

## Q1-FORMAT-001 {applies: go:gguf.byteSize,go:gguf.decodeTensor,go:gguf.Quantize,go:gguf.Dequantize,go:gguf.dequantQ1_0Into,go:gguf.dequantQ1_0,go:gguf.quantizeQ1_0,go:gguf.TestQ1FormatAPIsMatchPinnedLayout,go:gguf.TestQuantizeQ1MatchesPinnedReferenceLayout,go:gguf.TestDequantQ1IntoMatchesTensorDecoderExactly,go:gguf.TestQ1RejectsInvalidInputs}
WHEN ggml type 41 Q1_0 data is encoded or decoded, the format APIs SHALL preserve 18-byte blocks of 128 weights, f16 scale, LSB-first sign bits, and bit-one positive semantics.

Rationale: End-to-end support prevents a fast QMatMul path from accepting bytes that Read, QuantTensor.Dequantize, Quantize, or Dequantize cannot reproduce.

## Q1-PORTABLE-QMATMUL-001 {applies: go:gguf.QMatMul,go:gguf.Dequantize,go:gguf.dotQ1Row,go:gguf.dequantQ1_0Into,go:gguf.TestDotQ1RowMatchesMaterializedReferenceExactly,go:gguf.TestQMatMulQ1MatchesDequantizedReference,go:gguf.TestQMatMulQ1RejectsInvalidInputs}
WHEN Q1_0 weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve f16 scale, LSB-first signs, bit-one positive weights, ascending element mapping, and float64 accumulation.

Rationale: The portable row dot is the semantic oracle for every architecture-specific Q1_0 path.

## Q1-PORTABLE-SCRATCH-001 {applies: go:gguf.QMatMul,go:gguf.dequantQ1_0Into,go:gguf.TestQMatMulQ1ScratchAllocationsDoNotScaleWithOutputRows}
The portable Q1_0 QMatMul SHALL use exactly 1 scratch-set allocation per worker and perform 0 per-output-row tensor allocations.

Rationale: Caller-owned decode scratch prevents output-row count from multiplying temporary allocations.

## ARM64-Q1-FUSED-DOT-001 {applies: go:gguf.QMatMul,go:gguf.dotQ1RowASM,go:gguf.dotQ1RowNeon,asm:gguf.dotQ1RowNeon,go:gguf.TestDotQ1RowNeonKnownSigns,go:gguf.TestDotQ1AsmRandomRaw,go:gguf.TestDotQ1AsmCancellationHeavy,go:gguf.TestDotQ1AsmAllocs}
WHEN contiguous F32 M1 activations use Q1_0 weights, the Apple ARM64 Q1_0 selector SHALL dispatch 1 whole-row NEON sign-XOR and f16-scale dot with 0 leaf allocations and scalar-relative error at most 1e-4.

Rationale: Single-token direct-F32 decode is the M2 CPU hot path; sign-bit expansion can avoid materialized weights.

## ARM64-Q1-FUSED-DOT-SCOPE-001 {applies: go:gguf.QMatMul,go:gguf.dotQ1RowFn,go:gguf.dotQ1RowASM,go:gguf.TestQMatMulQ1SelectorScope}
The Q1_0 QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

Rationale: The native leaf assumes ARM64 F32 row layout and must not leak into portable, F64, or prefill semantics.

## TQ1-FORMAT-001
WHEN ggml type 34 TQ1_0 data is encoded or decoded, the format APIs SHALL preserve 54-byte blocks of 256 weights, 48 five-trit base-243 bytes, 4 four-trit base-243 tail bytes, 1 trailing f16 scale, and the pinned ggml element order.

Rationale: Exact GGUF interoperability requires both the mixed-radix block layout and its non-linear element permutation.

## TQ1-PORTABLE-QMATMUL-001
WHEN TQ1_0 weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve the trailing f16 scale, base-243 ternary decode, pinned 256-element mapping, ascending activation mapping, and float64 accumulation.

Rationale: The portable path defines public semantics and is the oracle for architecture-specific kernels.

## TQ1-PORTABLE-SCRATCH-001
The portable TQ1_0 QMatMul SHALL use exactly 1 reusable scratch set per worker for M greater than 1 and perform 0 per-output-row tensor allocations.

Rationale: TQ1_0 prefill must not reintroduce decode allocation growth.

## ARM64-TQ1-FUSED-DOT-001
WHEN contiguous F32 M1 activations use TQ1_0 weights, the Apple ARM64 TQ1_0 selector SHALL dispatch 1 whole-row fused NEON base-243 ternary and f16-scale dot with 0 leaf allocations and scalar-relative error at most 1e-4.

Rationale: Direct-F32 decode is the primary M2 single-token performance path.

## ARM64-TQ1-FUSED-DOT-SCOPE-001
The TQ1_0 QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

Rationale: The architecture-specific leaf is valid only for the benchmarked contiguous F32 M1 boundary.

## TQ2-FORMAT-001
WHEN ggml type 35 TQ2_0 data is encoded or decoded, the format APIs SHALL preserve 66-byte blocks of 256 weights, 64 bytes packing four two-bit codes in 32-lane groups, one trailing f16 scale, and the pinned ggml element order.

## TQ2-CODES-001
WHEN arbitrary raw TQ2_0 codes are decoded, the format APIs SHALL map codes 0, 1, 2, and 3 to minus 1, 0, plus 1, and plus 2 times the block scale, while reference encoding emits codes 0 through 2.

## TQ2-PORTABLE-QMATMUL-001
WHEN TQ2_0 weights are multiplied by F32 or F64 activations, the QMatMul SHALL preserve the trailing f16 scale, code-minus-one mapping, pinned 32-lane group order, ascending activation mapping, and float64 accumulation.

## TQ2-PORTABLE-SCRATCH-001
The portable TQ2_0 QMatMul SHALL use exactly 1 reusable scratch set per worker for M greater than 1 and perform 0 per-output-row tensor allocations.

## ARM64-TQ2-FUSED-DOT-001
WHEN contiguous F32 M1 activations use TQ2_0 weights, the Apple ARM64 TQ2_0 selector SHALL dispatch 1 whole-row fused NEON two-bit unpack and f16-scale dot with 0 leaf allocations and scalar-relative error at most 1e-4.

## ARM64-TQ2-FUSED-DOT-SCOPE-001
The TQ2_0 QMatMul dispatcher SHALL keep non-ARM64, M greater than 1, and non-F32 paths portable and dispatch 0 Apple ARM64 M1 kernel calls.

## GGUF-IQ-WIRE-ID-001
The GGUF quant type registry SHALL assign IQ1_S to wire type 19 and IQ2_S to wire type 22 exactly as pinned ggml does.

## GGUF-I8-UNSUPPORTED-001
WHEN a caller supplies GGUF wire type 24 before I8 support exists, the format APIs SHALL return an unsupported-type error without dispatching an IQ decoder.

## GGUF-IQ-ID-SEMANTICS-001
The IQ1_S and IQ2_S identifier correction SHALL preserve each existing block size, decoder math, QMatMul selector scope, and numerical tolerance.

## GGUF-IQ-WIRE-DISPATCH-001
WHEN Read or QuantTensor.Dequantize receives wire type 19 or 22, the GGUF decoder SHALL dispatch IQ1_S for type 19 and IQ2_S for type 22 with the same result as public Dequantize.

## GGUF-IQ-MXFP4-WIRE-DISPATCH-001
WHEN Read or QuantTensor.Dequantize receives wire type 16, 17, 18, 20, 21, 23, 29, or 39, the GGUF decoder SHALL dispatch 1 matching existing decoder and return F32 values exactly equal to public Dequantize.

## GGUF-IQ-MXFP4-WIRE-SCOPE-001
The GGUF wire dispatch extension SHALL preserve 8 wire identifiers, block sizes, decoder arithmetic, QMatMul selectors, platform routes, and unsupported-type errors.

## GGUF-Q4K-PAIRED-M1-001
WHEN 2 equal-shape Q4_K matrices receive 1 contiguous F32 M1 activation, the QMatMulPair SHALL allocate 2 F32 outputs, invoke qmatmulParallelChunks exactly once, and match 2 independent QMatMul outputs bit-for-bit.

## GGUF-MIXED-QKV-M1-001
WHEN 3 unequal-row Q4_K or Q6_K matrices receive 1 contiguous F32 M1 activation, the QMatMulTriple SHALL invoke qmatmulParallelChunks exactly 1 time and match 3 independent QMatMul outputs bit-for-bit.

Rationale: One work-sized fan-out removes 2 scheduler barriers without changing row arithmetic.

## GGUF-MIXED-QKV-OUTPUT-001
The QMatMulTriple SHALL return 3 F32 tensors with shapes [1,n0], [1,n1], and [1,n2].

Rationale: Unequal grouped projections retain every matrix output shape.

## GGUF-MIXED-QKV-BALANCE-001
WHEN 1 grouped fan-out combines Q4_K and Q6_K matrices with unequal row counts, the QMatMulTriple SHALL partition every matrix proportionally across every scheduler chunk, creating 0 quant-type-only tail chunks.

Rationale: Contiguous concatenation reduced allocations but lost 6 of 8 initial production pairs; proportional distribution produced the retained gain.

## GGUF-Q4K-PAIRED-APPLY-001
WHEN 2 Q4_K matrices receive 1 contiguous F32 M1 activation and an 8-lane consumer, the QMatMulPairApply SHALL invoke qmatmulParallelChunks once, return 1 F32 output, call the consumer per aligned chunk, and bit-match QMatMulPair plus that consumer.

## GGUF-Q4K-PAIRED-SCRATCH-001
The QMatMulPairApply SHALL borrow exactly 1 raw up scratch, return it after the final chunk, retain capacities no larger than 65,536 F32 values, and expose 0 scratch aliases to callers.

## GGUF-Q4K-PAIR-DUAL-DOT-001
WHEN QMatMulPairApply computes paired Q4_K rows, the ARM64 paired Q4_K row dot SHALL load every activation vector exactly once for 2 weight rows through dotQ4KPairBlockNeon while preserving the independent accumulation and reduction orders bit-for-bit.

## Q4K-PAIR-BULK-HEADER-EXACT-001
The paired Q4_K coefficient builder SHALL decode all 16 six-bit scale/min values per row exactly as 8 getScaleMinK4 calls and preserve pair-row output bits.

## Q4K-PAIR-BULK-HEADER-PERF-001
WHEN the K=2048 paired-row benchmark runs on Apple ARM64, the bulk Q4_K header path SHALL reach at least 1.03x median speedup across 7 interleaved campaigns with 0 allocation increase and no production-shape regression.

## Q4K-SINGLE-BULK-HEADER-EXACT-001
The independent Q4_K coefficient builder SHALL decode all 16 six-bit scale/min values exactly as 8 getScaleMinK4 calls and preserve row output bits.

## Q4K-SINGLE-BULK-HEADER-PERF-001
WHEN the K=2048 independent-row benchmark runs on Apple ARM64, the bulk independent Q4_K header path SHALL reach at least 1.03x median speedup across 7 interleaved campaigns with 0 allocation increase and no production-shape regression.

## Q4K-POSTINDEX-EXACT-001 {applies: go:gguf.dotQ4KBlockNeon,go:gguf.dotQ4KPairBlockNeon}
WHEN single or paired 256-weight leaf pointer arithmetic is folded into load addressing, the Apple ARM64 Q4_K block kernels SHALL return bits identical to the d43cdb4b leaf for every input and perform 0 leaf allocations.

Rationale: Post-index addressing must remain a pure integer-address-generation optimization around the established numerical kernel.

## Q4K-POSTINDEX-PERF-001 {applies: go:gguf.dotQ4KPairBlockNeon,go:gguf.dotQ4KPairRowASM}
WHEN the K=2048 paired-row benchmark runs on Apple ARM64, the post-index Q4_K paired leaf SHALL reach at least 1.02x median speedup across 7 interleaved campaigns, retain 0 allocations, and show no production-shape regression.

Rationale: The paired leaf is the dominant remaining Q4_K compute hotspot in the exact TinyLlama CPU profile.
