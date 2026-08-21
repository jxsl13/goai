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
