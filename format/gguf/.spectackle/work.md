---
schema: v1
---

## P-01M0JWHMHZEDQVCFCT3GJVPME7 Add IQ4_NL QMatMul and fuse its ARM64 decode dot
kind: proposal
state: active
created: 2026-08-21
refs: ADR-01M0JWG7KKFVE9V1KTEEQ556WF
grilled: 2026-08-21 open=0
targets: go:gguf.QMatMul, format/gguf/iq4.go, format/gguf/quant_matmul.go, format/gguf/iq4_test.go, format/gguf/bench_test.go, format/gguf/dot_iq4nl_asm_arm64.go, format/gguf/dot_iq4nl_asm_arm64.s

Extend QMatMul into the importance-quant family bottom-up by adding portable IQ4_NL semantics and a separately dispatched Apple ARM64 M1 kernel. The portable path SHALL preserve the existing nonlinear 16-entry codebook, low-half then high-half element order, f16 block scaling, f64 accumulation, and reusable scratch behavior for M greater than one. The ARM64 row-level leaf SHALL process all 32-element blocks in one noescape call, use vector table lookup for codebook values, avoid materialized weights, retain zero leaf allocations, and stay within 1e-4 scalar-relative error under arbitrary raw and cancellation-heavy inputs. Because merged main rejects IQ4_NL QMatMul, measure portable scalar versus NEON in one candidate binary at K4096 leaf, M1/N64/K1024, and M1/N4096/K1024, plus an untouched supported quant negative control; retain only material gains with n=10 alternating order. Preserve every existing quant path and make no llama.cpp leadership claim without a matched harness. Run full GGUF, race, portable cross-build, preflight, external perfscan with GOPROXY=direct, and post-implementation Spectackle validation; report generalized findings on perfscan.

## T-01M0JWPKA3E13AFRYG012RQ706 Implement and benchmark IQ4_NL QMatMul with ARM64 fused lookup dot
kind: task
state: active
created: 2026-08-21
parent: P-01M0JWHMHZEDQVCFCT3GJVPME7
refs: P-01M0JWHMHZEDQVCFCT3GJVPME7, ADR-01M0JWG7KKFVE9V1KTEEQ556WF
grilled: 2026-08-21 open=1
targets: format/gguf/iq4.go, format/gguf/quant_matmul.go, format/gguf/iq4_test.go, format/gguf/dot_iq4nl_asm_arm64.go, format/gguf/dot_iq4nl_asm_arm64.s, format/gguf/dot_iq4nl_scalar.go, format/gguf/dot_iq4nl_asm_arm64_test.go, format/gguf/bench_test.go

Add portable IQ4_NL QMatMul semantics for F32 and F64 inputs, preserving nonlinear lookup order and f16 block scaling with f64 accumulation. Reuse one row scratch per worker for M greater than one and retain zero per-output-row tensor allocations. Add an Apple ARM64 M1 contiguous-F32 selector and one row-level noescape fused leaf using vector table lookup without materializing decoded weights. Prove exact layout behavior, input immutability, zero leaf allocations, arbitrary and cancellation-heavy relative error at or below 1e-4, and selector boundaries. Benchmark candidate scalar versus NEON at K4096 leaf, M1/N64/K1024, and M1/N4096/K1024, with an untouched supported quant negative control using n=10 alternating order and excluded warmups. Commit reproducible evidence, run full GGUF and race suites, Linux amd64 and arm64 cross-builds, make preflight, external perfscan v1.71.0 with GOPROXY=direct, and post-implementation Spectackle validation. Report any generalizable performance findings on jxsl13/perfscan; open a PR and merge only after all CI checks pass.
