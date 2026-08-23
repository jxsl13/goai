# M2 CPU Q4_K pair-to-SwiGLU fusion

## Scope

This tranche removes the materialized `up` tensor between paired Q4_K gate/up
projection and eager CPU SwiGLU. `QMatMulPairApply` retains one gate tensor,
borrows one bounded raw scratch slice for `up`, and invokes the existing CPU
SiLU-times-up kernel inside aligned producer chunks. On ARM64, a dual-output
row kernel loads each activation vector once for both weight rows while
preserving the independent kernels' accumulation and reduction orders.

The route remains narrow: contiguous offset-zero F32 M1 input, matching Q4_K
Gate/Up geometry, eager CPU execution, and a backend exposing the chunk-fuser
capability. Recording, accelerators, batches, unsupported quant types, and
mismatched geometry retain the established paired or independent fallback.

## M2 Pro result

Five fresh-process pairs alternate candidate/control order. Each process loads
the same TinyLlama-1.1B Q4_K_M file, excludes one model/cache warm-up, then
retains three 64-step decode samples. Four of five pairs favor the candidate;
the median across process medians clears the predeclared 1.03x gate.

| 64-step CPU decode | `origin/main` | pair-to-SwiGLU | delta |
|---|---:|---:|---:|
| time | 1.808 s | 1.707 s | **-5.59% / 1.059x** |
| allocated bytes | 207,358,992 | 172,408,640 | **-16.85%** |
| allocations | 204,986 | 200,762 | **-2.06%** |

Every retained run produced final-logit digest `ea3df5516f17df83`.

Eight fresh-process fixed-iteration leaf pairs isolate the production
N=5,632/K=2,048 gate/up boundary:

| paired projection plus SwiGLU | composed | producer-apply | delta |
|---|---:|---:|---:|
| median time | 589.7 us | 513.9 us | **-12.85% / 1.148x** |
| median bytes/op | 51,168.5 | 26,482.5 | **-48.24%** |

All eight leaf pairs favor producer-apply. The scratch pool itself measures
zero steady-state allocations after warm-up, retains at most 65,536 F32
values, and exposes no scratch alias. ARM64 tests prove the dual row result is
bit-identical to two independent assembly row calls at K=256/512/2048/4096.

## Validation and claim boundary

The pinned Go 1.26.6 preflight passed formatting, build, vet, module drift,
API examples, perfscan fixtures, and the full short suite. Race-enabled CPU,
GGUF, and `nn` suites passed. The non-short diffusion grammar test fails
identically on candidate and clean `origin/main`, including the same generated
string, so it is recorded as pre-existing rather than attributed to this work.

This is an internal current-main improvement, not a cross-library leadership
claim. A matched llama.cpp comparison still requires equal KV dtype, token
stream, context, and measurement boundaries.

The general producer-consumer and shared-activation pattern is reported in
[perfscan issue #833](https://github.com/jxsl13/perfscan/issues/833).
