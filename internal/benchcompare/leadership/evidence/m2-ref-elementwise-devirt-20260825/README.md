# M2 reference elementwise bounds and dispatch evidence

## Scope

Spectackle task `T-01KYJREH8QF2H` separates two costs in the F32/F64
reference elementwise kernels:

1. loop bounds derived from `Tensor.Numel` that the compiler could not relate
   to the lengths of raw storage slices; and
2. a function-valued scalar operation called indirectly for every element.

The baseline is commit
`b5df153b6f3cbddca17c2e9d4b6c48ecbb92736d`. The reslice-only stage is
`c545d029d4bd1418c2374dfd308fdb4739ced901`. The final functor-dispatch stage
is `10f9993aef78d692ef458ebadc169ee684eb8bd9`.

## Bounds-check stage

The F32/F64 storage views are sliced once to `n`, and each loop ranges over
the output slice. The shared direct-loop cores prove peer slice lengths once
with their final required index.

Go 1.27 SSA bounds-check diagnostics changed the hot unary and binary loops
from per-index `IsInBounds` records to entry `IsSliceInBounds` or explicit
last-index checks. The final disassembly contains only entry panic branches;
the direct arithmetic loop backedges contain no bounds checks.

Seven alternating fresh-process pairs at 500 ms per benchmark isolate this
stage:

| benchmark | baseline median | resliced median | result |
|---|---:|---:|---:|
| Add F64 4K | 14,670 ns | 11,324 ns | **1.2955x** |
| Add F32 4K | 11,564 ns | 6,996 ns | **1.6529x** |
| ReLU F64 64K | 555,100 ns | 392,234 ns | **1.4152x** |
| Tanh F64 64K | 883,774 ns | 774,917 ns | **1.1405x** |

Bytes and allocations are unchanged. Raw pairs for both implementation stages
are in `staged-pairs.tsv`.

## Dispatch stage

The first generic design called an interface-constrained `op.apply` method
inside `elemUnary` and `elemBinary`. Despite zero-size concrete functors,
Go 1.27 shared one `go.shape.struct {}` instantiation and used per-operation
dictionaries. M2 disassembly still showed `blr x1` on every iteration.

The shipped design performs one functor-type dispatch before direct
operation-specific loops. The final F64 Add core is loads, `fadd`, store, and
loop control. The corresponding Sub, Mul, and Div arms use `fsub`, `fmul`,
and `fdiv`; no indirect branch and no `FMADD` occur in the elementwise
cores. Direct libm calls remain for operations such as Tanh, where the
transcendental dominates.

The separately measured reslice-to-functor medians are:

| benchmark | resliced median | direct-loop median | result |
|---|---:|---:|---:|
| Add F64 4K | 27,709 ns | 15,606 ns | **1.7755x** |
| Add F32 4K | 18,494 ns | 13,216 ns | **1.3994x** |
| ReLU F64 64K | 1,128,761 ns | 953,432 ns | **1.1839x** |
| Tanh F64 64K | 2,033,630 ns | 1,845,297 ns | **1.1021x** |

Absolute times moved with unrelated shared-host load, so the release claim
uses a longer final campaign rather than multiplying stage medians.

## Final Apple M2 Pro result

Nine process-order-alternated pairs run 1,000 complete operations per
benchmark in each fresh `GOMAXPROCS=1` process:

| benchmark | baseline median | candidate median | result | paired wins |
|---|---:|---:|---:|---:|
| Add F64 4K | 20,192 ns | 7,762 ns | **2.6014x** | 9/9 |
| Add F32 4K | 14,038 ns | 6,604 ns | **2.1257x** | 8/9 |
| ReLU F64 64K | 1,050,402 ns | 731,328 ns | **1.4363x** | 7/9 |
| Tanh F64 64K | 1,596,969 ns | 1,628,712 ns | neutral, 0.9805x | 4/9 |

All four benchmarks retain their original bytes and four allocations per
operation. Tanh is deliberately reported as neutral: removing dispatch cannot
materially improve a loop dominated by the direct libm call.

The raw final campaign is in `frozen-prepost.tsv`. Executable SHA-256 values:

- baseline:
  `996fc8986b777377caeb328ea45378af03d0fa830e8d65442cdf952c55261cdc`;
- reslice-only:
  `1916dbed60b6239d39a62d2e8e11b7a5d7e60556724ee52bbde5e54d2a11d3b3`;
- final:
  `3e2cc381c8c32de8d03988e998122962dd2029a51dc212da8f405b48b1ada612`.

The host used Go 1.27.0, macOS 26.5.1, and Apple M2 Pro ARM64.

## Numerical and structural gates

`TestElementwiseUnaryBitExact` and `TestElementwiseBinaryBitExact` freeze
the old scalar loops for every registered unary and binary operation. They
compare output bits for F64 and for the exact F32 widen/compute/narrow
sequence, including NaNs, infinities, and signed zeros.

A temporary Add-to-Sub production mutation makes both Add exactness cases fail,
which proves the oracle is live. The restored candidate passes the exact
suite. The reference contract and documented tolerances were not changed.

The generalized findings are tracked in
[perfscan issue #904](https://github.com/jxsl13/perfscan/issues/904) and
[perfscan issue #905](https://github.com/jxsl13/perfscan/issues/905).

## Reproduction

```sh
GOCACHE=/private/tmp/goai-elementwise-gocache GOPROXY=direct \
  go test -c -o /private/tmp/goai-elementwise.test ./backend/ref
GOMAXPROCS=1 /private/tmp/goai-elementwise.test \
  -test.run '^$' \
  -test.bench '^Benchmark(AddF64_4K|AddF32_4K|ReLUF64_64K|TanhF64_64K)$' \
  -test.benchmem -test.benchtime=1000x -test.count=1
```
