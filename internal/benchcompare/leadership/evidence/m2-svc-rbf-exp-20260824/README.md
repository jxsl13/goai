# M2 SVC RBF batched SIMD exponentials

## Scope

The ARM64 SIMD SVC route keeps the established scalar squared-distance
accumulation order, stores completed distances in the final kernel-column
buffer, and evaluates one existing `ExpScaledF64` batch per parallel band.
The default build, non-ARM64 builds, and non-RBF kernels retain the established
scalar path. The implementation does not add a public API or grow `SVC`.

Spectackle proposal `P-01M0TFVWHDFC7`, task `T-01M0TFXRWJEZE`, and rules
`SVC-RBF-SIMD-EXP-ROUTE-001` / `SVC-RBF-SIMD-EXP-GATE-001` bind the route and
acceptance boundary.

## Numerical and convergence gate

The scalar control and SIMD candidate both converge in exactly 79 SMO steps
and retain 42 support vectors on the fixed 4,000x20 workload. Their decision
signs match, the maximum absolute decision-value delta is
`3.3306690738754696e-15`, and two SIMD fits are bit-identical. This gate is
mandatory because earlier value-changing exponential approximations showed
that even one-ULP kernel noise can change the SMO trajectory non-monotonically.

## Apple M2 Pro result

The primary campaign uses binaries frozen from merged baseline `512fec03` and
implementation commit `6e88d7aa`. Each process runs the 4,000x20 RBF fit 100
times. Seven pairs alternate process order.

| frozen pre/post | baseline | SIMD candidate | result |
|---|---:|---:|---:|
| median time | 5.048190 ms | 4.518638 ms | **1.1536x faster** |
| paired wins | 1/7 | 6/7 | candidate |
| median bytes/op | 1,620,155 | 1,620,116 | neutral |
| median allocations/op | 1,040 | 1,040 | unchanged |

One pair favored baseline by 7.66%. The same campaign ranged from 4.89 to 8.57
ms on baseline while unrelated perfscan jobs were continuously active, so this
evidence treats paired medians as an internal improvement signal and makes no
absolute latency or universal leadership claim.

An earlier frozen campaign with the same executable semantics but before a
comment-only source correction won all seven pairs: 10.726122 to 8.948403 ms
at the medians, or 1.2218x. The exact committed-artifact campaign above is the
weaker and therefore primary release number.

Same-binary scalar controls isolate the route from source/build differences:

| control | pair wins | median candidate/base | speedup | allocations |
|---|---:|---:|---:|---:|
| GOMAXPROCS=12 | 6/7 | 0.764634 | 1.3078x | 1,041 -> 1,041 |
| GOMAXPROCS=1 | 7/7 | 0.725429 | 1.3785x | 182 -> 182 |

The retained streams are in `frozen-prepost.tsv`,
`frozen-prepost-corroborating.tsv`, and `same-binary-control.tsv`.

## Incumbent boundary

On identical generated rows, five fresh candidate processes produced a best
GoAI fit of 4.71 ms. scikit-learn 1.9.0 with NumPy 2.5.1 and Python 3.14.7
produced a best-of-five 3.621500 ms fit. The optimized Go route therefore
remains about 1.3006x behind the pinned libsvm incumbent on this cell. Batched
or decomposition SMO remains the likely next frontier; this change establishes
an internal M2 gain, not cross-library leadership.

## Generalized finding

A scalar transcendental inside an otherwise scalar reduction can be batched
without an extra allocation when completed inputs are staged directly in the
operation's final output buffer. The reusable detector opportunity is tracked
as [perfscan issue #899](https://github.com/jxsl13/perfscan/issues/899).

