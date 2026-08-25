# M2 CPU Conv1D four-channel register tile

Task `T-01M0VJKX5ZEYN` replaces the channel-at-a-time depthwise causal
Conv1D traversal with four independent channel accumulators. The reduction of
each output still visits taps in ascending order, while each tap now reads four
adjacent activation values and exposes four independent multiply/add chains.

## Scope and method

- baseline: clean `bfe6abe773a195d27b12ade5c0c714107fdd082c`
- host: Apple M2 Pro, Darwin/ARM64
- toolchain: Go 1.27.0
- production shape: `L=2048,D=1024,K=4`, bias enabled
- nine alternating baseline/candidate process pairs, 800 ms benchmark time
- exact old and candidate test binaries were frozen before comparison
- allocations and bytes are part of the gate

The host was under variable concurrent load. Paired directions and win counts
therefore bound the claim; absolute times are not a cross-library leadership
result.

| dtype | scalar median | tile median | median ratio | paired median | wins |
|---|---:|---:|---:|---:|---:|
| F32 | 2.517768 ms | 1.460454 ms | **1.724x** | **1.630x** | 9/9 |
| F64 | 1.534858 ms | 1.066286 ms | **1.439x** | **1.517x** | 9/9 |

Both variants allocate six times per operation. Bytes remain dominated by the
fresh output tensor: about 8 MiB for F32 and 16 MiB for F64.

Single-P same-binary controls isolate the traversal from worker scheduling:

| F32 shape | control median | tile median | paired median | wins |
|---|---:|---:|---:|---:|
| `L1,D1024,K4,bias` | 2.990 us | 2.103 us | **1.258x** | 9/9 |
| `L64,D1023,K4,no-bias` | 427.768 us | 219.570 us | **1.933x** | 9/9 |
| `L512,D256,K8,bias` | 1.280878 ms | 0.915066 ms | **1.488x** | 7/9 |

The Float64-bit oracle covers F32 and F64, bias and no-bias, `K=1`, causal
prefixes, channel widths below four, and every non-zero four-channel tail used
by the test shapes.

## Rejected variants

- A full tap-outer row sweep won on wall time, but exact F32 accumulation needed
  one F64 scratch row per worker. GC repeatedly dropped those pooled rows and
  increased the measured allocation footprint, so the variant was rejected.
- An eight-channel tile passed the exact oracle but split only 5/9 against the
  four-channel tile for each dtype under host variance. It added code and
  register pressure without establishing more leverage.

The generalized detector and remediation result is recorded in
[perfscan issue #915](https://github.com/jxsl13/perfscan/issues/915).

## Reproduction

```sh
go test -c -o /tmp/goai-conv1d.test ./backend/cpu
/tmp/goai-conv1d.test -test.run='^$' \
  -test.bench='^BenchmarkConv1D_(cpu|control)_(f32|f64)$' \
  -test.benchmem -test.benchtime=800ms -test.count=9
GOMAXPROCS=1 /tmp/goai-conv1d.test -test.run='^$' \
  -test.bench='^BenchmarkConv1D_(cpu|control)_f32_L' \
  -test.benchmem -test.benchtime=500ms -test.count=9
```
