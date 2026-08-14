# Rejected M2 Q4_K gate/up fusion seam — 2026-08-14

This bundle records Spectackle task `T-01M0111CEPFDX`. TinyLlama executes two
Q4_K K2048,N5632 SwiGLU projections from the same normalized input in each of
22 layers. The operation-only candidate concatenated the packed gate and up
weight rows into one valid Q4_K K2048,N11264 resident weight, issued one
cooperative matvec, and extracted its two output bands with existing device
blits. The control issued the two current resident matvecs. Both modes used one
command buffer and retained resident weights, input, and output buffers.

The candidate is bit-identical to the separate outputs. Ten paired alternating
500-iteration samples measured 247.0 us median for the separate control and
251.1 us for fused-plus-extraction. The difference is not significant
(`p=0.218`, `n=10`), misses the 1.10x leaf gate, and trends 1.7% slower. Both
paths remain 8 B/op and 1 alloc/op.

This corrects a misleading estimate obtained by adding two standalone leaf
times: standalone timings each pay their own command-buffer completion boundary,
whereas the real Decoder already records both projections into one command
buffer. Once measured at the correct operation boundary, concatenation adds no
projection win and the two extraction blits slightly outweigh any dispatch
amortization.

## Decision

No Decoder weight duplication, combined scratch workspace, prefill branch, or
production selector is added. The benchmark and its bit-identity test are kept
as a cheap reproducible guard; they only compose existing primitives and do not
alter production dispatch. A future gate/up optimization must reuse the input
within a genuinely fused quant kernel and emit SwiGLU directly; merely
concatenating rows and copying bands is not a lever.

This is also a reusable measurement rule: do not estimate graph-fusion leverage
by summing isolated synchronous leaf benchmarks when production already batches
the operations into one command buffer. Measure the full fused seam against the
batched control.

## Reproduce measured binary

```sh
go test -c -o /tmp/goai-metal-q4k-gateup.test ./backend/metal

# Ten pairs, reversing mode order on every second pair.
/tmp/goai-metal-q4k-gateup.test -test.run='^$' \
  -test.bench='^BenchmarkMetalQ4KGateUpFusion/separate$' \
  -test.benchtime=500x -test.count=1 -test.benchmem
/tmp/goai-metal-q4k-gateup.test -test.run='^$' \
  -test.bench='^BenchmarkMetalQ4KGateUpFusion/fused$' \
  -test.benchtime=500x -test.count=1 -test.benchmem
```

Normalize only the final mode component before running `benchstat`. Raw samples
are retained in `separate.txt` and `fused.txt`.
