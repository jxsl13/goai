# Rejected M2 Q4_K/Q6_K aligned-unroll candidate — 2026-08-14

This bundle preserves the negative result from Spectackle task
`T-01M00ZBMRBFQN`. The candidate duplicated the current cooperative Q4_K and
Q6_K M=1 kernels, removed the inner odd-row checks for `N % 4 == 0`, and added
`#pragma clang loop unroll(full)` to fixed-trip load, decode, row, and reduction
loops. The existing cooperative kernels remained selectable in the same binary.

The first 100-iteration screen showed apparent 1.03–1.36x wins on several
shapes. Ten paired, alternating 500-iteration samples did not reproduce them.
No measured shape was significant, every median missed the 1.10x promotion
gate, and candidate geomean time was 1.44% worse. Both paths remained 8 B/op and
1 alloc/op.

| M=1 shape | cooperative median | candidate median | result |
|---|---:|---:|---:|
| Q4_K K2048,N2560 | 158.5 us | 160.4 us | no difference, `p=0.912` |
| Q4_K K2048,N5632 | 168.3 us | 168.9 us | no difference, `p=0.529` |
| Q4_K K5632,N2048 | 172.0 us | 173.0 us | no difference, `p=0.684` |
| Q6_K K2048,N256 | 136.0 us | 139.4 us | no difference, `p=0.853` |
| Q6_K K5632,N2048 | 180.8 us | 185.5 us | no difference, `p=0.529` |

The baseline samples include early high-variance observations despite 20 warmup
executions per sub-benchmark. Paired mode order was reversed on every second
pair, so that drift is represented rather than discarded. `benchstat.txt`
contains the normalized-name comparison over all ten samples.

## Decision and reusable boundary

The candidate is rejected and removed. On the tested Apple M2 Pro toolchain,
forcing full unroll of these small constant loops and deleting a predictable
two-row tail predicate does not provide a reproducible speedup over the current
Metal compiler output. A static recommendation to add explicit unroll pragmas
or clone a tail-free kernel would therefore be a false positive without
hardware-and-shape evidence.

The next kernel experiment should change execution geometry rather than compiler
hints: autotune rows per SIMD group and SIMD groups per threadgroup, retain the
current cooperative implementation as the control, and cache the winner by
device, quant type, K, and N. Full-model work remains gated until a leaf variant
clears the standing threshold.

No full-model or cold-start run was performed because the leaf gate failed.
This prevents measurement spend and production complexity for a non-lever.

## Reproduce

```sh
go test -c -o /tmp/goai-metal-kquant-successor.test ./backend/metal

# Ten pairs. Reverse mode order on every second pair.
/tmp/goai-metal-kquant-successor.test -test.run='^$' \
  -test.bench='BenchmarkMetalQ(4|6)KDecodeLeaf/(K2048N256|K2048N5632|K5632N2048)/cooperative$' \
  -test.benchtime=500x -test.count=1 -test.benchmem
/tmp/goai-metal-kquant-successor.test -test.run='^$' \
  -test.bench='BenchmarkMetalQ(4|6)KDecodeLeaf/(K2048N256|K2048N5632|K5632N2048)/successor$' \
  -test.benchtime=500x -test.count=1 -test.benchmem

# Normalize only the final mode component to the same benchmark name, then:
benchstat baseline-normalized.txt candidate-normalized.txt
```

The measured experimental binary SHA-256 and source hashes are in
`metadata.json`; the candidate code itself is deliberately not retained.
