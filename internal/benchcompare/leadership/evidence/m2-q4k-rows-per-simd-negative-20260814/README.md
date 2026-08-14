# M2 Q4_K rows-per-SIMD negative result

This bundle records the T998 experiment against the production cooperative
Q4_K Metal kernel. The control retained two output rows per SIMD group and two
SIMD groups per threadgroup. Same-binary compile-time candidates used one,
four, or eight rows per SIMD group; the threadgroup width stayed at 64 threads.
All variants were bit-identical to the control at odd `N=15`.

The 200-iteration stop-gate screen covered every incident Q4_K leaf shape.
There was no universal winner. Two isolated observations crossed 1.10x:
rows=1 at `K2048,N2560` (1.128x) and rows=8 at `K2048,N2048` (1.143x).
Both disappeared under ten alternating 500-iteration samples:

- `K2048,N2560`: 161.5 us control versus 159.9 us rows=1,
  `p=0.971`, approximately 1.010x.
- `K2048,N2048`: 159.6 us control versus 160.7 us rows=8,
  `p=0.971`, approximately 0.993x.
- Geomean changed by only -0.13%; both paths remained 8 B/op and 1 alloc/op.

The leaf gate failed, so no shape selector or full-model run was justified.
The experimental pipelines, selector, public measurement hook, and benchmark
modes were removed. The production two-row kernel remains unchanged.

General lesson: a single short GPU screen can make compile-time work geometry
look shape-selective when warm-clock variance is larger than the apparent win.
A performance analyzer should not recommend rows-per-SIMD or rows-per-warp
changes from launch geometry alone; it needs repeated device-specific evidence,
register/occupancy data, and an end-to-end gate.

## Commands

```text
go test ./backend/metal -run 'TestMetalQ4KRowsPerSIMDMatchesControl|TestMetalQMatMulQ4KCooperativeMatchesScalar' -count=1
go test ./backend/metal -run '^$' -bench '^BenchmarkMetalQ4KDecodeLeaf/' -benchtime=200x -count=1 -benchmem
go test -c -o /tmp/goai-metal-q4rows.test ./backend/metal
/tmp/goai-metal-q4rows.test -test.run='^$' -test.bench='^BenchmarkMetalQ4KDecodeLeaf/K2048N2560/(cooperative|rows1)$' -test.benchtime=500x -test.count=1 -test.benchmem
/tmp/goai-metal-q4rows.test -test.run='^$' -test.bench='^BenchmarkMetalQ4KDecodeLeaf/K2048N2048/(cooperative|rows8)$' -test.benchtime=500x -test.count=1 -test.benchmem
benchstat baseline.txt candidate.txt
```
