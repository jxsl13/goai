---
schema: v1
---

## T-01KYJREGCMF3Q8H8D9SNSNT9ZE Fix reduceKernel — closure call, odometer and F32 pre-widening on every element
kind: task
state: draft
created: 2026-07-27

READ THIS FIRST, it reframes the whole file: backend/cpu registers 40 of 158 ops and backend/metal 39. NEITHER registers OpSum/OpMean/OpMax/OpMin/OpProd. So backend/ref is the PRODUCTION kernel for every reduction on this host, reached through the Execute fallback chain — not merely the numeric truth path.

MEASURED on this host (M2 Pro, darwin/arm64, go1.26.5, -count=3): BenchmarkSumF64_64K 226,251-228,360 ns = 3.46 ns/element. Comparators in the SAME package on the SAME host: BenchmarkReshapeF64_64K moves the identical 512 KB at 0.657 ns/element, and BenchmarkCumsumF64_256x256 does the same traffic PLUS an add per element, with no closure and no odometer, at 1.36 ns/element. The reduce loop is 5x slower than a memcpy while doing strictly less memory traffic.

SITE: backend/ref/reduce.go:65 reduceKernel, hot loop :109-119, registrations :269-273.

DEFECT, four compounding costs in one loop:
(a) combine is a CLOSURE VARIABLE — -gcflags=-S emits CALL (R1) at reduce.go:110: an indirect call per element to compute one a+x.
(b) The odometer at :111-118 runs even when provably a no-op: for a reduce-all, eff is all-zero and of is always 0.
(c) FOUR un-eliminated bounds checks per element — -d=ssa/check_bce/debug=1 reports IsInBounds at 110:26, 112:9, 113:15, 114:23, 124:7, and -S shows runtime.panicBounds calls.
(d) F32 inputs go through f64Data (backend/ref/devirt.go:24-40), which widens the ENTIRE input to a fresh []float64 before an O(n) reduction: BenchmarkSumAxisF32_256x256 shows 529,697 B/op — a 512 KB garbage buffer per call — and costs +43 us (+19%) over the F64 path.

FIX, four independently A/B-able steps, land in this order:
(a) reduce-all fast path when outNumel == 1: a local register accumulator, for _, v := range xs { a = combine(a, v) } — no odometer, no acc[of] load/store, no bounds checks.
(b) general case: compute the trailing run length L over which of is constant (product of trailing sizes whose eff is 0) and the run over which of advances by 1; drive an inner flat loop of length L and tick the odometer ONCE PER RUN.
(c) devirtualize combine/finalize with a zero-size functor generic (type sumOp struct{}; func (sumOp) do(a, x float64) float64 { return a + x }, core generic over O interface{ do(...) }) — Go monomorphizes non-interface type params, so -gcflags=-m should show the body inlined and the CALL (R1) gone.
(d) add an F32 branch that reads []float32 and widens per element instead of pre-widening the whole tensor, dropping the 512 KB allocation.

VALIDATION GATE (benchmark only): all existing — BenchmarkSumF64_64K, BenchmarkSumAxisF64_256x256, BenchmarkSumAxisF32_256x256 (backend/ref/perf_regress_test.go:41,46). Add BenchmarkMaxAxisF64_256x256 to cover the math.Max combine, whose inlining behavior differs from a literal closure.

EXPECTED: 3.46 -> 0.7-1.0 ns/element, i.e. 3.5-5x on SumF64_64K and SumAxisF64; F32 additionally -512 KB/op and -19%. High confidence for (a)(b)(d) — mechanism confirmed by disassembly, BCE dump and the in-package Cumsum/Reshape comparators. Medium for (c): generic devirtualization is reliable but must be confirmed with -gcflags=-m.

BIT-IDENTITY BAR, HIGHEST TIER — this IS the reference, so a drift here moves the goalposts for cpu, metal, cuda and vulkan simultaneously. The invariant is preservable EXACTLY: each accumulator must see the same combine sequence in the same ascending row-major pos order. (a) and (b) change only WHEN THE ODOMETER TICKS, not the visit order, so the per-accumulator add chain is unchanged and the result is bit-identical. (d) is exact — float32 to float64 widening is lossless and f64Data already performs precisely that conversion. (c) IS THE ONE TO WATCH: inlining a+x into a loop can create a new FMA-contraction opportunity on arm64 if a multiply ends up adjacent. For Sum/Mean/Max/Min there is no multiply so it cannot fire, but OpProd's a*x is the case to check. GATE THE CHANGE on -gcflags=-S showing no FMADDD in the reduce core, and re-run the cross-backend parity suite. THE PER-OP TOLERANCE MUST NOT BE WIDENED TO MAKE THIS PASS.

PERFSCAN RULES REQUIRED, one of which is a THIRD DETECTOR BUG: (i) PS1002 (per-element visitor closure) DID NOT FIRE HERE and should have. Widen it to match a loop whose body contains a CallExpr whose Fun is an *ast.Ident resolving to a PARAMETER OR FREE VARIABLE of function type (not a package-level FuncDecl), where the loop bound derives from len() or Numel(). Instances it currently misses: ref/reduce.go:110, ref/elementwise.go:37,44,81,89, and ref/reduce.go:271-272 where math.Max/math.Min are passed as values. (ii) NEW RULE, per-element odometer: a ForStmt whose body contains a nested ForStmt that is not data-dependent on the outer index, only mutates an []int counter and an integer offset, and reads a loop-invariant stride slice — shape for d := N-1; d >= 0; d-- containing idx[d]++, off += eff[d], if idx[d] < shape[d] { break }. Instances: ref/reduce.go:111, ref/broadcast.go:50,66, and the autograd odometers already tasked separately.
