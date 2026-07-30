package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized GEMM (§T12). C[M,N] = A[M,K]·B[K,N] on raw contiguous slices using
// the ikj loop order: for fixed (i,j) the k updates run 0..K-1 in the same order
// as the reference, so the result is bit-identical to backend/ref (§V3, §V11
// tol 0) — no summation reorder. ikj also gives row-wise access to B and C (the
// inner j-loop is a vectorizable SAXPY) and parallelizes cleanly over disjoint C
// row-bands. f32 accumulates in an f64 scratch matrix, narrowing once (§V10).
//
// §T74/§B28: BLIS-style kc×nc cache blocking with a packed B-panel was built and
// measured against this kernel on the arm64 (Apple M-series) host and DISCARDED —
// it tied on f64-512, regressed f32-1024 (~-9%), and its one marginal f64-1024
// median (~12%) sat inside the run-to-run noise while costing ~40% more memory.
// The M-series memory subsystem already feeds this streaming kernel near-optimally,
// so panel packing is net overhead here. The real remaining headroom is wider-FMA
// SIMD, which is host-blocked on arm64 (cf §T11b/§B13). Resume blocking only on a
// host whose cache hierarchy actually bottlenecks the unblocked stream (large-cache
// x86 server), re-measuring before merge.
//
// Update (amd64/Zen 3, this branch): that re-measurement happened. On a Ryzen
// 5700G the unblocked stream DOES bottleneck once the k×n B panel outgrows L2
// (512×2048×2048 f32 ran at ~half the 1024³ rate), and packing-free COLUMN
// blocking (j-splits sized to ~256 KiB of B per block, no packed buffer, no
// extra memory) recovered it: f32 +88%, f64 +60% on that shape, squares
// neutral, numerics untouched. Lives in gemm_simd.go (amd64+simd only).
//
// Update 2 (arm64, this branch): THE arm64 DISCARD ABOVE NO LONGER HOLDS, and the reason is
// that it was measured against a different kernel. At the time, the portable band ran p-outer
// with j inner and STREAMED its accumulator — four f64 loads and four stores per j to issue
// four FMAs — so it was load/store bound on C and B locality could not be the limit. Packing B
// against that kernel was indeed net overhead, exactly as recorded.
//
// Giving the band a 4x4 register tile removed the C traffic, and B locality then became the
// binding constraint. Packing measured -11.1% (f32) and -28.1% (f64) at n=1024 on this same
// M-series host, and the portable bands now pack behind measured per-dtype gates. Cumulatively
// the tile, the pack, the operand-widening hoist and the packed remainder took the vision
// benchmark suite -17.76% geomean end to end.
//
// The general lesson, since it cost a real win: a "measured and discarded" verdict is scoped to
// the code it was measured against. Re-open it when that code changes shape
// (PERF-THRESHOLD-IS-STALE-WHEN-ITS-ARM-CHANGES-001).

// matContiguousF32 returns a dense row-major copy of a rank-2 F32 tensor,
// parallelizing the gather over output ROWS. matmulKernel's backward calls
// materialize a TRANSPOSED operand (Wᵀ/Aᵀ) whose Contiguous() runs a single
// serial per-element gather (~2 GB/s, a textbook transpose cache-thrash) before
// the parallel GEMM — an Amdahl serialization the profile put at ~5-7% of a GPT
// training step. Each output row is independent, so a row-band split removes it.
// The gather order (out[r*C+c] = src[off + r·s0 + c·s1], ascending r then c) is
// identical to tensor.Contiguous, so the result is bit-for-bit the same. The
// parThreshold gate inside parallelWork keeps small operands serial (no fork
// cost), and nested calls degrade to inline via the pool, so this is safe even
// if a caller is already inside a parallel region. Non-rank-2 / already-dense
// tensors fall back to the shared Contiguous fast paths.
func matContiguousF32(t *tensor.Tensor) *tensor.Tensor {
	if t.IsContiguous() && t.Offset() == 0 {
		return t
	}
	sh := t.Shape()
	if len(sh) != 2 {
		return t.Contiguous()
	}
	rows, cols := sh[0], sh[1]
	st := t.Strides()
	s0, s1, off := st[0], st[1], t.Offset()
	src := t.Storage().F32()
	out := tensor.NewOn(t.Device(), tensor.F32, tensor.Shape{rows, cols})
	dst := out.Storage().F32()
	parallelWork(rows, cols, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			drow := dst[r*cols : r*cols+cols]
			so := off + r*s0
			if s1 == 1 {
				copy(drow, src[so:so+cols])
			} else {
				for c := 0; c < cols; c++ {
					drow[c] = src[so+c*s1]
				}
			}
		}
	})
	return out
}

// matmulInlineWork gates the F64 matmul's serial fast path: below this m·k·n it runs on
// the caller (see matmulKernel), above it the row bands fan out.
//
// RE-SWEPT after the band kernels were register-tiled. The old value, 1<<18, was calibrated
// against a band roughly twice as slow; fork/join cost did not change while the work being
// weighed against it shrank, so the crossover moved UP and the old gate was fanning out shapes
// that had stopped paying for it. Forcing each arm (BenchmarkMatmulInlineGate), square k=n:
//
//	m*k*n  262144 (n=64)  parallel +37.26%    592704 (n=84)  parallel  +9.41%
//	       512000 (n=80)  parallel +22.85%    681472 (n=88)  parallel -37.66%
//	                                          884736 (n=96)  parallel -45.07%
//
// So the crossover now sits between 592704 and 681472, not at 262144: everything in that band
// was paying 9-37% for a fan-out that no longer earned it. 640000 sits between the last loss and
// the first win. Not a power of two, deliberately — the measured boundary is not one, and
// rounding to 1<<19 would re-admit the +9.41% case while 1<<20 would forgo the -37.66% one.
//
// A var, not a const, so the benchmark can force each arm and this can be re-swept whenever the
// band kernels change again (PERF-THRESHOLD-IS-STALE-WHEN-ITS-ARM-CHANGES-001).
var matmulInlineWork = 640000

func matmulKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: matmul wants 2 inputs, got %d", len(in))
	}
	a, b := in[0], in[1]
	if a.Ndim() != 2 || b.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: matmul needs rank-2, got %dD and %dD", a.Ndim(), b.Ndim())
	}
	if a.Dtype() != b.Dtype() {
		return nil, fmt.Errorf("cpu: matmul dtype mismatch %v vs %v", a.Dtype(), b.Dtype())
	}
	m, k := a.Shape()[0], a.Shape()[1]
	k2, n := b.Shape()[0], b.Shape()[1]
	if k != k2 {
		return nil, fmt.Errorf("cpu: matmul inner dim mismatch %v · %v", a.Shape(), b.Shape())
	}
	ac, bc := a, b
	if a.Dtype() == tensor.F32 {
		ac, bc = matContiguousF32(a), matContiguousF32(b)
	} else {
		ac, bc = a.Contiguous(), b.Contiguous()
	}
	out := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{m, n})

	switch a.Dtype() {
	case tensor.F64:
		A, B, C := ac.Storage().F64(), bc.Storage().F64(), out.Storage().F64()
		if m == 1 {
			// Single-row (GEMV, the decode shape): the row-band split above
			// degenerates to ONE chunk — the whole product ran serial on the
			// caller while the §V22 decode profile showed it as the dominant
			// cost. Split over COLUMN ranges of C instead: each worker owns a
			// disjoint j-range, and every C[j] still accumulates its k products
			// in one ascending-p pass, so the result stays bit-identical to the
			// band kernel and ref (§V3, §V11 tol 0).
			parallelWork(n, k, func(loCol, hiCol int) {
				gemvF64Cols(A, B, C, k, n, loCol, hiCol)
			})
			return []*tensor.Tensor{out}, nil
		}
		if m*k*n < matmulInlineWork {
			// Small matmul (e.g. a training step's tiny layers): compute-cheap but it
			// would pay the pool's fork/join + worker-spin machinery per op. Run it
			// serial on the caller — measurably faster in the many-small-ops regime,
			// while larger matmuls still fan out. Bit-identical: one unsplit band is the
			// same ascending accumulation as the row-band chunks (§V3, §V11 tol 0). Only
			// matmul uses this higher gate — bandwidth-bound elementwise and expensive-
			// per-element transcendentals keep parThreshold so they still parallelize.
			gemmF64Band(A, B, C, 0, m, k, n)
		} else {
			gemmF64Rows(A, B, C, m, k, n)
		}
	case tensor.F32:
		A, B := ac.Storage().F32(), bc.Storage().F32()
		C := out.Storage().F32()
		gemmF32(A, B, C, m, k, n)
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", a.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// gemvF64Cols computes C[0, lo:hi] of the single-row product C[1,n] =
// A[1,k]·B[k,n]: for each owned column j the k contributions accumulate in
// ascending-p order — exactly the order gemmF64Band's remainder-row loop uses —
// so a column-split GEMV is bit-identical to the serial band (§V3). The p-outer
// / j-inner loop keeps B access contiguous (an axpy over the owned segment of
// each B row); C starts zeroed by the tensor allocation, matching the band
// kernel's += contract.
func gemvF64Cols(A, B, C []float64, k, n, lo, hi int) {
	c := C[lo:hi:hi]
	for p := range k {
		ap := A[p]
		bp := B[p*n+lo : p*n+hi : p*n+hi]
		for j, bv := range bp {
			c[j] += ap * bv
		}
	}
}

func init() {
	std.add(backend.OpMatMul, tensor.F32, matmulKernel)
	std.add(backend.OpMatMul, tensor.F64, matmulKernel)
}
