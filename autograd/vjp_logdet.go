package autograd

import (
	"runtime"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// logdetParallelIdx runs body(idx) for idx in [0,n) across GOMAXPROCS goroutines, STRIPED
// (worker w handles idx = w, w+nw, …) so a triangular per-index load balances. work is the
// total inner-iteration estimate; below a threshold it runs serially to avoid goroutine
// overhead. Callers must ensure each i writes DISJOINT memory (see the Ā loop).
//
//perfscan:ignore PS3048,PS3061,PS6021 init/RegisterVJP signature; one-time registration | init closure signature; one-time registration | verificati
func logdetParallelIdx(n, work int, body func(i int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > n {
		nw = n
	}
	if nw <= 1 || work < 1<<15 {
		for i := 0; i < n; i++ {
			body(i)
		}
		return
	}
	var wg sync.WaitGroup
	for w := 0; w < nw; w++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			for i := start; i < n; i += nw {
				body(i)
			}
		}(w)
	}
	wg.Wait()
}

// LogDet VJP — Jacobi's formula. For y = logdet(A) with scalar output cotangent ḡ,
// ∂y/∂A = A⁻ᵀ, which is A⁻¹ for the symmetric SPD input, so Ā = ḡ·A⁻¹. The inverse
// is formed from the same Cholesky factor L = chol(A): L⁻¹ by forward substitution,
// then A⁻¹ = L⁻ᵀ·L⁻¹ (symmetric, so no ½-symmetrization is needed — unlike the
// Cholesky VJP where S is not symmetric). All arithmetic is f64 (§V10).
func init() {
	RegisterVJP(backend.OpLogDet, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		a := in[0]
		lout, err := backend.Execute(ctx, backend.OpCholesky, []*tensor.Tensor{a}, nil)
		if err != nil {
			return nil, err
		}
		lt := lout[0]
		n := lt.Shape()[0]
		gv := g.AtF64() // scalar cotangent

		// dense f64 copy of L (lower triangle)
		l := make([][]float64, n)
		for i := range n {
			//perfscan:ignore PS2008,PS3064 resource-only alloc, no wallclock | abar-alloc indirection; subsumed by PS4006 flatten
			l[i] = make([]float64, n)
			for j := 0; j <= i; j++ {
				//perfscan:ignore PS3016 same abar loop; subsumed by PS4006 flatten
				l[i][j] = lt.AtF64(i, j)
			}
		}
		// Linv = L⁻¹ (lower-triangular) by forward substitution on L·X = I, solved DIRECTLY in
		// column-major: linvT[j][k] holds Linv[k,j]. The consumer below needs column-major anyway
		// (it contracts A⁻¹ = Linvᵀ·Linv), so building it here removes an entire O(n²) transpose
		// pass and n intermediate row allocations rather than paying for both layouts.
		//
		// It also fixes FALSE SHARING, which is the part a profile does not show. Solving in
		// row-major, worker j wrote linv[i][j] for every i — a COLUMN — so worker j+1's stores
		// landed in the adjacent eight bytes of the same rows, and every write contended for a
		// cache line with the other workers. Here worker j owns the contiguous row linvT[j]
		// exclusively.
		//
		// Each COLUMN j is an independent solve (reads only l and its own column), so the fan-out
		// is unchanged and bit-identical to serial.
		linvT := make([][]float64, n)
		for i := range n {
			//perfscan:ignore PS2008,PS3064 resource-only alloc, no wallclock | indirection; subsumed by PS4006 flatten
			linvT[i] = make([]float64, n)
		}
		logdetParallelIdx(n, n*n*n/6, func(j int) {
			col := linvT[j]
			col[j] = 1 / l[j][j]
			for i := j + 1; i < n; i++ {
				// Both operands are now contiguous slices cut to one length, so the whole inner
				// loop carries no bounds check at all. The row-major form had one surviving on
				// lv[t][j] plus a slice-header load per element, because it walked DOWN a column
				// of a [][]float64.
				//
				// Bit-identical: cv[t] is col[j+t] = Linv[j+t, j], which is exactly the lv[t][j]
				// the previous form read — same operands, same ascending order.
				lrow := l[i][j:i]
				cv := col[j:i]
				cv = cv[:len(lrow)]
				var s float64
				//perfscan:ignore PS3010 range on linv loop; subsumed by PS4006 flatten
				for t, lik := range lrow {
					s += lik * cv[t]
				}
				col[i] = -s / l[i][i]
			}
		})
		// Ā = ḡ·A⁻¹, A⁻¹ = Linvᵀ·Linv (symmetric): (A⁻¹)_ij = Σ_{k≥max(i,j)} Linv[k,i]·Linv[k,j].
		// linvT is already column-major, so each dot walks two CONTIGUOUS rows. Linv is
		// lower-triangular, so column i is nonzero only for k ≥ i, and A⁻¹'s symmetry lets the
		// O(n³) accumulation run only for i ≤ j and mirror (j,i)=(i,j). Both preserve the exact
		// per-element summation, so the result is bit-identical to the serial column-strided form.
		abar := tensor.New(a.Dtype(), a.Shape())
		// Rows are disjoint: worker for row i writes (i,·) and its mirror (·,i) only within its
		// own j ≥ i range, so no two i's touch the same (r,c). Chunk-parallel over i, STRIPED
		// (i = w, w+nw, …) to balance the triangular per-row load; bit-identical to serial.
		logdetParallelIdx(n, n*n/2, func(i int) {
			ri := linvT[i]
			for j := i; j < n; j++ {
				rj := linvT[j]
				var s float64
				//perfscan:ignore PS3010 same forward-sub loop; subsumed by PS4006 flatten
				for k := j; k < n; k++ { // k ≥ max(i,j) = j
					s += ri[k] * rj[k]
				}
				v := gv * s
				abar.SetF64(v, i, j)
				if i != j {
					abar.SetF64(v, j, i)
				}
			}
		})
		return []*tensor.Tensor{abar}, nil
	})
}
