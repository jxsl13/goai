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
			l[i] = make([]float64, n)
			for j := 0; j <= i; j++ {
				l[i][j] = lt.AtF64(i, j)
			}
		}
		// Linv = L⁻¹ (lower-triangular) by forward substitution on L·X = I.
		linv := make([][]float64, n)
		for i := range n {
			linv[i] = make([]float64, n)
		}
		// Forward substitution — each COLUMN j is an independent solve (reads only l and
		// its own column linv[·][j]); parallelize over j. Writes linv[i][j] touch disjoint
		// (row i, col j) slots, so concurrent columns never collide. Bit-identical to serial.
		logdetParallelIdx(n, n*n*n/6, func(j int) {
			linv[j][j] = 1 / l[j][j]
			for i := j + 1; i < n; i++ {
				var s float64
				for k := j; k < i; k++ {
					s += l[i][k] * linv[k][j]
				}
				linv[i][j] = -s / l[i][i]
			}
		})
		// Ā = ḡ·A⁻¹, A⁻¹ = Linvᵀ·Linv (symmetric): (A⁻¹)_ij = Σ_{k≥max(i,j)} Linv[k,i]·Linv[k,j].
		// Transpose Linv to column-major (linvT[i][k] = Linv[k,i]) so each dot walks two
		// CONTIGUOUS rows instead of striding down columns of a [][]float64 — bit-identical
		// (same k order), but cache-resident. Linv is lower-triangular, so column i is nonzero
		// only for k ≥ i. Then exploit A⁻¹'s symmetry: compute only i ≤ j and mirror (j,i)=(i,j),
		// halving the O(n³) accumulation. Both transforms preserve the exact per-element
		// summation, so the result is bit-identical to the serial column-strided double loop.
		linvT := make([][]float64, n)
		for i := range n {
			col := make([]float64, n)
			for k := i; k < n; k++ {
				col[k] = linv[k][i]
			}
			linvT[i] = col
		}
		abar := tensor.New(a.Dtype(), a.Shape())
		// Rows are disjoint: worker for row i writes (i,·) and its mirror (·,i) only within its
		// own j ≥ i range, so no two i's touch the same (r,c). Chunk-parallel over i, STRIPED
		// (i = w, w+nw, …) to balance the triangular per-row load; bit-identical to serial.
		logdetParallelIdx(n, n*n/2, func(i int) {
			ri := linvT[i]
			for j := i; j < n; j++ {
				rj := linvT[j]
				var s float64
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
