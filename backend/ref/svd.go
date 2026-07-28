package ref

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// svdKernel computes the reduced (economy) singular value decomposition A = U·Σ·Vᵀ
// of an m×n matrix with m ≥ n via one-sided Jacobi (Hestenes) rotations (numpy.linalg.svd
// full_matrices=False; Golub & Van Loan §8.6.3): U ∈ Rᵐˣⁿ orthonormal columns, s the
// singular values σ₁ ≥ … ≥ σₙ ≥ 0 (length n), V ∈ Rⁿˣⁿ orthonormal columns. One-sided
// Jacobi works on A directly (never forms AᵀA) → high relative accuracy. Three outputs
// [U, s, V]; f64 arithmetic (§V10), F32 factors in f64 and narrows on store.
func svdKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: svd wants 1 input, got %d", len(in))
	}
	a := in[0]
	if a.Ndim() != 2 {
		return nil, fmt.Errorf("ref: svd needs a rank-2 matrix, got shape %v", a.Shape())
	}
	m, n := a.Shape()[0], a.Shape()[1]
	if m < n {
		return nil, fmt.Errorf("ref: svd needs m ≥ n (tall/square), got %dx%d", m, n)
	}

	// Flat [m*n] and [n*n] row-major working buffers, not slices of rows. The Jacobi
	// sweep below walks COLUMNS — acol[k][i] and vmat[k][i] with k varying — so a
	// row-of-slices layout dereferences a different heap row per step. Flat makes
	// each walk a constant stride through one allocation and drops m+n allocations
	// to 2. Index arithmetic only, so the decomposition is bit-identical.
	acol := make([]float64, m*n) // working copy; columns get orthogonalized into U·Σ
	if as, ok := f64Data(a); ok {
		copy(acol, as)
	} else {
		for i := range m {
			for j := range n {
				acol[i*n+j] = a.AtF64(i, j)
			}
		}
	}
	vmat := make([]float64, n*n)
	for i := range n {
		vmat[i*n+i] = 1
	}
	const tol = 1e-14
	for range 100 { // sweeps until columns mutually orthogonal
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				var alpha, beta, gamma float64
				for k := range m {
					alpha += acol[k*n+i] * acol[k*n+i]
					beta += acol[k*n+j] * acol[k*n+j]
					gamma += acol[k*n+i] * acol[k*n+j]
				}
				if alpha == 0 || beta == 0 {
					continue
				}
				rel := math.Abs(gamma) / math.Sqrt(alpha*beta)
				if rel > off {
					off = rel
				}
				if rel <= tol {
					continue
				}
				zeta := (beta - alpha) / (2 * gamma)
				var t float64
				if zeta >= 0 {
					t = 1 / (zeta + math.Sqrt(1+zeta*zeta))
				} else {
					t = -1 / (-zeta + math.Sqrt(1+zeta*zeta))
				}
				c := 1 / math.Sqrt(1+t*t)
				sn := c * t
				for k := range m {
					ai, aj := acol[k*n+i], acol[k*n+j]
					acol[k*n+i] = c*ai - sn*aj
					acol[k*n+j] = sn*ai + c*aj
				}
				for k := range n {
					vi, vj := vmat[k*n+i], vmat[k*n+j]
					vmat[k*n+i] = c*vi - sn*vj
					vmat[k*n+j] = sn*vi + c*vj
				}
			}
		}
		if off <= tol {
			break
		}
	}
	sigma := make([]float64, n)
	for j := range n {
		var nrm float64
		for k := range m {
			nrm += acol[k*n+j] * acol[k*n+j]
		}
		sigma[j] = math.Sqrt(nrm)
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(x, y int) bool { return sigma[order[x]] > sigma[order[y]] })

	ut := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{m, n})
	st := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n})
	vt := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n, n})
	for jj, j := range order {
		st.SetF64(sigma[j], jj)
		if sigma[j] > 0 {
			for k := range m {
				ut.SetF64(acol[k*n+j]/sigma[j], k, jj)
			}
		}
		for k := range n {
			vt.SetF64(vmat[k*n+j], k, jj)
		}
	}
	return []*tensor.Tensor{ut, st, vt}, nil
}

func init() {
	std.add(backend.OpSVD, tensor.F32, svdKernel)
	std.add(backend.OpSVD, tensor.F64, svdKernel)
}
