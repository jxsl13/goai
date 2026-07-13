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

	acol := make([][]float64, m) // working copy; columns get orthogonalized into U·Σ
	for i := range m {
		acol[i] = make([]float64, n)
		for j := range n {
			acol[i][j] = a.AtF64(i, j)
		}
	}
	vmat := make([][]float64, n)
	for i := range n {
		vmat[i] = make([]float64, n)
		vmat[i][i] = 1
	}
	const tol = 1e-14
	for range 100 { // sweeps until columns mutually orthogonal
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				var alpha, beta, gamma float64
				for k := range m {
					alpha += acol[k][i] * acol[k][i]
					beta += acol[k][j] * acol[k][j]
					gamma += acol[k][i] * acol[k][j]
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
					ai, aj := acol[k][i], acol[k][j]
					acol[k][i] = c*ai - sn*aj
					acol[k][j] = sn*ai + c*aj
				}
				for k := range n {
					vi, vj := vmat[k][i], vmat[k][j]
					vmat[k][i] = c*vi - sn*vj
					vmat[k][j] = sn*vi + c*vj
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
			nrm += acol[k][j] * acol[k][j]
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
				ut.SetF64(acol[k][j]/sigma[j], k, jj)
			}
		}
		for k := range n {
			vt.SetF64(vmat[k][j], k, jj)
		}
	}
	return []*tensor.Tensor{ut, st, vt}, nil
}

func init() {
	std.add(backend.OpSVD, tensor.F32, svdKernel)
	std.add(backend.OpSVD, tensor.F64, svdKernel)
}
