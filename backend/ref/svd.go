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
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
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

	// COLUMN-MAJOR working buffers: acol[j*m+k] is row k of column j, vmat[j*n+k] likewise. The
	// Jacobi sweep is entirely column work — it forms three inner products over a column pair and
	// then rotates that pair — so with a row-major layout every one of those steps strides by n and
	// touches its own cache line to use one element. Column-major makes each of them a contiguous
	// run, and the rotation writes two contiguous columns.
	//
	// Flat rather than a slice of columns for the same reason the row-major version was flat: two
	// allocations instead of m+n. Index arithmetic only, so the decomposition is bit-identical —
	// every sum keeps its operands and its ascending-k order.
	acol := make([]float64, m*n) // working copy; columns get orthogonalized into U·Σ
	if as, ok := f64Data(a); ok {
		for i := range m {
			row := as[i*n : i*n+n]
			for j := range n {
				acol[j*m+i] = row[j]
			}
		}
	} else {
		for i := range m {
			for j := range n {
				acol[j*m+i] = a.AtF64(i, j)
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
				ci, cj := acol[i*m:i*m+m], acol[j*m:j*m+m]
				//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
				for k := range m {
					alpha += ci[k] * ci[k]
					beta += cj[k] * cj[k]
					gamma += ci[k] * cj[k]
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
					ai, aj := ci[k], cj[k]
					ci[k] = c*ai - sn*aj
					cj[k] = sn*ai + c*aj
				}
				vi, vj := vmat[i*n:i*n+n], vmat[j*n:j*n+n]
				for k := range n {
					a, b := vi[k], vj[k]
					vi[k] = c*a - sn*b
					vj[k] = sn*a + c*b
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
		cj := acol[j*m : j*m+m]
		//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
		for k := range m {
			nrm += cj[k] * cj[k]
		}
		sigma[j] = math.Sqrt(nrm)
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	//perfscan:ignore PS3002,PS6009 reference oracle: intentionally simple, correctness baseline not an optimization target
	sort.SliceStable(order, func(x, y int) bool { return sigma[order[x]] > sigma[order[y]] })

	ut := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{m, n})
	st := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n})
	vt := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n, n})
	for jj, j := range order {
		st.SetF64(sigma[j], jj)
		if sigma[j] > 0 {
			cj := acol[j*m : j*m+m]
			for k := range m {
				ut.SetF64(cj[k]/sigma[j], k, jj)
			}
		}
		vj := vmat[j*n : j*n+n]
		for k := range n {
			vt.SetF64(vj[k], k, jj)
		}
	}
	return []*tensor.Tensor{ut, st, vt}, nil
}

func init() {
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpSVD, tensor.F32, svdKernel)
	std.add(backend.OpSVD, tensor.F64, svdKernel)
}
