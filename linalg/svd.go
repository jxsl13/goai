package linalg

import (
	"math"
	"slices"

	"github.com/jxsl13/goai/tensor"
)

// SVD computes the reduced (economy) singular value decomposition A = U·Σ·Vᵀ of an m×n matrix via
// the one-sided Jacobi (Hestenes) method (§R117; Golub & Van Loan §8.6.3, Demmel §5.4.3, Hestenes
// 1958). With p = min(m,n): U ∈ R^{m×p} has orthonormal columns (the left singular vectors), s holds
// the singular values σ₁ ≥ σ₂ ≥ … ≥ σ_p ≥ 0 (a length-p vector), and V ∈ R^{n×p} has orthonormal
// columns (the right singular vectors), so A = U · diag(s) · Vᵀ. One-sided Jacobi works directly on
// A (never forming AᵀA), giving high relative accuracy even for tiny singular values. Returns fresh
// f64 tensors; the input is not mutated.
func SVD(a *tensor.Tensor) (u, s, v *tensor.Tensor, err error) {
	m, n, err := shapeMN(a)
	if err != nil {
		return nil, nil, nil, err
	}
	if m < n {
		// SVD(A) from SVD(Aᵀ): Aᵀ = U'Σ'V'ᵀ ⇒ A = V'Σ'U'ᵀ, so U=V', V=U', s unchanged.
		ut, st, vt, e := SVD(transposeT(a, m, n))
		if e != nil {
			return nil, nil, nil, e
		}
		return vt, st, ut, nil
	}
	// one-sided Jacobi on the columns of a working copy; V accumulates the right rotations.
	// V is held TRANSPOSED in a flat [n*n] buffer — vT[i*n+k] is V[k][i] — because the rotation
	// touches two COLUMNS of V, and transposed they are two contiguous rows. Held row-major it
	// strode by n on every one of the 2n reads and 2n writes per rotation, which at n=192 was
	// about a sixth of the whole factorization; the columns of A next to it were already
	// contiguous for exactly this reason.
	//
	// Layout only: each entry takes the same two multiplies and one add, in the same order, from
	// the same operands, so every bit of the result is unchanged. The transpose back happens once
	// when the output is assembled.
	col := toColMajor(a, m, n)
	vT := make([]float64, n*n) // vT[i*n+k] = V[k][i]
	for i := range n {
		vT[i*n+i] = 1 // the identity is its own transpose
	}
	const tol = 1e-14
	for range 100 { // sweeps until convergence (one-sided Jacobi converges in ~10)
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				ci, cj := col[i], col[j] // contiguous columns i,j (streamed by every loop below)
				var alpha, beta, gamma float64
				for k := range m {
					alpha += ci[k] * ci[k]
					beta += cj[k] * cj[k]
					gamma += ci[k] * cj[k]
				}
				if alpha == 0 || beta == 0 {
					continue // a zero column can't be rotated meaningfully
				}
				rel := math.Abs(gamma) / math.Sqrt(alpha*beta)
				if rel > off {
					off = rel
				}
				if rel <= tol {
					continue // already orthogonal
				}
				// symmetric Jacobi rotation zeroing the (i,j) inner product; smaller root t=tanθ
				zeta := (beta - alpha) / (2 * gamma)
				var t float64
				if zeta >= 0 {
					t = 1 / (zeta + math.Sqrt(1+zeta*zeta))
				} else {
					t = -1 / (-zeta + math.Sqrt(1+zeta*zeta))
				}
				c := 1 / math.Sqrt(1+t*t)
				sn := c * t
				for k := range m { // rotate columns i,j of A (in place on their contiguous slices)
					ai, aj := ci[k], cj[k]
					ci[k] = c*ai - sn*aj
					cj[k] = sn*ai + c*aj
				}
				vi, vj := vT[i*n:i*n+n], vT[j*n:j*n+n] // rows of vT = columns of V
				for k := range n {
					a, b := vi[k], vj[k]
					vi[k] = c*a - sn*b
					vj[k] = sn*a + c*b
				}
			}
		}
		if off <= tol {
			break // all columns mutually orthogonal to working precision
		}
	}
	// singular values = final column norms; left vectors = normalized columns
	sigma := make([]float64, n)
	for j := range n {
		cj := col[j]
		var nrm float64
		for k := range m {
			nrm += cj[k] * cj[k]
		}
		sigma[j] = math.Sqrt(nrm)
	}
	// sort descending, permute U and V columns
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	// slices.SortStableFunc, not sort.SliceStable: the latter reaches its swap through reflection
	// and allocates on every call (PS6009). Same comparator, same stable order.
	slices.SortStableFunc(order, func(a, b int) int {
		switch {
		case sigma[a] > sigma[b]:
			return -1
		case sigma[a] < sigma[b]:
			return 1
		}
		return 0
	})

	uMat := make([]float64, m*n)
	sVec := make([]float64, n)
	vOut := make([]float64, n*n)
	for jj, j := range order {
		sVec[jj] = sigma[j]
		cj := col[j]
		for k := range m {
			if sigma[j] > 0 {
				uMat[k*n+jj] = cj[k] / sigma[j]
			}
		}
		for k := range n {
			vOut[k*n+jj] = vT[j*n+k]
		}
	}
	return tensor.FromFloat64(tensor.Shape{m, n}, uMat),
		tensor.FromFloat64(tensor.Shape{n}, sVec),
		tensor.FromFloat64(tensor.Shape{n, n}, vOut), nil
}

// transposeT returns the n×m transpose of the m×n tensor a.
func transposeT(a *tensor.Tensor, m, n int) *tensor.Tensor {
	d := make([]float64, m*n)
	for i := range n {
		for j := range m {
			d[i*m+j] = a.AtF64(j, i)
		}
	}
	return tensor.FromFloat64(tensor.Shape{n, m}, d)
}

// toColMajor copies the m×n input into a COLUMN-major working buffer: col[j] is column j
// as one contiguous length-m slice. The one-sided Jacobi body is entirely column-oriented
// (every dot-product and rotation runs down the m rows of two columns), so this layout lets
// each inner loop stream two contiguous arrays instead of striding across m separate row
// slices — the cache-locality win for tall (m≫n) matrices. Same values as toRect, just
// transposed, so the k-accumulation order and every float64 result are bit-identical.
func toColMajor(a *tensor.Tensor, m, n int) [][]float64 {
	c := make([][]float64, n)
	for j := range n {
		c[j] = make([]float64, m)
		for k := range m {
			c[j][k] = a.AtF64(k, j)
		}
	}
	return c
}
