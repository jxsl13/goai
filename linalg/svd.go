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
	col := toColMajor(a, m, n)
	// Same one-backing-array treatment as toColMajor: n row views over a single n*n buffer
	// instead of n allocations. Values and traversal are untouched.
	vbuf := make([]float64, n*n)
	vmat := make([][]float64, n)
	for i := range n {
		vmat[i] = vbuf[i*n : (i+1)*n : (i+1)*n]
		vmat[i][i] = 1
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
				for k := range n { // rotate columns i,j of V
					vi, vj := vmat[k][i], vmat[k][j]
					vmat[k][i] = c*vi - sn*vj
					vmat[k][j] = sn*vi + c*vj
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
	// SortStableFunc, not SortFunc: this comparator orders on sigma alone and relies on
	// stability to keep equal singular values in their original index order, which is what
	// makes the returned basis reproducible. sort.SliceStable reaches its swap through
	// reflectlite.Swapper and ALLOCATES on every call (PS6009), once per SVD.
	slices.SortStableFunc(order, func(x, y int) int {
		switch a, b := sigma[x], sigma[y]; {
		case a > b:
			return -1
		case a < b:
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
			vOut[k*n+jj] = vmat[k][j]
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
	// One backing array carved into n column views rather than n separate allocations. The
	// [][]float64 type and every downstream index are unchanged; only the addresses move, so
	// this is bit-identical. Three-index slices cap each column so an accidental append can
	// never reach into the next one.
	buf := make([]float64, n*m)
	c := make([][]float64, n)
	for j := range n {
		c[j] = buf[j*m : (j+1)*m : (j+1)*m]
	}
	if d, ok := flatRowMajor(a); ok {
		for j := range n {
			cj := c[j]
			for k := range m {
				cj[k] = d[k*n+j]
			}
		}
		return c
	}
	for j := range n {
		cj := c[j]
		for k := range m {
			cj[k] = a.AtF64(k, j)
		}
	}
	return c
}
