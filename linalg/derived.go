package linalg

import (
	"math"

	"github.com/jxsl13/goai/tensor"
)

// f64Eps is machine epsilon for IEEE-754 binary64 (≈2.220446e-16), used for the standard
// numpy-style singular-value cutoffs.
const f64Eps = 2.220446049250313e-16

// Pinv returns the Moore-Penrose pseudoinverse A⁺ of an m×n matrix via the SVD (§R118; Golub & Van
// Loan §5.5.2): with A = U·diag(σ)·Vᵀ, A⁺ = V·diag(σ⁺)·Uᵀ where σ⁺ᵢ = 1/σᵢ for σᵢ > cutoff and 0
// otherwise, cutoff = rcond·σ_max with numpy's default rcond = 1e-15. A⁺ satisfies the four
// Moore-Penrose conditions; for a full-rank tall matrix it is the least-squares left-inverse
// (AᵀA)⁻¹Aᵀ, and A⁺·b is the least-squares/minimum-norm solution. Result is [n×m] f64.
//
//perfscan:ignore PS6004 correctness/invariant check, no wall-clock; Pinv is utility
func Pinv(a *tensor.Tensor) (*tensor.Tensor, error) {
	u, s, v, err := SVD(a)
	if err != nil {
		return nil, err
	}
	m, n := a.Shape()[0], a.Shape()[1]
	p := s.Numel()
	cutoff := 1e-15 * s.AtF64(0) // rcond·σ_max
	out := make([]float64, n*m)  // A⁺ is [n,m]
	// The innermost accumulation runs O(p·n·m) times and read U through AtF64 —
	// an interface hop into storage plus a variadic index — for one multiply-add.
	// SVD returns freshly built contiguous tensors, so a flat row-major view is
	// available and the read becomes uf[j*p+k]. Same operands, same order, same
	// accumulation into out, so bit-identical; the accessor path is retained for
	// tensors a flat view cannot expose.
	uf, uok := flatRowMajor(u)
	vf, vok := flatRowMajor(v)
	//perfscan:ignore PS1006,PS3046 Pinv already has flat fast path; SVD-dominated utility, no hot caller | Pinv reconstruction, SVD-dominated fit
	for k := range p {
		sk := s.AtF64(k)
		if sk <= cutoff {
			continue
		}
		inv := 1 / sk
		//perfscan:ignore PS3040 Pinv inner loop already flat fast-pathed; SVD-dominated utility
		for i := range n { // A⁺[i,j] += V[i,k]·(1/σ_k)·U[j,k]
			var vik float64
			if vok {
				//perfscan:ignore PS6011 strided uf read but flat fast path present; utility, no hot caller
				vik = vf[i*p+k] * inv
			} else {
				vik = v.AtF64(i, k) * inv
			}
			row := out[i*m : i*m+m]
			if uok {
				for j := range m {
					//perfscan:ignore PS6011 strided uf read, flat fast path present; SVD-dominated utility
					row[j] += vik * uf[j*p+k]
				}
			} else {
				for j := range m {
					row[j] += vik * u.AtF64(j, k)
				}
			}
		}
	}
	return tensor.FromFloat64(tensor.Shape{n, m}, out), nil
}

// Rank returns the numerical rank of a — the number of singular values greater than the tolerance
// tol = σ_max · max(m,n) · eps (numpy.linalg.matrix_rank's default). §R118.
func Rank(a *tensor.Tensor) (int, error) {
	m, n, err := shapeMN(a)
	if err != nil {
		return 0, err
	}
	_, s, _, err := SVD(a)
	if err != nil {
		return 0, err
	}
	tol := s.AtF64(0) * float64(max(m, n)) * f64Eps
	r := 0
	//perfscan:ignore PS1001 Rank AtF64 over min(m,n) values (small); SVD-dominated utility
	for k := range s.Numel() {
		if s.AtF64(k) > tol {
			r++
		}
	}
	return r, nil
}

// Cond returns the 2-norm condition number σ_max/σ_min of a (numpy.linalg.cond); +Inf when a is
// singular (σ_min = 0). §R118.
func Cond(a *tensor.Tensor) (float64, error) {
	_, s, _, err := SVD(a)
	if err != nil {
		return 0, err
	}
	smin := s.AtF64(s.Numel() - 1)
	if smin == 0 {
		return math.Inf(1), nil
	}
	return s.AtF64(0) / smin, nil
}

// Norm2 returns the spectral (induced 2-) norm ‖A‖₂ = σ_max, the largest singular value. §R118.
func Norm2(a *tensor.Tensor) (float64, error) {
	_, s, _, err := SVD(a)
	if err != nil {
		return 0, err
	}
	return s.AtF64(0), nil
}

// NormFro returns the Frobenius norm ‖A‖_F = √(Σ_ij a_ij²). §R118.
func NormFro(a *tensor.Tensor) (float64, error) {
	m, n, err := shapeMN(a)
	if err != nil {
		return 0, err
	}
	var s float64
	for i := range m {
		//perfscan:ignore PS1005 NormFro linalg util, zero DL-path callers; cold analysis fn
		for j := range n {
			v := a.AtF64(i, j)
			s += v * v
		}
	}
	return math.Sqrt(s), nil
}

// Norm1 returns the induced 1-norm ‖A‖₁ = max_j Σ_i |a_ij| (maximum absolute column sum). §R118.
func Norm1(a *tensor.Tensor) (float64, error) {
	m, n, err := shapeMN(a)
	if err != nil {
		return 0, err
	}
	best := 0.0
	for j := range n {
		var c float64
		//perfscan:ignore PS1005 Norm1 linalg util, no hot-path caller; cold analysis fn
		for i := range m {
			c += math.Abs(a.AtF64(i, j))
		}
		if c > best {
			best = c
		}
	}
	return best, nil
}

// NormInf returns the induced ∞-norm ‖A‖_∞ = max_i Σ_j |a_ij| (maximum absolute row sum). §R118.
func NormInf(a *tensor.Tensor) (float64, error) {
	m, n, err := shapeMN(a)
	if err != nil {
		return 0, err
	}
	best := 0.0
	for i := range m {
		var r float64
		//perfscan:ignore PS1005 NormInf linalg util, no hot-path caller; cold analysis fn
		for j := range n {
			r += math.Abs(a.AtF64(i, j))
		}
		if r > best {
			best = r
		}
	}
	return best, nil
}

// flatRowMajor exposes a rank-2 tensor's storage as a row-major []float64 when the
// layout permits it, so a hot loop can index arithmetically instead of dispatching
// through AtF64. Reports false for anything strided, offset, or of another dtype,
// leaving the caller's accessor path in charge.
func flatRowMajor(t *tensor.Tensor) ([]float64, bool) {
	if t == nil || t.Ndim() != 2 || t.Dtype() != tensor.F64 || !t.IsContiguous() || t.Offset() != 0 {
		return nil, false
	}
	d := t.Storage().F64()
	if len(d) < t.Numel() {
		return nil, false
	}
	return d, true
}
