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
//perfscan:ignore PS6004 dual path unverifiable: the fallback is unreachable, u and v are SVD outputs
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
	// The accessor arm below is UNREACHABLE in practice and kept only as a defensive fallback:
	// u and v are SVD's own outputs, freshly built contiguous f64 tensors, so both guards always
	// succeed no matter what the caller passed in. Confirmed by panicking in that arm — the whole
	// linalg suite stayed green. The PS6004 suppression for it sits on the func declaration, which
	// is where that check reports.
	uf, uok := flatRowMajor(u)
	vf, vok := flatRowMajor(v)
	for k := range p {
		sk := s.AtF64(k)
		if sk <= cutoff {
			continue
		}
		inv := 1 / sk
		for i := range n { // A⁺[i,j] += V[i,k]·(1/σ_k)·U[j,k]
			var vik float64
			if vok {
				vik = vf[i*p+k] * inv
			} else {
				vik = v.AtF64(i, k) * inv
			}
			row := out[i*m : i*m+m]
			if uok {
				for j := range m {
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
	// AtF64 is not inlinable — it is a real call plus a storage type switch, against a
	// three-cycle add chain — so on the contiguous F64 path it dominates this loop. Same
	// (i,j) traversal, same accumulation sequence into s, so bit-identical; flatRowMajor
	// rejects strided, offset, non-F64 and non-rank-2 inputs, which keep the accessor arm.
	if d, ok := flatRowMajor(a); ok {
		for i := range m {
			row := d[i*n : i*n+n : i*n+n]
			for j := range n {
				v := row[j]
				s += v * v
			}
		}
		return math.Sqrt(s), nil
	}
	for i := range m {
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
	// Row sums are already the row-major traversal, so only the accessor changes. Each row
	// accumulates its elements in the same ascending-j order, and the comparison sequence
	// over i is unchanged, so the result is bit-identical.
	if d, ok := flatRowMajor(a); ok {
		for i := range m {
			row := d[i*n : i*n+n : i*n+n]
			var r float64
			for j := range n {
				r += math.Abs(row[j])
			}
			if r > best {
				best = r
			}
		}
		return best, nil
	}
	for i := range m {
		var r float64
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
	if t == nil || t.Ndim() != 2 {
		return nil, false
	}
	return flatContig(t)
}

// flatContig is flatRowMajor without the rank restriction: the contiguous f64 backing slice of a
// tensor of ANY rank, in row-major order. CholSolve's right-hand side is a vector or a matrix
// depending on the call, and both want the same typed fast path.
func flatContig(t *tensor.Tensor) ([]float64, bool) {
	if t == nil || t.Dtype() != tensor.F64 || !t.IsContiguous() || t.Offset() != 0 {
		return nil, false
	}
	d := t.Storage().F64()
	if len(d) < t.Numel() {
		return nil, false
	}
	return d, true
}
