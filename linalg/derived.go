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
func Pinv(a *tensor.Tensor) (*tensor.Tensor, error) {
	u, s, v, err := SVD(a)
	if err != nil {
		return nil, err
	}
	m, n := a.Shape()[0], a.Shape()[1]
	p := s.Numel()
	cutoff := 1e-15 * s.AtF64(0) // rcond·σ_max
	out := make([]float64, n*m)  // A⁺ is [n,m]
	for k := range p {
		sk := s.AtF64(k)
		if sk <= cutoff {
			continue
		}
		inv := 1 / sk
		for i := range n { // A⁺[i,j] += V[i,k]·(1/σ_k)·U[j,k]
			vik := v.AtF64(i, k) * inv
			for j := range m {
				out[i*m+j] += vik * u.AtF64(j, k)
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
