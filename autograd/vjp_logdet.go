package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

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
		for j := range n {
			linv[j][j] = 1 / l[j][j]
			for i := j + 1; i < n; i++ {
				var s float64
				for k := j; k < i; k++ {
					s += l[i][k] * linv[k][j]
				}
				linv[i][j] = -s / l[i][i]
			}
		}
		// Ā = ḡ·A⁻¹, A⁻¹ = Linvᵀ·Linv (symmetric): (A⁻¹)_ij = Σ_{k≥max(i,j)} Linv[k,i]·Linv[k,j].
		abar := tensor.New(a.Dtype(), a.Shape())
		for i := range n {
			for j := range n {
				lo := max(i, j)
				var s float64
				for k := lo; k < n; k++ {
					s += linv[k][i] * linv[k][j]
				}
				abar.SetF64(gv*s, i, j)
			}
		}
		return []*tensor.Tensor{abar}, nil
	})
}
