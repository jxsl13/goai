package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SolveSPD VJP — standard reverse-mode of a linear solve. For X = A⁻¹·B (A SPD) with
// output cotangent Ḡ, the differential dX = −A⁻¹·dA·X + A⁻¹·dB gives, via
// ⟨Ḡ, dX⟩ = ⟨B̄, dB⟩ + ⟨Ā, dA⟩:
//
//	B̄ = A⁻ᵀ·Ḡ = A⁻¹·Ḡ   (another SPD solve, A symmetric)
//	Ā = −B̄·Xᵀ           (raw), symmetrized to −½(B̄·Xᵀ + X·B̄ᵀ)
//
// The ½-symmetrization is the composition-correct convention for the symmetric input
// A (the tape's full Frobenius product over symmetric dA), matching the Cholesky VJP.
// B̄ is obtained by reusing the forward op (Cholesky solve on the same A). All
// arithmetic is f64 (§V10).
func init() {
	RegisterVJP(backend.OpSolveSPD, func(ctx *backend.Context, in, out []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		a, b := in[0], in[1]
		x := out[0]
		n := a.Shape()[0]

		// B̄ = A⁻¹·Ḡ (same SPD solve as the forward).
		bbarOut, err := backend.Execute(ctx, backend.OpSolveSPD, []*tensor.Tensor{a, g}, nil)
		if err != nil {
			return nil, err
		}
		bbar := bbarOut[0]

		// column count + accessor shared by X and B̄ (vector [n] → k=1).
		vector := b.Ndim() == 1
		k := 1
		if !vector {
			k = b.Shape()[1]
		}
		at := func(t *tensor.Tensor, i, c int) float64 {
			if vector {
				return t.AtF64(i)
			}
			return t.AtF64(i, c)
		}

		// Ā = −½(B̄·Xᵀ + X·B̄ᵀ): Ā[i,j] = −½·Σ_c (B̄[i,c]·X[j,c] + X[i,c]·B̄[j,c]).
		abar := tensor.New(a.Dtype(), a.Shape())
		for i := range n {
			for j := range n {
				var s float64
				for c := range k {
					s += at(bbar, i, c)*at(x, j, c) + at(x, i, c)*at(bbar, j, c)
				}
				abar.SetF64(-0.5*s, i, j)
			}
		}
		return []*tensor.Tensor{abar, bbar}, nil
	})
}
