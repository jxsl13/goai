package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DoRA weight-decomposition VJP (Liu et al. 2024, arXiv:2402.09353 Eq. 5). With
// n[j] = ‖V[:,j]‖, W'[i,j] = m[j]·V[i,j]/n[j], and s[j] = Σ_i g[i,j]·V[i,j], the
// exact gradients (differentiating THROUGH the column norm) are:
//
//	∂m[j]   = s[j] / n[j]
//	∂V[a,j] = (m[j]/n[j])·( g[a,j] − V[a,j]·s[j]/n[j]² )
//
// This is the true gradient of the forward, so it passes a finite-difference
// gradient check (§V2). NOTE: the reference PEFT/DoRA implementation DETACHES the
// column norm in its backward (dropping the V·s/n² term) purely to save memory —
// a deliberate approximation; we compute the exact gradient instead, a documented
// divergence in the backward only (the forward is identical) (§R70).
func init() {
	RegisterVJP(backend.OpDoRAWeight, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		v, m := in[0], in[1]
		rows, cols := v.Shape()[0], v.Shape()[1]

		dv := tensor.New(v.Dtype(), v.Shape())
		dm := tensor.New(m.Dtype(), m.Shape())
		for j := range cols {
			var ss, s float64
			for i := range rows {
				x := v.AtF64(i, j)
				ss += x * x
				s += g.AtF64(i, j) * x
			}
			n := math.Sqrt(ss)
			if n == 0 {
				continue // degenerate zero column → zero gradients
			}
			mj := m.AtF64(j)
			dm.SetF64(s/n, j)
			for i := range rows {
				dv.SetF64(mj/n*(g.AtF64(i, j)-v.AtF64(i, j)*s/(n*n)), i, j)
			}
		}
		return []*tensor.Tensor{dv, dm}, nil
	})
}
