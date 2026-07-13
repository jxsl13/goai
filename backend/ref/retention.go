package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// retentionKernel is the parallel form of RetNet retention (§R114; Sun et al. 2023 arXiv:2307.08621
// Eq.5): O = (QKᵀ ⊙ D)·V, where D is the causal decay mask D_nm = γ^(n−m) for n≥m else 0 (γ⁰=1 on
// the diagonal, 0 above). It replaces attention's softmax with the decay mask, so it has an exactly
// equivalent O(1)-per-step recurrent form (Eq.6, see nn.RetentionRecurrent). Q,K,V are [L,d] (a
// single head; the reference operates per head). Accumulation is f64 (§V10).
func retentionKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: retention wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("ref: retention needs rank-2 Q,K,V, got %dD", t.Ndim())
		}
	}
	l, dk := q.Shape()[0], q.Shape()[1] // Q,K key dim
	dv := v.Shape()[1]                  // V value dim (may differ from dk — RetNet value expansion)
	if k.Shape()[0] != l || v.Shape()[0] != l || k.Shape()[1] != dk {
		return nil, fmt.Errorf("ref: retention needs Q,K [L,dk] and V [L,dv], got Q%v K%v V%v", q.Shape(), k.Shape(), v.Shape())
	}
	pa, _ := attrs.(backend.RetentionAttrs)
	g := pa.Gamma
	if g < 0 || g > 1 {
		return nil, fmt.Errorf("ref: retention gamma %g out of [0,1]", g)
	}

	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{l, dv})
	for n := range l {
		// P_nm = (Σ_i Q[n,i]·K[m,i]) · γ^(n−m) for m ≤ n, i over the key dim dk
		p := make([]float64, n+1)
		for m := 0; m <= n; m++ {
			var a float64
			for i := range dk {
				a += q.AtF64(n, i) * k.AtF64(m, i)
			}
			p[m] = a * math.Pow(g, float64(n-m))
		}
		for j := range dv { // j over the value dim dv
			var acc float64
			for m := 0; m <= n; m++ {
				acc += p[m] * v.AtF64(m, j)
			}
			out.SetF64(acc, n, j)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpRetention, tensor.F32, retentionKernel)
	std.add(backend.OpRetention, tensor.F64, retentionKernel)
}
