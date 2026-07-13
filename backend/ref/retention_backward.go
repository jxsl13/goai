package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// retentionBackwardKernel is the RetNet retention backward (§T174, gradient of §R114's OpRetention).
// Inputs are (Q,K,V,dO) each [L,d] (dO the upstream gradient of the retention output); it returns
// (dQ,dK,dV). Making the backward a dispatched op — instead of a hand-rolled loop inside the VJP —
// lets it run on whichever backend is active (GPU when training on Metal/Vulkan), exactly as the
// matmul and attention backwards do. Forward O=(A⊙D)V, A_nm=Σ_i Q_ni K_mi, D_nm=γ^(n−m) (m≤n),
// P=A⊙D, O=PV. Given dO:
//
//	dP_nm = Σ_j dO_nj·V_mj ;  dA_nm = dP_nm·γ^(n−m)
//	dQ_ni = Σ_{m≤n} dA_nm·K_mi ;  dK_mi = Σ_{n≥m} dA_nm·Q_ni ;  dV_mj = Σ_{n≥m} P_nm·dO_nj
//
// f64 accumulation (§V10).
func retentionBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("ref: retention-backward wants (Q,K,V,dO), got %d inputs", len(in))
	}
	q, k, v, g := in[0], in[1], in[2], in[3]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("ref: retention-backward needs rank-2 tensors")
		}
	}
	l, kd := q.Shape()[0], q.Shape()[1] // key dim
	vd := v.Shape()[1]                  // value dim (may differ)
	if k.Shape()[0] != l || g.Shape()[0] != l || k.Shape()[1] != kd || g.Shape()[1] != vd {
		return nil, fmt.Errorf("ref: retention-backward needs Q,K [L,dk], V,dO [L,dv]; got Q%v K%v V%v dO%v", q.Shape(), k.Shape(), v.Shape(), g.Shape())
	}
	pa, _ := attrs.(backend.RetentionAttrs)
	gamma := pa.Gamma

	dq := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	dk := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())
	dv := tensor.NewOn(ctx.Device(), v.Dtype(), v.Shape())
	for n := range l {
		for m := 0; m <= n; m++ {
			decay := math.Pow(gamma, float64(n-m))
			var a float64
			for i := range kd {
				a += q.AtF64(n, i) * k.AtF64(m, i)
			}
			pnm := a * decay
			var dp float64
			for j := range vd { // value dim
				gnj := g.AtF64(n, j)
				dp += gnj * v.AtF64(m, j)
				dv.SetF64(dv.AtF64(m, j)+pnm*gnj, m, j)
			}
			dA := dp * decay
			for i := range kd { // key dim
				dq.SetF64(dq.AtF64(n, i)+dA*k.AtF64(m, i), n, i)
				dk.SetF64(dk.AtF64(m, i)+dA*q.AtF64(n, i), m, i)
			}
		}
	}
	return []*tensor.Tensor{dq, dk, dv}, nil
}

func init() {
	std.add(backend.OpRetentionBackward, tensor.F32, retentionBackwardKernel)
	std.add(backend.OpRetentionBackward, tensor.F64, retentionBackwardKernel)
}
