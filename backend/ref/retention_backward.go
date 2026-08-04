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
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
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
	// Devirtualised typed core (§T646 follow-up): inputs exposed once as flat
	// []float64 (exact widening), γ^t precomputed once (same math.Pow calls,
	// identical values), gradients accumulated through a generic refFloat core so
	// the F32 instantiation reproduces the generic path's per-update
	// widen/add/narrow sequence exactly — bit-identical.
	if q.Dtype() == k.Dtype() && q.Dtype() == v.Dtype() && q.Dtype() == g.Dtype() {
		if qs, ok := f64Data(q); ok {
			ks, _ := f64Data(k)
			vs, _ := f64Data(v)
			gs, _ := f64Data(g)
			switch q.Dtype() {
			case tensor.F64:
				retentionBackwardCore(qs, ks, vs, gs,
					dq.Storage().F64(), dk.Storage().F64(), dv.Storage().F64(), l, kd, vd, gamma)
			case tensor.F32:
				retentionBackwardCore(qs, ks, vs, gs,
					dq.Storage().F32(), dk.Storage().F32(), dv.Storage().F32(), l, kd, vd, gamma)
			}
			return []*tensor.Tensor{dq, dk, dv}, nil
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
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

// retentionBackwardCore runs the retention backward over flat row-major slices.
// Reads come from exact-widened []float64; writes go to []T with every
// intermediate store narrowed for T=float32 (identical to the generic
// SetF64(AtF64+δ) chain). Loop structure and accumulation order match the
// generic loops exactly — bit-identical.
func retentionBackwardCore[T refFloat](qs, ks, vs, gs []float64, dqs, dks, dvs []T,
	l, kd, vd int, gamma float64) {
	pow := make([]float64, l)
	for t := range pow {
		pow[t] = math.Pow(gamma, float64(t))
	}
	for n := 0; n < l; n++ {
		qrow := qs[n*kd : n*kd+kd]
		grow := gs[n*vd : n*vd+vd]
		dqrow := dqs[n*kd : n*kd+kd]
		//perfscan:ignore PS3053 reference oracle: intentionally simple, correctness baseline not an optimization target
		for m := 0; m <= n; m++ {
			decay := pow[n-m]
			krow := ks[m*kd : m*kd+kd]
			var a float64
			//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
			for i, qv := range qrow {
				//perfscan:ignore PS3025 reference oracle: intentionally simple, correctness baseline not an optimization target
				a += qv * krow[i]
			}
			pnm := a * decay
			vrow := vs[m*vd : m*vd+vd]
			dvrow := dvs[m*vd : m*vd+vd]
			var dp float64
			//perfscan:ignore PS3010,PS3074 reference oracle: intentionally simple, correctness baseline not an optimization target
			for j, gnj := range grow { // value dim
				//perfscan:ignore PS3025 reference oracle: intentionally simple, correctness baseline not an optimization target
				dp += gnj * vrow[j]
				dvrow[j] = T(float64(dvrow[j]) + pnm*gnj)
			}
			dA := dp * decay
			dkrow := dks[m*kd : m*kd+kd]
			for i := range dqrow { // key dim
				dqrow[i] = T(float64(dqrow[i]) + dA*krow[i])
				dkrow[i] = T(float64(dkrow[i]) + dA*qrow[i])
			}
		}
	}
}

func init() {
	std.add(backend.OpRetentionBackward, tensor.F32, retentionBackwardKernel)
	std.add(backend.OpRetentionBackward, tensor.F64, retentionBackwardKernel)
}
