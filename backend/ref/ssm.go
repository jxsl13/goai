package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// selectiveScanKernel is the fused Mamba selective-scan (S6) recurrence (Gu & Dao
// 2023, arXiv:2312.00752 §3.2 Alg. 2, §R76), matching the official mamba_ssm
// selective_scan_ref. It is a linear-time (O(L·D·N)) input-dependent state-space
// model: instead of attention it carries a per-channel latent state h forward in
// time, with the step size Δ, input matrix B and output matrix C all functions of
// the input (that "selectivity" is what lets it route information like attention).
//
// Inputs: u[L,D] (sequence), delta[L,D] (>0 discretization step), A[D,N] (state
// matrix, negative for stability), B[L,N], C[L,N] (input-dependent, shared across
// the D channels), and an OPTIONAL skip D_skip[D]. With the zero-order-hold
// discretization Ā = exp(Δ⊙A) and the simplified B̄ = Δ⊙B (as in the reference):
//
//	h[t,d,n] = Ā[t,d,n]·h[t-1,d,n] + Δ[t,d]·B[t,n]·u[t,d]     (h[-1]=0)
//	y[t,d]   = Σ_n C[t,n]·h[t,d,n] + D_skip[d]·u[t,d]
//
// Output y[L,D]. f64 accumulation (§V10).
func selectiveScanKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) < 5 || len(in) > 6 {
		return nil, fmt.Errorf("ref: ssm wants (u,delta,A,B,C[,D_skip]), got %d inputs", len(in))
	}
	u, delta, A, B, C := in[0], in[1], in[2], in[3], in[4]
	var dskip *tensor.Tensor
	if len(in) == 6 {
		dskip = in[5]
	}
	if u.Ndim() != 2 || delta.Ndim() != 2 || A.Ndim() != 2 || B.Ndim() != 2 || C.Ndim() != 2 {
		return nil, fmt.Errorf("ref: ssm needs rank-2 inputs")
	}
	L, D := u.Shape()[0], u.Shape()[1]
	N := A.Shape()[1]
	if !delta.Shape().Equal(u.Shape()) {
		return nil, fmt.Errorf("ref: ssm delta %v != u %v", delta.Shape(), u.Shape())
	}
	if A.Shape()[0] != D {
		return nil, fmt.Errorf("ref: ssm A rows %d != D %d", A.Shape()[0], D)
	}
	if B.Shape()[0] != L || B.Shape()[1] != N || !C.Shape().Equal(B.Shape()) {
		return nil, fmt.Errorf("ref: ssm B/C must be [%d,%d], got %v/%v", L, N, B.Shape(), C.Shape())
	}
	if dskip != nil && (dskip.Ndim() != 1 || dskip.Shape()[0] != D) {
		return nil, fmt.Errorf("ref: ssm D_skip must be [%d], got %v", D, dskip.Shape())
	}

	out := tensor.NewOn(ctx.Device(), u.Dtype(), u.Shape())
	h := make([]float64, D*N) // recurrent state per (d,n), persists across t
	for t := range L {
		for d := range D {
			dt := delta.AtF64(t, d)
			ut := u.AtF64(t, d)
			var y float64
			for n := range N {
				abar := math.Exp(dt * A.AtF64(d, n))
				hv := abar*h[d*N+n] + dt*B.AtF64(t, n)*ut
				h[d*N+n] = hv
				y += C.AtF64(t, n) * hv
			}
			if dskip != nil {
				y += dskip.AtF64(d) * ut
			}
			out.SetF64(y, t, d)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpSSM, tensor.F32, selectiveScanKernel)
	std.add(backend.OpSSM, tensor.F64, selectiveScanKernel)
}
