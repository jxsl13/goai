package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/parallel"
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
	h := make([]float64, D*N) // recurrent state per (d,n), persists across t (f64 on ALL paths)

	// Devirtualised fast paths (§T645): the generic AtF64/SetF64 loop below pays a
	// dtype dispatch + flat-offset computation per element. When every read tensor
	// shares the output dtype we instead grab the raw typed slices once (row-major:
	// (t,d) of [L,D] at t*D+d, (d,n) of [D,N] at d*N+n, (t,n) of [L,N] at t*N+n) and
	// index directly. Iteration order, state buffer indexing and accumulation are
	// byte-for-byte identical to the generic path; the scan state stays float64
	// everywhere and the F32 path only rounds the STORED output. Contiguous() is
	// called once per tensor (returns self when already contiguous).
	switch out.Dtype() {
	case tensor.F64:
		uc, dc := u.Contiguous(), delta.Contiguous()
		ac, bc, cc := A.Contiguous(), B.Contiguous(), C.Contiguous()
		if uc.Dtype() == tensor.F64 && dc.Dtype() == tensor.F64 && ac.Dtype() == tensor.F64 &&
			bc.Dtype() == tensor.F64 && cc.Dtype() == tensor.F64 &&
			(dskip == nil || dskip.Dtype() == tensor.F64) {
			us, ds := uc.Storage().F64(), dc.Storage().F64()
			as, bs, cs := ac.Storage().F64(), bc.Storage().F64(), cc.Storage().F64()
			os := out.Storage().F64()
			var dsk []float64
			if dskip != nil {
				dsk = dskip.Contiguous().Storage().F64()
			}
			// Interchange to channel-outer/time-inner: each channel d is an independent
			// sequential scan over t with its own N-state, so the outer loop parallelizes
			// over D. Per-worker hloc[N] reproduces the shared h[d*N:] evolution in the same
			// t-order and arithmetic — byte-identical, disjoint output columns.
			parallel.Rows(D, func(dlo, dhi int) {
				hloc := make([]float64, N)
				for d := dlo; d < dhi; d++ {
					base := d * N
					for n := range N {
						hloc[n] = 0
					}
					for t := range L {
						dt := ds[t*D+d]
						ut := us[t*D+d]
						tn := t * N
						var y float64
						for n := range N {
							abar := math.Exp(dt * as[base+n])
							hv := abar*hloc[n] + dt*bs[tn+n]*ut
							hloc[n] = hv
							y += cs[tn+n] * hv
						}
						if dskip != nil {
							y += dsk[d] * ut
						}
						os[t*D+d] = y
					}
				}
			})
			return []*tensor.Tensor{out}, nil
		}
	case tensor.F32:
		uc, dc := u.Contiguous(), delta.Contiguous()
		ac, bc, cc := A.Contiguous(), B.Contiguous(), C.Contiguous()
		if uc.Dtype() == tensor.F32 && dc.Dtype() == tensor.F32 && ac.Dtype() == tensor.F32 &&
			bc.Dtype() == tensor.F32 && cc.Dtype() == tensor.F32 &&
			(dskip == nil || dskip.Dtype() == tensor.F32) {
			us, ds := uc.Storage().F32(), dc.Storage().F32()
			as, bs, cs := ac.Storage().F32(), bc.Storage().F32(), cc.Storage().F32()
			os := out.Storage().F32()
			var dsk []float32
			if dskip != nil {
				dsk = dskip.Contiguous().Storage().F32()
			}
			parallel.Rows(D, func(dlo, dhi int) {
				hloc := make([]float64, N)
				for d := dlo; d < dhi; d++ {
					base := d * N
					for n := range N {
						hloc[n] = 0
					}
					for t := range L {
						dt := float64(ds[t*D+d])
						ut := float64(us[t*D+d])
						tn := t * N
						var y float64 // scan state accumulates in float64; only the store rounds
						for n := range N {
							abar := math.Exp(dt * float64(as[base+n]))
							hv := abar*hloc[n] + dt*float64(bs[tn+n])*ut
							hloc[n] = hv
							y += float64(cs[tn+n]) * hv
						}
						if dskip != nil {
							y += float64(dsk[d]) * ut
						}
						os[t*D+d] = float32(y)
					}
				}
			})
			return []*tensor.Tensor{out}, nil
		}
	}

	// Generic fallback for exotic dtypes / mixed inputs (verbatim original loop).
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
