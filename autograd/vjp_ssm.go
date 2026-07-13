package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Mamba selective-scan VJP (Gu & Dao 2023, §R76). The forward carries a state
// h[t,d,n] = Ā·h[t-1] + Δ·B·u forward in time; the backward is the matching
// REVERSE-time scan. With Ā = exp(Δ·A) and dy the upstream gradient:
//
//	dh[t] = Cᵀ·dy[t] + Ā[t+1]·dh[t+1]        (reverse recurrence, dh[L]=0)
//	dC[t,n]  = Σ_d h[t,d,n]·dy[t,d]
//	du[t,d]  = D_skip[d]·dy[t,d] + Σ_n dh[t,d,n]·Δ[t,d]·B[t,n]
//	dB[t,n]  = Σ_d dh[t,d,n]·Δ[t,d]·u[t,d]
//	dA[d,n]  = Σ_t dh[t,d,n]·h[t-1,d,n]·Δ[t,d]·Ā[t,d,n]
//	dΔ[t,d]  = Σ_n dh[t,d,n]·( h[t-1,d,n]·A[d,n]·Ā[t,d,n] + B[t,n]·u[t,d] )
//	dD_skip[d] = Σ_t u[t,d]·dy[t,d]
//
// This is the exact gradient of the forward scan, so it passes a finite-difference
// check (§V2). D_skip is optional (5- or 6-input op).
func init() {
	RegisterVJP(backend.OpSSM, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		u, delta, A, B, C := in[0], in[1], in[2], in[3], in[4]
		hasSkip := len(in) == 6
		L, D := u.Shape()[0], u.Shape()[1]
		N := A.Shape()[1]
		idx := func(t, d, n int) int { return (t*D+d)*N + n }

		// recompute forward, storing Ā and h for every timestep
		abar := make([]float64, L*D*N)
		h := make([]float64, L*D*N)
		state := make([]float64, D*N)
		for t := range L {
			for d := range D {
				dt := delta.AtF64(t, d)
				ut := u.AtF64(t, d)
				for n := range N {
					ab := math.Exp(dt * A.AtF64(d, n))
					hv := ab*state[d*N+n] + dt*B.AtF64(t, n)*ut
					abar[idx(t, d, n)] = ab
					h[idx(t, d, n)] = hv
					state[d*N+n] = hv
				}
			}
		}

		du := tensor.New(u.Dtype(), u.Shape())
		ddelta := tensor.New(delta.Dtype(), delta.Shape())
		dA := tensor.New(A.Dtype(), A.Shape())
		dB := tensor.New(B.Dtype(), B.Shape())
		dC := tensor.New(C.Dtype(), C.Shape())
		var dskipT *tensor.Tensor
		if hasSkip {
			dskipT = tensor.New(in[5].Dtype(), in[5].Shape())
		}

		dhNext := make([]float64, D*N) // dh[t+1]; starts at dh[L]=0
		dht := make([]float64, D*N)
		for t := L - 1; t >= 0; t-- {
			for d := range D {
				gy := g.AtF64(t, d)
				dt := delta.AtF64(t, d)
				ut := u.AtF64(t, d)
				var duTD, ddeltaTD float64
				for n := range N {
					dh := C.AtF64(t, n) * gy
					if t < L-1 {
						dh += abar[idx(t+1, d, n)] * dhNext[d*N+n]
					}
					dht[d*N+n] = dh

					hPrev := 0.0
					if t > 0 {
						hPrev = h[idx(t-1, d, n)]
					}
					ab := abar[idx(t, d, n)]
					bt := B.AtF64(t, n)

					dC.SetF64(dC.AtF64(t, n)+h[idx(t, d, n)]*gy, t, n)
					dB.SetF64(dB.AtF64(t, n)+dh*dt*ut, t, n)
					dA.SetF64(dA.AtF64(d, n)+dh*hPrev*dt*ab, d, n)
					duTD += dh * dt * bt
					ddeltaTD += dh * (hPrev*A.AtF64(d, n)*ab + bt*ut)
				}
				if hasSkip {
					duTD += in[5].AtF64(d) * gy
					dskipT.SetF64(dskipT.AtF64(d)+ut*gy, d)
				}
				du.SetF64(duTD, t, d)
				ddelta.SetF64(ddeltaTD, t, d)
			}
			copy(dhNext, dht)
		}

		grads := []*tensor.Tensor{du, ddelta, dA, dB, dC}
		if hasSkip {
			grads = append(grads, dskipT)
		}
		return grads, nil
	})
}
