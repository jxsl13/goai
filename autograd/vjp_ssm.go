package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
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
//
// The sequential scan carries hidden state across time, so the time-iteration
// order, the state[d*N+n] buffer indexing, and every += accumulation are held
// exactly as written. Only the element-access mechanism is devirtualised: when
// every tensor shares a float dtype the loops read/write raw row-major slices
// (element (t,d,n) of a [T,D,N] tensor at (t*D+d)*N+n); mixed/exotic dtypes fall
// through to the generic AtF64/SetF64 loop.
func init() {
	RegisterVJP(backend.OpSSM, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		u, delta, A, B, C := in[0], in[1], in[2], in[3], in[4]
		hasSkip := len(in) == 6
		L, D := u.Shape()[0], u.Shape()[1]
		N := A.Shape()[1]
		idx := func(t, d, n int) int { return (t*D+d)*N + n }

		du := tensor.New(u.Dtype(), u.Shape())
		ddelta := tensor.New(delta.Dtype(), delta.Shape())
		dA := tensor.New(A.Dtype(), A.Shape())
		dB := tensor.New(B.Dtype(), B.Shape())
		dC := tensor.New(C.Dtype(), C.Shape())
		var dskipT *tensor.Tensor
		if hasSkip {
			dskipT = tensor.New(in[5].Dtype(), in[5].Shape())
		}

		grads := func() []*tensor.Tensor {
			gr := []*tensor.Tensor{du, ddelta, dA, dB, dC}
			if hasSkip {
				gr = append(gr, dskipT)
			}
			return gr
		}

		// Fast paths require every read tensor (and D_skip, if present) to share
		// one float dtype; the freshly-allocated grad tensors then share it too.
		dt := u.Dtype()
		same := dt == delta.Dtype() && dt == A.Dtype() && dt == B.Dtype() &&
			dt == C.Dtype() && dt == g.Dtype()
		if hasSkip {
			same = same && dt == in[5].Dtype()
		}

		if same {
			switch dt {
			case tensor.F64:
				us := u.Contiguous().Storage().F64()
				ds := delta.Contiguous().Storage().F64()
				as := A.Contiguous().Storage().F64()
				bs := B.Contiguous().Storage().F64()
				cs := C.Contiguous().Storage().F64()
				gs := g.Contiguous().Storage().F64()
				dus, ddeltas := du.Storage().F64(), ddelta.Storage().F64()
				dAs, dBs, dCs := dA.Storage().F64(), dB.Storage().F64(), dC.Storage().F64()
				var skips, dskips []float64
				if hasSkip {
					skips = in[5].Contiguous().Storage().F64()
					dskips = dskipT.Storage().F64()
				}

				// recompute forward, storing Ā and h for every timestep
				abar := make([]float64, L*D*N)
				h := make([]float64, L*D*N)
				state := make([]float64, D*N)
				abrow := make([]float64, N) // Ā over the state dim, one 4-wide exp per (t,d)
				for t := range L {
					for d := range D {
						dtv := ds[t*D+d]
						ut := us[t*D+d]
						base := d * N
						simd.ExpScaledF64(abrow, as[base:base+N], dtv) // abrow[n]=exp(Δ·A[d,n]), A<0⇒arg≤0
						for n := range N {
							ab := abrow[n]
							hv := ab*state[base+n] + dtv*bs[t*N+n]*ut
							abar[idx(t, d, n)] = ab
							h[idx(t, d, n)] = hv
							state[base+n] = hv
						}
					}
				}

				dhNext := make([]float64, D*N) // dh[t+1]; starts at dh[L]=0
				dht := make([]float64, D*N)
				for t := L - 1; t >= 0; t-- {
					for d := range D {
						gy := gs[t*D+d]
						dtv := ds[t*D+d]
						ut := us[t*D+d]
						var duTD, ddeltaTD float64
						for n := range N {
							dh := cs[t*N+n] * gy
							if t < L-1 {
								dh += abar[idx(t+1, d, n)] * dhNext[d*N+n]
							}
							dht[d*N+n] = dh

							hPrev := 0.0
							if t > 0 {
								hPrev = h[idx(t-1, d, n)]
							}
							ab := abar[idx(t, d, n)]
							bt := bs[t*N+n]

							dCs[t*N+n] += h[idx(t, d, n)] * gy
							dBs[t*N+n] += dh * dtv * ut
							dAs[d*N+n] += dh * hPrev * dtv * ab
							duTD += dh * dtv * bt
							ddeltaTD += dh * (hPrev*as[d*N+n]*ab + bt*ut)
						}
						if hasSkip {
							duTD += skips[d] * gy
							dskips[d] += ut * gy
						}
						dus[t*D+d] = duTD
						ddeltas[t*D+d] = ddeltaTD
					}
					copy(dhNext, dht)
				}
				return grads(), nil

			case tensor.F32:
				us := u.Contiguous().Storage().F32()
				ds := delta.Contiguous().Storage().F32()
				as := A.Contiguous().Storage().F32()
				bs := B.Contiguous().Storage().F32()
				cs := C.Contiguous().Storage().F32()
				gs := g.Contiguous().Storage().F32()
				dus, ddeltas := du.Storage().F32(), ddelta.Storage().F32()
				dAs, dBs, dCs := dA.Storage().F32(), dB.Storage().F32(), dC.Storage().F32()
				var skips, dskips []float32
				if hasSkip {
					skips = in[5].Contiguous().Storage().F32()
					dskips = dskipT.Storage().F32()
				}

				// recompute forward, storing Ā and h for every timestep (state and
				// the saved buffers stay float64, exactly as the generic path — which
				// reads inputs through AtF64 — computes them)
				abar := make([]float64, L*D*N)
				h := make([]float64, L*D*N)
				state := make([]float64, D*N)
				abrow := make([]float64, N) // Ā over the state dim
				asrow := make([]float64, N) // the F32 A row widened to f64 for the vector exp
				for t := range L {
					for d := range D {
						dtv := float64(ds[t*D+d])
						ut := float64(us[t*D+d])
						base := d * N
						for n := range N {
							asrow[n] = float64(as[base+n])
						}
						simd.ExpScaledF64(abrow, asrow, dtv)
						for n := range N {
							ab := abrow[n]
							hv := ab*state[base+n] + dtv*float64(bs[t*N+n])*ut
							abar[idx(t, d, n)] = ab
							h[idx(t, d, n)] = hv
							state[base+n] = hv
						}
					}
				}

				dhNext := make([]float64, D*N) // dh[t+1]; starts at dh[L]=0
				dht := make([]float64, D*N)
				for t := L - 1; t >= 0; t-- {
					for d := range D {
						gy := float64(gs[t*D+d])
						dtv := float64(ds[t*D+d])
						ut := float64(us[t*D+d])
						var duTD, ddeltaTD float64
						for n := range N {
							dh := float64(cs[t*N+n]) * gy
							if t < L-1 {
								dh += abar[idx(t+1, d, n)] * dhNext[d*N+n]
							}
							dht[d*N+n] = dh

							hPrev := 0.0
							if t > 0 {
								hPrev = h[idx(t-1, d, n)]
							}
							ab := abar[idx(t, d, n)]
							bt := float64(bs[t*N+n])

							// per-accumulation rounding to float32 matches the generic
							// AtF64+SetF64 path this fast path replaces
							dCs[t*N+n] = float32(float64(dCs[t*N+n]) + h[idx(t, d, n)]*gy)
							dBs[t*N+n] = float32(float64(dBs[t*N+n]) + dh*dtv*ut)
							dAs[d*N+n] = float32(float64(dAs[d*N+n]) + dh*hPrev*dtv*ab)
							duTD += dh * dtv * bt
							ddeltaTD += dh * (hPrev*float64(as[d*N+n])*ab + bt*ut)
						}
						if hasSkip {
							duTD += float64(skips[d]) * gy
							dskips[d] = float32(float64(dskips[d]) + ut*gy)
						}
						dus[t*D+d] = float32(duTD)
						ddeltas[t*D+d] = float32(ddeltaTD)
					}
					copy(dhNext, dht)
				}
				return grads(), nil
			}
		}

		// generic fallback (mixed or exotic dtypes) — the original AtF64/SetF64 loop
		abar := make([]float64, L*D*N)
		h := make([]float64, L*D*N)
		state := make([]float64, D*N)
		for t := range L {
			for d := range D {
				dtv := delta.AtF64(t, d)
				ut := u.AtF64(t, d)
				for n := range N {
					ab := math.Exp(dtv * A.AtF64(d, n))
					hv := ab*state[d*N+n] + dtv*B.AtF64(t, n)*ut
					abar[idx(t, d, n)] = ab
					h[idx(t, d, n)] = hv
					state[d*N+n] = hv
				}
			}
		}

		dhNext := make([]float64, D*N) // dh[t+1]; starts at dh[L]=0
		dht := make([]float64, D*N)
		for t := L - 1; t >= 0; t-- {
			for d := range D {
				gy := g.AtF64(t, d)
				dtv := delta.AtF64(t, d)
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
					dB.SetF64(dB.AtF64(t, n)+dh*dtv*ut, t, n)
					dA.SetF64(dA.AtF64(d, n)+dh*hPrev*dtv*ab, d, n)
					duTD += dh * dtv * bt
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
		return grads(), nil
	})
}
