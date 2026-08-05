package autograd

import (
	"math"
	"runtime"
	"sync"

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
// The sequential scan carries hidden state across time, so within each channel the
// time-iteration order and every += accumulation are held exactly as written. The
// D channels are INDEPENDENT recurrences, though, so the shared-float-dtype fast
// path (ssmVJPScan) fans them out across the worker pool — the same axis the forward
// prefill parallelises (nn SSM #928/#929) — recomputing Ā,h and the reverse scan per
// channel; only dB/dC sum across channels and are reduced from per-worker partials.
// Mixed/exotic dtypes fall through to the generic serial AtF64/SetF64 loop.
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
				var skips, dskips []float64
				if hasSkip {
					skips = in[5].Contiguous().Storage().F64()
					dskips = dskipT.Storage().F64()
				}
				ssmVJPScan(
					u.Contiguous().Storage().F64(), delta.Contiguous().Storage().F64(),
					A.Contiguous().Storage().F64(), B.Contiguous().Storage().F64(),
					C.Contiguous().Storage().F64(), g.Contiguous().Storage().F64(),
					du.Storage().F64(), ddelta.Storage().F64(), dA.Storage().F64(),
					dB.Storage().F64(), dC.Storage().F64(),
					skips, dskips, hasSkip, L, D, N)
				return grads(), nil
			case tensor.F32:
				var skips, dskips []float32
				if hasSkip {
					skips = in[5].Contiguous().Storage().F32()
					dskips = dskipT.Storage().F32()
				}
				ssmVJPScan(
					u.Contiguous().Storage().F32(), delta.Contiguous().Storage().F32(),
					A.Contiguous().Storage().F32(), B.Contiguous().Storage().F32(),
					C.Contiguous().Storage().F32(), g.Contiguous().Storage().F32(),
					du.Storage().F32(), ddelta.Storage().F32(), dA.Storage().F32(),
					dB.Storage().F32(), dC.Storage().F32(),
					skips, dskips, hasSkip, L, D, N)
				return grads(), nil
			}
		}

		// generic fallback (mixed or exotic dtypes) — the original AtF64/SetF64 loop
		//perfscan:ignore PS3028 generic fallback scratch alloc, exotic-dtype path
		abar := make([]float64, L*D*N)
		//perfscan:ignore PS3028 generic fallback h alloc, exotic-dtype path
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
			//perfscan:ignore PS3040,PS3067 generic exotic-dtype fallback loop, fast paths handle F64/F32 | exotic-dtype fallback d-loop, declined-dtype b
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

// ssmVJPScan is the parallel selective-scan backward core shared by the F64 and F32
// fast paths. Each channel d ∈ [0,D) is an INDEPENDENT scalar recurrence (its own
// state, Ā, and reverse-scan dh), so the D channels fan out across the worker pool —
// the same axis the forward prefill parallelises. Per channel it recomputes the
// forward (Ā,h in f64 scratch) then runs the reverse-time scan, writing du/dΔ/dA/
// dD_skip directly (disjoint rows, no contention). Only dB[t,n]/dC[t,n] sum over ALL
// channels, so each worker accumulates them into a PRIVATE f64 band partial that is
// reduced in band order afterwards.
//
// That reduction re-associates the per-channel dB/dC sums (grouped by band rather
// than strict ascending d); the scan arithmetic is otherwise identical, so the
// gradient matches the serial one within the same rounding the finite-difference
// gradcheck (§V2) and the F32↔F64 parity test already allow — there is NO byte-exact
// contract on the backward. All accumulation is f64 (F32 rounds only on the final
// store), matching the generic AtF64 fallback's precision.
func ssmVJPScan[T float32 | float64](
	us, ds, as, bs, cs, gs []T,
	dus, ddeltas, dAs, dBs, dCs []T,
	skips, dskips []T, hasSkip bool,
	L, D, N int,
) {
	nw := runtime.GOMAXPROCS(0)
	if nw > D {
		nw = D
	}
	if nw < 1 || D*L*N < 1<<15 {
		nw = 1
	}
	chunk := (D + nw - 1) / nw
	nBands := (D + chunk - 1) / chunk
	//perfscan:ignore PS3028 per-worker dB/dC reverse-scan partials, resource-only
	dBpart := make([]float64, nBands*L*N)
	//perfscan:ignore PS3028 per-worker dB/dC reverse-scan partials, resource-only
	dCpart := make([]float64, nBands*L*N)

	var wg sync.WaitGroup
	for band := 0; band < nBands; band++ {
		dlo := band * chunk
		dhi := dlo + chunk
		if dhi > D {
			dhi = D
		}
		wg.Add(1)
		go func(band, dlo, dhi int) {
			defer wg.Done()
			dBp := dBpart[band*L*N : (band+1)*L*N]
			dCp := dCpart[band*L*N : (band+1)*L*N]
			abar := make([]float64, L*N)
			h := make([]float64, L*N)
			state := make([]float64, N)
			abrow := make([]float64, N)
			asrow := make([]float64, N)
			dhNext := make([]float64, N)
			dAloc := make([]float64, N)
			for d := dlo; d < dhi; d++ {
				base := d * N
				// recompute the forward for this channel (Ā, h saved per timestep)
				for n := 0; n < N; n++ {
					state[n] = 0
					asrow[n] = float64(as[base+n])
				}
				for t := 0; t < L; t++ {
					dtv := float64(ds[t*D+d])
					ut := float64(us[t*D+d])
					simd.ExpScaledF64(abrow, asrow, dtv) // Ā[n]=exp(Δ·A[d,n]), A<0⇒arg≤0
					row := t * N
					for n := 0; n < N; n++ {
						ab := abrow[n]
						hv := ab*state[n] + dtv*float64(bs[row+n])*ut
						abar[row+n] = ab
						h[row+n] = hv
						state[n] = hv
					}
				}
				// reverse-time scan; dhNext[n] carries dh[t+1] (overwritten in place
				// since dh[t,n] reads only dhNext[n] of the same n).
				for n := 0; n < N; n++ {
					dhNext[n] = 0
					dAloc[n] = 0
				}
				var dskipAcc float64
				for t := L - 1; t >= 0; t-- {
					gy := float64(gs[t*D+d])
					dtv := float64(ds[t*D+d])
					ut := float64(us[t*D+d])
					row := t * N
					var duTD, ddeltaTD float64
					for n := 0; n < N; n++ {
						dh := float64(cs[row+n]) * gy
						if t < L-1 {
							dh += abar[row+N+n] * dhNext[n]
						}
						hPrev := 0.0
						if t > 0 {
							hPrev = h[row-N+n]
						}
						ab := abar[row+n]
						bt := float64(bs[row+n])
						dCp[row+n] += h[row+n] * gy
						dBp[row+n] += dh * dtv * ut
						dAloc[n] += dh * hPrev * dtv * ab
						duTD += dh * dtv * bt
						ddeltaTD += dh * (hPrev*asrow[n]*ab + bt*ut)
						dhNext[n] = dh
					}
					if hasSkip {
						duTD += float64(skips[d]) * gy
						dskipAcc += ut * gy
					}
					dus[t*D+d] = T(duTD)
					ddeltas[t*D+d] = T(ddeltaTD)
				}
				for n := 0; n < N; n++ {
					dAs[base+n] = T(dAloc[n])
				}
				if hasSkip {
					dskips[d] = T(dskipAcc)
				}
			}
		}(band, dlo, dhi)
	}
	wg.Wait()

	// Reduce the per-band dB/dC partials in band order (deterministic given nw).
	for band := 0; band < nBands; band++ {
		dBp := dBpart[band*L*N : (band+1)*L*N]
		dCp := dCpart[band*L*N : (band+1)*L*N]
		for i := 0; i < L*N; i++ {
			dBs[i] = T(float64(dBs[i]) + dBp[i])
			dCs[i] = T(float64(dCs[i]) + dCp[i])
		}
	}
}
