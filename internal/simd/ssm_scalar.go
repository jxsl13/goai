package simd

import "math"

// ssmScanDRangeScalar runs the full selective scan over the CHANNEL range [dLo,dHi) with
// d-outer/t-inner order — the channel-parallel form. Each (t,d) is independent of every
// other d (h[d·N+n] and out[t·D+d] are per-d disjoint), so this is byte-for-byte equal to
// the whole ssmScanScalar(...,0,N) despite the swapped loop order: the y reduction stays
// ascending-n and h[d·N+n] evolves in the same t sequence. The portable scan and the AVX
// range scan's non-AVX / N-tail fall back here.
func ssmScanDRangeScalar(u, delta, as, bs, cs, dsk, out, h []float64, L, D, N, dLo, dHi int) {
	for d := dLo; d < dHi; d++ {
		base := d * N
		for t := 0; t < L; t++ {
			dt := delta[t*D+d]
			ut := u[t*D+d]
			tn := t * N
			var y float64
			for n := 0; n < N; n++ {
				abar := math.Exp(dt * as[base+n])
				hv := abar*h[base+n] + dt*bs[tn+n]*ut
				h[base+n] = hv
				y += cs[tn+n] * hv
			}
			if dsk != nil {
				y += dsk[d] * ut
			}
			out[t*D+d] = y
		}
	}
}

// ssmScanScalar runs the Mamba/S6 selective-scan recurrence (Gu & Dao 2023) over the
// state-dim range [nLo,nHi) — the portable full scan (nLo=0) and the AVX version's
// len%4 tail. It mirrors the reference kernel (backend/ref/ssm.go) byte-for-byte: per
// (t,d) a decay abar = exp(Δ·A[d,n]) (A<0 ⟹ argument ≤ 0), a per-(d,n) state update
// h = abar·h + Δ·B·u, and the output reduction y = Σ_n C·h. The state h (D·N,
// row-major (d,n) at d·N+n) persists across t and stays float64.
//
// The state dim n is independent per lane (h[base+n] evolves only in t), so a scan can
// be split across n-ranges: the AVX path owns [0,nMain) and this owns [nMain,N),
// accumulating into the same out[t,d]. nLo==0 is the sole owner of the store (= y) and
// of the per-d D-skip; a tail pass (nLo>0) adds its partial (+= y) and must NOT re-add
// the skip — the AVX main pass already did.
//
//	u,delta [L,D]; as (A) [D,N]; bs (B), cs (C) [L,N]; dsk (D-skip) [D] or nil; out [L,D].
func ssmScanScalar(u, delta, as, bs, cs, dsk, out, h []float64, L, D, N, nLo, nHi int) {
	for t := 0; t < L; t++ {
		for d := 0; d < D; d++ {
			dt := delta[t*D+d]
			ut := u[t*D+d]
			base := d * N
			tn := t * N
			var y float64
			// FOUR bounds checks in this loop — as, h, bs and cs, each at a computed offset.
			// Cutting all four to the n-range and ranging the first removes every one; the
			// others are re-cut to that length so the compiler can relate them. Bit-identical:
			// the same operands in the same ascending-n order, and hrow aliases h so the
			// read-modify-write is unchanged.
			arow := as[base+nLo : base+nHi]
			hrow := h[base+nLo : base+nHi]
			brow := bs[tn+nLo : tn+nHi]
			crow := cs[tn+nLo : tn+nHi]
			hrow, brow, crow = hrow[:len(arow)], brow[:len(arow)], crow[:len(arow)]
			for i, av := range arow {
				abar := math.Exp(dt * av)
				hv := abar*hrow[i] + dt*brow[i]*ut
				hrow[i] = hv
				y += crow[i] * hv
			}
			if nLo == 0 {
				if dsk != nil {
					y += dsk[d] * ut
				}
				out[t*D+d] = y
			} else {
				out[t*D+d] += y
			}
		}
	}
}
