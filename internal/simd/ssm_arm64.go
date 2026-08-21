//go:build arm64 && goexperiment.simd

package simd

import "math"

//go:noescape
func ssmChannelNegNeonF64(as, bs, cs, h, u, delta, out *float64, dskip float64, L, D, N, pairs int)

// ssmNeonRangeSafe proves the complete channel range before the first state or
// output write. For each channel, max(delta)*min(A) is the most-negative decay
// product because delta is finite nonnegative and A is finite nonpositive.
func ssmNeonRangeSafe(delta, as []float64, L, D, N, dLo, dHi int) bool {
	delta = delta[:L*D]
	as = as[:D*N]
	for d := dLo; d < dHi; d++ {
		maxDelta := 0.0
		for t := 0; t < L; t++ {
			dt := delta[t*D+d]
			if dt < 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
				return false
			}
			maxDelta = max(maxDelta, dt)
		}
		minA := 0.0
		base := d * N
		for n := 0; n < N; n++ {
			a := as[base+n]
			if a > 0 || math.IsNaN(a) || math.IsInf(a, 0) {
				return false
			}
			minA = min(minA, a)
		}
		if maxDelta*minA < expF64NeonLo {
			return false
		}
	}
	return true
}

func ssmScanRangeNeonF64Unchecked(u, delta, as, bs, cs, dsk, out, h []float64, L, D, N, dLo, dHi int) {
	pairs := N >> 1
	for d := dLo; d < dHi; d++ {
		dskip := 0.0
		if dsk != nil {
			dskip = dsk[d]
		}
		base := d * N
		ssmChannelNegNeonF64(
			&as[base], &bs[0], &cs[0], &h[base],
			&u[d], &delta[d], &out[d], dskip, L, D, N, pairs,
		)
		if N&1 == 0 {
			continue
		}
		n := N - 1
		for t := 0; t < L; t++ {
			dt := delta[t*D+d]
			ut := u[t*D+d]
			tn := t*N + n
			hn := base + n
			abar := math.Exp(dt * as[hn])
			hv := abar*h[hn] + dt*bs[tn]*ut
			h[hn] = hv
			out[t*D+d] += cs[tn] * hv
		}
	}
}

// SSMScanF64 runs the arm64 two-lane fused negative-exp/state/reduction leaf.
// Unsupported domains select the byte-for-byte scalar implementation before
// either recurrent state or output is mutated.
func SSMScanF64(u, delta, as, bs, cs, dsk, out, h []float64, L, D, N int) {
	if L == 0 || D == 0 || N < 2 || !ssmNeonRangeSafe(delta, as, L, D, N, 0, D) {
		ssmScanScalar(u, delta, as, bs, cs, dsk, out, h, L, D, N, 0, N)
		return
	}
	ssmScanRangeNeonF64Unchecked(u, delta, as, bs, cs, dsk, out, h, L, D, N, 0, D)
}

// SSMScanRangeF64 is the channel-parallel entry point. Each range proves its
// own disjoint delta/A domain before handing a channel to the fused leaf.
func SSMScanRangeF64(u, delta, as, bs, cs, dsk, out, h []float64, L, D, N, dLo, dHi int) {
	if L == 0 || dLo == dHi || N < 2 || !ssmNeonRangeSafe(delta, as, L, D, N, dLo, dHi) {
		ssmScanDRangeScalar(u, delta, as, bs, cs, dsk, out, h, L, D, N, dLo, dHi)
		return
	}
	ssmScanRangeNeonF64Unchecked(u, delta, as, bs, cs, dsk, out, h, L, D, N, dLo, dHi)
}
