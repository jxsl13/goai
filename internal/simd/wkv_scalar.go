package simd

import "math"

// wkvScanScalar runs the RWKV-4 WKV recurrence for channels [cLo,cHi) — the
// numerically-stable log-space scan (Peng et al. 2023). It mirrors the reference
// kernel (backend/ref/wkv.go) byte-for-byte: per channel, a running numerator aa /
// denominator bb / max-exponent pp, with max-tracking exp rescaling. Shared by the
// scalar WKVScanF64 and the AVX version's len%4 channel tail.
func wkvScanScalar(k, v, w, u, out []float64, seq, d, cLo, cHi int) {
	for c := cLo; c < cHi; c++ {
		wc, uc := w[c], u[c]
		aa, bb, pp := 0.0, 0.0, -1e38
		for t := 0; t < seq; t++ {
			base := t*d + c
			kk, vv := k[base], v[base]
			ww := uc + kk
			q := math.Max(pp, ww)
			e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
			out[base] = (e1*aa + e2*vv) / (e1*bb + e2)
			q = math.Max(pp-wc, kk)
			e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
			aa = e1*aa + e2*vv
			bb = e1*bb + e2
			pp = q
		}
	}
}
