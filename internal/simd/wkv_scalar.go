package simd

import "math"

// wkvScanScalar runs the RWKV-4 WKV recurrence for channels [cLo,cHi) — the
// numerically-stable log-space scan (Peng et al. 2023). It mirrors the reference
// kernel (backend/ref/wkv.go) byte-for-byte: per channel, a running numerator aa /
// denominator bb / max-exponent pp, with max-tracking exp rescaling. Shared by the
// scalar WKVScanF64 and the AVX version's len%4 channel tail. It is the fresh-state
// case of wkvScanStateScalar (aa/bb/pp start at 0/0/-1e38).
func wkvScanScalar(k, v, w, u, out []float64, seq, d, cLo, cHi int) {
	wkvScanStateScalar(k, v, w, u, out, nil, nil, nil, seq, d, cLo, cHi)
}

// wkvScanStateScalar is wkvScanScalar generalized to a CONTINUING scan: when the
// aa0/bb0/pp0 state slices are non-nil, each channel's recurrence starts from the
// carried (aa0[c], bb0[c], pp0[c]) instead of the fresh (0, 0, -1e38), and the final
// per-channel state is written back to those slices so the caller can resume with the
// next token chunk. Passing nil for all three reproduces wkvScanScalar exactly. This
// is what the RWKV decode/prefill step needs (it absorbs tokens against persistent
// AA/BB/PP), and chunking a sequence through it is bit-identical to one whole-sequence
// scan — the StoreSlice/Load round-trip carries the f64 state losslessly.
func wkvScanStateScalar(k, v, w, u, out, aa0, bb0, pp0 []float64, seq, d, cLo, cHi int) {
	for c := cLo; c < cHi; c++ {
		wc, uc := w[c], u[c]
		aa, bb, pp := 0.0, 0.0, -1e38
		if aa0 != nil {
			aa, bb, pp = aa0[c], bb0[c], pp0[c]
		}
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
		if aa0 != nil {
			aa0[c], bb0[c], pp0[c] = aa, bb, pp
		}
	}
}
