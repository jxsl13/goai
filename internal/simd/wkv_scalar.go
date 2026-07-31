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
			// HALF THESE EXPS ARE PROVABLY ONE. q is the max of its two arguments, so
			// whichever argument it equals gives exp(x-x) = exp(0), which is exactly 1 — no
			// call needed. Testing q against each argument rather than branching on the
			// comparison keeps NaN behavior identical: with a NaN operand math.Max yields NaN,
			// both equality tests fail, and both exps are evaluated exactly as before.
			// Bit-identical, and math.Exp(0) is exactly 1 so the surviving arithmetic is
			// unchanged. exp was 75.6%% of this kernel's profile.
			q := math.Max(pp, ww)
			e1, e2 := 1.0, 1.0
			if q != pp {
				e1 = math.Exp(pp - q)
			}
			if q != ww {
				e2 = math.Exp(ww - q)
			}
			out[base] = (e1*aa + e2*vv) / (e1*bb + e2)
			ppw := pp - wc
			q = math.Max(ppw, kk)
			e1, e2 = 1.0, 1.0
			if q != ppw {
				e1 = math.Exp(ppw - q)
			}
			if q != kk {
				e2 = math.Exp(kk - q)
			}
			aa = e1*aa + e2*vv
			bb = e1*bb + e2
			pp = q
		}
		if aa0 != nil {
			aa0[c], bb0[c], pp0[c] = aa, bb, pp
		}
	}
}
