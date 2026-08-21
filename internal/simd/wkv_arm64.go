//go:build arm64 && goexperiment.simd

package simd

import (
	"math"

	"github.com/jxsl13/goai/internal/fmath"
)

//go:noescape
func wkvPairNegNeonF64(k, v, w, u, out, aa, bb, pp *float64, seq, d int)

func wkvNeonFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// wkvNeonPairSafe proves the PP/max evolution for one adjacent channel pair.
// AA/BB do not influence either exponent argument, so this pass needs no exp
// evaluation and leaves output plus persistent state untouched on rejection.
func wkvNeonPairSafe(k, v, w, u, aa0, bb0, pp0 []float64, seq, d, cLo int) bool {
	for c := cLo; c < cLo+2; c++ {
		wc, uc := w[c], u[c]
		aa, bb, pp := 0.0, 0.0, -1e38
		if aa0 != nil {
			aa, bb, pp = aa0[c], bb0[c], pp0[c]
		}
		if !wkvNeonFinite(wc) || !wkvNeonFinite(uc) ||
			!wkvNeonFinite(aa) || !wkvNeonFinite(bb) || !wkvNeonFinite(pp) {
			return false
		}
		for t := 0; t < seq; t++ {
			base := t*d + c
			kk, vv := k[base], v[base]
			if !wkvNeonFinite(kk) || !wkvNeonFinite(vv) {
				return false
			}
			ww := uc + kk
			if !wkvNeonFinite(ww) {
				return false
			}
			q := fmath.Max(pp, ww)
			x := min(pp-q, ww-q)
			if !wkvNeonFinite(x) || x > 0 {
				return false
			}
			ppw := pp - wc
			if !wkvNeonFinite(ppw) {
				return false
			}
			q = fmath.Max(ppw, kk)
			x = min(ppw-q, kk-q)
			if !wkvNeonFinite(x) || x > 0 {
				return false
			}
			pp = q
		}
	}
	return true
}

func wkvScanStateRangeNeonF64(k, v, w, u, out, aa0, bb0, pp0 []float64, seq, d, cLo, cHi int) {
	if seq == 0 || cLo == cHi {
		return
	}
	c := cLo
	if c&1 != 0 {
		wkvScanStateScalar(k, v, w, u, out, aa0, bb0, pp0, seq, d, c, c+1)
		c++
	}
	for ; c+2 <= cHi; c += 2 {
		if !wkvNeonPairSafe(k, v, w, u, aa0, bb0, pp0, seq, d, c) {
			wkvScanStateScalar(k, v, w, u, out, aa0, bb0, pp0, seq, d, c, c+2)
			continue
		}
		if aa0 != nil {
			wkvPairNegNeonF64(
				&k[c], &v[c], &w[c], &u[c], &out[c],
				&aa0[c], &bb0[c], &pp0[c], seq, d,
			)
			continue
		}
		aa := [2]float64{}
		bb := [2]float64{}
		pp := [2]float64{-1e38, -1e38}
		wkvPairNegNeonF64(
			&k[c], &v[c], &w[c], &u[c], &out[c],
			&aa[0], &bb[0], &pp[0], seq, d,
		)
	}
	if c < cHi {
		wkvScanStateScalar(k, v, w, u, out, aa0, bb0, pp0, seq, d, c, cHi)
	}
}

// WKVScanF64 runs the fresh-state WKV recurrence over adjacent channel pairs.
func WKVScanF64(k, v, w, u, out []float64, seq, d int) {
	wkvScanStateRangeNeonF64(k, v, w, u, out, nil, nil, nil, seq, d, 0, d)
}

// WKVScanStateF64 resumes WKV from persistent AA/BB/PP state.
func WKVScanStateF64(k, v, w, u, out, aa0, bb0, pp0 []float64, seq, d int) {
	wkvScanStateRangeNeonF64(k, v, w, u, out, aa0, bb0, pp0, seq, d, 0, d)
}

// WKVScanRangeF64 scans a disjoint channel range. Two-aligned boundaries retain
// the same pair grouping as WKVScanF64; unaligned edge channels stay scalar.
func WKVScanRangeF64(k, v, w, u, out []float64, seq, d, cLo, cHi int) {
	wkvScanStateRangeNeonF64(k, v, w, u, out, nil, nil, nil, seq, d, cLo, cHi)
}
