package simd

import (
	"math"
	"math/rand"
	"testing"
)

// TestWKVScanF64Accuracy checks the channel-vectorized WKV scan against the scalar
// reference over the full recurrence (the error accumulates across seq steps, so
// this validates the log-space stabilization survives the ~1-ulp SIMD exp). Non-
// multiple-of-4 d exercises the scalar channel tail.
func TestWKVScanF64Accuracy(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, dims := range [][2]int{{128, 512}, {64, 1024}, {130, 17}, {200, 64}} {
		seq, d := dims[0], dims[1]
		k := make([]float64, seq*d)
		v := make([]float64, seq*d)
		w := make([]float64, d)
		u := make([]float64, d)
		for i := range k {
			k[i] = rng.NormFloat64()
			v[i] = rng.NormFloat64() * 2
		}
		for c := range w {
			w[c] = 0.5 + rng.Float64()     // positive decay, RWKV-like
			u[c] = rng.NormFloat64() * 0.5 // bonus
		}
		got := make([]float64, seq*d)
		WKVScanF64(k, v, w, u, got, seq, d)
		want := make([]float64, seq*d)
		wkvScanScalar(k, v, w, u, want, seq, d, 0, d)
		var maxRel float64
		for i := range got {
			den := math.Max(1e-6, math.Abs(want[i]))
			if rel := math.Abs(got[i]-want[i]) / den; rel > maxRel {
				maxRel = rel
			}
		}
		t.Logf("seq=%d d=%d maxRel=%.3e", seq, d, maxRel)
		if maxRel > 1e-10 {
			t.Fatalf("seq=%d d=%d: WKVScanF64 maxRel=%.3e exceeds 1e-10 vs scalar", seq, d, maxRel)
		}
	}
}

func benchWKVScan(b *testing.B, seq, d int, simd bool) {
	rng := rand.New(rand.NewSource(1))
	k := make([]float64, seq*d)
	v := make([]float64, seq*d)
	w := make([]float64, d)
	u := make([]float64, d)
	for i := range k {
		k[i], v[i] = rng.NormFloat64(), rng.NormFloat64()
	}
	for c := range w {
		w[c], u[c] = 0.5+rng.Float64(), rng.NormFloat64()*0.5
	}
	out := make([]float64, seq*d)
	b.ResetTimer()
	for range b.N {
		if simd {
			WKVScanF64(k, v, w, u, out, seq, d)
		} else {
			wkvScanScalar(k, v, w, u, out, seq, d, 0, d)
		}
	}
}

func BenchmarkWKVScan_SIMD_512x1024(b *testing.B)   { benchWKVScan(b, 512, 1024, true) }
func BenchmarkWKVScan_Scalar_512x1024(b *testing.B) { benchWKVScan(b, 512, 1024, false) }
