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

// TestWKVScanStateF64ChunkEqualsWhole is the parity property RWKV's forward/decode
// consistency rides on: absorbing a sequence in chunks through the CONTINUING scan
// (carrying aa/bb/pp across calls — what decode does) must equal one whole-sequence
// fresh scan (what the forward op does). Because the f64 state round-trips through the
// state slices losslessly and the per-token arithmetic is identical either way, this
// must hold BIT-EXACTLY — not just within tolerance. Non-multiple-of-4 d also exercises
// the scalar channel tail's state carry.
func TestWKVScanStateF64ChunkEqualsWhole(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, dims := range [][2]int{{96, 512}, {96, 130}, {50, 17}} {
		seq, d := dims[0], dims[1]
		k := make([]float64, seq*d)
		v := make([]float64, seq*d)
		w := make([]float64, d)
		u := make([]float64, d)
		for i := range k {
			k[i], v[i] = rng.NormFloat64(), rng.NormFloat64()*2
		}
		for c := range w {
			w[c], u[c] = 0.5+rng.Float64(), rng.NormFloat64()*0.5
		}
		whole := make([]float64, seq*d)
		WKVScanF64(k, v, w, u, whole, seq, d)

		// Absorb the same sequence in three uneven chunks against persistent state.
		aa := make([]float64, d)
		bb := make([]float64, d)
		pp := make([]float64, d)
		for c := range pp {
			pp[c] = -1e38 // fresh start = the -1e38 the forward scan seeds
		}
		chunked := make([]float64, seq*d)
		for _, span := range [][2]int{{0, 13}, {13, 40}, {40, seq}} {
			t0, t1 := span[0], span[1]
			n := t1 - t0
			WKVScanStateF64(k[t0*d:t1*d], v[t0*d:t1*d], w, u, chunked[t0*d:t1*d], aa, bb, pp, n, d)
		}
		for i := range whole {
			if whole[i] != chunked[i] {
				t.Fatalf("seq=%d d=%d i=%d: chunked decode %.17g != whole forward %.17g (must be bit-exact)",
					seq, d, i, chunked[i], whole[i])
			}
		}
		t.Logf("seq=%d d=%d: chunked continuing scan bit-exact vs whole scan", seq, d)
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
