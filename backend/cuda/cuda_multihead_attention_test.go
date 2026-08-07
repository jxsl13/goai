//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestMultiHeadAttentionBackward validates the multi-head attention VJP (dQ, dK, dV) against a central
// finite-difference of L = Σ dO·O, where O concatenates per-head scaled-dot-product attention. Proves the
// per-head gather/AttentionBackward/scatter composition is correct for real (multi-head) transformers.
func TestMultiHeadAttentionBackward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const L, nHeads, hd = 8, 4, 8
	const D = nHeads * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rng := rand.New(rand.NewSource(41))
	Q := randM(rng, L, D)
	K := randM(rng, L, D)
	V := randM(rng, L, D)
	dO := randM(rng, L, D)

	up := func(h []float32) *cuda.DeviceF32 {
		x, err := cuda.NewDeviceF32(L, D)
		if err != nil {
			t.Fatal(err)
		}
		if h != nil {
			if err := x.UploadF32(h); err != nil {
				t.Fatal(err)
			}
		}
		return x
	}
	qd, kd, vd, dod := up(Q), up(K), up(V), up(dO)
	dqd, dkd, dvd := up(nil), up(nil), up(nil)
	defer qd.Free()
	defer kd.Free()
	defer vd.Free()
	defer dod.Free()
	defer dqd.Free()
	defer dkd.Free()
	defer dvd.Free()
	if err := cuda.MultiHeadAttentionBackward(dqd, dkd, dvd, qd, kd, vd, dod, nHeads, scale); err != nil {
		t.Fatal(err)
	}
	gotDQ := dl(t, dqd, L*D)
	gotDK := dl(t, dkd, L*D)
	gotDV := dl(t, dvd, L*D)

	sc := float64(scale)
	loss := func(Qh, Kh, Vh []float32) float64 {
		var l float64
		for head := 0; head < nHeads; head++ {
			off := head * hd
			for i := 0; i < L; i++ {
				s := make([]float64, L)
				mx := math.Inf(-1)
				for j := 0; j < L; j++ {
					var dot float64
					for k := 0; k < hd; k++ {
						dot += float64(Qh[i*D+off+k]) * float64(Kh[j*D+off+k])
					}
					s[j] = dot * sc
					if s[j] > mx {
						mx = s[j]
					}
				}
				var z float64
				for j := 0; j < L; j++ {
					s[j] = math.Exp(s[j] - mx)
					z += s[j]
				}
				for j := 0; j < L; j++ {
					s[j] /= z
				}
				for k := 0; k < hd; k++ {
					var o float64
					for j := 0; j < L; j++ {
						o += s[j] * float64(Vh[j*D+off+k])
					}
					l += float64(dO[i*D+off+k]) * o
				}
			}
		}
		return l
	}
	const h = 1e-3
	fd := func(base []float32) []float64 {
		g := make([]float64, len(base))
		for idx := range base {
			orig := base[idx]
			base[idx] = orig + float32(h)
			lp := loss(Q, K, V)
			base[idx] = orig - float32(h)
			lm := loss(Q, K, V)
			base[idx] = orig
			g[idx] = (lp - lm) / (2 * h)
		}
		return g
	}
	fdQ, fdK, fdV := fd(Q), fd(K), fd(V)
	maxOf := func(got []float32, ref []float64) float64 {
		var m float64
		for i := range ref {
			if x := math.Abs(float64(got[i]) - ref[i]); x > m {
				m = x
			}
		}
		return m
	}
	mQ, mK, mV := maxOf(gotDQ, fdQ), maxOf(gotDK, fdK), maxOf(gotDV, fdV)
	t.Logf("multi-head attention backward vs finite-diff: dQ %.3e dK %.3e dV %.3e", mQ, mK, mV)
	if mQ > 5e-3 || mK > 5e-3 || mV > 5e-3 {
		t.Fatalf("multi-head attention backward diverges: dQ %.3e dK %.3e dV %.3e", mQ, mK, mV)
	}
}
