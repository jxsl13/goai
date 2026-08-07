//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestCausalAttentionBackward validates the causal attention VJP (dQ, dK, dV) against a central
// finite-difference of L = Σ dO·O, where O = causal-softmax((Q·Kᵀ)/√d)·V (query i attends only to keys
// j ≤ i). Proves the causal-masked attention backward is correct — the attention needed for
// autoregressive-LM training.
func TestCausalAttentionBackward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const L, d = 8, 16
	scale := float64(1.0 / math.Sqrt(float64(d)))
	rng := rand.New(rand.NewSource(43))
	Q := randM(rng, L, d)
	K := randM(rng, L, d)
	V := randM(rng, L, d)
	dO := randM(rng, L, d)

	up := func(h []float32) *cuda.DeviceF32 {
		x, err := cuda.NewDeviceF32(L, d)
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
	if err := cuda.AttentionBackwardCausal(dqd, dkd, dvd, qd, kd, vd, dod, float32(scale)); err != nil {
		t.Fatal(err)
	}
	gotDQ := dl(t, dqd, L*d)
	gotDK := dl(t, dkd, L*d)
	gotDV := dl(t, dvd, L*d)

	loss := func(Qh, Kh, Vh []float32) float64 {
		var l float64
		for i := 0; i < L; i++ {
			s := make([]float64, i+1) // causal: keys 0..i only
			mx := math.Inf(-1)
			for j := 0; j <= i; j++ {
				var dot float64
				for k := 0; k < d; k++ {
					dot += float64(Qh[i*d+k]) * float64(Kh[j*d+k])
				}
				s[j] = dot * scale
				if s[j] > mx {
					mx = s[j]
				}
			}
			var z float64
			for j := 0; j <= i; j++ {
				s[j] = math.Exp(s[j] - mx)
				z += s[j]
			}
			for k := 0; k < d; k++ {
				var o float64
				for j := 0; j <= i; j++ {
					o += (s[j] / z) * float64(Vh[j*d+k])
				}
				l += float64(dO[i*d+k]) * o
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
	t.Logf("causal attention backward vs finite-diff: dQ %.3e dK %.3e dV %.3e", mQ, mK, mV)
	if mQ > 5e-3 || mK > 5e-3 || mV > 5e-3 {
		t.Fatalf("causal attention backward diverges: dQ %.3e dK %.3e dV %.3e", mQ, mK, mV)
	}
}
