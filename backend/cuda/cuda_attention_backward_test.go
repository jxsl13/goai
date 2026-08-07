//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestAttentionBackward validates the composed single-head attention VJP (dQ, dK, dV) against a central
// finite-difference of L = Σ dO·O, where O = softmax((Q·Kᵀ)·scale)·V. This proves the attention backward
// — matmul GEMMs + softmax backward + scale, composed on device — is correct, i.e. attention trains on GPU.
func TestAttentionBackward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const L, d = 8, 16
	scale := float32(1.0 / math.Sqrt(float64(d)))
	rng := rand.New(rand.NewSource(19))
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
	if err := cuda.AttentionBackward(dqd, dkd, dvd, qd, kd, vd, dod, scale); err != nil {
		t.Fatal(err)
	}
	gotDQ := dl(t, dqd, L*d)
	gotDK := dl(t, dkd, L*d)
	gotDV := dl(t, dvd, L*d)

	sc := float64(scale)
	// Host forward → L = Σ dO·O.
	lossWith := func(Qh, Kh, Vh []float32) float64 {
		var l float64
		for i := 0; i < L; i++ {
			// scores row i, softmax
			s := make([]float64, L)
			mx := math.Inf(-1)
			for j := 0; j < L; j++ {
				var dot float64
				for k := 0; k < d; k++ {
					dot += float64(Qh[i*d+k]) * float64(Kh[j*d+k])
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
			// O_i = Σ_j P_ij V_j ; contribute dO_i·O_i
			for k := 0; k < d; k++ {
				var o float64
				for j := 0; j < L; j++ {
					o += s[j] * float64(Vh[j*d+k])
				}
				l += float64(dO[i*d+k]) * o
			}
		}
		return l
	}
	const h = 1e-3
	fdGrad := func(base []float32) []float64 {
		g := make([]float64, len(base))
		for idx := range base {
			orig := base[idx]
			base[idx] = orig + float32(h)
			lp := lossWith(Q, K, V)
			base[idx] = orig - float32(h)
			lm := lossWith(Q, K, V)
			base[idx] = orig
			g[idx] = (lp - lm) / (2 * h)
		}
		return g
	}
	// Sample a subset of coords for each of Q,K,V to keep the FD cost low.
	fdQ := fdGrad(Q)
	fdK := fdGrad(K)
	fdV := fdGrad(V)

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
	t.Logf("attention backward vs finite-diff: dQ %.3e dK %.3e dV %.3e", mQ, mK, mV)
	if mQ > 5e-3 || mK > 5e-3 || mV > 5e-3 {
		t.Fatalf("attention backward diverges: dQ %.3e dK %.3e dV %.3e", mQ, mK, mV)
	}
}

func randM(rng *rand.Rand, r, c int) []float32 {
	s := make([]float32, r*c)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}

func dl(t *testing.T, d *cuda.DeviceF32, n int) []float32 {
	out := make([]float32, n)
	if err := d.DownloadF32(out); err != nil {
		t.Fatal(err)
	}
	return out
}
