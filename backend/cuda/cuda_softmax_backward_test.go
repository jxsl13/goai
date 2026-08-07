//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestSoftmaxBackward validates the GPU softmax VJP: the kernel matches the host analytic gradient
// ds_j = p_j*(dp_j - Σ p_k dp_k), AND that gradient matches a central finite-difference of
// L = Σ dp·softmax(s) w.r.t. the pre-softmax s (confirming ds is dL/ds — the core of attention backward).
func TestSoftmaxBackward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const R, C = 20, 96
	rng := rand.New(rand.NewSource(17))
	s := make([]float32, R*C)  // pre-softmax logits
	dp := make([]float32, R*C) // upstream gradient
	for i := range s {
		s[i] = float32(rng.NormFloat64() * 1.5)
		dp[i] = float32(rng.NormFloat64())
	}
	// Host softmax → p.
	sm := func(row int, sv []float32) []float64 {
		mx := math.Inf(-1)
		for j := 0; j < C; j++ {
			if v := float64(sv[row*C+j]); v > mx {
				mx = v
			}
		}
		p := make([]float64, C)
		var z float64
		for j := 0; j < C; j++ {
			p[j] = math.Exp(float64(sv[row*C+j]) - mx)
			z += p[j]
		}
		for j := 0; j < C; j++ {
			p[j] /= z
		}
		return p
	}
	p := make([]float32, R*C)
	for r := 0; r < R; r++ {
		pr := sm(r, s)
		for j := 0; j < C; j++ {
			p[r*C+j] = float32(pr[j])
		}
	}

	up := func(h []float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(R, C)
		if err != nil {
			t.Fatal(err)
		}
		if h != nil {
			if err := d.UploadF32(h); err != nil {
				t.Fatal(err)
			}
		}
		return d
	}
	pd, dpd, dsd := up(p), up(dp), up(nil)
	defer pd.Free()
	defer dpd.Free()
	defer dsd.Free()
	if err := cuda.SoftmaxBackward(dsd, pd, dpd); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, R*C)
	dsd.DownloadF32(got)

	// Analytic.
	refDs := make([]float64, R*C)
	var maxA float64
	for r := 0; r < R; r++ {
		pr := sm(r, s)
		var dot float64
		for j := 0; j < C; j++ {
			dot += pr[j] * float64(dp[r*C+j])
		}
		for j := 0; j < C; j++ {
			refDs[r*C+j] = pr[j] * (float64(dp[r*C+j]) - dot)
			if d := math.Abs(refDs[r*C+j] - float64(got[r*C+j])); d > maxA {
				maxA = d
			}
		}
	}

	// Finite-difference of L_row = Σ_k dp_k·softmax(s)_k w.r.t. s_rowj.
	rowL := func(row int, sv []float32) float64 {
		pr := sm(row, sv)
		var l float64
		for k := 0; k < C; k++ {
			l += float64(dp[row*C+k]) * pr[k]
		}
		return l
	}
	const h = 1e-3
	var maxFD float64
	for _, rj := range [][2]int{{0, 0}, {5, 40}, {13, 95}, {19, 7}} {
		r, j := rj[0], rj[1]
		orig := s[r*C+j]
		s[r*C+j] = orig + float32(h)
		lp := rowL(r, s)
		s[r*C+j] = orig - float32(h)
		lm := rowL(r, s)
		s[r*C+j] = orig
		fd := (lp - lm) / (2 * h)
		if d := math.Abs(fd - refDs[r*C+j]); d > maxFD {
			maxFD = d
		}
	}

	t.Logf("softmax backward: GPU vs analytic %.3e; analytic vs finite-diff %.3e", maxA, maxFD)
	if maxA > 1e-5 {
		t.Fatalf("GPU softmax backward diverges from analytic: %.3e", maxA)
	}
	if maxFD > 1e-3 {
		t.Fatalf("analytic softmax gradient disagrees with finite-difference: %.3e", maxFD)
	}
}
