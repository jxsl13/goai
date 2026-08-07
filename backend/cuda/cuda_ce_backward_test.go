//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestCrossEntropyBackward validates the GPU softmax-cross-entropy loss gradient against a host-analytic
// reference AND a central finite-difference of the loss L = -mean(log softmax(logits)[target]) — the
// latter confirms the formula dlogits = (1/M)(softmax - onehot), not just the kernel.
func TestCrossEntropyBackward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, V = 12, 500
	rng := rand.New(rand.NewSource(13))
	logits := make([]float32, M*V)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64() * 2)
	}
	targets := make([]int32, M)
	for i := range targets {
		targets[i] = int32(rng.Intn(V))
	}
	scale := float32(1.0 / M)

	ld, err := cuda.NewDeviceF32(M, V)
	if err != nil {
		t.Fatal(err)
	}
	defer ld.Free()
	if err := ld.UploadF32(logits); err != nil {
		t.Fatal(err)
	}
	dld, err := cuda.NewDeviceF32(M, V)
	if err != nil {
		t.Fatal(err)
	}
	defer dld.Free()
	if err := cuda.CrossEntropyBackward(dld, ld, targets, scale); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, M*V)
	dld.DownloadF32(got)

	// Host softmax + analytic gradient.
	softmax := func(m int) []float64 {
		mx := math.Inf(-1)
		for j := 0; j < V; j++ {
			if v := float64(logits[m*V+j]); v > mx {
				mx = v
			}
		}
		p := make([]float64, V)
		var z float64
		for j := 0; j < V; j++ {
			p[j] = math.Exp(float64(logits[m*V+j]) - mx)
			z += p[j]
		}
		for j := 0; j < V; j++ {
			p[j] /= z
		}
		return p
	}
	var maxG float64
	for m := 0; m < M; m++ {
		p := softmax(m)
		for j := 0; j < V; j++ {
			ref := float64(scale) * (p[j] - b2f(j == int(targets[m])))
			if d := math.Abs(ref - float64(got[m*V+j])); d > maxG {
				maxG = d
			}
		}
	}

	// Finite-difference of L = -mean(log p[target]) on a few (row, class) coords.
	loss := func() float64 {
		var l float64
		for m := 0; m < M; m++ {
			p := softmax(m)
			l += -math.Log(p[int(targets[m])])
		}
		return l / float64(M)
	}
	const h = 1e-3
	var maxFD float64
	for _, mj := range [][2]int{{0, 0}, {3, 100}, {7, int(targets[7])}, {11, 499}} {
		m, j := mj[0], mj[1]
		orig := logits[m*V+j]
		logits[m*V+j] = orig + float32(h)
		lp := loss()
		logits[m*V+j] = orig - float32(h)
		lm := loss()
		logits[m*V+j] = orig
		fd := (lp - lm) / (2 * h)
		p := softmax(m)
		ref := float64(scale) * (p[j] - b2f(j == int(targets[m])))
		if d := math.Abs(fd - ref); d > maxFD {
			maxFD = d
		}
	}

	t.Logf("cross-entropy backward: GPU vs analytic %.3e; analytic vs finite-diff %.3e", maxG, maxFD)
	if maxG > 1e-4 {
		t.Fatalf("GPU cross-entropy backward diverges from analytic: %.3e", maxG)
	}
	if maxFD > 1e-3 {
		t.Fatalf("analytic cross-entropy gradient disagrees with finite-difference: %.3e", maxFD)
	}
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
