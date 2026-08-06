//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

// genMMQ builds random int8 A[m,k], W[n,k] (row n = weight col n over K), and per-32-block f32
// scales aSc[m][k/32], wSc[n][k/32] — the layout cu_matmul_i8_mmq{,_lm} consume.
func genMMQ(m, k, n int) (a8, w8 []int8, aSc, wSc []float32) {
	rng := rand.New(rand.NewSource(7))
	nb := k / 32
	a8 = make([]int8, m*k)
	w8 = make([]int8, n*k)
	aSc = make([]float32, m*nb)
	wSc = make([]float32, n*nb)
	for i := range a8 {
		a8[i] = int8(rng.Intn(255) - 127)
	}
	for i := range w8 {
		w8[i] = int8(rng.Intn(255) - 127)
	}
	for i := range aSc {
		aSc[i] = float32(rng.NormFloat64())*0.01 + 0.02
	}
	for i := range wSc {
		wSc[i] = float32(rng.NormFloat64())*0.01 + 0.02
	}
	return
}

// TestI8MMQLMMatchesMMQ: the ldmatrix core must produce the SAME dequantized f32 result as the
// manual-load core (identical math, only the fragment-load path differs).
func TestI8MMQLMMatchesMMQ(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	m, k, n := 64, 256, 128
	a8, w8, aSc, wSc := genMMQ(m, k, n)
	ref := i8MMQ(a8, w8, aSc, wSc, m, k, n)
	got := i8MMQLM(a8, w8, aSc, wSc, m, k, n)
	if ref == nil || got == nil {
		t.Fatalf("kernel returned nil (ref=%v got=%v)", ref == nil, got == nil)
	}
	var maxAbs float64
	for i := range ref {
		d := math.Abs(float64(ref[i] - got[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("i8mmq vs i8mmqlm max abs diff: %.3e", maxAbs)
	if maxAbs > 1e-3 {
		t.Fatalf("ldmatrix core diverges from manual core: maxAbs=%.3e", maxAbs)
	}
}

func benchI8MMQvsLM(b *testing.B, m, k, n int) {
	if !Available() {
		b.Skip("no gpu")
	}
	a8, w8, aSc, wSc := genMMQ(m, k, n)
	dA := i8Upload(a8)
	dW := i8Upload(w8)
	dAs := f32uploadRaw(aSc)
	dWs := f32uploadRaw(wSc)
	dC := devAllocBytes(m * n * 4)
	defer func() { devFree(dA); devFree(dW); devFree(dAs); devFree(dWs); devFree(dC) }()
	tflops := func(el float64) float64 { return 2 * float64(m) * float64(k) * float64(n) * float64(b.N) / el / 1e12 }
	b.Run("manual", func(b *testing.B) {
		i8mmqForBench(dA, dW, dAs, dWs, dC, m, k, n)
		devSync()
		b.ResetTimer()
		for range b.N {
			i8mmqForBench(dA, dW, dAs, dWs, dC, m, k, n)
		}
		devSync()
		b.StopTimer()
		b.ReportMetric(tflops(b.Elapsed().Seconds()), "TOPS")
	})
	b.Run("ldmatrix", func(b *testing.B) {
		i8mmqLmForBench(dA, dW, dAs, dWs, dC, m, k, n)
		devSync()
		b.ResetTimer()
		for range b.N {
			i8mmqLmForBench(dA, dW, dAs, dWs, dC, m, k, n)
		}
		devSync()
		b.StopTimer()
		b.ReportMetric(tflops(b.Elapsed().Seconds()), "TOPS")
	})
}

func BenchmarkI8MMQvsLM_512x2048x2048(b *testing.B) { benchI8MMQvsLM(b, 512, 2048, 2048) }
