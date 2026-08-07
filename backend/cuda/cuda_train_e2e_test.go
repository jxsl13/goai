//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestGPULinearTrainStep trains a linear layer Y = X·W to fit a realizable target T = X·W* ENTIRELY on
// the GPU — forward (MatMul), MSE gradient (SubScaled), weight gradient dW = Xᵀ·dY (MatMulGradW), and the
// AdamW update (DeviceAdam) — with no gradient or optimizer state leaving the device. It asserts the loss
// collapses toward zero, proving the GPU training path works end to end (the gap: training was CPU-only).
func TestGPULinearTrainStep(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N, steps = 64, 32, 16, 2000
	rng := rand.New(rand.NewSource(7))
	randMat := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	X := randMat(M * K)
	Wstar := randMat(K * N)
	T := make([]float32, M*N) // realizable target T = X·W*
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var s float64
			for k := 0; k < K; k++ {
				s += float64(X[m*K+k]) * float64(Wstar[k*N+n])
			}
			T[m*N+n] = float32(s)
		}
	}
	Winit := randMat(K * N)

	up := func(rows, cols int, h []float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(rows, cols)
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
	dX := up(M, K, X)
	dT := up(M, N, T)
	dW := up(K, N, Winit)
	dY := up(M, N, nil)
	dDY := up(M, N, nil)
	dDW := up(K, N, nil)
	defer dX.Free()
	defer dT.Free()
	defer dW.Free()
	defer dY.Free()
	defer dDY.Free()
	defer dDW.Free()

	opt, err := cuda.NewDeviceAdam([]int{K * N}, 1e-2, 0.9, 0.999, 1e-8, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer opt.Free()

	loss := func() float64 {
		if err := cuda.MatMul(dX, dW, dY); err != nil {
			t.Fatal(err)
		}
		y := make([]float32, M*N)
		if err := dY.DownloadF32(y); err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range y {
			d := float64(y[i] - T[i])
			s += d * d
		}
		return s / float64(M*N)
	}

	l0 := loss()
	for s := 0; s < steps; s++ {
		if err := cuda.MatMul(dX, dW, dY); err != nil { // Y = X·W
			t.Fatal(err)
		}
		if err := cuda.SubScaled(dDY, dY, dT, 2.0/float32(M)); err != nil { // dY = (2/M)(Y-T)
			t.Fatal(err)
		}
		if err := cuda.MatMulGradW(dX, dDY, dDW); err != nil { // dW = Xᵀ·dY
			t.Fatal(err)
		}
		if err := opt.Step([]*cuda.DeviceF32{dW}, []*cuda.DeviceF32{dDW}); err != nil { // AdamW update
			t.Fatal(err)
		}
	}
	l1 := loss()

	t.Logf("GPU linear training: MSE %.4e -> %.4e over %d steps (all on device)", l0, l1, steps)
	if l1 > l0/100 {
		t.Fatalf("GPU training did not converge: loss %.4e -> %.4e (want < %.4e)", l0, l1, l0/100)
	}
}
