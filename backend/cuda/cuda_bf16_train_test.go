//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestBf16TrainingConverges trains a linear layer with the bf16 tensor-core GEMMs (MatMulBf16 forward,
// MatMulGradWBf16 weight gradient) and an f32 master weight updated by DeviceAdam — the mixed-precision
// recipe. bf16's 8-bit mantissa sets a coarser floor than f32/TF32, but the loss must still drop
// substantially, proving bf16 training (the 2.6× GEMM speedup) is usable.
func TestBf16TrainingConverges(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N, steps = 64, 64, 32, 400
	rng := rand.New(rand.NewSource(5))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	X := rnd(M * K)
	Wstar := rnd(K * N)
	T := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var s float64
			for k := 0; k < K; k++ {
				s += float64(X[m*K+k]) * float64(Wstar[k*N+n])
			}
			T[m*N+n] = float32(s)
		}
	}
	up := func(r, c int, h []float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(r, c)
		if err != nil {
			t.Fatal(err)
		}
		if h != nil {
			d.UploadF32(h)
		}
		return d
	}
	dX, dT, dW := up(M, K, X), up(M, N, T), up(K, N, rnd(K*N))
	dY, dDY, dDW := up(M, N, nil), up(M, N, nil), up(K, N, nil)
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
		cuda.MatMulBf16(dX, dW, dY)
		y := make([]float32, M*N)
		dY.DownloadF32(y)
		var s float64
		for i := range y {
			d := float64(y[i] - T[i])
			s += d * d
		}
		return s / float64(M*N)
	}
	l0 := loss()
	for i := 0; i < steps; i++ {
		if err := cuda.MatMulBf16(dX, dW, dY); err != nil { // Y = X·W  (bf16)
			t.Fatal(err)
		}
		if err := cuda.SubScaled(dDY, dY, dT, 2.0/float32(M)); err != nil {
			t.Fatal(err)
		}
		if err := cuda.MatMulGradWBf16(dX, dDY, dDW); err != nil { // dW = Xᵀ·dY  (bf16)
			t.Fatal(err)
		}
		if err := opt.Step([]*cuda.DeviceF32{dW}, []*cuda.DeviceF32{dDW}); err != nil {
			t.Fatal(err)
		}
	}
	l1 := loss()
	t.Logf("bf16 GPU training: MSE %.4e -> %.4e over %d steps", l0, l1, steps)
	if l1 > l0/20 {
		t.Fatalf("bf16 training did not converge: %.4e -> %.4e", l0, l1)
	}
}
