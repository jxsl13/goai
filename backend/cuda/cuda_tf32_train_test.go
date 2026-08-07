//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestTF32TrainingConverges checks that GPU training still converges with TF32 tensor cores enabled — the
// ~10-bit-mantissa GEMMs are the precision PyTorch trains with on Ampere, so a linear layer trained under
// SetGemmTF32(true) must still drive the loss down (not to f32-machine-zero, but substantially).
func TestTF32TrainingConverges(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	if err := cuda.SetGemmTF32(true); err != nil {
		t.Fatal(err)
	}
	defer cuda.SetGemmTF32(false)

	const M, K, N, steps = 64, 64, 32, 400
	rng := rand.New(rand.NewSource(3))
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
		cuda.MatMul(dX, dW, dY)
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
		if err := cuda.MatMul(dX, dW, dY); err != nil {
			t.Fatal(err)
		}
		if err := cuda.SubScaled(dDY, dY, dT, 2.0/float32(M)); err != nil {
			t.Fatal(err)
		}
		if err := cuda.MatMulGradW(dX, dDY, dDW); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step([]*cuda.DeviceF32{dW}, []*cuda.DeviceF32{dDW}); err != nil {
			t.Fatal(err)
		}
	}
	l1 := loss()
	t.Logf("TF32 GPU training: MSE %.4e -> %.4e over %d steps", l0, l1, steps)
	if l1 > l0/50 {
		t.Fatalf("TF32 training did not converge: %.4e -> %.4e", l0, l1)
	}
}
