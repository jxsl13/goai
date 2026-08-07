//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestBf16MLPTrainsOnGPU trains a 2-layer linear MLP y=(x·W1)·W2 entirely with the bf16 tensor-core
// GEMMs — forward MatMulBf16, weight gradients MatMulGradWBf16, and the INPUT gradient MatMulGradXBf16
// (dh1 = dY·W2ᵀ) that chains the gradient back to the first layer. Convergence proves all three bf16
// GEMMs compose correctly for multi-layer mixed-precision training.
func TestBf16MLPTrainsOnGPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, H, N, steps = 64, 32, 16, 32, 500
	rng := rand.New(rand.NewSource(7))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	X := rnd(M * K)
	// Realizable low-rank target T = X·W1*·W2*.
	W1s, W2s := rnd(K*H), rnd(H*N)
	hid := make([]float32, M*H)
	for m := 0; m < M; m++ {
		for h := 0; h < H; h++ {
			var s float64
			for k := 0; k < K; k++ {
				s += float64(X[m*K+k]) * float64(W1s[k*H+h])
			}
			hid[m*H+h] = float32(s)
		}
	}
	T := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var s float64
			for h := 0; h < H; h++ {
				s += float64(hid[m*H+h]) * float64(W2s[h*N+n])
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
	dX, dT := up(M, K, X), up(M, N, T)
	W1, W2 := up(K, H, rnd(K*H)), up(H, N, rnd(H*N))
	h1, Y := up(M, H, nil), up(M, N, nil)
	dY, dh1 := up(M, N, nil), up(M, H, nil)
	dW1, dW2 := up(K, H, nil), up(H, N, nil)
	for _, b := range []*cuda.DeviceF32{dX, dT, W1, W2, h1, Y, dY, dh1, dW1, dW2} {
		defer b.Free()
	}
	opt, err := cuda.NewDeviceAdam([]int{K * H, H * N}, 1e-2, 0.9, 0.999, 1e-8, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer opt.Free()

	must := func(e error) {
		if e != nil {
			t.Fatal(e)
		}
	}
	loss := func() float64 {
		must(cuda.MatMulBf16(dX, W1, h1))
		must(cuda.MatMulBf16(h1, W2, Y))
		y := make([]float32, M*N)
		Y.DownloadF32(y)
		var s float64
		for i := range y {
			d := float64(y[i] - T[i])
			s += d * d
		}
		return s / float64(M*N)
	}
	l0 := loss()
	for i := 0; i < steps; i++ {
		must(cuda.MatMulBf16(dX, W1, h1)) // h1 = X·W1
		must(cuda.MatMulBf16(h1, W2, Y))  // Y  = h1·W2
		must(cuda.SubScaled(dY, Y, dT, 2.0/float32(M*N)))
		must(cuda.MatMulGradWBf16(h1, dY, dW2))  // dW2 = h1ᵀ·dY
		must(cuda.MatMulGradXBf16(dY, W2, dh1))  // dh1 = dY·W2ᵀ  (input grad → chains to layer 1)
		must(cuda.MatMulGradWBf16(dX, dh1, dW1)) // dW1 = Xᵀ·dh1
		must(opt.Step([]*cuda.DeviceF32{W1, W2}, []*cuda.DeviceF32{dW1, dW2}))
	}
	l1 := loss()
	t.Logf("bf16 2-layer MLP GPU training: MSE %.4e -> %.4e over %d steps", l0, l1, steps)
	if l1 > l0/20 {
		t.Fatalf("bf16 MLP did not converge: %.4e -> %.4e", l0, l1)
	}
}
