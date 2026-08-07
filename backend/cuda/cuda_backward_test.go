//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestMatMulBackward validates the linear-layer backward GEMMs (dW = xᵀ·dY, dX = dY·Wᵀ) against host
// references — the matmul VJP that, with DeviceAdam, forms a GPU training path. The cuBLAS transpose
// mapping is easy to get wrong, so this pins it.
func TestMatMulBackward(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 40, 48, 32
	rng := rand.New(rand.NewSource(5))
	x := make([]float32, M*K)  // layer input [M,K]
	w := make([]float32, K*N)  // weight [K,N]
	dY := make([]float32, M*N) // output grad [M,N]
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	for i := range dY {
		dY[i] = float32(rng.NormFloat64())
	}

	up := func(rows, cols int, h []float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		if err := d.UploadF32(h); err != nil {
			t.Fatal(err)
		}
		return d
	}
	dx := up(M, K, x)
	dw := up(K, N, w)
	ddy := up(M, N, dY)
	defer dx.Free()
	defer dw.Free()
	defer ddy.Free()

	// dW = xᵀ·dY  [K,N]
	dW := up(K, N, make([]float32, K*N))
	defer dW.Free()
	if err := cuda.MatMulGradW(dx, ddy, dW); err != nil {
		t.Fatal(err)
	}
	gotDW := make([]float32, K*N)
	dW.DownloadF32(gotDW)
	var maxW float64
	for k := 0; k < K; k++ {
		for n := 0; n < N; n++ {
			var ref float64
			for m := 0; m < M; m++ {
				ref += float64(x[m*K+k]) * float64(dY[m*N+n])
			}
			if d := math.Abs(ref - float64(gotDW[k*N+n])); d > maxW {
				maxW = d
			}
		}
	}

	// dX = dY·Wᵀ  [M,K]
	dX := up(M, K, make([]float32, M*K))
	defer dX.Free()
	if err := cuda.MatMulGradX(ddy, dw, dX); err != nil {
		t.Fatal(err)
	}
	gotDX := make([]float32, M*K)
	dX.DownloadF32(gotDX)
	var maxX float64
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			var ref float64
			for n := 0; n < N; n++ {
				ref += float64(dY[m*N+n]) * float64(w[k*N+n])
			}
			if d := math.Abs(ref - float64(gotDX[m*K+k])); d > maxX {
				maxX = d
			}
		}
	}

	t.Logf("dW max abs diff %.3e, dX max abs diff %.3e", maxW, maxX)
	if maxW > 1e-3 {
		t.Fatalf("MatMulGradW (dW=xᵀ·dY) diverges: %.3e", maxW)
	}
	if maxX > 1e-3 {
		t.Fatalf("MatMulGradX (dX=dY·Wᵀ) diverges: %.3e", maxX)
	}
}
