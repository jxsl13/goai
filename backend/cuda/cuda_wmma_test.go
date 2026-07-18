//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// The nvcc-compiled WMMA GEMM must track an f64 host reference within the f16-input envelope
// (inputs round to f16 -> ~1e-3 rel per element; f32 accumulate). This is the load-bearing
// proof that the nvcc -> fatbin -> cuModuleLoadDataEx pipeline runs a real tensor-core kernel
// on the 3060 — the mma.h primitive nvrtc could not compile.
func TestCUDAWMMAGemm(t *testing.T) {
	skipNoGPU(t)
	const M, K, N = 64, 128, 96
	rng := rand.New(rand.NewSource(19))
	a := make([]float32, M*K)
	b := make([]float32, K*N)
	for i := range a {
		a[i] = float32(rng.NormFloat64()) * 0.25
	}
	for i := range b {
		b[i] = float32(rng.NormFloat64()) * 0.25
	}
	c := make([]float32, M*N)
	if err := cuda.WMMAGemm(a, b, c, M, K, N); err != nil {
		t.Fatal(err)
	}
	var num, den float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var ref float64
			for k := 0; k < K; k++ {
				ref += float64(a[m*K+k]) * float64(b[k*N+n])
			}
			d := float64(c[m*N+n]) - ref
			num += d * d
			den += ref * ref
		}
	}
	relRMS := math.Sqrt(num / den)
	t.Logf("nvcc WMMA GEMM vs f64 ref: norm-rel-RMS %.3e", relRMS)
	if relRMS > 5e-3 {
		t.Fatalf("nvcc WMMA GEMM off reference: norm-rel-RMS %.3e (f16-input envelope ~1e-3)", relRMS)
	}
}
