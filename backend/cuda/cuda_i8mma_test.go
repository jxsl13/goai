//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math/rand"
	"testing"
)

func TestZZI8MMACorrect(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 32, 64, 16
	rng := rand.New(rand.NewSource(1))
	a := make([]int8, M*K)
	w := make([]int8, K*N)
	for i := range a {
		a[i] = int8(rng.Intn(15) - 7)
	}
	for i := range w {
		w[i] = int8(rng.Intn(13) - 6)
	}
	got := i8MMA(a, w, M, K, N)
	if got == nil {
		t.Fatal("i8MMA failed")
	}
	var maxErr int32
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var ref int32
			for k := 0; k < K; k++ {
				ref += int32(a[m*K+k]) * int32(w[k*N+n])
			}
			e := got[m*N+n] - ref
			if e < 0 {
				e = -e
			}
			if e > maxErr {
				maxErr = e
			}
		}
	}
	t.Logf("tiled i8 mma GEMM %dx%dx%d maxErr = %d", M, K, N, maxErr)
	if maxErr != 0 {
		t.Fatalf("tiled int8 mma GEMM WRONG: maxErr %d", maxErr)
	}
	t.Log("TILED INT8 TENSOR-CORE GEMM CORRECT")
}
