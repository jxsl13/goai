//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math/rand"
	"testing"
)

func TestZZI8MMARBCorrect(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	// Non-square, multi-block on every axis: M=128 (2 M-blocks), N=128 (2 N-blocks), K=96 (3 K-steps).
	const M, K, N = 128, 96, 128
	rng := rand.New(rand.NewSource(7))
	a := make([]int8, M*K)
	w := make([]int8, K*N)
	for i := range a {
		a[i] = int8(rng.Intn(15) - 7)
	}
	for i := range w {
		w[i] = int8(rng.Intn(13) - 6)
	}
	got := i8MMARB(a, w, M, K, N)
	if got == nil {
		t.Fatal("i8MMARB failed")
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
	t.Logf("register-blocked i8 mma GEMM %dx%dx%d maxErr = %d", M, K, N, maxErr)
	if maxErr != 0 {
		t.Fatalf("REG-BLOCKED int8 mma GEMM WRONG: maxErr %d", maxErr)
	}
	t.Log("REGISTER-BLOCKED INT8 GEMM CORRECT")
}

func BenchmarkI8MMARB_prefill(b *testing.B) {
	if !Available() {
		b.Skip("no gpu")
	}
	const M, K, N = 128, 2048, 2048
	a := make([]int8, M*K)
	w := make([]int8, K*N)
	dA := i8uploadRaw(a)
	dW := i8uploadRaw(w)
	dC := i8allocRaw(M * N * 4)
	i8mmarbRaw(dA, dW, dC, M, K, N)
	i8syncRaw()
	b.ResetTimer()
	for range b.N {
		i8mmarbRaw(dA, dW, dC, M, K, N)
	}
	i8syncRaw()
	b.StopTimer()
	b.ReportMetric(2.0*float64(M)*float64(K)*float64(N)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GOP/s")
}
