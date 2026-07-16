//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math/rand"
	"testing"
)

func TestZZI8MMALMCorrect(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 128, 96, 128 // multi-block, W is [N][K]
	rng := rand.New(rand.NewSource(41))
	a := make([]int8, M*K)
	wt := make([]int8, N*K)
	for i := range a {
		a[i] = int8(rng.Intn(15) - 7)
	}
	for i := range wt {
		wt[i] = int8(rng.Intn(13) - 6)
	}
	got := i8MMALM(a, wt, M, K, N)
	if got == nil {
		t.Fatal("i8MMALM failed")
	}
	var maxErr int32
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var ref int32
			for k := 0; k < K; k++ {
				ref += int32(a[m*K+k]) * int32(wt[n*K+k]) // W[n][k]
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
	t.Logf("ldmatrix i8 GEMM %dx%dx%d maxErr = %d", M, K, N, maxErr)
	if maxErr != 0 {
		t.Fatalf("LDMATRIX int8 GEMM WRONG: maxErr %d", maxErr)
	}
	t.Log("LDMATRIX INT8 GEMM CORRECT (ldmatrix.x4 A + ldmatrix.x2 B)")
}

func BenchmarkI8MMALM_prefill(b *testing.B) {
	if !Available() {
		b.Skip("no gpu")
	}
	const M, K, N = 128, 2048, 2048
	a := make([]int8, M*K)
	wt := make([]int8, N*K)
	dA := i8uploadRaw(a)
	dW := i8uploadRaw(wt)
	dC := i8allocRaw(M * N * 4)
	i8mmalmRaw(dA, dW, dC, M, K, N)
	i8syncRaw()
	b.ResetTimer()
	for range b.N {
		i8mmalmRaw(dA, dW, dC, M, K, N)
	}
	i8syncRaw()
	b.StopTimer()
	b.ReportMetric(2.0*float64(M)*float64(K)*float64(N)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GOP/s")
}
