//go:build cuda && cgo && (linux || windows)

package cuda

import "testing"

func BenchmarkI8MMA_prefill(b *testing.B) {
	if !Available() {
		b.Skip("no gpu")
	}
	const M, K, N = 128, 2048, 2048
	a := make([]int8, M*K)
	w := make([]int8, K*N)
	for i := range a {
		a[i] = int8(i % 7)
	}
	for i := range w {
		w[i] = int8(i % 5)
	}
	dA := i8uploadRaw(a)
	dW := i8uploadRaw(w)
	dC := i8allocRaw(M * N * 4)
	i8mmaRaw(dA, dW, dC, M, K, N)
	i8syncRaw()
	b.ResetTimer()
	for range b.N {
		i8mmaRaw(dA, dW, dC, M, K, N)
	}
	i8syncRaw()
	b.StopTimer()
	b.ReportMetric(2.0*float64(M)*float64(K)*float64(N)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GOP/s")
}
