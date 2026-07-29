//go:build amd64 && goexperiment.simd

package gguf

import (
	"math"
	"testing"
)

func TestDotQ4KAsm(t *testing.T) {
	for _, k := range []int{256, 512, 4096} {
		x := make([]float32, k)
		w := make([]float32, k)
		for i := range x {
			x[i] = float32(math.Sin(float64(i) * 0.01))
			w[i] = float32(math.Cos(float64(i) * 0.013))
		}
		raw := quantizeQ4_K(w)
		got := dotQ4_KRowASM(x, raw, k)
		want := dotQ4_KRow(x, raw, k)
		rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
		if rel > 1e-4 {
			t.Fatalf("k=%d: asm %v vs scalar %v (rel %g)", k, got, want, rel)
		}
		t.Logf("k=%d: asm=%v scalar=%v rel=%g", k, got, want, rel)
	}
}

var q4dsink float64
func BenchmarkDotQ4KAsm(b *testing.B) {
	const k = 4096
	x := make([]float32, k); w := make([]float32, k)
	for i := range x { x[i]=float32(i%7); w[i]=float32(i%5) }
	raw := quantizeQ4_K(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ { q4dsink = dotQ4_KRowASM(x, raw, k) }
}
func BenchmarkDotQ4KScalar(b *testing.B) {
	const k = 4096
	x := make([]float32, k); w := make([]float32, k)
	for i := range x { x[i]=float32(i%7); w[i]=float32(i%5) }
	raw := quantizeQ4_K(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ { q4dsink = dotQ4_KRow(x, raw, k) }
}
