package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSSDQuadratic covers the Mamba-2 SSD quadratic (prefill/attention) form.
// T=256 tokens, d=64 head dim, n=16 state dim.
func BenchmarkSSDQuadratic(b *testing.B) {
	const T, d, n = 256, 64, 16
	mk := func(fn func(i int) float64, shape ...int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape(shape))
		s := t.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return t
	}
	x := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) }, T, d)
	a := mk(func(i int) float64 { return 0.9 + 0.05*math.Cos(float64(i)) }, T) // decay in (0,1)
	bb := mk(func(i int) float64 { return math.Cos(float64(i) * 0.02) }, T, n)
	cc := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) }, T, n)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.SSDQuadratic(x, a, bb, cc); err != nil {
			b.Fatal(err)
		}
	}
}

// benchSSDRecurrent covers the Mamba-2 SSD linear-time scan. The state h=[n,d]
// dominates the output-mix loop; large n·d (exceeding L2) exposes the strided
// vs contiguous memory-access difference in that reduction.
func benchSSDRecurrent(b *testing.B, T, d, n int) {
	mk := func(fn func(i int) float64, shape ...int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape(shape))
		s := t.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return t
	}
	x := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) }, T, d)
	a := mk(func(i int) float64 { return 0.9 + 0.05*math.Cos(float64(i)) }, T)
	bb := mk(func(i int) float64 { return math.Cos(float64(i) * 0.02) }, T, n)
	cc := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) }, T, n)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.SSDRecurrent(x, a, bb, cc); err != nil {
			b.Fatal(err)
		}
	}
}

// State h=[128,256]·8B = 256 KB — exceeds L2; the strided output reduction is
// the memory-access hotspot.
func BenchmarkSSDRecurrent_T256_d256_n128(b *testing.B) { benchSSDRecurrent(b, 256, 256, 128) }

// State h=[16,64]·8B = 8 KB — fits L1; spatial-locality gain only.
func BenchmarkSSDRecurrent_T256_d64_n16(b *testing.B) { benchSSDRecurrent(b, 256, 64, 16) }
