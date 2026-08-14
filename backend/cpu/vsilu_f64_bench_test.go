package cpu

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func BenchmarkSiLUF64Kernel(b *testing.B) {
	x := tensor.New(tensor.F64, tensor.Shape{256, 1408}) // cache-resident FFN tile (compute-bound)
	s := x.Storage().F64()
	for i := range s {
		s[i] = -4 + 8*float64(i%1000)/1000
	}
	ctx := backend.NewContext()
	inputs := []*tensor.Tensor{x}
	b.SetBytes(int64(len(s) * 16)) // one read plus one write
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.Execute(ctx, backend.OpSiLU, inputs, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSiLUF64Into measures the eager dispatch path with caller-owned
// output storage. Input and output descriptor slices are built before the timer;
// any reported bytes therefore come from ExecuteInto or the selected kernel.
func BenchmarkSiLUF64Into(b *testing.B) {
	shape := tensor.Shape{256, 1408}
	x := tensor.New(tensor.F64, shape)
	out := tensor.New(tensor.F64, shape)
	s := x.Storage().F64()
	for i := range s {
		s[i] = -4 + 8*float64(i%1000)/1000
	}
	ctx := backend.NewContext()
	inputs := []*tensor.Tensor{x}
	outputs := []*tensor.Tensor{out}
	b.SetBytes(int64(len(s) * 16)) // one read plus one write
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := backend.ExecuteInto(ctx, backend.OpSiLU, inputs, outputs, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSiLUF64Leaf isolates the L1 arithmetic from tensor construction and
// garbage collection. Compare it with PyTorch aten.silu.out on the same shape.
func BenchmarkSiLUF64Leaf(b *testing.B) {
	src := make([]float64, 256*1408)
	dst := make([]float64, len(src))
	for i := range src {
		src[i] = -4 + 8*float64(i%1000)/1000
	}
	b.SetBytes(int64(len(src) * 16)) // one read plus one write
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		vsiluF64(dst, src)
	}
}
