//go:build amd64 && goexperiment.simd

package cpu

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"testing"
)

func BenchmarkSiLUF64Kernel(b *testing.B) {
	x := tensor.New(tensor.F64, tensor.Shape{256, 1408}) // cache-resident FFN tile (compute-bound)
	s := x.Storage().F64()
	for i := range s {
		s[i] = -4 + 8*float64(i%1000)/1000
	}
	ctx := backend.NewContext()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := backend.Execute(ctx, backend.OpSiLU, []*tensor.Tensor{x}, nil); err != nil {
			b.Fatal(err)
		}
	}
}
