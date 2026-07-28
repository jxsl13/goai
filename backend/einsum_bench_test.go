package backend_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func benchEinsum(b *testing.B, in []string, out string, dims [][]int) {
	ops := make([]*tensor.Tensor, len(in))
	subs := make([][]byte, len(in))
	for k := range in {
		ops[k] = bench.RandF64(tensor.Shape(dims[k]), uint64(k+1))
		subs[k] = []byte(in[k])
	}
	o := []byte(out)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.EinsumContract(subs, o, ops); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEinsumMatmul_32(b *testing.B) {
	benchEinsum(b, []string{"ij", "jk"}, "ik", [][]int{{32, 32}, {32, 32}})
}
func BenchmarkEinsumBatched_8x16(b *testing.B) {
	benchEinsum(b, []string{"bij", "bjk"}, "bik", [][]int{{8, 16, 16}, {8, 16, 16}})
}
