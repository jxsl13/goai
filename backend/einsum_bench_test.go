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

// Control for einsum measurements: exercises the shared backend plumbing without
// touching EinsumContract, so a run that moves this is measuring the machine rather
// than the change. Added after d58e40b shipped WITHOUT an in-package control.
func BenchmarkEinsumControlStrides(b *testing.B) {
	sh := tensor.Shape{8, 16, 16}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		s := tensor.RowMajorStrides(sh)
		if s[0] == 0 {
			b.Fatal("unreachable")
		}
	}
}

func benchEinsumF32(b *testing.B, in []string, out string, dims [][]int) {
	ops := make([]*tensor.Tensor, len(in))
	subs := make([][]byte, len(in))
	for k := range in {
		f := bench.RandF64(tensor.Shape(dims[k]), uint64(k+1))
		t32 := tensor.New(tensor.F32, tensor.Shape(dims[k]))
		ds, ss := t32.Storage().F32(), f.Storage().F64()
		for i := range ds {
			ds[i] = float32(ss[i])
		}
		ops[k] = t32
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

// F32 hot-layer specs (memorizing-attention scores/output, TPA), the common inference dtype.
func BenchmarkEinsumF32_MemAttnScore(b *testing.B) { // td,tmd->tm
	benchEinsumF32(b, []string{"td", "tmd"}, "tm", [][]int{{128, 64}, {128, 32, 64}})
}
func BenchmarkEinsumF32_MemAttnOut(b *testing.B) { // tm,tmd->td
	benchEinsumF32(b, []string{"tm", "tmd"}, "td", [][]int{{128, 32}, {128, 32, 64}})
}
func BenchmarkEinsumF32_TPA(b *testing.B) { // trh,trd->thd
	benchEinsumF32(b, []string{"trh", "trd"}, "thd", [][]int{{128, 8, 16}, {128, 8, 64}})
}
