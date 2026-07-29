package nlp

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func benchGatherSetup() (*tensor.Tensor, []int) {
	const n, d = 4096, 1024
	t := tensor.New(tensor.F32, tensor.Shape{n, d})
	s := t.Storage().F32()
	rng := rand.New(rand.NewPCG(1, 2))
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	idx := make([]int, 1024) // evict to 1024 kept rows
	for i := range idx {
		idx[i] = (i * 3) % n
	}
	return t, idx
}

func BenchmarkGatherRowsTyped(b *testing.B) {
	t, idx := benchGatherSetup()
	b.ResetTimer()
	for range b.N {
		gsink = GatherRows(t, idx)
	}
}

func BenchmarkGatherRowsPerElement(b *testing.B) {
	t, idx := benchGatherSetup()
	d := t.Shape()[1]
	b.ResetTimer()
	for range b.N {
		out := tensor.New(t.Dtype(), tensor.Shape{len(idx), d})
		for r, src := range idx {
			for j := range d {
				out.SetF64(t.AtF64(src, j), r, j)
			}
		}
		gsink = out
	}
}

var gsink *tensor.Tensor

func TestGatherRowsTypedByteIdentical(t *testing.T) {
	tn, idx := benchGatherSetup()
	d := tn.Shape()[1]
	got := GatherRows(tn, idx)
	for r, src := range idx {
		for j := 0; j < d; j++ {
			if got.AtF64(r, j) != tn.AtF64(src, j) {
				t.Fatalf("GatherRows[%d,%d] != src row %d col %d", r, j, src, j)
			}
		}
	}
}
