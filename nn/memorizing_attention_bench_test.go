package nn

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMemAddSegment measures folding a key/value segment into the Memorizing
// Transformer's memory bank (cap = one segment, so it reaches the add-then-evict steady
// state). Each row still needs its own retained backing slice, but the detaching copy
// takes the typed contiguous path instead of m.dim per-element AtF64 dispatches per row.
func BenchmarkMemAddSegment(b *testing.B) {
	const s, dim = 128, 512
	mem := newMemMemory(dim, s)
	k := tensor.New(tensor.F64, tensor.Shape{s, dim})
	v := tensor.New(tensor.F64, tensor.Shape{s, dim})
	kf, vf := k.Storage().F64(), v.Storage().F64()
	for i := range kf {
		kf[i] = 0.01 * float64(i%17)
		vf[i] = 0.01 * float64(i%13)
	}
	b.ReportAllocs()
	for b.Loop() {
		if err := mem.AddSegment(k, v); err != nil {
			b.Fatal(err)
		}
	}
}
