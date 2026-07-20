package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// T870 follow-up: NewEmbeddingPadded fills the FULL embedding table [vocab,dim]
// via SetF64(Unravel(i)...) — an embedding table is one of T870's own textbook
// examples of a LARGE/hot tensor (vocab×dim easily reaches into the millions of
// elements). The fill is a pure gen()-shaped draw (rng.NormFloat64() per
// element, no dependency on i beyond ordering), exactly like KaimingNormal, so
// it reuses fillGen directly.
//
// No non-contiguous-view test accompanies this conversion: w is always a
// freshly allocated tensor.New(...) inside the function (never a caller-
// supplied view), so the fast path is unconditionally taken and the fallback
// branch can never be reached through this call site. fillGen's fallback
// correctness is already proven independently in init_fastpath_test.go.

// slowNewEmbeddingPadded is the verbatim pre-conversion per-element path.
func slowNewEmbeddingPadded(dtype tensor.Dtype, vocab, dim, padIdx int, seed uint64) *Embedding {
	w := tensor.New(dtype, tensor.Shape{vocab, dim})
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	for i := range w.Numel() {
		w.SetF64(rng.NormFloat64(), tensor.Unravel(i, w.Shape())...)
	}
	e := &Embedding{W: w, PadIdx: -1}
	if padIdx >= 0 {
		e.PadIdx = padIdx
		e.ZeroPadRow()
	}
	return e
}

func TestNewEmbeddingPaddedBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, vd := range [][2]int{{1, 1}, {5, 3}, {37, 11}, {257, 64}} {
			vocab, dim := vd[0], vd[1]
			fast := NewEmbeddingPadded(dt, vocab, dim, 0, 314)
			slow := slowNewEmbeddingPadded(dt, vocab, dim, 0, 314)
			n := fast.W.Numel()
			for i := range n {
				idx := tensor.Unravel(i, fast.W.Shape())
				a, b := fast.W.AtF64(idx...), slow.W.AtF64(idx...)
				if math.Float64bits(a) != math.Float64bits(b) {
					t.Fatalf("dt=%v vocab=%d dim=%d elem %d: fast %v != slow %v (not bit-identical)", dt, vocab, dim, i, a, b)
				}
			}
		}
	}
}

func BenchmarkNewEmbeddingPaddedFast(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewEmbeddingPadded(tensor.F32, 8192, 256, -1, uint64(i))
	}
}

func BenchmarkNewEmbeddingPaddedSlow(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = slowNewEmbeddingPadded(tensor.F32, 8192, 256, -1, uint64(i))
	}
}
