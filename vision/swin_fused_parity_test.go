package vision_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// tapeRecorder is a recorder that records nothing. Its only job is to be non-nil, which is what
// selects the dispatched windowed-attention path — the fused arm runs only when no tape is active.
type tapeRecorder struct{ n int }

func (r *tapeRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {
	r.n++
}

// TestSwinFusedWindowAttentionIsBitIdentical is the gate on the fused inference arm. The arm
// replaces seven of about ten dispatches per (window, head) with index arithmetic, and it claims
// the result is bit-identical, not merely close — so the assertion is exact equality of every
// element, with no tolerance to hide behind.
//
// Both blocks are exercised: index 0 is W-MSA, index 1 is SW-MSA and additionally carries the
// cross-region mask add, which is one of the folded operations.
//
// The two arms are selected by the recorder alone, which is also what pins the gate itself: if the
// recorder ever stopped selecting the dispatched path, this test would compare the fused arm
// against itself and pass for free. The recorder's own call count is asserted to be non-zero for
// exactly that reason.
func TestSwinFusedWindowAttentionIsBitIdentical(t *testing.T) {
	const B, C, size, classes, embedC, grid = 2, 3, 32, 10, 96, 8
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		m, err := vision.NewSwin(dt, size, 4, 4, embedC, []int{2, 2}, []int{3, 6}, classes, 7,
			vision.WithSwinRelativeBias(true), vision.WithSwinChannels(C))
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(11))
		x := tensor.New(dt, tensor.Shape{B * grid * grid, embedC})
		for i := range B * grid * grid * embedC {
			x.SetF64(rng.NormFloat64(), i/embedC, i%embedC)
		}
		for bi, blk := range m.Stages[0] {
			fused, err := blk.Forward(backend.NewContext(), x, B*grid, grid)
			if err != nil {
				t.Fatal(err)
			}
			rec := &tapeRecorder{}
			dispatched, err := blk.Forward(backend.NewContext().WithRecorder(rec), x, B*grid, grid)
			if err != nil {
				t.Fatal(err)
			}
			if rec.n == 0 {
				t.Fatalf("%v block %d: the recorder saw no ops, so both arms were the same one", dt, bi)
			}
			n := fused.Shape()[0] * fused.Shape()[1]
			for i := range n {
				r, c := i/embedC, i%embedC
				f, d := fused.AtF64(r, c), dispatched.AtF64(r, c)
				if math.Float64bits(f) != math.Float64bits(d) {
					t.Fatalf("%v block %d [%d,%d]: fused %v, dispatched %v — not bit-identical",
						dt, bi, r, c, f, d)
				}
			}
		}
	}
}
