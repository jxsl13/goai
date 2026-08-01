package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMHAMaskedBackwardF32ProducesGradients is the regression floor for a silent F32 failure: the
// devirtualized fast path took its four OUTPUT buffers from f64Data, which is an INPUT view — for
// F32 it returns a detached widened copy. The kernel accumulated every gradient into buffers
// nobody read and returned four all-zero tensors. Nothing caught it because every test that
// touched this op built F64 tensors.
//
// The floor therefore asserts two things. The gradients must be non-zero, which is what actually
// failed; and they must agree with the F64 run to f32 precision, so a future edit cannot satisfy
// the first check with garbage. Both matter — a buffer full of noise is also non-zero.
func TestMHAMaskedBackwardF32ProducesGradients(t *testing.T) {
	const sq, sk, dm, heads = 6, 6, 8, 2
	build := func(dt tensor.Dtype) []*tensor.Tensor {
		mk := func(sh tensor.Shape, seed float64) *tensor.Tensor {
			x := tensor.New(dt, sh)
			n := x.Numel()
			for i := range n {
				v := 0.1 + seed + 0.01*float64(i%17) - 0.005*float64(i%7)
				switch len(sh) {
				case 2:
					x.SetF64(v, i/sh[1], i%sh[1])
				default:
					t.Fatalf("unexpected rank %d", len(sh))
				}
			}
			return x
		}
		return []*tensor.Tensor{
			mk(tensor.Shape{sq, dm}, 0.1), mk(tensor.Shape{sk, dm}, 0.2),
			mk(tensor.Shape{sk, dm}, 0.3), mk(tensor.Shape{sq, sk}, 0.0),
			mk(tensor.Shape{sq, dm}, 0.4),
		}
	}
	run := func(dt tensor.Dtype) []*tensor.Tensor {
		out, err := mhaMaskedBackwardKernel(backend.NewContext(), build(dt),
			backend.AttnAttrs{Heads: heads})
		if err != nil {
			t.Fatalf("%v: %v", dt, err)
		}
		return out
	}
	ref64, got32 := run(tensor.F64), run(tensor.F32)

	for oi, name := range []string{"dQ", "dK", "dV", "dMask"} {
		g, w := got32[oi], ref64[oi]
		rows, cols := g.Shape()[0], g.Shape()[1]
		nz, worst := 0, 0.0
		for r := range rows {
			for c := range cols {
				gv, wv := g.AtF64(r, c), w.AtF64(r, c)
				if gv != 0 {
					nz++
				}
				if d := math.Abs(gv - wv); d > worst {
					worst = d
				}
			}
		}
		if nz == 0 {
			t.Fatalf("%s: every F32 gradient is zero — the kernel accumulated into a detached "+
				"copy of the output storage", name)
		}
		if tol := 1e-5 * (1 + math.Abs(w.AtF64(0, 0))); worst > tol {
			t.Fatalf("%s: F32 gradients disagree with F64 by %g (tol %g)", name, worst, tol)
		}
	}
}
