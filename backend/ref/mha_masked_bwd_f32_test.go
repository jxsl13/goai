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

// TestMHAMaskedBackwardBufferReuseIsClean guards the pooled per-head contribution buffer. Recycling
// it is only safe because every slot is written before it is read; if that ever stops being true,
// the second call through a warm pool would see the first call's leftovers.
//
// The check is a round trip: run A, run B with different inputs so the pooled buffer comes back
// carrying B's values, then run A again and require the same bits. A stale slot would surface as a
// difference in exactly one of the four gradients.
func TestMHAMaskedBackwardBufferReuseIsClean(t *testing.T) {
	const sq, sk, dm, heads = 8, 8, 8, 2
	build := func(seed float64) []*tensor.Tensor {
		mk := func(sh tensor.Shape, off float64) *tensor.Tensor {
			x := tensor.New(tensor.F64, sh)
			n := x.Numel()
			for i := range n {
				x.SetF64(math.Sin(float64(i)*0.31+seed+off)*0.7, i/sh[1], i%sh[1])
			}
			return x
		}
		return []*tensor.Tensor{
			mk(tensor.Shape{sq, dm}, 0.1), mk(tensor.Shape{sk, dm}, 0.2),
			mk(tensor.Shape{sk, dm}, 0.3), mk(tensor.Shape{sq, sk}, 0.4),
			mk(tensor.Shape{sq, dm}, 0.5),
		}
	}
	run := func(seed float64) []*tensor.Tensor {
		out, err := mhaMaskedBackwardKernel(backend.NewContext(), build(seed),
			backend.AttnAttrs{Heads: heads})
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	first := run(0)
	run(11.7) // different inputs; the pooled buffer comes back holding these
	again := run(0)

	for oi, name := range []string{"dQ", "dK", "dV", "dMask"} {
		a, b := first[oi], again[oi]
		for r := range a.Shape()[0] {
			for c := range a.Shape()[1] {
				if math.Float64bits(a.AtF64(r, c)) != math.Float64bits(b.AtF64(r, c)) {
					t.Fatalf("%s[%d,%d]: %v then %v — a pooled buffer carried state between calls",
						name, r, c, a.AtF64(r, c), b.AtF64(r, c))
				}
			}
		}
	}
}
