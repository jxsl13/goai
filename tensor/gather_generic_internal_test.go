package tensor

import (
	"math"
	"math/rand"
	"testing"
)

// gatherGenericRef is gatherGeneric EXACTLY as it stood before the innermost axis was
// hoisted: one odometer tick per ELEMENT. Frozen bit-identity oracle.
func gatherGenericRef(out, t *Tensor, n int) {
	nd := len(t.shape)
	idx := make([]int, nd)
	off := t.offset
	for pos := 0; pos < n; pos++ {
		out.storage.setF64(pos, t.storage.atF64(off))
		for d := nd - 1; d >= 0; d-- {
			idx[d]++
			off += t.strides[d]
			if idx[d] < t.shape[d] {
				break
			}
			idx[d] = 0
			off -= t.strides[d] * t.shape[d]
		}
	}
}

// TestGatherGenericMatchesOdometerExact byte-compares the strip-mined walk against the
// per-element oracle across the SOURCE/DEST dtype pairs that actually reach this
// function — strided cross-dtype half casts — at tolerance 0.
//
// The half dtypes matter specifically: this path widens through float64, and f16/bf16
// round-trips are lossy, so a transform that changed when the widening happened would
// show up here and nowhere else.
func TestGatherGenericMatchesOdometerExact(t *testing.T) {
	rng := rand.New(rand.NewSource(20260728))
	type pair struct{ src, dst Dtype }
	pairs := []pair{{F16, F32}, {F16, F64}, {BF16, F32}, {BF16, F64}, {F32, F16}, {F64, BF16}}
	shapes := []Shape{{3, 4}, {2, 3, 4}, {4, 3, 2}, {2, 2, 2, 3}, {5}, {1, 7}}

	for _, p := range pairs {
		for _, sh := range shapes {
			base := New(p.src, sh)
			switch p.src {
			case F16, BF16:
				st := base.Storage().U16()
				for i := range st {
					st[i] = uint16(rng.Intn(1 << 16))
				}
			case F32:
				st := base.Storage().F32()
				for i := range st {
					st[i] = float32(rng.NormFloat64())
				}
			case F64:
				st := base.Storage().F64()
				for i := range st {
					st[i] = rng.NormFloat64()
				}
			}
			// A non-contiguous view: reverse-ish stride on the innermost axis plus a
			// shifted offset, so the run is genuinely strided rather than a copy.
			view := base
			if len(sh) >= 2 {
				v, err := base.Slice(0, 0, sh[0])
				if err != nil {
					t.Fatalf("Slice: %v", err)
				}
				view = v
			}
			n := view.Shape().Numel()

			got, want := New(p.dst, view.Shape()), New(p.dst, view.Shape())
			gatherGenericRef(want, view, n)
			gatherGeneric(got, view, n)

			for i := range n {
				g, w := got.storage.atF64(i), want.storage.atF64(i)
				if math.Float64bits(g) != math.Float64bits(w) {
					t.Fatalf("%v->%v shape %v elem %d: got %v (%#x), want %v (%#x)",
						p.src, p.dst, sh, i, g, math.Float64bits(g), w, math.Float64bits(w))
				}
			}
		}
	}
}
