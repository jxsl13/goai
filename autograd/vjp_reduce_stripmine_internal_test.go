package autograd

import (
	"math"
	"math/rand"
	"testing"
)

// broadcastFillRef is broadcastVJP's inner loop EXACTLY as it stood before the
// strip-mine: one odometer tick per ELEMENT. Frozen bit-identity oracle.
//
// It exists because the existing autograd suite did NOT gate this path at all:
// breaking the strided gather (gs[of+j*s] -> gs[of]) and disabling the outer odometer
// entirely both left every test green. A rewrite of an ungated hot loop is exactly
// where a silent numerical regression lands, so the oracle is the gate.
func broadcastFillRef[T float32 | float64](ds, gs []T, xs, axStride []int, n int, scale T) {
	nd := len(xs)
	coord := make([]int, nd)
	of := 0
	for i := 0; i < n; i++ {
		ds[i] = gs[of] * scale
		for ax := nd - 1; ax >= 0; ax-- {
			coord[ax]++
			of += axStride[ax]
			if coord[ax] < xs[ax] {
				break
			}
			coord[ax] = 0
			of -= axStride[ax] * xs[ax]
		}
	}
}

// axStrideFor builds the source-offset stride vector broadcastVJP derives for a
// reduction over the given axes: a reduced axis contributes 0 (the gradient is shared
// across it), a kept axis contributes its row-major stride in the REDUCED shape.
func axStrideFor(xs []int, reduced map[int]bool) []int {
	nd := len(xs)
	out := make([]int, nd)
	stride := 1
	for ax := nd - 1; ax >= 0; ax-- {
		if reduced[ax] {
			out[ax] = 0
			continue
		}
		out[ax] = stride
		stride *= xs[ax]
	}
	return out
}

func numelOf(xs []int) int {
	n := 1
	for _, d := range xs {
		n *= d
	}
	return n
}

// TestBroadcastFillRowsMatchesOdometerExact holds the strip-mined fill bit-identical
// to the per-element odometer, tolerance 0, across ranks 1..4 and every subset of
// reduced axes — so both branches (zero innermost stride = constant fill, non-zero =
// strided gather) and both the single-row and many-row odometer cases are covered.
func TestBroadcastFillRowsMatchesOdometerExact(t *testing.T) {
	rng := rand.New(rand.NewSource(20260728))
	shapes := [][]int{
		{7}, {1}, {4, 5}, {5, 4}, {1, 6}, {6, 1}, {2, 3, 4}, {4, 3, 2}, {2, 1, 3}, {3, 2, 2, 2},
	}
	for _, xs := range shapes {
		nd, n := len(xs), numelOf(xs)
		for mask := 0; mask < 1<<nd; mask++ {
			reduced := map[int]bool{}
			for ax := range nd {
				if mask&(1<<ax) != 0 {
					reduced[ax] = true
				}
			}
			axStride := axStrideFor(xs, reduced)
			// Source extent: the product of the kept axes (what the reduction leaves).
			src := 1
			for ax := range nd {
				if !reduced[ax] {
					src *= xs[ax]
				}
			}
			gs := make([]float64, src)
			for i := range gs {
				gs[i] = rng.NormFloat64() * math.Pow(2, float64(rng.Intn(21)-10))
			}
			scale := rng.NormFloat64()

			got, want := make([]float64, n), make([]float64, n)
			broadcastFillRef(want, gs, xs, axStride, n, scale)
			if !broadcastFillRows(got, gs, xs, axStride, n, scale) {
				t.Fatalf("shape %v mask %d: strip-mine declined a shape it should handle", xs, mask)
			}
			for i := range want {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
					t.Fatalf("shape %v mask %d elem %d: got %v (%#x), want %v (%#x)",
						xs, mask, i, got[i], math.Float64bits(got[i]),
						want[i], math.Float64bits(want[i]))
				}
			}

			// F32 too: the caller converts scale ONCE before calling, and the fill must
			// round exactly as the odometer did.
			gs32 := make([]float32, src)
			for i := range gs32 {
				gs32[i] = float32(gs[i])
			}
			sc := float32(scale)
			got32, want32 := make([]float32, n), make([]float32, n)
			broadcastFillRef(want32, gs32, xs, axStride, n, sc)
			if !broadcastFillRows(got32, gs32, xs, axStride, n, sc) {
				t.Fatalf("shape %v mask %d: f32 strip-mine declined", xs, mask)
			}
			for i := range want32 {
				if math.Float32bits(got32[i]) != math.Float32bits(want32[i]) {
					t.Fatalf("f32 shape %v mask %d elem %d: got %v, want %v",
						xs, mask, i, got32[i], want32[i])
				}
			}
		}
	}
}

// TestBroadcastFillRowsDeclinesUnhandled pins the fallback contract: for a shape the
// strip-mine does not handle it must return false and leave the destination ALONE, so
// the caller's original loop still produces the answer. A version that returned true
// after writing nothing would silently zero every gradient.
func TestBroadcastFillRowsDeclinesUnhandled(t *testing.T) {
	ds := []float64{1, 2, 3}
	if broadcastFillRows(ds, []float64{9}, nil, nil, 3, 1) {
		t.Fatal("rank-0 shape: want decline")
	}
	if broadcastFillRows(ds, []float64{9}, []int{0}, []int{0}, 3, 1) {
		t.Fatal("zero innermost extent: want decline")
	}
	if broadcastFillRows(ds, []float64{9}, []int{2}, []int{0}, 3, 1) {
		t.Fatal("innermost extent not dividing n: want decline")
	}
	for i, v := range []float64{1, 2, 3} {
		if ds[i] != v {
			t.Fatalf("declined call mutated dst[%d] = %v, want %v", i, ds[i], v)
		}
	}
}
