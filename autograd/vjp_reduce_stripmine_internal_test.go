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

// bcastSumRef is bcastSumInto's inner loop EXACTLY as it stood before the strip-mine:
// one odometer tick per ELEMENT, accumulating into dst. Frozen bit-identity oracle.
func bcastSumRef[T float32 | float64](dst, src []T, gs []int, eff []int) {
	nd := len(gs)
	idx := make([]int, nd)
	dOff := 0
	for _, v := range src {
		dst[dOff] += v
		for d := nd - 1; d >= 0; d-- {
			idx[d]++
			dOff += eff[d]
			if idx[d] < gs[d] {
				break
			}
			idx[d] = 0
			dOff -= eff[d] * gs[d]
		}
	}
}

// TestBcastSumRowsMatchesOdometerExact holds the accumulating strip-mine bit-identical
// to the per-element odometer at tolerance 0, over every broadcast pattern of ranks
// 1..4 — so all three inner-run shapes are covered: unit stride, broadcast axis
// (eff 0, the scalar-accumulator branch) and strided scatter.
//
// Unlike its storing twin, this one accumulates, so bit-identity is CONDITIONAL: the
// summation order per destination must survive, and the scalar accumulator must keep
// the element type. F32 is the case that actually discriminates — a float64
// accumulator would be more accurate and would show up here as differing bits.
func TestBcastSumRowsMatchesOdometerExact(t *testing.T) {
	rng := rand.New(rand.NewSource(31337))
	shapes := [][]int{{6}, {3, 4}, {4, 3}, {1, 5}, {5, 1}, {2, 3, 4}, {3, 1, 2}, {2, 2, 3, 2}}
	for _, gs := range shapes {
		nd, n := len(gs), numelOf(gs)
		for mask := 0; mask < 1<<nd; mask++ {
			// Build eff exactly as bcastSumInto does: 0 on broadcast axes, the
			// destination's row-major stride on kept axes.
			eff := make([]int, nd)
			stride := 1
			dstLen := 1
			for ax := nd - 1; ax >= 0; ax-- {
				if mask&(1<<ax) != 0 { // broadcast this axis
					eff[ax] = 0
					continue
				}
				eff[ax] = stride
				stride *= gs[ax]
				dstLen *= gs[ax]
			}
			src := make([]float64, n)
			for i := range src {
				src[i] = rng.NormFloat64() * math.Pow(2, float64(rng.Intn(21)-10))
			}
			seed := make([]float64, dstLen)
			for i := range seed {
				seed[i] = rng.NormFloat64()
			}

			gotD, wantD := append([]float64(nil), seed...), append([]float64(nil), seed...)
			bcastSumRef(wantD, src, gs, eff)
			if !bcastSumRows(gotD, src, gs, eff) {
				t.Fatalf("shape %v mask %d: strip-mine declined", gs, mask)
			}
			for i := range wantD {
				if math.Float64bits(gotD[i]) != math.Float64bits(wantD[i]) {
					t.Fatalf("f64 shape %v mask %d dst[%d]: got %v (%#x), want %v (%#x)",
						gs, mask, i, gotD[i], math.Float64bits(gotD[i]),
						wantD[i], math.Float64bits(wantD[i]))
				}
			}

			src32 := make([]float32, n)
			for i := range src32 {
				src32[i] = float32(src[i])
			}
			seed32 := make([]float32, dstLen)
			for i := range seed32 {
				seed32[i] = float32(seed[i])
			}
			got32, want32 := append([]float32(nil), seed32...), append([]float32(nil), seed32...)
			bcastSumRef(want32, src32, gs, eff)
			if !bcastSumRows(got32, src32, gs, eff) {
				t.Fatalf("shape %v mask %d: f32 strip-mine declined", gs, mask)
			}
			for i := range want32 {
				if math.Float32bits(got32[i]) != math.Float32bits(want32[i]) {
					t.Fatalf("f32 shape %v mask %d dst[%d]: got %v, want %v — a widened "+
						"accumulator would look exactly like this", gs, mask, i, got32[i], want32[i])
				}
			}
		}
	}
}

// TestBcastSumRowsDeclinesUnhandled pins the fallback contract, and specifically that
// declining happens BEFORE any row is written. This function ACCUMULATES into dst, so
// a version that bailed mid-loop would leave the destination half-updated and the
// caller's fallback would then add those rows a second time.
func TestBcastSumRowsDeclinesUnhandled(t *testing.T) {
	dst := []float64{1, 2, 3, 4}
	src := []float64{5, 6, 7, 8}
	// An innermost stride that is neither 0 nor 1 cannot arise from a row-major
	// destination; it must be declined rather than handled by a branch no test reaches.
	if bcastSumRows(dst, src, []int{2, 2}, []int{0, 2}) {
		t.Fatal("innermost stride 2: want decline")
	}
	if bcastSumRows(dst, src, nil, nil) {
		t.Fatal("rank-0: want decline")
	}
	for i, v := range []float64{1, 2, 3, 4} {
		if dst[i] != v {
			t.Fatalf("declined call mutated dst[%d] = %v, want %v — a mid-loop bail would "+
				"double-count these rows when the caller's fallback runs", i, dst[i], v)
		}
	}
}
