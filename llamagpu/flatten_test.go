package llamagpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// --- Pre-change reference implementations (the §V22 baseline + the bit-identity oracle) ---
//
// These are verbatim copies of the per-element AtF64 loops that flat1D / flat2D / flat2DT /
// embedRow replaced. They stay in the test file for two reasons: they are the "before" arm of
// the benchmark, and they are the golden oracle the fast paths are asserted BIT-identical
// against (a transpose is a pure permutation, so equality is exact, not approximate).

func flat2DTOld(t *tensor.Tensor) []float32 {
	rows, cols := t.Shape()[0], t.Shape()[1]
	o := make([]float32, rows*cols)
	for j := range rows {
		for i := range cols {
			o[i*rows+j] = float32(t.AtF64(j, i))
		}
	}
	return o
}

func flat2DOld(t *tensor.Tensor) []float32 {
	r, c := t.Shape()[0], t.Shape()[1]
	o := make([]float32, r*c)
	for i := range r {
		for j := range c {
			o[i*c+j] = float32(t.AtF64(i, j))
		}
	}
	return o
}

func flat1DOld(t *tensor.Tensor) []float32 {
	n := t.Shape()[0]
	o := make([]float32, n)
	for i := range n {
		o[i] = float32(t.AtF64(i))
	}
	return o
}

func embedRowOld(table *tensor.Tensor, row, cols int) []float32 {
	o := make([]float32, cols)
	for j := range cols {
		o[j] = float32(table.AtF64(row, j))
	}
	return o
}

// fillSeq writes a deterministic, dtype-representable value pattern into t. The values are chosen
// to survive an f16 round-trip exactly (small halves and integers), so a bit-identity assertion is
// meaningful for every dtype rather than accidentally passing on flushed-to-zero data.
func fillSeq(t *tensor.Tensor) {
	sh := t.Shape()
	v := func(i int) float64 { return float64(i%97)*0.5 - 24 }
	if len(sh) == 1 {
		for i := range sh[0] {
			t.SetF64(v(i), i)
		}
		return
	}
	for i := range sh[0] {
		for j := range sh[1] {
			t.SetF64(v(i*sh[1]+j), i, j)
		}
	}
}

func newFilled(dt tensor.Dtype, shape tensor.Shape) *tensor.Tensor {
	t := tensor.New(dt, shape)
	fillSeq(t)
	return t
}

func bitEqual(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		g, w := math.Float32bits(got[i]), math.Float32bits(want[i])
		if g != w {
			t.Fatalf("%s: element %d = %v (bits %#x), want %v (bits %#x)", name, i, got[i], g, want[i], w)
		}
	}
}

// TestFlattenHelpersBitIdentical pins the bulk-slice flatten helpers to the per-element AtF64
// loops they replaced: identical float32 BIT patterns on every shape/dtype combination, including
// the degenerate 1×N / N×1 / 1×1 shapes and the strided (transposed-view) input case.
func TestFlattenHelpersBitIdentical(t *testing.T) {
	shapes := []tensor.Shape{{1, 1}, {1, 37}, {37, 1}, {4, 4}, {3, 512}, {512, 3}, {64, 65}, {129, 33}}
	dtypes := []tensor.Dtype{tensor.F32, tensor.F64, tensor.F16, tensor.BF16}
	for _, dt := range dtypes {
		for _, sh := range shapes {
			x := newFilled(dt, sh)
			bitEqual(t, "flat2DT", flat2DT(x), flat2DTOld(x))
			bitEqual(t, "flat2D", flat2D(x), flat2DOld(x))
			for r := range sh[0] {
				bitEqual(t, "embedRow", embedRow(x, r, sh[1]), embedRowOld(x, r, sh[1]))
			}
			v := newFilled(dt, tensor.Shape{sh[0] * sh[1]})
			bitEqual(t, "flat1D", flat1D(v), flat1DOld(v))
		}
	}
}

// TestFlattenHelpersStridedInput covers non-contiguous inputs: a transposed VIEW handed to the
// helpers must still flatten by logical index, not by storage order.
func TestFlattenHelpersStridedInput(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64, tensor.F16, tensor.BF16} {
		base := newFilled(dt, tensor.Shape{17, 23})
		view, err := base.Transpose(0, 1)
		if err != nil {
			t.Fatalf("Transpose: %v", err)
		}
		bitEqual(t, "flat2DT/view", flat2DT(view), flat2DTOld(view))
		bitEqual(t, "flat2D/view", flat2D(view), flat2DOld(view))
		bitEqual(t, "embedRow/view", embedRow(view, 5, 17), embedRowOld(view, 5, 17))
	}
}

// --- §V22 benchmarks: before (per-element AtF64) vs after (cache-tiled bulk-slice walk) ---
//
// flat2DT is a WEIGHT-UPLOAD-time cost: it runs once per tied lm_head / Mamba projection when a
// Decoder is built, not per token. The shape that motivated the fix is a tied lm_head, which the
// benchmark takes at [vocab=32000, dim=2048] — 65M elements, a 256 MB source and a 256 MB result.
//
// Gemma's [256000, 2048] is 8× that (524M elements, 2 GB each side) and is deliberately NOT the
// default benchmark shape: a 4 GB live working set puts the machine into memory-compression /
// swap, which flattens every arm toward the pagefault rate and understates the difference. Measure
// it with -benchtime 3x on a machine with headroom if you want the real Gemma figure.
//
// The f64 arm covers the widening path (Cast's F64→F32 fast path vs the per-element AtF64→float32
// round-trip), at the smaller shape a unit-scale model actually builds.

func benchFlat2DT(b *testing.B, dt tensor.Dtype, vocab, dim int, old bool) {
	b.Helper()
	x := tensor.New(dt, tensor.Shape{vocab, dim})
	b.SetBytes(int64(vocab) * int64(dim) * 4)
	b.ResetTimer()
	for b.Loop() {
		if old {
			_ = flat2DTOld(x)
		} else {
			_ = flat2DT(x)
		}
	}
}

func BenchmarkFlat2DTHeadF32Old(b *testing.B)  { benchFlat2DT(b, tensor.F32, 32000, 2048, true) }
func BenchmarkFlat2DTHeadF32New(b *testing.B)  { benchFlat2DT(b, tensor.F32, 32000, 2048, false) }
func BenchmarkFlat2DTSmallF64Old(b *testing.B) { benchFlat2DT(b, tensor.F64, 4096, 512, true) }
func BenchmarkFlat2DTSmallF64New(b *testing.B) { benchFlat2DT(b, tensor.F64, 4096, 512, false) }

func BenchmarkEmbedRowOld(b *testing.B) {
	x := tensor.New(tensor.F32, tensor.Shape{32000, 2048})
	for b.Loop() {
		_ = embedRowOld(x, 12345, 2048)
	}
}

func BenchmarkEmbedRowNew(b *testing.B) {
	x := tensor.New(tensor.F32, tensor.Shape{32000, 2048})
	for b.Loop() {
		_ = embedRow(x, 12345, 2048)
	}
}
