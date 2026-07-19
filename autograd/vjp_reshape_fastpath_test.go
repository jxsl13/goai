package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// slowReshapeVJP is the verbatim pre-fast-path per-element reshape VJP, kept as
// the bit-identity oracle: the contiguous copy() fast path must reproduce it
// exactly, and the non-contiguous fallback IS this loop.
func slowReshapeVJP(x, g *tensor.Tensor) *tensor.Tensor {
	dx := tensor.New(x.Dtype(), x.Shape())
	xs, gs := x.Shape(), g.Shape()
	for i := range x.Numel() {
		dx.SetF64(g.AtF64(tensor.Unravel(i, gs)...), tensor.Unravel(i, xs)...)
	}
	return dx
}

// fillSeq writes a distinct value per element (so a scrambled copy is caught).
func fillSeq(t *tensor.Tensor, start float64) {
	sh := t.Shape()
	for i := range t.Numel() {
		t.SetF64(start+float64(i), tensor.Unravel(i, sh)...)
	}
}

func reshapeRule(t *testing.T) VJP {
	t.Helper()
	rule, ok := vjps[backend.OpReshape]
	if !ok {
		t.Fatal("no VJP registered for OpReshape")
	}
	return rule
}

// The contiguous fast path must be bit-identical to the per-element oracle for
// every fast-path dtype, across ranks (the gradient must not be scrambled and
// must survive the dtype round-trip exactly).
func TestReshapeVJPFastPathBitIdentical(t *testing.T) {
	rule := reshapeRule(t)
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, sh := range []tensor.Shape{{1}, {7}, {4, 5}, {2, 3, 4}, {128, 96}} {
			// reshape maps xShape -> some gShape with the same element count.
			x := tensor.New(dt, sh) // VJP reads only x's shape/dtype, not its values
			g := tensor.New(dt, tensor.Shape{sh.Numel()})
			fillSeq(g, 0.25) // 0.25 is exact in every float dtype; +i stays exact for these sizes
			gins, err := rule(nil, []*tensor.Tensor{x}, nil, nil, g)
			if err != nil {
				t.Fatalf("dt=%v sh=%v: %v", dt, sh, err)
			}
			fast, slow := gins[0], slowReshapeVJP(x, g)
			n := fast.Numel()
			for i := range n {
				a := fast.AtF64(tensor.Unravel(i, fast.Shape())...)
				b := slow.AtF64(tensor.Unravel(i, slow.Shape())...)
				if math.Float64bits(a) != math.Float64bits(b) {
					t.Fatalf("dt=%v sh=%v elem %d: fast %v != slow %v (not bit-identical)", dt, sh, i, a, b)
				}
			}
			// The output must carry x's shape (reshaped back), not g's.
			if fast.Shape().String() != sh.String() {
				t.Fatalf("dt=%v: output shape %v, want %v", dt, fast.Shape(), sh)
			}
		}
	}
}

// A non-contiguous cotangent must route to the fallback and stay bit-identical to
// the oracle (VJPs frequently receive non-contiguous cotangents; a wrong stride
// here corrupts training silently).
func TestReshapeVJPNonContiguousCotangent(t *testing.T) {
	rule := reshapeRule(t)
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		base := tensor.New(dt, tensor.Shape{4, 5})
		fillSeq(base, 1.0)
		g, err := base.Transpose(0, 1) // [5,4], non-contiguous
		if err != nil {
			t.Fatal(err)
		}
		if g.IsContiguous() {
			t.Fatal("expected a non-contiguous cotangent")
		}
		x := tensor.New(dt, tensor.Shape{2, 10}) // same 20 elements, reshaped
		gins, err := rule(nil, []*tensor.Tensor{x}, nil, nil, g)
		if err != nil {
			t.Fatal(err)
		}
		fast, slow := gins[0], slowReshapeVJP(x, g)
		for i := range fast.Numel() {
			a := fast.AtF64(tensor.Unravel(i, fast.Shape())...)
			b := slow.AtF64(tensor.Unravel(i, slow.Shape())...)
			if math.Float64bits(a) != math.Float64bits(b) {
				t.Fatalf("dt=%v elem %d: fast %v != slow %v (non-contiguous fallback not bit-identical)", dt, i, a, b)
			}
		}
	}
}

// The half-float dtypes fall through to the general path; confirm the guard does
// not wrongly engage the typed copy and the result stays bit-identical.
func TestReshapeVJPHalfFloatFallThrough(t *testing.T) {
	rule := reshapeRule(t)
	for _, dt := range []tensor.Dtype{tensor.F16, tensor.BF16} {
		x := tensor.New(dt, tensor.Shape{3, 4})
		g := tensor.New(dt, tensor.Shape{12})
		for i := range g.Numel() {
			g.SetF64(float64(i-6), i) // small integers: exact in F16/BF16
		}
		gins, err := rule(nil, []*tensor.Tensor{x}, nil, nil, g)
		if err != nil {
			t.Fatal(err)
		}
		fast, slow := gins[0], slowReshapeVJP(x, g)
		for i := range fast.Numel() {
			a := fast.AtF64(tensor.Unravel(i, fast.Shape())...)
			b := slow.AtF64(tensor.Unravel(i, slow.Shape())...)
			if math.Float64bits(a) != math.Float64bits(b) {
				t.Fatalf("dt=%v elem %d: %v != %v", dt, i, a, b)
			}
		}
	}
}

func benchReshapeVJP(b *testing.B, dt tensor.Dtype) {
	rule := vjps[backend.OpReshape]
	x := tensor.New(dt, tensor.Shape{512, 512})
	g := tensor.New(dt, tensor.Shape{512 * 512})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := rule(nil, []*tensor.Tensor{x}, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func benchReshapeVJPSlow(b *testing.B, dt tensor.Dtype) {
	x := tensor.New(dt, tensor.Shape{512, 512})
	g := tensor.New(dt, tensor.Shape{512 * 512})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = slowReshapeVJP(x, g)
	}
}

func BenchmarkReshapeVJPFastF32(b *testing.B) { benchReshapeVJP(b, tensor.F32) }
func BenchmarkReshapeVJPSlowF32(b *testing.B) { benchReshapeVJPSlow(b, tensor.F32) }
func BenchmarkReshapeVJPFastF64(b *testing.B) { benchReshapeVJP(b, tensor.F64) }
func BenchmarkReshapeVJPSlowF64(b *testing.B) { benchReshapeVJPSlow(b, tensor.F64) }
