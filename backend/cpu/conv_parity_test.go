package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// cpu registers OpConv2D unconditionally for both dtypes, so cpu/conv.go is the
// production convolution and ref is its reference. PS6001 flags it as a dual-path
// kernel — a devirtualized fast path plus a generic fallback, asserting bit-identity
// with nothing checking it.
//
// Guarded by parity, as MHA was. cpu and ref were MEASURED to agree bit-for-bit in
// both dtypes first (0 of 336 elements differing); bitwise agreement is never
// inferred from algebraic equivalence, per the flashattn collapse that failed in 27
// of 48 (R-01KYM5J5Z8EK9). Registration was also checked for dtype-partial and
// build-gated forms before calling cpu production here, since cross-entropy looked
// production and was not (R-01KYM7SSDRETE).
func TestCPUConv2DBitIdenticalToRef(t *testing.T) {
	rb, ok := backend.Get(backend.Ref)
	if !ok {
		t.Skip("ref backend unavailable")
	}
	cb, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend unavailable")
	}
	if _, has := cb.Kernel(backend.OpConv2D, tensor.F64); !has {
		t.Skip("cpu registers no OpConv2D kernel in this build; parity would be ref-vs-ref")
	}
	cases := []struct{ n, c, h, w, f, kh, kw, stride, pad int }{
		{1, 1, 3, 3, 1, 1, 1, 1, 0},
		{2, 3, 7, 6, 4, 3, 3, 1, 1},
		{1, 2, 8, 8, 3, 3, 3, 2, 1},
		{2, 1, 5, 9, 2, 2, 4, 1, 0},
		{1, 4, 6, 6, 2, 5, 5, 1, 2},
	}
	for _, c := range cases {
		for _, f32 := range []bool{false, true} {
			mk := func(sh tensor.Shape, s uint64) *tensor.Tensor {
				if f32 {
					return bench.RandF32(sh, s)
				}
				return bench.RandF64(sh, s)
			}
			seed := uint64(c.h*100 + c.w)
			in := []*tensor.Tensor{
				mk(tensor.Shape{c.n, c.c, c.h, c.w}, seed),
				mk(tensor.Shape{c.f, c.c, c.kh, c.kw}, seed+1),
			}
			at := backend.ConvAttrs{Stride: c.stride, Pad: c.pad}
			want, err := backend.Execute(backend.NewContext().WithBackend(rb), backend.OpConv2D, in, at)
			if err != nil {
				t.Fatal(err)
			}
			got, err := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpConv2D, in, at)
			if err != nil {
				t.Fatal(err)
			}
			for i := range want[0].Numel() {
				co := tensor.Unravel(i, want[0].Shape())
				g, w := got[0].AtF64(co...), want[0].AtF64(co...)
				if math.Float64bits(g) != math.Float64bits(w) {
					t.Fatalf("%+v f32=%v elem %d: cpu %v != ref %v", c, f32, i, g, w)
				}
			}
		}
	}
}
