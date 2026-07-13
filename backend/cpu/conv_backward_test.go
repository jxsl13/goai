package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §T460 V-CROSS: the GEMM-based conv2d backward matches the reference for every gradient across
// shapes, strides, padding and dtypes. dW/dX sums are reassociated vs the reference loop nest, so
// parity is tolerance-based (f64 1e-12 rel, f32 1e-4 rel); dBias uses the same order (near-exact).
func TestConv2DBackwardCrossReference(t *testing.T) {
	cpuBE, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend not registered")
	}
	refBE := backend.Reference()
	cases := []struct {
		n, c, h, w, f, kh, kw, s, p int
	}{
		{1, 1, 5, 5, 1, 3, 3, 1, 0},
		{2, 3, 8, 8, 4, 3, 3, 1, 1},
		{2, 2, 9, 7, 3, 3, 2, 2, 1},
		{1, 4, 6, 6, 8, 5, 5, 1, 2},
		{3, 1, 7, 7, 2, 2, 2, 2, 0},
	}
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		tol := 1e-12
		if dt == tensor.F32 {
			tol = 1e-4
		}
		for _, tc := range cases {
			x := tensor.New(dt, tensor.Shape{tc.n, tc.c, tc.h, tc.w})
			w := tensor.New(dt, tensor.Shape{tc.f, tc.c, tc.kh, tc.kw})
			ho := (tc.h+2*tc.p-tc.kh)/tc.s + 1
			wo := (tc.w+2*tc.p-tc.kw)/tc.s + 1
			g := tensor.New(dt, tensor.Shape{tc.n, tc.f, ho, wo})
			for i := range x.Numel() {
				x.SetF64(math.Sin(float64(i)*0.7), tensor.Unravel(i, x.Shape())...)
			}
			for i := range w.Numel() {
				w.SetF64(math.Cos(float64(i)*1.3), tensor.Unravel(i, w.Shape())...)
			}
			for i := range g.Numel() {
				g.SetF64(math.Sin(float64(i)*0.3+1), tensor.Unravel(i, g.Shape())...)
			}
			attrs := backend.ConvAttrs{Stride: tc.s, Pad: tc.p}
			got, err := backend.Execute(backend.NewContext().WithBackend(cpuBE), backend.OpConv2DBackward,
				[]*tensor.Tensor{x, w, g}, attrs)
			if err != nil {
				t.Fatalf("%+v %v: cpu: %v", tc, dt, err)
			}
			want, err := backend.Execute(backend.NewContext().WithBackend(refBE), backend.OpConv2DBackward,
				[]*tensor.Tensor{x, w, g}, attrs)
			if err != nil {
				t.Fatalf("%+v %v: ref: %v", tc, dt, err)
			}
			names := []string{"dX", "dW", "dBias"}
			for oi := range want {
				a, b := got[oi], want[oi]
				if !a.Shape().Equal(b.Shape()) {
					t.Fatalf("%+v %v %s: shape %v vs %v", tc, dt, names[oi], a.Shape(), b.Shape())
				}
				for i := range b.Numel() {
					idx := tensor.Unravel(i, b.Shape())
					av, bv := a.AtF64(idx...), b.AtF64(idx...)
					if math.IsNaN(av) || math.Abs(av-bv) > tol*math.Max(1, math.Abs(bv)) {
						t.Fatalf("%+v %v %s[%v]: cpu %.17g vs ref %.17g", tc, dt, names[oi], idx, av, bv)
					}
				}
			}
		}
	}
}
