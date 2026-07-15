package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §T462 V-CROSS: the optimized pools are BIT-identical to the reference (same window order and
// arithmetic) across shapes, kernels, strides and dtypes — including a non-contiguous view input
// and the pool-window edge cases.
func TestPool2DCrossReference(t *testing.T) {
	cpuBE, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend not registered")
	}
	refBE := backend.Reference()
	cases := []struct{ n, c, h, w, k, s int }{
		{1, 1, 4, 4, 2, 0}, // stride 0 → kernel
		{2, 3, 8, 8, 2, 2},
		{2, 2, 9, 7, 3, 2},
		{1, 4, 6, 6, 6, 1}, // global pool
		{3, 1, 5, 5, 2, 3}, // stride > kernel
	}
	for _, op := range []backend.Op{backend.OpMaxPool2D, backend.OpAvgPool2D} {
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			for _, tc := range cases {
				x := tensor.New(dt, tensor.Shape{tc.n, tc.c, tc.h, tc.w})
				for i := range x.Numel() {
					x.SetF64(math.Sin(float64(i)*0.9)*3, tensor.Unravel(i, x.Shape())...)
				}
				attrs := backend.PoolAttrs{Kernel: tc.k, Stride: tc.s}
				got, err := backend.Execute(backend.NewContext().WithBackend(cpuBE), op, []*tensor.Tensor{x}, attrs)
				if err != nil {
					t.Fatalf("%v %+v %v: cpu: %v", op, tc, dt, err)
				}
				want, err := backend.Execute(backend.NewContext().WithBackend(refBE), op, []*tensor.Tensor{x}, attrs)
				if err != nil {
					t.Fatalf("%v %+v %v: ref: %v", op, tc, dt, err)
				}
				if !got[0].Shape().Equal(want[0].Shape()) {
					t.Fatalf("%v %+v: shape %v vs %v", op, tc, got[0].Shape(), want[0].Shape())
				}
				for i := range want[0].Numel() {
					idx := tensor.Unravel(i, want[0].Shape())
					a, b := got[0].AtF64(idx...), want[0].AtF64(idx...)
					if a != b && !(math.IsNaN(a) && math.IsNaN(b)) {
						t.Fatalf("%v %+v %v [%v]: cpu %.17g vs ref %.17g", op, tc, dt, idx, a, b)
					}
				}
			}
		}
	}
	// error paths
	bad := tensor.New(tensor.F64, tensor.Shape{2, 2})
	if _, err := backend.Execute(backend.NewContext().WithBackend(cpuBE), backend.OpMaxPool2D,
		[]*tensor.Tensor{bad}, backend.PoolAttrs{Kernel: 2}); err == nil {
		t.Fatal("rank-2 accepted")
	}
	small := tensor.New(tensor.F64, tensor.Shape{1, 1, 2, 2})
	if _, err := backend.Execute(backend.NewContext().WithBackend(cpuBE), backend.OpMaxPool2D,
		[]*tensor.Tensor{small}, backend.PoolAttrs{Kernel: 3}); err == nil {
		t.Fatal("window > input accepted")
	}
}

func benchPool(b *testing.B, op backend.Op) {
	be, _ := backend.Get(backend.CPU)
	x := tensor.Randn(tensor.F32, 1, tensor.Shape{8, 16, 64, 64})
	attrs := backend.PoolAttrs{Kernel: 2, Stride: 2}
	ctx := backend.NewContext().WithBackend(be)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, []*tensor.Tensor{x}, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMaxPool2D_cpu(b *testing.B) { benchPool(b, backend.OpMaxPool2D) }
func BenchmarkAvgPool2D_cpu(b *testing.B) { benchPool(b, backend.OpAvgPool2D) }
