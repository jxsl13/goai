package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// TestReduceAxisCPUMatchesRefBitExact pins the CPU trailing-contiguous axis-reduction fast path
// (devirtualised per-segment closure, parallel) against the reference kernel, BIT-FOR-BIT across
// ops/dtypes/axes/keepdims — including special values where Max/Min IEEE tie rules matter.
func TestReduceAxisCPUMatchesRefBitExact(t *testing.T) {
	cpuBE, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend not registered")
	}
	refBE := backend.Reference()
	ops := []backend.Op{backend.OpSum, backend.OpMean, backend.OpMax, backend.OpMin, backend.OpProd}
	type tc struct {
		shape tensor.Shape
		axes  []int
	}
	cases := []tc{
		{tensor.Shape{64, 128}, []int{1}},       // trailing 2D
		{tensor.Shape{64, 128}, []int{-1}},      // negative axis == trailing
		{tensor.Shape{8, 16, 32}, []int{2}},     // trailing of 3D
		{tensor.Shape{8, 16, 32}, []int{1, 2}},  // trailing suffix (multi-axis)
		{tensor.Shape{7, 5}, []int{1}},          // ragged, count not multiple of 4
		{tensor.Shape{64, 128}, []int{0}},       // LEADING axis of 2D (prefix fast path)
		{tensor.Shape{16, 8, 32}, []int{0}},     // leading axis of 3D → segStride 256
		{tensor.Shape{8, 16, 32}, []int{0, 1}},  // leading PREFIX (multi-axis) → segStride 32
		{tensor.Shape{7, 5}, []int{0}},          // ragged leading, segStride 5 (not mult of 4)
		{tensor.Shape{5, 6, 7}, []int{0, 1, 2}}, // explicit all-axes
	}
	run := func(be backend.Backend, op backend.Op, x *tensor.Tensor, axes []int, keep bool) *tensor.Tensor {
		ctx := backend.NewContext().WithBackend(be)
		out, err := backend.Execute(ctx, op, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: axes, KeepDims: keep})
		if err != nil {
			t.Fatalf("%v: %v", op, err)
		}
		return out[0]
	}
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, c := range cases {
			for _, keep := range []bool{false, true} {
				var x *tensor.Tensor
				if dt == tensor.F64 {
					x = bench.RandF64(c.shape, 7)
				} else {
					x = bench.RandF32(c.shape, 7)
				}
				for _, op := range ops {
					g := run(cpuBE, op, x, c.axes, keep)
					r := run(refBE, op, x, c.axes, keep)
					if !g.Shape().Equal(r.Shape()) {
						t.Fatalf("%v %v shape=%v axes=%v keep=%v: cpu shape %v != ref %v", dt, op, c.shape, c.axes, keep, g.Shape(), r.Shape())
					}
					n := g.Numel()
					for i := 0; i < n; i++ {
						idx := tensor.Unravel(i, g.Shape())
						gv, rv := g.AtF64(idx...), r.AtF64(idx...)
						if math.Float64bits(gv) != math.Float64bits(rv) {
							t.Fatalf("%v %v shape=%v axes=%v keep=%v idx=%v: cpu %.17g != ref %.17g", dt, op, c.shape, c.axes, keep, idx, gv, rv)
						}
					}
				}
			}
		}
	}
}

// TestReduceAxisSpecialValues checks Max/Min over the trailing axis with NaN, ±0, ±Inf present in
// each row — the IEEE tie cases the 4-way unroll must preserve — against the reference kernel.
func TestReduceAxisSpecialValues(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	refBE := backend.Reference()
	nan, pInf, nInf := math.NaN(), math.Inf(1), math.Inf(-1)
	rows := [][]float64{
		{1, -2, 3, nan, 5, 6, 7, 8, 9, 10},                           // NaN mid-row
		{0, math.Copysign(0, -1), pInf, nInf, 2, 3, 4, 5, 6, 7},      // +0,-0,±Inf
		{nInf, nInf, nInf, nInf, nInf, nInf, nInf, nInf, nInf, nInf}, // all -Inf
		{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10},
	}
	k := len(rows[0])
	data := make([]float64, len(rows)*k)
	for i, r := range rows {
		copy(data[i*k:], r)
	}
	x := tensor.FromFloat64(tensor.Shape{len(rows), k}, data)
	// axis 1 = trailing fast path; axis 0 = leading (prefix) fast path — both must match the
	// reference on the IEEE tie cases (NaN absorbing, math.Max(+0,-0)=+0, math.Min(+0,-0)=-0).
	for _, axis := range []int{1, 0} {
		for _, op := range []backend.Op{backend.OpMax, backend.OpMin} {
			gctx := backend.NewContext().WithBackend(cpuBE)
			rctx := backend.NewContext().WithBackend(refBE)
			attrs := backend.ReduceAttrs{Axes: []int{axis}, KeepDims: false}
			g, err := backend.Execute(gctx, op, []*tensor.Tensor{x}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			r, _ := backend.Execute(rctx, op, []*tensor.Tensor{x}, attrs)
			for i := 0; i < g[0].Numel(); i++ {
				gv, rv := g[0].AtF64(i), r[0].AtF64(i)
				if math.Float64bits(gv) != math.Float64bits(rv) {
					t.Fatalf("%v axis=%d out[%d]: cpu %v (bits %x) != ref %v (bits %x)", op, axis, i, gv, math.Float64bits(gv), rv, math.Float64bits(rv))
				}
			}
		}
	}
}

func benchAxisCPUAx(b *testing.B, op backend.Op, rows, cols, axis int) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	x := bench.RandF64(tensor.Shape{rows, cols}, 3)
	attrs := backend.ReduceAttrs{Axes: []int{axis}, KeepDims: false}
	ins := []*tensor.Tensor{x}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

// Leading (axis-0) reductions — the strided ref walk vs the new row-major accumulator path.
func BenchmarkReduceAxis0SumF64_4096x4096_cpu(b *testing.B) {
	benchAxisCPUAx(b, backend.OpSum, 4096, 4096, 0)
}
func BenchmarkReduceAxis0MaxF64_4096x4096_cpu(b *testing.B) {
	benchAxisCPUAx(b, backend.OpMax, 4096, 4096, 0)
}
func BenchmarkReduceAxis0MeanF64_4096x4096_cpu(b *testing.B) {
	benchAxisCPUAx(b, backend.OpMean, 4096, 4096, 0)
}

func benchAxisCPU(b *testing.B, op backend.Op, rows, cols int) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	x := bench.RandF64(tensor.Shape{rows, cols}, 3)
	attrs := backend.ReduceAttrs{Axes: []int{1}, KeepDims: false}
	ins := []*tensor.Tensor{x}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReduceAxisMaxF64_4096x4096_cpu(b *testing.B) {
	benchAxisCPU(b, backend.OpMax, 4096, 4096)
}
func BenchmarkReduceAxisMinF64_4096x4096_cpu(b *testing.B) {
	benchAxisCPU(b, backend.OpMin, 4096, 4096)
}
func BenchmarkReduceAxisSumF64_4096x4096_cpu(b *testing.B) {
	benchAxisCPU(b, backend.OpSum, 4096, 4096)
}
