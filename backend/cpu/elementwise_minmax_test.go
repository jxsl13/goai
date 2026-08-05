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

// TestElementwiseMinMaxAbsSqrtCPUMatchesRef pins the new CPU kernels for OpMaximum/OpMinimum/OpAbs/
// OpSqrt BIT-FOR-BIT against the reference kernels they replaced, across dtypes and including the
// IEEE special values (NaN/±0/±Inf/negatives) where Max/Min tie rules and Sqrt-of-negative matter.
func TestElementwiseMinMaxAbsSqrtCPUMatchesRef(t *testing.T) {
	cpuBE, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend not registered")
	}
	refBE := backend.Reference()
	run := func(be backend.Backend, op backend.Op, ins ...*tensor.Tensor) *tensor.Tensor {
		out, err := backend.Execute(backend.NewContext().WithBackend(be), op, ins, nil)
		if err != nil {
			t.Fatalf("%v: %v", op, err)
		}
		return out[0]
	}
	eqBits := func(t *testing.T, op backend.Op, dt tensor.Dtype, g, r *tensor.Tensor) {
		n := g.Numel()
		for i := 0; i < n; i++ {
			idx := tensor.Unravel(i, g.Shape())
			gv, rv := g.AtF64(idx...), r.AtF64(idx...)
			if math.Float64bits(gv) != math.Float64bits(rv) {
				t.Fatalf("%v %v idx=%v: cpu %v (%x) != ref %v (%x)", dt, op, idx, gv, math.Float64bits(gv), rv, math.Float64bits(rv))
			}
		}
	}
	// Special-value probe values exercised on every op.
	nan, pInf, nInf, nz := math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1)
	special := []float64{0, nz, 1, -1, 2.5, -2.5, pInf, nInf, nan, 1e-20, 1e20}
	mk := func(dt tensor.Dtype, vals []float64) *tensor.Tensor {
		x := tensor.New(dt, tensor.Shape{len(vals)})
		for i, v := range vals {
			x.SetF64(v, i)
		}
		return x
	}
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		// Random large tensors (exercise the parallel path).
		var xa, xb *tensor.Tensor
		if dt == tensor.F64 {
			xa, xb = bench.RandF64(tensor.Shape{300000}, 1), bench.RandF64(tensor.Shape{300000}, 2)
		} else {
			xa, xb = bench.RandF32(tensor.Shape{300000}, 1), bench.RandF32(tensor.Shape{300000}, 2)
		}
		eqBits(t, backend.OpMaximum, dt, run(cpuBE, backend.OpMaximum, xa, xb), run(refBE, backend.OpMaximum, xa, xb))
		eqBits(t, backend.OpMinimum, dt, run(cpuBE, backend.OpMinimum, xa, xb), run(refBE, backend.OpMinimum, xa, xb))
		eqBits(t, backend.OpAbs, dt, run(cpuBE, backend.OpAbs, xa), run(refBE, backend.OpAbs, xa))
		eqBits(t, backend.OpSqrt, dt, run(cpuBE, backend.OpSqrt, xa), run(refBE, backend.OpSqrt, xa))
		// Special values.
		sa, sb := mk(dt, special), mk(dt, append([]float64{nInf, pInf, nan, nz, 0, -1, 1, 2.5, -2.5, 1e20, 1e-20}, nil...)[:len(special)])
		eqBits(t, backend.OpMaximum, dt, run(cpuBE, backend.OpMaximum, sa, sb), run(refBE, backend.OpMaximum, sa, sb))
		eqBits(t, backend.OpMinimum, dt, run(cpuBE, backend.OpMinimum, sa, sb), run(refBE, backend.OpMinimum, sa, sb))
		eqBits(t, backend.OpAbs, dt, run(cpuBE, backend.OpAbs, sa), run(refBE, backend.OpAbs, sa))
		eqBits(t, backend.OpSqrt, dt, run(cpuBE, backend.OpSqrt, sa), run(refBE, backend.OpSqrt, sa))
	}
}

func benchElem(b *testing.B, op backend.Op, binary bool) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	x := bench.RandF64(tensor.Shape{262144}, 1)
	ins := []*tensor.Tensor{x}
	if binary {
		ins = append(ins, bench.RandF64(tensor.Shape{262144}, 2))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMaximumF64_256K_cpu(b *testing.B) { benchElem(b, backend.OpMaximum, true) }
func BenchmarkMinimumF64_256K_cpu(b *testing.B) { benchElem(b, backend.OpMinimum, true) }
func BenchmarkAbsF64_256K_cpu(b *testing.B)     { benchElem(b, backend.OpAbs, false) }
func BenchmarkSqrtF64_256K_cpu(b *testing.B)    { benchElem(b, backend.OpSqrt, false) }

func benchElemRef(b *testing.B, op backend.Op, binary bool) {
	ctx := backend.NewContext().WithBackend(backend.Reference())
	x := bench.RandF64(tensor.Shape{262144}, 1)
	ins := []*tensor.Tensor{x}
	if binary {
		ins = append(ins, bench.RandF64(tensor.Shape{262144}, 2))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMaximumF64_256K_ref(b *testing.B) { benchElemRef(b, backend.OpMaximum, true) }
func BenchmarkAbsF64_256K_ref(b *testing.B)     { benchElemRef(b, backend.OpAbs, false) }
func BenchmarkSqrtF64_256K_ref(b *testing.B)    { benchElemRef(b, backend.OpSqrt, false) }
