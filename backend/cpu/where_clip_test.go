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

// TestWhereClipCPUMatchesRef pins the new CPU OpWhere/OpClip kernels bit-for-bit against the ref
// kernels they parallelize, across dtypes, including special values and the Lo>Hi error delegation.
func TestWhereClipCPUMatchesRef(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	refBE := backend.Reference()
	exec := func(be backend.Backend, op backend.Op, at backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
		out, err := backend.Execute(backend.NewContext().WithBackend(be), op, ins, at)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	eqBits := func(t *testing.T, tag string, g, r *tensor.Tensor) {
		for i := 0; i < g.Numel(); i++ {
			idx := tensor.Unravel(i, g.Shape())
			if math.Float64bits(g.AtF64(idx...)) != math.Float64bits(r.AtF64(idx...)) {
				t.Fatalf("%s idx=%v: cpu %v != ref %v", tag, idx, g.AtF64(idx...), r.AtF64(idx...))
			}
		}
	}
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		n := tensor.Shape{200000}
		var a, b, cond *tensor.Tensor
		if dt == tensor.F64 {
			a, b, cond = bench.RandF64(n, 1), bench.RandF64(n, 2), bench.RandF64(n, 3)
		} else {
			a, b, cond = bench.RandF32(n, 1), bench.RandF32(n, 2), bench.RandF32(n, 3)
		}
		// make cond a mix of 0 and nonzero
		for i := 0; i < cond.Numel(); i++ {
			if i%3 == 0 {
				cond.SetF64(0, i)
			}
		}
		gw, _ := exec(cpuBE, backend.OpWhere, nil, cond, a, b)
		rw, _ := exec(refBE, backend.OpWhere, nil, cond, a, b)
		eqBits(t, "where "+dt.String(), gw, rw)

		for _, lohi := range [][2]float64{{-0.5, 0.5}, {0, 1e9}, {-1e9, 0}} {
			at := backend.ClipAttrs{Lo: lohi[0], Hi: lohi[1]}
			gc, _ := exec(cpuBE, backend.OpClip, at, a)
			rc, _ := exec(refBE, backend.OpClip, at, a)
			eqBits(t, "clip "+dt.String(), gc, rc)
		}
	}
	// Lo>Hi must delegate to ref and produce the SAME error.
	x := bench.RandF64(tensor.Shape{8}, 1)
	_, ge := exec(cpuBE, backend.OpClip, backend.ClipAttrs{Lo: 1, Hi: 0}, x)
	_, re := exec(refBE, backend.OpClip, backend.ClipAttrs{Lo: 1, Hi: 0}, x)
	if (ge == nil) != (re == nil) {
		t.Fatalf("clip Lo>Hi: cpu err %v vs ref err %v", ge, re)
	}
}

func BenchmarkWhereF64_256K_cpu(b *testing.B) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{bench.RandF64(tensor.Shape{262144}, 3), bench.RandF64(tensor.Shape{262144}, 1), bench.RandF64(tensor.Shape{262144}, 2)}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpWhere, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkWhereF64_256K_ref(b *testing.B) {
	ctx := backend.NewContext().WithBackend(backend.Reference())
	ins := []*tensor.Tensor{bench.RandF64(tensor.Shape{262144}, 3), bench.RandF64(tensor.Shape{262144}, 1), bench.RandF64(tensor.Shape{262144}, 2)}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpWhere, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}
func benchClip(b *testing.B, be backend.Backend) {
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{bench.RandF64(tensor.Shape{262144}, 1)}
	at := backend.ClipAttrs{Lo: -0.5, Hi: 0.5}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpClip, ins, at); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkClipF64_256K_cpu(b *testing.B) { be, _ := backend.Get(backend.CPU); benchClip(b, be) }
func BenchmarkClipF64_256K_ref(b *testing.B) { benchClip(b, backend.Reference()) }
