package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchUnaryVJP times a registered unary VJP over a 512×512 F32 tensor — the
// training backward path for activation gradients. Guards the §T632 devirtualization
// (typed flat loop, no per-element Unravel/AtF64/SetF64) against regressing back to
// the per-element path.
func benchUnaryVJP(b *testing.B, op backend.Op) {
	vjp := vjps[op]
	if vjp == nil {
		b.Skipf("no vjp for %v", op)
	}
	mk := func() *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape{512, 512})
		s := t.Storage().F32()
		for i := range s {
			s[i] = float32((i%17)-8) * 0.1
		}
		return t
	}
	x, y, g := mk(), mk(), mk()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUnaryVJPTanh(b *testing.B)    { benchUnaryVJP(b, backend.OpTanh) }
func BenchmarkUnaryVJPExp(b *testing.B)     { benchUnaryVJP(b, backend.OpExp) }
func BenchmarkUnaryVJPSigmoid(b *testing.B) { benchUnaryVJP(b, backend.OpSigmoid) }

// benchReduceVJP times the sum/mean backward over a 512×512 F32 tensor reduced
// along one axis — the §T633 incremental-offset devirtualization guard.
func benchReduceVJP(b *testing.B, op backend.Op) {
	vjp := vjps[op]
	x := tensor.New(tensor.F32, tensor.Shape{512, 512})
	attrs := backend.ReduceAttrs{Axes: []int{1}}
	g := tensor.New(tensor.F32, tensor.Shape{512}) // reduced output shape
	gs := g.Storage().F32()
	for i := range gs {
		gs[i] = float32(i%13) * 0.1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, nil, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReduceVJPSum(b *testing.B)  { benchReduceVJP(b, backend.OpSum) }
func BenchmarkReduceVJPMean(b *testing.B) { benchReduceVJP(b, backend.OpMean) }

// broadcastVJPNaive is the pre-§T633 per-element path (double Unravel + AtF64/SetF64)
// — the A/B baseline for the incremental-offset rewrite (§V22).
func broadcastVJPNaive(x *tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) *tensor.Tensor {
	outShape, mapIdx, _, _ := reduceOutMap(x.Shape(), attrs)
	gin := tensor.New(x.Dtype(), x.Shape())
	for i := 0; i < x.Numel(); i++ {
		idx := tensor.Unravel(i, x.Shape())
		gin.SetF64(g.AtF64(tensor.Unravel(mapIdx(idx), outShape)...), idx...)
	}
	return gin
}

func BenchmarkReduceVJPSumNaive(b *testing.B) {
	x := tensor.New(tensor.F32, tensor.Shape{512, 512})
	attrs := backend.ReduceAttrs{Axes: []int{1}}
	g := tensor.New(tensor.F32, tensor.Shape{512})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = broadcastVJPNaive(x, attrs, g)
	}
}
