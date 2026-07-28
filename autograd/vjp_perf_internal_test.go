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

// benchBcastReduce times bcastReduce for the bias-gradient shape — reducing a
// [batch, features] output cotangent to a [features] input — the backward of
// every broadcast Add/Mul (bias adds, per-channel scales).
func benchBcastReduce(b *testing.B, dt tensor.Dtype) {
	g := tensor.New(dt, tensor.Shape{128, 1024})
	switch dt {
	case tensor.F32:
		s := g.Storage().F32()
		for i := range s {
			s[i] = float32(i%11) * 0.1
		}
	case tensor.F64:
		s := g.Storage().F64()
		for i := range s {
			s[i] = float64(i%11) * 0.1
		}
	}
	inShape := tensor.Shape{1024}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bcastReduce(g, inShape)
	}
}

func BenchmarkBcastReduceBiasF32(b *testing.B) { benchBcastReduce(b, tensor.F32) }
func BenchmarkBcastReduceBiasF64(b *testing.B) { benchBcastReduce(b, tensor.F64) }

// BenchmarkNrm2VJP guards the §T634 scaleInto devirtualization of the L2-norm
// backward — on the gradient-clipping hot path (a global grad-norm every step).
func BenchmarkNrm2VJP(b *testing.B) {
	vjp := vjps[backend.OpNrm2]
	x := tensor.New(tensor.F32, tensor.Shape{512, 512})
	s := x.Storage().F32()
	for i := range s {
		s[i] = float32(i%7) * 0.1
	}
	out := tensor.New(tensor.F32, tensor.Shape{})
	out.Storage().F32()[0] = 100
	g := tensor.New(tensor.F32, tensor.Shape{})
	g.Storage().F32()[0] = 1
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{out}, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

// benchExtremumVJP covers the OpMax/OpMin backward, which routes the gradient to the
// first element attaining each group's extremum. It needs the FORWARD output y as
// well, since the VJP compares x against it — benchReduceVJP passes no `out` because
// the sum/mean backward does not need one.
func benchExtremumVJP(b *testing.B, op backend.Op, dt tensor.Dtype) {
	vjp := vjps[op]
	x := tensor.New(dt, tensor.Shape{512, 512})
	y := tensor.New(dt, tensor.Shape{512})
	g := tensor.New(dt, tensor.Shape{512})
	switch dt {
	case tensor.F32:
		xs, ys, gs := x.Storage().F32(), y.Storage().F32(), g.Storage().F32()
		for i := range xs {
			xs[i] = float32(i%97) * 0.25
		}
		for i := range ys {
			ys[i] = 24.0 // the max of i%97*0.25 over a row
			gs[i] = float32(i%13) * 0.1
		}
	case tensor.F64:
		xs, ys, gs := x.Storage().F64(), y.Storage().F64(), g.Storage().F64()
		for i := range xs {
			xs[i] = float64(i%97) * 0.25
		}
		for i := range ys {
			ys[i] = 24.0
			gs[i] = float64(i%13) * 0.1
		}
	}
	attrs := backend.ReduceAttrs{Axes: []int{1}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExtremumVJPMaxF32(b *testing.B) { benchExtremumVJP(b, backend.OpMax, tensor.F32) }
func BenchmarkExtremumVJPMaxF64(b *testing.B) { benchExtremumVJP(b, backend.OpMax, tensor.F64) }

// benchProdVJP covers the OpProd backward, the remaining reduceOffsets consumer. It is
// a genuine TWO-PASS algorithm — one pass tallies zeros and the non-zero product per
// group, the second builds the gradient — which is why it kept the materialized offset
// table when extremumVJP moved to forEachReduceRow.
//
// The fixture deliberately contains NO zeros: the zero-handling branches are
// correctness paths exercised by prod_test.go, and seeding zeros here would measure
// branch behavior rather than the traversal this benchmark exists to compare.
func benchProdVJP(b *testing.B, dt tensor.Dtype) {
	vjp := vjps[backend.OpProd]
	x := tensor.New(dt, tensor.Shape{512, 512})
	y := tensor.New(dt, tensor.Shape{512})
	g := tensor.New(dt, tensor.Shape{512})
	switch dt {
	case tensor.F32:
		xs, ys, gs := x.Storage().F32(), y.Storage().F32(), g.Storage().F32()
		for i := range xs {
			xs[i] = 1 + float32(i%7)*0.01
		}
		for i := range ys {
			ys[i] = 1
			gs[i] = float32(i%13) * 0.1
		}
	case tensor.F64:
		xs, ys, gs := x.Storage().F64(), y.Storage().F64(), g.Storage().F64()
		for i := range xs {
			xs[i] = 1 + float64(i%7)*0.01
		}
		for i := range ys {
			ys[i] = 1
			gs[i] = float64(i%13) * 0.1
		}
	}
	attrs := backend.ReduceAttrs{Axes: []int{1}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProdVJPF32(b *testing.B) { benchProdVJP(b, tensor.F32) }
func BenchmarkProdVJPF64(b *testing.B) { benchProdVJP(b, tensor.F64) }
