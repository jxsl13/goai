package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"testing"
)

// realistic transformer shape: split a [batch,seq,3*dim] into Q/K/V along the last
// axis — the slice backward scatters the grad of one [batch,seq,dim] chunk back.
func benchSliceVJP(b *testing.B, dt tensor.Dtype, axis, start, end int, shape tensor.Shape) {
	vjp := vjps[backend.OpSlice]
	x := tensor.New(dt, shape)
	xs := x.Storage()
	if dt == tensor.F64 {
		for i := range xs.F64() {
			xs.F64()[i] = float64(i%13) * 0.1
		}
	}
	gShape := append(tensor.Shape(nil), shape...)
	gShape[axis] = end - start
	g := tensor.New(dt, gShape)
	attrs := backend.SliceAttrs{Axis: axis, Start: start, End: end}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, nil, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}

// [32, 128, 2304] split into a [32,128,768] chunk on the last axis (QKV split).
func BenchmarkSliceVJP_QKV_F32(b *testing.B) {
	benchSliceVJP(b, tensor.F32, 2, 0, 768, tensor.Shape{32, 128, 2304})
}
func BenchmarkSliceVJP_QKV_F64(b *testing.B) {
	benchSliceVJP(b, tensor.F64, 2, 0, 768, tensor.Shape{32, 128, 2304})
}

func benchConcatVJP(b *testing.B, dt tensor.Dtype) {
	vjp := vjps[backend.OpConcat]
	mk := func() *tensor.Tensor { return tensor.New(dt, tensor.Shape{32, 128, 768}) }
	a, c, e := mk(), mk(), mk()
	g := tensor.New(dt, tensor.Shape{32, 128, 2304}) // concat of 3 along last axis
	attrs := backend.ConcatAttrs{Axis: 2}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{a, c, e}, nil, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkConcatVJP_F32(b *testing.B) { benchConcatVJP(b, tensor.F32) }
func BenchmarkConcatVJP_F64(b *testing.B) { benchConcatVJP(b, tensor.F64) }

func benchSoftCapVJP(b *testing.B, dt tensor.Dtype) {
	vjp := vjps[backend.OpSoftCap]
	mk := func() *tensor.Tensor { return tensor.New(dt, tensor.Shape{512, 512}) }
	x, y, g := mk(), mk(), mk()
	attrs := backend.SoftCapAttrs{Cap: 30.0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkSoftCapVJP_F32(b *testing.B) { benchSoftCapVJP(b, tensor.F32) }
func BenchmarkSoftCapVJP_F64(b *testing.B) { benchSoftCapVJP(b, tensor.F64) }
