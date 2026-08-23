package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

var swigluFusionSink *tensor.Tensor

func TestSwiGLUInPlaceFusionMatchesComposedCPU(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("CPU backend is not registered")
	}
	fuser, ok := be.(backend.SwiGLUInPlaceFuser)
	if !ok {
		t.Fatal("CPU backend lacks SwiGLUInPlaceFuser")
	}
	ctx := backend.NewContext().WithBackend(be)
	for _, n := range []int{1, 3, 4, 31, 5632} {
		t.Run(tensor.Shape{n}.String(), func(t *testing.T) {
			gateData := make([]float32, n)
			upData := make([]float32, n)
			for i := range n {
				gateData[i] = float32(math.Sin(float64(i)*0.17)*9 - 2)
				upData[i] = float32(math.Cos(float64(i)*0.11)*3 + 0.25)
			}
			gate := tensor.FromFloat32(tensor.Shape{1, n}, gateData)
			up := tensor.FromFloat32(tensor.Shape{1, n}, upData)
			act, err := backend.Execute(ctx, backend.OpSiLU, []*tensor.Tensor{gate}, nil)
			if err != nil {
				t.Fatal(err)
			}
			want, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{act[0], up}, nil)
			if err != nil {
				t.Fatal(err)
			}
			got := tensor.FromFloat32(tensor.Shape{1, n}, gateData)
			if !fuser.FuseSwiGLUInPlace(got, up) {
				t.Fatal("CPU fuser rejected contiguous F32 projections")
			}
			for i, value := range got.Storage().F32() {
				if math.Float32bits(value) != math.Float32bits(want[0].Storage().F32()[i]) {
					t.Fatalf("element %d = %08x, want %08x", i, math.Float32bits(value), math.Float32bits(want[0].Storage().F32()[i]))
				}
				if math.Float32bits(up.Storage().F32()[i]) != math.Float32bits(upData[i]) {
					t.Fatalf("up element %d was mutated", i)
				}
			}
		})
	}
	alias := tensor.New(tensor.F32, tensor.Shape{1, 4})
	if fuser.FuseSwiGLUInPlace(alias, alias) {
		t.Fatal("CPU fuser accepted aliased gate and up storage")
	}
}

func TestSwiGLUF32ChunksMatchWholeTensor(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("CPU backend is not registered")
	}
	whole := be.(backend.SwiGLUInPlaceFuser)
	chunked := be.(backend.SwiGLUF32ChunkFuser)
	for _, n := range []int{1, 3, 4, 7, 8, 31, 5632, 5633, 65_537} {
		gateData := make([]float32, n)
		upData := make([]float32, n)
		for i := range n {
			gateData[i] = float32(math.Sin(float64(i)*0.17)*9 - 2)
			upData[i] = float32(math.Cos(float64(i)*0.11)*3 + 0.25)
		}
		want := tensor.FromFloat32(tensor.Shape{1, n}, gateData)
		up := tensor.FromFloat32(tensor.Shape{1, n}, upData)
		if !whole.FuseSwiGLUInPlace(want, up) {
			t.Fatal("whole-tensor fuser rejected input")
		}
		got := append([]float32(nil), gateData...)
		for lo := 0; lo < n; lo += 704 {
			hi := min(lo+704, n)
			chunked.FuseSwiGLUF32Chunk(got[lo:hi], upData[lo:hi])
		}
		for i := range n {
			if math.Float32bits(got[i]) != math.Float32bits(want.Storage().F32()[i]) {
				t.Fatalf("n=%d element %d = %08x, want %08x", n, i, math.Float32bits(got[i]), math.Float32bits(want.Storage().F32()[i]))
			}
		}
	}
}

func BenchmarkSwiGLUInPlaceFusion(b *testing.B) {
	const hidden = 5632
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend is not registered")
	}
	fuser, ok := be.(backend.SwiGLUInPlaceFuser)
	if !ok {
		b.Fatal("CPU backend lacks SwiGLUInPlaceFuser")
	}
	ctx := backend.NewContext().WithBackend(be)
	src := make([]float32, hidden)
	upData := make([]float32, hidden)
	for i := range hidden {
		src[i] = float32(math.Sin(float64(i)*0.17)*9 - 2)
		upData[i] = float32(math.Cos(float64(i)*0.11)*3 + 0.25)
	}
	up := tensor.FromFloat32(tensor.Shape{1, hidden}, upData)
	b.Run("composed", func(b *testing.B) {
		gate := tensor.New(tensor.F32, tensor.Shape{1, hidden})
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			copy(gate.Storage().F32(), src)
			act, err := backend.Execute(ctx, backend.OpSiLU, []*tensor.Tensor{gate}, nil)
			if err != nil {
				b.Fatal(err)
			}
			out, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{act[0], up}, nil)
			if err != nil {
				b.Fatal(err)
			}
			swigluFusionSink = out[0]
		}
	})
	b.Run("in_place", func(b *testing.B) {
		gate := tensor.New(tensor.F32, tensor.Shape{1, hidden})
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			copy(gate.Storage().F32(), src)
			if !fuser.FuseSwiGLUInPlace(gate, up) {
				b.Fatal("CPU fuser rejected contiguous F32 projections")
			}
			swigluFusionSink = gate
		}
	})
}
