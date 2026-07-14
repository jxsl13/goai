//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// The on-device SiLU (nvrtc) must match the Pure-Go reference within tolerance.
func TestCUDADeviceSiLUMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	for _, s := range []struct{ r, c int }{{1, 1}, {4, 8}, {33, 17}, {8, 4096}} {
		x := bench.RandF32(tensor.Shape{s.r, s.c}, 6)
		d, _ := cuda.UploadF32(x)
		if err := d.SiLU(); err != nil {
			t.Fatal(err)
		}
		got, _ := d.ToHost()
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpSiLU, []*tensor.Tensor{x}, nil)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, r := got.AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("%v [%d]: device-silu %v vs ref %v", s, i, g, r)
			}
		}
		d.Free()
	}
}

// The on-device residual add (d += other) must match the reference elementwise add.
func TestCUDADeviceAddMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	for _, s := range []struct{ r, c int }{{1, 1}, {4, 8}, {33, 17}, {8, 512}} {
		a := bench.RandF32(tensor.Shape{s.r, s.c}, 1)
		b := bench.RandF32(tensor.Shape{s.r, s.c}, 2)
		da, _ := cuda.UploadF32(a)
		db, _ := cuda.UploadF32(b)
		if err := da.Add(db); err != nil {
			t.Fatal(err)
		}
		got, _ := da.ToHost()
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpAdd, []*tensor.Tensor{a, b}, nil)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, r := got.AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-6*math.Max(1, math.Abs(r)) {
				t.Fatalf("%v [%d]: device-add %v vs ref %v", s, i, g, r)
			}
		}
		da.Free()
		db.Free()
	}
}

func TestCUDASiLUAddUseAfterFree(t *testing.T) {
	skipNoGPU(t)
	d, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{4, 4}, 1))
	other, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{4, 4}, 2))
	d.Free()
	if err := d.SiLU(); err == nil {
		t.Fatal("expected SiLU error after Free")
	}
	if err := d.Add(other); err == nil {
		t.Fatal("expected Add error after Free")
	}
	other.Free()
}

// A residual FFN block y = x + SiLU(x·W1)·... stays fully device-resident: matmul,
// SiLU and the residual add all run on the stream with no host round-trip.
func BenchmarkResidualBlockOnDevice_M8_D2048(b *testing.B) {
	if _, ok := backend.Get(backend.CUDA); !ok {
		b.Skip("cuda not available")
	}
	const m, d = 8, 2048
	x := bench.RandF32(tensor.Shape{m, d}, 1)
	rb, _ := cuda.NewResidentB(bench.RandF32(tensor.Shape{d, d}, 2))
	defer rb.Free()
	b.ResetTimer()
	for range b.N {
		dx, _ := cuda.UploadF32(x)
		dh, err := rb.MatMulDevice(dx)
		if err != nil {
			b.Fatal(err)
		}
		if err := dh.SiLU(); err != nil {
			b.Fatal(err)
		}
		if err := dh.Add(dx); err != nil { // residual
			b.Fatal(err)
		}
		if _, err := dh.ToHost(); err != nil {
			b.Fatal(err)
		}
		dx.Free()
		dh.Free()
	}
}
