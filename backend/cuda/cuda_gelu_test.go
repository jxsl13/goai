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

// The on-device GELU (nvrtc-compiled, driver-launched) must match the Pure-Go
// reference GELU within the f32 tolerance, across shapes.
func TestCUDADeviceGELUMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	for _, s := range []struct{ r, c int }{{1, 1}, {4, 8}, {33, 17}, {8, 4096}} {
		x := bench.RandF32(tensor.Shape{s.r, s.c}, 5)
		d, err := cuda.UploadF32(x)
		if err != nil {
			t.Fatalf("%v: %v", s, err)
		}
		if err := d.GELU(); err != nil {
			t.Fatal(err)
		}
		got, err := d.ToHost()
		if err != nil {
			t.Fatal(err)
		}
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpGELU, []*tensor.Tensor{x}, nil)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, r := got.AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
				t.Fatalf("%v [%d]: device-gelu %v vs ref %v", s, i, g, r)
			}
		}
		d.Free()
	}
}

func TestCUDAGELUUseAfterFree(t *testing.T) {
	skipNoGPU(t)
	d, err := cuda.UploadF32(bench.RandF32(tensor.Shape{4, 4}, 1))
	if err != nil {
		t.Fatal(err)
	}
	d.Free()
	if err := d.GELU(); err == nil {
		t.Fatal("expected error on GELU after Free")
	}
}

// An FFN block x·W1 → GELU → ·W2. On-device: the whole block stays resident (the
// nvrtc GELU runs on the matmul stream). The residency-BREAKING path must bring
// the intermediate to host for the (Pure-Go) GELU and re-upload it. Same numeric
// result; the delta is the intermediate round-trip + host activation.
func benchFFN(b *testing.B, onDevice bool, m, d int) {
	be, ok := backend.Get(backend.CUDA)
	if !ok {
		b.Skip("cuda not available")
	}
	ref, _ := backend.Get(backend.Ref)
	x := bench.RandF32(tensor.Shape{m, d}, 1)
	w1 := bench.RandF32(tensor.Shape{d, d}, 2)
	w2 := bench.RandF32(tensor.Shape{d, d}, 3)
	rb1, _ := cuda.NewResidentB(w1)
	rb2, _ := cuda.NewResidentB(w2)
	defer rb1.Free()
	defer rb2.Free()
	b.ResetTimer()
	for range b.N {
		dx, _ := cuda.UploadF32(x)
		dh, err := rb1.MatMulDevice(dx)
		if err != nil {
			b.Fatal(err)
		}
		if onDevice {
			if err := dh.GELU(); err != nil {
				b.Fatal(err)
			}
		} else {
			h, _ := dh.ToHost()
			hg, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpGELU, []*tensor.Tensor{h}, nil)
			dh.Free()
			dh, _ = cuda.UploadF32(hg[0])
		}
		dy, _ := rb2.MatMulDevice(dh)
		if _, err := dy.ToHost(); err != nil {
			b.Fatal(err)
		}
		dx.Free()
		dh.Free()
		dy.Free()
	}
	_ = be
}

func BenchmarkFFNOnDevice_M8_D4096(b *testing.B) { benchFFN(b, true, 8, 4096) }
func BenchmarkFFNHostGELU_M8_D4096(b *testing.B) { benchFFN(b, false, 8, 4096) }
