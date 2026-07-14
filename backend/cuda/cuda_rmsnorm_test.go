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

// The on-device RMSNorm (nvrtc, block-per-row reduction) must match the Pure-Go
// reference within tolerance, including cols not a multiple of the block size.
func TestCUDADeviceRMSNormMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	for _, s := range []struct{ rows, cols int }{{1, 1}, {4, 17}, {8, 100}, {33, 256}, {8, 4096}} {
		x := bench.RandF32(tensor.Shape{s.rows, s.cols}, 5)
		gamma := bench.RandF32(tensor.Shape{s.cols}, 9)
		dx, _ := cuda.UploadF32(x)
		gv, err := cuda.NewResidentVec(gamma)
		if err != nil {
			t.Fatalf("%v: %v", s, err)
		}
		if err := dx.RMSNorm(gv, 1e-5); err != nil {
			t.Fatal(err)
		}
		got, _ := dx.ToHost()
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRMSNorm,
			[]*tensor.Tensor{x, gamma}, backend.NormAttrs{Eps: 1e-5})
		if err != nil {
			t.Fatal(err)
		}
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, r := got.AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-4*math.Max(1, math.Abs(r)) {
				t.Fatalf("%v [%d]: device-rmsnorm %v vs ref %v", s, i, g, r)
			}
		}
		dx.Free()
		gv.Free()
	}
}

func TestCUDARMSNormUseAfterFree(t *testing.T) {
	skipNoGPU(t)
	dx, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{4, 8}, 1))
	gv, _ := cuda.NewResidentVec(bench.RandF32(tensor.Shape{8}, 2))
	dx.Free()
	if err := dx.RMSNorm(gv, 1e-5); err == nil {
		t.Fatal("expected error on RMSNorm after Free")
	}
	gv.Free()
}

// A Llama-style pre-norm projection: RMSNorm(x) then ·W, fully device-resident.
func BenchmarkRMSNormProjOnDevice_M8_D2048(b *testing.B) {
	if _, ok := backend.Get(backend.CUDA); !ok {
		b.Skip("cuda not available")
	}
	const m, d = 8, 2048
	x := bench.RandF32(tensor.Shape{m, d}, 1)
	rb, _ := cuda.NewResidentB(bench.RandF32(tensor.Shape{d, d}, 2))
	gv, _ := cuda.NewResidentVec(bench.RandF32(tensor.Shape{d}, 3))
	defer rb.Free()
	defer gv.Free()
	b.ResetTimer()
	for range b.N {
		dx, _ := cuda.UploadF32(x)
		if err := dx.RMSNorm(gv, 1e-5); err != nil {
			b.Fatal(err)
		}
		dy, err := rb.MatMulDevice(dx)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := dy.ToHost(); err != nil {
			b.Fatal(err)
		}
		dx.Free()
		dy.Free()
	}
}
