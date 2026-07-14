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

// The on-device stable softmax (nvrtc, two-reduction block-per-row) must match
// the Pure-Go reference within tolerance, incl. cols not a block-size multiple.
func TestCUDADeviceSoftmaxMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	for _, s := range []struct{ rows, cols int }{{1, 1}, {4, 17}, {8, 100}, {33, 256}, {8, 4096}} {
		x := bench.RandF32(tensor.Shape{s.rows, s.cols}, 5)
		dx, _ := cuda.UploadF32(x)
		if err := dx.Softmax(); err != nil {
			t.Fatal(err)
		}
		got, _ := dx.ToHost()
		gr, _ := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpSoftmax, []*tensor.Tensor{x}, nil)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, r := got.AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-5*math.Max(1e-3, math.Abs(r)) {
				t.Fatalf("%v [%d]: device-softmax %v vs ref %v", s, i, g, r)
			}
		}
		// each row must sum to ~1
		for r := 0; r < s.rows; r++ {
			var sum float64
			for c := 0; c < s.cols; c++ {
				sum += got.AtF64(r, c)
			}
			if math.Abs(sum-1) > 1e-4 {
				t.Fatalf("%v row %d sums to %v, want 1", s, r, sum)
			}
		}
		dx.Free()
	}
}

func TestCUDASoftmaxUseAfterFree(t *testing.T) {
	skipNoGPU(t)
	dx, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{4, 8}, 1))
	dx.Free()
	if err := dx.Softmax(); err == nil {
		t.Fatal("expected error on Softmax after Free")
	}
}
