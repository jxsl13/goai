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

// The on-device RoPE (nvrtc, one thread per dim-pair; frequencies from
// backend.RoPEFreqs) must match the Pure-Go reference within tolerance across
// single/multi-head, a KV-cache position offset, and PI position scaling.
func TestCUDADeviceRoPEMatchesRef(t *testing.T) {
	skipNoGPU(t)
	ref, _ := backend.Get(backend.Ref)
	cases := []struct {
		name       string
		seq, width int
		attrs      backend.RoPEAttrs
	}{
		{"single-head", 8, 64, backend.RoPEAttrs{}},
		{"multi-head", 8, 64, backend.RoPEAttrs{Heads: 4}},
		{"pos-offset", 6, 32, backend.RoPEAttrs{PosOffset: 100}},
		{"pi-scale", 8, 64, backend.RoPEAttrs{Heads: 4, PosScale: 2}},
		{"larger", 32, 128, backend.RoPEAttrs{Heads: 8}},
	}
	for _, c := range cases {
		x := bench.RandF32(tensor.Shape{c.seq, c.width}, 5)
		dx, _ := cuda.UploadF32(x)
		if err := dx.RoPE(c.attrs); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		got, _ := dx.ToHost()
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpRoPE, []*tensor.Tensor{x}, c.attrs)
		if err != nil {
			t.Fatalf("%s ref: %v", c.name, err)
		}
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, r := got.AtF64(idx...), gr[0].AtF64(idx...)
			if math.Abs(g-r) > 1e-3*math.Max(1, math.Abs(r)) {
				t.Fatalf("%s [%d]: device-rope %v vs ref %v", c.name, i, g, r)
			}
		}
		dx.Free()
	}
}

func TestCUDARoPEUseAfterFree(t *testing.T) {
	skipNoGPU(t)
	dx, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{4, 8}, 1))
	dx.Free()
	if err := dx.RoPE(backend.RoPEAttrs{}); err == nil {
		t.Fatal("expected error on RoPE after Free")
	}
}
