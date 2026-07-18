package goai

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestAvgPool2DAcceleratedBackends closes a coverage gap that the parity test in
// package nn cannot close by itself.
//
// nn/avgpool_test.go iterates backend.Available() and is backend-agnostic by
// design, but package nn deliberately does not blank-import the accelerated
// backends — doing so would drag cgo into every nn test binary. So in a plain
// `go test ./nn/` only cpu and ref are registered, and the GPU kernels go
// unexercised however green that run looks.
//
// The root package already registers metal (register_darwin.go) and vulkan under
// its build tag, so it is the natural place to assert the GPU path really agrees.
// Run it as `CGO_ENABLED=1 go test .` for metal and add `-tags vulkan` for vulkan;
// under plain `go test` it silently covers only cpu and ref, which is why the
// same assertion lives in both places rather than only here.
func TestAvgPool2DAcceleratedBackends(t *testing.T) {
	x := tensor.Zeros(tensor.F32, tensor.Shape{1, 1, 4, 4})
	for i := 0; i < 16; i++ {
		x.Storage().F32()[i] = float32(i)
	}
	// Hand-computed 2x2 means of the 4x4 ramp 0..15, stride 2.
	want := []float32{2.5, 4.5, 10.5, 12.5}
	p := &nn.AvgPool2D{Kernel: 2, Stride: 2}

	for _, name := range backend.Available() {
		b, ok := backend.Get(name)
		if !ok {
			continue
		}
		out, err := p.Forward(&backend.Context{Backend: b}, x)
		if err != nil {
			t.Errorf("%s: Forward: %v", name, err)
			continue
		}
		got := out.Contiguous().Storage().F32()
		if len(got) != len(want) {
			t.Errorf("%s: got %d values, want %d", name, len(got), len(want))
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("%s: got %v, want %v", name, got, want)
				break
			}
		}
	}
	t.Logf("verified on: %v", backend.Available())
}
