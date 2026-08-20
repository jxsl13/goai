//go:build darwin && cgo && arm64

package metal_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

func unaryRouteRGLRUFixture(tb testing.TB) (*nn.RGLRU, *tensor.Tensor) {
	tb.Helper()
	model, err := nn.NewRGLRU(tensor.F32, 64, nn.WithRGLRUSeed(7))
	if err != nil {
		tb.Fatal(err)
	}
	x := tensor.New(tensor.F32, tensor.Shape{32, 64})
	values := x.Storage().F32()
	for i := range values {
		values[i] = float32((i*2654435761)&0xffff)/65536 - 0.5
	}
	return model, x
}

func TestMetalRGLRUUnaryRouteWorkloadParity(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("Metal available but not registered")
	}
	model, x := unaryRouteRGLRUFixture(t)
	got, err := model.Forward(backend.NewContext().WithBackend(mb), x)
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.Forward(backend.NewContext().WithBackend(backend.Reference()), x)
	if err != nil {
		t.Fatal(err)
	}
	for i, gotValue := range got.Storage().F32() {
		wantValue := want.Storage().F32()[i]
		if diff := math.Abs(float64(gotValue - wantValue)); diff > 2e-3+2e-3*math.Abs(float64(wantValue)) {
			t.Fatalf("RGLRU output[%d] = %g, reference %g (difference %.3g)", i, gotValue, wantValue, diff)
		}
	}
}

// BenchmarkMetalRGLRUUnaryRouteWorkload is an affected model-level check, not
// a synthetic unary chain. Griffin's real-gated recurrent unit dispatches
// Sigmoid, Neg, Exp, and Sqrt while computing its input-dependent decay and
// normalized recurrence input. The 32x64 gate tensors sit on the 2K route edge;
// its parallel contribution matrix contains 65,536 elements.
func BenchmarkMetalRGLRUUnaryRouteWorkload(b *testing.B) {
	if !metal.Available() {
		b.Skip("Metal unavailable")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		b.Fatal("Metal available but not registered")
	}
	model, x := unaryRouteRGLRUFixture(b)
	ctx := backend.NewContext().WithBackend(mb)
	for range 5 {
		if _, err := model.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := model.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}
