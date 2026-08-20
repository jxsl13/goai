//go:build darwin && cgo && arm64

package metal

import (
	"math"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// reluRouteWideMLPFixture is a real two-layer ReLU MLP with narrow external
// dimensions and a wide hidden layer. Its 256x1365 activation is the 349,440
// element model shape retained by the unary route matrix, while the surrounding
// projections keep this an end-to-end nn workload rather than a unary chain.
func reluRouteWideMLPFixture() (*nn.Sequential, *tensor.Tensor) {
	model := nn.NewSequential(
		nn.NewLinear(tensor.F32, 1, 1365, 41),
		nn.ReLU(),
		nn.NewLinear(tensor.F32, 1365, 1, 43),
	)
	x := tensor.New(tensor.F32, tensor.Shape{256, 1})
	for i := range x.Storage().F32() {
		x.Storage().F32()[i] = float32((i*97)%251)/125.5 - 1
	}
	return model, x
}

func TestMetalReLUWideMLPRouteWorkloadParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("Metal available but not registered")
	}
	model, x := reluRouteWideMLPFixture()
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
			t.Fatalf("wide ReLU MLP output[%d] = %g, reference %g (difference %.3g)", i, gotValue, wantValue, diff)
		}
	}
}

type directReLUBackend struct {
	backend.Backend
}

func (b directReLUBackend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpReLU && dtype == tensor.F32 {
		return unaryMetalF32(backend.OpReLU, unaryReLU), true
	}
	return b.Backend.Kernel(op, dtype)
}

func BenchmarkMetalReLUWideMLPRouteWorkload(b *testing.B) {
	if !Available() {
		b.Skip("Metal unavailable")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		b.Fatal("Metal available but not registered")
	}
	model, x := reluRouteWideMLPFixture()
	candidateCtx := backend.NewContext().WithBackend(mb)
	controlCtx := backend.NewContext().WithBackend(directReLUBackend{Backend: mb})
	for range 20 {
		if _, err := model.Forward(controlCtx, x); err != nil {
			b.Fatal(err)
		}
		if _, err := model.Forward(candidateCtx, x); err != nil {
			b.Fatal(err)
		}
	}
	var controlElapsed, candidateElapsed time.Duration
	run := func(ctx *backend.Context, elapsed *time.Duration) {
		start := time.Now()
		if _, err := model.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
		*elapsed += time.Since(start)
	}
	b.ResetTimer()
	for i := range b.N {
		if i&1 == 0 {
			run(controlCtx, &controlElapsed)
			run(candidateCtx, &candidateElapsed)
		} else {
			run(candidateCtx, &candidateElapsed)
			run(controlCtx, &controlElapsed)
		}
	}
	b.StopTimer()
	controlNs := float64(controlElapsed.Nanoseconds()) / float64(b.N)
	candidateNs := float64(candidateElapsed.Nanoseconds()) / float64(b.N)
	b.ReportMetric(controlNs, "control-ns/op")
	b.ReportMetric(candidateNs, "candidate-ns/op")
	b.ReportMetric(controlNs/candidateNs, "speedup")
}
