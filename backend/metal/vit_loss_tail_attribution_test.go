//go:build darwin && cgo

package metal_test

import (
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

type noCrossEntropyMetalBackend struct{ backend.Backend }

func (b noCrossEntropyMetalBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpCrossEntropy || op == backend.OpCrossEntropyBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func runCrossEntropyBoundary(t testing.TB, be backend.Backend, logits, targets *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	loss, err := nn.CrossEntropy(tape.Context(), logits, targets)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	grad := tape.Grad(logits)
	if grad == nil {
		t.Fatal("cross-entropy logits gradient is nil")
	}
	return grad
}

func runOrderedLossTailBenchmarks(b *testing.B, control, candidate func(*testing.B)) {
	b.Helper()
	if os.Getenv("GOAI_VIT_LOSS_HOST_FIRST") == "1" {
		b.Run("candidate", candidate)
		b.Run("control", control)
		return
	}
	b.Run("control", control)
	b.Run("candidate", candidate)
}

func BenchmarkViTLossTailCrossEntropyBoundaryAttribution(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	logits := bench.RandF32(tensor.Shape{8, 10}, 611)
	targets := tensor.New(tensor.F32, tensor.Shape{8})
	for i := range 8 {
		targets.SetF64(float64(i%10), i)
	}
	host := noCrossEntropyMetalBackend{be}
	runCrossEntropyBoundary(b, be, logits, targets)
	runCrossEntropyBoundary(b, host, logits, targets)
	runOrderedLossTailBenchmarks(b, func(b *testing.B) {
		for range b.N {
			runCrossEntropyBoundary(b, be, logits, targets)
		}
	}, func(b *testing.B) {
		for range b.N {
			runCrossEntropyBoundary(b, host, logits, targets)
		}
	})
}

func BenchmarkViTLossTailTrainStepAttribution(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	m, err := vision.NewViT(3, 32, 10, 1,
		vision.WithViTPatch(4), vision.WithViTDim(128),
		vision.WithViTDepth(4), vision.WithViTHeads(4))
	if err != nil {
		b.Fatal(err)
	}
	x, targets := preNormFFNViTInputs()
	host := noCrossEntropyMetalBackend{be}
	runPreNormFFNViTStep(b, be, m, x, targets)
	runPreNormFFNViTStep(b, host, m, x, targets)
	runOrderedLossTailBenchmarks(b, func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, be, m, x, targets)
		}
		b.ReportMetric(float64(8*b.N)/b.Elapsed().Seconds(), "img/s")
	}, func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, host, m, x, targets)
		}
		b.ReportMetric(float64(8*b.N)/b.Elapsed().Seconds(), "img/s")
	})
}
