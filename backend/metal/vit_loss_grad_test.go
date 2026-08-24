//go:build darwin && cgo

package metal_test

import (
	"math"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

type vitLossGradCapability interface {
	ViTLossAndGradF32([]*tensor.Tensor, backend.ViTLossAndGradAttrs) (*tensor.Tensor, []*tensor.Tensor, bool, error)
}

type countingViTLossGradBackend struct {
	backend.Backend
	capability vitLossGradCapability
	calls      int
}

func (b *countingViTLossGradBackend) ViTLossAndGradF32(
	in []*tensor.Tensor, attrs backend.ViTLossAndGradAttrs,
) (*tensor.Tensor, []*tensor.Tensor, bool, error) {
	b.calls++
	return b.capability.ViTLossAndGradF32(in, attrs)
}

func controlViTLossAndGrad(
	t testing.TB, be backend.Backend, m *vision.ViT, images, targets *tensor.Tensor,
) (*tensor.Tensor, []*tensor.Tensor) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	logits, err := m.Forward(tape.Context(), images)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := nn.CrossEntropy(tape.Context(), logits, targets)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	params := m.Params()
	grads := make([]*tensor.Tensor, len(params))
	for i, parameter := range params {
		grads[i] = tape.Grad(parameter)
		if grads[i] == nil {
			t.Fatalf("control parameter %d has no gradient", i)
		}
	}
	return loss, grads
}

func TestViTLossAndGradMetalParityAndImmutability(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	capability, ok := be.(vitLossGradCapability)
	if !ok {
		t.Fatal("Metal backend lacks ViT loss-and-gradient capability")
	}
	m := newPreNormFFNViT(t)
	images, targets := preNormFFNViTInputs()
	allInputs := append([]*tensor.Tensor{images, targets}, m.Params()...)
	before := make([][]float32, len(allInputs))
	for i, value := range allInputs {
		before[i] = slices.Clone(value.Storage().F32())
	}
	wantLoss, wantGrads := controlViTLossAndGrad(t, be, m, images, targets)
	counting := &countingViTLossGradBackend{Backend: be, capability: capability}
	gotLoss, gotGrads, err := m.LossAndGrad(
		backend.NewContext().WithBackend(counting), images, targets)
	if err != nil {
		t.Fatal(err)
	}
	if counting.calls != 1 {
		t.Fatalf("whole-objective capability calls = %d, want 1", counting.calls)
	}
	closeViTLossGradF32(t, "loss", gotLoss, wantLoss)
	if len(gotGrads) != len(wantGrads) {
		t.Fatalf("gradient count = %d, want %d", len(gotGrads), len(wantGrads))
	}
	for i := range gotGrads {
		closeViTLossGradF32(t, "parameter gradient", gotGrads[i], wantGrads[i])
	}
	for i, value := range allInputs {
		if !slices.Equal(value.Storage().F32(), before[i]) {
			t.Fatalf("input or parameter %d mutated", i)
		}
	}
}

func BenchmarkViTLossAndGradObjectivePaired(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(b)
	images, targets := preNormFFNViTInputs()
	controlViTLossAndGrad(b, be, m, images, targets)
	if _, _, err := m.LossAndGrad(backend.NewContext().WithBackend(be), images, targets); err != nil {
		b.Fatal(err)
	}
	candidateFirst := os.Getenv("GOAI_VIT_LOSS_GRAD_CANDIDATE_FIRST") == "1"
	var controlDuration, candidateDuration time.Duration
	b.ResetTimer()
	for i := range b.N {
		candidateLeads := (i%2 == 0) == candidateFirst
		if candidateLeads {
			start := time.Now()
			if _, _, err := m.LossAndGrad(backend.NewContext().WithBackend(be), images, targets); err != nil {
				b.Fatal(err)
			}
			candidateDuration += time.Since(start)
			start = time.Now()
			controlViTLossAndGrad(b, be, m, images, targets)
			controlDuration += time.Since(start)
			continue
		}
		start := time.Now()
		controlViTLossAndGrad(b, be, m, images, targets)
		controlDuration += time.Since(start)
		start = time.Now()
		if _, _, err := m.LossAndGrad(backend.NewContext().WithBackend(be), images, targets); err != nil {
			b.Fatal(err)
		}
		candidateDuration += time.Since(start)
	}
	b.StopTimer()
	controlNS := float64(controlDuration.Nanoseconds()) / float64(b.N)
	candidateNS := float64(candidateDuration.Nanoseconds()) / float64(b.N)
	b.ReportMetric(controlNS, "control-ns/objective")
	b.ReportMetric(candidateNS, "candidate-ns/objective")
	b.ReportMetric(controlNS/candidateNS, "speedup")
}

func closeViTLossGradF32(t *testing.T, name string, got, want *tensor.Tensor) {
	t.Helper()
	if got == nil || want == nil || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s shape mismatch: got %v want %v", name, got, want)
	}
	for i, value := range got.Storage().F32() {
		w := want.Storage().F32()[i]
		tolerance := float32(3e-4 + 5e-4*math.Abs(float64(w)))
		if delta := float32(math.Abs(float64(value - w))); delta > tolerance {
			t.Fatalf("%s[%d]: candidate=%g control=%g delta=%g tolerance=%g", name, i, value, w, delta, tolerance)
		}
	}
}
