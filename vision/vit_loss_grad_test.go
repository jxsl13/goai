package vision_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

type fallbackViTLossGradBackend struct {
	backend.Backend
	calls int
}

func (b *fallbackViTLossGradBackend) ViTLossAndGradF32(
	[]*tensor.Tensor, backend.ViTLossAndGradAttrs,
) (*tensor.Tensor, []*tensor.Tensor, bool, error) {
	b.calls++
	return nil, nil, false, nil
}

type noopViTRecorder struct{}

func (noopViTRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {}

func TestViTLossAndGradPortableParity(t *testing.T) {
	m, err := vision.NewViT(1, 8, 3, 17,
		vision.WithViTPatch(4), vision.WithViTDim(8), vision.WithViTDepth(1),
		vision.WithViTHeads(2), vision.WithViTMLP(16), vision.WithViTDtype(tensor.F64))
	if err != nil {
		t.Fatal(err)
	}
	images := tensor.Randn(tensor.F64, 19, tensor.Shape{2, 1, 8, 8})
	targets := tensor.New(tensor.F64, tensor.Shape{2})
	targets.SetF64(1, 0)
	targets.SetF64(2, 1)
	be := backend.Reference()

	tape := autograd.NewTapeOn(be)
	logits, err := m.Forward(tape.Context(), images)
	if err != nil {
		t.Fatal(err)
	}
	wantLoss, err := nn.CrossEntropy(tape.Context(), logits, targets)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(wantLoss); err != nil {
		t.Fatal(err)
	}

	gotLoss, gotGrads, err := m.LossAndGrad(backend.NewContext().WithBackend(be), images, targets)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotGrads) != len(m.Params()) {
		t.Fatalf("got %d gradients, want %d", len(gotGrads), len(m.Params()))
	}
	closeViTLossGradF64(t, "loss", gotLoss, wantLoss)
	for i, parameter := range m.Params() {
		closeViTLossGradF64(t, "parameter gradient", gotGrads[i], tape.Grad(parameter))
	}
}

func TestViTLossAndGradRejectsNilInputs(t *testing.T) {
	m, err := vision.NewViT(1, 8, 3, 23)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := m.LossAndGrad(nil, nil, nil); err == nil {
		t.Fatal("LossAndGrad accepted nil inputs")
	}
}

func TestViTLossAndGradCapabilityFallbackAndRecorderIsolation(t *testing.T) {
	m, err := vision.NewViT(1, 8, 3, 29,
		vision.WithViTPatch(4), vision.WithViTDim(8), vision.WithViTDepth(1),
		vision.WithViTHeads(2), vision.WithViTMLP(16))
	if err != nil {
		t.Fatal(err)
	}
	images := tensor.Randn(tensor.F32, 31, tensor.Shape{2, 1, 8, 8})
	targets := tensor.New(tensor.F32, tensor.Shape{2})
	targets.SetF64(1, 0)
	targets.SetF64(2, 1)
	be := &fallbackViTLossGradBackend{Backend: backend.Reference()}
	ctx := backend.NewContext().WithBackend(be)
	if _, grads, err := m.LossAndGrad(ctx, images, targets); err != nil || len(grads) != len(m.Params()) {
		t.Fatalf("capability fallback: gradients=%d error=%v", len(grads), err)
	}
	if be.calls != 1 {
		t.Fatalf("unsupported capability calls = %d, want 1", be.calls)
	}
	if _, grads, err := m.LossAndGrad(ctx.WithRecorder(noopViTRecorder{}), images, targets); err != nil || len(grads) != len(m.Params()) {
		t.Fatalf("recorder fallback: gradients=%d error=%v", len(grads), err)
	}
	if be.calls != 1 {
		t.Fatalf("capability called through recorder: calls=%d, want 1", be.calls)
	}
}

func closeViTLossGradF64(t *testing.T, name string, got, want *tensor.Tensor) {
	t.Helper()
	if got == nil || want == nil || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s shape mismatch: got %v want %v", name, got, want)
	}
	for i, value := range got.Storage().F64() {
		if delta := math.Abs(value - want.Storage().F64()[i]); delta > 1e-12 {
			t.Fatalf("%s[%d]: got %g want %g delta %g", name, i, value, want.Storage().F64()[i], delta)
		}
	}
}
