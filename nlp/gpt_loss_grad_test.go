package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type fallbackGPTLossGradBackend struct {
	backend.Backend
	calls int
}

func (b *fallbackGPTLossGradBackend) GPTLossAndGradF32(
	[]*tensor.Tensor, backend.GPTLossAndGradAttrs,
) (*tensor.Tensor, []*tensor.Tensor, bool, error) {
	b.calls++
	return nil, nil, false, nil
}

type noopGPTRecorder struct{}

func (noopGPTRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {}

func TestGPTLossAndGradPortableParity(t *testing.T) {
	model, golden := loadGPT(t)
	targets := tensor.New(tensor.F64, tensor.Shape{len(golden.Tokens)})
	for i, token := range golden.Tokens {
		targets.SetF64(float64((token+1)%golden.Config.Vocab), i)
	}
	be := backend.Reference()
	tape := autograd.NewTapeOn(be)
	logits, err := model.Forward(tape.Context(), golden.Tokens)
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

	gotLoss, gotGrads, err := model.LossAndGrad(
		backend.NewContext().WithBackend(be), golden.Tokens, targets)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotGrads) != len(model.Params()) {
		t.Fatalf("got %d gradients, want %d", len(gotGrads), len(model.Params()))
	}
	closeGPTLossGradF64(t, "loss", gotLoss, wantLoss)
	for i, parameter := range model.Params() {
		closeGPTLossGradF64(t, "parameter gradient", gotGrads[i], tape.Grad(parameter))
	}
}

func TestGPTLossAndGradCapabilityFallbackAndRecorderIsolation(t *testing.T) {
	model, golden := loadGPT(t)
	f32 := make(map[string]*tensor.Tensor, len(model.Safetensors()))
	for name, value := range model.Safetensors() {
		converted := tensor.New(tensor.F32, value.Shape())
		for i := range value.Numel() {
			index := tensor.Unravel(i, value.Shape())
			converted.SetF64(value.AtF64(index...), index...)
		}
		f32[name] = converted
	}
	model, err := nlp.FromSafetensors(nlp.GPTConfig{
		Vocab: golden.Config.Vocab, Ctx: golden.Config.Ctx, Dim: golden.Config.Dim,
		Heads: golden.Config.Heads, Layers: golden.Config.Layers, Eps: golden.Config.Eps,
	}, f32)
	if err != nil {
		t.Fatal(err)
	}
	targets := tensor.New(tensor.F32, tensor.Shape{len(golden.Tokens)})
	for i, token := range golden.Tokens {
		targets.SetF64(float64((token+1)%golden.Config.Vocab), i)
	}
	be := &fallbackGPTLossGradBackend{Backend: backend.Reference()}
	ctx := backend.NewContext().WithBackend(be)
	if _, grads, err := model.LossAndGrad(ctx, golden.Tokens, targets); err != nil || len(grads) != len(model.Params()) {
		t.Fatalf("capability fallback: gradients=%d error=%v", len(grads), err)
	}
	if be.calls != 1 {
		t.Fatalf("unsupported capability calls = %d, want 1", be.calls)
	}
	if _, grads, err := model.LossAndGrad(ctx.WithRecorder(noopGPTRecorder{}), golden.Tokens, targets); err != nil || len(grads) != len(model.Params()) {
		t.Fatalf("recorder fallback: gradients=%d error=%v", len(grads), err)
	}
	if be.calls != 1 {
		t.Fatalf("capability called through recorder: calls=%d, want 1", be.calls)
	}
}

func TestGPTLossAndGradRejectsInvalidInputs(t *testing.T) {
	model, golden := loadGPT(t)
	if _, _, err := model.LossAndGrad(nil, nil, nil); err == nil {
		t.Fatal("LossAndGrad accepted nil inputs")
	}
	targets := tensor.New(tensor.F64, tensor.Shape{1})
	if _, _, err := model.LossAndGrad(nil, []int{golden.Config.Vocab}, targets); err == nil {
		t.Fatal("LossAndGrad accepted an out-of-range token")
	}
}

func closeGPTLossGradF64(t *testing.T, name string, got, want *tensor.Tensor) {
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
