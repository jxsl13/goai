package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func cloneGPTF32(t *testing.T) (*nlp.GPT, []int) {
	t.Helper()
	model, golden := loadGPT(t)
	values := make(map[string]*tensor.Tensor, len(model.Safetensors()))
	for name, value := range model.Safetensors() {
		converted := tensor.New(tensor.F32, value.Shape())
		for i := range value.Numel() {
			index := tensor.Unravel(i, value.Shape())
			converted.SetF64(value.AtF64(index...), index...)
		}
		values[name] = converted
	}
	got, err := nlp.FromSafetensors(nlp.GPTConfig{
		Vocab: golden.Config.Vocab, Ctx: golden.Config.Ctx, Dim: golden.Config.Dim,
		Heads: golden.Config.Heads, Layers: golden.Config.Layers, Eps: golden.Config.Eps,
	}, values)
	if err != nil {
		t.Fatal(err)
	}
	return got, golden.Tokens
}

func TestGPTAdamWSessionPortableParityAndLifetime(t *testing.T) {
	control, tokens := cloneGPTF32(t)
	candidate, _ := cloneGPTF32(t)
	targets := tensor.New(tensor.F32, tensor.Shape{len(tokens)})
	for i, token := range tokens {
		targets.SetF64(float64((token+1)%candidate.Config.Vocab), i)
	}
	config := nlp.DefaultGPTAdamWConfig(1e-3, 0.1)
	session, err := candidate.NewAdamWSession(
		backend.NewContext().WithBackend(backend.Reference()), len(tokens), config)
	if err != nil {
		t.Fatal(err)
	}
	controlParams := control.Params()
	optimizer := nn.NewAdamWF32(controlParams, config.LR, config.WeightDecay)
	optimizer.Beta1, optimizer.Beta2, optimizer.Eps = config.Beta1, config.Beta2, config.Eps
	indices := make(map[*tensor.Tensor]int, len(controlParams))
	for i, parameter := range controlParams {
		indices[parameter] = i
	}
	ctx := backend.NewContext().WithBackend(backend.Reference())
	for step := 0; step < 3; step++ {
		wantLoss, grads, err := control.LossAndGrad(ctx, tokens, targets)
		if err != nil {
			t.Fatal(err)
		}
		if err := optimizer.Step(func(parameter *tensor.Tensor) *tensor.Tensor {
			return grads[indices[parameter]]
		}); err != nil {
			t.Fatal(err)
		}
		gotLoss, err := session.Step(tokens, targets)
		if err != nil {
			t.Fatal(err)
		}
		closeSessionF32(t, "loss", gotLoss, wantLoss, 1e-6)
	}
	if err := session.Sync(); err != nil {
		t.Fatal(err)
	}
	for i, want := range control.Params() {
		closeSessionF32(t, "parameter", candidate.Params()[i], want, 1e-6)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := session.Step(tokens, targets); err == nil {
		t.Fatal("Step succeeded after Close")
	}
}

func TestGPTAdamWSessionRejectsInvalidConfigurationAndInputs(t *testing.T) {
	model, tokens := cloneGPTF32(t)
	ctx := backend.NewContext().WithBackend(backend.Reference())
	if _, err := model.NewAdamWSession(ctx, len(tokens), nlp.GPTAdamWConfig{}); err == nil {
		t.Fatal("zero configuration accepted")
	}
	session, err := model.NewAdamWSession(ctx, len(tokens), nlp.DefaultGPTAdamWConfig(1e-3, 0.1))
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if _, err := session.Step(tokens[:len(tokens)-1], tensor.New(tensor.F32, tensor.Shape{len(tokens)})); err == nil {
		t.Fatal("wrong token count accepted")
	}
	badTargets := tensor.New(tensor.F32, tensor.Shape{len(tokens)})
	badTargets.SetF64(float64(model.Config.Vocab), 0)
	if _, err := session.Step(tokens, badTargets); err == nil {
		t.Fatal("out-of-range target accepted")
	}
}

func closeSessionF32(t *testing.T, name string, got, want *tensor.Tensor, tolerance float64) {
	t.Helper()
	if got == nil || want == nil || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s shape mismatch: got %v want %v", name, got, want)
	}
	for i, value := range got.Storage().F32() {
		w := want.Storage().F32()[i]
		if delta := math.Abs(float64(value - w)); delta > tolerance*(1+math.Abs(float64(w))) {
			t.Fatalf("%s[%d]: got %g want %g delta %g", name, i, value, w, delta)
		}
	}
}
