package vision_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

type fallbackViTAdamWBackend struct {
	backend.Backend
	calls int
}

func (b *fallbackViTAdamWBackend) NewViTAdamWSessionF32(
	[]*tensor.Tensor, backend.ViTLossAndGradAttrs, backend.ViTAdamWAttrs,
) (backend.ViTAdamWSession, bool, error) {
	b.calls++
	return nil, false, nil
}

func newViTAdamWF32(t *testing.T) *vision.ViT {
	t.Helper()
	m, err := vision.NewViT(1, 8, 3, 41,
		vision.WithViTPatch(4), vision.WithViTDim(8), vision.WithViTDepth(1),
		vision.WithViTHeads(2), vision.WithViTMLP(16), vision.WithViTDtype(tensor.F32))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestViTAdamWSessionPortableParityFallbackAndLifetime(t *testing.T) {
	control := newViTAdamWF32(t)
	candidate := newViTAdamWF32(t)
	images := tensor.Randn(tensor.F32, 43, tensor.Shape{2, 1, 8, 8})
	targets := tensor.New(tensor.F32, tensor.Shape{2})
	targets.SetF64(1, 0)
	targets.SetF64(2, 1)
	config := vision.DefaultViTAdamWConfig(1e-3, 0.1)
	be := &fallbackViTAdamWBackend{Backend: backend.Reference()}
	session, err := candidate.NewAdamWSession(backend.NewContext().WithBackend(be), 2, config)
	if err != nil {
		t.Fatal(err)
	}
	if be.calls != 1 {
		t.Fatalf("resident capability calls = %d, want 1", be.calls)
	}
	controlParams := control.Params()
	optimizer := nn.NewAdamWF32(controlParams, config.LR, config.WeightDecay)
	optimizer.Beta1, optimizer.Beta2, optimizer.Eps = config.Beta1, config.Beta2, config.Eps
	indices := make(map[*tensor.Tensor]int, len(controlParams))
	for i, parameter := range controlParams {
		indices[parameter] = i
	}
	ctx := backend.NewContext().WithBackend(backend.Reference())
	for range 3 {
		wantLoss, grads, err := control.LossAndGrad(ctx, images, targets)
		if err != nil {
			t.Fatal(err)
		}
		if err := optimizer.Step(func(parameter *tensor.Tensor) *tensor.Tensor {
			return grads[indices[parameter]]
		}); err != nil {
			t.Fatal(err)
		}
		gotLoss, err := session.Step(images, targets)
		if err != nil {
			t.Fatal(err)
		}
		closeViTAdamWF32(t, "loss", gotLoss, wantLoss, 1e-6)
	}
	if err := session.Sync(); err != nil {
		t.Fatal(err)
	}
	for i, want := range control.Params() {
		closeViTAdamWF32(t, "parameter", candidate.Params()[i], want, 1e-6)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := session.Step(images, targets); err == nil {
		t.Fatal("Step succeeded after Close")
	}
	if err := session.Sync(); err == nil {
		t.Fatal("Sync succeeded after Close")
	}
}

func TestViTAdamWSessionRejectsInvalidConfigurationAndInputs(t *testing.T) {
	m := newViTAdamWF32(t)
	ctx := backend.NewContext().WithBackend(backend.Reference())
	if _, err := m.NewAdamWSession(ctx, 2, vision.ViTAdamWConfig{}); err == nil {
		t.Fatal("zero configuration accepted")
	}
	session, err := m.NewAdamWSession(ctx, 2, vision.DefaultViTAdamWConfig(1e-3, 0.1))
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	badImages := tensor.New(tensor.F32, tensor.Shape{1, 1, 8, 8})
	if _, err := session.Step(badImages, tensor.New(tensor.F32, tensor.Shape{2})); err == nil {
		t.Fatal("wrong image batch accepted")
	}
	images := tensor.New(tensor.F32, tensor.Shape{2, 1, 8, 8})
	badTargets := tensor.New(tensor.F32, tensor.Shape{2})
	badTargets.SetF64(3, 0)
	if _, err := session.Step(images, badTargets); err == nil {
		t.Fatal("out-of-range target accepted")
	}
}

func closeViTAdamWF32(t *testing.T, name string, got, want *tensor.Tensor, tolerance float64) {
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
