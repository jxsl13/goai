//go:build darwin && cgo

package metal_test

import (
	"os"
	"slices"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

type vitAdamWSessionCapability interface {
	NewViTAdamWSessionF32([]*tensor.Tensor, backend.ViTLossAndGradAttrs, backend.ViTAdamWAttrs) (
		backend.ViTAdamWSession, bool, error)
}

func TestViTAdamWSessionMetalParitySyncAndLifetime(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	control := newPreNormFFNViT(t)
	candidate := newPreNormFFNViT(t)
	capability, ok := be.(vitAdamWSessionCapability)
	if !ok {
		t.Fatal("Metal backend lacks ViT AdamW session capability")
	}
	unsupported, supported, err := capability.NewViTAdamWSessionF32(
		candidate.Params(), backend.ViTLossAndGradAttrs{}, backend.ViTAdamWAttrs{})
	if err != nil || supported || unsupported != nil {
		t.Fatalf("invalid geometry: session=%v supported=%v err=%v", unsupported, supported, err)
	}
	images, targets := preNormFFNViTInputs()
	imagesBefore := slices.Clone(images.Storage().F32())
	targetsBefore := slices.Clone(targets.Storage().F32())
	config := vision.DefaultViTAdamWConfig(1e-3, 0.1)
	session, err := candidate.NewAdamWSession(backend.NewContext().WithBackend(be), 8, config)
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
	ctx := backend.NewContext().WithBackend(be)
	initialCandidate := slices.Clone(candidate.Params()[0].Storage().F32())
	for step := range 3 {
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
		closeMetalAdamWF32(t, "loss", gotLoss, wantLoss, 1e-4)
		if step == 0 && !slices.Equal(initialCandidate, candidate.Params()[0].Storage().F32()) {
			t.Fatal("resident Step mutated host parameters before Sync")
		}
		if step == 1 {
			if err := session.Sync(); err != nil {
				t.Fatal(err)
			}
			for i, want := range control.Params() {
				closeMetalAdamWF32(t, "checkpoint parameter", candidate.Params()[i], want, 8e-4)
			}
		}
	}
	if !slices.Equal(images.Storage().F32(), imagesBefore) || !slices.Equal(targets.Storage().F32(), targetsBefore) {
		t.Fatal("resident session mutated images or targets")
	}
	if err := session.Sync(); err != nil {
		t.Fatal(err)
	}
	for i, want := range control.Params() {
		closeMetalAdamWF32(t, "parameter", candidate.Params()[i], want, 8e-4)
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

func BenchmarkViTAdamWSessionPaired(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	control := newPreNormFFNViT(b)
	candidate := newPreNormFFNViT(b)
	images, targets := preNormFFNViTInputs()
	config := vision.DefaultViTAdamWConfig(1e-3, 0.1)
	session, err := candidate.NewAdamWSession(backend.NewContext().WithBackend(be), 8, config)
	if err != nil {
		b.Fatal(err)
	}
	controlParams := control.Params()
	optimizer := nn.NewAdamWF32(controlParams, config.LR, config.WeightDecay)
	optimizer.Beta1, optimizer.Beta2, optimizer.Eps = config.Beta1, config.Beta2, config.Eps
	indices := make(map[*tensor.Tensor]int, len(controlParams))
	for i, parameter := range controlParams {
		indices[parameter] = i
	}
	ctx := backend.NewContext().WithBackend(be)
	controlStep := func() {
		_, grads, err := control.LossAndGrad(ctx, images, targets)
		if err != nil {
			b.Fatal(err)
		}
		if err := optimizer.Step(func(parameter *tensor.Tensor) *tensor.Tensor {
			return grads[indices[parameter]]
		}); err != nil {
			b.Fatal(err)
		}
	}
	candidateStep := func() {
		if _, err := session.Step(images, targets); err != nil {
			b.Fatal(err)
		}
	}
	controlStep()
	candidateStep()
	candidateFirst := os.Getenv("GOAI_VIT_ADAMW_CANDIDATE_FIRST") == "1"
	var controlDuration, candidateDuration time.Duration
	b.ResetTimer()
	for i := range b.N {
		candidateLeads := (i%2 == 0) == candidateFirst
		if candidateLeads {
			start := time.Now()
			candidateStep()
			candidateDuration += time.Since(start)
			start = time.Now()
			controlStep()
			controlDuration += time.Since(start)
			continue
		}
		start := time.Now()
		controlStep()
		controlDuration += time.Since(start)
		start = time.Now()
		candidateStep()
		candidateDuration += time.Since(start)
	}
	b.StopTimer()
	if err := session.Close(); err != nil {
		b.Fatal(err)
	}
	controlNS := float64(controlDuration.Nanoseconds()) / float64(b.N)
	candidateNS := float64(candidateDuration.Nanoseconds()) / float64(b.N)
	b.ReportMetric(controlNS, "control-ns/step")
	b.ReportMetric(candidateNS, "candidate-ns/step")
	b.ReportMetric(controlNS/candidateNS, "speedup")
	b.ReportMetric(8e9/candidateNS, "img/s")
}
