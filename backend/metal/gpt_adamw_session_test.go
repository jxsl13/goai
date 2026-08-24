//go:build darwin && cgo

package metal_test

import (
	"math"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type gptAdamWSessionCapability interface {
	NewGPTAdamWSessionF32([]*tensor.Tensor, backend.GPTLossAndGradAttrs, backend.GPTAdamWAttrs) (
		backend.GPTAdamWSession, bool, error)
}

func TestGPTAdamWSessionMetalParitySyncAndLifetime(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	cfg := nlp.GPTConfig{Vocab: 16, Ctx: 6, Dim: 8, Heads: 2, Layers: 2, Eps: 1e-5}
	control := randGPTf32(t, cfg, 4*cfg.Dim)
	candidate := randGPTf32(t, cfg, 4*cfg.Dim)
	capability, ok := be.(gptAdamWSessionCapability)
	if !ok {
		t.Fatal("Metal backend lacks GPT AdamW session capability")
	}
	unsupported, supported, err := capability.NewGPTAdamWSessionF32(
		candidate.Params(), backend.GPTLossAndGradAttrs{}, backend.GPTAdamWAttrs{})
	if err != nil || supported || unsupported != nil {
		t.Fatalf("invalid geometry: session=%v supported=%v err=%v", unsupported, supported, err)
	}
	tokens := []int{2, 5, 2, 9}
	targets := tensor.New(tensor.F32, tensor.Shape{len(tokens)})
	for i, token := range tokens {
		targets.SetF64(float64((token+1)%cfg.Vocab), i)
	}
	config := nlp.DefaultGPTAdamWConfig(1e-3, 0.1)
	session, err := candidate.NewAdamWSession(
		backend.NewContext().WithBackend(be), len(tokens), config)
	if err != nil {
		t.Fatal(err)
	}
	controlParams := control.Params()
	optimizer := nn.NewAdamWF32(controlParams, config.LR, config.WeightDecay)
	indices := make(map[*tensor.Tensor]int, len(controlParams))
	for i, parameter := range controlParams {
		indices[parameter] = i
	}
	ctx := backend.NewContext().WithBackend(be)
	initialCandidate := slices.Clone(candidate.Params()[0].Storage().F32())
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
		closeMetalAdamWF32(t, "loss", gotLoss, wantLoss, 1e-4)
		if step == 0 && !slices.Equal(initialCandidate, candidate.Params()[0].Storage().F32()) {
			t.Fatal("resident Step mutated host parameters before Sync")
		}
		if err := session.Sync(); err != nil {
			t.Fatal(err)
		}
		for i, want := range control.Params() {
			closeMetalAdamWF32(t, "parameter", candidate.Params()[i], want, 8e-4)
		}
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

func BenchmarkGPTAdamWSessionPaired(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	cfg := nlp.GPTConfig{Vocab: 4096, Ctx: 256, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	control := randGPTf32(b, cfg, 4*cfg.Dim)
	candidate := randGPTf32(b, cfg, 4*cfg.Dim)
	tokens := make([]int, cfg.Ctx)
	targets := tensor.New(tensor.F32, tensor.Shape{cfg.Ctx})
	for i := range tokens {
		tokens[i] = i % cfg.Vocab
		targets.SetF64(float64(i%cfg.Vocab), i)
	}
	config := nlp.DefaultGPTAdamWConfig(1e-3, 0.1)
	session, err := candidate.NewAdamWSession(
		backend.NewContext().WithBackend(be), len(tokens), config)
	if err != nil {
		b.Fatal(err)
	}
	controlParams := control.Params()
	optimizer := nn.NewAdamWF32(controlParams, config.LR, config.WeightDecay)
	indices := make(map[*tensor.Tensor]int, len(controlParams))
	for i, parameter := range controlParams {
		indices[parameter] = i
	}
	ctx := backend.NewContext().WithBackend(be)
	controlStep := func() {
		_, grads, err := control.LossAndGrad(ctx, tokens, targets)
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
		if _, err := session.Step(tokens, targets); err != nil {
			b.Fatal(err)
		}
	}
	controlStep()
	candidateStep()
	candidateFirst := os.Getenv("GOAI_GPT_ADAMW_CANDIDATE_FIRST") == "1"
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
	b.ReportMetric(float64(len(tokens))*1e9/candidateNS, "tok/s")
}

func closeMetalAdamWF32(t *testing.T, name string, got, want *tensor.Tensor, tolerance float64) {
	t.Helper()
	if got == nil || want == nil || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s shape mismatch: got %v want %v", name, got, want)
	}
	for i, value := range got.Storage().F32() {
		w := want.Storage().F32()[i]
		if delta := math.Abs(float64(value - w)); delta > tolerance*(1+math.Abs(float64(w))) {
			t.Fatalf("%s[%d]: got %g want %g delta %g tolerance %g", name, i, value, w, delta, tolerance)
		}
	}
}
