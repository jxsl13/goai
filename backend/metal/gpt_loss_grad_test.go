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
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type gptLossGradCapability interface {
	GPTLossAndGradF32([]*tensor.Tensor, backend.GPTLossAndGradAttrs) (*tensor.Tensor, []*tensor.Tensor, bool, error)
}

type countingGPTLossGradBackend struct {
	backend.Backend
	capability gptLossGradCapability
	calls      int
}

func (b *countingGPTLossGradBackend) GPTLossAndGradF32(
	in []*tensor.Tensor, attrs backend.GPTLossAndGradAttrs,
) (*tensor.Tensor, []*tensor.Tensor, bool, error) {
	b.calls++
	return b.capability.GPTLossAndGradF32(in, attrs)
}

func controlGPTLossAndGrad(
	t testing.TB, be backend.Backend, model *nlp.GPT, tokens []int, targets *tensor.Tensor,
) (*tensor.Tensor, []*tensor.Tensor) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	logits, err := model.Forward(tape.Context(), tokens)
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
	params := model.Params()
	grads := make([]*tensor.Tensor, len(params))
	for i, parameter := range params {
		grads[i] = tape.Grad(parameter)
		if grads[i] == nil {
			t.Fatalf("control parameter %d has no gradient", i)
		}
	}
	return loss, grads
}

func TestGPTLossAndGradMetalParityAndImmutability(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	capability, ok := be.(gptLossGradCapability)
	if !ok {
		t.Fatal("Metal backend lacks GPT loss-and-gradient capability")
	}
	cfg := nlp.GPTConfig{Vocab: 16, Ctx: 6, Dim: 8, Heads: 2, Layers: 2, Eps: 1e-5}
	model := randGPTf32(t, cfg, 4*cfg.Dim)
	tokens := []int{2, 5, 2, 9}
	targets := tensor.New(tensor.F32, tensor.Shape{len(tokens)})
	for i := range tokens {
		targets.SetF64(float64((tokens[i]+1)%cfg.Vocab), i)
	}
	allInputs := append([]*tensor.Tensor{targets}, model.Params()...)
	before := make([][]float32, len(allInputs))
	for i, value := range allInputs {
		before[i] = slices.Clone(value.Storage().F32())
	}
	wantLoss, wantGrads := controlGPTLossAndGrad(t, be, model, tokens, targets)
	counting := &countingGPTLossGradBackend{Backend: be, capability: capability}
	gotLoss, gotGrads, err := model.LossAndGrad(
		backend.NewContext().WithBackend(counting), tokens, targets)
	if err != nil {
		t.Fatal(err)
	}
	if counting.calls != 1 {
		t.Fatalf("whole-objective capability calls = %d, want 1", counting.calls)
	}
	closeGPTLossGradF32(t, "loss", gotLoss, wantLoss)
	if len(gotGrads) != len(wantGrads) {
		t.Fatalf("gradient count = %d, want %d", len(gotGrads), len(wantGrads))
	}
	for i := range gotGrads {
		closeGPTLossGradF32(t, "parameter gradient", gotGrads[i], wantGrads[i])
	}
	for i, value := range allInputs {
		if !slices.Equal(value.Storage().F32(), before[i]) {
			t.Fatalf("input or parameter %d mutated", i)
		}
	}
}

func benchmarkGPTObjectiveModel(t testing.TB) (*nlp.GPT, []int, *tensor.Tensor) {
	t.Helper()
	cfg := nlp.GPTConfig{Vocab: 4096, Ctx: 256, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	model := randGPTf32(t, cfg, 4*cfg.Dim)
	tokens := make([]int, cfg.Ctx)
	targets := tensor.New(tensor.F32, tensor.Shape{cfg.Ctx})
	for i := range tokens {
		tokens[i] = i % cfg.Vocab
		targets.SetF64(float64((i+1)%cfg.Vocab), i)
	}
	return model, tokens, targets
}

func BenchmarkGPTLossAndGradObjective(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	model, tokens, targets := benchmarkGPTObjectiveModel(b)
	ctx := backend.NewContext().WithBackend(be)
	if _, _, err := model.LossAndGrad(ctx, tokens, targets); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if _, _, err := model.LossAndGrad(ctx, tokens, targets); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(tokens))*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkGPTLossAndGradObjectivePaired(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	model, tokens, targets := benchmarkGPTObjectiveModel(b)
	candidateCtx := backend.NewContext().WithBackend(be)
	controlGPTLossAndGrad(b, be, model, tokens, targets)
	if _, _, err := model.LossAndGrad(candidateCtx, tokens, targets); err != nil {
		b.Fatal(err)
	}
	candidateFirst := os.Getenv("GOAI_GPT_LOSS_GRAD_CANDIDATE_FIRST") == "1"
	var controlDuration, candidateDuration time.Duration
	b.ResetTimer()
	for i := range b.N {
		candidateLeads := (i%2 == 0) == candidateFirst
		if candidateLeads {
			start := time.Now()
			if _, _, err := model.LossAndGrad(candidateCtx, tokens, targets); err != nil {
				b.Fatal(err)
			}
			candidateDuration += time.Since(start)
			start = time.Now()
			controlGPTLossAndGrad(b, be, model, tokens, targets)
			controlDuration += time.Since(start)
			continue
		}
		start := time.Now()
		controlGPTLossAndGrad(b, be, model, tokens, targets)
		controlDuration += time.Since(start)
		start = time.Now()
		if _, _, err := model.LossAndGrad(candidateCtx, tokens, targets); err != nil {
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

func closeGPTLossGradF32(t *testing.T, name string, got, want *tensor.Tensor) {
	t.Helper()
	if got == nil || want == nil || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s shape mismatch: got %v want %v", name, got, want)
	}
	for i, value := range got.Storage().F32() {
		w := want.Storage().F32()[i]
		tolerance := float32(5e-4 + 8e-4*math.Abs(float64(w)))
		if delta := float32(math.Abs(float64(value - w))); delta > tolerance {
			t.Fatalf("%s[%d]: candidate=%g control=%g delta=%g tolerance=%g", name, i, value, w, delta, tolerance)
		}
	}
}
