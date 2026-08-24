//go:build darwin && cgo

package metal_test

import (
	"os"
	"slices"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

const preNormTransformerStackDepth = 4

type preNormTransformerStackFixture struct {
	x      *tensor.Tensor
	blocks [preNormTransformerStackDepth]preNormTransformerBlockFixture
	dOut   *tensor.Tensor
}

func newPreNormTransformerStackFixture() preNormTransformerStackFixture {
	f := preNormTransformerStackFixture{}
	for i := range f.blocks {
		f.blocks[i] = newPreNormTransformerBlockFixture()
	}
	f.x = f.blocks[0].x
	f.dOut = f.blocks[0].dOut
	return f
}

func (f preNormTransformerStackFixture) inputs() []*tensor.Tensor {
	inputs := make([]*tensor.Tensor, 1, 1+12*len(f.blocks))
	inputs[0] = f.x
	for _, block := range f.blocks {
		inputs = append(inputs, block.inputs()[1:]...)
	}
	return inputs
}

func (f preNormTransformerStackFixture) layers(t testing.TB) []nlp.PreNormTransformerBlock {
	t.Helper()
	blocks := make([]nlp.PreNormTransformerBlock, len(f.blocks))
	for i, block := range f.blocks {
		mha, norm1, norm2, up, down := block.layers(t, 2e-5, 3e-5)
		blocks[i] = nlp.PreNormTransformerBlock{
			Attention: mha, Norm1: norm1, Norm2: norm2, Up: up, Down: down,
		}
	}
	return blocks
}

func runPreNormTransformerStack(t testing.TB, be backend.Backend, f preNormTransformerStackFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	out, err := nlp.ForwardPreNormTransformerStack(tape.Context(), f.x, f.layers(t), preNormAttentionBatch)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out, f.dOut); err != nil {
		t.Fatal(err)
	}
	return out, tape
}

type noPreNormTransformerStackMetalBackend struct{ backend.Backend }

func (b noPreNormTransformerStackMetalBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpPreNormTransformerStack || op == backend.OpPreNormTransformerStackBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func runOrderedPreNormTransformerStackBenchmarks(
	b *testing.B,
	control func(*testing.B),
	candidate func(*testing.B),
) {
	b.Helper()
	if os.Getenv("GOAI_PRENORM_STACK_CANDIDATE_FIRST") == "1" {
		b.Run("candidate", candidate)
		b.Run("control", control)
		return
	}
	b.Run("control", control)
	b.Run("candidate", candidate)
}

func TestPreNormTransformerStackMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	f := newPreNormTransformerStackFixture()
	allInputs := append(slices.Clone(f.inputs()), f.dOut)
	before := make([][]float32, len(allInputs))
	for i, input := range allInputs {
		before[i] = slices.Clone(input.Storage().F32())
	}
	wantY, wantTape := runPreNormTransformerStack(t, noPreNormTransformerStackMetalBackend{be}, f)
	gotY, gotTape := runPreNormTransformerStack(t, be, f)
	closePreNormAttentionTensor(t, "Y", gotY, wantY)
	for i, input := range f.inputs() {
		got, want := gotTape.Grad(input), wantTape.Grad(input)
		if got == nil || want == nil {
			t.Fatalf("gradient %d is nil", i)
		}
		closePreNormAttentionTensor(t, "gradient", got, want)
	}
	for i, input := range allInputs {
		if !slices.Equal(input.Storage().F32(), before[i]) {
			t.Fatalf("input %d mutated", i)
		}
	}
}

func BenchmarkPreNormTransformerStackTrainingBoundary(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	f := newPreNormTransformerStackFixture()
	control := noPreNormTransformerStackMetalBackend{be}
	runPreNormTransformerStack(b, control, f)
	runPreNormTransformerStack(b, be, f)
	runOrderedPreNormTransformerStackBenchmarks(b, func(b *testing.B) {
		for range b.N {
			runPreNormTransformerStack(b, control, f)
		}
	}, func(b *testing.B) {
		for range b.N {
			runPreNormTransformerStack(b, be, f)
		}
	})
}

func TestPreNormTransformerStackViTMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(t)
	x, targets := preNormFFNViTInputs()
	controlY, controlTape := runPreNormFFNViTStep(t, noPreNormTransformerStackMetalBackend{be}, m, x, targets)
	candidateY, candidateTape := runPreNormFFNViTStep(t, be, m, x, targets)
	closePreNormAttentionTensor(t, "logits", candidateY, controlY)
	for i, parameter := range m.Params() {
		candidateGrad, controlGrad := candidateTape.Grad(parameter), controlTape.Grad(parameter)
		if candidateGrad == nil || controlGrad == nil {
			t.Fatalf("parameter %d has no gradient", i)
		}
		closePreNormAttentionTensor(t, "param-gradient", candidateGrad, controlGrad)
	}
}

func BenchmarkPreNormTransformerStackViTTrainStep(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(b)
	x, targets := preNormFFNViTInputs()
	control := noPreNormTransformerStackMetalBackend{be}
	runPreNormFFNViTStep(b, control, m, x, targets)
	runPreNormFFNViTStep(b, be, m, x, targets)
	runOrderedPreNormTransformerStackBenchmarks(b, func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, control, m, x, targets)
		}
		b.ReportMetric(float64(8*b.N)/b.Elapsed().Seconds(), "img/s")
	}, func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, be, m, x, targets)
		}
		b.ReportMetric(float64(8*b.N)/b.Elapsed().Seconds(), "img/s")
	})
}
