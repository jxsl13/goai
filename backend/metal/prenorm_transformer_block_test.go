//go:build darwin && cgo

package metal_test

import (
	"os"
	"slices"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type preNormTransformerBlockFixture struct {
	x, gamma1, beta1, wq, wk, wv, wo *tensor.Tensor
	gamma2, beta2, w1, b1, w2, b2    *tensor.Tensor
	dOut                             *tensor.Tensor
}

func newPreNormTransformerBlockFixture() preNormTransformerBlockFixture {
	attention := newPreNormAttentionFixture()
	ffn := newPreNormFFNFixture()
	return preNormTransformerBlockFixture{
		x: attention.x, gamma1: attention.gamma, beta1: attention.beta,
		wq: attention.wq, wk: attention.wk, wv: attention.wv, wo: attention.wo,
		gamma2: ffn.gamma, beta2: ffn.beta, w1: ffn.w1, b1: ffn.b1, w2: ffn.w2, b2: ffn.b2,
		dOut: ffn.dOut,
	}
}

func (f preNormTransformerBlockFixture) inputs() []*tensor.Tensor {
	return []*tensor.Tensor{
		f.x, f.gamma1, f.beta1, f.wq, f.wk, f.wv, f.wo,
		f.gamma2, f.beta2, f.w1, f.b1, f.w2, f.b2,
	}
}

func (f preNormTransformerBlockFixture) layers(t testing.TB, eps1, eps2 float64) (*nlp.MHA, *nn.LayerNorm, *nn.LayerNorm, *nn.Linear, *nn.Linear) {
	t.Helper()
	mha, err := nlp.NewMHA(preNormAttentionHeads, f.wq, f.wk, f.wv, f.wo)
	if err != nil {
		t.Fatal(err)
	}
	return mha,
		&nn.LayerNorm{Gamma: f.gamma1, Beta: f.beta1, Eps: eps1},
		&nn.LayerNorm{Gamma: f.gamma2, Beta: f.beta2, Eps: eps2},
		&nn.Linear{W: f.w1, B: f.b1}, &nn.Linear{W: f.w2, B: f.b2}
}

func runPreNormTransformerBlock(t testing.TB, be backend.Backend, f preNormTransformerBlockFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	mha, norm1, norm2, up, down := f.layers(t, 2e-5, 3e-5)
	out, err := nlp.ForwardPreNormTransformerBlock(tape.Context(), f.x, mha, norm1, norm2, up, down, preNormAttentionBatch)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out, f.dOut); err != nil {
		t.Fatal(err)
	}
	return out, tape
}

func forwardPreNormTransformerBlock(t testing.TB, be backend.Backend, f preNormTransformerBlockFixture, eps1, eps2 float64) *tensor.Tensor {
	t.Helper()
	mha, norm1, norm2, up, down := f.layers(t, eps1, eps2)
	out, err := nlp.ForwardPreNormTransformerBlock(
		backend.NewContext().WithBackend(be), f.x, mha, norm1, norm2, up, down, preNormAttentionBatch)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func backwardPreNormTransformerBlockControl(t testing.TB, be backend.Backend, f preNormTransformerBlockFixture, attentionOut *tensor.Tensor) {
	t.Helper()
	ctx := backend.NewContext().WithBackend(be)
	ffnGrad, err := backend.Execute(ctx, backend.OpPreNormFFNBackward, []*tensor.Tensor{
		attentionOut, f.gamma2, f.beta2, f.w1, f.b1, f.w2, f.b2, f.dOut,
	}, backend.NormAttrs{Eps: 3e-5})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := backend.Execute(ctx, backend.OpPreNormAttentionBackward, []*tensor.Tensor{
		f.x, f.gamma1, f.beta1, f.wq, f.wk, f.wv, f.wo, ffnGrad[0],
	}, backend.PreNormAttentionAttrs{Heads: preNormAttentionHeads, Batch: preNormAttentionBatch, Eps: 2e-5}); err != nil {
		t.Fatal(err)
	}
}

func backwardPreNormTransformerBlockCandidate(t testing.TB, be backend.Backend, f preNormTransformerBlockFixture) {
	t.Helper()
	if _, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpPreNormTransformerBlockBackward,
		append(f.inputs(), f.dOut), backend.PreNormTransformerBlockAttrs{
			Heads: preNormAttentionHeads, Batch: preNormAttentionBatch, Eps1: 2e-5, Eps2: 3e-5,
		}); err != nil {
		t.Fatal(err)
	}
}

type noPreNormTransformerBlockMetalBackend struct{ backend.Backend }

func runOrderedPreNormTransformerBlockBenchmarks(
	b *testing.B,
	control func(*testing.B),
	candidate func(*testing.B),
) {
	b.Helper()
	if os.Getenv("GOAI_PRENORM_BLOCK_CANDIDATE_FIRST") == "1" {
		b.Run("candidate", candidate)
		b.Run("control", control)
		return
	}
	b.Run("control", control)
	b.Run("candidate", candidate)
}

func (b noPreNormTransformerBlockMetalBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpPreNormTransformerBlock || op == backend.OpPreNormTransformerBlockBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func TestPreNormTransformerBlockMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	f := newPreNormTransformerBlockFixture()
	allInputs := append(slices.Clone(f.inputs()), f.dOut)
	before := make([][]float32, len(allInputs))
	for i, input := range allInputs {
		before[i] = slices.Clone(input.Storage().F32())
	}
	wantY, wantTape := runPreNormTransformerBlock(t, noPreNormTransformerBlockMetalBackend{be}, f)
	gotY, gotTape := runPreNormTransformerBlock(t, be, f)
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

func TestPreNormTransformerBlockMetalRuntimeEpsilons(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	f := newPreNormTransformerBlockFixture()
	control := noPreNormTransformerBlockMetalBackend{be}
	var first []float32
	for _, eps := range [][2]float64{{1e-5, 1e-5}, {2e-2, 3e-2}} {
		want := forwardPreNormTransformerBlock(t, control, f, eps[0], eps[1])
		got := forwardPreNormTransformerBlock(t, be, f, eps[0], eps[1])
		closePreNormAttentionTensor(t, "runtime-epsilons", got, want)
		if first == nil {
			first = slices.Clone(got.Storage().F32())
		} else if slices.Equal(first, got.Storage().F32()) {
			t.Fatal("changing both runtime epsilon feeds did not change output")
		}
	}
}

func BenchmarkPreNormTransformerBlockTrainingBoundary(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	f := newPreNormTransformerBlockFixture()
	control := noPreNormTransformerBlockMetalBackend{be}
	runPreNormTransformerBlock(b, control, f)
	runPreNormTransformerBlock(b, be, f)
	runOrderedPreNormTransformerBlockBenchmarks(b, func(b *testing.B) {
		for range b.N {
			runPreNormTransformerBlock(b, control, f)
		}
	}, func(b *testing.B) {
		for range b.N {
			runPreNormTransformerBlock(b, be, f)
		}
	})
}

func BenchmarkPreNormTransformerBlockForwardBoundary(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	f := newPreNormTransformerBlockFixture()
	control := noPreNormTransformerBlockMetalBackend{be}
	forwardPreNormTransformerBlock(b, control, f, 2e-5, 3e-5)
	forwardPreNormTransformerBlock(b, be, f, 2e-5, 3e-5)
	runOrderedPreNormTransformerBlockBenchmarks(b, func(b *testing.B) {
		for range b.N {
			forwardPreNormTransformerBlock(b, control, f, 2e-5, 3e-5)
		}
	}, func(b *testing.B) {
		for range b.N {
			forwardPreNormTransformerBlock(b, be, f, 2e-5, 3e-5)
		}
	})
}

func BenchmarkPreNormTransformerBlockBackwardBoundary(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	f := newPreNormTransformerBlockFixture()
	attentionOut, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpPreNormAttention, []*tensor.Tensor{
		f.x, f.gamma1, f.beta1, f.wq, f.wk, f.wv, f.wo,
	}, backend.PreNormAttentionAttrs{Heads: preNormAttentionHeads, Batch: preNormAttentionBatch, Eps: 2e-5})
	if err != nil {
		b.Fatal(err)
	}
	backwardPreNormTransformerBlockControl(b, be, f, attentionOut[0])
	backwardPreNormTransformerBlockCandidate(b, be, f)
	runOrderedPreNormTransformerBlockBenchmarks(b, func(b *testing.B) {
		for range b.N {
			backwardPreNormTransformerBlockControl(b, be, f, attentionOut[0])
		}
	}, func(b *testing.B) {
		for range b.N {
			backwardPreNormTransformerBlockCandidate(b, be, f)
		}
	})
}

func TestPreNormTransformerBlockViTMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(t)
	x, targets := preNormFFNViTInputs()
	controlY, controlTape := runPreNormFFNViTStep(t, noPreNormTransformerBlockMetalBackend{be}, m, x, targets)
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

func BenchmarkPreNormTransformerBlockViTTrainStep(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(b)
	x, targets := preNormFFNViTInputs()
	control := noPreNormTransformerBlockMetalBackend{be}
	runPreNormFFNViTStep(b, control, m, x, targets)
	runPreNormFFNViTStep(b, be, m, x, targets)
	runOrderedPreNormTransformerBlockBenchmarks(b, func(b *testing.B) {
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
