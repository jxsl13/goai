//go:build darwin && cgo

package metal_test

import (
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

const (
	preNormAttentionBatch = 8
	preNormAttentionSeq   = 65
	preNormAttentionDim   = 128
	preNormAttentionHeads = 4
	preNormAttentionEps   = 1e-5
)

type preNormAttentionFixture struct {
	x, gamma, beta, wq, wk, wv, wo, dOut *tensor.Tensor
}

func newPreNormAttentionFixture() preNormAttentionFixture {
	g := bench.RandF32(tensor.Shape{preNormAttentionDim}, 102)
	for i := range g.Numel() {
		g.Storage().F32()[i] = 1 + 0.1*g.Storage().F32()[i]
	}
	beta := bench.RandF32(tensor.Shape{preNormAttentionDim}, 103)
	for i := range beta.Numel() {
		beta.Storage().F32()[i] *= 0.1
	}
	wq := bench.RandF32(tensor.Shape{preNormAttentionDim, preNormAttentionDim}, 104)
	wk := bench.RandF32(tensor.Shape{preNormAttentionDim, preNormAttentionDim}, 105)
	wv := bench.RandF32(tensor.Shape{preNormAttentionDim, preNormAttentionDim}, 106)
	wo := bench.RandF32(tensor.Shape{preNormAttentionDim, preNormAttentionDim}, 107)
	for _, w := range []*tensor.Tensor{wq, wk, wv, wo} {
		for i := range w.Numel() {
			w.Storage().F32()[i] *= 0.02
		}
	}
	return preNormAttentionFixture{
		x:     bench.RandF32(tensor.Shape{preNormAttentionBatch * preNormAttentionSeq, preNormAttentionDim}, 101),
		gamma: g,
		beta:  beta,
		wq:    wq,
		wk:    wk,
		wv:    wv,
		wo:    wo,
		dOut:  bench.RandF32(tensor.Shape{preNormAttentionBatch * preNormAttentionSeq, preNormAttentionDim}, 108),
	}
}

func newFixtureMHA(t testing.TB, f preNormAttentionFixture) *nlp.MHA {
	t.Helper()
	m, err := nlp.NewMHA(preNormAttentionHeads, f.wq, f.wk, f.wv, f.wo)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func compositePreNormAttention(t testing.TB, be backend.Backend, f preNormAttentionFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	ctx := tape.Context()
	h, err := (&nn.LayerNorm{Gamma: f.gamma, Beta: f.beta, Eps: preNormAttentionEps}).Forward(ctx, f.x)
	if err != nil {
		t.Fatal(err)
	}
	h, err = newFixtureMHA(t, f).ForwardBatched(ctx, h, preNormAttentionBatch)
	if err != nil {
		t.Fatal(err)
	}
	out, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{f.x, h}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out[0], f.dOut); err != nil {
		t.Fatal(err)
	}
	return out[0], tape
}

func fusedPreNormAttention(t testing.TB, be backend.Backend, f preNormAttentionFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	out, err := backend.Execute(tape.Context(), backend.OpPreNormAttention, []*tensor.Tensor{
		f.x, f.gamma, f.beta, f.wq, f.wk, f.wv, f.wo,
	}, backend.PreNormAttentionAttrs{
		Heads: preNormAttentionHeads, Batch: preNormAttentionBatch, Eps: preNormAttentionEps,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out[0], f.dOut); err != nil {
		t.Fatal(err)
	}
	return out[0], tape
}

func closePreNormAttentionTensor(t *testing.T, name string, got, want *tensor.Tensor) {
	t.Helper()
	if got == nil || want == nil || !got.Shape().Equal(want.Shape()) || got.Dtype() != tensor.F32 || want.Dtype() != tensor.F32 {
		t.Fatalf("%s tensor mismatch: got=%v want=%v", name, got, want)
	}
	var maxRatio float64
	var maxIndex int
	var maxGot, maxWant float64
	for i := range got.Numel() {
		g, w := float64(got.Storage().F32()[i]), float64(want.Storage().F32()[i])
		diff := math.Abs(g - w)
		ratio := diff / math.Max(1, math.Abs(w))
		if ratio > maxRatio {
			maxRatio, maxIndex, maxGot, maxWant = ratio, i, g, w
		}
	}
	const tolerance = 2e-3
	if maxRatio > tolerance {
		t.Fatalf("%s[%d]: fused=%g composite=%g normalized-diff=%g", name, maxIndex, maxGot, maxWant, maxRatio)
	}
}

func TestPreNormAttentionMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	f := newPreNormAttentionFixture()
	inputs := []*tensor.Tensor{f.x, f.gamma, f.beta, f.wq, f.wk, f.wv, f.wo, f.dOut}
	before := make([][]float32, len(inputs))
	for i, in := range inputs {
		before[i] = slices.Clone(in.Storage().F32())
	}
	wantY, wantTape := compositePreNormAttention(t, be, f)
	gotY, gotTape := fusedPreNormAttention(t, be, f)
	closePreNormAttentionTensor(t, "Y", gotY, wantY)
	for _, pair := range []struct {
		name string
		in   *tensor.Tensor
	}{
		{"dX", f.x}, {"dGamma", f.gamma}, {"dBeta", f.beta}, {"dWq", f.wq},
		{"dWk", f.wk}, {"dWv", f.wv}, {"dWo", f.wo},
	} {
		closePreNormAttentionTensor(t, pair.name, gotTape.Grad(pair.in), wantTape.Grad(pair.in))
	}
	for i, in := range inputs {
		if !slices.Equal(in.Storage().F32(), before[i]) {
			t.Fatalf("input %d mutated", i)
		}
	}
}

func TestPreNormAttentionMetalRuntimeEpsilon(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	f := newPreNormAttentionFixture()
	ctx := backend.NewContext().WithBackend(be)
	var first []float32
	for _, eps := range []float64{1e-5, 2e-2} {
		h, err := (&nn.LayerNorm{Gamma: f.gamma, Beta: f.beta, Eps: eps}).Forward(ctx, f.x)
		if err != nil {
			t.Fatal(err)
		}
		h, err = newFixtureMHA(t, f).ForwardBatched(ctx, h, preNormAttentionBatch)
		if err != nil {
			t.Fatal(err)
		}
		want, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{f.x, h}, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := backend.Execute(ctx, backend.OpPreNormAttention, []*tensor.Tensor{
			f.x, f.gamma, f.beta, f.wq, f.wk, f.wv, f.wo,
		}, backend.PreNormAttentionAttrs{
			Heads: preNormAttentionHeads, Batch: preNormAttentionBatch, Eps: eps,
		})
		if err != nil {
			t.Fatal(err)
		}
		closePreNormAttentionTensor(t, "runtime-epsilon", got[0], want[0])
		if first == nil {
			first = slices.Clone(got[0].Storage().F32())
		} else if slices.Equal(first, got[0].Storage().F32()) {
			t.Fatal("changing runtime epsilon did not change the fused output")
		}
	}
}

func BenchmarkPreNormAttentionTrainingBoundary(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	f := newPreNormAttentionFixture()
	compositePreNormAttention(b, be, f)
	fusedPreNormAttention(b, be, f)
	b.Run("control", func(b *testing.B) {
		for range b.N {
			compositePreNormAttention(b, be, f)
		}
	})
	b.Run("candidate", func(b *testing.B) {
		for range b.N {
			fusedPreNormAttention(b, be, f)
		}
	})
}

type noPreNormAttentionMetalBackend struct{ backend.Backend }

func (b noPreNormAttentionMetalBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpPreNormAttention || op == backend.OpPreNormAttentionBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func runPreNormAttentionViTStep(t testing.TB, be backend.Backend, m *vision.ViT, x, targets *tensor.Tensor) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	return runPreNormFFNViTStep(t, be, m, x, targets)
}

func TestPreNormAttentionViTMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(t)
	x, targets := preNormFFNViTInputs()
	controlY, controlTape := runPreNormAttentionViTStep(t, noPreNormAttentionMetalBackend{be}, m, x, targets)
	candidateY, candidateTape := runPreNormAttentionViTStep(t, be, m, x, targets)
	closePreNormAttentionTensor(t, "logits", candidateY, controlY)
	for i, p := range m.Params() {
		candidateGrad, controlGrad := candidateTape.Grad(p), controlTape.Grad(p)
		if candidateGrad == nil || controlGrad == nil {
			t.Fatalf("parameter %d has no candidate gradient", i)
		}
		closePreNormAttentionTensor(t, "param-gradient", candidateGrad, controlGrad)
	}
}

func BenchmarkPreNormAttentionViTTrainStep(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(b)
	x, targets := preNormFFNViTInputs()
	control := noPreNormAttentionMetalBackend{be}
	runPreNormAttentionViTStep(b, control, m, x, targets)
	runPreNormAttentionViTStep(b, be, m, x, targets)
	b.Run("control", func(b *testing.B) {
		for range b.N {
			runPreNormAttentionViTStep(b, control, m, x, targets)
		}
		b.ReportMetric(float64(8*b.N)/b.Elapsed().Seconds(), "img/s")
	})
	b.Run("candidate", func(b *testing.B) {
		for range b.N {
			runPreNormAttentionViTStep(b, be, m, x, targets)
		}
		b.ReportMetric(float64(8*b.N)/b.Elapsed().Seconds(), "img/s")
	})
}
