//go:build darwin && cgo

package metal_test

import (
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

const (
	preNormFFNRows   = 520
	preNormFFNDim    = 128
	preNormFFNHidden = 512
	preNormFFNEps    = 1e-5
)

type preNormFFNFixture struct {
	x, gamma, beta, w1, b1, w2, b2, dOut *tensor.Tensor
}

func newPreNormFFNFixture() preNormFFNFixture {
	g := bench.RandF32(tensor.Shape{preNormFFNDim}, 2)
	for i := range g.Numel() {
		g.Storage().F32()[i]++
	}
	return preNormFFNFixture{
		x:     bench.RandF32(tensor.Shape{preNormFFNRows, preNormFFNDim}, 1),
		gamma: g,
		beta:  bench.RandF32(tensor.Shape{preNormFFNDim}, 3),
		w1:    bench.RandF32(tensor.Shape{preNormFFNDim, preNormFFNHidden}, 4),
		b1:    bench.RandF32(tensor.Shape{preNormFFNHidden}, 5),
		w2:    bench.RandF32(tensor.Shape{preNormFFNHidden, preNormFFNDim}, 6),
		b2:    bench.RandF32(tensor.Shape{preNormFFNDim}, 7),
		dOut:  bench.RandF32(tensor.Shape{preNormFFNRows, preNormFFNDim}, 8),
	}
}

func compositePreNormFFN(t testing.TB, be backend.Backend, f preNormFFNFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	ctx := tape.Context()
	ln := &nn.LayerNorm{Gamma: f.gamma, Beta: f.beta, Eps: preNormFFNEps}
	up := &nn.Linear{W: f.w1, B: f.b1}
	down := &nn.Linear{W: f.w2, B: f.b2}
	h, err := ln.Forward(ctx, f.x)
	if err != nil {
		t.Fatal(err)
	}
	if h, err = up.Forward(ctx, h); err != nil {
		t.Fatal(err)
	}
	out, err := backend.Execute(ctx, backend.OpGELU, []*tensor.Tensor{h}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if h, err = down.Forward(ctx, out[0]); err != nil {
		t.Fatal(err)
	}
	out, err = backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{f.x, h}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out[0], f.dOut); err != nil {
		t.Fatal(err)
	}
	return out[0], tape
}

func fusedPreNormFFN(t testing.TB, be backend.Backend, f preNormFFNFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	out, err := backend.Execute(tape.Context(), backend.OpPreNormFFN, []*tensor.Tensor{
		f.x, f.gamma, f.beta, f.w1, f.b1, f.w2, f.b2,
	}, backend.NormAttrs{Eps: preNormFFNEps})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out[0], f.dOut); err != nil {
		t.Fatal(err)
	}
	return out[0], tape
}

func closePreNormFFNTensor(t *testing.T, name string, got, want *tensor.Tensor) {
	t.Helper()
	for i := range got.Numel() {
		g, w := float64(got.Storage().F32()[i]), float64(want.Storage().F32()[i])
		tol := 3e-4 + 5e-4*math.Abs(w)
		if math.Abs(g-w) > tol {
			t.Fatalf("%s[%d]: fused=%g composite=%g tol=%g", name, i, g, w, tol)
		}
	}
}

func TestPreNormFFNMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	f := newPreNormFFNFixture()
	inputs := []*tensor.Tensor{f.x, f.gamma, f.beta, f.w1, f.b1, f.w2, f.b2, f.dOut}
	before := make([][]float32, len(inputs))
	for i, in := range inputs {
		before[i] = slices.Clone(in.Storage().F32())
	}
	wantY, wantTape := compositePreNormFFN(t, be, f)
	gotY, gotTape := fusedPreNormFFN(t, be, f)
	closePreNormFFNTensor(t, "Y", gotY, wantY)
	for _, pair := range []struct {
		name string
		in   *tensor.Tensor
	}{
		{"dX", f.x}, {"dGamma", f.gamma}, {"dBeta", f.beta}, {"dW1", f.w1},
		{"dB1", f.b1}, {"dW2", f.w2}, {"dB2", f.b2},
	} {
		closePreNormFFNTensor(t, pair.name, gotTape.Grad(pair.in), wantTape.Grad(pair.in))
	}
	for i, in := range inputs {
		for j, v := range before[i] {
			if in.Storage().F32()[j] != v {
				t.Fatalf("input %d mutated at %d", i, j)
			}
		}
	}
}

func BenchmarkPreNormFFNTrainingBoundary(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	f := newPreNormFFNFixture()
	compositePreNormFFN(b, be, f)
	fusedPreNormFFN(b, be, f)
	b.Run("control", func(b *testing.B) {
		for range b.N {
			compositePreNormFFN(b, be, f)
		}
	})
	b.Run("candidate", func(b *testing.B) {
		for range b.N {
			fusedPreNormFFN(b, be, f)
		}
	})
}

type noPreNormMetalBackend struct{ backend.Backend }

func (b noPreNormMetalBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpPreNormFFN || op == backend.OpPreNormFFNBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func newPreNormFFNViT(t testing.TB) *vision.ViT {
	t.Helper()
	m, err := vision.NewViT(3, 32, 10, 1,
		vision.WithViTPatch(4), vision.WithViTDim(128),
		vision.WithViTDepth(4), vision.WithViTHeads(4))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func preNormFFNViTInputs() (*tensor.Tensor, *tensor.Tensor) {
	x := bench.RandF32(tensor.Shape{8, 3, 32, 32}, 21)
	targets := tensor.New(tensor.F32, tensor.Shape{8})
	for i := range 8 {
		targets.SetF64(float64(i%10), i)
	}
	return x, targets
}

func runPreNormFFNViTStep(t testing.TB, be backend.Backend, m *vision.ViT, x, targets *tensor.Tensor) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	logits, err := m.Forward(tape.Context(), x)
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
	return logits, tape
}

func TestPreNormFFNViTMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(t)
	x, targets := preNormFFNViTInputs()
	controlY, controlTape := runPreNormFFNViTStep(t, noPreNormMetalBackend{be}, m, x, targets)
	candidateY, candidateTape := runPreNormFFNViTStep(t, be, m, x, targets)
	closePreNormFFNTensor(t, "logits", candidateY, controlY)
	for i, p := range m.Params() {
		candidateGrad, controlGrad := candidateTape.Grad(p), controlTape.Grad(p)
		if candidateGrad == nil || controlGrad == nil {
			t.Fatalf("parameter %d has no fused gradient", i)
		}
		closePreNormFFNTensor(t, "param-gradient", candidateGrad, controlGrad)
	}
}

func BenchmarkPreNormFFNViTTrainStep(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(b)
	x, targets := preNormFFNViTInputs()
	control := noPreNormMetalBackend{be}
	runPreNormFFNViTStep(b, control, m, x, targets)
	runPreNormFFNViTStep(b, be, m, x, targets)
	b.Run("control", func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, control, m, x, targets)
		}
		b.ReportMetric(float64(8*b.N)/b.Elapsed().Seconds(), "img/s")
	})
	b.Run("candidate", func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, be, m, x, targets)
		}
		b.ReportMetric(float64(8*b.N)/b.Elapsed().Seconds(), "img/s")
	})
}
