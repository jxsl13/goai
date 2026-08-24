//go:build darwin && cgo

package metal_test

import (
	"os"
	"slices"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

const (
	sequenceClassifierBatch   = 8
	sequenceClassifierSeq     = 65
	sequenceClassifierDim     = 128
	sequenceClassifierClasses = 10
	sequenceClassifierEps     = 2e-5
)

type sequenceClassifierFixture struct {
	x, gamma, beta, w, bias, dOut *tensor.Tensor
}

func newSequenceClassifierFixture() sequenceClassifierFixture {
	gamma := bench.RandF32(tensor.Shape{sequenceClassifierDim}, 501)
	for i := range gamma.Numel() {
		gamma.Storage().F32()[i]++
	}
	return sequenceClassifierFixture{
		x:     bench.RandF32(tensor.Shape{sequenceClassifierBatch * sequenceClassifierSeq, sequenceClassifierDim}, 500),
		gamma: gamma,
		beta:  bench.RandF32(tensor.Shape{sequenceClassifierDim}, 502),
		w:     bench.RandF32(tensor.Shape{sequenceClassifierDim, sequenceClassifierClasses}, 503),
		bias:  bench.RandF32(tensor.Shape{sequenceClassifierClasses}, 504),
		dOut:  bench.RandF32(tensor.Shape{sequenceClassifierBatch, sequenceClassifierClasses}, 505),
	}
}

func compositeSequenceClassifier(t testing.TB, be backend.Backend, f sequenceClassifierFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	ctx := tape.Context()
	norm := &nn.LayerNorm{Gamma: f.gamma, Beta: f.beta, Eps: sequenceClassifierEps}
	head := &nn.Linear{W: f.w, B: f.bias}
	h, err := norm.Forward(ctx, f.x)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([]*tensor.Tensor, sequenceClassifierBatch)
	for batch := range sequenceClassifierBatch {
		out, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{h}, backend.SliceAttrs{
			Axis: 0, Start: batch * sequenceClassifierSeq, End: batch*sequenceClassifierSeq + 1,
		})
		if err != nil {
			t.Fatal(err)
		}
		rows[batch] = out[0]
	}
	selected, err := backend.Execute(ctx, backend.OpConcat, rows, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		t.Fatal(err)
	}
	logits, err := head.Forward(ctx, selected[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(logits, f.dOut); err != nil {
		t.Fatal(err)
	}
	return logits, tape
}

func fusedSequenceClassifier(t testing.TB, be backend.Backend, f sequenceClassifierFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	out, err := backend.Execute(tape.Context(), backend.OpLayerNormSequenceClassifier, []*tensor.Tensor{
		f.x, f.gamma, f.beta, f.w, f.bias,
	}, backend.LayerNormSequenceClassifierAttrs{Batch: sequenceClassifierBatch, Eps: sequenceClassifierEps})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out[0], f.dOut); err != nil {
		t.Fatal(err)
	}
	return out[0], tape
}

type noSequenceClassifierMetalBackend struct{ backend.Backend }

func (b noSequenceClassifierMetalBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpLayerNormSequenceClassifier || op == backend.OpLayerNormSequenceClassifierBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func runOrderedSequenceClassifierBenchmarks(b *testing.B, control, candidate func(*testing.B)) {
	b.Helper()
	if os.Getenv("GOAI_SEQUENCE_CLASSIFIER_CANDIDATE_FIRST") == "1" {
		b.Run("candidate", candidate)
		b.Run("control", control)
		return
	}
	b.Run("control", control)
	b.Run("candidate", candidate)
}

func TestLayerNormSequenceClassifierMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	f := newSequenceClassifierFixture()
	inputs := []*tensor.Tensor{f.x, f.gamma, f.beta, f.w, f.bias, f.dOut}
	before := make([][]float32, len(inputs))
	for i, input := range inputs {
		before[i] = slices.Clone(input.Storage().F32())
	}
	wantY, wantTape := compositeSequenceClassifier(t, be, f)
	gotY, gotTape := fusedSequenceClassifier(t, be, f)
	closePreNormAttentionTensor(t, "logits", gotY, wantY)
	for i, input := range inputs[:5] {
		got, want := gotTape.Grad(input), wantTape.Grad(input)
		if got == nil || want == nil {
			t.Fatalf("gradient %d is nil", i)
		}
		closePreNormAttentionTensor(t, "gradient", got, want)
	}
	for i, input := range inputs {
		if !slices.Equal(input.Storage().F32(), before[i]) {
			t.Fatalf("input %d mutated", i)
		}
	}
}

func BenchmarkLayerNormSequenceClassifierTrainingBoundary(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	f := newSequenceClassifierFixture()
	compositeSequenceClassifier(b, be, f)
	fusedSequenceClassifier(b, be, f)
	runOrderedSequenceClassifierBenchmarks(b, func(b *testing.B) {
		for range b.N {
			compositeSequenceClassifier(b, be, f)
		}
	}, func(b *testing.B) {
		for range b.N {
			fusedSequenceClassifier(b, be, f)
		}
	})
}

func TestLayerNormSequenceClassifierViTMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(t)
	x, targets := preNormFFNViTInputs()
	controlY, controlTape := runPreNormFFNViTStep(t, noSequenceClassifierMetalBackend{be}, m, x, targets)
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

func BenchmarkLayerNormSequenceClassifierViTTrainStep(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(b)
	x, targets := preNormFFNViTInputs()
	control := noSequenceClassifierMetalBackend{be}
	runPreNormFFNViTStep(b, control, m, x, targets)
	runPreNormFFNViTStep(b, be, m, x, targets)
	runOrderedSequenceClassifierBenchmarks(b, func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, control, m, x, targets)
		}
		b.ReportMetric(float64(sequenceClassifierBatch*b.N)/b.Elapsed().Seconds(), "img/s")
	}, func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, be, m, x, targets)
		}
		b.ReportMetric(float64(sequenceClassifierBatch*b.N)/b.Elapsed().Seconds(), "img/s")
	})
}
