package ref_test

import (
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type layerNormSequenceClassifierFixture struct {
	x, gamma, beta, w, bias, dOut *tensor.Tensor
	batch, seq                    int
	eps                           float64
}

func newLayerNormSequenceClassifierFixture(dt tensor.Dtype) layerNormSequenceClassifierFixture {
	const batch, seq, dim, classes = 3, 5, 7, 4
	f := layerNormSequenceClassifierFixture{
		x:     tensor.Randn(dt, 401, tensor.Shape{batch * seq, dim}),
		gamma: tensor.Randn(dt, 402, tensor.Shape{dim}),
		beta:  tensor.Randn(dt, 403, tensor.Shape{dim}),
		w:     tensor.Randn(dt, 404, tensor.Shape{dim, classes}),
		bias:  tensor.Randn(dt, 405, tensor.Shape{classes}),
		dOut:  tensor.Randn(dt, 406, tensor.Shape{batch, classes}),
		batch: batch,
		seq:   seq,
		eps:   2e-5,
	}
	for i := range f.gamma.Numel() {
		f.gamma.SetF64(f.gamma.AtF64(i)+1, i)
	}
	return f
}

func compositeLayerNormSequenceClassifier(t *testing.T, f layerNormSequenceClassifierFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(backend.Reference())
	ctx := tape.Context()
	norm := &nn.LayerNorm{Gamma: f.gamma, Beta: f.beta, Eps: f.eps}
	head := &nn.Linear{W: f.w, B: f.bias}
	h, err := norm.Forward(ctx, f.x)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([]*tensor.Tensor, f.batch)
	for batch := range f.batch {
		out, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{h}, backend.SliceAttrs{
			Axis: 0, Start: batch * f.seq, End: batch*f.seq + 1,
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

func fusedLayerNormSequenceClassifier(t *testing.T, f layerNormSequenceClassifierFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(backend.Reference())
	out, err := backend.Execute(tape.Context(), backend.OpLayerNormSequenceClassifier, []*tensor.Tensor{
		f.x, f.gamma, f.beta, f.w, f.bias,
	}, backend.LayerNormSequenceClassifierAttrs{Batch: f.batch, Eps: f.eps})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out[0], f.dOut); err != nil {
		t.Fatal(err)
	}
	return out[0], tape
}

func closeLayerNormSequenceClassifier(t *testing.T, name string, got, want *tensor.Tensor, atol, rtol float64) {
	t.Helper()
	if got == nil || want == nil || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s shape mismatch: got=%v want=%v", name, got, want)
	}
	for i := range got.Numel() {
		gv, wv := got.AtF64(tensor.Unravel(i, got.Shape())...), want.AtF64(tensor.Unravel(i, want.Shape())...)
		if math.Abs(gv-wv) > atol+rtol*math.Abs(wv) {
			t.Fatalf("%s[%d]: fused=%g composite=%g", name, i, gv, wv)
		}
	}
}

func TestLayerNormSequenceClassifierReferenceMatchesComposite(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		t.Run(dt.String(), func(t *testing.T) {
			f := newLayerNormSequenceClassifierFixture(dt)
			inputs := []*tensor.Tensor{f.x, f.gamma, f.beta, f.w, f.bias, f.dOut}
			before := make([][]float64, len(inputs))
			for i, input := range inputs {
				before[i] = make([]float64, input.Numel())
				for j := range input.Numel() {
					before[i][j] = input.AtF64(tensor.Unravel(j, input.Shape())...)
				}
			}
			wantY, wantTape := compositeLayerNormSequenceClassifier(t, f)
			gotY, gotTape := fusedLayerNormSequenceClassifier(t, f)
			atol, rtol := 1e-11, 1e-10
			if dt == tensor.F32 {
				atol, rtol = 2e-5, 2e-5
			}
			closeLayerNormSequenceClassifier(t, "logits", gotY, wantY, atol, rtol)
			for _, input := range inputs[:5] {
				closeLayerNormSequenceClassifier(t, "gradient", gotTape.Grad(input), wantTape.Grad(input), atol, rtol)
			}
			for i, input := range inputs {
				got := make([]float64, input.Numel())
				for j := range input.Numel() {
					got[j] = input.AtF64(tensor.Unravel(j, input.Shape())...)
				}
				if !slices.Equal(got, before[i]) {
					t.Fatalf("input %d mutated", i)
				}
			}
		})
	}
}
