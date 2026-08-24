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

type patchEmbedSequenceFixture struct {
	patches, class, pos, weight, bias, dOut *tensor.Tensor
	batch, patchesPerImage                  int
}

func newPatchEmbedSequenceFixture(dt tensor.Dtype) patchEmbedSequenceFixture {
	const batch, patches, patchDim, dim = 3, 5, 7, 11
	return patchEmbedSequenceFixture{
		patches:         tensor.Randn(dt, 701, tensor.Shape{batch * patches, patchDim}),
		class:           tensor.Randn(dt, 702, tensor.Shape{1, dim}),
		pos:             tensor.Randn(dt, 703, tensor.Shape{patches + 1, dim}),
		weight:          tensor.Randn(dt, 704, tensor.Shape{patchDim, dim}),
		bias:            tensor.Randn(dt, 705, tensor.Shape{dim}),
		dOut:            tensor.Randn(dt, 706, tensor.Shape{batch * (patches + 1), dim}),
		batch:           batch,
		patchesPerImage: patches,
	}
}

func compositePatchEmbedSequence(t *testing.T, f patchEmbedSequenceFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(backend.Reference())
	ctx := tape.Context()
	linear := &nn.Linear{W: f.weight, B: f.bias}
	projected, err := linear.Forward(ctx, f.patches)
	if err != nil {
		t.Fatal(err)
	}
	seqs := make([]*tensor.Tensor, f.batch)
	for batch := range f.batch {
		rows, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{projected}, backend.SliceAttrs{
			Axis: 0, Start: batch * f.patchesPerImage, End: (batch + 1) * f.patchesPerImage,
		})
		if err != nil {
			t.Fatal(err)
		}
		seq, err := backend.Execute(ctx, backend.OpConcat, []*tensor.Tensor{f.class, rows[0]}, backend.ConcatAttrs{Axis: 0})
		if err != nil {
			t.Fatal(err)
		}
		seq, err = backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{seq[0], f.pos}, nil)
		if err != nil {
			t.Fatal(err)
		}
		seqs[batch] = seq[0]
	}
	packed, err := backend.Execute(ctx, backend.OpConcat, seqs, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(packed[0], f.dOut); err != nil {
		t.Fatal(err)
	}
	return packed[0], tape
}

func fusedPatchEmbedSequence(t *testing.T, f patchEmbedSequenceFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(backend.Reference())
	out, err := backend.Execute(tape.Context(), backend.OpPatchEmbedSequence, []*tensor.Tensor{
		f.patches, f.class, f.pos, f.weight, f.bias,
	}, backend.PatchEmbedSequenceAttrs{Batch: f.batch})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out[0], f.dOut); err != nil {
		t.Fatal(err)
	}
	return out[0], tape
}

func closePatchEmbedSequence(t *testing.T, name string, got, want *tensor.Tensor, atol, rtol float64) {
	t.Helper()
	if got == nil || want == nil || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s shape mismatch: got=%v want=%v", name, got, want)
	}
	for i := range got.Numel() {
		index := tensor.Unravel(i, got.Shape())
		gv, wv := got.AtF64(index...), want.AtF64(index...)
		if math.Abs(gv-wv) > atol+rtol*math.Abs(wv) {
			t.Fatalf("%s[%d]: fused=%g composite=%g", name, i, gv, wv)
		}
	}
}

func TestPatchEmbedSequenceReferenceMatchesComposite(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		t.Run(dt.String(), func(t *testing.T) {
			f := newPatchEmbedSequenceFixture(dt)
			inputs := []*tensor.Tensor{f.patches, f.class, f.pos, f.weight, f.bias, f.dOut}
			total := 0
			for _, input := range inputs {
				total += input.Numel()
			}
			before := make([][]float64, len(inputs))
			beforeSlab := make([]float64, total)
			offset := 0
			for i, input := range inputs {
				n := input.Numel()
				before[i] = beforeSlab[offset : offset+n : offset+n]
				offset += n
				for j := range input.Numel() {
					before[i][j] = input.AtF64(tensor.Unravel(j, input.Shape())...)
				}
			}
			wantY, wantTape := compositePatchEmbedSequence(t, f)
			gotY, gotTape := fusedPatchEmbedSequence(t, f)
			atol, rtol := 1e-11, 1e-10
			if dt == tensor.F32 {
				atol, rtol = 2e-5, 2e-5
			}
			closePatchEmbedSequence(t, "output", gotY, wantY, atol, rtol)
			for _, input := range inputs[:5] {
				closePatchEmbedSequence(t, "gradient", gotTape.Grad(input), wantTape.Grad(input), atol, rtol)
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
