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
	patchSequenceBatch   = 8
	patchSequencePatches = 64
	patchSequenceInput   = 48
	patchSequenceDim     = 128
)

type patchSequenceFixture struct {
	patches, class, pos, weight, bias, dOut *tensor.Tensor
}

func newPatchSequenceFixture() patchSequenceFixture {
	return patchSequenceFixture{
		patches: bench.RandF32(tensor.Shape{patchSequenceBatch * patchSequencePatches, patchSequenceInput}, 601),
		class:   bench.RandF32(tensor.Shape{1, patchSequenceDim}, 602),
		pos:     bench.RandF32(tensor.Shape{patchSequencePatches + 1, patchSequenceDim}, 603),
		weight:  bench.RandF32(tensor.Shape{patchSequenceInput, patchSequenceDim}, 604),
		bias:    bench.RandF32(tensor.Shape{patchSequenceDim}, 605),
		dOut:    bench.RandF32(tensor.Shape{patchSequenceBatch * (patchSequencePatches + 1), patchSequenceDim}, 606),
	}
}

func compositePatchSequence(t testing.TB, be backend.Backend, f patchSequenceFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	ctx := tape.Context()
	projected, err := (&nn.Linear{W: f.weight, B: f.bias}).Forward(ctx, f.patches)
	if err != nil {
		t.Fatal(err)
	}
	seqs := make([]*tensor.Tensor, patchSequenceBatch)
	for batch := range patchSequenceBatch {
		rows, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{projected}, backend.SliceAttrs{
			Axis: 0, Start: batch * patchSequencePatches, End: (batch + 1) * patchSequencePatches,
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

func fusedPatchSequence(t testing.TB, be backend.Backend, f patchSequenceFixture) (*tensor.Tensor, *autograd.Tape) {
	t.Helper()
	tape := autograd.NewTapeOn(be)
	out, err := backend.Execute(tape.Context(), backend.OpPatchEmbedSequence, []*tensor.Tensor{
		f.patches, f.class, f.pos, f.weight, f.bias,
	}, backend.PatchEmbedSequenceAttrs{Batch: patchSequenceBatch})
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.BackwardGrad(out[0], f.dOut); err != nil {
		t.Fatal(err)
	}
	return out[0], tape
}

type noPatchSequenceMetalBackend struct{ backend.Backend }

func (b noPatchSequenceMetalBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpPatchEmbedSequence || op == backend.OpPatchEmbedSequenceBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func runOrderedPatchSequenceBenchmarks(b *testing.B, control, candidate func(*testing.B)) {
	b.Helper()
	if os.Getenv("GOAI_PATCH_SEQUENCE_CANDIDATE_FIRST") == "1" {
		b.Run("candidate", candidate)
		b.Run("control", control)
		return
	}
	b.Run("control", control)
	b.Run("candidate", candidate)
}

func TestPatchEmbedSequenceMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	f := newPatchSequenceFixture()
	inputs := []*tensor.Tensor{f.patches, f.class, f.pos, f.weight, f.bias, f.dOut}
	before := make([][]float32, len(inputs))
	for i, input := range inputs {
		before[i] = slices.Clone(input.Storage().F32())
	}
	wantY, wantTape := compositePatchSequence(t, be, f)
	gotY, gotTape := fusedPatchSequence(t, be, f)
	closePreNormAttentionTensor(t, "packed", gotY, wantY)
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

func TestPatchEmbedSequenceViTMetalParity(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(t)
	x, targets := preNormFFNViTInputs()
	beforeX := slices.Clone(x.Storage().F32())
	controlY, controlTape := runPreNormFFNViTStep(t, noPatchSequenceMetalBackend{be}, m, x, targets)
	candidateY, candidateTape := runPreNormFFNViTStep(t, be, m, x, targets)
	closePreNormAttentionTensor(t, "logits", candidateY, controlY)
	for i, parameter := range m.Params() {
		candidateGrad, controlGrad := candidateTape.Grad(parameter), controlTape.Grad(parameter)
		if candidateGrad == nil || controlGrad == nil {
			t.Fatalf("parameter %d has no gradient", i)
		}
		closePreNormAttentionTensor(t, "parameter-gradient", candidateGrad, controlGrad)
	}
	if !slices.Equal(x.Storage().F32(), beforeX) {
		t.Fatal("ViT input mutated")
	}
}

func TestPatchEmbedSequenceMetalOffsetFallback(t *testing.T) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("Metal unavailable")
	}
	f := newPatchSequenceFixture()
	base := bench.RandF32(tensor.Shape{patchSequenceBatch*patchSequencePatches + 1, patchSequenceInput}, 607)
	view, err := base.Slice(0, 1, patchSequenceBatch*patchSequencePatches+1)
	if err != nil {
		t.Fatal(err)
	}
	if view.Offset() == 0 {
		t.Fatal("fallback fixture did not create an offset view")
	}
	inputs := []*tensor.Tensor{view, f.class, f.pos, f.weight, f.bias}
	attrs := backend.PatchEmbedSequenceAttrs{Batch: patchSequenceBatch}
	want, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpPatchEmbedSequence, inputs, attrs)
	if err != nil {
		t.Fatal(err)
	}
	got, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpPatchEmbedSequence, inputs, attrs)
	if err != nil {
		t.Fatal(err)
	}
	closePreNormAttentionTensor(t, "fallback-output", got[0], want[0])
	backwardInputs := append(slices.Clone(inputs), f.dOut)
	wantGrad, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpPatchEmbedSequenceBackward, backwardInputs, attrs)
	if err != nil {
		t.Fatal(err)
	}
	gotGrad, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpPatchEmbedSequenceBackward, backwardInputs, attrs)
	if err != nil {
		t.Fatal(err)
	}
	for i := range gotGrad {
		closePreNormAttentionTensor(t, "fallback-gradient", gotGrad[i], wantGrad[i])
	}
}

func BenchmarkPatchEmbedSequenceTrainingBoundary(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	f := newPatchSequenceFixture()
	compositePatchSequence(b, be, f)
	fusedPatchSequence(b, be, f)
	runOrderedPatchSequenceBenchmarks(b, func(b *testing.B) {
		for range b.N {
			compositePatchSequence(b, be, f)
		}
	}, func(b *testing.B) {
		for range b.N {
			fusedPatchSequence(b, be, f)
		}
	})
}

func BenchmarkPatchEmbedSequenceViTTrainStep(b *testing.B) {
	be, ok := backend.Get(backend.Metal)
	if !ok {
		b.Skip("Metal unavailable")
	}
	m := newPreNormFFNViT(b)
	x, targets := preNormFFNViTInputs()
	control := noPatchSequenceMetalBackend{be}
	runPreNormFFNViTStep(b, control, m, x, targets)
	runPreNormFFNViTStep(b, be, m, x, targets)
	runOrderedPatchSequenceBenchmarks(b, func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, control, m, x, targets)
		}
		b.ReportMetric(float64(patchSequenceBatch*b.N)/b.Elapsed().Seconds(), "img/s")
	}, func(b *testing.B) {
		for range b.N {
			runPreNormFFNViTStep(b, be, m, x, targets)
		}
		b.ReportMetric(float64(patchSequenceBatch*b.N)/b.Elapsed().Seconds(), "img/s")
	})
}
