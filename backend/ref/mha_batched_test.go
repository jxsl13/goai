package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func TestMHABatchedMatchesIndependentSequences(t *testing.T) {
	const batch, seq, dm, heads = 3, 5, 8, 2
	q := mkQKV(11, batch*seq, dm)
	k := mkQKV(12, batch*seq, dm)
	v := mkQKV(13, batch*seq, dm)
	g := mkQKV(14, batch*seq, dm)
	ctx := backend.NewContext().WithBackend(backend.Reference())
	attrs := backend.AttnAttrs{Heads: heads, Batch: batch, Causal: true}
	got, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q, k, v}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	gotGrad, err := backend.Execute(ctx, backend.OpMHABackward, []*tensor.Tensor{q, k, v, g}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	one := attrs
	one.Batch = 1
	for b := range batch {
		q0, q1 := b*seq, (b+1)*seq
		qb, _ := q.Slice(0, q0, q1)
		kb, _ := k.Slice(0, q0, q1)
		vb, _ := v.Slice(0, q0, q1)
		gb, _ := g.Slice(0, q0, q1)
		want, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{qb, kb, vb}, one)
		if err != nil {
			t.Fatal(err)
		}
		wantGrad, err := backend.Execute(ctx, backend.OpMHABackward, []*tensor.Tensor{qb, kb, vb, gb}, one)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want[0].Numel() {
			if math.Float64bits(got[0].AtF64(q0+i/dm, i%dm)) != math.Float64bits(want[0].AtF64(i/dm, i%dm)) {
				t.Fatalf("batch %d forward element %d crossed a sequence boundary", b, i)
			}
		}
		for output := range gotGrad {
			width := gotGrad[output].Shape()[1]
			for i := range wantGrad[output].Numel() {
				if math.Float64bits(gotGrad[output].AtF64(q0+i/width, i%width)) != math.Float64bits(wantGrad[output].AtF64(i/width, i%width)) {
					t.Fatalf("batch %d gradient %d element %d crossed a sequence boundary", b, output, i)
				}
			}
		}
	}
}

func TestMHABatchedRejectsUnevenRows(t *testing.T) {
	q := mkQKV(21, 10, 8)
	attrs := backend.AttnAttrs{Heads: 2, Batch: 3}
	if _, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMHA, []*tensor.Tensor{q, q, q}, attrs); err == nil {
		t.Fatal("uneven packed batch unexpectedly accepted")
	}
}
