package autograd_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func randFA(shape tensor.Shape, seed int) *tensor.Tensor {
	t := tensor.New(tensor.F64, shape)
	for i := range t.Numel() {
		idx := tensor.Unravel(i, shape)
		t.SetF64(math.Sin(float64(i*7+seed)*0.3+0.1), idx...)
	}
	return t
}

// §V16 tier-1: FlashAttention-2 produces the SAME output as the naive softmax
// attention (OpMHA) at f64 1e-12, for causal and non-causal, single and multi
// head, and across key-block sizes (tiling actually exercised: block < seq).
func TestFlashAttnMatchesMHA(t *testing.T) {
	for _, seq := range []int{5, 8} {
		for _, heads := range []int{1, 2} {
			dm := heads * 4
			q, k, v := randFA(tensor.Shape{seq, dm}, 1), randFA(tensor.Shape{seq, dm}, 2), randFA(tensor.Shape{seq, dm}, 3)
			for _, causal := range []bool{false, true} {
				ref, err := backend.Execute(backend.NewContext(), backend.OpMHA,
					[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: heads, Causal: causal})
				if err != nil {
					t.Fatal(err)
				}
				for _, block := range []int{1, 2, 3, seq} {
					got, err := backend.Execute(backend.NewContext(), backend.OpFlashAttn,
						[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: heads, Causal: causal, Block: block})
					if err != nil {
						t.Fatal(err)
					}
					for idx := range got[0].Numel() {
						id := tensor.Unravel(idx, got[0].Shape())
						a, b := got[0].AtF64(id...), ref[0].AtF64(id...)
						if math.Abs(a-b) > 1e-12*math.Max(1, math.Abs(b)) {
							t.Fatalf("seq=%d heads=%d causal=%v block=%d [%d]: flash %.15g vs mha %.15g", seq, heads, causal, block, idx, a, b)
						}
					}
				}
			}
		}
	}
}

// §V16 tier-1: FlashAttention with grouped-query / multi-query attention (kv_heads <
// heads) matches the naive OpMHA with the same kv_heads at f64 1e-12 — query head h
// shares K/V head h/(heads/kv_heads) — across block sizes so the GQA indexing is
// exercised under tiling.
func TestFlashAttnGQAMatchesMHA(t *testing.T) {
	cases := []struct{ heads, kv, dk int }{
		{4, 2, 4}, // grouped-query
		{6, 3, 4}, // grouped-query
		{4, 1, 8}, // multi-query
	}
	const seq = 7
	for _, c := range cases {
		dm, dkv := c.heads*c.dk, c.kv*c.dk
		q := randFA(tensor.Shape{seq, dm}, 1)
		k := randFA(tensor.Shape{seq, dkv}, 2)
		v := randFA(tensor.Shape{seq, dkv}, 3)
		for _, causal := range []bool{false, true} {
			ref, err := backend.Execute(backend.NewContext(), backend.OpMHA,
				[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv, Causal: causal})
			if err != nil {
				t.Fatal(err)
			}
			for _, block := range []int{1, 3, seq} {
				got, err := backend.Execute(backend.NewContext(), backend.OpFlashAttn,
					[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv, Causal: causal, Block: block})
				if err != nil {
					t.Fatal(err)
				}
				for idx := range got[0].Numel() {
					id := tensor.Unravel(idx, got[0].Shape())
					a, b := got[0].AtF64(id...), ref[0].AtF64(id...)
					if math.Abs(a-b) > 1e-12*math.Max(1, math.Abs(b)) {
						t.Fatalf("heads=%d kv=%d causal=%v block=%d [%d]: flash %.15g vs mha %.15g", c.heads, c.kv, causal, block, idx, a, b)
					}
				}
			}
		}
	}
}

// §V2: the FlashAttention op is differentiable (shared standard softmax-attention
// backward); central finite differences match the analytic gradients into Q,K,V.
func TestFlashAttnGradCheck(t *testing.T) {
	seq, heads, dm := 6, 2, 8
	attrs := backend.AttnAttrs{Heads: heads, Causal: true, Block: 2}
	q, k, v := randFA(tensor.Shape{seq, dm}, 4), randFA(tensor.Shape{seq, dm}, 5), randFA(tensor.Shape{seq, dm}, 6)

	loss := func(qq, kk, vv *tensor.Tensor) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpFlashAttn, []*tensor.Tensor{qq, kk, vv}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out[0].Numel() {
			s += out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
		}
		return s
	}

	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), backend.OpFlashAttn, []*tensor.Tensor{q, k, v}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	for _, in := range []*tensor.Tensor{q, k, v} {
		grad := tape.Grad(in)
		if grad == nil {
			t.Fatal("flashattn input received no gradient")
		}
		for idx := range in.Numel() {
			id := tensor.Unravel(idx, in.Shape())
			o := in.AtF64(id...)
			in.SetF64(o+h, id...)
			lp := loss(q, k, v)
			in.SetF64(o-h, id...)
			lm := loss(q, k, v)
			in.SetF64(o, id...)
			if num, ana := (lp-lm)/(2*h), grad.AtF64(id...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", idx, num, ana)
			}
		}
	}
}

// FlashAttention-2 computes exact attention without ever building the full score
// matrix, streaming over key blocks with an online softmax. Its output is
// identical to the standard attention up to floating-point reassociation — here a
// tiny causal example with a key-block size of 2 matches the naive attention.
func Example_flashAttention() {
	seq, dm := 4, 4
	q := tensor.New(tensor.F64, tensor.Shape{seq, dm})
	kk := tensor.New(tensor.F64, tensor.Shape{seq, dm})
	vv := tensor.New(tensor.F64, tensor.Shape{seq, dm})
	for i := range seq {
		for j := range dm {
			q.SetF64(0.1*float64(i+j), i, j)
			kk.SetF64(0.1*float64(i-j), i, j)
			vv.SetF64(float64(i), i, j)
		}
	}
	flash, _ := backend.Execute(backend.NewContext(), backend.OpFlashAttn,
		[]*tensor.Tensor{q, kk, vv}, backend.AttnAttrs{Heads: 1, Causal: true, Block: 2})
	naive, _ := backend.Execute(backend.NewContext(), backend.OpMHA,
		[]*tensor.Tensor{q, kk, vv}, backend.AttnAttrs{Heads: 1, Causal: true})

	var maxDiff float64
	for i := range flash[0].Numel() {
		id := tensor.Unravel(i, flash[0].Shape())
		maxDiff = math.Max(maxDiff, math.Abs(flash[0].AtF64(id...)-naive[0].AtF64(id...)))
	}
	fmt.Println("flash == naive attention:", maxDiff < 1e-12)
	// Output: flash == naive attention: true
}
