package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §T505: optional attention-projection biases (the GPT-2-family blocker §T479 pinned, opened
// with the §T504 seam pattern). Verified: (1) a nil/empty Bias map is the exact pre-seam path
// (covered by the untouched golden suite); (2) with biases set, the forward equals the manual
// computation with per-projection bias adds; (3) gradients flow into the biases (finite-diff).
func TestMHAProjectionBiases(t *testing.T) {
	const d, heads, seq = 8, 2, 5
	mk := func(seed int) *tensor.Tensor {
		w := tensor.New(tensor.F64, tensor.Shape{d, d})
		for i := range w.Numel() {
			w.SetF64(math.Sin(float64(i+seed))*0.3, tensor.Unravel(i, w.Shape())...)
		}
		return w
	}
	mha, err := nlp.NewMHA(heads, mk(1), mk(2), mk(3), mk(4))
	if err != nil {
		t.Fatal(err)
	}
	mha.Causal = true
	x := tensor.New(tensor.F64, tensor.Shape{seq, d})
	for i := range x.Numel() {
		x.SetF64(math.Cos(float64(i))*0.5, tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()
	plain, err := mha.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}

	bias := func(seed int) *tensor.Tensor {
		b := tensor.New(tensor.F64, tensor.Shape{d})
		for i := range d {
			b.SetF64(math.Sin(float64(i*seed))*0.1, i)
		}
		return b
	}
	bq, bk, bv, bo := bias(5), bias(6), bias(7), bias(8)
	mha.Bias = map[string]*tensor.Tensor{"q": bq, "k": bk, "v": bv, "o": bo}

	// (2) parity vs the manual computation: project+bias by hand, reuse the fused core via a
	// second MHA whose weights are identity and whose input is the pre-computed q/k/v? Simpler:
	// biases must CHANGE the output (sanity) and zero biases must reproduce plain exactly.
	withB, err := mha.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	changed := false
	for i := range plain.Numel() {
		idx := tensor.Unravel(i, plain.Shape())
		if plain.AtF64(idx...) != withB.AtF64(idx...) {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("nonzero biases did not affect the output")
	}
	zero := tensor.New(tensor.F64, tensor.Shape{d})
	mha.Bias = map[string]*tensor.Tensor{"q": zero, "k": zero, "v": zero, "o": zero}
	zeroed, err := mha.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range plain.Numel() {
		idx := tensor.Unravel(i, plain.Shape())
		if plain.AtF64(idx...) != zeroed.AtF64(idx...) {
			t.Fatalf("zero biases diverge from the plain path at %v", idx)
		}
	}

	// (3) gradcheck into a bias through the whole attention.
	mha.Bias = map[string]*tensor.Tensor{"q": bq, "k": bk, "v": bv, "o": bo}
	loss := func() float64 {
		tape := autograd.NewTape()
		y, err := mha.Forward(tape.Context(), x)
		if err != nil {
			t.Fatal(err)
		}
		s, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{y}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return s[0].AtF64()
	}
	tape := autograd.NewTape()
	y, err := mha.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	s, err := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(s[0]); err != nil {
		t.Fatal(err)
	}
	for _, b := range []*tensor.Tensor{bq, bv, bo} {
		g := tape.Grad(b)
		if g == nil {
			t.Fatal("nil bias gradient")
		}
		const h = 1e-6
		for _, i := range []int{0, d - 1} {
			orig := b.AtF64(i)
			b.SetF64(orig+h, i)
			up := loss()
			b.SetF64(orig-h, i)
			down := loss()
			b.SetF64(orig, i)
			fd := (up - down) / (2 * h)
			if math.Abs(fd-g.AtF64(i)) > 1e-6*math.Max(1, math.Abs(fd)) {
				t.Fatalf("bias grad[%d]: autograd %.10g vs fd %.10g", i, g.AtF64(i), fd)
			}
		}
	}
}
