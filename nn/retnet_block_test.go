package nn_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: the MSR block maps [L,d_model]→[L,d_model] and exposes 5 weight matrices per head
// (Wq,Wk,Wv,Wg,Wo); the affine-free RMSNorm gain is fixed and not a parameter.
func TestRetNetBlockShapes(t *testing.T) {
	const dModel, heads, l = 8, 2, 5
	b, err := nn.NewRetNetBlock(tensor.F64, dModel, heads, 7)
	if err != nil {
		t.Fatal(err)
	}
	if b.KeyDim != 4 || b.ValDim != 8 {
		t.Errorf("KeyDim=%d ValDim=%d, want 4/8", b.KeyDim, b.ValDim)
	}
	x := tensor.New(tensor.F64, tensor.Shape{l, dModel})
	rng := rand.New(rand.NewSource(1))
	for i := range l * dModel {
		x.SetF64(rng.NormFloat64(), tensor.Unravel(i, x.Shape())...)
	}
	out, err := b.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{l, dModel}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), l, dModel)
	}
	if got := len(b.Params()); got != 5*heads {
		t.Errorf("Params count %d, want %d", got, 5*heads)
	}
}

// §V2 GRADCHECK (end-to-end): the whole MSR block is differentiable — the gradients of Σ MSR(x) w.r.t.
// the input and a representative weight matrix match central finite differences, so the block trains.
func TestRetNetBlockGradcheckE2E(t *testing.T) {
	const dModel, heads, l, h, rtol = 4, 2, 3, 1e-6, 2e-3
	b, _ := nn.NewRetNetBlock(tensor.F64, dModel, heads, 3)
	rng := rand.New(rand.NewSource(9))
	x := tensor.New(tensor.F64, tensor.Shape{l, dModel})
	for i := range l * dModel {
		x.SetF64(rng.NormFloat64()*0.5, tensor.Unravel(i, x.Shape())...)
	}
	// analytic grads via the tape
	tape := autograd.NewTape()
	ctx := tape.Context()
	o, err := b.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{o}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	// forward-only loss reading the live tensors (for finite differences)
	sumLoss := func() float64 {
		out, err := b.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		s := 0.0
		for i := range out.Numel() {
			s += out.AtF64(tensor.Unravel(i, out.Shape())...)
		}
		return s
	}
	check := func(name string, p *tensor.Tensor, grad *tensor.Tensor) {
		if grad == nil {
			t.Fatalf("nil grad for %s", name)
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			orig := p.AtF64(idx...)
			p.SetF64(orig+h, idx...)
			plus := sumLoss()
			p.SetF64(orig-h, idx...)
			minus := sumLoss()
			p.SetF64(orig, idx...)
			numer := (plus - minus) / (2 * h)
			if ana := grad.AtF64(idx...); math.Abs(numer-ana) > rtol*math.Max(1, math.Abs(numer)) {
				t.Errorf("%s grad[%v]: numeric %.8g vs analytic %.8g", name, idx, numer, ana)
			}
		}
	}
	check("x", x, tape.Grad(x))
	check("Wv0", b.Wv[0].W, tape.Grad(b.Wv[0].W))
	check("Wo1", b.Wo[1].W, tape.Grad(b.Wo[1].W))
	check("Wq0", b.Wq[0].W, tape.Grad(b.Wq[0].W)) // gradient through the RoPE-rotated query path
	check("Wk1", b.Wk[1].W, tape.Grad(b.Wk[1].W)) // gradient through the RoPE-rotated key path
}

func TestRetNetBlockErrors(t *testing.T) {
	if _, err := nn.NewRetNetBlock(tensor.F64, 10, 3, 1); err == nil {
		t.Error("d_model=10 not divisible by heads=3 must error")
	}
	// odd per-head key dim (d_model/heads = 6/2 = 3) has no valid rotary pairing (§R114)
	if _, err := nn.NewRetNetBlock(tensor.F64, 6, 2, 1); err == nil {
		t.Error("odd per-head key dim must error (rotary needs even head dim)")
	}
	b, _ := nn.NewRetNetBlock(tensor.F64, 8, 2, 1)
	if _, err := b.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 7})); err == nil {
		t.Error("wrong input width must error")
	}
}

// A RetNet MSR block is a drop-in sequence layer: it maps [L, d_model] to [L, d_model], replacing
// attention with multi-scale retention (a parallel-trainable, recurrent-decodable operator). Here a
// 4-wide, 2-head block processes a length-3 sequence.
func ExampleRetNetBlock() {
	b, _ := nn.NewRetNetBlock(tensor.F64, 4, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	out, _ := b.Forward(backend.NewContext(), x)
	fmt.Println(out.Shape())
	// Output: (3, 4)
}
