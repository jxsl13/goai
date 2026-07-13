package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §T456: the Conv2D layer is exactly OpConv2D with the layer's attrs and bias — parity against
// the direct backend call, and shape math (H' = (H+2P−K)/S + 1).
func TestConv2DLayerParity(t *testing.T) {
	l, err := nn.NewConv2D(tensor.F64, 2, 3, 3, 1, 1, 7)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{2, 2, 5, 5})
	for i := range x.Numel() {
		x.SetF64(math.Sin(float64(i)), tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()
	got, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	want, err := backend.Execute(ctx, backend.OpConv2D,
		[]*tensor.Tensor{x, l.W, l.B}, backend.ConvAttrs{Stride: 1, Pad: 1})
	if err != nil {
		t.Fatal(err)
	}
	if !got.Shape().Equal(tensor.Shape{2, 3, 5, 5}) {
		t.Fatalf("shape %v, want [2 3 5 5]", got.Shape())
	}
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		if a, b := got.AtF64(idx...), want[0].AtF64(idx...); a != b {
			t.Fatalf("[%v]: %g != %g", idx, a, b)
		}
	}
	if len(l.Params()) != 2 {
		t.Fatalf("params %d, want 2", len(l.Params()))
	}
	if _, err := nn.NewConv2D(tensor.F64, 0, 1, 3, 1, 0, 1); err == nil {
		t.Fatal("inC=0 accepted")
	}
	if _, err := l.Forward(ctx, tensor.New(tensor.F64, tensor.Shape{2, 3})); err == nil {
		t.Fatal("rank-2 input accepted")
	}
}

// §T456: gradients flow through the layer (the fused OpConv2DBackward VJP) — finite-difference
// check on both the input and the weight.
func TestConv2DLayerGradCheck(t *testing.T) {
	l, err := nn.NewConv2D(tensor.F64, 1, 2, 2, 1, 0, 3)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{1, 1, 4, 4})
	for i := range x.Numel() {
		x.SetF64(0.3*math.Cos(float64(i)), tensor.Unravel(i, x.Shape())...)
	}
	loss := func() float64 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		y, err := l.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		s, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{y}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return s[0].AtF64()
	}
	tape := autograd.NewTape()
	ctx := tape.Context()
	y, err := l.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	s, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(s[0]); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for _, i := range []int{0, p.Numel() / 2, p.Numel() - 1} {
			idx := tensor.Unravel(i, p.Shape())
			orig := p.AtF64(idx...)
			p.SetF64(orig+h, idx...)
			up := loss()
			p.SetF64(orig-h, idx...)
			down := loss()
			p.SetF64(orig, idx...)
			fd := (up - down) / (2 * h)
			if math.Abs(fd-g.AtF64(idx...)) > 1e-6*math.Max(1, math.Abs(fd)) {
				t.Fatalf("%s grad[%v]: autograd %g vs finite-diff %g", name, idx, g.AtF64(idx...), fd)
			}
		}
	}
	check("x", x)
	check("W", l.W)
	check("B", l.B)
}

// §T456: MaxPool2D pools windows to their maximum and carries no params.
func TestMaxPool2DLayer(t *testing.T) {
	p := &nn.MaxPool2D{Kernel: 2}
	x := tensor.New(tensor.F64, tensor.Shape{1, 1, 4, 4})
	for i := range 16 {
		x.SetF64(float64(i), tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()
	y, err := p.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	if !y.Shape().Equal(tensor.Shape{1, 1, 2, 2}) {
		t.Fatalf("shape %v", y.Shape())
	}
	for i, want := range []float64{5, 7, 13, 15} {
		idx := tensor.Unravel(i, y.Shape())
		if got := y.AtF64(idx...); got != want {
			t.Fatalf("pool[%v]=%g, want %g", idx, got, want)
		}
	}
	if p.Params() != nil {
		t.Fatal("pool must have no params")
	}
	if _, err := p.Forward(ctx, tensor.New(tensor.F64, tensor.Shape{4, 4})); err == nil {
		t.Fatal("rank-2 accepted")
	}
}

// §T456 end-to-end: a small CNN — Conv2D → ReLU → MaxPool → GlobalAvg → logits — trains on a
// synthetic two-class image task (bright top half vs bright bottom half) with the standard loop,
// proving the layers compose and the fused conv backward learns.
func TestCNNTrains(t *testing.T) {
	conv, err := nn.NewConv2D(tensor.F64, 1, 4, 3, 1, 1, 5)
	if err != nil {
		t.Fatal(err)
	}
	head := nn.NewLinear(tensor.F64, 4, 2, 6)
	relu := nn.ReLU()

	const n, hw = 8, 8
	x := tensor.New(tensor.F64, tensor.Shape{n, 1, hw, hw})
	targets := tensor.New(tensor.F64, tensor.Shape{n})
	for s := range n {
		cls := s % 2
		targets.SetF64(float64(cls), s)
		for i := range hw {
			for j := range hw {
				v := 0.05 * math.Sin(float64(s*7+i*3+j))
				if (cls == 0 && i < hw/2) || (cls == 1 && i >= hw/2) {
					v += 1
				}
				x.SetF64(v, s, 0, i, j)
			}
		}
	}

	params := append(conv.Params(), head.Params()...)
	opt := nn.NewAdamW(params, 0.02, 0)
	var first, last float64
	for step := range 60 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		y, err := conv.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		if y, err = relu.Forward(ctx, y); err != nil {
			t.Fatal(err)
		}
		pooled, err := backend.Execute(ctx, backend.OpAvgPool2D, []*tensor.Tensor{y}, backend.PoolAttrs{Kernel: hw})
		if err != nil {
			t.Fatal(err)
		}
		flat, err := backend.Execute(ctx, backend.OpReshape, []*tensor.Tensor{pooled[0]}, backend.ReshapeAttrs{Shape: tensor.Shape{n, 4}})
		if err != nil {
			t.Fatal(err)
		}
		logits, err := head.Forward(ctx, flat[0])
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(params, tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	if last >= first*0.2 {
		t.Fatalf("CNN did not train: CE %.4f → %.4f", first, last)
	}
	t.Logf("CNN trained: CE %.4f → %.4f (60 steps)", first, last)
}
