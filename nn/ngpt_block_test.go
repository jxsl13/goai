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

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// ngptRandInput fills an [L,d_model] tensor with N(0,scale) noise (deliberately OFF the sphere so the
// entry-normalization is exercised).
func ngptRandInput(rng *rand.Rand, l, d int, scale float64) *tensor.Tensor {
	x := tensor.New(tensor.F64, tensor.Shape{l, d})
	for i := range l * d {
		x.SetF64(rng.NormFloat64()*scale, tensor.Unravel(i, x.Shape())...)
	}
	return x
}

// §V16 tier-1: the block maps [L,d_model]→[L,d_model] and exposes the expected parameter surface
// (7 shared tensors + 5 per head).
func TestNGPTBlockShapes(t *testing.T) {
	const dModel, heads, hidden, l = 8, 2, 16, 5
	b, err := nn.NewNGPTBlock(tensor.F64, dModel, heads, hidden, 7)
	if err != nil {
		t.Fatal(err)
	}
	if b.HeadDim != 4 {
		t.Errorf("HeadDim=%d, want 4", b.HeadDim)
	}
	x := ngptRandInput(rand.New(rand.NewSource(1)), l, dModel, 1)
	out, err := b.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{l, dModel}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), l, dModel)
	}
	if got, want := len(b.Params()), 7+5*heads; got != want {
		t.Errorf("Params count %d, want %d", got, want)
	}
}

// §V16 sphere invariant (§R243 §2, the defining nGPT property): every output row of Forward lies on
// the unit hypersphere (L2 norm 1), regardless of the input scale — Forward ends in a Normalize.
func TestNGPTBlockOutputOnSphere(t *testing.T) {
	const dModel, heads, hidden, l = 12, 3, 24, 6
	b, err := nn.NewNGPTBlock(tensor.F64, dModel, heads, hidden, 4)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(2))
	for _, scale := range []float64{0.01, 1, 50} { // far off the sphere in both directions
		out, err := b.Forward(backend.NewContext(), ngptRandInput(rng, l, dModel, scale))
		if err != nil {
			t.Fatal(err)
		}
		for i := range l {
			var ss float64
			for j := range dModel {
				v := out.AtF64(i, j)
				ss += v * v
			}
			// unit-norm holds up to the L2-normalize eps floor (1e-6 ⇒ ~5e-7 shortfall).
			if n := math.Sqrt(ss); math.Abs(n-1) > 1e-5 {
				t.Errorf("scale=%g row %d L2 norm %.12g, want 1", scale, i, n)
			}
		}
	}
}

// §V16 weight invariant (§R243 §2.1): after NormalizeWeights every weight matrix's columns (an output
// feature's weight vector over the input axis) are unit-norm. Weights are scrambled first so the
// re-normalization does real work (the constructor already normalizes once).
func TestNGPTBlockNormalizeWeights(t *testing.T) {
	const dModel, heads, hidden = 8, 2, 16
	b, err := nn.NewNGPTBlock(tensor.F64, dModel, heads, hidden, 5)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	mats := func() []*tensor.Tensor {
		m := []*tensor.Tensor{b.Wgate, b.Wup, b.Wdown}
		for h := range heads {
			m = append(m, b.Wq[h], b.Wk[h], b.Wv[h], b.Wo[h])
		}
		return m
	}
	for _, w := range mats() { // scramble every weight off the sphere
		for i := range w.Numel() {
			idx := tensor.Unravel(i, w.Shape())
			w.SetF64(rng.NormFloat64()*3, idx...)
		}
	}
	b.NormalizeWeights()
	for wi, w := range mats() {
		in, out := w.Shape()[0], w.Shape()[1]
		for j := range out {
			var ss float64
			for i := range in {
				v := w.AtF64(i, j)
				ss += v * v
			}
			if n := math.Sqrt(ss); math.Abs(n-1) > 1e-12 {
				t.Errorf("matrix %d column %d L2 norm %.12g, want 1", wi, j, n)
			}
		}
	}
}

// §V2 gradient flow (finite-difference): the whole block is differentiable — gradients of Σ Forward(x)
// w.r.t. the input, a representative weight matrix, a per-head q/k scale, an MLP scale, and both
// eigen-LRs match central finite differences. Confirms backprop reaches weights AND scales AND α.
func TestNGPTBlockGradcheck(t *testing.T) {
	const dModel, heads, hidden, l, fd, rtol = 4, 2, 6, 3, 1e-6, 3e-4
	b, err := nn.NewNGPTBlock(tensor.F64, dModel, heads, hidden, 3)
	if err != nil {
		t.Fatal(err)
	}
	x := ngptRandInput(rand.New(rand.NewSource(9)), l, dModel, 0.7)

	tape := autograd.NewTape()
	o, err := b.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), o)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		out, err := b.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out.Numel() {
			s += out.AtF64(tensor.Unravel(i, out.Shape())...)
		}
		return s
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad (no gradient flow)", name)
		}
		var nonzero bool
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			orig := p.AtF64(idx...)
			p.SetF64(orig+fd, idx...)
			plus := sumLoss()
			p.SetF64(orig-fd, idx...)
			minus := sumLoss()
			p.SetF64(orig, idx...)
			numer := (plus - minus) / (2 * fd)
			if ana := g.AtF64(idx...); math.Abs(numer-ana) > rtol*math.Max(1, math.Abs(numer)) {
				t.Errorf("%s grad[%v]: numeric %.8g vs analytic %.8g", name, idx, numer, ana)
			}
			if math.Abs(g.AtF64(idx...)) > 1e-9 {
				nonzero = true
			}
		}
		if !nonzero {
			t.Errorf("%s: all gradients ~0 (parameter not effectively trained)", name)
		}
	}
	check("x", x)
	check("Wq0", b.Wq[0])   // weight (attention projection)
	check("Wo1", b.Wo[1])   // weight (output projection)
	check("Sqk0", b.Sqk[0]) // per-head q/k scale
	check("Su", b.Su)       // MLP gate scale
	check("Sv", b.Sv)       // MLP value scale
	check("AlphaA", b.AlphaA)
	check("AlphaM", b.AlphaM)
}

// §V16 e2e (training convergence): a full optimize loop — Forward, alignment loss, backward, SGD Step,
// then NormalizeWeights (the mandatory post-step renorm) — drives the loss down monotonically-ish and
// keeps the weights on the sphere throughout. Exercises the whole trainable surface end to end.
func TestNGPTBlockTrainDecreases(t *testing.T) {
	const dModel, heads, hidden, l = 8, 2, 16, 4
	b, err := nn.NewNGPTBlock(tensor.F64, dModel, heads, hidden, 11)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(21))
	x := ngptRandInput(rng, l, dModel, 1)
	target := ngptRandInput(rng, l, dModel, 1) // fixed direction to align the output toward
	opt := nn.NewSGD(b.Params(), 0.5, 0.9)

	// loss = −Σ out ⊙ target : minimized by aligning each (unit-norm) output row with target.
	step := func(train bool) float64 {
		tape := autograd.NewTape()
		out, err := b.Forward(tape.Context(), x)
		if err != nil {
			t.Fatal(err)
		}
		prod, err := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out, target}, nil)
		if err != nil {
			t.Fatal(err)
		}
		s, err := sumAll(tape.Context(), prod[0])
		if err != nil {
			t.Fatal(err)
		}
		neg, err := backend.Execute(tape.Context(), backend.OpNeg, []*tensor.Tensor{s}, nil)
		if err != nil {
			t.Fatal(err)
		}
		var loss float64
		for i := range neg[0].Numel() {
			loss += neg[0].AtF64(tensor.Unravel(i, neg[0].Shape())...)
		}
		if train {
			if err := tape.Backward(neg[0]); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
			b.NormalizeWeights() // mandatory nGPT post-step weight renorm (§R243 §2.1)
		}
		return loss
	}

	first := step(true)
	var last float64
	for range 30 {
		last = step(true)
	}
	if last >= first {
		t.Errorf("loss did not decrease: first %.6g, last %.6g", first, last)
	}
	// weights stayed on the sphere across all the NormalizeWeights calls.
	for j := range b.Wgate.Shape()[1] {
		var ss float64
		for i := range b.Wgate.Shape()[0] {
			v := b.Wgate.AtF64(i, j)
			ss += v * v
		}
		if n := math.Sqrt(ss); math.Abs(n-1) > 1e-12 {
			t.Errorf("post-train Wgate column %d norm %.12g, want 1", j, n)
		}
	}
}

func TestNGPTBlockErrors(t *testing.T) {
	if _, err := nn.NewNGPTBlock(tensor.F64, 10, 3, 16, 1); err == nil {
		t.Error("d_model=10 not divisible by heads=3 must error")
	}
	if _, err := nn.NewNGPTBlock(tensor.F64, 8, 2, 0, 1); err == nil {
		t.Error("hidden=0 must error")
	}
	b, _ := nn.NewNGPTBlock(tensor.F64, 8, 2, 16, 1)
	if _, err := b.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 7})); err == nil {
		t.Error("wrong input width must error")
	}
}

// An nGPT block is a drop-in decoder layer that keeps every hidden state on the unit hypersphere:
// there is no LayerNorm, and the residual is a normalized linear interpolation (§R243). A 4-wide,
// 2-head block maps a length-3 sequence to unit-norm rows.
func ExampleNGPTBlock() {
	b, _ := nn.NewNGPTBlock(tensor.F64, 4, 2, 8, 1, nn.WithNGPTCausal(true))
	b.NormalizeWeights() // the mandatory post-optimizer-step renorm keeps weight columns on the sphere
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := b.Forward(backend.NewContext(), x)
	var ss float64
	for j := range 4 {
		v := out.AtF64(0, j)
		ss += v * v
	}
	fmt.Printf("%v norm=%.4f\n", out.Shape(), math.Sqrt(ss))
	// Output: (3, 4) norm=1.0000
}
