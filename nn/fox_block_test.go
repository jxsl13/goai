package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: the FoX block maps [T,dim]→[T,dim] and exposes the per-head Q/K/V
// projections + fused output + forget-gate projection as trainable parameters
// (each Linear contributes W and bias → 2 tensors): 3·heads·2 + 2 (Wo) + 2 (Wf).
func TestFoXBlockShapes(t *testing.T) {
	const dim, heads, seq = 8, 2, 5
	b, err := nn.NewFoXBlock(tensor.F64, dim, heads, 7)
	if err != nil {
		t.Fatal(err)
	}
	if b.HeadDim != dim/heads {
		t.Errorf("HeadDim=%d, want %d", b.HeadDim, dim/heads)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, seq, dim)
	out, err := b.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{seq, dim}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), seq, dim)
	}
	if got, want := len(b.Params()), 3*heads*2+2+2; got != want {
		t.Errorf("Params count %d, want %d", got, want)
	}
}

// §V16: NewFoXBlock rejects a dim not divisible by heads and an out-of-range eps.
func TestFoXBlockErrors(t *testing.T) {
	if _, err := nn.NewFoXBlock(tensor.F64, 7, 2, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := nn.NewFoXBlock(tensor.F64, 8, 2, 1, nn.WithFoXEps(0)); err == nil {
		t.Error("eps=0: want error")
	}
	b, _ := nn.NewFoXBlock(tensor.F64, 8, 2, 1)
	if _, err := b.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
}

// §V16 (a) DECAY-BIAS CORRECTNESS: for a known per-step log-forget column, the
// decay bias D_ij must equal the hand-computed log-cumsum difference
// c_i − c_j = Σ_{l=j+1..i} log f_l (0 on the diagonal, ≤ 0 below it).
func TestFoXDecayBias(t *testing.T) {
	logf := []float64{-0.1, -0.3, -0.2, -0.5} // arbitrary log f_t (each ≤ 0)
	const n = 4
	col := tensor.New(tensor.F64, tensor.Shape{n, 1})
	for i, v := range logf {
		col.SetF64(v, i, 0)
	}
	// hand-computed cumulative sum c_t = Σ_{l≤t} log f_l
	c := make([]float64, n)
	run := 0.0
	for i, v := range logf {
		run += v
		c[i] = run
	}
	d, err := nn.FoXDecayBias(backend.NewContext(), col)
	if err != nil {
		t.Fatal(err)
	}
	if !d.Shape().Equal(tensor.Shape{n, n}) {
		t.Fatalf("bias shape %v, want [%d %d]", d.Shape(), n, n)
	}
	for i := range n {
		for j := range n {
			want := c[i] - c[j]
			if got := d.AtF64(i, j); math.Abs(got-want) > 1e-12 {
				t.Errorf("D[%d,%d]=%.6g, want c_i−c_j=%.6g", i, j, got, want)
			}
			if i == j && d.AtF64(i, j) != 0 {
				t.Errorf("diagonal D[%d,%d]=%.6g, want 0", i, j, d.AtF64(i, j))
			}
			if j < i && d.AtF64(i, j) > 0 {
				t.Errorf("below-diagonal D[%d,%d]=%.6g must be ≤ 0", i, j, d.AtF64(i, j))
			}
		}
	}
}

// §V16 (b) CAUSALITY: output row i must not depend on any future token j>i.
// Perturbing a later token leaves every earlier output row bit-identical.
func TestFoXBlockCausality(t *testing.T) {
	const dim, heads, seq = 6, 3, 5
	b, _ := nn.NewFoXBlock(tensor.F64, dim, heads, 4)
	rng := rand.New(rand.NewPCG(11, 5))
	x := randMat(rng, seq, dim)
	base, err := b.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	const pert = 3 // perturb token 3 → rows 0..2 must be unchanged
	for c := range dim {
		x.SetF64(x.AtF64(pert, c)+1.7, pert, c)
	}
	after, err := b.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range pert {
		for c := range dim {
			if base.AtF64(i, c) != after.AtF64(i, c) {
				t.Errorf("row %d col %d changed after perturbing future token %d: %.12g → %.12g",
					i, c, pert, base.AtF64(i, c), after.AtF64(i, c))
			}
		}
	}
}

// §V16 (c) GRADIENT FLOW: central finite differences confirm gradients reach the
// forget-gate projection W_f AND the Q/K/V/O projections, so the whole block trains.
func TestFoXBlockGradcheck(t *testing.T) {
	const dim, heads, seq, h = 4, 2, 3, 1e-6
	b, _ := nn.NewFoXBlock(tensor.F64, dim, heads, 3)
	rng := rand.New(rand.NewPCG(29, 7))
	x := randMat(rng, seq, dim)

	tape := autograd.NewTape()
	out, err := b.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		o, err := b.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad — no gradient reached this parameter", name)
		}
		var nz bool
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := sumLoss()
			p.SetF64(o-h, coord...)
			lm := sumLoss()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(ana) > 1e-9 {
				nz = true
			}
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("%s: all analytic grads zero — parameter not exercised", name)
		}
	}
	check("Wf.W", b.Wf.W)       // forget-gate projection — the FoX-specific path
	check("Wf.B", b.Wf.B)       // forget-gate bias
	check("Wq[0].W", b.Wq[0].W) // query
	check("Wk[1].W", b.Wk[1].W) // key (2nd head)
	check("Wv[0].W", b.Wv[0].W) // value
	check("Wo.W", b.Wo.W)       // output projection
}

// §V16 (d) RECENCY / FORGETTING PROPERTY: forcing the forget gate toward strong
// forgetting (large negative logit → f→0 → strongly negative decay bias) makes the
// attention concentrate on the most recent token, whereas a near-1 gate (large
// positive logit → f→1 → D≈0) recovers plain uniform-ish causal softmax. We assert
// this structurally on the attention weights of the last query row by driving the
// forget-gate projection directly (bias term) with the two regimes.
func TestFoXForgettingProperty(t *testing.T) {
	const dim, heads, seq = 4, 1, 6
	// helper: build a block, force every forget-gate logit to a constant via the
	// bias (zero the weight so the logit is data-independent), and return the last
	// query row's attention distribution over keys.
	attnLastRow := func(logit float64) []float64 {
		b, _ := nn.NewFoXBlock(tensor.F64, dim, heads, 2)
		nn.Zeros(b.Wf.W)       // forget logit independent of x
		for i := range heads { // constant logit for the single head
			b.Wf.B.SetF64(logit, i)
		}
		// Zero Q and K so the raw QKᵀ scores are all equal → the decay bias alone
		// shapes the softmax, isolating the forget gate's effect.
		nn.Zeros(b.Wq[0].W)
		nn.Zeros(b.Wq[0].B)
		nn.Zeros(b.Wk[0].W)
		nn.Zeros(b.Wk[0].B)
		rng := rand.New(rand.NewPCG(3, 9))
		x := randMat(rng, seq, dim)
		// Reconstruct the last-row attention weights from the decay bias directly:
		// with equal QKᵀ scores, softmax over keys j≤i of (D_ij) gives the weights.
		fl, _ := b.Wf.Forward(backend.NewContext(), x)
		fg, _ := backend.Execute(backend.NewContext(), backend.OpSigmoid, []*tensor.Tensor{fl}, nil)
		logf, _ := backend.Execute(backend.NewContext(), backend.OpLog, fg, nil)
		col, _ := backend.Execute(backend.NewContext(), backend.OpSlice, logf, backend.SliceAttrs{Axis: 1, Start: 0, End: 1})
		d, err := nn.FoXDecayBias(backend.NewContext(), col[0])
		if err != nil {
			t.Fatal(err)
		}
		i := seq - 1
		w := make([]float64, seq)
		var z float64
		for j := 0; j <= i; j++ {
			w[j] = math.Exp(d.AtF64(i, j))
			z += w[j]
		}
		for j := range w {
			w[j] /= z
		}
		return w
	}
	strong := attnLastRow(-15) // f≈3e-7 → aggressive forgetting
	weak := attnLastRow(15)    // f≈1 → ~plain attention

	last := seq - 1
	// Strong forgetting: the most recent key dominates.
	if strong[last] < 0.99 {
		t.Errorf("strong forget: recent-token weight %.4f, want ≈1", strong[last])
	}
	// Near-1 gate: weight is spread ~uniformly, so the recent token is far from dominant.
	if weak[last] > 0.30 {
		t.Errorf("near-1 gate: recent-token weight %.4f, want spread out (≪ strong-forget case)", weak[last])
	}
	if strong[last] <= weak[last] {
		t.Errorf("forgetting did not increase recency focus: strong %.4f vs weak %.4f", strong[last], weak[last])
	}
}

// §V16 (e) E2E TRAINABILITY: a few SGD steps on a fixed alignment objective reduce
// the loss, exercising the full forward+backward+update path (grads through the
// forget gate and all projections actually move the weights in a useful direction).
func TestFoXBlockTrainsE2E(t *testing.T) {
	const dim, heads, seq = 6, 2, 4
	b, _ := nn.NewFoXBlock(tensor.F64, dim, heads, 5)
	rng := rand.New(rand.NewPCG(21, 8))
	x := randMat(rng, seq, dim)
	target := randMat(rng, seq, dim)
	opt := nn.NewSGD(b.Params(), 0.05, 0.9)

	step := func(train bool) float64 {
		tape := autograd.NewTape()
		out, err := b.Forward(tape.Context(), x)
		if err != nil {
			t.Fatal(err)
		}
		diff, err := backend.Execute(tape.Context(), backend.OpSub, []*tensor.Tensor{out, target}, nil)
		if err != nil {
			t.Fatal(err)
		}
		sq, err := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := sumAll(tape.Context(), sq[0])
		if err != nil {
			t.Fatal(err)
		}
		var l float64
		for i := range loss.Numel() {
			l += loss.AtF64(tensor.Unravel(i, loss.Shape())...)
		}
		if train {
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
		return l
	}

	first := step(true)
	var last float64
	for range 40 {
		last = step(true)
	}
	if last >= first {
		t.Errorf("MSE did not decrease over training: first %.6g, last %.6g", first, last)
	}
}

func ExampleFoXBlock() {
	b, _ := nn.NewFoXBlock(tensor.F64, 4, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := b.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(b.Params()))
	// Output: (3, 4) params=16
}
