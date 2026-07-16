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

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 anchor 1 (THE anchor — PARALLEL ≡ SEQUENTIAL duality): the parallel
// cumulative-log-gate form (Forward) and the step-by-step recurrent scan
// (ForwardSequential) produce the SAME output at 1e-10 over random inputs. This is
// the gold-standard correctness check for a linear recurrence (mirrors RetNet's
// parallel/recurrent duality).
func TestRGLRUDuality(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, dims := range [][2]int{{1, 1}, {7, 4}, {16, 8}} {
		l, d := dims[0], dims[1]
		m, err := nn.NewRGLRU(tensor.F64, d, nn.WithRGLRUSeed(42))
		if err != nil {
			t.Fatal(err)
		}
		x := rglruRand(rng, l, d)
		par, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		seq, err := m.ForwardSequential(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		if !par.Shape().Equal(tensor.Shape{l, d}) {
			t.Fatalf("Forward shape %v, want [%d %d]", par.Shape(), l, d)
		}
		for i := range par.Numel() {
			coord := tensor.Unravel(i, par.Shape())
			if diff := math.Abs(par.AtF64(coord...) - seq.AtF64(coord...)); diff > 1e-10 {
				t.Errorf("L=%d d=%d out%v: parallel %.17g vs sequential %.17g (Δ%.2g)",
					l, d, coord, par.AtF64(coord...), seq.AtF64(coord...), diff)
			}
		}
	}
}

// §V16 anchor 2 (CONSTANT-GATE COLLAPSE): with the gate weights driven to zero the
// gates become constant (r_t = i_t = σ(0) = ½), so a_t = σ(Λ)^(c·½) = ā is a FIXED
// per-channel decay and the recurrence collapses to a fixed-decay EMA h_t =
// ā·h_{t-1} + √(1−ā²)·(½·x_t). Assert Forward matches a hand-computed EMA at 1e-10 —
// pins the recurrence arithmetic independently of the gating.
func TestRGLRUConstGateEMA(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const l, d = 12, 4
	m, err := nn.NewRGLRU(tensor.F64, d, nn.WithRGLRUSeed(7))
	if err != nil {
		t.Fatal(err)
	}
	nn.Zeros(m.Wa.W) // r_t = σ(0·x + 0) = ½ for all t
	nn.Zeros(m.Wx.W) // i_t = ½ for all t
	x := rglruRand(rng, l, d)
	got, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// Reference fixed-decay EMA. r=i=½ ⇒ a_t = σ(Λ_k)^(c·½), constant per channel.
	const c = 8.0
	for k := range d {
		lam := m.Lambda.AtF64(k)
		sig := 1 / (1 + math.Exp(-lam))
		abar := math.Pow(sig, c*0.5)
		norm := math.Sqrt(1 - abar*abar)
		var h float64
		for tt := range l {
			h = abar*h + norm*(0.5*x.AtF64(tt, k))
			if diff := math.Abs(got.AtF64(tt, k) - h); diff > 1e-10 {
				t.Errorf("EMA out[%d,%d]=%.17g want %.17g (Δ%.2g)", tt, k, got.AtF64(tt, k), h, diff)
			}
		}
	}
}

// §V16 anchor 4 (GRADCHECK): finite-difference gradcheck through W_a, b_a, W_x, b_x
// and Λ. Λ is set to a moderate range so the decay a_t stays clear of 1 (where the
// √(1−a²) derivative diverges — the flagged failure mode); the gate/cumsum/exp/√/
// einsum path must still match numeric gradients to ~1e-4.
func TestRGLRUGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	const l, d = 4, 3
	m, err := nn.NewRGLRU(tensor.F64, d, nn.WithRGLRUSeed(11))
	if err != nil {
		t.Fatal(err)
	}
	for k := range d { // moderate Λ ⇒ a_t mid-range, well-conditioned √(1−a²)
		m.Lambda.SetF64(0.4*rng.NormFloat64(), k)
	}
	x := rglruRand(rng, l, d)

	tape := autograd.NewTape()
	out, err := m.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := rglruSum(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	fwd := func() float64 {
		o, _ := m.Forward(backend.NewContext(), x)
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
	}
	const h = 1e-6
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := fwd()
			p.SetF64(o-h, coord...)
			lm := fwd()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
	}
	check("Wa.W", m.Wa.W)
	check("Wa.B", m.Wa.B)
	check("Wx.W", m.Wx.W)
	check("Wx.B", m.Wx.B)
	check("Lambda", m.Lambda)
}

// §V16 anchor 5 (THE VALUE — long-range recall): a selective-copy task. A signed
// signal sits at t=0 behind a one-hot marker channel; channel 0 is then flooded with
// random distractors for the rest of the sequence, and the model must reproduce the
// t=0 signal from the LAST step's hidden state. Input-dependent gating can learn to
// write at the marker and hold while suppressing distractors; a fixed-decay EMA
// (gates frozen, the ungated ablation) integrates the distractors and cannot. Train
// a few hundred steps and report loss for both.
func TestRGLRULongRangeRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping training loop in -short")
	}
	const (
		d     = 4
		l     = 16
		batch = 16
		steps = 400
	)
	rng := rand.New(rand.NewPCG(9, 10))
	makeBatch := func() ([]*tensor.Tensor, []float64) {
		xs := make([]*tensor.Tensor, batch)
		ys := make([]float64, batch)
		for b := range batch {
			x := tensor.New(tensor.F64, tensor.Shape{l, d})
			s := 1.0
			if rng.Float64() < 0.5 {
				s = -1
			}
			x.SetF64(s, 0, 0) // the signal to recall
			x.SetF64(1, 0, 1) // one-hot marker at t=0
			for tt := 1; tt < l; tt++ {
				x.SetF64(2*rng.Float64()-1, tt, 0) // distractors flood channel 0
				for k := 2; k < d; k++ {
					x.SetF64(0.1*rng.NormFloat64(), tt, k)
				}
			}
			xs[b], ys[b] = x, s
		}
		return xs, ys
	}
	// train runs the loop for a model + readout over the given trainable params;
	// returns (firstLoss, lastLoss).
	train := func(m *nn.RGLRU, readout *nn.Linear, params []*tensor.Tensor) (float64, float64) {
		opt := nn.NewAdam(params, 5e-3)
		var first, last float64
		for step := range steps {
			xs, ys := makeBatch()
			tape := autograd.NewTape()
			ctx := tape.Context()
			var loss *tensor.Tensor
			for b := range batch {
				h, err := m.Forward(ctx, xs[b])
				if err != nil {
					t.Fatal(err)
				}
				last1, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{h}, backend.SliceAttrs{Axis: 0, Start: l - 1, End: l})
				if err != nil {
					t.Fatal(err)
				}
				pred, err := readout.Forward(ctx, last1[0]) // [1,1]
				if err != nil {
					t.Fatal(err)
				}
				tgt := tensor.New(tensor.F64, tensor.Shape{1, 1})
				tgt.SetF64(ys[b], 0, 0)
				diff, err := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{pred, tgt}, nil)
				if err != nil {
					t.Fatal(err)
				}
				sq, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
				if err != nil {
					t.Fatal(err)
				}
				if loss == nil {
					loss = sq[0]
				} else {
					add, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{loss, sq[0]}, nil)
					if err != nil {
						t.Fatal(err)
					}
					loss = add[0]
				}
			}
			lv := loss.AtF64(0, 0) / float64(batch)
			if step == 0 {
				first = lv
			}
			last = lv
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
		return first, last
	}

	// Full gated RG-LRU: all gate params + Λ + readout trainable.
	gated, err := nn.NewRGLRU(tensor.F64, d, nn.WithRGLRUSeed(100))
	if err != nil {
		t.Fatal(err)
	}
	rdG := nn.NewLinear(tensor.F64, d, 1, 101)
	gParams := append(gated.Params(), rdG.Params()...)
	gFirst, gLast := train(gated, rdG, gParams)
	t.Logf("gated RG-LRU MSE: %.4f → %.4f", gFirst, gLast)

	// Ungated fixed-decay ablation: freeze the gate weights at 0 (r=i=½ constant),
	// train only Λ and the readout ⇒ a per-channel fixed-decay EMA with no input
	// selectivity.
	abl, err := nn.NewRGLRU(tensor.F64, d, nn.WithRGLRUSeed(200))
	if err != nil {
		t.Fatal(err)
	}
	nn.Zeros(abl.Wa.W)
	nn.Zeros(abl.Wx.W)
	rdA := nn.NewLinear(tensor.F64, d, 1, 201)
	aParams := append([]*tensor.Tensor{abl.Lambda}, rdA.Params()...)
	aFirst, aLast := train(abl, rdA, aParams)
	t.Logf("ungated fixed-decay ablation MSE: %.4f → %.4f", aFirst, aLast)

	if gLast >= 0.5*gFirst {
		t.Errorf("gated RG-LRU did not learn the recall task: %.4f → %.4f", gFirst, gLast)
	}
	if gLast >= aLast {
		t.Errorf("gating did not help vs fixed-decay ablation: gated %.4f, ablation %.4f", gLast, aLast)
	}
}

// §V16 anchor 6 (sanity): construction/shape errors and determinism.
func TestRGLRUErrors(t *testing.T) {
	if _, err := nn.NewRGLRU(tensor.F64, 0); err == nil {
		t.Error("dModel=0: want error")
	}
	if _, err := nn.NewRGLRU(tensor.F64, -3); err == nil {
		t.Error("dModel<0: want error")
	}
	m, err := nn.NewRGLRU(tensor.F64, 4, nn.WithRGLRUSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3})); err == nil {
		t.Error("rank-1 input: want error")
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 5})); err == nil {
		t.Error("wrong channel width: want error")
	}
	if _, err := m.ForwardSequential(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 5})); err == nil {
		t.Error("sequential wrong width: want error")
	}
	// Determinism: same seed ⇒ identical params and identical output.
	a, _ := nn.NewRGLRU(tensor.F64, 4, nn.WithRGLRUSeed(77))
	b, _ := nn.NewRGLRU(tensor.F64, 4, nn.WithRGLRUSeed(77))
	x := rglruRand(rand.New(rand.NewPCG(1, 1)), 5, 4)
	oa, _ := a.Forward(backend.NewContext(), x)
	ob, _ := b.Forward(backend.NewContext(), x)
	for i := range oa.Numel() {
		coord := tensor.Unravel(i, oa.Shape())
		if oa.AtF64(coord...) != ob.AtF64(coord...) {
			t.Errorf("same seed diverged at %v", coord)
		}
	}
}

func rglruRand(rng *rand.Rand, r, c int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			t.SetF64(rng.NormFloat64(), i, j)
		}
	}
	return t
}

func rglruSum(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	axes := make([]int, x.Ndim())
	for i := range axes {
		axes[i] = i
	}
	o, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: axes})
	if err != nil {
		return nil, err
	}
	return o[0], nil
}

func ExampleRGLRU() {
	// Zeroing the gate weights collapses the RG-LRU to a fixed-decay EMA (constant
	// gates r=i=σ(0)=½). With Λ=0 the per-channel decay is a = σ(0)^(c·½) = 0.5⁴ =
	// 0.0625, so h_t = 0.0625·h_{t-1} + √(1−0.0625²)·(½·x_t) over x=[1,1].
	m, _ := nn.NewRGLRU(tensor.F64, 1, nn.WithRGLRUSeed(0))
	nn.Zeros(m.Wa.W)
	nn.Zeros(m.Wx.W)
	m.Lambda.SetF64(0, 0)
	x := tensor.New(tensor.F64, tensor.Shape{2, 1})
	x.SetF64(1, 0, 0)
	x.SetF64(1, 1, 0)
	out, _ := m.Forward(backend.NewContext(), x)
	fmt.Printf("%.4f %.4f\n", out.AtF64(0, 0), out.AtF64(1, 0))
	// Output:
	// 0.4990 0.5302
}

func ExampleRGLRU_forwardSequential() {
	// ForwardSequential is the recurrent scan; it produces the SAME output as the
	// parallel Forward (the parallel/recurrent duality) — here the fixed-decay EMA
	// of ExampleRGLRU, computed step by step instead of via the cumulative-log form.
	m, _ := nn.NewRGLRU(tensor.F64, 1, nn.WithRGLRUSeed(0))
	nn.Zeros(m.Wa.W)
	nn.Zeros(m.Wx.W)
	m.Lambda.SetF64(0, 0)
	x := tensor.New(tensor.F64, tensor.Shape{2, 1})
	x.SetF64(1, 0, 0)
	x.SetF64(1, 1, 0)
	out, _ := m.ForwardSequential(backend.NewContext(), x)
	fmt.Printf("%.4f %.4f\n", out.AtF64(0, 0), out.AtF64(1, 0))
	// Output:
	// 0.4990 0.5302
}
