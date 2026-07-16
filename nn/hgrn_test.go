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
// cumulative-log-forget form (Forward) and the step-by-step recurrent scan
// (ForwardSequential) produce the SAME output at 1e-10 over random inputs, several
// lower bounds γ, and with/without the output gate. Gold-standard correctness check
// for the linear recurrence (mirrors RG-LRU's parallel/recurrent duality).
func TestHGRNDuality(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, gamma := range []float64{0, 0.1, 0.5, 0.9} {
		for _, outGate := range []bool{false, true} {
			for _, dims := range [][2]int{{1, 1}, {7, 4}, {16, 8}} {
				l, d := dims[0], dims[1]
				opts := []nn.HGRNOption{nn.WithHGRNSeed(42), nn.WithHGRNLowerBound(gamma)}
				if outGate {
					opts = append(opts, nn.WithHGRNOutputGate())
				}
				m, err := nn.NewHGRN(tensor.F64, d, opts...)
				if err != nil {
					t.Fatal(err)
				}
				x := hgrnRand(rng, l, d)
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
						t.Errorf("γ=%g outGate=%v L=%d d=%d out%v: parallel %.17g vs sequential %.17g (Δ%.2g)",
							gamma, outGate, l, d, coord, par.AtF64(coord...), seq.AtF64(coord...), diff)
					}
				}
			}
		}
	}
}

// §V16 anchor 2 (γ=0 COLLAPSE — no floor): with γ=0 the forget gate is f_t = σ(f̃_t)
// with NO lower bound, i.e. a standard gated linear RNN. Assert Forward matches an
// independent, hand-computed no-floor recurrence h_t = σ(f̃_t)·h_{t-1} +
// (1−σ(f̃_t))·SiLU(ĩ_t) at 1e-10 (σ, SiLU and the projections recomputed by hand) —
// pins the recurrence arithmetic against a reference that shares no code with gates().
func TestHGRNZeroBoundNoFloor(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const l, d = 10, 4
	m, err := nn.NewHGRN(tensor.F64, d, nn.WithHGRNSeed(7), nn.WithHGRNLowerBound(0))
	if err != nil {
		t.Fatal(err)
	}
	x := hgrnRand(rng, l, d)
	got, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// Independent reference: f̃ = x·W_f + b_f, ĩ = x·W_i + b_i (manual matmul), then
	// f = σ(f̃), i = 1−f, c = SiLU(ĩ), h_t = f·h_{t-1} + i·c.
	sigmoid := func(z float64) float64 { return 1 / (1 + math.Exp(-z)) }
	silu := func(z float64) float64 { return z * sigmoid(z) }
	proj := func(w, bias *tensor.Tensor, tt, k int) float64 {
		acc := bias.AtF64(k)
		for j := range d {
			acc += x.AtF64(tt, j) * w.AtF64(j, k)
		}
		return acc
	}
	h := make([]float64, d)
	for tt := range l {
		for k := range d {
			f := sigmoid(proj(m.Wf.W, m.Wf.B, tt, k))
			c := silu(proj(m.Wi.W, m.Wi.B, tt, k))
			h[k] = f*h[k] + (1-f)*c
			if diff := math.Abs(got.AtF64(tt, k) - h[k]); diff > 1e-10 {
				t.Errorf("no-floor out[%d,%d]=%.17g want %.17g (Δ%.2g)", tt, k, got.AtF64(tt, k), h[k], diff)
			}
		}
	}
}

// §V16 anchor 2 (γ→1 COLLAPSE — pure retention): as γ→1 the forget gate f_t→1 and the
// input gate i_t = 1−f_t→0, so h_t ≈ h_{t-1}: nothing new is written and the state
// freezes at its (zero) initial carry. With γ = 1−1e-10 assert every step's output
// equals the first within 1e-8 (the state is frozen / retained).
func TestHGRNHighBoundRetention(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	const l, d = 8, 4
	m, err := nn.NewHGRN(tensor.F64, d, nn.WithHGRNSeed(9), nn.WithHGRNLowerBound(1-1e-10))
	if err != nil {
		t.Fatal(err)
	}
	x := hgrnRand(rng, l, d)
	got, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for k := range d {
		h0 := got.AtF64(0, k)
		for tt := range l {
			if diff := math.Abs(got.AtF64(tt, k) - h0); diff > 1e-8 {
				t.Errorf("retention out[%d,%d]=%.3g not frozen at h0=%.3g (Δ%.2g)", tt, k, got.AtF64(tt, k), h0, diff)
			}
		}
	}
}

// §V16 anchor 3 (HIERARCHY monotonicity): HGRNLayerBounds(K) is strictly increasing
// and lies in [0,1) for several K, and matches the closed-form (k−1)/K of the uniform
// cumulative-softmax rule at 1e-10.
func TestHGRNLayerBoundsMonotone(t *testing.T) {
	for _, K := range []int{1, 2, 3, 5, 8} {
		g := nn.HGRNLayerBounds(K)
		if len(g) != K {
			t.Fatalf("K=%d: got %d bounds", K, len(g))
		}
		for k := range K {
			if g[k] < 0 || g[k] >= 1 {
				t.Errorf("K=%d γ_%d=%g out of [0,1)", K, k, g[k])
			}
			if want := float64(k) / float64(K); math.Abs(g[k]-want) > 1e-10 {
				t.Errorf("K=%d γ_%d=%.17g want %.17g", K, k, g[k], want)
			}
			if k > 0 && g[k] <= g[k-1] {
				t.Errorf("K=%d not strictly increasing at %d: %g ≤ %g", K, k, g[k], g[k-1])
			}
		}
	}
	if g := nn.HGRNLayerBounds(0); g != nil {
		t.Errorf("K=0: want nil, got %v", g)
	}
}

// §V16 anchor 4 (GRADCHECK): finite-difference gradcheck through W_f, b_f, W_i, b_i and
// the output-gate W_g, b_g. γ=0.3 keeps f_t mid-range (clear of 1, where log f and the
// gate saturate), so the sigmoid/SiLU/log/cumsum/exp/einsum path matches numeric
// gradients to ~1e-4.
func TestHGRNGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const l, d = 4, 3
	m, err := nn.NewHGRN(tensor.F64, d, nn.WithHGRNSeed(11), nn.WithHGRNLowerBound(0.3), nn.WithHGRNOutputGate())
	if err != nil {
		t.Fatal(err)
	}
	// Nudge biases off zero so b_f/b_i/b_g gradients are exercised at a generic point.
	for _, bias := range []*tensor.Tensor{m.Wf.B, m.Wi.B, m.Wg.B} {
		for k := range d {
			bias.SetF64(0.2*rng.NormFloat64(), k)
		}
	}
	x := hgrnRand(rng, l, d)

	tape := autograd.NewTape()
	out, err := m.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := hgrnSum(tape.Context(), out)
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
	const step = 1e-6
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+step, coord...)
			lp := fwd()
			p.SetF64(o-step, coord...)
			lm := fwd()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*step), g.AtF64(coord...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
	}
	check("Wf.W", m.Wf.W)
	check("Wf.B", m.Wf.B)
	check("Wi.W", m.Wi.W)
	check("Wi.B", m.Wi.B)
	check("Wg.W", m.Wg.W)
	check("Wg.B", m.Wg.B)
}

// §V16 anchor 5 (THE VALUE — the hierarchy pays off): a long-range retention task. A
// signed signal sits at t=0 on channel 0; every later step is small noise (on the
// other channels), and the model must reproduce the t=0 signal from the LAST hidden
// state. To isolate the LOWER BOUND itself, the forget-gate weights are frozen at 0 in
// BOTH configs, so f_t = (1+γ)/2 is a constant decay set purely by γ: a HIGH bound
// (deep layer) keeps the early signal alive across the whole sequence; a γ≈0 bound
// (shallow layer) decays it away in a few steps and forgets. Train the candidate +
// readout for each and report losses — a direct read on the memory-horizon the bound
// buys. (Distractors are kept OFF the signal channel so retention, not conditional
// selection, is the sole bottleneck — the pure test of the lower bound.)
func TestHGRNHierarchyPaysOff(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping training loop in -short")
	}
	const (
		d     = 4
		l     = 16
		batch = 16
		steps = 500
		noise = 0.05
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
			x.SetF64(s, 0, 0) // the signal to recall, on channel 0 at t=0
			for tt := range l {
				for k := range d {
					if tt == 0 && k == 0 {
						continue // keep the signal channel clean
					}
					x.SetF64(noise*rng.NormFloat64(), tt, k) // background noise
				}
			}
			xs[b], ys[b] = x, s
		}
		return xs, ys
	}
	// train freezes the forget-gate weights (constant decay from γ) and trains only the
	// candidate W_i and the readout; returns (firstLoss, lastLoss).
	train := func(gamma float64, seed uint64) (float64, float64) {
		m, err := nn.NewHGRN(tensor.F64, d, nn.WithHGRNSeed(seed), nn.WithHGRNLowerBound(gamma))
		if err != nil {
			t.Fatal(err)
		}
		nn.Zeros(m.Wf.W) // f_t = γ + (1−γ)·σ(0) = (1+γ)/2, constant per step
		readout := nn.NewLinear(tensor.F64, d, 1, seed+50)
		params := append([]*tensor.Tensor{m.Wi.W, m.Wi.B}, readout.Params()...)
		opt := nn.NewAdam(params, 1e-2)
		var first, last float64
		for st := range steps {
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
			if st == 0 {
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

	hiFirst, hiLast := train(0.99, 100) // deep-layer high bound: retains
	loFirst, loLast := train(0.0, 200)  // shallow γ≈0: forgets fast
	t.Logf("high-γ (0.99) MSE: %.4f → %.4f", hiFirst, hiLast)
	t.Logf("low-γ  (0.00) MSE: %.4f → %.4f", loFirst, loLast)

	if hiLast >= 0.5*hiFirst {
		t.Errorf("high-γ HGRU did not learn the recall task: %.4f → %.4f", hiFirst, hiLast)
	}
	if hiLast >= loLast {
		t.Errorf("lower bound did not help long-range recall: high-γ %.4f, low-γ %.4f", hiLast, loLast)
	}
}

// §V16 anchor 6 (sanity): construction/shape errors, γ range enforcement, determinism.
func TestHGRNErrors(t *testing.T) {
	if _, err := nn.NewHGRN(tensor.F64, 0); err == nil {
		t.Error("dModel=0: want error")
	}
	if _, err := nn.NewHGRN(tensor.F64, -3); err == nil {
		t.Error("dModel<0: want error")
	}
	if _, err := nn.NewHGRN(tensor.F64, 4, nn.WithHGRNLowerBound(1)); err == nil {
		t.Error("γ=1: want error (must be < 1)")
	}
	if _, err := nn.NewHGRN(tensor.F64, 4, nn.WithHGRNLowerBound(-0.1)); err == nil {
		t.Error("γ<0: want error")
	}
	m, err := nn.NewHGRN(tensor.F64, 4, nn.WithHGRNSeed(1))
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
	// Determinism: same seed ⇒ identical output.
	a, _ := nn.NewHGRN(tensor.F64, 4, nn.WithHGRNSeed(77), nn.WithHGRNLowerBound(0.4))
	b, _ := nn.NewHGRN(tensor.F64, 4, nn.WithHGRNSeed(77), nn.WithHGRNLowerBound(0.4))
	x := hgrnRand(rand.New(rand.NewPCG(1, 1)), 5, 4)
	oa, _ := a.Forward(backend.NewContext(), x)
	ob, _ := b.Forward(backend.NewContext(), x)
	for i := range oa.Numel() {
		coord := tensor.Unravel(i, oa.Shape())
		if oa.AtF64(coord...) != ob.AtF64(coord...) {
			t.Errorf("same seed diverged at %v", coord)
		}
	}
}

func hgrnRand(rng *rand.Rand, r, c int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			t.SetF64(rng.NormFloat64(), i, j)
		}
	}
	return t
}

func hgrnSum(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
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

func ExampleHGRN() {
	// A single HGRU layer with a HIGH lower bound γ=0.9 forces the forget gate to stay
	// ≥ 0.9, so the state retains. Its two forward paths — the parallel training form
	// and the recurrent scan — agree exactly (the duality). Over x=[1,1] the state
	// accumulates the per-step writes b_t = (1−f_t)·SiLU(ĩ_t).
	m, _ := nn.NewHGRN(tensor.F64, 1, nn.WithHGRNSeed(0), nn.WithHGRNLowerBound(0.9))
	x := tensor.New(tensor.F64, tensor.Shape{2, 1})
	x.SetF64(1, 0, 0)
	x.SetF64(1, 1, 0)
	par, _ := m.Forward(backend.NewContext(), x)
	seq, _ := m.ForwardSequential(backend.NewContext(), x)
	fmt.Printf("parallel:   %.4f %.4f\n", par.AtF64(0, 0), par.AtF64(1, 0))
	fmt.Printf("sequential: %.4f %.4f\n", seq.AtF64(0, 0), seq.AtF64(1, 0))
	// hierarchy: shallow forgets fast (γ_1=0), deep retains (γ_4→1)
	fmt.Printf("bounds(4): %v\n", nn.HGRNLayerBounds(4))
	// Output:
	// parallel:   0.0037 0.0073
	// sequential: 0.0037 0.0073
	// bounds(4): [0 0.25 0.5 0.75]
}
