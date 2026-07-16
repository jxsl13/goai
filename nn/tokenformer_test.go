package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// geluExact is the exact-erf GELU the ref backend's OpGELU computes, used to
// hand-check Θ independently of the layer.
func geluExact(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

// scoreRow returns S_i = X_i·Kᵀ for one input row.
func scoreRow(xd, kd []float64, dIn, nTok, b int) []float64 {
	s := make([]float64, nTok)
	for tk := 0; tk < nTok; tk++ {
		for c := 0; c < dIn; c++ {
			s[tk] += xd[b*dIn+c] * kd[tk*dIn+c]
		}
	}
	return s
}

// §V16.2 (default/coupled): Y = Θ(X·Kᵀ)·V with the paper's Eq. 5 normalization
// Θ(S)_ij = GELU(S_ij·τ/√(ΣS²+ε)), τ=√n, hand-recomputed at f64 1e-10.
func TestPattentionHandCheckCoupled(t *testing.T) {
	const dIn, dOut, nTok, batch = 3, 2, 2, 2
	xd := []float64{1, -1, 0.5, 0.3, 2, -0.4}
	kd := []float64{0.2, -0.5, 1.0, 0.7, 0.1, -0.3}
	vd := []float64{1.5, -0.5, 0.25, 2.0}
	eps := 1e-12
	tau := math.Sqrt(float64(nTok))
	p := &nn.PattentionLayer{
		K:   tensor.FromFloat64(tensor.Shape{nTok, dIn}, kd),
		V:   tensor.FromFloat64(tensor.Shape{nTok, dOut}, vd),
		Tau: tau, Eps: eps,
	}
	y, err := p.Forward(backend.NewContext(), tensor.FromFloat64(tensor.Shape{batch, dIn}, xd))
	if err != nil {
		t.Fatal(err)
	}
	for b := 0; b < batch; b++ {
		s := scoreRow(xd, kd, dIn, nTok, b)
		var ss float64
		for _, v := range s {
			ss += v * v
		}
		den := math.Sqrt(ss + eps)
		a := make([]float64, nTok)
		for tk := range a {
			a[tk] = geluExact(s[tk] * tau / den)
		}
		for o := 0; o < dOut; o++ {
			var want float64
			for tk := 0; tk < nTok; tk++ {
				want += a[tk] * vd[tk*dOut+o]
			}
			if got := y.AtF64(b, o); math.Abs(got-want) > 1e-10 {
				t.Errorf("coupled Y[%d,%d]=%.12g want %.12g", b, o, got, want)
			}
		}
	}
}

// §V16.2 (uncoupled option): Y = GELU(S·scale)·V hand-recomputed at f64 1e-10.
func TestPattentionHandCheckUncoupled(t *testing.T) {
	const dIn, dOut, nTok, batch = 3, 2, 2, 2
	xd := []float64{1, -1, 0.5, 0.3, 2, -0.4}
	kd := []float64{0.2, -0.5, 1.0, 0.7, 0.1, -0.3}
	vd := []float64{1.5, -0.5, 0.25, 2.0}
	scale := 0.5
	p := &nn.PattentionLayer{
		K:         tensor.FromFloat64(tensor.Shape{nTok, dIn}, kd),
		V:         tensor.FromFloat64(tensor.Shape{nTok, dOut}, vd),
		Uncoupled: true, Scale: scale,
	}
	y, err := p.Forward(backend.NewContext(), tensor.FromFloat64(tensor.Shape{batch, dIn}, xd))
	if err != nil {
		t.Fatal(err)
	}
	for b := 0; b < batch; b++ {
		s := scoreRow(xd, kd, dIn, nTok, b)
		for o := 0; o < dOut; o++ {
			var want float64
			for tk := 0; tk < nTok; tk++ {
				want += geluExact(s[tk]*scale) * vd[tk*dOut+o]
			}
			if got := y.AtF64(b, o); math.Abs(got-want) > 1e-10 {
				t.Errorf("uncoupled Y[%d,%d]=%.12g want %.12g", b, o, got, want)
			}
		}
	}
}

// §V16.1 (uncoupled option): appending token pairs with ZERO value rows leaves the
// output BIT-EXACT (max|Δ|=0) — the per-score normalization's exact-growth guarantee.
func TestPattentionGrowBitExactUncoupled(t *testing.T) {
	p := nn.NewPattention(4, 3, 5, 42, nn.WithPattentionUncoupledNorm())
	x := tensor.FromFloat64(tensor.Shape{3, 4}, []float64{
		1, -1, 0.5, 2, 0.3, 1, -2, 0.7, -0.4, 0.9, 1.2, -0.1,
	})
	before, err := p.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	p.Grow(7, 99) // random keys, ZERO values
	if p.NumTokens() != 12 {
		t.Fatalf("NumTokens after grow = %d, want 12", p.NumTokens())
	}
	after, err := p.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(before, after); d > 1e-10 {
		t.Fatalf("uncoupled Grow changed output: max|Δ|=%g (want bit-exact)", d)
	}
	for tk := 5; tk < 12; tk++ {
		for o := 0; o < 3; o++ {
			if v := p.V.AtF64(tk, o); v != 0 {
				t.Errorf("appended value row %d col %d = %g, want 0", tk, o, v)
			}
		}
	}
	t.Logf("uncoupled Grow 5→12: output invariant, max|Δ|=0")
}

// coupledGrown clones a base coupled layer and appends delta token pairs whose new
// key rows are (deterministic random) scaled by newKeyScale and new value rows are
// ZERO, keeping τ FROZEN at the base value. Used to probe the approximate-growth bound.
func coupledGrown(base *nn.PattentionLayer, delta int, newKeyScale float64, seed uint64) *nn.PattentionLayer {
	old := base.K.Shape()[0]
	dIn := base.K.Shape()[1]
	dOut := base.V.Shape()[1]
	dk := tensor.New(tensor.F64, tensor.Shape{delta, dIn})
	nn.XavierUniform(dk, dIn, delta, seed)
	k := tensor.New(tensor.F64, tensor.Shape{old + delta, dIn})
	v := tensor.New(tensor.F64, tensor.Shape{old + delta, dOut})
	for r := 0; r < old; r++ {
		for c := 0; c < dIn; c++ {
			k.SetF64(base.K.AtF64(r, c), r, c)
		}
		for c := 0; c < dOut; c++ {
			v.SetF64(base.V.AtF64(r, c), r, c)
		}
	}
	for r := 0; r < delta; r++ {
		for c := 0; c < dIn; c++ {
			k.SetF64(dk.AtF64(r, c)*newKeyScale, old+r, c)
		}
	}
	return &nn.PattentionLayer{K: k, V: v, Tau: base.Tau, Eps: base.Eps}
}

// §V16.1 (default/coupled): growth is APPROXIMATE (resume-from-checkpoint). The
// output perturbation from appending zero-value tokens is BOUNDED and shrinks
// monotonically toward 0 as the new keys' scores → 0, hitting exactly 0 (bit-exact)
// when the new keys are zero — the paper's zero-init-keys resume property.
func TestPattentionGrowApproximateCoupled(t *testing.T) {
	base := nn.NewPattention(4, 3, 5, 42) // default = coupled Eq. 5
	x := tensor.FromFloat64(tensor.Shape{3, 4}, []float64{
		1, -1, 0.5, 2, 0.3, 1, -2, 0.7, -0.4, 0.9, 1.2, -0.1,
	})
	y0, err := base.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}

	// Ever-smaller new-key scales → the perturbation must shrink monotonically and
	// vanish. scale 0 (zero-scored new keys) is bit-exact given the frozen τ.
	scales := []float64{0.1, 0.03, 0.01, 0}
	diffs := make([]float64, len(scales))
	prev := math.Inf(1)
	for i, s := range scales {
		g := coupledGrown(base, 7, s, 99)
		ys, err := g.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		d := maxAbsDiff(y0, ys)
		if d > prev+1e-12 {
			t.Errorf("perturbation not shrinking: scale %.3g gave max|Δ|=%g > previous %g", s, d, prev)
		}
		diffs[i] = d
		prev = d
		t.Logf("coupled Grow, new-key scale %.3g → max|Δ|=%.3e", s, d)
	}
	if smallest := diffs[len(diffs)-2]; smallest > 2e-2 { // scale 0.01 → near-lossless resume
		t.Errorf("coupled Grow not near-lossless at small keys: max|Δ|=%g", smallest)
	}
	if last := diffs[len(diffs)-1]; last > 1e-10 { // scale 0 → bit-exact
		t.Fatalf("zero-scored coupled Grow not bit-exact: max|Δ|=%g", last)
	}
}

// §V16.3: finite-difference gradcheck of Σ(Pattention) w.r.t. X, K and V, in BOTH
// normalization modes.
func TestPattentionGradCheck(t *testing.T) {
	for _, uncoupled := range []bool{false, true} {
		name := "coupled"
		var opts []nn.PattentionOption
		if uncoupled {
			name, opts = "uncoupled", []nn.PattentionOption{nn.WithPattentionUncoupledNorm()}
		}
		t.Run(name, func(t *testing.T) {
			p := nn.NewPattention(3, 2, 4, 7, opts...)
			x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{0.5, -1, 1.2, -0.3, 0.8, 0.4})

			sumOut := func() float64 {
				y, err := p.Forward(backend.NewContext(), x)
				if err != nil {
					t.Fatal(err)
				}
				var s float64
				for k := 0; k < y.Numel(); k++ {
					s += y.AtF64(tensor.Unravel(k, y.Shape())...)
				}
				return s
			}
			tape := autograd.NewTape()
			y, err := p.Forward(tape.Context(), x)
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(y); err != nil {
				t.Fatal(err)
			}
			const h = 1e-6
			for label, in := range map[string]*tensor.Tensor{"X": x, "K": p.K, "V": p.V} {
				grad := tape.Grad(in)
				if grad == nil {
					t.Fatalf("%s received no gradient", label)
				}
				for k := 0; k < in.Numel(); k++ {
					idx := tensor.Unravel(k, in.Shape())
					o := in.AtF64(idx...)
					in.SetF64(o+h, idx...)
					lp := sumOut()
					in.SetF64(o-h, idx...)
					lm := sumOut()
					in.SetF64(o, idx...)
					if num, ana := (lp-lm)/(2*h), grad.AtF64(idx...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
						t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", label, k, num, ana)
					}
				}
			}
		})
	}
}

// teacher builds a fixed nonlinear target Y = GELU(X·A)·B for random A,B — a map
// outside the layer's exact parameterization, so fitting it exercises Pattention
// as a genuine trainable nonlinear projection.
func teacher(x *tensor.Tensor, dIn, dOut, hidden int) *tensor.Tensor {
	a := tensor.New(tensor.F64, tensor.Shape{dIn, hidden})
	nn.XavierUniform(a, dIn, hidden, 1234)
	b := tensor.New(tensor.F64, tensor.Shape{hidden, dOut})
	nn.XavierUniform(b, hidden, dOut, 5678)
	ctx := backend.NewContext()
	h, _ := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, a}, nil)
	g, _ := backend.Execute(ctx, backend.OpGELU, h, nil)
	y, _ := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{g[0], b}, nil)
	return y[0]
}

// §V16.4 (the value): in the DEFAULT coupled (Eq. 5) mode, a Pattention module
// trained on a nonlinear target reduces its loss; Grow perturbs it only slightly
// (near-lossless resume) and the added capacity keeps reducing the loss further.
func TestPattentionLearnsAndGrowsCoupled(t *testing.T) {
	const dIn, dOut, batch, hidden = 4, 3, 24, 6
	x := tensor.New(tensor.F64, tensor.Shape{batch, dIn})
	nn.XavierUniform(x, dIn, batch, 2024)
	target := teacher(x, dIn, dOut, hidden)

	p := nn.NewPattention(dIn, dOut, 4, 11) // coupled Eq. 5

	train := func(opt nn.Optimizer, steps int) {
		for s := 0; s < steps; s++ {
			tape := autograd.NewTape()
			y, err := p.Forward(tape.Context(), x)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.MSE(tape.Context(), y, target)
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
	}
	loss0 := func() float64 {
		y, _ := p.Forward(backend.NewContext(), x)
		l, _ := nn.MSE(backend.NewContext(), y, target)
		return l.AtF64()
	}

	start := loss0()
	train(nn.NewAdam(p.Params(), 0.03), 300)
	preGrow := loss0()
	if preGrow >= start {
		t.Fatalf("Pattention did not learn: loss %.5f → %.5f", start, preGrow)
	}

	p.Grow(4, 77)
	atGrow := loss0()
	bump := math.Abs(atGrow - preGrow)
	if bump > 0.2*preGrow+1e-6 { // near-lossless resume: only a small approximate perturbation
		t.Fatalf("coupled Grow perturbed loss too much: %.6f → %.6f (bump %.6f)", preGrow, atGrow, bump)
	}
	train(nn.NewAdam(p.Params(), 0.03), 300)
	grownFinal := loss0()
	if grownFinal >= preGrow {
		t.Fatalf("added capacity did not help: base %.5f, grown %.5f", preGrow, grownFinal)
	}
	t.Logf("coupled MSE %.5f → %.5f (4 tok) → grow (bump %.2e) → %.5f (8 tok)", start, preGrow, bump, grownFinal)
}

// §V16.5: identical seeds give identical layers (determinism) and the shape/arg
// guards reject bad calls.
func TestPattentionDeterminismAndGuards(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	a := nn.NewPattention(3, 2, 5, 314)
	b := nn.NewPattention(3, 2, 5, 314)
	ya, _ := a.Forward(backend.NewContext(), x)
	yb, _ := b.Forward(backend.NewContext(), x)
	if d := maxAbsDiff(ya, yb); d != 0 {
		t.Fatalf("same seed diverged: max|Δ|=%g", d)
	}

	if _, err := a.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3})); err == nil {
		t.Error("rank-1 x must error")
	}
	if _, err := a.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 9})); err == nil {
		t.Error("wrong dIn must error")
	}
	mustPanic(t, "nTokens<1", func() { nn.NewPattention(3, 2, 0, 1) })
	mustPanic(t, "deltaTokens<1", func() { a.Grow(0, 1) })
}

func mustPanic(t *testing.T, what string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic", what)
		}
	}()
	fn()
}

// ExamplePattentionLayer shows a Pattention projection and incremental growth: the
// default coupled Eq. 5 normalization means appending zero-value token pairs is a
// near-lossless resume (bounded perturbation), not a bit-exact one.
func ExamplePattentionLayer() {
	p := nn.NewPattention(4, 3, 8, 1)
	x := tensor.FromFloat64(tensor.Shape{1, 4}, []float64{0.5, -1, 0.25, 2})

	before, _ := p.Forward(backend.NewContext(), x)
	p.Grow(4, 2) // 8 → 12 tokens: small keys, zero-initialized value rows
	after, _ := p.Forward(backend.NewContext(), x)

	var maxDiff float64
	for i := 0; i < before.Numel(); i++ {
		idx := tensor.Unravel(i, before.Shape())
		if d := math.Abs(before.AtF64(idx...) - after.AtF64(idx...)); d > maxDiff {
			maxDiff = d
		}
	}
	fmt.Printf("tokens %d→%d, near-lossless resume: %t\n", 8, p.NumTokens(), maxDiff < 0.05)
	// Output: tokens 8→12, near-lossless resume: true
}
