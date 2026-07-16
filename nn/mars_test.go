package nn_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// marsRef is an independent straight-line transcription of approximate MARS-AdamW
// (arXiv:2411.10438, is_approx=True; the reference AGI-Arena/MARS mars.py), used by the
// parity and variance-reduction tests below. It carries the full state — corrected
// gradient c, first/second moments m/v, previous gradient gPrev — so a mismatch in any of
// them surfaces through the parameters it drives.
type marsRef struct {
	lr, b1, b2, eps, wd, gamma, clip float64
	p, m, v, gPrev, c                []float64
	seen                             bool
	t                                int
}

func newMARSRef(p0 []float64, lr float64) *marsRef {
	return &marsRef{
		lr: lr, b1: 0.95, b2: 0.99, eps: 1e-8, gamma: 0.025,
		p:     append([]float64(nil), p0...),
		m:     make([]float64, len(p0)),
		v:     make([]float64, len(p0)),
		gPrev: make([]float64, len(p0)),
		c:     make([]float64, len(p0)),
	}
}

// step advances the reference by one MARS update on gradient g and returns the corrected
// gradient c_t it used (a copy).
func (r *marsRef) step(g []float64) []float64 {
	r.t++
	c1 := 1 - math.Pow(r.b1, float64(r.t))
	c2 := 1 - math.Pow(r.b2, float64(r.t))
	factor := 0.0
	if r.gamma != 0 {
		factor = r.gamma * r.b1 / (1 - r.b1)
	}
	for i := range r.p {
		if factor != 0 && r.seen {
			r.c[i] = g[i] + factor*(g[i]-r.gPrev[i])
		} else {
			r.c[i] = g[i]
		}
	}
	if r.clip > 0 {
		ss := 0.0
		for _, v := range r.c {
			ss += v * v
		}
		if norm := math.Sqrt(ss); norm > r.clip {
			s := r.clip / norm
			for i := range r.c {
				r.c[i] *= s
			}
		}
	}
	for i := range r.p {
		r.m[i] = r.b1*r.m[i] + (1-r.b1)*r.c[i]
		r.v[i] = r.b2*r.v[i] + (1-r.b2)*r.c[i]*r.c[i]
		r.p[i] = r.p[i]*(1-r.lr*r.wd) - r.lr*(r.m[i]/c1)/(math.Sqrt(r.v[i]/c2)+r.eps)
		r.gPrev[i] = g[i]
	}
	r.seen = true
	return append([]float64(nil), r.c...)
}

// §V16 tier-1: the MARS trajectory reproduces the independent straight-line transcription
// of approximate MARS-AdamW step by step at rtol 1e-12. Because the update
// x ← x·(1−lr·wd) − lr·(m/c1)/(√(v/c2)+ε) is a smooth invertible function of (m, v) and
// m, v are driven by c_t, matching the PARAMETERS bit-for-bit over many steps transitively
// pins m, v and c_t. Two configs are exercised: (a) weight decay + explicit betas/eps, and
// (b) the reference's unit-norm corrected-gradient clip — both starting from the
// first-step (no g_{t-1}) case c_1 = g_1.
func TestMARSAlgorithmParity(t *testing.T) {
	p0 := []float64{1.0, -2.0, 0.5}
	grads := [][]float64{
		{0.5, 0.5, -0.3},
		{0.2, 0.4, -0.1},
		{-0.4, 0.6, 0.2},
		{0.1, 0.5, 0.0},
		{0.3, -0.2, 0.4},
	}
	const lr = 0.01

	for _, cfg := range []struct {
		name     string
		setupRef func(r *marsRef)
		opts     []nn.MARSOption
	}{
		{
			"wd+betas+eps",
			func(r *marsRef) { r.wd, r.b1, r.b2, r.eps = 0.02, 0.9, 0.999, 1e-7 },
			[]nn.MARSOption{
				nn.WithMARSWeightDecay(0.02), nn.WithMARSBetas(0.9, 0.999), nn.WithMARSEps(1e-7),
			},
		},
		{
			"clip",
			func(r *marsRef) { r.clip = 1.0 },
			[]nn.MARSOption{nn.WithMARSClip(1.0)},
		},
	} {
		t.Run(cfg.name, func(t *testing.T) {
			ref := newMARSRef(p0, lr)
			cfg.setupRef(ref)

			p := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
			opt := nn.NewMARS([]*tensor.Tensor{p}, lr, cfg.opts...)
			for k := range grads {
				gs := grads[k]
				ref.step(gs)
				if err := opt.Step(func(q *tensor.Tensor) *tensor.Tensor {
					if q != p {
						return nil
					}
					return tensor.FromFloat64(p.Shape(), gs)
				}); err != nil {
					t.Fatal(err)
				}
				for i := range p.Numel() {
					got := p.AtF64(i)
					if math.Abs(got-ref.p[i]) > 1e-12*math.Max(1, math.Abs(ref.p[i])) {
						t.Errorf("step %d p[%d]: got %.16g want %.16g", k, i, got, ref.p[i])
					}
				}
			}
		})
	}
}

// §V16 tier-1 COLLAPSE anchor: γ=0 makes the correction identically zero, so c_t=g_t and
// MARS-AdamW is AdamW BIT-FOR-BIT (== equality, not just rtol) over a multi-step run with
// weight decay, including the very first step. Clipping is off (the default), so nothing
// perturbs c_t.
func TestMARSGammaZeroIsAdamW(t *testing.T) {
	p0 := []float64{1.0, -2.0, 0.5, 3.0}
	grads := [][]float64{
		{0.5, 0.5, -0.3, 0.9},
		{0.2, 0.4, -0.1, -0.7},
		{-0.4, 0.6, 0.2, 0.3},
		{0.1, 0.5, 0.0, -0.2},
	}
	const lr, wd = 0.02, 0.05

	pM := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
	pA := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
	optM := nn.NewMARS([]*tensor.Tensor{pM}, lr,
		nn.WithMARSGamma(0), nn.WithMARSBetas(0.9, 0.999), nn.WithMARSWeightDecay(wd))
	optA := nn.NewAdamW([]*tensor.Tensor{pA}, lr, wd)

	for k := range grads {
		gs := grads[k]
		mk := func(p *tensor.Tensor) nn.GradFn {
			return func(q *tensor.Tensor) *tensor.Tensor {
				if q != p {
					return nil
				}
				return tensor.FromFloat64(p.Shape(), gs)
			}
		}
		if err := optM.Step(mk(pM)); err != nil {
			t.Fatal(err)
		}
		if err := optA.Step(mk(pA)); err != nil {
			t.Fatal(err)
		}
		for i := range pM.Numel() {
			if m, a := pM.AtF64(i), pA.AtF64(i); m != a {
				t.Errorf("step %d p[%d]: MARS(γ=0)=%.17g != AdamW=%.17g (must be bit-identical)", k, i, m, a)
			}
		}
	}
}

// §V16 tier-1 first-step ≡ plain AdamW: with no stored previous gradient the corrected
// gradient is c_1=g_1, so MARS's very first update — even at the default γ>0 — is exactly
// AdamW's. Bit-identical single step.
func TestMARSFirstStepIsAdamW(t *testing.T) {
	p0 := []float64{0.7, -1.3, 2.1}
	g := []float64{0.4, -0.6, 0.2}
	const lr = 0.01

	pM := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
	pA := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
	optM := nn.NewMARS([]*tensor.Tensor{pM}, lr, nn.WithMARSBetas(0.9, 0.999)) // γ=0.025 default
	optA := nn.NewAdam([]*tensor.Tensor{pA}, lr)
	step := func(opt nn.Optimizer, p *tensor.Tensor) {
		if err := opt.Step(func(q *tensor.Tensor) *tensor.Tensor {
			if q != p {
				return nil
			}
			return tensor.FromFloat64(p.Shape(), g)
		}); err != nil {
			t.Fatal(err)
		}
	}
	step(optM, pM)
	step(optA, pA)
	for i := range pM.Numel() {
		if m, a := pM.AtF64(i), pA.AtF64(i); m != a {
			t.Errorf("first step p[%d]: MARS=%.17g != AdamW=%.17g", i, m, a)
		}
	}
}

// §V16 tier-1 convergence: MARS reaches the noise floor of a small noisy least-squares
// regression, comparable to hand-tuned Adam at the same learning rate.
func TestMARSConvergence(t *testing.T) {
	pr := newRegressionProblem()
	const steps, lr = 3000, 0.01

	pM := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	excessM := pr.run(t, nn.NewMARS([]*tensor.Tensor{pM}, lr), pM, steps)

	pA := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	excessA := pr.run(t, nn.NewAdam([]*tensor.Tensor{pA}, lr), pA, steps)

	t.Logf("excess loss after %d steps: MARS=%.4g  Adam=%.4g", steps, excessM, excessA)
	// Both optimizers drive the finite-sample loss essentially to (and slightly past) the
	// w* floor, so excess is near zero and may be negative; assert MARS reaches the floor
	// (|excess| small) and is no materially worse than Adam at the same lr.
	if math.Abs(excessM) > 0.01 {
		t.Errorf("MARS must reach the noise floor: excess %.4g (|·|>0.01)", excessM)
	}
	if excessM > excessA+0.005 {
		t.Errorf("MARS (%.4g) should be comparable to Adam (%.4g) at the same lr", excessM, excessA)
	}
}

// §V16 tier-1 VARIANCE-REDUCTION value. HONEST DEVIATION from the naive brief: raw
// pointwise Var(c_t) < Var(g_t) is PROVABLY FALSE — for a stationary noisy gradient,
// c_t=(1+f)g_t−f·g_{t-1} has variance ≥ Var(g_t) always (equality only at perfect
// correlation), so the difference term can only ADD per-step noise. This test asserts what
// is actually true and is the property MARS is built on: on a MOVING true gradient, the
// corrected-gradient momentum (MARS's m̂, the direction that drives the step) tracks the
// true gradient with LOWER time-averaged squared error than the plain-gradient momentum
// (AdamW's m̂) — variance reduction via cancelled EMA lag. Computed on the reference
// recurrence (tied to the optimizer bit-for-bit by TestMARSAlgorithmParity); the raw-Var
// inequality is asserted too, to pin the honest direction.
func TestMARSVarianceReduction(t *testing.T) {
	const (
		b1, gamma = 0.9, 0.1
		steps     = 8000
		speed     = 0.03
		noise     = 0.3
		amp       = 2.0
	)
	factor := gamma * b1 / (1 - b1)
	rng := rand.New(rand.NewSource(1))

	var gPrev float64
	haveGprev := false
	var mg, mc float64 // plain-gradient and corrected-gradient EMAs
	var sqErrG, sqErrC float64
	var sumG, sumG2, sumC, sumC2 float64
	n := 0
	for k := range steps {
		tg := math.Sin(float64(k)*speed) * amp // moving true gradient
		g := tg + rng.NormFloat64()*noise      // noisy observation
		c := g
		if haveGprev {
			c = g + factor*(g-gPrev)
		}
		gPrev, haveGprev = g, true

		mg = b1*mg + (1-b1)*g
		mc = b1*mc + (1-b1)*c
		bc := 1 - math.Pow(b1, float64(k+1))
		if k > 300 { // past EMA warmup
			eg := mg/bc - tg
			ec := mc/bc - tg
			sqErrG += eg * eg
			sqErrC += ec * ec
			sumG += g
			sumG2 += g * g
			sumC += c
			sumC2 += c * c
			n++
		}
	}
	varG := sumG2/float64(n) - (sumG/float64(n))*(sumG/float64(n))
	varC := sumC2/float64(n) - (sumC/float64(n))*(sumC/float64(n))
	mseG := sqErrG / float64(n)
	mseC := sqErrC / float64(n)
	t.Logf("raw Var: g=%.4f c=%.4f (c≥g expected);  momentum-tracking MSE: g=%.5f c=%.5f (ratio %.3f)",
		varG, varC, mseG, mseC, mseC/mseG)

	// Honest direction: raw pointwise variance is NOT reduced (it rises).
	if varC < varG {
		t.Errorf("premise check: raw Var(c)=%.4f unexpectedly < Var(g)=%.4f", varC, varG)
	}
	// The real value: the corrected-gradient momentum tracks the moving true gradient with
	// materially lower error than the plain-gradient momentum.
	if mseC >= mseG {
		t.Errorf("variance reduction: momentum-of-c MSE %.5f must be < momentum-of-g MSE %.5f", mseC, mseG)
	}
	if mseC > 0.95*mseG {
		t.Errorf("variance reduction too weak: MSE ratio %.3f (want < 0.95)", mseC/mseG)
	}
}

// §V16 tier-1 fast-path guard: the typed contiguous f64/f32 fast paths must produce a
// BIT-IDENTICAL trajectory to the generic widening-accessor fallback, which is forced via
// non-contiguous transposed views. Exercised with the correction ON (γ>0) and clipping ON
// so both extra code paths are covered.
func TestMARSFastPathParity(t *testing.T) {
	const steps = 6
	pval := func(i int) float64 { return math.Sin(float64(i)*0.7) + 0.1 }
	gval := func(i, s int) float64 { return math.Cos(float64(i)*0.3+float64(s)) * 0.05 }

	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		t.Run(fmt.Sprintf("%v", dt), func(t *testing.T) {
			shapes := []tensor.Shape{{3, 5}, {2, 3, 4}}
			fast := make([]*tensor.Tensor, len(shapes))
			slow := make([]*tensor.Tensor, len(shapes))
			for pi, s := range shapes {
				fast[pi] = tensor.New(dt, s)
				slow[pi] = mustTransposedView(t, dt, s)
				for i := range s.Numel() {
					idx := tensor.Unravel(i, s)
					fast[pi].SetF64(pval(i), idx...)
					slow[pi].SetF64(pval(i), idx...)
				}
			}
			mk := func(ps []*tensor.Tensor) *nn.MARS {
				return nn.NewMARS(ps, 0.01, nn.WithMARSGamma(0.05),
					nn.WithMARSClip(0.2), nn.WithMARSWeightDecay(0.01))
			}
			optFast, optSlow := mk(fast), mk(slow)
			for s := range steps {
				gFast := map[*tensor.Tensor]*tensor.Tensor{}
				gSlow := map[*tensor.Tensor]*tensor.Tensor{}
				for pi, shape := range shapes {
					cg := tensor.New(dt, shape)
					vg := mustTransposedView(t, dt, shape)
					for i := range shape.Numel() {
						idx := tensor.Unravel(i, shape)
						cg.SetF64(gval(i, s), idx...)
						vg.SetF64(gval(i, s), idx...)
					}
					gFast[fast[pi]] = cg
					gSlow[slow[pi]] = vg
				}
				if err := optFast.Step(func(p *tensor.Tensor) *tensor.Tensor { return gFast[p] }); err != nil {
					t.Fatalf("fast step %d: %v", s, err)
				}
				if err := optSlow.Step(func(p *tensor.Tensor) *tensor.Tensor { return gSlow[p] }); err != nil {
					t.Fatalf("slow step %d: %v", s, err)
				}
				for pi, shape := range shapes {
					for i := range shape.Numel() {
						idx := tensor.Unravel(i, shape)
						fv, sv := fast[pi].AtF64(idx...), slow[pi].AtF64(idx...)
						if fv != sv {
							t.Fatalf("step %d p%d[%v]: fast %.17g != generic %.17g", s, pi, idx, fv, sv)
						}
					}
				}
			}
		})
	}
}

// §V16 sanity: nil gradients are skipped, all-zero gradients are a NaN-free finite step,
// and a shape mismatch errors like the other optimizers.
func TestMARSSanity(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	opt := nn.NewMARS([]*tensor.Tensor{p}, 0.01)

	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return nil }); err != nil {
		t.Fatal(err)
	}
	if p.AtF64(0) != 1 || p.AtF64(1) != 2 {
		t.Error("nil-grad param must be skipped (no motion)")
	}

	zero := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 0})
	for range 3 {
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return zero }); err != nil {
			t.Fatal(err)
		}
	}
	if math.IsNaN(p.AtF64(0)) || math.IsNaN(p.AtF64(1)) {
		t.Errorf("zero-grad step must stay finite: p=[%g %g]", p.AtF64(0), p.AtF64(1))
	}

	bad := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return bad }); err == nil {
		t.Error("shape mismatch must error")
	}
}

// MARS is AdamW driven by a variance-reduced gradient: it adds a small extrapolation from
// how the gradient just moved (scaled by γ) so the momentum keeps up with a changing
// gradient. Here it minimizes ½(x−3)² from x=0; after 400 steps x reaches the optimum.
// Setting γ=0 would recover plain AdamW exactly.
func ExampleMARS() {
	x := tensor.FromFloat64(tensor.Shape{1}, []float64{0})
	opt := nn.NewMARS([]*tensor.Tensor{x}, 0.05)
	for range 400 {
		_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
			return tensor.FromFloat64(tensor.Shape{1}, []float64{x.AtF64(0) - 3.0}) // ∇½(x−3)²
		})
	}
	fmt.Printf("x=%.3f\n", x.AtF64(0))
	// Output: x=3.000
}
