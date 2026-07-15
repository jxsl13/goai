package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// prodigyRef is an independent straight-line transcription of the authors' reference
// implementation (konstmish/prodigy prodigy.py; the refined form of arXiv:2306.06101
// Algorithm 3), used by the parity tests below.
type prodigyRef struct {
	lr, b1, b2, b3, eps, wd float64
	d0, dCoef, growth       float64
	useBC, safeguard        bool
	d, dMax, rNum           float64
	p, x0, m, v, s          []float64
	k                       int
}

func newProdigyRef(p0 []float64) *prodigyRef {
	r := &prodigyRef{
		lr: 1, b1: 0.9, b2: 0.999, eps: 1e-8,
		d0: 1e-6, dCoef: 1, growth: math.Inf(1),
	}
	r.b3 = math.Sqrt(r.b2)
	r.d, r.dMax = r.d0, r.d0
	r.p = append([]float64(nil), p0...)
	r.x0 = append([]float64(nil), p0...)
	r.m = make([]float64, len(p0))
	r.v = make([]float64, len(p0))
	r.s = make([]float64, len(p0))
	return r
}

func (r *prodigyRef) step(g []float64) {
	bc := 1.0
	if r.useBC {
		t1 := float64(r.k + 1)
		bc = math.Sqrt(1-math.Pow(r.b2, t1)) / (1 - math.Pow(r.b1, t1))
	}
	dlr := r.d * r.lr * bc
	sCoef := (r.d / r.d0) * dlr
	if r.safeguard {
		sCoef = (r.d / r.d0) * r.d
	}
	delta := 0.0
	for i := range r.p {
		delta += (r.d / r.d0) * dlr * g[i] * (r.x0[i] - r.p[i])
	}
	dDenom := 0.0
	for i := range r.p {
		r.m[i] = r.b1*r.m[i] + (1-r.b1)*r.d*g[i]
		r.v[i] = r.b2*r.v[i] + (1-r.b2)*r.d*r.d*g[i]*g[i]
		r.s[i] = r.b3*r.s[i] + sCoef*g[i]
		dDenom += math.Abs(r.s[i])
	}
	if dDenom == 0 {
		return
	}
	rNum := r.b3*r.rNum + delta
	dHat := r.dCoef * rNum / dDenom
	d := r.d
	if d == r.d0 {
		d = math.Max(d, dHat)
	}
	r.dMax = math.Max(r.dMax, dHat)
	d = math.Min(r.dMax, d*r.growth)
	r.rNum = rNum
	r.d = d
	for i := range r.p {
		r.p[i] = r.p[i]*(1-r.wd*dlr) - dlr*r.m[i]/(math.Sqrt(r.v[i])+d*r.eps)
	}
	r.k++
}

// §V16 tier-1: the Prodigy trajectory (parameters AND the d_k estimate) reproduces the
// authors' reference implementation (konstmish/prodigy prodigy.py; the refined form of
// arXiv:2306.06101 Algorithm 3) step by step, checked against an independent straight-line
// transcription of the same recurrence at rtol 1e-12. Two configurations are exercised:
// (a) bias correction + weight decay + a finite growth cap + explicit β₃, and (b) the
// safeguard-warmup accumulation variant. Per step (config a):
//
//	dlr = d·lr·√(1−β₂ᵏ⁺¹)/(1−β₁ᵏ⁺¹)
//	m ← β₁m + (1−β₁)·d·g;  v ← β₂v + (1−β₂)·d²·g²
//	r ← β₃r + (d/d₀)·dlr·⟨g, x₀−x⟩;  s ← β₃s + (d/d₀)·dlr·g
//	d̂ = d_coef·r/‖s‖₁;  d_max ← max(d_max, d̂);  d ← min(d_max, d·growth)
//	x ← (1−λ·dlr)·x − dlr·m/(√v + d_new·ε)
func TestProdigyAlgorithmParity(t *testing.T) {
	p0 := []float64{1.0, -2.0, 0.5}
	grads := [][]float64{
		{0.5, 0.5, -0.3},
		{0.2, 0.4, -0.1},
		{-0.4, 0.6, 0.2},
		{0.1, 0.5, 0.0},
		{0.3, -0.2, 0.4},
	}

	for _, cfg := range []struct {
		name      string
		setupRef  func(r *prodigyRef)
		setupOpts []nn.ProdigyOption
	}{
		{
			"bc+wd+growth+beta3",
			func(r *prodigyRef) {
				r.useBC, r.wd, r.growth, r.b3, r.dCoef = true, 0.02, 1.5, 0.995, 0.8
			},
			[]nn.ProdigyOption{
				nn.WithProdigyBiasCorrection(true), nn.WithProdigyWeightDecay(0.02),
				nn.WithProdigyGrowthRate(1.5), nn.WithProdigyBeta3(0.995),
				nn.WithProdigyDCoef(0.8),
			},
		},
		{
			"safeguard-warmup",
			func(r *prodigyRef) { r.safeguard = true },
			[]nn.ProdigyOption{nn.WithProdigySafeguardWarmup(true)},
		},
	} {
		t.Run(cfg.name, func(t *testing.T) {
			ref := newProdigyRef(p0)
			cfg.setupRef(ref)

			p := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
			opt := nn.NewProdigy([]*tensor.Tensor{p}, cfg.setupOpts...)
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
				if got := opt.D(); math.Abs(got-ref.d) > 1e-12*math.Max(1, math.Abs(ref.d)) {
					t.Errorf("step %d d: got %.16g want %.16g", k, got, ref.d)
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

// §V16 tier-1, THE value test: Prodigy with its DEFAULTS (nominal LR=1, no tuning at all)
// reaches the noise floor of a problem whose true scale is ~0.03, comparable to the best
// hand-tuned Adam, while a fixed-lr Adam at the same nominal lr=1 stalls two orders of
// magnitude higher. This is the learning-rate-free property the method exists for.
func TestProdigyLRFreeConvergence(t *testing.T) {
	pr := newRegressionProblem()
	const steps = 2000

	pP := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	optP := nn.NewProdigy([]*tensor.Tensor{pP}) // NO learning rate anywhere
	excessP := pr.run(t, optP, pP, steps)

	pA1 := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	excessA1 := pr.run(t, nn.NewAdam([]*tensor.Tensor{pA1}, 1.0), pA1, steps) // same nominal lr

	pAT := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	excessAT := pr.run(t, nn.NewAdam([]*tensor.Tensor{pAT}, 0.01), pAT, steps) // best hand-tuned lr

	t.Logf("excess loss after %d steps: Prodigy(default)=%.4g  Adam(lr=1)=%.4g  Adam(tuned lr=0.01)=%.4g  d=%.4g",
		steps, excessP, excessA1, excessAT, optP.D())

	if excessP > 0.01 {
		t.Errorf("Prodigy with defaults must reach the noise floor untuned: excess %.4g > 0.01", excessP)
	}
	if excessA1 < 0.05 {
		t.Errorf("premise broken: fixed-lr Adam at lr=1 should stall on this problem, excess %.4g", excessA1)
	}
	if excessP > excessA1/10 {
		t.Errorf("Prodigy (%.4g) must beat untuned Adam@1 (%.4g) by >10x", excessP, excessA1)
	}
	if excessP > 10*excessAT {
		t.Errorf("Prodigy (%.4g) not comparable to tuned Adam (%.4g)", excessP, excessAT)
	}
	if d := optP.D(); d < 1e-3 || d > 0.5 {
		t.Errorf("d should land near the problem scale ~0.03: got %.4g", d)
	}
}

// §V16 tier-1: Prodigy's d_k estimate grows from its tiny 1e-6 init toward the problem
// scale within a handful of steps (the exponential growth phase the paper's d_k-weighted
// numerator exists for, arXiv:2306.06101 §1), is monotone nondecreasing, STABILIZES once
// the scale is found (no further growth over the remaining 90% of training), and on a
// 10x-scaled copy of the problem lands ~10x higher — d tracks the initial distance
// ‖x₀−x*‖ it is defined to estimate.
func TestProdigyDTracksProblemScale(t *testing.T) {
	a := []float64{2.0, 0.5, 1.0, 4.0}
	base := []float64{3.0, -5.0, 2.0, -1.0}

	finalD := func(scale float64) (d200, dEnd, loss float64) {
		star := make([]float64, len(base))
		for i, v := range base {
			star[i] = v * scale
		}
		p := tensor.FromFloat64(tensor.Shape{len(star)}, make([]float64, len(star)))
		opt := nn.NewProdigy([]*tensor.Tensor{p})
		g := quadGrad(p, a, star)
		prev := opt.D()
		for k := range 2000 {
			if err := opt.Step(g); err != nil {
				t.Fatal(err)
			}
			if opt.D() < prev {
				t.Fatalf("d must be monotone nondecreasing: %g -> %g", prev, opt.D())
			}
			prev = opt.D()
			if k == 199 {
				d200 = opt.D()
			}
		}
		return d200, opt.D(), quadLoss(p, a, star)
	}

	d200, d1, loss1 := finalD(1)
	_, d10, _ := finalD(10)
	t.Logf("d after 200/2000 steps: %.4g/%.4g;  final loss %.3g;  d(scale=10)=%.4g ratio=%.2f",
		d200, d1, loss1, d10, d10/d1)

	if d1 < 100*1e-6 {
		t.Errorf("d must grow far beyond its 1e-6 init: got %.4g", d1)
	}
	if d1 > 1.2*d200 {
		t.Errorf("d must stabilize once the scale is found: d200=%.4g grew to %.4g", d200, d1)
	}
	if loss1 > 1e-8 {
		t.Errorf("Prodigy must also solve the quadratic it adapted to: loss %.3g", loss1)
	}
	if ratio := d10 / d1; ratio < 3 || ratio > 30 {
		t.Errorf("d should scale ~10x with a 10x-scaled problem: ratio %.2f", ratio)
	}
}

// §V16 tier-1 fast-path guard (§V22 style): the typed contiguous f64/f32 fast paths must
// produce a BIT-IDENTICAL trajectory (parameters and d) to the generic widening-accessor
// fallback, which is forced via non-contiguous transposed views.
func TestProdigyFastPathParity(t *testing.T) {
	testLRFreeFastPathParity(t, func(ps []*tensor.Tensor) dOptimizer {
		return nn.NewProdigy(ps, nn.WithProdigyWeightDecay(0.01))
	})
}

// §V16 tier-1 collapse check: with DCoef=0 the estimate d̂ is always 0, d can never leave
// d₀, and Prodigy reduces exactly to Adam (without bias correction) at lr = d₀·LR — the
// d factors in m, v and the d·ε floor cancel algebraically. Verified against a
// straight-line Adam recurrence at rtol 1e-12; D() stays at D0 throughout.
func TestProdigyDCoefZeroPinsToAdam(t *testing.T) {
	const (
		lr, b1, b2, eps, d0 = 1.0, 0.9, 0.999, 1e-8, 1e-3
	)
	p0 := []float64{1.0, -2.0, 0.5}
	grads := [][]float64{
		{0.5, 0.5, -0.3},
		{0.2, 0.4, -0.1},
		{-0.4, 0.6, 0.2},
	}

	// straight-line Adam without bias correction at fixed lr = d0*lr:
	// Prodigy's m = d0·m̃, √v = d0·√ṽ, denom floor d0·ε, so
	// dlr·m/(√v+d0·ε) = d0·lr·m̃/(√ṽ+ε) exactly.
	ref := append([]float64(nil), p0...)
	rm := make([]float64, len(p0))
	rv := make([]float64, len(p0))
	for k := range grads {
		for i := range ref {
			g := grads[k][i]
			rm[i] = b1*rm[i] + (1-b1)*g
			rv[i] = b2*rv[i] + (1-b2)*g*g
			ref[i] -= d0 * lr * rm[i] / (math.Sqrt(rv[i]) + eps)
		}
	}

	p := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
	opt := nn.NewProdigy([]*tensor.Tensor{p},
		nn.WithProdigyD0(d0), nn.WithProdigyDCoef(0))
	for k := range grads {
		gs := grads[k]
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor {
			return tensor.FromFloat64(p.Shape(), gs)
		}); err != nil {
			t.Fatal(err)
		}
		if opt.D() != d0 {
			t.Fatalf("DCoef=0 must pin d at d0: got %g", opt.D())
		}
	}
	for i := range p.Numel() {
		got := p.AtF64(i)
		if math.Abs(got-ref[i]) > 1e-12*math.Max(1, math.Abs(ref[i])) {
			t.Errorf("pinned-d update p[%d]: got %.16g want Adam %.16g", i, got, ref[i])
		}
	}
}

// §V16 sanity: all-zero gradients are a NaN-free no-op (no d update, no motion), nil
// gradients are skipped, and a shape mismatch errors like the other optimizers.
// the fast-path guard lives in TestProdigyFastPathParity.
func TestProdigySanity(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	opt := nn.NewProdigy([]*tensor.Tensor{p})
	zero := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 0})
	for range 3 {
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return zero }); err != nil {
			t.Fatal(err)
		}
	}
	if p.AtF64(0) != 1 || p.AtF64(1) != 2 || opt.D() != 1e-6 || math.IsNaN(opt.D()) {
		t.Errorf("zero-grad step must be a no-op: p=[%g %g] d=%g", p.AtF64(0), p.AtF64(1), opt.D())
	}

	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return nil }); err != nil {
		t.Fatal(err)
	}
	if p.AtF64(0) != 1 {
		t.Error("nil-grad param must be skipped")
	}

	bad := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return bad }); err == nil {
		t.Error("shape mismatch must error")
	}
}

// Prodigy needs no learning rate: NewProdigy takes only the parameters, starts its
// internal step-size estimate d at a tiny 1e-6 and grows it to the problem's scale while
// training. Here it minimizes ½(x−3)² from x=0 — after 300 steps x has reached the
// optimum to 3 decimals and D() has grown by orders of magnitude from its 1e-6 seed.
func ExampleProdigy() {
	x := tensor.FromFloat64(tensor.Shape{1}, []float64{0})
	opt := nn.NewProdigy([]*tensor.Tensor{x}) // note: no learning rate
	for range 300 {
		_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
			return tensor.FromFloat64(tensor.Shape{1}, []float64{x.AtF64(0) - 3.0}) // ∇½(x−3)²
		})
	}
	fmt.Printf("x=%.3f d grew: %v\n", x.AtF64(0), opt.D() > 1e-3)
	// Output: x=3.000 d grew: true
}
