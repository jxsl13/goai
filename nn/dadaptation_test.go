package nn_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// quadLoss/quadGrad: the toy convex problem ½·Σ aᵢ(xᵢ−x*ᵢ)² shared by the LR-free tests.
func quadLoss(p *tensor.Tensor, a, star []float64) float64 {
	l := 0.0
	for i := range star {
		d := p.AtF64(i) - star[i]
		l += 0.5 * a[i] * d * d
	}
	return l
}

func quadGrad(p *tensor.Tensor, a, star []float64) nn.GradFn {
	return func(q *tensor.Tensor) *tensor.Tensor {
		if q != p {
			return nil
		}
		g := make([]float64, len(star))
		for i := range star {
			g[i] = a[i] * (q.AtF64(i) - star[i])
		}
		return tensor.FromFloat64(p.Shape(), g)
	}
}

// §V16 tier-1: the DAdaptAdam trajectory (parameters AND the d_k estimate) reproduces the
// authors' reference implementation (facebookresearch/dadaptation dadapt_adam.py; the
// refined form of arXiv:2301.07733's D-Adapted Adam algorithm box) step by step, checked
// against an independent straight-line transcription of the same recurrence at rtol 1e-12.
// Bias correction, weight decay and a finite growth cap are all exercised: per step
//
//	dlr = d·lr·√(1−β₂ᵏ⁺¹)/(1−β₁ᵏ⁺¹);   q = Σ g·s/(√v+ε) with OLD s,v
//	m ← β₁m + (1−β₁)·dlr·g;  v ← β₂v + (1−β₂)g²;  s ← √β₂·s + (1−√β₂)·dlr·g
//	r ← √β₂·r + (1−√β₂)·dlr·q;  d̂ = r/((1−√β₂)‖s‖₁);  d ← max(d, min(d̂, d·growth))
//	x ← (1−λ·dlr)·x − m/(√v+ε)
func TestDAdaptAdamAlgorithmParity(t *testing.T) {
	const (
		lr, b1, b2, eps, wd = 1.0, 0.9, 0.999, 1e-8, 0.02
		d0, growth          = 1e-6, 1.5
	)
	p0v := []float64{1.0, -2.0, 0.5}
	grads := [][]float64{
		{0.5, 0.5, -0.3},
		{0.2, 0.4, -0.1},
		{-0.4, 0.6, 0.2},
		{0.1, 0.5, 0.0},
		{0.3, -0.2, 0.4},
	}

	// independent reference recurrence (a second transcription of the reference code)
	sqb2 := math.Sqrt(b2)
	ref := append([]float64(nil), p0v...)
	rm := make([]float64, len(p0v))
	rv := make([]float64, len(p0v))
	rs := make([]float64, len(p0v))
	rd, rnum := d0, 0.0
	wantP := make([][]float64, len(grads))
	wantD := make([]float64, len(grads))
	for k := range grads {
		g := grads[k]
		bc := math.Sqrt(1-math.Pow(b2, float64(k+1))) / (1 - math.Pow(b1, float64(k+1)))
		dlr := rd * lr * bc
		numAcum := 0.0
		for i := range ref {
			numAcum += dlr * g[i] * rs[i] / (math.Sqrt(rv[i]) + eps) // old s, old v
		}
		for i := range ref {
			rm[i] = b1*rm[i] + (1-b1)*dlr*g[i]
			rv[i] = b2*rv[i] + (1-b2)*g[i]*g[i]
			rs[i] = sqb2*rs[i] + (1-sqb2)*dlr*g[i]
		}
		skl1 := 0.0
		for i := range ref {
			skl1 += math.Abs(rs[i])
		}
		rnum = sqb2*rnum + (1-sqb2)*numAcum
		dHat := rnum / ((1 - sqb2) * skl1)
		rd = math.Max(rd, math.Min(dHat, rd*growth))
		for i := range ref {
			ref[i] = ref[i]*(1-wd*dlr) - rm[i]/(math.Sqrt(rv[i])+eps)
		}
		wantP[k] = append([]float64(nil), ref...)
		wantD[k] = rd
	}

	// optimizer under test
	p := tensor.FromFloat64(tensor.Shape{len(p0v)}, append([]float64(nil), p0v...))
	opt := nn.NewDAdaptAdam([]*tensor.Tensor{p},
		nn.WithDAdaptLR(lr), nn.WithDAdaptBetas(b1, b2), nn.WithDAdaptEps(eps),
		nn.WithDAdaptWeightDecay(wd), nn.WithDAdaptD0(d0),
		nn.WithDAdaptGrowthRate(growth), nn.WithDAdaptBiasCorrection(true))
	for k := range grads {
		gs := grads[k]
		if err := opt.Step(func(q *tensor.Tensor) *tensor.Tensor {
			if q != p {
				return nil
			}
			return tensor.FromFloat64(p.Shape(), gs)
		}); err != nil {
			t.Fatal(err)
		}
		if got := opt.D(); math.Abs(got-wantD[k]) > 1e-12*math.Max(1, math.Abs(wantD[k])) {
			t.Errorf("step %d d: got %.16g want %.16g", k, got, wantD[k])
		}
		for i := range p.Numel() {
			got := p.AtF64(i)
			if math.Abs(got-wantP[k][i]) > 1e-12*math.Max(1, math.Abs(wantP[k][i])) {
				t.Errorf("step %d p[%d]: got %.16g want %.16g", k, i, got, wantP[k][i])
			}
		}
	}
}

// regressionProblem is the shared LR-free value-test fixture: least-squares regression
// whose true weights are TINY (‖w*‖≈0.06 — three orders of magnitude below the nominal
// lr=1) with label noise, trained on deterministic pseudo-random minibatches. A fixed-lr
// Adam at the nominal lr=1 wanders at O(1) scale and stalls far above the noise floor; a
// learning-rate-free optimizer must discover the ~0.03 scale by itself and reach a floor
// comparable to the best hand-tuned Adam. Everything is seeded — fully deterministic.
type regressionProblem struct {
	dim    int
	z      [][]float64
	y      []float64
	wstar  []float64
	minima float64 // loss at w* (≈ the Bayes floor on this sample)
}

func newRegressionProblem() *regressionProblem {
	rng := rand.New(rand.NewSource(42))
	const dim, n, sigma = 8, 256, 0.5
	pr := &regressionProblem{dim: dim}
	pr.wstar = make([]float64, dim)
	for i := range pr.wstar {
		pr.wstar[i] = rng.NormFloat64() * 0.02
	}
	pr.z = make([][]float64, n)
	pr.y = make([]float64, n)
	for j := range pr.z {
		pr.z[j] = make([]float64, dim)
		for i := range pr.z[j] {
			pr.z[j][i] = rng.NormFloat64()
		}
		for i := range pr.wstar {
			pr.y[j] += pr.wstar[i] * pr.z[j][i]
		}
		pr.y[j] += sigma * rng.NormFloat64()
	}
	w := tensor.FromFloat64(tensor.Shape{dim}, pr.wstar)
	pr.minima = pr.loss(w)
	return pr
}

// loss is the full-dataset objective 1/n·Σ ½(w·zⱼ−yⱼ)².
func (pr *regressionProblem) loss(p *tensor.Tensor) float64 {
	l := 0.0
	for j := range pr.z {
		r := -pr.y[j]
		for i := range pr.dim {
			r += p.AtF64(i) * pr.z[j][i]
		}
		l += 0.5 * r * r
	}
	return l / float64(len(pr.z))
}

// grad returns a seeded minibatch GradFn (batch 8) for p.
func (pr *regressionProblem) grad(p *tensor.Tensor, seed int64) nn.GradFn {
	r2 := rand.New(rand.NewSource(seed))
	const batch = 8
	return func(q *tensor.Tensor) *tensor.Tensor {
		if q != p {
			return nil
		}
		g := make([]float64, pr.dim)
		for range batch {
			j := r2.Intn(len(pr.z))
			res := -pr.y[j]
			for i := range pr.dim {
				res += q.AtF64(i) * pr.z[j][i]
			}
			for i := range pr.dim {
				g[i] += res * pr.z[j][i] / batch
			}
		}
		return tensor.FromFloat64(q.Shape(), g)
	}
}

// run trains opt for steps on the problem and returns the final excess loss over the floor.
func (pr *regressionProblem) run(t *testing.T, opt nn.Optimizer, p *tensor.Tensor, steps int) float64 {
	t.Helper()
	g := pr.grad(p, 7)
	for range steps {
		if err := opt.Step(g); err != nil {
			t.Fatal(err)
		}
	}
	return pr.loss(p) - pr.minima
}

// §V16 tier-1, THE value test: DAdaptAdam with its DEFAULTS (nominal LR=1, no tuning at
// all) reaches the noise floor of a problem whose scale is ~0.03, comparable to the best
// hand-tuned Adam, while a fixed-lr Adam at the same nominal lr=1 stalls two orders of
// magnitude higher. This is the learning-rate-free property the method exists for.
func TestDAdaptAdamLRFreeConvergence(t *testing.T) {
	pr := newRegressionProblem()
	const steps = 2000

	pD := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	optD := nn.NewDAdaptAdam([]*tensor.Tensor{pD}) // NO learning rate anywhere
	excessD := pr.run(t, optD, pD, steps)

	pA1 := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	excessA1 := pr.run(t, nn.NewAdam([]*tensor.Tensor{pA1}, 1.0), pA1, steps) // same nominal lr

	pAT := tensor.FromFloat64(tensor.Shape{pr.dim}, make([]float64, pr.dim))
	excessAT := pr.run(t, nn.NewAdam([]*tensor.Tensor{pAT}, 0.01), pAT, steps) // best hand-tuned lr

	t.Logf("excess loss after %d steps: DAdaptAdam(default)=%.4g  Adam(lr=1)=%.4g  Adam(tuned lr=0.01)=%.4g  d=%.4g",
		steps, excessD, excessA1, excessAT, optD.D())

	if excessD > 0.01 {
		t.Errorf("DAdaptAdam with defaults must reach the noise floor untuned: excess %.4g > 0.01", excessD)
	}
	if excessA1 < 0.05 {
		t.Errorf("premise broken: fixed-lr Adam at lr=1 should stall on this problem, excess %.4g", excessA1)
	}
	if excessD > excessA1/10 {
		t.Errorf("DAdaptAdam (%.4g) must beat untuned Adam@1 (%.4g) by >10x", excessD, excessA1)
	}
	if excessD > 10*excessAT {
		t.Errorf("DAdaptAdam (%.4g) not comparable to tuned Adam (%.4g)", excessD, excessAT)
	}
	// d must have discovered the problem's tiny scale: far above its 1e-6 init, far below
	// the naive O(1) guess a fixed lr=1 embodies.
	if d := optD.D(); d < 1e-3 || d > 0.5 {
		t.Errorf("d should land near the problem scale ~0.03: got %.4g", d)
	}
}

// §V16 tier-1: the d_k estimate grows from its tiny init d₀=1e-6 toward the problem scale
// ‖x₀−x*‖ and stabilizes (monotone, bounded); and on a 10×-scaled copy of the problem the
// final d is ~10× larger — d tracks the initial distance it is defined to estimate.
func TestDAdaptAdamDTracksProblemScale(t *testing.T) {
	a := []float64{2.0, 0.5, 1.0, 4.0}
	base := []float64{3.0, -5.0, 2.0, -1.0}

	finalD := func(scale float64) float64 {
		star := make([]float64, len(base))
		for i, v := range base {
			star[i] = v * scale
		}
		p := tensor.FromFloat64(tensor.Shape{len(star)}, make([]float64, len(star)))
		opt := nn.NewDAdaptAdam([]*tensor.Tensor{p})
		g := quadGrad(p, a, star)
		prev := opt.D()
		for range 3000 {
			if err := opt.Step(g); err != nil {
				t.Fatal(err)
			}
			if opt.D() < prev {
				t.Fatalf("d must be monotone nondecreasing: %g -> %g", prev, opt.D())
			}
			prev = opt.D()
		}
		return opt.D()
	}

	d1 := finalD(1)
	d10 := finalD(10)
	dist := 0.0
	for _, v := range base {
		dist += v * v
	}
	dist = math.Sqrt(dist)
	t.Logf("d(scale=1)=%.4g d(scale=10)=%.4g ratio=%.2f  ‖x0−x*‖=%.3g", d1, d10, d10/d1, dist)

	if d1 < 100*1e-6 {
		t.Errorf("d must grow far beyond its 1e-6 init: got %.3g", d1)
	}
	if ratio := d10 / d1; ratio < 3 || ratio > 30 {
		t.Errorf("d should scale ~10x with a 10x-scaled problem: ratio %.2f", ratio)
	}
}

// §V16 tier-1 collapse check: with the growth cap pinned to 1.0, d can never leave d₀ and
// DAdaptAdam reduces exactly to Adam (without bias correction) at lr = d₀·LR — verified
// against a straight-line Adam recurrence at rtol 1e-12, and D() stays at D0 throughout.
func TestDAdaptAdamGrowthOnePinsToAdam(t *testing.T) {
	const (
		lr, b1, b2, eps, d0 = 1.0, 0.9, 0.999, 1e-8, 1e-3
	)
	p0v := []float64{1.0, -2.0, 0.5}
	grads := [][]float64{
		{0.5, 0.5, -0.3},
		{0.2, 0.4, -0.1},
		{-0.4, 0.6, 0.2},
	}

	// straight-line Adam without bias correction at fixed lr = d0*lr
	ref := append([]float64(nil), p0v...)
	rm := make([]float64, len(p0v))
	rv := make([]float64, len(p0v))
	for k := range grads {
		for i := range ref {
			g := grads[k][i]
			rm[i] = b1*rm[i] + (1-b1)*d0*lr*g
			rv[i] = b2*rv[i] + (1-b2)*g*g
			ref[i] -= rm[i] / (math.Sqrt(rv[i]) + eps)
		}
	}

	p := tensor.FromFloat64(tensor.Shape{len(p0v)}, append([]float64(nil), p0v...))
	opt := nn.NewDAdaptAdam([]*tensor.Tensor{p},
		nn.WithDAdaptD0(d0), nn.WithDAdaptGrowthRate(1.0))
	for k := range grads {
		gs := grads[k]
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor {
			return tensor.FromFloat64(p.Shape(), gs)
		}); err != nil {
			t.Fatal(err)
		}
		if opt.D() != d0 {
			t.Fatalf("growth=1 must pin d at d0: got %g", opt.D())
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
// gradients are skipped, and a shape mismatch errors like the other optimizers. Also
// checks the f32 fast path agrees with the f64 path at f32 precision.
func TestDAdaptAdamSanity(t *testing.T) {
	// zero grads: params and d unchanged, no NaN
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	opt := nn.NewDAdaptAdam([]*tensor.Tensor{p})
	zero := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 0})
	for range 3 {
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return zero }); err != nil {
			t.Fatal(err)
		}
	}
	if p.AtF64(0) != 1 || p.AtF64(1) != 2 || opt.D() != 1e-6 || math.IsNaN(opt.D()) {
		t.Errorf("zero-grad step must be a no-op: p=[%g %g] d=%g", p.AtF64(0), p.AtF64(1), opt.D())
	}

	// nil grad: skipped
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return nil }); err != nil {
		t.Fatal(err)
	}
	if p.AtF64(0) != 1 {
		t.Error("nil-grad param must be skipped")
	}

	// shape mismatch: error
	bad := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return bad }); err == nil {
		t.Error("shape mismatch must error")
	}

}

// dOptimizer is an Optimizer exposing its distance estimate — what both LR-free
// optimizers implement; the fast-path harness compares D() across paths too.
type dOptimizer interface {
	nn.Optimizer
	D() float64
}

// testLRFreeFastPathParity is the shared §V22-style fast-path guard for the LR-free
// optimizers: the typed contiguous f64/f32 fast paths must produce a trajectory
// (parameters and the d estimate) that MATCHES the generic widening-accessor fallback,
// which is forced via non-contiguous transposed views (mustTransposedView).
//
// Tolerance (not bit-exact): the typed paths of DAdaptAdam/Prodigy fan their d-estimate's
// numerator and ‖s‖₁ denominator across cores via parSumF64x2, which REASSOCIATES those
// reductions (~1 ULP, exactly like parSumSq in LAMB/ClipGradNorm), while the generic
// fallback sums serially. The estimate d — and the parameters it scales — therefore agree
// to a reassociation tolerance, not bit-for-bit; a real fast-path bug (wrong index/formula)
// still diverges far beyond it. The per-index moment writes remain bit-exact and are pinned
// separately by the single-step toggle tests in optim_parallel_internal_test.go.
func testLRFreeFastPathParity(t *testing.T, mk func(ps []*tensor.Tensor) dOptimizer) {
	t.Helper()
	const steps = 6
	// relTol compounds gently over the steps as d's tiny reassociation difference propagates.
	close := func(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Abs(a)+1e-12 }
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
			optFast := mk(fast)
			optSlow := mk(slow)
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
				if !close(optFast.D(), optSlow.D()) {
					t.Fatalf("step %d: fast-path d %.17g != generic-path d %.17g (beyond reassoc tol)", s, optFast.D(), optSlow.D())
				}
				for pi, shape := range shapes {
					for i := range shape.Numel() {
						idx := tensor.Unravel(i, shape)
						fv, sv := fast[pi].AtF64(idx...), slow[pi].AtF64(idx...)
						if !close(fv, sv) {
							t.Fatalf("step %d param %d idx %v: fast path %.17g != generic path %.17g (beyond reassoc tol)", s, pi, idx, fv, sv)
						}
					}
				}
			}
		})
	}
}

// §V16 tier-1 fast-path guard (§V22 style): typed f64/f32 fast paths vs the generic
// fallback, bit-identical including the d trajectory.
func TestDAdaptAdamFastPathParity(t *testing.T) {
	testLRFreeFastPathParity(t, func(ps []*tensor.Tensor) dOptimizer {
		return nn.NewDAdaptAdam(ps, nn.WithDAdaptWeightDecay(0.01))
	})
}

// DAdaptAdam needs no learning rate: NewDAdaptAdam takes only the parameters, starts its
// internal step-size estimate d at a tiny 1e-6 and grows it to the problem's scale while
// training. Here it minimizes ½(x−3)² from x=0 — after 300 steps x has reached the
// optimum to 3 decimals and D() has grown by orders of magnitude from its 1e-6 seed.
func ExampleDAdaptAdam() {
	x := tensor.FromFloat64(tensor.Shape{1}, []float64{0})
	opt := nn.NewDAdaptAdam([]*tensor.Tensor{x}) // note: no learning rate
	for range 300 {
		_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
			return tensor.FromFloat64(tensor.Shape{1}, []float64{x.AtF64(0) - 3.0}) // ∇½(x−3)²
		})
	}
	fmt.Printf("x=%.3f d grew: %v\n", x.AtF64(0), opt.D() > 1e-3)
	// Output: x=3.000 d grew: true
}
