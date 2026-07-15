package nn_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 COLLAPSE anchor (§V16.3): with PreconditionerLR 0 the factors are frozen at
// the identity (all ones), so Ĝ=G and PSGDKron is BIT-IDENTICAL to plain SGD over a
// multi-step run (2-D + 1-D params). If this drifts, the preconditioned-step arithmetic
// diverged from SGD's — the whole "opt-in preconditioner on the same step" claim breaks.
func TestPSGDKronIdentityCollapsesToSGD(t *testing.T) {
	const lr, steps = 0.05, 25
	shapes := []tensor.Shape{{3, 4}, {5}}
	pval := func(i int) float64 { return math.Sin(float64(i)*1.3) - 0.2 }
	gval := func(i, s int) float64 { return math.Cos(float64(i)*0.7+float64(s)*0.31) * 0.4 }

	mk := func() []*tensor.Tensor {
		ps := make([]*tensor.Tensor, len(shapes))
		for pi, s := range shapes {
			ps[pi] = tensor.New(tensor.F64, s)
			for i := range s.Numel() {
				ps[pi].SetF64(pval(i+13*pi), tensor.Unravel(i, s)...)
			}
		}
		return ps
	}
	pSGD, pKron := mk(), mk()
	optSGD := nn.NewSGD(pSGD, lr, 0)
	optKron := nn.NewPSGDKron(pKron, lr, nn.WithPSGDKronPreconditionerLR(0)) // frozen ⇒ SGD

	gradFor := func(ps []*tensor.Tensor, s int) nn.GradFn {
		return func(q *tensor.Tensor) *tensor.Tensor {
			for pi, p := range ps {
				if q != p {
					continue
				}
				g := tensor.New(tensor.F64, p.Shape())
				for i := range p.Numel() {
					g.SetF64(gval(i+13*pi, s), tensor.Unravel(i, p.Shape())...)
				}
				return g
			}
			return nil
		}
	}
	for s := range steps {
		if err := optSGD.Step(gradFor(pSGD, s)); err != nil {
			t.Fatal(err)
		}
		if err := optKron.Step(gradFor(pKron, s)); err != nil {
			t.Fatal(err)
		}
	}
	for pi := range shapes {
		for i := range shapes[pi].Numel() {
			idx := tensor.Unravel(i, shapes[pi])
			a, b := pSGD[pi].AtF64(idx...), pKron[pi].AtF64(idx...)
			if a != b { // BIT-identical, not just close
				t.Errorf("param %d idx %v: PSGDKron(frozen) %.17g != SGD %.17g", pi, idx, b, a)
			}
		}
	}
}

// refKron is an INDEPENDENT transcription of the diagonal PSGD-Kron algorithm used to pin
// the implementation (§V16.1). It mirrors NewPSGDKron/Step/updateFactors from scratch on
// plain []float64 state — same probe RNG (seed, standard-normal, row-major, per-param),
// same normalized multiplicative factor update, same preconditioned step.
type refKron struct {
	dL, dR  [][]float64
	cols    []int
	numel   []int
	rng     *rand.Rand
	lr, plr float64
	eps     float64
}

func newRefKron(shapes []tensor.Shape, lr, plr float64, seed int64) *refKron {
	r := &refKron{lr: lr, plr: plr, eps: 1e-30, rng: rand.New(rand.NewSource(seed))}
	for _, s := range shapes {
		n := s.Numel()
		r.numel = append(r.numel, n)
		if len(s) >= 2 && s[0] > 0 {
			rows := s[0]
			cols := n / rows
			r.cols = append(r.cols, cols)
			r.dL = append(r.dL, refOnes(rows))
			r.dR = append(r.dR, refOnes(cols))
		} else {
			r.cols = append(r.cols, 0)
			r.dL = append(r.dL, refOnes(n))
			r.dR = append(r.dR, nil)
		}
	}
	return r
}

func refOnes(n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

func (r *refKron) step(params [][]float64, grads [][]float64) {
	for pi := range params {
		g := grads[pi]
		n := r.numel[pi]
		allZero := true
		for _, v := range g {
			if v != 0 {
				allZero = false
				break
			}
		}
		dL, dR, cols := r.dL[pi], r.dR[pi], r.cols[pi]
		if r.plr > 0 && !allZero {
			v := make([]float64, n)
			for k := range v {
				v[k] = r.rng.NormFloat64()
			}
			if cols > 0 {
				rows := n / cols
				t1L, t2L := make([]float64, rows), make([]float64, rows)
				t1R, t2R := make([]float64, cols), make([]float64, cols)
				for k := 0; k < n; k++ {
					i, j := k/cols, k%cols
					f := dL[i] * dR[j]
					a, b := f*g[k], v[k]/f
					t1L[i] += a * a
					t2L[i] += b * b
					t1R[j] += a * a
					t2R[j] += b * b
				}
				normL := r.eps
				for i := range t1L {
					if s := t1L[i] + t2L[i]; s > normL {
						normL = s
					}
				}
				muL := r.plr / normL
				for i := range dL {
					dL[i] *= 1 - muL*(t1L[i]-t2L[i])
				}
				normR := r.eps
				for j := range t1R {
					if s := t1R[j] + t2R[j]; s > normR {
						normR = s
					}
				}
				muR := r.plr / normR
				for j := range dR {
					dR[j] *= 1 - muR*(t1R[j]-t2R[j])
				}
			} else {
				t1, t2 := make([]float64, n), make([]float64, n)
				norm := r.eps
				for k := 0; k < n; k++ {
					f := dL[k]
					a, b := f*g[k], v[k]/f
					t1[k], t2[k] = a*a, b*b
					if s := t1[k] + t2[k]; s > norm {
						norm = s
					}
				}
				mu := r.plr / norm
				for k := 0; k < n; k++ {
					dL[k] *= 1 - mu*(t1[k]-t2[k])
				}
			}
		}
		for k := 0; k < n; k++ {
			f := 1.0
			if cols > 0 {
				f = dL[k/cols] * dR[k%cols]
			} else {
				f = dL[k]
			}
			params[pi][k] -= r.lr * f * g[k]
		}
	}
}

// §V16 tier-1 ALGORITHM PARITY (§V16.1): the optimizer must match the independent refKron
// transcription to 1e-10 on BOTH the parameters and the learned factors, over a multi-step
// run with a 2-D and a 1-D parameter and a live preconditioner (PreconditionerLR>0). The
// gradients are a fixed function of step (independent of W) so any divergence is a pure
// algorithm mismatch, not feedback drift.
func TestPSGDKronAlgorithmParity(t *testing.T) {
	const lr, plr, steps = 0.03, 0.1, 30
	const seed int64 = 1
	shapes := []tensor.Shape{{4, 3}, {6}}
	gval := func(pi, i, s int) float64 { return math.Cos(float64(i)*0.5+float64(s)*0.2+float64(pi)) * 0.3 }

	// Optimizer over tensors.
	ps := make([]*tensor.Tensor, len(shapes))
	for pi, s := range shapes {
		ps[pi] = tensor.New(tensor.F64, s)
	}
	opt := nn.NewPSGDKron(ps, lr, nn.WithPSGDKronPreconditionerLR(plr), nn.WithPSGDKronSeed(seed))

	// Independent reference over []float64.
	ref := newRefKron(shapes, lr, plr, seed)
	rparams := make([][]float64, len(shapes))
	for pi, s := range shapes {
		rparams[pi] = make([]float64, s.Numel())
	}

	for s := range steps {
		grad := func(q *tensor.Tensor) *tensor.Tensor {
			for pi, p := range ps {
				if q != p {
					continue
				}
				g := tensor.New(tensor.F64, p.Shape())
				for i := range p.Numel() {
					g.SetF64(gval(pi, i, s), tensor.Unravel(i, p.Shape())...)
				}
				return g
			}
			return nil
		}
		if err := opt.Step(grad); err != nil {
			t.Fatal(err)
		}
		rgrads := make([][]float64, len(shapes))
		for pi, sh := range shapes {
			rgrads[pi] = make([]float64, sh.Numel())
			for i := range sh.Numel() {
				rgrads[pi][i] = gval(pi, i, s)
			}
		}
		ref.step(rparams, rgrads)
	}

	const tol = 1e-10
	for pi, sh := range shapes {
		for i := range sh.Numel() {
			got := ps[pi].AtF64(tensor.Unravel(i, sh)...)
			want := rparams[pi][i]
			if math.Abs(got-want) > tol*math.Max(1, math.Abs(want)) {
				t.Errorf("param %d[%d]: opt %.16g != ref %.16g", pi, i, got, want)
			}
		}
		left, right := opt.Factors(pi)
		for i, want := range ref.dL[pi] {
			if math.Abs(left[i]-want) > tol*math.Max(1, math.Abs(want)) {
				t.Errorf("param %d dL[%d]: opt %.16g != ref %.16g", pi, i, left[i], want)
			}
		}
		for j, want := range ref.dR[pi] {
			if math.Abs(right[j]-want) > tol*math.Max(1, math.Abs(want)) {
				t.Errorf("param %d dR[%d]: opt %.16g != ref %.16g", pi, j, right[j], want)
			}
		}
	}
}

// §V16 tier-1 PRECONDITIONER VALUE, part A — the fixed point (§V16.2). Fed a CONSTANT
// gradient, the diagonal factor must converge toward the inverse square root of the
// per-coordinate gradient magnitude (the whitening fixed point d[k]→|g[k]|^{-1/2}). Uses a
// 1-D parameter so the single per-element factor is the preconditioner directly (no
// dL/dR product degeneracy). The factor is time-averaged over the tail to beat the probe
// noise, then compared to |g|^{-1/2}.
//
// NOTE on the tolerance: the normalized multiplicative update divides by the LARGEST
// row/column energy, so the single STIFFEST coordinate (the one that is almost always the
// argmax) carries a small structural downward bias — its factor settles a bit below the
// ideal |g|^{-1/2}. This is inherent to PSGD's global step normalization (real PSGD uses a
// matrix norm and rarely has one coordinate dominate so extremely). The test therefore
// asserts a strict monotone trend for every coordinate plus a tight (<8%) fit for all but
// the stiffest, and a looser (<25%) fit for the stiffest.
func TestPSGDKronWhiteningFixedPoint(t *testing.T) {
	g := []float64{0.04, 0.1, 0.25, 1.0, 4.0} // ascending magnitude; last is the stiffest
	p := tensor.New(tensor.F64, tensor.Shape{len(g)})
	// A tiny LR keeps the parameter (hence the constant gradient premise) essentially
	// fixed while the preconditioner adapts.
	opt := nn.NewPSGDKron([]*tensor.Tensor{p}, 1e-9, nn.WithPSGDKronPreconditionerLR(0.05))
	grad := func(*tensor.Tensor) *tensor.Tensor { return tensor.FromFloat64(p.Shape(), g) }

	const warm, avg = 4000, 4000
	for range warm {
		if err := opt.Step(grad); err != nil {
			t.Fatal(err)
		}
	}
	acc := make([]float64, len(g))
	for range avg {
		if err := opt.Step(grad); err != nil {
			t.Fatal(err)
		}
		left, _ := opt.Factors(0)
		for i, v := range left {
			acc[i] += v
		}
	}
	prev := math.Inf(1)
	for i, gv := range g {
		want := 1 / math.Sqrt(math.Abs(gv)) // |g|^{-1/2}
		got := acc[i] / avg
		rel := math.Abs(got-want) / want
		t.Logf("g=%.3g  factor≈%.4g  target |g|^-1/2=%.4g  rel=%.1f%%", gv, got, want, 100*rel)
		if got >= prev { // strictly decreasing in |g| (the inverse-sqrt trend)
			t.Errorf("factor for g=%.3g (%.4g) must be < the softer coordinate's (%.4g)", gv, got, prev)
		}
		prev = got
		tol := 0.08
		if i == len(g)-1 { // stiffest coordinate carries the normalization bias
			tol = 0.25
		}
		if rel > tol {
			t.Errorf("factor for g=%.3g: got %.4g want ~%.4g (inverse-sqrt), rel %.1f%% > %.0f%%",
				gv, got, want, 100*rel, 100*tol)
		}
	}
}

// §V16 tier-1 PRECONDITIONER VALUE, part B — ill-conditioned convergence (§V16.2). On a
// diagonal quadratic with condition number ~1000, PSGD-Kron (with momentum) reaches the
// loss threshold in FAR fewer steps than well-tuned SGD, and fewer than well-tuned Adam.
// SGD is throttled by the stiffest direction (lr < 2/h_max) so its softest direction
// crawls — it needs ~1000× more steps in that direction; the learned per-coordinate
// preconditioner balances the directions so a single lr makes even progress. Adam is given
// its BEST learning rate over a small sweep for a fair comparison. "Reached" means the loss
// stays below the threshold for 50 consecutive steps (no transient dips counted).
func TestPSGDKronIllConditioned(t *testing.T) {
	const dim = 20
	h := make([]float64, dim) // curvatures log-spaced 1 … 1000 (cond ≈ 1000)
	for i := range h {
		h[i] = math.Pow(1000, float64(i)/float64(dim-1))
	}
	loss := func(p *tensor.Tensor) float64 {
		l := 0.0
		for i := range dim {
			x := p.AtF64(i)
			l += 0.5 * h[i] * x * x
		}
		return l
	}
	gradOf := func(p *tensor.Tensor) nn.GradFn {
		return func(q *tensor.Tensor) *tensor.Tensor {
			if q != p {
				return nil
			}
			g := make([]float64, dim)
			for i := range dim {
				g[i] = h[i] * q.AtF64(i) // ∇½Σhᵢxᵢ²
			}
			return tensor.FromFloat64(q.Shape(), g)
		}
	}
	start := make([]float64, dim)
	for i := range start {
		start[i] = 1
	}
	fresh := func() *tensor.Tensor { return tensor.FromFloat64(tensor.Shape{dim}, start) }

	const threshold, maxSteps, hold = 5e-3, 40000, 50
	stepsTo := func(mk func(p *tensor.Tensor) nn.Optimizer) int {
		p := fresh()
		opt := mk(p)
		below := 0
		for s := 1; s <= maxSteps; s++ {
			if err := opt.Step(gradOf(p)); err != nil {
				t.Fatal(err)
			}
			if loss(p) < threshold {
				if below++; below >= hold {
					return s
				}
			} else {
				below = 0
			}
		}
		return maxSteps + 1 // did not reach the threshold
	}

	sSGD := stepsTo(func(p *tensor.Tensor) nn.Optimizer {
		return nn.NewSGD([]*tensor.Tensor{p}, 1.9/1000.0, 0.0) // lr just under the 2/h_max limit
	})
	sAdam := maxSteps + 1
	for _, lr := range []float64{0.05, 0.02, 0.01} { // give Adam its best shot
		if n := stepsTo(func(p *tensor.Tensor) nn.Optimizer {
			return nn.NewAdam([]*tensor.Tensor{p}, lr)
		}); n < sAdam {
			sAdam = n
		}
	}
	sKron := stepsTo(func(p *tensor.Tensor) nn.Optimizer {
		return nn.NewPSGDKron([]*tensor.Tensor{p}, 0.003,
			nn.WithPSGDKronPreconditionerLR(0.3), nn.WithPSGDKronMomentum(0.9))
	})

	t.Logf("steps to sustained loss<%.0e (cond~1000): SGD=%d  Adam(best)=%d  PSGDKron=%d",
		threshold, sSGD, sAdam, sKron)
	if sKron > maxSteps {
		t.Fatalf("PSGDKron did not reach the threshold in %d steps", maxSteps)
	}
	if sKron*3 >= sSGD { // huge margin expected (SGD is conditioning-limited)
		t.Errorf("PSGDKron (%d) must decisively beat SGD (%d) on an ill-conditioned problem", sKron, sSGD)
	}
	if sKron >= sAdam {
		t.Errorf("PSGDKron (%d) must beat well-tuned Adam (%d)", sKron, sAdam)
	}
}

// §V16 tier-1 FAST-PATH guard (§V22 style): the typed contiguous f64/f32 read/write paths
// must produce a BIT-IDENTICAL trajectory to the generic widening-accessor fallback,
// forced via non-contiguous transposed views. Both optimizers share the seed, so the probe
// streams match and only the tensor access path differs.
func TestPSGDKronFastPathParity(t *testing.T) {
	const steps = 8
	pval := func(i int) float64 { return math.Sin(float64(i)*0.7) + 0.1 }
	gval := func(i, s int) float64 { return math.Cos(float64(i)*0.3+float64(s)) * 0.05 }

	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		t.Run(fmt.Sprintf("%v", dt), func(t *testing.T) {
			shapes := []tensor.Shape{{3, 5}, {2, 3, 4}, {7}}
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
			mk := func(ps []*tensor.Tensor) nn.Optimizer {
				return nn.NewPSGDKron(ps, 0.05, nn.WithPSGDKronPreconditionerLR(0.1),
					nn.WithPSGDKronSeed(7))
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
							t.Fatalf("step %d param %d idx %v: fast %.17g != generic %.17g", s, pi, idx, fv, sv)
						}
					}
				}
			}
		})
	}
}

// §V16 sanity (§V16.4): nil grad is skipped, an all-zero grad is a NaN-free no-op (no
// motion, no probe drawn), a shape mismatch errors, and an f32 parameter trains without
// producing NaNs.
func TestPSGDKronSanity(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	opt := nn.NewPSGDKron([]*tensor.Tensor{p}, 0.1)

	zero := tensor.New(tensor.F64, p.Shape())
	for range 3 {
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return zero }); err != nil {
			t.Fatal(err)
		}
	}
	for i := range 4 {
		if v := p.AtF64(tensor.Unravel(i, p.Shape())...); math.IsNaN(v) || v != float64(i+1) {
			t.Errorf("all-zero grad must be a NaN-free no-op: p[%d]=%g", i, v)
		}
	}

	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return nil }); err != nil {
		t.Fatal(err)
	}
	if p.AtF64(0, 0) != 1 {
		t.Error("nil-grad param must be skipped")
	}

	bad := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return bad }); err == nil {
		t.Error("shape mismatch must error")
	}

	// f32 parameter: a few live-preconditioner steps must stay finite.
	pf := tensor.New(tensor.F32, tensor.Shape{3, 2})
	for i := range 6 {
		pf.SetF64(0.5-0.1*float64(i), tensor.Unravel(i, pf.Shape())...)
	}
	optf := nn.NewPSGDKron([]*tensor.Tensor{pf}, 0.05, nn.WithPSGDKronMomentum(0.9))
	for range 10 {
		if err := optf.Step(func(*tensor.Tensor) *tensor.Tensor {
			g := tensor.New(tensor.F32, pf.Shape())
			for i := range 6 {
				g.SetF64(0.2, tensor.Unravel(i, pf.Shape())...)
			}
			return g
		}); err != nil {
			t.Fatal(err)
		}
	}
	for i := range 6 {
		if v := pf.AtF64(tensor.Unravel(i, pf.Shape())...); math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("f32 step produced non-finite p[%d]=%g", i, v)
		}
	}
}

// ExamplePSGDKron drives a [4,3] weight to a target with the diagonal Kronecker
// preconditioner learned online, then inspects the learned per-row/per-column factors.
func ExamplePSGDKron() {
	target := []float64{1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12}
	w := tensor.New(tensor.F64, tensor.Shape{4, 3})
	opt := nn.NewPSGDKron([]*tensor.Tensor{w}, 0.1)
	for range 2000 {
		_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
			g := tensor.New(tensor.F64, w.Shape())
			for i := range w.Numel() {
				idx := tensor.Unravel(i, w.Shape())
				g.SetF64(w.AtF64(idx...)-target[i], idx...) // ∇½‖w−target‖²
			}
			return g
		})
	}
	maxErr := 0.0
	for i := range w.Numel() {
		idx := tensor.Unravel(i, w.Shape())
		maxErr = math.Max(maxErr, math.Abs(w.AtF64(idx...)-target[i]))
	}
	left, right := opt.Factors(0)
	fmt.Printf("converged: %v\n", maxErr < 1e-2)
	fmt.Printf("preconditioner: %d row factors, %d col factors\n", len(left), len(right))
	// Output:
	// converged: true
	// preconditioner: 4 row factors, 3 col factors
}
