package nn

import (
	"fmt"
	"math/rand"

	"github.com/jxsl13/goai/tensor"
)

// PSGDKron is the diagonal Kronecker-factored variant of Preconditioned SGD (PSGD;
// Xi-Lin Li 2015, "Preconditioned Stochastic Gradient Descent", arXiv:1512.04202, and
// the modern "black-box Lie-group" whitening form of arXiv:2211.04422 that the popular
// evanatyourservice/kron implementation follows). PSGD learns a preconditioner ONLINE —
// jointly with the parameters — and applies it to the raw gradient before the step, so
// the optimizer adapts to the local geometry without ever forming or inverting a Hessian.
//
// In plain terms: plain SGD steps blindly downhill and crawls through long, thin valleys
// (ill-conditioned problems); PSGD watches the gradients go by and continuously fits a
// cheap per-coordinate rescaling that stretches the cramped directions and shrinks the
// steep ones, so the same step size makes balanced progress in every direction. It learns
// this rescaling as it trains — there is no separate curvature pass.
//
// KRONECKER FACTORS (the memory win). For a matrix parameter W[m,n] a full preconditioner
// would need m·n numbers per entry; PSGD-Kron factors it as an OUTER PRODUCT of two
// diagonal vectors — a per-ROW scale dL[m] and a per-COLUMN scale dR[n] — so the
// preconditioner for entry (i,j) is just dL[i]·dR[j] and the whole thing costs O(m+n)
// state instead of O(m·n). The preconditioned gradient and the step are:
//
//	Ĝ[i,j] = dL[i] · dR[j] · G[i,j]        (apply the factored preconditioner)
//	W      ← W − lr · Ĝ                     (preconditioned SGD step)
//
// For a 1-D parameter (bias, norm gain) there is a single per-element diagonal
// preconditioner d[k] and Ĝ[k] = d[k]·G[k]. A parameter with ≥3 dims is treated as the
// matrix [shape[0], (rest)] — one factor over the leading axis, one over the flattened
// trailing axes.
//
// THE PRECONDITIONER CRITERION (how the factors are learned). Each step draws a random
// probe V (standard normal, same shape as G) and nudges the factors to satisfy the PSGD
// whitening criterion — balancing the preconditioned gradient against the inverse-
// preconditioned probe:
//
//	minimize over the factors   E_V‖ (Q⊙G)(Q⊙G)ᵀ − (V⊘Q)(V⊘Q)ᵀ ‖   where Q[i,j]=dL[i]·dR[j]
//
// (⊙ elementwise multiply, ⊘ elementwise divide). Setting the per-coordinate gradient of
// this criterion to zero gives the fixed point Q[i,j] = |G[i,j]|^{−1/2}: the factors
// converge to the INVERSE SQUARE ROOT of the per-coordinate gradient magnitude (for a
// quadratic, the inverse-sqrt of the local curvature scale). The preconditioned step is
// then sign(G)·√|G| — gentler than Adam's sign(G), and crucially it VANISHES as the
// gradient vanishes near the optimum, so PSGD keeps tightening where Adam's fixed-size
// sign step floors out. The factors are updated by the normalized multiplicative
// (Lie-group) rule that keeps them strictly positive:
//
//	aᵢⱼ = dL[i]·dR[j]·G[i,j] ,  bᵢⱼ = V[i,j]/(dL[i]·dR[j])
//	gradL[i] = Σⱼ(aᵢⱼ² − bᵢⱼ²) ,  normL = maxᵢ Σⱼ(aᵢⱼ² + bᵢⱼ²)
//	dL[i] ← dL[i] · (1 − (PreconditionerLR/normL)·gradL[i])     (and symmetrically for dR)
//
// The normalization by the largest row/column energy bounds the multiplicative change by
// PreconditionerLR < 1, so the factors never cross zero. Preconditioner state is float64
// (§V10). This is the DIAGONAL variant: each factor is a diagonal (per-row/per-column)
// scale. A full TRIANGULAR factor Q per axis — PSGD's stronger form that also captures
// within-axis rotations (the source of Kron's biggest wins over Adam on transformers) —
// is a documented follow-up; the diagonal form already delivers the ill-conditioning
// robustness with the smallest, most easily validated implementation.
//
// COLLAPSE TO SGD. With PreconditionerLR = 0 the factors are frozen at their identity
// initialization (all ones), Ĝ = G, and PSGDKron is bit-identical to plain SGD — the
// preconditioner is opt-in curvature adaptation layered on the same step.
//
// Further reading: Li 2015 (arXiv:1512.04202, the original PSGD) and Li 2022
// (arXiv:2211.04422, the black-box Lie-group / whitening form); the evanatyourservice/kron
// reference implementation; Martens & Grosse 2015 (K-FAC, arXiv:1503.05671) for the
// related Kronecker-factored curvature idea; Goodfellow et al. "Deep Learning" ch. 8 for
// second-order and preconditioned optimization background.
type PSGDKron struct {
	Params []*tensor.Tensor // parameters this optimizer updates

	// LR is the learning rate (step size) applied to the PRECONDITIONED gradient. In plain
	// terms: how far to step downhill each update after the per-coordinate rescaling — too
	// small and training crawls, too large and it diverges. Boundary behavior: LR→0 stalls
	// learning; past the (problem-specific) stability threshold the loss oscillates or
	// explodes. Because the preconditioner already balances the coordinates, PSGD tolerates
	// a single well-chosen LR across a wider conditioning range than SGD. No universal
	// default (it is the one hyperparameter to tune) — 0.1–1.0 are common with a learned
	// preconditioner in Li's experiments (arXiv:1512.04202 §4).
	LR float64
	// PreconditionerLR is the step size of the online preconditioner update (Li's
	// precond_lr). In plain terms: how fast the row/column rescalings adapt to the incoming
	// gradients — larger tracks changing curvature faster but noisier, smaller is steadier.
	// Boundary behavior: MUST be < 1 for the normalized multiplicative update to keep the
	// factors positive (values near 1 make the factors jump); as it →0 the preconditioner
	// stops learning. SPECIAL VALUE: 0 = freeze the factors at the identity ⇒ PSGDKron is
	// exactly plain SGD (the collapse used to validate the step arithmetic). Default 0.1,
	// the value used by Li's Lie-group PSGD and the evanatyourservice/kron reference
	// (arXiv:2211.04422).
	PreconditionerLR float64
	// Momentum is the classical heavy-ball coefficient μ applied to the PRECONDITIONED
	// gradient: v ← μ·v + Ĝ; W ← W − lr·v. In plain terms: how much of the previous
	// (preconditioned) step to carry forward, smoothing noise and accelerating down long
	// valleys. Boundary behavior: 0 = no inertia; →1 overshoots and oscillates. SPECIAL
	// VALUE: 0 = disabled (no velocity buffer allocated, and the SGD-collapse stays
	// bit-identical). Default 0 (plain preconditioned SGD); 0.9 is the standard momentum
	// value when enabled.
	Momentum float64
	// Eps is the tiny floor added to the normalizer of the multiplicative factor update,
	// guarding the division when a row/column has (near-)zero gradient+probe energy. In
	// plain terms: a numerical safety net, not a tuning knob. Boundary behavior: any value
	// this small is inert on normal steps (the real energy dominates); too large would damp
	// the preconditioner update. Default 1e-30 (a safe float64 floor, matching the tiny
	// constant Li's reference uses to stabilize the normalized update).
	Eps float64
	// Seed seeds the deterministic probe RNG. In plain terms: fixes the random probes V so
	// a run is reproducible (the probes are drawn standard-normal, one per gradient element,
	// in row-major order, per parameter, only on steps that actually update the
	// preconditioner). Boundary behavior: any int64 is valid; changing it reshuffles the
	// probe stream (the trajectory shifts slightly but the fixed point is unchanged).
	// Default 1. SPECIAL: has no effect when PreconditionerLR = 0 (no probes are drawn).
	Seed int64

	rng  *rand.Rand
	dL   [][]float64 // per-param left/leading-axis (or 1-D single) diagonal factor
	dR   [][]float64 // per-param right/trailing-axis diagonal factor (nil for 1-D params)
	cols []int       // trailing-flattened column count per param (0 marks a 1-D param)
	vel  [][]float64 // per-param momentum buffer (nil when Momentum==0)

	// step-scratch reused across params within one Step (single-threaded)
	probe, gbuf, step, t1L, t2L []float64 // sized to the largest Numel
	t1R, t2R                    []float64 // sized to the largest column count
}

// PSGDKronOption configures a PSGDKron optimizer (functional-options idiom, §C12).
type PSGDKronOption func(*PSGDKron)

// WithPSGDKronPreconditionerLR sets the online preconditioner-update step size (see the
// PreconditionerLR field).
//
// In plain terms: how fast the row/column rescalings adapt. Boundary behavior: must be
// < 1 (the normalized update relies on it to keep the factors positive); →0 stops the
// preconditioner learning. SPECIAL VALUE: 0 = freeze at the identity ⇒ plain SGD. Default
// 0.1 (Li's Lie-group PSGD / evanatyourservice/kron, arXiv:2211.04422).
func WithPSGDKronPreconditionerLR(plr float64) PSGDKronOption {
	return func(o *PSGDKron) { o.PreconditionerLR = plr }
}

// WithPSGDKronMomentum sets the heavy-ball momentum μ on the preconditioned gradient (see
// the Momentum field).
//
// In plain terms: carry a fraction of the previous step forward. Boundary behavior: 0 =
// none; →1 overshoots. SPECIAL VALUE: 0 = disabled. Default 0; 0.9 is the standard value
// when enabled (Sutskever et al. 2013).
func WithPSGDKronMomentum(m float64) PSGDKronOption {
	return func(o *PSGDKron) { o.Momentum = m }
}

// WithPSGDKronEps sets the normalizer floor ε for the factor update (see the Eps field).
//
// In plain terms: a numerical safety net, not a tuning knob. Boundary behavior: inert on
// normal steps; too large damps the update. Default 1e-30.
func WithPSGDKronEps(eps float64) PSGDKronOption { return func(o *PSGDKron) { o.Eps = eps } }

// WithPSGDKronSeed sets the probe RNG seed for reproducibility (see the Seed field).
//
// In plain terms: fix the random probes so a run repeats exactly. Default 1. Has no effect
// when PreconditionerLR = 0.
func WithPSGDKronSeed(seed int64) PSGDKronOption { return func(o *PSGDKron) { o.Seed = seed } }

// NewPSGDKron builds a diagonal PSGD-Kron optimizer over params with learning rate lr and
// the research-grounded defaults PreconditionerLR=0.1, Momentum=0, ε=1e-30, seed=1
// (Li's Lie-group PSGD / evanatyourservice/kron, arXiv:2211.04422). The factors start at
// the identity (all ones), so the very first step equals an SGD step and the preconditioner
// warms up from there. Set PreconditionerLR=0 (via [WithPSGDKronPreconditionerLR]) to
// freeze the factors and recover plain SGD exactly. Only lr has no universal default and
// must be chosen for the task (see the LR field).
func NewPSGDKron(params []*tensor.Tensor, lr float64, opts ...PSGDKronOption) *PSGDKron {
	o := &PSGDKron{
		Params: params, LR: lr,
		PreconditionerLR: 0.1, Eps: 1e-30, Seed: 1,
	}
	for _, opt := range opts {
		opt(o)
	}
	o.rng = rand.New(rand.NewSource(o.Seed))
	o.dL = make([][]float64, len(params))
	o.dR = make([][]float64, len(params))
	o.cols = make([]int, len(params))
	maxN, maxCols := 0, 0
	//perfscan:ignore PS3065 NewPSGDKron construction, one-time init
	for i, p := range params {
		n := p.Numel()
		sh := p.Shape()
		if len(sh) >= 2 && sh[0] > 0 {
			rows := sh[0]
			cols := n / rows
			o.cols[i] = cols
			o.dL[i] = ones(rows)
			o.dR[i] = ones(cols)
			if cols > maxCols {
				maxCols = cols
			}
		} else {
			o.cols[i] = 0
			o.dL[i] = ones(n) // 1-D / scalar: single per-element diagonal preconditioner
		}
		if n > maxN {
			maxN = n
		}
	}
	if o.Momentum != 0 {
		o.vel = make([][]float64, len(params))
		for i, p := range params {
			o.vel[i] = make([]float64, p.Numel())
		}
	}
	o.probe = make([]float64, maxN)
	o.gbuf = make([]float64, maxN)
	o.step = make([]float64, maxN)
	o.t1L = make([]float64, maxN)
	o.t2L = make([]float64, maxN)
	o.t1R = make([]float64, maxCols)
	o.t2R = make([]float64, maxCols)
	return o
}

func ones(n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

// Factors returns copies of the learned diagonal preconditioner factors for parameter i:
// the leading-axis (or, for a 1-D parameter, the single) factor as left, and the
// trailing-axis factor as right (nil for a 1-D parameter). In plain terms: inspect what
// the optimizer has learned — each entry is the per-row/per-column rescaling applied to
// the gradient, converging toward the inverse square root of that coordinate's gradient
// magnitude.
func (o *PSGDKron) Factors(i int) (left, right []float64) {
	left = append([]float64(nil), o.dL[i]...)
	if o.cols[i] > 0 {
		right = append([]float64(nil), o.dR[i]...)
	}
	return left, right
}

// Step applies one preconditioned update. Parameters with a nil gradient are skipped; an
// all-zero gradient is a NaN-free no-op that neither moves the parameter nor perturbs the
// preconditioner (and draws no probe). A grad/param shape mismatch errors like the other
// optimizers.
func (o *PSGDKron) Step(grad GradFn) error {
	//perfscan:ignore PS3044 outer param-loop low-trip; body already flat typed buffers
	for pi, p := range o.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: PSGDKron grad shape %v != param %v", g.Shape(), p.Shape())
		}
		n := p.Numel()
		if n == 0 {
			continue
		}
		gbuf := o.gbuf[:n]
		readFlat64(g, gbuf)

		allZero := true
		for _, v := range gbuf {
			if v != 0 {
				allZero = false
				break
			}
		}
		dL, dR, cols := o.dL[pi], o.dR[pi], o.cols[pi]
		if o.PreconditionerLR > 0 && !allZero {
			o.updateFactors(gbuf, dL, dR, cols)
		}

		// Preconditioned gradient Ĝ = (dL⊗dR)⊙G, then the (optionally momentum-smoothed) step.
		var vel []float64
		if o.vel != nil {
			vel = o.vel[pi]
		}
		step := o.step[:n]
		if cols > 0 { // ≥2-D: factor dL[row]·dR[col]
			for k := 0; k < n; k++ {
				gh := dL[k/cols] * dR[k%cols] * gbuf[k]
				if vel != nil {
					vel[k] = o.Momentum*vel[k] + gh
					gh = vel[k]
				}
				step[k] = gh
			}
		} else { // 1-D: per-element factor dL[k]
			for k := 0; k < n; k++ {
				gh := dL[k] * gbuf[k]
				if vel != nil {
					vel[k] = o.Momentum*vel[k] + gh
					gh = vel[k]
				}
				step[k] = gh
			}
		}

		// Apply W ← W − lr·step (typed fast paths + generic fallback). With PreconditionerLR
		// 0 and Momentum 0 the factors are 1, step==G, and this is SGD's exact arithmetic.
		lr := o.LR
		if pf := flatF64(p); pf != nil {
			for k := range pf {
				pf[k] -= lr * step[k]
			}
		} else if pf := flatF32(p); pf != nil {
			for k := range pf {
				pf[k] = float32(float64(pf[k]) - lr*step[k])
			}
		} else {
			sh := p.Shape()
			for k := 0; k < n; k++ {
				idx := tensor.Unravel(k, sh)
				p.SetF64(p.AtF64(idx...)-lr*step[k], idx...)
			}
		}
	}
	return nil
}

// updateFactors performs one normalized multiplicative PSGD update of the diagonal
// factors from the gradient (gbuf, float64 row-major) and a fresh standard-normal probe.
func (o *PSGDKron) updateFactors(gbuf, dL, dR []float64, cols int) {
	n := len(gbuf)
	v := o.probe[:n]
	for k := 0; k < n; k++ {
		v[k] = o.rng.NormFloat64()
	}
	plr, tiny := o.PreconditionerLR, o.Eps
	if cols > 0 {
		rows := n / cols
		t1L, t2L := o.t1L[:rows], o.t2L[:rows]
		t1R, t2R := o.t1R[:cols], o.t2R[:cols]
		for i := range t1L {
			t1L[i], t2L[i] = 0, 0
		}
		for j := range t1R {
			t1R[j], t2R[j] = 0, 0
		}
		for k := 0; k < n; k++ {
			i, j := k/cols, k%cols
			f := dL[i] * dR[j]
			a := f * gbuf[k]
			b := v[k] / f
			aa, bb := a*a, b*b
			t1L[i] += aa
			t2L[i] += bb
			t1R[j] += aa
			t2R[j] += bb
		}
		normL := tiny
		for i := range t1L {
			if s := t1L[i] + t2L[i]; s > normL {
				normL = s
			}
		}
		muL := plr / normL
		for i := range dL {
			dL[i] *= 1 - muL*(t1L[i]-t2L[i])
		}
		normR := tiny
		for j := range t1R {
			if s := t1R[j] + t2R[j]; s > normR {
				normR = s
			}
		}
		muR := plr / normR
		for j := range dR {
			dR[j] *= 1 - muR*(t1R[j]-t2R[j])
		}
		return
	}
	// 1-D: each factor governs exactly one element (t1/t2 are per-element).
	t1, t2 := o.t1L[:n], o.t2L[:n]
	norm := tiny
	for k := 0; k < n; k++ {
		f := dL[k]
		a := f * gbuf[k]
		b := v[k] / f
		t1[k], t2[k] = a*a, b*b
		if s := t1[k] + t2[k]; s > norm {
			norm = s
		}
	}
	mu := plr / norm
	for k := 0; k < n; k++ {
		dL[k] *= 1 - mu*(t1[k]-t2[k])
	}
}

// readFlat64 copies t's logical row-major values into buf (len ≥ t.Numel()) as float64:
// bulk copy for contiguous f64/f32, widening accessor otherwise.
func readFlat64(t *tensor.Tensor, buf []float64) {
	if f := flatF64(t); f != nil {
		copy(buf, f)
		return
	}
	if f := flatF32(t); f != nil {
		for i, v := range f {
			buf[i] = float64(v)
		}
		return
	}
	sh := t.Shape()
	for i := range t.Numel() {
		buf[i] = t.AtF64(tensor.Unravel(i, sh)...)
	}
}
