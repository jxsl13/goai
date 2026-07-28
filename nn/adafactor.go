package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Adafactor (Shazeer & Stern 2018, arXiv:1804.04235, §R80): an Adam-class optimizer
// with SUBLINEAR memory. For a matrix parameter it does not store the full
// second-moment matrix V (as Adam does) but only its per-row and per-column sums R
// and C, reconstructing V̂[i,j] = R[i]·C[j]/ΣR — the rank-1 (independence)
// approximation — so state is O(rows+cols) instead of O(rows·cols). It also drops
// the manual learning rate: the step is RELATIVE to the parameter scale
// (α_t = max(ε₂, RMS(θ))·ρ_t) and updates are clipped by their RMS. First moment is
// off by default (the memory-saving configuration used to train T5).
//
//	matrix:  R ← β̂₂R + (1−β̂₂)rowsum(g²+ε₁);  C ← β̂₂C + (1−β̂₂)colsum(g²+ε₁)
//	         V̂ = R⊗C/ΣR
//	vector:  V̂ ← β̂₂V̂ + (1−β̂₂)(g²+ε₁)            (no factoring)
//	U = g/√V̂;  Û = U/max(1, RMS(U)/d);  θ ← θ − max(ε₂,RMS(θ))·ρ_t·Û
//	with β̂₂,t = 1−t^(−0.8) and ρ_t = min(LR, 1/√t).
type Adafactor struct {
	Params []*tensor.Tensor // parameters this optimizer updates
	LR     float64          // relative-step cap ρ (default 1e-2); ρ_t = min(LR, 1/√t)
	Beta1  float64          // first-moment EMA decay; 0 = off (default, memory-saving)
	Eps1   float64          // second-moment regularization added to g² (default 1e-30)
	Eps2   float64          // relative-step floor on RMS(θ) (default 1e-3)
	Clip   float64          // update-clipping threshold d (default 1)

	t    int
	r, c [][]float64 // factored row/col second moments for rank-2 params (else nil)
	v    [][]float64 // full second moment for non-matrix params (else nil)
	m    [][]float64 // first moment (allocated only when Beta1 > 0)

	// Per-Step scratch, reused across steps AND across params within a step (params are
	// processed serially, each finishing before the next). Hoisting these off the hot
	// path drops Adafactor.Step from 12 allocs/3.1 MB per call to zero — no GC churn. Each
	// is fully overwritten (or explicitly zeroed) before use, so reuse is bit-identical.
	uScr, vhatScr, rowScr, colScr []float64
}

// growF64 returns b resliced to length n, reallocating only when its capacity is short.
func growF64(b []float64, n int) []float64 {
	if cap(b) < n {
		return make([]float64, n)
	}
	return b[:n]
}

// AdafactorOption configures an Adafactor optimizer (functional-options idiom, §C12).
type AdafactorOption func(*Adafactor)

// WithAdafactorLR sets the relative-step cap ρ, applied as ρ_t = min(LR, 1/√t).
//
// In plain terms: an upper bound on Adafactor's automatically-decaying step size — it starts
// near this value and shrinks as 1/√t. Boundary behavior — larger LR lets early steps be
// bigger (faster but riskier); smaller caps them throughout. Unlike Adam this is a CAP on a
// self-scheduling step, not a fixed rate.
//
// Default 1e-2 (research-grounded): the ρ cap from Shazeer & Stern 2018 (§R80).
func WithAdafactorLR(lr float64) AdafactorOption { return func(a *Adafactor) { a.LR = lr } }

// WithAdafactorBeta1 enables a first-moment (momentum) EMA with decay β1.
//
// In plain terms: turn on momentum. Adafactor's whole point is sublinear memory, and momentum
// costs a full-size extra buffer — so it is OFF by default. Boundary behavior — 0 = no
// momentum (the memory-saving configuration); a value like 0.9 adds Adam-style momentum at the
// cost of that buffer. SPECIAL VALUE: 0 = disabled (no first-moment state allocated).
//
// Default 0 (research-grounded): Shazeer & Stern 2018 (§R80) run Adafactor without the first
// moment to keep state at O(rows+cols).
func WithAdafactorBeta1(b1 float64) AdafactorOption { return func(a *Adafactor) { a.Beta1 = b1 } }

// WithAdafactorEps sets the two epsilons: ε1 regularizes the squared gradient, ε2 floors the
// relative step size.
//
// In plain terms: ε1 is a tiny floor added to g² so the variance estimate never collapses to
// zero; ε2 keeps the relative step from vanishing on parameters whose values are near zero.
// Boundary behavior — ε1 too large biases the variance estimate; ε2 too large stops the step
// from ever shrinking below that floor.
//
// Defaults 1e-30, 1e-3 (research-grounded): the ε1/ε2 from Shazeer & Stern 2018 (§R80).
func WithAdafactorEps(eps1, eps2 float64) AdafactorOption {
	return func(a *Adafactor) { a.Eps1, a.Eps2 = eps1, eps2 }
}

// WithAdafactorClip sets the update root-mean-square clipping threshold d.
//
// In plain terms: cap how large a typical coordinate of the update may be, which stabilizes
// training when gradients spike. Boundary behavior — small d clamps hard (slower, very
// stable); large d effectively disables clipping. The update is scaled by 1/max(1, RMS(U)/d).
//
// Default 1 (research-grounded): the clipping threshold d=1 from Shazeer & Stern 2018 (§R80).
func WithAdafactorClip(d float64) AdafactorOption { return func(a *Adafactor) { a.Clip = d } }

// NewAdafactor builds an Adafactor optimizer over params with the paper's defaults
// (ρ cap 1e-2, ε1 1e-30, ε2 1e-3, clip d 1, no first moment). Matrix (rank-2)
// parameters use the factored second moment; others fall back to the full one.
func NewAdafactor(params []*tensor.Tensor, opts ...AdafactorOption) *Adafactor {
	a := &Adafactor{Params: params, LR: 1e-2, Eps1: 1e-30, Eps2: 1e-3, Clip: 1}
	for _, o := range opts {
		o(a)
	}
	a.r = make([][]float64, len(params))
	a.c = make([][]float64, len(params))
	a.v = make([][]float64, len(params))
	if a.Beta1 > 0 {
		a.m = make([][]float64, len(params))
	}
	for i, p := range params {
		if p.Ndim() == 2 {
			a.r[i] = make([]float64, p.Shape()[0])
			a.c[i] = make([]float64, p.Shape()[1])
		} else {
			a.v[i] = make([]float64, p.Numel())
		}
		if a.Beta1 > 0 {
			a.m[i] = make([]float64, p.Numel())
		}
	}
	return a
}

// adafactorVhat reconstructs the factored second moment V̂[i,j] = R[i]·C[j]/ΣR (row-
// major, len(R)×len(C)). This rank-1 form is EXACT when g² is itself rank-1, which
// is the property the factorization exploits (Shazeer & Stern §3).
func adafactorVhat(out, r, c []float64) {
	var sumR float64
	for _, rv := range r {
		sumR += rv
	}
	// V̂ = R⊗C/ΣR — hoist the loop-invariant 1/ΣR to a per-row reciprocal-multiply
	// (rᵢ/ΣR once, then ·cⱼ). V̂ is a continuous second-moment estimate fed through
	// √ and a division, so the ½-ulp reassociation rides the optimizer's tolerance
	// (like the other reciprocal-multiplied moments), not a discrete/golden output.
	invSumR := 1 / sumR
	nc := len(c)
	for i, rv := range r {
		ri := rv * invSumR
		base := i * nc
		for j, cv := range c {
			out[base+j] = ri * cv
		}
	}
}

// Step applies one Adafactor update. Parameters with a nil gradient are skipped.
func (a *Adafactor) Step(grad GradFn) error {
	a.t++
	t := float64(a.t)
	beta2t := 1 - math.Pow(t, -0.8)
	rho := math.Min(a.LR, 1/math.Sqrt(t))

	for pi, p := range a.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: Adafactor grad shape %v != param %v", g.Shape(), p.Shape())
		}
		n := p.Numel()
		// Typed fast paths (contiguous f64/f32): flat loops over the row-major
		// order the generic Unravel walk visits, arithmetic in float64 exactly as
		// the generic path computes it.
		pf64, gf64 := flatF64(p), flatF64(g)
		pf32, gf32 := flatF32(p), flatF32(g)

		// Second-moment estimate V-hat fused directly into U = g/sqrt(V-hat), so the
		// full-size vhat scratch (an n-element write+read round-trip) is eliminated:
		// the 2D path reconstructs V-hat[i,j]=(R[i]/sumR)*C[j] inline; the vector
		// path divides by V directly. Values are bit-identical to the two-step form.
		u := growF64(a.uScr, n)
		a.uScr = u
		if p.Ndim() == 2 {
			rows, cols := p.Shape()[0], p.Shape()[1]
			rowsum := growF64(a.rowScr, rows)
			colsum := growF64(a.colScr, cols)
			a.rowScr, a.colScr = rowsum, colsum
			for i := range rowsum { // reused buffer: zero before accumulating
				rowsum[i] = 0
			}
			for j := range colsum {
				colsum[j] = 0
			}
			switch {
			case gf64 != nil:
				for i := range rows {
					base := i * cols
					for j := range cols {
						gv := gf64[base+j]
						g2 := gv*gv + a.Eps1
						rowsum[i] += g2
						colsum[j] += g2
					}
				}
			case gf32 != nil:
				for i := range rows {
					base := i * cols
					for j := range cols {
						gv := float64(gf32[base+j])
						g2 := gv*gv + a.Eps1
						rowsum[i] += g2
						colsum[j] += g2
					}
				}
			default:
				for i := range rows {
					for j := range cols {
						gv := g.AtF64(i, j)
						g2 := gv*gv + a.Eps1
						rowsum[i] += g2
						colsum[j] += g2
					}
				}
			}
			R, C := a.r[pi], a.c[pi]
			for i := range R {
				R[i] = beta2t*R[i] + (1-beta2t)*rowsum[i]
			}
			for j := range C {
				C[j] = beta2t*C[j] + (1-beta2t)*colsum[j]
			}
			// U = g / sqrt(V-hat) with V-hat[i,j]=(R[i]/sumR)*C[j] inline (same per-row
			// reciprocal-multiply adafactorVhat did, then divide) -> no vhat array.
			var sumR float64
			for _, rv := range R {
				sumR += rv
			}
			invSumR := 1 / sumR
			nc := len(C)
			switch {
			case gf64 != nil:
				for i, rv := range R {
					ri := rv * invSumR
					base := i * nc
					for j, cv := range C {
						u[base+j] = gf64[base+j] / math.Sqrt(ri*cv)
					}
				}
			case gf32 != nil:
				for i, rv := range R {
					ri := rv * invSumR
					base := i * nc
					for j, cv := range C {
						u[base+j] = float64(gf32[base+j]) / math.Sqrt(ri*cv)
					}
				}
			default:
				for i, rv := range R {
					ri := rv * invSumR
					base := i * nc
					for j, cv := range C {
						u[base+j] = g.AtF64(i, j) / math.Sqrt(ri*cv)
					}
				}
			}
		} else {
			V := a.v[pi]
			switch {
			case gf64 != nil:
				for i, gv := range gf64 {
					V[i] = beta2t*V[i] + (1-beta2t)*(gv*gv+a.Eps1)
					u[i] = gv / math.Sqrt(V[i])
				}
			case gf32 != nil:
				for i := range gf32 {
					gv := float64(gf32[i])
					V[i] = beta2t*V[i] + (1-beta2t)*(gv*gv+a.Eps1)
					u[i] = gv / math.Sqrt(V[i])
				}
			default:
				for i := range n {
					idx := tensor.Unravel(i, p.Shape())
					gv := g.AtF64(idx...)
					V[i] = beta2t*V[i] + (1-beta2t)*(gv*gv+a.Eps1)
					u[i] = gv / math.Sqrt(V[i])
				}
			}
		}
		if a.Beta1 > 0 {
			M := a.m[pi]
			for i := range n {
				M[i] = a.Beta1*M[i] + (1-a.Beta1)*u[i]
				u[i] = M[i]
			}
		}
		var su, sp float64
		switch {
		case pf64 != nil:
			for i, pv := range pf64 {
				su += u[i] * u[i]
				sp += pv * pv
			}
		case pf32 != nil:
			for i := range pf32 {
				su += u[i] * u[i]
				pv := float64(pf32[i])
				sp += pv * pv
			}
		default:
			for i := range n {
				su += u[i] * u[i]
				idx := tensor.Unravel(i, p.Shape())
				pv := p.AtF64(idx...)
				sp += pv * pv
			}
		}
		clip := math.Max(1, math.Sqrt(su/float64(n))/a.Clip)
		alpha := math.Max(a.Eps2, math.Sqrt(sp/float64(n))) * rho
		switch {
		case pf64 != nil:
			for i := range pf64 {
				pf64[i] = pf64[i] - alpha*(u[i]/clip)
			}
		case pf32 != nil:
			for i := range pf32 {
				pf32[i] = float32(float64(pf32[i]) - alpha*(u[i]/clip))
			}
		default:
			for i := range n {
				idx := tensor.Unravel(i, p.Shape())
				p.SetF64(p.AtF64(idx...)-alpha*(u[i]/clip), idx...)
			}
		}
	}
	return nil
}
