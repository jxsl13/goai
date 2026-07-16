package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 anchor 3 (GATE RANGES / STABILITY). Two parts:
//
//   - moderate Λ / inputs: the decay is strictly a_t ∈ (0,1) and the input
//     normalizer √(1−a_t²) ∈ (0,1] (the mathematical guarantee — the log-space
//     exponent c·r_t·logσ(Λ) is strictly < 0 and finite);
//   - extreme Λ (±50) with large inputs: the computation stays FINITE — a_t ∈ [0,1]
//     (it may saturate to the closed boundary in f64), the radicand 1−a_t² is never
//     negative (√ stays real), and b is finite (no NaN/Inf). This is the stability
//     claim: the log-space formulation never overflows for extreme decay parameters.
func TestRGLRUGateBoundsAndStability(t *testing.T) {
	const l, d = 6, 5

	// Part 1 — strict bounds under a moderate regime.
	m, err := NewRGLRU(tensor.F64, d, WithRGLRUSeed(3))
	if err != nil {
		t.Fatal(err)
	}
	for k := range d { // moderate Λ so a_t stays clear of the f64 boundary
		m.Lambda.SetF64(0.3*float64(k)-0.6, k)
	}
	x := tensor.New(tensor.F64, tensor.Shape{l, d})
	for i := range x.Numel() {
		coord := tensor.Unravel(i, x.Shape())
		x.SetF64(math.Sin(float64(i)), coord...)
	}
	_, a, b, err := m.gates(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.Numel() {
		coord := tensor.Unravel(i, a.Shape())
		av := a.AtF64(coord...)
		if !(av > 0 && av < 1) {
			t.Errorf("moderate: a%v = %.17g not strictly in (0,1)", coord, av)
		}
		norm := math.Sqrt(1 - av*av)
		if !(norm > 0 && norm <= 1) {
			t.Errorf("moderate: √(1−a²) at %v = %.17g not in (0,1]", coord, norm)
		}
		if bv := b.AtF64(coord...); math.IsNaN(bv) || math.IsInf(bv, 0) {
			t.Errorf("moderate: b%v = %v not finite", coord, bv)
		}
	}

	// Part 2 — finiteness/stability under extreme Λ and large inputs.
	m2, err := NewRGLRU(tensor.F64, d, WithRGLRUSeed(3))
	if err != nil {
		t.Fatal(err)
	}
	m2.Lambda.SetF64(50, 0)  // a = σ(50) ≈ 1
	m2.Lambda.SetF64(-50, 1) // a = σ(−50) ≈ 0
	m2.Lambda.SetF64(0, 2)
	xb := tensor.New(tensor.F64, tensor.Shape{l, d})
	for i := range xb.Numel() {
		coord := tensor.Unravel(i, xb.Shape())
		xb.SetF64(20*math.Sin(float64(i)), coord...)
	}
	_, a2, b2, err := m2.gates(backend.NewContext(), xb)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a2.Numel() {
		coord := tensor.Unravel(i, a2.Shape())
		av := a2.AtF64(coord...)
		if math.IsNaN(av) || math.IsInf(av, 0) || av < 0 || av > 1 {
			t.Errorf("extreme: a%v = %.17g not finite in [0,1]", coord, av)
		}
		if rad := 1 - av*av; rad < 0 { // √ must stay real: radicand never negative
			t.Errorf("extreme: 1−a² at %v = %.17g negative (√ would be complex)", coord, rad)
		}
		if bv := b2.AtF64(coord...); math.IsNaN(bv) || math.IsInf(bv, 0) {
			t.Errorf("extreme: b%v = %v not finite", coord, bv)
		}
	}
}
