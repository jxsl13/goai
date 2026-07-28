package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// softcapVJPGenericRef is the SoftCap VJP's generic per-element fallback, written out
// as the oracle: Unravel + AtF64/SetF64, using the same hoisted invCap the typed arms
// use. Frozen — editing it to match a changed fast path would make the gate
// self-fulfilling.
func softcapVJPGenericRef(x, y, g *tensor.Tensor, cap float64) *tensor.Tensor {
	gin := tensor.New(x.Dtype(), x.Shape())
	invCap := 1 / cap
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		t := y.AtF64(idx...) * invCap
		gin.SetF64(g.AtF64(idx...)*(1-t*t), idx...)
	}
	return gin
}

// TestSoftCapVJPTypedMatchesGenericExact verifies the claim the source makes: that the
// F64 and F32 typed fast paths stay BIT-IDENTICAL to the generic AtF64 path because all
// three share one hoisted invCap.
//
// That claim was load-bearing and unverified. The reciprocal-multiply it justifies is a
// shipped optimization (1.28x/1.29x), and the exactness sweep showed NO test in the
// package detects a 1-ulp perturbation of either typed store — so nothing held the
// three paths together. This is the oracle that does.
//
// Note what is and is not asserted: the hoist from y/cap to y*invCap is a deliberate
// half-ulp reassociation against the UNOPTIMIZED form, and rides the gradient-check
// tolerance. What must be exact is the agreement BETWEEN the three paths, which is what
// a caller switching dtype or contiguity actually observes.
//
// COVERAGE, measured rather than assumed. Mutation probes: un-hoisting the F64 arm
// turns this red, and so does removing the F32 arm float64 intermediate. Un-hoisting
// the F32 arm does NOT, and that is not a fixture weakness — it is unobservable BY
// CONSTRUCTION. Both that arm and the generic path narrow their result to float32 at
// the end, so a half-ulp difference in t is absorbed far below f32 resolution.
// Widening the cap set from powers of two to inexact reciprocals (3, 7, 0.3, 1.1,
// 49.7) did not change it, which is what distinguishes a structural property from a
// missing test case. The F32 arm hoist is therefore safe for a reason no test can
// state; do not read the green as proof that arm is pinned.
func TestSoftCapVJPTypedMatchesGenericExact(t *testing.T) {
	vjp := vjps[backend.OpSoftCap]
	if vjp == nil {
		t.Fatal("no VJP registered for OpSoftCap")
	}
	rng := rand.New(rand.NewSource(20260728))
	// Caps whose reciprocal is INEXACT are what make the hoist observable; powers of
	// two (1, 0.125) divide exactly and cannot distinguish y/cap from y*invCap at all.
	// The F32 arm needed this: with only 1, 2.5, 30 and 0.125 a deliberate un-hoist of
	// that arm went undetected, so the gate did not in fact cover it.
	for _, cap := range []float64{1, 2.5, 30, 0.125, 3, 7, 0.3, 1.1, 49.7} {
		for _, shape := range []tensor.Shape{{7}, {3, 4}, {2, 3, 4}, {64}} {
			n := shape.Numel()
			for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
				x := tensor.New(dt, shape)
				y := tensor.New(dt, shape)
				g := tensor.New(dt, shape)
				for i := range n {
					idx := tensor.Unravel(i, shape)
					// y must lie in (-cap, cap), the range softcap's forward produces.
					y.SetF64(cap*math.Tanh(rng.NormFloat64()*1.5), idx...)
					g.SetF64(rng.NormFloat64()*math.Pow(2, float64(rng.Intn(9)-4)), idx...)
					x.SetF64(rng.NormFloat64(), idx...)
				}
				got, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y},
					backend.SoftCapAttrs{Cap: cap}, g)
				if err != nil {
					t.Fatalf("cap=%v %v %v: %v", cap, shape, dt, err)
				}
				want := softcapVJPGenericRef(x, y, g, cap)
				for i := range n {
					idx := tensor.Unravel(i, shape)
					a, b := got[0].AtF64(idx...), want.AtF64(idx...)
					if math.Float64bits(a) != math.Float64bits(b) {
						t.Fatalf("cap=%v shape=%v %v elem %d: typed %v (%#x) != generic %v (%#x)",
							cap, shape, dt, i, a, math.Float64bits(a), b, math.Float64bits(b))
					}
				}
			}
		}
	}
}
