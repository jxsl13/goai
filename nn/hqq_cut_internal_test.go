package nn

import (
	"math"
	"math/rand"
	"testing"
)

// hqqGoldenFixture is the deterministic input the golden below pins, kept in one place so the
// generator and the assertion cannot drift apart.
func hqqGoldenFixture() []float64 {
	rng := rand.New(rand.NewSource(20260806))
	w := make([]float64, 512)
	for i := range w {
		w[i] = rng.NormFloat64() * (1 + 0.3*math.Sin(float64(i)))
	}
	return w
}

// TestHQQShrinkIsZeroBelowTheCut is the correctness argument for skipping the pow, stated as a
// property rather than assumed.
//
// The shrink returns 0 exactly when |d| ≤ (1/β)·|d|^(p−1). Multiplying both sides by |d|^(1−p),
// which is positive, gives |d|^(2−p) ≤ 1/β — so the test has the closed form |d| ≤ (1/β)^(1/(2−p)),
// computable once per iteration instead of once per weight.
//
// The quantizer uses that bound pulled DOWN by a relative margin, and what has to hold is that
// every |d| below the lowered cut really does shrink to zero. This sweeps β over the range the
// iteration walks (β₀·κⁿ) and p over the configurable range, checking values just under the cut,
// far under it, and — as a control that the cut is not trivially large — that values just ABOVE
// the unlowered bound do NOT shrink to zero.
func TestHQQShrinkIsZeroBelowTheCut(t *testing.T) {
	for _, p := range []float64{0.5, 0.7, 0.9, 1.5} {
		beta := 1.0
		for range 40 {
			cut := math.Pow(1/beta, 1/(2-p))
			cutLow := cut * (1 - 1e-12)
			for _, frac := range []float64{1e-6, 0.5, 0.9, 0.999, 1 - 1e-9} {
				d := cutLow * frac
				if got := shrinkLp(d, beta, p); got != 0 {
					t.Fatalf("p=%g beta=%g |d|=%g (%.6g of the cut): shrink gave %g, want 0 —"+
						" the skip would change the result", p, beta, d, frac, got)
				}
				if got := shrinkLp(-d, beta, p); got != 0 {
					t.Fatalf("p=%g beta=%g d=-%g: shrink gave %g, want 0", p, beta, d, got)
				}
			}
			// Control: just above the true cut the shrink must be non-zero, or the bound is
			// vacuously large and the test above proves nothing.
			if d := cut * (1 + 1e-6); shrinkLp(d, beta, p) == 0 {
				t.Fatalf("p=%g beta=%g |d|=%g just above the cut: shrink gave 0, so the cut is"+
					" not the boundary it claims to be", p, beta, d)
			}
			beta *= 1.01
		}
	}
}

// hqqLargeFixture is deliberately WIDE. The cut is (1/β)^(1/(2−p)), which starts at 1 and only
// shrinks as β grows, so a fixture of unit-scale weights leaves every residual far BELOW it — the
// pow is skipped for all of them and the cut's exact value cannot matter. Mutations that raised
// the cut by half, or dropped the negative half of the band, passed against such a fixture.
//
// Scaling the weights up puts most residuals ABOVE the cut, so the pow path runs and the boundary
// is what the golden depends on.
func hqqLargeFixture() []float64 {
	rng := rand.New(rand.NewSource(20260807))
	w := make([]float64, 512)
	for i := range w {
		w[i] = rng.NormFloat64() * 100
	}
	return w
}

// TestHQQuantizeWideOutputIsFrozen is that fixture's golden, generated the same way.
func TestHQQuantizeWideOutputIsFrozen(t *testing.T) {
	codes, scale, zero := HQQuantize(hqqLargeFixture(), 4, 64)
	cs, ss, zs := hqqDigest(codes, scale, zero)
	const wantCodes, wantScale, wantZero uint64 = 16569551713832293144, 8373000343230715716, 9291206234723566854
	if cs != wantCodes || ss != wantScale || zs != wantZero {
		t.Fatalf("wide HQQ output changed:\n codes %d want %d\n scale %d want %d\n zero  %d want %d",
			cs, wantCodes, ss, wantScale, zs, wantZero)
	}
}

// hqqDigest folds the three outputs into comparable hashes.
func hqqDigest(codes []int, scale, zero []float64) (cs, ss, zs uint64) {
	for _, c := range codes {
		cs = cs*1099511628211 ^ uint64(c+1)
	}
	for i := range scale {
		ss = ss*1099511628211 ^ math.Float64bits(scale[i])
		zs = zs*1099511628211 ^ math.Float64bits(zero[i])
	}
	return cs, ss, zs
}

// TestHQQuantizeOutputIsFrozen pins the exact codes, scales and zeros for a fixed input. The
// expected values were generated from the implementation as it stood BEFORE the pow skip, which
// is what makes this evidence that the skip changed nothing rather than a restatement of what it
// does.
func TestHQQuantizeOutputIsFrozen(t *testing.T) {
	codes, scale, zero := HQQuantize(hqqGoldenFixture(), 4, 64)
	cs, ss, zs := hqqDigest(codes, scale, zero)
	const wantCodes, wantScale, wantZero uint64 = 3640501409108930686, 379505289789257397, 13231318097518461017
	if cs != wantCodes || ss != wantScale || zs != wantZero {
		t.Fatalf("HQQ output changed:\n codes %d want %d\n scale %d want %d\n zero  %d want %d",
			cs, wantCodes, ss, wantScale, zs, wantZero)
	}
}
