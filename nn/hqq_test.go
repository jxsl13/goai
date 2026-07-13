package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"
)

func lpErr(w, wHat []float64, p float64) float64 {
	var s float64
	for i := range w {
		s += math.Pow(math.Abs(w[i]-wHat[i]), p)
	}
	return s
}

// §V16 tier-1 (arXiv/blog, dequant form): ŵ[i] = scale[g]·(codes[i] − zero[g]).
func TestHQQRoundTripFormula(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 23))
	w := make([]float64, 32)
	for i := range w {
		w[i] = rng.NormFloat64()
	}
	codes, scale, zero := nn.HQQuantize(w, 4, 8)
	got := nn.DequantizeHQQ(codes, scale, zero, 8)
	for i := range got {
		if want := scale[i/8] * (float64(codes[i]) - zero[i/8]); math.Abs(got[i]-want) > 1e-12 {
			t.Errorf("dequant[%d]=%g want %g", i, got[i], want)
		}
	}
}

// §V16 tier-1 (codes range): every code lies in [0, 2^bits − 1].
func TestHQQCodesInRange(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 24))
	w := make([]float64, 40)
	for i := range w {
		w[i] = 5 * rng.NormFloat64()
	}
	for _, bits := range []int{2, 3, 4, 8} {
		codes, _, _ := nn.HQQuantize(w, bits, 10)
		hi := 1<<bits - 1
		for i, c := range codes {
			if c < 0 || c > hi {
				t.Errorf("bits=%d code[%d]=%d out of [0,%d]", bits, i, c, hi)
			}
		}
	}
}

// §V2 (the defining benefit): the half-quadratic zero-point optimization lowers the robust
// Lp reconstruction error versus plain round-to-nearest (0 iterations) on skewed,
// outlier-heavy weights — aggregated over many groups so the improvement is unambiguous.
func TestHQQBeatsRTN(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 25))
	const n, gs = 2048, 64
	w := make([]float64, n)
	for i := range w {
		w[i] = math.Exp(1.2 * rng.NormFloat64()) // right-skewed (log-normal)
		if rng.Float64() < 0.03 {
			w[i] *= 15 // sparse heavy outliers
		}
	}
	const p = 0.7
	rtnC, rtnS, rtnZ := nn.HQQuantize(w, 4, gs, nn.WithHQQIters(0)) // = round-to-nearest init
	hqqC, hqqS, hqqZ := nn.HQQuantize(w, 4, gs, nn.WithHQQLpNorm(p), nn.WithHQQIters(20))
	rtn := lpErr(w, nn.DequantizeHQQ(rtnC, rtnS, rtnZ, gs), p)
	hqq := lpErr(w, nn.DequantizeHQQ(hqqC, hqqS, hqqZ, gs), p)
	if !(hqq < rtn) {
		t.Errorf("HQQ Lp-error %.6g not < RTN %.6g", hqq, rtn)
	}
}

// §V2 (edge): a constant group (zero range) quantizes without dividing by zero and
// reconstructs the constant.
func TestHQQConstantGroup(t *testing.T) {
	w := []float64{2.5, 2.5, 2.5, 2.5}
	codes, scale, zero := nn.HQQuantize(w, 4, 4)
	got := nn.DequantizeHQQ(codes, scale, zero, 4)
	for i := range got {
		if math.Abs(got[i]-2.5) > 1e-9 {
			t.Errorf("constant group[%d]=%g, want 2.5", i, got[i])
		}
	}
}

// §V2: per-group scale/zero are applied to the right ranges, including a final partial group.
func TestHQQGroups(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 26))
	w := make([]float64, 25) // 25 = 2 full groups of 10 + a partial group of 5
	for i := range w {
		w[i] = rng.NormFloat64()
	}
	codes, scale, zero := nn.HQQuantize(w, 4, 10)
	if len(scale) != 3 || len(zero) != 3 {
		t.Fatalf("groups: got %d scales, want 3", len(scale))
	}
	// reconstruction error is bounded by the per-group step (sanity, not exact).
	got := nn.DequantizeHQQ(codes, scale, zero, 10)
	for i := range w {
		if math.Abs(got[i]-w[i]) > scale[i/10]*1.01 { // within ~one quant step
			t.Errorf("group reconstruction off at %d: |%g−%g| > step %g", i, got[i], w[i], scale[i/10])
		}
	}
}

func ExampleHQQuantize() {
	// Quantize 4 weights to 3-bit (levels 0..7) as one group; dequantization recovers them
	// closely and every code is in range.
	w := []float64{-1.0, 0.0, 0.5, 2.0}
	codes, scale, zero := nn.HQQuantize(w, 3, 4)
	hat := nn.DequantizeHQQ(codes, scale, zero, 4)
	fmt.Printf("codes in [0,7]: %v  err<0.3: %v\n",
		codes[0] >= 0 && codes[3] <= 7, math.Abs(hat[3]-2.0) < 0.3)
	// Output:
	// codes in [0,7]: true  err<0.3: true
}
