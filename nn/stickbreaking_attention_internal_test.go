package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// sbSigmoid is a local reference logistic σ(x)=1/(1+e^-x).
func sbSigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// sbDirectWeights is a small, NON-log reference of the stick-breaking allocation:
// A_{ij} = β_{ij}·∏_{j<k≤i}(1−β_{ik}) with β=σ(z), for j≤i (0 above the diagonal).
// This is the plain product form the log-space path in stickBreakingWeights must
// reproduce (§V16.3).
func sbDirectWeights(z *tensor.Tensor) [][]float64 {
	t := z.Shape()[0]
	a := make([][]float64, t)
	for i := range t {
		a[i] = make([]float64, t)
		for j := 0; j <= i; j++ {
			beta := sbSigmoid(z.AtF64(i, j))
			prod := 1.0
			for k := j + 1; k <= i; k++ {
				prod *= 1 - sbSigmoid(z.AtF64(i, k))
			}
			a[i][j] = beta * prod
		}
	}
	return a
}

// sbWeightsF64 runs the production stickBreakingWeights on z and returns a plain
// [][]float64 for comparison.
func sbWeightsF64(t *testing.T, z *tensor.Tensor) [][]float64 {
	t.Helper()
	n := z.Shape()[0]
	mask := sbCausalMask01(z.Dtype(), n)
	w, err := stickBreakingWeights(backend.NewContext(), z, mask)
	if err != nil {
		t.Fatal(err)
	}
	out := make([][]float64, n)
	for i := range n {
		out[i] = make([]float64, n)
		for j := range n {
			out[i][j] = w.AtF64(i, j)
		}
	}
	return out
}

func sbRandZ(rng *rand.Rand, n int) *tensor.Tensor {
	z := tensor.New(tensor.F64, tensor.Shape{n, n})
	for i := range n {
		for j := range n {
			z.SetF64(rng.NormFloat64(), i, j)
		}
	}
	return z
}

// §V16.1 SINGLE-KEY / RECENCY (anchor): the most-recent key j=i takes exactly the
// first stick break A_{ii}=σ(z_{ii}) @1e-10, and the two-key closed form
// A_{i,i-1}=β_{i-1}(1−β_i) matches @1e-10.
func TestStickBreakingSingleKeyRecency(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	z := sbRandZ(rng, 5)
	w := sbWeightsF64(t, z)
	for i := range 5 {
		want := sbSigmoid(z.AtF64(i, i))
		if d := math.Abs(w[i][i] - want); d > 1e-10 {
			t.Errorf("A[%d,%d]=%.12g, want σ(z)=%.12g (diff %.3g)", i, i, w[i][i], want, d)
		}
	}
	// Two-key hand case at row i=3, key i-1=2.
	i := 3
	betaCur := sbSigmoid(z.AtF64(i, i))
	betaPrev := sbSigmoid(z.AtF64(i, i-1))
	want := betaPrev * (1 - betaCur)
	if d := math.Abs(w[i][i-1] - want); d > 1e-10 {
		t.Errorf("A[%d,%d]=%.12g, want β_{i-1}(1-β_i)=%.12g (diff %.3g)", i, i-1, w[i][i-1], want, d)
	}
}

// §V16.2 VALID ALLOCATION (anchor): every A_{ij}∈[0,1] and Σ_j A_{ij} ≤ 1 (the
// stick-breaking signature — NOT =1); and as the OLDEST key's β→1 the stick is
// fully consumed so the row sum →1.
func TestStickBreakingValidAllocation(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 99))
	z := sbRandZ(rng, 8)
	w := sbWeightsF64(t, z)
	for i := range 8 {
		var sum float64
		for j := range 8 {
			if w[i][j] < 0 || w[i][j] > 1 {
				t.Errorf("A[%d,%d]=%.6g out of [0,1]", i, j, w[i][j])
			}
			sum += w[i][j]
		}
		if sum > 1+1e-9 {
			t.Errorf("row %d sums to %.12g > 1 (stick-breaking must be ≤1)", i, sum)
		}
	}
	// Drive the OLDEST key (column 0) hard positive → β_{i0}→1 → remainder→0 →
	// each row's mass is fully allocated (Σ→1).
	z2 := sbRandZ(rng, 8)
	for i := range 8 {
		z2.SetF64(30, i, 0)
	}
	w2 := sbWeightsF64(t, z2)
	for i := range 8 {
		var sum float64
		for j := range 8 {
			sum += w2[i][j]
		}
		if math.Abs(sum-1) > 1e-6 {
			t.Errorf("oldest β→1: row %d sums to %.10g, want ≈1", i, sum)
		}
	}
}

// §V16.3 LOG-SPACE ↔ DIRECT (anchor): the log-space reverse-cumsum weights equal a
// direct non-log β·∏(1−β) reference @1e-8 on a tiny case; this pins the numerical
// path (logσ via −softplus, cumsum, exp) against the definition.
func TestStickBreakingLogMatchesDirect(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	z := sbRandZ(rng, 6)
	got := sbWeightsF64(t, z)
	want := sbDirectWeights(z)
	var maxd float64
	for i := range 6 {
		for j := range 6 {
			if d := math.Abs(got[i][j] - want[i][j]); d > maxd {
				maxd = d
			}
		}
	}
	if maxd > 1e-8 {
		t.Errorf("log-space vs direct max abs diff %.3g > 1e-8", maxd)
	}
	t.Logf("log-space vs direct: max abs diff %.3g", maxd)
}

// §V16.4 CAUSALITY (anchor): A_{ij}=0 bit-exact for future keys j>i.
func TestStickBreakingWeightsCausalZero(t *testing.T) {
	rng := rand.New(rand.NewPCG(55, 66))
	z := sbRandZ(rng, 7)
	w := sbWeightsF64(t, z)
	for i := range 7 {
		for j := i + 1; j < 7; j++ {
			if w[i][j] != 0 {
				t.Errorf("A[%d,%d]=%v, want exact 0 (future key)", i, j, w[i][j])
			}
		}
	}
}
