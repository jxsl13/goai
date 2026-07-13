package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func exactMatMul(t *testing.T, x, w *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	o, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{x, w}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return o[0]
}

func froErr(a, b *tensor.Tensor) float64 {
	var s float64
	for i := range a.Numel() {
		c := tensor.Unravel(i, a.Shape())
		d := a.AtF64(c...) - b.AtF64(c...)
		s += d * d
	}
	return math.Sqrt(s)
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2208.07339 §3.2): a feature dimension is an
// outlier iff any |X[:,h]| ≥ threshold.
func TestLLMInt8OutlierDetection(t *testing.T) {
	x := toT([][]float64{{1, 8, 0.5, 2}, {2, -1, 3, 7}}) // col1 has 8≥6, col3 has 7≥6
	w := toT([][]float64{{1}, {1}, {1}, {1}})
	_, outIdx, err := nn.LLMInt8MatMul(x, w, 6.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(outIdx) != 2 || outIdx[0] != 1 || outIdx[1] != 3 {
		t.Errorf("outlier dims = %v, want [1 3]", outIdx)
	}
}

// §V16 tier-1 (exact split): with threshold 0 every dimension is an outlier, so the
// whole matmul runs in high precision and equals X·W exactly.
func TestLLMInt8DecompositionExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 140))
	x, w := randMat(rng, 4, 5), randMat(rng, 5, 3)
	got, outIdx, err := nn.LLMInt8MatMul(x, w, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(outIdx) != 5 {
		t.Errorf("threshold 0 flagged %d/5 dims as outliers, want all", len(outIdx))
	}
	want := exactMatMul(t, x, w)
	if e := froErr(got, want); e > 1e-9 {
		t.Errorf("all-fp16 decomposition error %g, want 0", e)
	}
}

// §V2 (the defining benefit): with a systematic outlier feature dimension, keeping it
// in high precision (LLM.int8, α=6) is DRAMATICALLY more accurate than quantizing
// everything to int8 (which lets the outlier's absmax crush the other values).
func TestLLMInt8BeatsNaiveWithOutliers(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 141))
	const tokens, cin, cout = 8, 6, 4
	x, w := randMat(rng, tokens, cin), randMat(rng, cin, cout)
	for tt := 0; tt < tokens; tt++ { // make column 2 a large-magnitude outlier feature
		x.SetF64(60+10*rng.NormFloat64(), tt, 2)
	}
	exact := exactMatMul(t, x, w)
	mixed, outIdx, _ := nn.LLMInt8MatMul(x, w, 6.0)    // outlier col in fp16
	naive, _, _ := nn.LLMInt8MatMul(x, w, math.Inf(1)) // everything int8
	if len(outIdx) == 0 {
		t.Fatal("expected the ×60 column to be flagged as an outlier")
	}
	mixedErr, naiveErr := froErr(mixed, exact), froErr(naive, exact)
	if mixedErr >= 0.1*naiveErr {
		t.Errorf("mixed-precision error %g not ≪ naive-int8 error %g", mixedErr, naiveErr)
	}
}

// §V2: with no outliers, pure vector-wise int8 approximates the matmul with small
// relative error.
func TestLLMInt8NoOutlierApprox(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 142))
	x, w := randMat(rng, 6, 8), randMat(rng, 8, 5)
	got, outIdx, _ := nn.LLMInt8MatMul(x, w, 6.0) // random N(0,1) ⇒ no |value|≥6
	if len(outIdx) != 0 {
		t.Fatalf("unexpected outliers %v in N(0,1) data", outIdx)
	}
	exact := exactMatMul(t, x, w)
	if rel := froErr(got, exact) / froErr(exact, tensor.New(tensor.F64, exact.Shape())); rel > 0.05 {
		t.Errorf("int8 relative error %g, want < 5%%", rel)
	}
}

func TestLLMInt8Errors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, _, err := nn.LLMInt8MatMul(r(3, 4), r(5, 2), 6); err == nil {
		t.Error("contraction mismatch: want error")
	}
	if _, _, err := nn.LLMInt8MatMul(r(3, 4, 1), r(4, 2), 6); err == nil {
		t.Error("rank-3: want error")
	}
}

func ExampleLLMInt8MatMul() {
	// column 1 is a ×50 outlier feature: LLM.int8() keeps it in high precision (so the
	// result is accurate) and flags it.
	x := toT([][]float64{{1, 50, 2}, {2, 50, 1}})
	w := toT([][]float64{{1, 0}, {0, 1}, {1, 1}})
	got, outIdx, _ := nn.LLMInt8MatMul(x, w, 6.0)
	// exact X·W row0 = [1+2, 50+2] = [3, 52].
	fmt.Printf("outliers=%v row0=[%.1f %.1f]\n", outIdx, got.AtF64(0, 0), got.AtF64(0, 1))
	// Output:
	// outliers=[1] row0=[3.0 52.0]
}
