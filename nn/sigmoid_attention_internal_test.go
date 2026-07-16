package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// sigExec runs a single-output op through a fresh context (internal test helper).
func sigExec(t *testing.T, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	o, err := backend.Execute(backend.NewContext(), op, in, at)
	if err != nil {
		t.Fatal(err)
	}
	return o[0]
}

func sigScores(vals [][]float64) *tensor.Tensor {
	m := tensor.New(tensor.F64, tensor.Shape{len(vals), len(vals[0])})
	for i := range vals {
		for j := range vals[i] {
			m.SetF64(vals[i][j], i, j)
		}
	}
	return m
}

// sigWeightsMust runs sigmoidWeights(scores, [1,1]=b) or fails the test.
func sigWeightsMust(t *testing.T, scores *tensor.Tensor, b float64) *tensor.Tensor {
	t.Helper()
	w, err := sigmoidWeights(backend.NewContext(), scores, scalarTensor(tensor.F64, b))
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// §V16 (1) SINGLE-KEY / small-case CORRECTNESS — the anchor. With ONE key
// position the sigmoid-attention output is O = σ(q·k/√d + b)·v exactly, no
// normalization; and a known multi-key (Q,K,V,b) case must equal the hand
// computation σ(scores+b)·V to 1e-10.
func TestSigmoidSingleKeyAndSmallCase(t *testing.T) {
	// (a) single key: scores is [1,1] = [s]; weight is a lone σ(s+b), output s·v.
	const s, b = 0.7, -0.3
	scores := sigScores([][]float64{{s}})
	w := sigWeightsMust(t, scores, b)
	wantW := 1.0 / (1.0 + math.Exp(-(s + b)))
	if d := math.Abs(w.AtF64(0, 0) - wantW); d > 1e-10 {
		t.Errorf("single-key weight σ(s+b): got %.15g want %.15g (diff %.3g)", w.AtF64(0, 0), wantW, d)
	}
	v := sigScores([][]float64{{2.0, -5.0, 0.25}}) // one value row [1,3]
	o := sigExec(t, backend.OpMatMul, nil, w, v)   // [1,3]
	for j, vv := range []float64{2.0, -5.0, 0.25} {
		want := wantW * vv
		if d := math.Abs(o.AtF64(0, j) - want); d > 1e-10 {
			t.Errorf("single-key O[%d]: got %.15g want %.15g (diff %.3g)", j, o.AtF64(0, j), want, d)
		}
	}

	// (b) general small case: known scores + bias, two queries over three keys,
	// V [3,2]. O = σ(scores+b)·V, computed independently by hand.
	scores2 := sigScores([][]float64{
		{0.5, -1.0, 2.0},
		{-2.0, 0.0, 0.75},
	})
	const b2 = -math.Ln2 // arbitrary finite bias (−log 2)
	w2 := sigWeightsMust(t, scores2, b2)
	vmat := sigScores([][]float64{
		{1.0, -1.0},
		{0.5, 2.0},
		{-2.0, 0.5},
	})
	o2 := sigExec(t, backend.OpMatMul, nil, w2, vmat) // [2,2]
	rows := [][]float64{{0.5, -1.0, 2.0}, {-2.0, 0.0, 0.75}}
	vv := [][]float64{{1.0, -1.0}, {0.5, 2.0}, {-2.0, 0.5}}
	for i := range 2 {
		for c := range 2 {
			var want float64
			for j := range 3 {
				sig := 1.0 / (1.0 + math.Exp(-(rows[i][j] + b2)))
				want += sig * vv[j][c]
			}
			if d := math.Abs(o2.AtF64(i, c) - want); d > 1e-10 {
				t.Errorf("small-case O[%d,%d]: got %.15g want %.15g (diff %.3g)", i, c, o2.AtF64(i, c), want, d)
			}
		}
	}
}

// §V16 (2) BIAS LIMITS — the bracketing anchors. b → −∞ ⇒ every weight σ → 0 ⇒
// A = 0 ⇒ O = 0 (bit-exact). b → +∞ (no mask) ⇒ every weight σ → 1 ⇒ O = Σ_j v_j
// (column sums of V) to 1e-10. These two limits bracket the mechanism.
func TestSigmoidBiasLimits(t *testing.T) {
	scores := sigScores([][]float64{
		{0.5, -1.0, 2.0, 0.0},
		{-3.0, 1.5, 0.25, -0.5},
	})
	v := sigScores([][]float64{
		{1.0, -2.0},
		{0.5, 3.0},
		{-1.5, 1.0},
		{2.0, -0.5},
	})

	// b → −∞: weights all 0, output exactly 0.
	wLo := sigWeightsMust(t, scores, -1e30)
	for i := range scores.Shape()[0] {
		for j := range scores.Shape()[1] {
			if g := wLo.AtF64(i, j); g != 0 {
				t.Errorf("b→−∞ weight[%d,%d] = %g, want exactly 0", i, j, g)
			}
		}
	}
	oLo := sigExec(t, backend.OpMatMul, nil, wLo, v)
	for i := range oLo.Shape()[0] {
		for j := range oLo.Shape()[1] {
			if g := oLo.AtF64(i, j); g != 0 {
				t.Errorf("b→−∞ O[%d,%d] = %g, want bit-exact 0", i, j, g)
			}
		}
	}

	// b → +∞ (no mask): weights all 1, output = column sums of V.
	wHi := sigWeightsMust(t, scores, 1e30)
	for i := range scores.Shape()[0] {
		for j := range scores.Shape()[1] {
			if d := math.Abs(wHi.AtF64(i, j) - 1); d > 1e-10 {
				t.Errorf("b→+∞ weight[%d,%d] = %.15g, want 1 (diff %.3g)", i, j, wHi.AtF64(i, j), d)
			}
		}
	}
	oHi := sigExec(t, backend.OpMatMul, nil, wHi, v)
	nKeys := scores.Shape()[1]
	for c := range 2 {
		var colSum float64
		for j := range nKeys {
			colSum += v.AtF64(j, c)
		}
		for i := range oHi.Shape()[0] {
			if d := math.Abs(oHi.AtF64(i, c) - colSum); d > 1e-10 {
				t.Errorf("b→+∞ O[%d,%d] = %.15g, want Σ_j v = %.15g (diff %.3g)", i, c, oHi.AtF64(i, c), colSum, d)
			}
		}
	}
	t.Logf("bias limits: b→−∞ O all 0 (bit-exact); b→+∞ O = Σ_j v_j @1e-10")
}
