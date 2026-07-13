package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// logSoftmaxRef is an independent stable log-softmax for the parity oracle.
func logSoftmaxRef(logits []float64) []float64 {
	mx := math.Inf(-1)
	for _, v := range logits {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(v - mx)
	}
	ls := math.Log(sum) + mx
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = v - ls
	}
	return out
}

func argmax(x []float64) int {
	best, bv := 0, math.Inf(-1)
	for i, v := range x {
		if v > bv {
			best, bv = i, v
		}
	}
	return best
}

// §V16 tier-1 (JSD definition): JSD([1,0],[0,1]) = ln2, symmetric, and 0 on equal
// distributions.
func TestDoLaJensenShannon(t *testing.T) {
	if d := nlp.JensenShannon([]float64{1, 0}, []float64{0, 1}); math.Abs(d-math.Ln2) > 1e-12 {
		t.Errorf("JSD([1,0],[0,1])=%g want ln2=%g", d, math.Ln2)
	}
	p, q := []float64{0.7, 0.2, 0.1}, []float64{0.1, 0.3, 0.6}
	if a, b := nlp.JensenShannon(p, q), nlp.JensenShannon(q, p); math.Abs(a-b) > 1e-12 {
		t.Errorf("JSD not symmetric: %g vs %g", a, b)
	}
	if d := nlp.JensenShannon(p, p); math.Abs(d) > 1e-12 {
		t.Errorf("JSD(p,p)=%g want 0", d)
	}
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2309.03883): the premature layer is the one
// MOST divergent (max JSD) from the mature distribution.
func TestDoLaSelectsMostDivergentLayer(t *testing.T) {
	mature := []float64{4, 0, 0} // peaked on token 0
	premature := [][]float64{
		{3.5, 0.5, 0}, // close to mature ⇒ small JSD
		{0, 0, 4},     // opposite peak ⇒ large JSD (should win)
	}
	_, sel := nlp.DoLaLogits(mature, premature, 0.1)
	if sel != 1 {
		t.Errorf("selected premature layer %d, want 1 (most divergent)", sel)
	}
	// single candidate ⇒ index 0.
	if _, s := nlp.DoLaLogits(mature, [][]float64{{1, 1, 1}}, 0.1); s != 0 {
		t.Errorf("single premature layer selected %d, want 0", s)
	}
}

// §V16 tier-1 (contrastive score): for plausible tokens F(v) = log q_N(v) − log q_M(v).
func TestDoLaContrastiveFormula(t *testing.T) {
	mature := []float64{2, 1, 0.5, -1}
	premature := [][]float64{{0, 0, 0, 3}}           // single layer ⇒ it is selected
	out, sel := nlp.DoLaLogits(mature, premature, 0) // alpha=0 ⇒ no masking, check all
	le := logSoftmaxRef(mature)
	lm := logSoftmaxRef(premature[sel])
	for v := range out {
		want := le[v] - lm[v]
		if math.Abs(out[v]-want) > 1e-12 {
			t.Errorf("out[%d]=%g want le−lm=%g", v, out[v], want)
		}
	}
}

// §V16 tier-1 (adaptive plausibility): tokens with mature prob < α·max are −∞.
func TestDoLaAdaptivePlausibility(t *testing.T) {
	mature := []float64{5, 0, -10} // softmax ≈ [0.993, 0.0067, ~0]; max≈0.993
	out, _ := nlp.DoLaLogits(mature, [][]float64{{0, 0, 0}}, 0.1)
	// token 0 (dominant) kept; tokens 1,2 (p ≪ 0.1·max) excluded.
	if math.IsInf(out[0], -1) {
		t.Error("dominant token 0 was excluded")
	}
	if !math.IsInf(out[1], -1) || !math.IsInf(out[2], -1) {
		t.Errorf("implausible tokens not masked: out=%v", out)
	}
}

// §V2: against a uniform premature layer the contrast only shifts by a constant, so
// the mature layer's argmax is preserved (DoLa amplifies, never inverts, the top token).
func TestDoLaUniformPrematurePreservesArgmax(t *testing.T) {
	mature := []float64{0.3, 2.5, 1.0, -0.2, 0.8}
	uniform := []float64{0, 0, 0, 0, 0} // softmax uniform ⇒ constant log-prob
	out, _ := nlp.DoLaLogits(mature, [][]float64{uniform}, 0.1)
	if argmax(out) != argmax(mature) {
		t.Errorf("DoLa argmax=%d != mature argmax=%d", argmax(out), argmax(mature))
	}
}

func TestDoLaPanics(t *testing.T) {
	assertPanic := func(name string, f func()) {
		defer func() {
			if recover() == nil {
				t.Errorf("%s: expected panic", name)
			}
		}()
		f()
	}
	assertPanic("no premature", func() { nlp.DoLaLogits([]float64{1, 2}, nil, 0.1) })
	assertPanic("length mismatch", func() { nlp.DoLaLogits([]float64{1, 2}, [][]float64{{1, 2, 3}}, 0.1) })
	assertPanic("JSD mismatch", func() { nlp.JensenShannon([]float64{1}, []float64{1, 0}) })
}

func ExampleDoLaLogits() {
	// mature layer favors token 1; the (only) premature layer favors token 0, so
	// contrasting sharpens token 1 as the factual choice.
	mature := []float64{1, 2, 0}
	premature := [][]float64{{3, 0, 0}}
	out, sel := nlp.DoLaLogits(mature, premature, 0.1)
	fmt.Printf("selected=%d argmax=%d\n", sel, argmax(out))
	// Output:
	// selected=0 argmax=1
}
