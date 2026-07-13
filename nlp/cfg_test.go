package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// cfgSoftmax turns logits into probabilities (stable). logSoftmaxRef/argmax are
// defined in dola_test.go (same test package).
func cfgSoftmax(logits []float64) []float64 {
	ls := logSoftmaxRef(logits)
	p := make([]float64, len(ls))
	for i, v := range ls {
		p[i] = math.Exp(v)
	}
	return p
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2306.17806): γ=1 recovers the ordinary
// conditional distribution.
func TestCFGGammaOneIsConditional(t *testing.T) {
	cond := []float64{1, 2, 0.5, -1}
	uncond := []float64{0, 1, 3, 0}
	guided := nlp.GuidedLogits(cond, uncond, 1)
	pg, pc := cfgSoftmax(guided), cfgSoftmax(cond)
	for v := range pg {
		if math.Abs(pg[v]-pc[v]) > 1e-9 {
			t.Errorf("γ=1 P[%d]=%g != conditional %g", v, pg[v], pc[v])
		}
	}
}

// §V16 tier-1 (Eq.): guided = log P(w) + γ·(log P(w|c) − log P(w)).
func TestCFGFormula(t *testing.T) {
	cond := []float64{2, 1, 0, -1}
	uncond := []float64{0, 0, 1, 2}
	const gamma = 1.5
	guided := nlp.GuidedLogits(cond, uncond, gamma)
	lc, lu := logSoftmaxRef(cond), logSoftmaxRef(uncond)
	for v := range guided {
		want := lu[v] + gamma*(lc[v]-lu[v])
		if math.Abs(guided[v]-want) > 1e-12 {
			t.Errorf("guided[%d]=%g want %g", v, guided[v], want)
		}
	}
}

// §V2 (defining property): γ>1 sharpens toward tokens the prompt favors and away from
// the unconditional/negative branch. With cond favoring token 0 and uncond (negative)
// favoring token 1, raising γ raises P(0) and lowers P(1).
func TestCFGAmplifiesPromptAdherence(t *testing.T) {
	cond := []float64{2, 0}   // prompt favors token 0
	uncond := []float64{0, 2} // negative prompt favors token 1
	p1 := cfgSoftmax(nlp.GuidedLogits(cond, uncond, 1))
	p2 := cfgSoftmax(nlp.GuidedLogits(cond, uncond, 2))
	if p2[0] <= p1[0] {
		t.Errorf("P(favored)=%g at γ=2 not > %g at γ=1", p2[0], p1[0])
	}
	if p2[1] >= p1[1] {
		t.Errorf("P(negative)=%g at γ=2 not < %g at γ=1 (should be pushed away)", p2[1], p1[1])
	}
	// γ=1 equals the plain conditional softmax.
	pc := cfgSoftmax(cond)
	if math.Abs(p1[0]-pc[0]) > 1e-9 {
		t.Errorf("γ=1 P(0)=%g != conditional %g", p1[0], pc[0])
	}
}

func TestCFGPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("length mismatch: want panic")
		}
	}()
	nlp.GuidedLogits([]float64{1, 2}, []float64{1, 2, 3}, 1.5)
}

func ExampleGuidedLogits() {
	// prompt strongly favors token 1; guidance γ=1.5 sharpens that preference, so the
	// greedy (argmax) choice is token 1.
	cond := []float64{0, 3, 1}
	uncond := []float64{1, 0, 1} // no/negative prompt
	guided := nlp.GuidedLogits(cond, uncond, 1.5)
	fmt.Println(argmax(guided))
	// Output:
	// 1
}
