package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §V16 tier-1 (paper CONFIRMED, arXiv:2202.06417): score(v)=(1−α)·prob(v)−α·maxSim(v).
func TestContrastiveScoreFormula(t *testing.T) {
	probs := []float64{0.5, 0.3, 0.2}
	maxSim := []float64{0.9, 0.1, 0.4}
	const alpha = 0.6
	got := nlp.ContrastiveScore(probs, maxSim, alpha)
	for v := range got {
		want := (1-alpha)*probs[v] - alpha*maxSim[v]
		if math.Abs(got[v]-want) > 1e-12 {
			t.Errorf("score[%d]=%g want %g", v, got[v], want)
		}
	}
}

// §V16 tier-1 (α=0): reduces to greedy selection by model probability.
func TestContrastiveScoreGreedyAtZero(t *testing.T) {
	probs := []float64{0.2, 0.5, 0.3}
	maxSim := []float64{0.9, 0.9, 0.1} // ignored at α=0
	got := nlp.ContrastiveScore(probs, maxSim, 0)
	if argmax(got) != argmax(probs) {
		t.Errorf("α=0 argmax=%d != prob argmax=%d", argmax(got), argmax(probs))
	}
}

// §V16 tier-1 (α=1): pure degeneration penalty — pick the candidate LEAST similar to
// the context (lowest maxSim).
func TestContrastiveScoreAntiRepetitionAtOne(t *testing.T) {
	probs := []float64{0.6, 0.3, 0.1}
	maxSim := []float64{0.9, 0.2, 0.5}
	got := nlp.ContrastiveScore(probs, maxSim, 1)
	if argmax(got) != 1 { // token 1 has the lowest similarity
		t.Errorf("α=1 argmax=%d, want 1 (least similar)", argmax(got))
	}
}

// §V2 (the defining trade-off): a high-probability but repetitive candidate loses to a
// slightly-less-probable, non-repetitive one once α weights the degeneration penalty.
func TestContrastiveScoreTradeoff(t *testing.T) {
	// A: high prob but nearly repeats the context; B: lower prob but novel.
	probs := []float64{0.5, 0.4}
	maxSim := []float64{0.9, 0.1}
	if a := nlp.ContrastiveScore(probs, maxSim, 0); argmax(a) != 0 {
		t.Errorf("α=0 should pick the high-prob token 0, got %d", argmax(a))
	}
	if a := nlp.ContrastiveScore(probs, maxSim, 0.6); argmax(a) != 1 {
		t.Errorf("α=0.6 should pick the non-repetitive token 1, got %d", argmax(a))
	}
}

// §V16 tier-1: the degeneration penalty is the MAX cosine similarity to the context.
func TestMaxContextCosine(t *testing.T) {
	candReps := [][]float64{{1, 0}, {0, 1}}
	contextReps := [][]float64{{1, 1}, {1, 0}} // cand0 matches [1,0] exactly (cos 1)
	got := nlp.MaxContextCosine(candReps, contextReps)
	want := []float64{1, 1 / math.Sqrt2}
	for v := range want {
		if math.Abs(got[v]-want[v]) > 1e-9 {
			t.Errorf("maxSim[%d]=%g want %g", v, got[v], want[v])
		}
	}
	// no context ⇒ zero penalty.
	if z := nlp.MaxContextCosine(candReps, nil); z[0] != 0 || z[1] != 0 {
		t.Errorf("empty-context penalty=%v, want zeros", z)
	}
}

func TestContrastiveScorePanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("length mismatch: want panic")
		}
	}()
	nlp.ContrastiveScore([]float64{1, 2}, []float64{1}, 0.6)
}

func ExampleContrastiveScore() {
	// two candidates: token 0 is likely but repeats the context; token 1 is a bit less
	// likely but novel. With α=0.6 contrastive search prefers the novel token 1.
	probs := []float64{0.55, 0.45}
	maxSim := []float64{0.95, 0.10}
	scores := nlp.ContrastiveScore(probs, maxSim, 0.6)
	fmt.Println(argmax(scores))
	// Output:
	// 1
}
