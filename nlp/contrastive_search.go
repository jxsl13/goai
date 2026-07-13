package nlp

import (
	"fmt"
	"math"
)

// ContrastiveScore implements the token ranking of contrastive search (Su, Lan, Wang,
// Yogatama, Kong & Collier 2022, "A Contrastive Framework for Neural Text Generation"
// / SimCTG, NeurIPS, arXiv:2202.06417). Ordinary likelihood decoding degenerates into
// repetitive text because the model keeps re-selecting tokens whose representations are
// nearly identical to the recent context. Contrastive search counters this by ranking
// the top-k candidates with a balance of model confidence and a DEGENERATION PENALTY:
//
//	score(v) = (1−α)·prob(v)  −  α·max_j cos(h_v, h_{x_j})
//
// The first term is the model's probability for candidate v; the second is the maximum
// cosine similarity of v's representation h_v to the representations of the already
// generated tokens (a high value means v would repeat the context). The candidate with
// the highest score is chosen. α ∈ [0,1] balances the two: α=0 is greedy decoding by
// probability, larger α more strongly discourages repetition (the paper uses α=0.6,
// top-k=4-8).
//
// probs holds the candidates' model probabilities and maxSim their per-candidate
// maximum cosine similarity to the context (compute it with MaxContextCosine). The
// returned scores are aligned with the candidates; the argmax is the selected token.
func ContrastiveScore(probs, maxSim []float64, alpha float64) []float64 {
	if len(probs) != len(maxSim) {
		panic(fmt.Sprintf("nlp: ContrastiveScore probs/maxSim length mismatch %d != %d", len(probs), len(maxSim)))
	}
	out := make([]float64, len(probs))
	for v := range probs {
		out[v] = (1-alpha)*probs[v] - alpha*maxSim[v]
	}
	return out
}

// MaxContextCosine returns, for each candidate representation, the maximum cosine
// similarity to any of the context (previously generated) token representations — the
// degeneration-penalty term of contrastive search. candReps is [numCandidates][dim],
// contextReps is [numContext][dim]. With no context every penalty is 0.
func MaxContextCosine(candReps, contextReps [][]float64) []float64 {
	out := make([]float64, len(candReps))
	for v, cand := range candReps {
		best := 0.0
		for _, ctx := range contextReps {
			if s := cosine(cand, ctx); s > best {
				best = s
			}
		}
		out[v] = best
	}
	return out
}

// cosine returns the cosine similarity a·b/(‖a‖‖b‖) ∈ [−1,1]; 0 if either is zero.
func cosine(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("nlp: cosine length mismatch %d != %d", len(a), len(b)))
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
