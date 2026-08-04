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
	if len(candReps) == 0 {
		return out
	}
	dim := len(candReps[0])
	// Pre-validate widths serially so the parallel body carries no panic path (a panic in a
	// worker goroutine would be unrecoverable). This preserves the original per-pair
	// length-mismatch panic while keeping the hot loop clean.
	for v, cand := range candReps {
		if len(cand) != dim {
			panic(fmt.Sprintf("nlp: MaxContextCosine candidate %d width %d != %d", v, len(cand), dim))
		}
	}
	for j, ctx := range contextReps {
		if len(ctx) != dim {
			panic(fmt.Sprintf("nlp: MaxContextCosine context %d width %d != %d", j, len(ctx), dim))
		}
	}
	// Each candidate's penalty is independent — it writes only out[v] and reads the shared
	// read-only contextReps — so the candidate loop fans out over GOMAXPROCS bit-identically to
	// the serial loop. Within a candidate its own squared norm ‖cand‖² is loop-invariant across
	// the context reps, so hoist it out of the inner scan instead of recomputing it for every
	// context token (a third of the inner-loop multiplies — the original recomputed dot, ‖cand‖²
	// and ‖ctx‖² together per pair). Bit-identical: dot and ‖ctx‖² keep the same ascending-i
	// summation order, and the divisor sqrt(na)·sqrt(nb) is the same product, so every compared
	// cosine value is unchanged.
	parallelChunks(len(candReps), len(contextReps)*dim, func(lo, hi int) {
		for v := lo; v < hi; v++ {
			cand := candReps[v]
			var na float64
			//perfscan:ignore PS3010 niche decode path, cosine dot << per-step forward
			for i := range cand {
				na += cand[i] * cand[i]
			}
			best := 0.0
			if na != 0 {
				sna := math.Sqrt(na)
				//perfscan:ignore PS3053 two scalar sqrts per cosine, niche path
				for _, ctx := range contextReps {
					var dot, nb float64
					//perfscan:ignore PS3010 cosine dot, niche decode path, low leverage
					for i := range cand {
						dot += cand[i] * ctx[i]
						nb += ctx[i] * ctx[i]
					}
					if nb == 0 {
						continue // zero-norm context: similarity is 0, best is already >= 0
					}
					if s := dot / (sna * math.Sqrt(nb)); s > best {
						best = s
					}
				}
			}
			out[v] = best
		}
	})
	return out
}
