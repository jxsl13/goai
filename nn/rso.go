package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// Statistical Rejection Sampling Optimization (RSO; Liu, Zhao, Joshi, Khalman, Saleh, Peter J.
// Liu & Jialu Liu 2024, "Statistical Rejection Sampling Improves Preference Optimization", ICLR,
// arXiv:2309.06657). Direct preference methods (DPO §R48, SLiC §R214) train on pairs drawn from
// an arbitrary policy, but the preference model they fit is defined w.r.t. the OPTIMAL policy
//
//	π*(y|x) ∝ π_sft(y|x)·exp(r(x,y)/β)
//
// RSO closes that gap: it draws candidate responses from the SFT policy and uses statistical
// REJECTION SAMPLING (with proposal π_sft, target π*) to keep a subset distributed as π*, from
// which the preference pairs are then labeled. Rejection sampling accepts a candidate with
// probability π*/(M·π_sft); taking the tightest M over the candidate set gives the acceptance
// probability
//
//	p(y) = exp((r(x,y) − r_max)/β)
//
// where r_max is the largest reward among the candidates — so the best candidate is always kept
// (p=1) and lower-reward ones are dropped with exponentially decaying probability. β is the KL
// temperature (smaller β ⇒ sharper target ⇒ more aggressive rejection). These are pure reward
// statistics (no gradient); feed the surviving responses to DPO/SLiC.
//
// This implements the single-pass acceptance rule with r_max the STATIC batch maximum — the
// acceptance-probability form the paper states in §3.2, and a valid rejection sampler for π*
// (M = exp(r_max/β) bounds π*/π_sft over the batch). The paper's Algorithm 2 refines this by
// recomputing r_max over the not-yet-accepted subset as it iterates; that iterative variant is a
// parked follow-up.

// RSOAcceptProbs returns the RSO acceptance probability p_i = exp((r_i − max r)/β) for each
// candidate reward. The maximum-reward candidate gets probability 1; β must be positive.
func RSOAcceptProbs(rewards []float64, beta float64) ([]float64, error) {
	if len(rewards) == 0 {
		return nil, fmt.Errorf("nn: RSOAcceptProbs needs at least one reward")
	}
	if beta <= 0 {
		return nil, fmt.Errorf("nn: RSOAcceptProbs beta must be positive, got %g", beta)
	}
	rMax := rewards[0]
	for _, r := range rewards[1:] {
		if r > rMax {
			rMax = r
		}
	}
	p := make([]float64, len(rewards))
	for i, r := range rewards {
		p[i] = math.Exp((r - rMax) / beta) // r ≤ rMax ⇒ p ∈ (0,1]
	}
	return p, nil
}

// RSOAccept runs one pass of RSO rejection sampling over the candidate rewards: it draws u∼U(0,1)
// per candidate and accepts candidate i iff u < exp((r_i − max r)/β). It returns the accepted
// indices (ascending); the surviving set is distributed approximately as the optimal policy π*.
func RSOAccept(rewards []float64, beta float64, rng *rand.Rand) ([]int, error) {
	p, err := RSOAcceptProbs(rewards, beta)
	if err != nil {
		return nil, err
	}
	var kept []int
	for i := range rewards {
		if rng.Float64() < p[i] {
			kept = append(kept, i)
		}
	}
	return kept, nil
}
