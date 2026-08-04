package nn

import "sort"

// LossFreeBalancer implements the Auxiliary-Loss-Free Load Balancing strategy for
// Mixture-of-Experts (Wang, Gao, Chen, Xie & Dai 2024, "Auxiliary-Loss-Free Load
// Balancing Strategy for Mixture-of-Experts", arXiv:2408.15664; the routing used in
// DeepSeek-V3). Classic MoE balancing adds an auxiliary load-balance loss
// (MoEBalanceLoss), but its gradient fights the language-modeling gradient and hurts
// quality. Loss-Free Balancing instead keeps a per-expert BIAS b_i that is adjusted
// by a gradient-free control rule:
//
//   - Routing SELECTION uses the biased score s_i + b_i: the top-K experts by
//     (s_i + b_i) are chosen. The bias steers tokens toward under-used experts.
//   - The combination WEIGHT of a selected expert is its ORIGINAL affinity s_i — the
//     bias changes WHICH experts fire, never HOW MUCH they contribute, so it leaves
//     no bias term in the output and needs no gradient.
//   - After each batch the bias is nudged by the sign of each expert's load error:
//     b_i ← b_i + u·sign(c̄ − c_i), where c_i is the expert's token count, c̄ the mean,
//     and u a fixed update rate (default 0.001). Under-loaded experts (c_i < c̄) get a
//     higher bias, over-loaded ones a lower bias.
//
// The bias is a control variable, NOT a trained parameter — it never appears in the
// autograd graph.
type LossFreeBalancer struct {
	Bias []float64 // per-expert routing bias b_i (starts at 0)
	Rate float64   // bias update rate u (default 0.001)
}

// NewLossFreeBalancer builds a balancer for numExperts experts with bias update rate
// u (u ≤ 0 falls back to the paper's default 0.001).
func NewLossFreeBalancer(numExperts int, u float64) *LossFreeBalancer {
	if u <= 0 {
		u = 0.001
	}
	return &LossFreeBalancer{Bias: make([]float64, numExperts), Rate: u}
}

// Route selects the top-k experts for a token by the BIASED score (affinity_i +
// bias_i) and returns their indices (highest biased score first) together with their
// ORIGINAL affinities as the combination weights. The bias affects the selection
// only, not the weights. Ties break toward the lower expert index.
func (b *LossFreeBalancer) Route(affinity []float64, k int) (experts []int, weights []float64) {
	n := len(affinity)
	if k > n {
		k = n
	}
	if k <= 0 {
		return nil, nil
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	//perfscan:ignore PS3002,PS6009 radix on small numExperts loses to pdqsort | permutation-sort transform on small n loses
	sort.SliceStable(order, func(a, c int) bool {
		return affinity[order[a]]+b.Bias[order[a]] > affinity[order[c]]+b.Bias[order[c]]
	})
	experts = make([]int, k)
	weights = make([]float64, k)
	for i := range k {
		experts[i] = order[i]
		weights[i] = affinity[order[i]] // original affinity, NOT the biased score
	}
	return experts, weights
}

// Update adjusts the per-expert bias from the batch's per-expert token counts (loads)
// using the sign rule b_i ← b_i + u·sign(c̄ − c_i): raise the bias of under-loaded
// experts and lower it for over-loaded ones. loads must have one entry per expert.
func (b *LossFreeBalancer) Update(loads []int) {
	n := len(loads)
	if n == 0 {
		return
	}
	var total int
	for _, c := range loads {
		total += c
	}
	mean := float64(total) / float64(n)
	for i := 0; i < n && i < len(b.Bias); i++ {
		switch e := mean - float64(loads[i]); {
		case e > 0:
			b.Bias[i] += b.Rate // under-loaded → more attractive
		case e < 0:
			b.Bias[i] -= b.Rate // over-loaded → less attractive
		}
	}
}
