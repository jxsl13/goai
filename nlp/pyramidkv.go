package nlp

import "math"

// PyramidKV per-layer KV-cache budget allocation (Cai, Zhang, Yue, Ren, Zhang, Wang, Yin,
// Zhang, Zhang, Liu, Yang, Wang, Wang & Wei 2024, "PyramidKV: Dynamic KV Cache Compression
// based on Pyramidal Information Funneling", arXiv:2406.02069). Attention is broad in the
// lower transformer layers but "funnels" onto a few tokens in the higher layers, so a uniform
// per-layer KV budget over-allocates to the top. PyramidKV instead spreads the SAME total
// budget as a decreasing PYRAMID: more cache in the lower layers, less in the higher ones,
// then selects which tokens to keep per layer with a SnapKV-style score at that layer's
// budget (see SnapKVKeep).

// PyramidKVBudgets returns the per-layer KV budgets: a linearly decreasing sequence from a
// maximum at layer 0 (bottom) down to minBudget at the top layer, with the total conserved
// (Σ budgets == numLayers·(totalBudget/numLayers) == totalBudget). The maximum is
// b_max = 2·avg − minBudget so the mean of the endpoints equals the average budget avg =
// totalBudget/numLayers. minBudget is clamped to [0, avg] (minBudget = avg gives a uniform
// allocation). Integer rounding residue is folded into the bottom (largest) layer so the sum
// is exact.
func PyramidKVBudgets(numLayers, totalBudget, minBudget int) []int {
	if numLayers <= 0 {
		return nil
	}
	if numLayers == 1 {
		return []int{totalBudget}
	}
	avg := float64(totalBudget) / float64(numLayers)
	if minBudget < 0 {
		minBudget = 0
	}
	if float64(minBudget) > avg {
		minBudget = int(math.Floor(avg)) // keep it a real (non-increasing) pyramid
	}
	bMax := 2*avg - float64(minBudget) // (bMax + minBudget)/2 = avg conserves the total
	out := make([]int, numLayers)
	sum := 0
	for l := range out {
		b := bMax - (bMax-float64(minBudget))*float64(l)/float64(numLayers-1)
		out[l] = int(math.Round(b))
		sum += out[l]
	}
	out[0] += totalBudget - sum // fold rounding residue into the bottom layer, exact total
	return out
}
