package llamagpu

import (
	"os"
	"sort"

	"github.com/jxsl13/goai/nlp"
)

// deviceTopKer is implemented by a device logits buffer that can return its k highest (index, value)
// pairs computed ON DEVICE — the CUDA logits buffer (cBuf, via cuda.DeviceF32.TopK). Metal/vulkan
// buffers do not implement it, so the on-device sampling fast-path type-asserts this and falls back to
// the full-vocab host path when it is absent. Keeping this an interface keeps decoder.go backend-neutral.
type deviceTopKer interface {
	// TopKN returns the k highest (index, value) pairs over the FIRST n elements (n = vocab; the decode
	// logits occupy [0,n) of a buffer that may be over-allocated for prefill/batch).
	TopKN(n, k int) ([]int32, []float32, error)
}

// fastTopKSampler reports (sampler, K) when s is a penalty-free *nlp.Sampler with TopK>0 whose selection
// is confined to the top-TopK — so a device TopK(K>TopK) superset lets the CPU draw EXACTLY on the K
// candidates (see sampleTopKCandidates), avoiding the whole-vocab logits D2H + CPU sort per token.
// Returns (nil, 0) for any other sampler (penalties on, TopK off/too large, a non-*Sampler impl, or
// GOAI_CUDA_TOPK_SAMPLE=0) → the flexible full-vocab host fallback.
func fastTopKSampler(s nlp.TokenSampler, vocab int) (*nlp.Sampler, int) {
	if os.Getenv("GOAI_CUDA_TOPK_SAMPLE") == "0" {
		return nil, 0
	}
	sp, ok := s.(*nlp.Sampler)
	if !ok {
		return nil, 0
	}
	// Penalties rewrite arbitrary tokens' logits from history, so a raw-logit top-k is not the effective
	// top-k → require them off (RepeatPenalty 0 or 1 are both no-ops). XTC/MinP/Typical/TopP are fine:
	// they only ever shrink the post-TopK set, which stays ⊆ the K candidates.
	if !(sp.RepeatPenalty == 0 || sp.RepeatPenalty == 1) || sp.FreqPenalty != 0 || sp.PresencePenalty != 0 || sp.DRYMultiplier != 0 {
		return nil, 0
	}
	if sp.TopK <= 0 || sp.TopK > 240 { // K = TopK+16 must stay within the cu_topk cap (256)
		return nil, 0
	}
	k := sp.TopK + 16 // margin so a tie at rank TopK can't push a real top-TopK token out of the K set
	if k > vocab {
		k = vocab
	}
	return sp, k
}

// sampleTopKCandidates draws exactly as the full-vocab sampler would, but over just the K device top-k
// candidates: it feeds them to sp.Sample in ASCENDING VOCAB-INDEX order so the multinomial's cumulative
// scan visits the selected tokens in the same order (and consumes the same rng) as the full-vocab path,
// then maps the chosen local index back to its vocab id. Exact for the penalty-free TopK case (K⊇top-k).
func sampleTopKCandidates(sp *nlp.Sampler, gi []int32, gv []float32) int {
	k := len(gi)
	order := make([]int, k)
	for j := range order {
		order[j] = j
	}
	sort.Slice(order, func(a, b int) bool { return gi[order[a]] < gi[order[b]] })
	vals := make([]float64, k)
	for j, o := range order {
		vals[j] = float64(gv[o])
	}
	local := sp.Sample(vals)
	if local < 0 || local >= k {
		local = 0 // degenerate (no positive mass) — the sampler already fell back to argmax on vals
	}
	return int(gi[order[local]])
}
