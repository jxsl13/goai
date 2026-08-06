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

// toppCandidateK is the device-TopK width used to resolve a pure-top-p nucleus on-device. 64 covers a
// realistic confident-decode nucleus (typically 10–40 tokens at top-p 0.9–0.95) while keeping the O(n·K)
// cu_topk_f32 cheap — K=256 measured ~6× slower (3.1ms vs 0.5ms at 128k), erasing the win. A nucleus that
// exceeds 64 overflows to the host fallback (cheaply predicted by the C/Zexp guard before the TopK runs).
const toppCandidateK = 64

// fastTopPSampler reports (sampler, C) when s is a penalty-free PURE top-p *nlp.Sampler — TopP∈(0,1)
// with every OTHER truncation filter off (no TopK/MinP/Typical/Eta/Epsilon/TopNSigma, no penalties,
// no XTC) and Temperature>0 — so a device TopK(C) superset lets SampleTopPFromCandidates reconstruct
// the exact full-vocab nucleus and draw, avoiding the whole-vocab logits D2H + host sort per token.
// The nucleus is Z-dependent (unlike top-k), so the caller must also supply the full-vocab softmax
// stats (maxLogit, Zexp); see SampleTopPFromCandidates. Returns (nil,0) for any richer sampler →
// the flexible full-vocab host fallback. Disabled by GOAI_CUDA_TOPK_SAMPLE=0 (shared switch).
func fastTopPSampler(s nlp.TokenSampler, vocab int) (*nlp.Sampler, int) {
	if os.Getenv("GOAI_CUDA_TOPK_SAMPLE") == "0" {
		return nil, 0
	}
	sp, ok := s.(*nlp.Sampler)
	if !ok {
		return nil, 0
	}
	// Pure top-p only: any other set-shrinking filter changes the kept set (and Typical/MinP have their
	// own Z-dependence), and penalties rewrite arbitrary logits — none are handled by the top-p nucleus
	// reconstruction, so require them all off. (Typical==1 is "off" per its field docs.)
	if sp.Temperature <= 0 || sp.TopP <= 0 || sp.TopP >= 1 {
		return nil, 0
	}
	if sp.TopK > 0 || sp.MinP != 0 || sp.Epsilon != 0 || sp.Eta != 0 || sp.TopNSigma > 0 {
		return nil, 0
	}
	if sp.Typical != 0 && sp.Typical != 1 {
		return nil, 0
	}
	if !(sp.RepeatPenalty == 0 || sp.RepeatPenalty == 1) || sp.FreqPenalty != 0 || sp.PresencePenalty != 0 || sp.DRYMultiplier != 0 {
		return nil, 0
	}
	if sp.XTCProbability != 0 {
		return nil, 0
	}
	c := toppCandidateK
	if c > vocab {
		c = vocab
	}
	return sp, c
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
