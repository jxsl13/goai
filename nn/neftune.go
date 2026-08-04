package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// NEFTune adds NEFTune noise (Jain et al. 2023, "NEFTune: Noisy Embeddings Improve
// Instruction Finetuning", arXiv:2310.05914, §R96) to a token embedding for
// noisy-embedding fine-tuning: given emb [L, d] (L tokens, d embedding dim), it returns
//
//	emb + (α/√(L·d))·ε ,  ε ~ Uniform(−1, 1) i.i.d. per entry
//
// The uniform magnitude α/√(L·d) scales the perturbation to the embedding size; α is
// the noise scale (the paper sweeps {5,10,15}). Apply it ONLY during training, between
// the embedding and the transformer blocks (see the models' ForwardFromEmbed), and
// never at inference — the added regularization noise would corrupt generation. The
// noise is a constant additive perturbation, so gradients flow normally into the
// embeddings; a bigger, more robust representation is learned. rng makes it
// deterministic.
func NEFTune(ctx *backend.Context, emb *tensor.Tensor, alpha float64, rng *rand.Rand) (*tensor.Tensor, error) {
	if emb.Ndim() != 2 {
		return nil, fmt.Errorf("nn: NEFTune needs a [seq,dim] embedding, got %v", emb.Shape())
	}
	l, d := emb.Shape()[0], emb.Shape()[1]
	mag := alpha / math.Sqrt(float64(l*d))
	noise := tensor.New(emb.Dtype(), emb.Shape())
	n := emb.Numel()
	// noise is freshly tensor.New'd, hence contiguous: its flat index IS its storage
	// index, so a typed slice walk replaces the per-element Unravel+SetF64 dispatch
	// (§T919 fast path). The rng is drawn once per element in flat order in EVERY
	// branch, so the Uniform(−mag,+mag) sequence — and the noise tensor — is
	// bit-identical to the generic fallback (F16/BF16/strided).
	if nf := flatF64(noise); nf != nil {
		for i := range nf {
			nf[i] = (rng.Float64()*2 - 1) * mag // Uniform(−mag, +mag)
		}
	} else if nf := flatF32(noise); nf != nil {
		for i := range nf {
			nf[i] = float32((rng.Float64()*2 - 1) * mag)
		}
	} else {
		for i := range n {
			idx := tensor.Unravel(i, emb.Shape())
			noise.SetF64((rng.Float64()*2-1)*mag, idx...) // Uniform(−mag, +mag)
		}
	}
	//perfscan:ignore PS3038 resource-only, time flat (alloc-only)
	out, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{emb, noise}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
