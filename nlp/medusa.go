package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Medusa tree-based speculative decoding primitives (Cai, Li, Geng, Peng, Chen, Lee &
// Dao 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding
// Heads", ICML 2024, arXiv:2401.10774). Medusa adds K extra decoding heads that predict
// the tokens at positions t+1…t+K in parallel; their top candidates are assembled into a
// TREE of continuations and verified in a SINGLE forward pass. Two mechanisms make this
// work and are provided here as pure functions: the TREE ATTENTION MASK (§2.3) and
// TYPICAL ACCEPTANCE (§2.3.1).

// Default typical-acceptance thresholds: a candidate is accepted when the original model's
// probability exceeds min(epsilon, delta·exp(−H)). These numeric defaults are from the
// reference implementation (FasterDecoding/Medusa medusa/model/utils.py:
// posterior_threshold=0.3, posterior_alpha=0.09); the paper itself gives no fixed values
// (it sweeps ε∈[0.01,0.25] and notes the Hewitt α=√ε relation, §3.3.2).
const (
	MedusaEpsilon = 0.3  // hard probability floor ε (posterior_threshold)
	MedusaDelta   = 0.09 // entropy-adaptive scale δ (posterior_alpha)
)

// MedusaTreeMask builds the tree attention mask and depth-based position ids for a
// candidate tree whose node i has parent parent[i] (a root has parent −1). Nodes must be
// given in an order where every parent precedes its children (parent[i] ∈ {−1} ∪ [0,i)).
// In Medusa's single-pass verification each candidate token may attend ONLY to its
// ancestors along the path back to the root (its true prefix), never to tokens in sibling
// branches — so one forward pass scores every root-to-node path independently. The
// returned mask is the additive pre-softmax bias: mask[i][j] = 0 when j is an ancestor of
// i or j==i, and −∞ otherwise (add it to the attention logits, as a causal mask). depth[i]
// is node i's depth in the tree (root 0), used as its position id. A linear chain
// reproduces the ordinary causal mask.
func MedusaTreeMask(parent []int) (mask *tensor.Tensor, depth []int) {
	n := len(parent)
	for i, p := range parent {
		if p < -1 || p >= i {
			panic(fmt.Sprintf("nlp: MedusaTreeMask parent[%d]=%d invalid (want −1 or an earlier node <%d)", i, p, i))
		}
	}
	mask = tensor.New(tensor.F64, tensor.Shape{n, n})
	depth = make([]int, n)
	neg := math.Inf(-1)
	for i := range n {
		for j := range n {
			mask.SetF64(neg, i, j)
		}
		mask.SetF64(0, i, i) // a token always attends to itself
		for a := parent[i]; a != -1; a = parent[a] {
			mask.SetF64(0, i, a) // …and to every ancestor on its path to the root
		}
		if parent[i] != -1 {
			depth[i] = depth[parent[i]] + 1
		}
	}
	return mask, depth
}

// TypicalAcceptance reports whether a Medusa-head candidate token should be accepted under
// the paper's TYPICAL ACCEPTANCE scheme (§2.3.1, after Hewitt et al. 2022 typical
// sampling): accept iff the ORIGINAL model's probability for the token exceeds
//
//	min(epsilon, delta·exp(−H(probs)))
//
// where H(probs) is the Shannon entropy (in nats) of the original model's next-token
// distribution at that position. The hard floor epsilon guarantees a minimum quality bar;
// the entropy term relaxes it where the model is uncertain (high H ⇒ lower threshold ⇒
// more candidates accepted) and tightens it where the model is confident. This replaces
// speculative decoding's rejection sampling: it is not distribution-exact but keeps only
// "typical" tokens, which the paper finds preserves generation quality while accepting
// more tokens per step. probs is the original model's distribution over the vocabulary;
// token indexes it. In a decode loop the first (greedy) token of a path is always accepted
// and decoding takes the longest prefix of a candidate path that each pass this test.
func TypicalAcceptance(probs []float64, token int, epsilon, delta float64) bool {
	if token < 0 || token >= len(probs) {
		panic(fmt.Sprintf("nlp: TypicalAcceptance token %d out of range [0,%d)", token, len(probs)))
	}
	thresh := math.Min(epsilon, delta*math.Exp(-shannonEntropy(probs)))
	return probs[token] > thresh
}

// MedusaHeads are K trainable decoding heads over the base model's final hidden
// state (§T443): head k is a linear map [dim,vocab] predicting the token at offset
// t+2+k from hidden state h_t (the base LM head already covers t+1). This is the
// simplest head form — the reference implementation uses a small residual block per
// head; a linear head is its first-order variant and trains with the library's own
// loop (frozen base: compute ForwardHidden outside the tape, then tape over Logits).
type MedusaHeads struct {
	W []*tensor.Tensor // head k: [dim, vocab]
}

// MedusaHeadsOption configures NewMedusaHeads (functional options, §C12).
type MedusaHeadsOption func(*medusaHeadsCfg)

type medusaHeadsCfg struct{ dtype tensor.Dtype }

// WithMedusaHeadsDtype sets the head weight dtype (default F32); match the base
// model's hidden-state dtype.
func WithMedusaHeadsDtype(d tensor.Dtype) MedusaHeadsOption {
	return func(c *medusaHeadsCfg) { c.dtype = d }
}

// NewMedusaHeads builds K linear heads with a small deterministic random init.
func NewMedusaHeads(k, dim, vocab int, seed uint64, opts ...MedusaHeadsOption) (*MedusaHeads, error) {
	if k <= 0 || dim <= 0 || vocab <= 0 {
		return nil, fmt.Errorf("nlp: NewMedusaHeads needs k, dim, vocab > 0 (got %d, %d, %d)", k, dim, vocab)
	}
	cfg := medusaHeadsCfg{dtype: tensor.F32}
	for _, o := range opts {
		o(&cfg)
	}
	rng := rand.New(rand.NewPCG(seed, 0x6d3d))
	m := &MedusaHeads{W: make([]*tensor.Tensor, k)}
	for h := range k {
		w := tensor.New(cfg.dtype, tensor.Shape{dim, vocab})
		for i := range w.Numel() {
			w.SetF64(rng.NormFloat64()*0.02, tensor.Unravel(i, w.Shape())...)
		}
		m.W[h] = w
	}
	return m, nil
}

// Params returns the head weights for optimizers.
func (m *MedusaHeads) Params() []*tensor.Tensor { return m.W }

// Logits projects the hidden states [seq,dim] through every head, returning K logit
// tensors [seq,vocab] (differentiable — training a frozen-base Medusa is a tape over
// this call with the base's ForwardHidden output as constant input).
func (m *MedusaHeads) Logits(ctx *backend.Context, hidden *tensor.Tensor) ([]*tensor.Tensor, error) {
	out := make([]*tensor.Tensor, len(m.W))
	for h, w := range m.W {
		l, err := exec1(ctx, backend.OpMatMul, nil, hidden, w)
		if err != nil {
			return nil, err
		}
		out[h] = l
	}
	return out, nil
}

// MedusaGenerate generates up to maxNew tokens with Medusa chain drafting (§T444,
// the decode loop over MedusaHeads + TypicalAcceptance; the single-path variant of
// the paper's candidate tree — tree assembly over top-k head candidates plugs into
// the same verification via MedusaTreeMask). Each round: (1) one forward computes
// the base's next greedy token x₁ (always emitted — the paper's guarantee that a
// round never stalls) and the K head proposals x₂…x_{K+1} from the same last hidden
// state; (2) ONE verification forward over seq+candidates scores every proposal
// under the base model, and the longest prefix passing TypicalAcceptance(ε, δ) is
// emitted. ε/δ ≤ 0 select the reference defaults (MedusaEpsilon/MedusaDelta).
// Typical acceptance is deliberately NOT distribution-exact (see TypicalAcceptance)
// — the output is greedy-anchored but may take plausible non-argmax tokens where
// the heads propose them. Greedy-only by construction (candidates are argmax
// proposals). Returns prompt+generated and the proposal/acceptance stats. Every
// round is O(seq) full forwards (analysis-scale, like DoLaDecode); the batched GPU
// variant belongs to llamagpu.
func MedusaGenerate(model *GPT, heads *MedusaHeads, prompt []int, maxNew int, epsilon, delta float64) ([]int, SpecStats, error) {
	var stats SpecStats
	if len(prompt) == 0 {
		return nil, stats, fmt.Errorf("nlp: MedusaGenerate needs a non-empty prompt")
	}
	if heads == nil || len(heads.W) == 0 {
		return nil, stats, fmt.Errorf("nlp: MedusaGenerate needs ≥1 Medusa head")
	}
	if epsilon <= 0 {
		epsilon = MedusaEpsilon
	}
	if delta <= 0 {
		delta = MedusaDelta
	}
	ctx := backend.NewContext()
	out := append([]int(nil), prompt...)
	for len(out)-len(prompt) < maxNew && len(out) < model.Config.Ctx {
		hidden, err := model.ForwardHidden(ctx, out)
		if err != nil {
			return nil, stats, err
		}
		last := hidden.Shape()[0] - 1
		lastRow, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{hidden},
			backend.SliceAttrs{Axis: 0, Start: last, End: last + 1})
		if err != nil {
			return nil, stats, err
		}
		mature, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{lastRow[0], model.Head}, nil)
		if err != nil {
			return nil, stats, err
		}
		cand := []int{argmax(rowAt(mature[0], 0))} // base greedy token, always emitted
		hl, err := heads.Logits(ctx, lastRow[0])
		if err != nil {
			return nil, stats, err
		}
		for _, l := range hl {
			cand = append(cand, argmax(rowAt(l, 0)))
		}
		room := min(maxNew-(len(out)-len(prompt)), model.Config.Ctx-len(out))
		if len(cand) > room {
			cand = cand[:room]
		}
		// one verification forward over seq+candidates scores every proposal under the base.
		ver := append(append([]int(nil), out...), cand...)
		vlog, err := model.Forward(ctx, ver)
		if err != nil {
			return nil, stats, err
		}
		n := 1 // accepted count: the greedy token
		stats.Proposed += len(cand) - 1
		for j := 1; j < len(cand); j++ {
			probs := softmaxProb(rowAt(vlog, len(out)+j-1))
			if !TypicalAcceptance(probs, cand[j], epsilon, delta) {
				break
			}
			n++
			stats.Accepted++
		}
		out = append(out, cand[:n]...)
	}
	return out, stats, nil
}

// shannonEntropy returns H(p) = −Σ p·ln p (in nats) over the positive entries of p.
func shannonEntropy(p []float64) float64 {
	var h float64
	for _, v := range p {
		if v > 0 {
			h -= v * math.Log(v)
		}
	}
	return h
}
