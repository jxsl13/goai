package nlp

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// EAGLE speculative decoding (Li, Wei, Zhang & Zhang 2024, "EAGLE: Speculative
// Sampling Requires Rethinking Feature Uncertainty", arXiv:2401.15077). EAGLE's
// key observation is that the base model's FEATURE sequence (the final hidden
// states, ForwardHidden) is far more regular — and thus more predictable — than
// the token sequence. So instead of a separate small draft MODEL (classic
// speculative decoding, speculative.go) or K parallel token heads over one hidden
// state (Medusa, medusa.go), EAGLE drafts with ONE small AUTOREGRESSIVE head that
// operates at the feature level: given the base's hidden state f_t and the
// embedding of the already-known next token t+1, it predicts the next hidden
// state f_{t+1}; feeding predictions back yields a whole chain of future features.
// The base model's own frozen LM head turns each drafted feature into draft-token
// logits, and one base forward pass verifies the drafted tokens — accepting the
// longest correct prefix keeps decoding LOSSLESS (the greedy output is
// token-for-token what plain greedy base decoding would produce).

// eagleCEWeight scales the token cross-entropy term of the EAGLE training loss
// relative to the feature-regression term (paper §4.1: L = L_reg + w_cls·L_cls
// with w_cls = 0.1 — the classification term carries a much larger raw scale
// over real vocabularies, so it is down-weighted).
const eagleCEWeight = 0.1

// EagleHead is the EAGLE-1 draft module: a single small network that maps the
// pair (current feature f_t, embedding of the next token t+1) to a prediction of
// the next feature f_{t+1}. Concretely: a fusion linear [2·dim → dim] over the
// concatenation [embedding ; feature], one pre-LN residual MLP block (the "one
// transformer block" of the paper in its position-wise form — the attention
// sublayer over past drafted features is a follow-up), and a final LayerNorm so
// predicted features live on the same normalized scale as the base's
// ForwardHidden output (which ends in the base's final LayerNorm). The head owns
// NO embedding table and NO LM head — it reuses the frozen base model's TokEmb
// and Head, which is what makes it so much smaller than a classic draft model.
type EagleHead struct {
	Dim   int // base model feature width (must equal GPTConfig.Dim)
	Vocab int // base model vocabulary size (must equal GPTConfig.Vocab)

	Wf *tensor.Tensor // fusion weight [2·dim, dim] over [embedding ; feature]
	Bf *tensor.Tensor // fusion bias [dim]

	LN *nn.LayerNorm  // pre-MLP LayerNorm of the residual block
	W1 *tensor.Tensor // MLP up-projection [dim, mult·dim]
	B1 *tensor.Tensor // MLP up bias [mult·dim]
	W2 *tensor.Tensor // MLP down-projection [mult·dim, dim]
	B2 *tensor.Tensor // MLP down bias [dim]

	LNf *nn.LayerNorm // final LayerNorm onto the base's feature scale
}

// EagleHeadOption configures NewEagleHead (functional options, §C12).
type EagleHeadOption func(*eagleHeadCfg)

type eagleHeadCfg struct{ mult int }

// WithEagleFFNMult sets the expansion factor of the head's MLP block: the hidden
// layer has mult·dim units (default 4, the standard transformer FFN ratio).
//
// In plain terms: how much scratch capacity the little draft network gets.
// Boundary behavior: must be ≥ 1 (NewEagleHead rejects smaller values); larger
// values buy drafting accuracy at the cost of a bigger head.
func WithEagleFFNMult(mult int) EagleHeadOption {
	return func(c *eagleHeadCfg) { c.mult = mult }
}

// NewEagleHead builds an EAGLE draft head for a base model of feature width dim
// and vocabulary vocab, with a small deterministic random init (seed). The head
// starts untrained: EagleGenerate is already lossless with it (the base verifies
// every token), it just accepts nothing yet — train it with EagleLoss on a
// frozen base first.
func NewEagleHead(dtype tensor.Dtype, dim, vocab int, seed uint64, opts ...EagleHeadOption) (*EagleHead, error) {
	if dim <= 0 || vocab <= 0 {
		return nil, fmt.Errorf("nlp: NewEagleHead needs dim, vocab > 0 (got %d, %d)", dim, vocab)
	}
	cfg := eagleHeadCfg{mult: 4}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.mult < 1 {
		return nil, fmt.Errorf("nlp: NewEagleHead FFN mult must be ≥ 1 (got %d)", cfg.mult)
	}
	rng := rand.New(rand.NewPCG(seed, 0xea91e))
	randn := func(shape ...int) *tensor.Tensor {
		w := tensor.New(dtype, tensor.Shape(shape))
		for i := range w.Numel() {
			w.SetF64(rng.NormFloat64()*0.02, tensor.Unravel(i, w.Shape())...)
		}
		return w
	}
	h := cfg.mult * dim
	return &EagleHead{
		Dim: dim, Vocab: vocab,
		Wf: randn(2*dim, dim), Bf: tensor.New(dtype, tensor.Shape{dim}),
		LN: nn.NewLayerNorm(dtype, dim),
		W1: randn(dim, h), B1: tensor.New(dtype, tensor.Shape{h}),
		W2: randn(h, dim), B2: tensor.New(dtype, tensor.Shape{dim}),
		LNf: nn.NewLayerNorm(dtype, dim),
	}, nil
}

// Params returns the head's trainable tensors for optimizers. The base model is
// NOT included — EAGLE trains only this small head against a frozen base.
func (e *EagleHead) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		e.Wf, e.Bf,
		e.LN.Gamma, e.LN.Beta,
		e.W1, e.B1, e.W2, e.B2,
		e.LNf.Gamma, e.LNf.Beta,
	}
}

// Predict maps features feats[n,dim] and next-token embeddings embs[n,dim] to
// the predicted next features [n,dim], row-wise: row t is the head's estimate of
// f_{t+1} from (f_t, embed(token_{t+1})). Differentiable — EAGLE training is a
// tape over this call (plus the frozen base Head) with ForwardHidden output as
// constant input, exactly like MedusaHeads.Logits. In a generation loop the same
// call runs one row at a time, feeding each predicted feature back in.
func (e *EagleHead) Predict(ctx *backend.Context, feats, embs *tensor.Tensor) (*tensor.Tensor, error) {
	for name, t := range map[string]*tensor.Tensor{"feats": feats, "embs": embs} {
		if t == nil || len(t.Shape()) != 2 || t.Shape()[1] != e.Dim {
			return nil, fmt.Errorf("nlp: EagleHead.Predict %s must be [n,%d]", name, e.Dim)
		}
	}
	if feats.Shape()[0] != embs.Shape()[0] {
		return nil, fmt.Errorf("nlp: EagleHead.Predict row mismatch: feats %d vs embs %d",
			feats.Shape()[0], embs.Shape()[0])
	}
	// fusion: z = [emb ; feat]·Wf + Bf
	z, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, embs, feats)
	if err != nil {
		return nil, err
	}
	if z, err = exec1(ctx, backend.OpMatMul, nil, z, e.Wf); err != nil {
		return nil, err
	}
	if z, err = exec1(ctx, backend.OpAddBias, nil, z, e.Bf); err != nil {
		return nil, err
	}
	// pre-LN residual MLP block: z += W2·gelu(W1·LN(z)+B1)+B2
	h, err := e.LN.Forward(ctx, z)
	if err != nil {
		return nil, err
	}
	if h, err = exec1(ctx, backend.OpMatMul, nil, h, e.W1); err != nil {
		return nil, err
	}
	if h, err = exec1(ctx, backend.OpAddBias, nil, h, e.B1); err != nil {
		return nil, err
	}
	if h, err = exec1(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	if h, err = exec1(ctx, backend.OpMatMul, nil, h, e.W2); err != nil {
		return nil, err
	}
	if h, err = exec1(ctx, backend.OpAddBias, nil, h, e.B2); err != nil {
		return nil, err
	}
	if z, err = exec1(ctx, backend.OpAdd, nil, z, h); err != nil {
		return nil, err
	}
	// final LayerNorm onto the base's (post-LNf) feature scale
	return e.LNf.Forward(ctx, z)
}

// EagleLoss is the EAGLE training loss for one sequence on a FROZEN base:
// feature regression + down-weighted token classification (paper §4.1,
// L = L_reg + 0.1·L_cls; MSE stands in for the paper's Smooth L1). tokens is the
// training sequence (≥ 3 tokens) and hidden the base's ForwardHidden(tokens)
// [len(tokens),dim] computed OUTSIDE the tape (the frozen-base recipe shared
// with MedusaHeads training). For every position t the head predicts f̂_{t+1}
// from (f_t, embed(token_{t+1})); the loss is
//
//	MSE(f̂_{t+1}, f_{t+1})  +  0.1·CE(f̂_{t+1}·Head, token_{t+2})
//
// where TokEmb and Head are the base's frozen tensors (they sit on the tape but
// only head.Params() are stepped). Run under a recording context to train.
func EagleLoss(ctx *backend.Context, head *EagleHead, base *GPT, tokens []int, hidden *tensor.Tensor) (*tensor.Tensor, error) {
	if head == nil || base == nil {
		return nil, fmt.Errorf("nlp: EagleLoss needs a head and a base model")
	}
	if head.Dim != base.Config.Dim || head.Vocab != base.Config.Vocab {
		return nil, fmt.Errorf("nlp: EagleLoss head geometry (dim %d, vocab %d) does not match base (dim %d, vocab %d)",
			head.Dim, head.Vocab, base.Config.Dim, base.Config.Vocab)
	}
	n := len(tokens)
	if n < 3 {
		return nil, fmt.Errorf("nlp: EagleLoss needs ≥ 3 tokens (got %d)", n)
	}
	if hidden == nil || len(hidden.Shape()) != 2 || hidden.Shape()[0] != n || hidden.Shape()[1] != head.Dim {
		return nil, fmt.Errorf("nlp: EagleLoss hidden must be [%d,%d] (base.ForwardHidden of tokens)", n, head.Dim)
	}
	// draft inputs: features f_0..f_{n-2} and embeddings of tokens 1..n-1
	feats, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: 0, End: n - 1}, hidden)
	if err != nil {
		return nil, err
	}
	idx := tensor.New(base.TokEmb.Dtype(), tensor.Shape{n - 1})
	for i := 0; i < n-1; i++ {
		idx.SetF64(float64(tokens[i+1]), i)
	}
	embs, err := exec1(ctx, backend.OpEmbed, nil, base.TokEmb, idx)
	if err != nil {
		return nil, err
	}
	pred, err := head.Predict(ctx, feats, embs) // row t ≈ f_{t+1}
	if err != nil {
		return nil, err
	}
	// feature regression against the true next features f_1..f_{n-1}
	next, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: 1, End: n}, hidden)
	if err != nil {
		return nil, err
	}
	lossF, err := nn.MSE(ctx, pred, next)
	if err != nil {
		return nil, err
	}
	// token classification through the frozen base Head: f̂_{t+1} → token_{t+2}
	predT, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: 0, End: n - 2}, pred)
	if err != nil {
		return nil, err
	}
	logits, err := exec1(ctx, backend.OpMatMul, nil, predT, base.Head)
	if err != nil {
		return nil, err
	}
	targets := tensor.New(tensor.F32, tensor.Shape{n - 2})
	for i := 0; i < n-2; i++ {
		targets.SetF64(float64(tokens[i+2]), i)
	}
	lossT, err := nn.CrossEntropy(ctx, logits, targets)
	if err != nil {
		return nil, err
	}
	w := tensor.New(lossT.Dtype(), tensor.Shape{})
	w.SetF64(eagleCEWeight)
	lossT, err = exec1(ctx, backend.OpMul, nil, lossT, w)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpAdd, nil, lossF, lossT)
}

// eagleEmbed gathers the base token embedding row [1,dim] for one token (no
// position term — EAGLE conditions the draft on the token identity only).
func eagleEmbed(ctx *backend.Context, table *tensor.Tensor, token int) (*tensor.Tensor, error) {
	idx := tensor.New(table.Dtype(), tensor.Shape{1})
	idx.SetF64(float64(token), 0)
	return exec1(ctx, backend.OpEmbed, nil, table, idx)
}

// EagleGenerate generates up to maxNew tokens after prompt with EAGLE
// speculative decoding (linear draft chain; the paper's candidate TREE over
// top-k draft tokens is the follow-up — it plugs the same verification into
// MedusaTreeMask-style tree attention). Each round:
//
//  1. one base ForwardHidden over the current sequence yields the features; the
//     last feature through the base Head gives the exact next greedy token;
//  2. the head drafts draftLen further tokens autoregressively at the FEATURE
//     level — predict f̂ from (feature, embed(token)), map f̂ through the frozen
//     base Head, take the argmax, feed both back;
//  3. ONE base forward over sequence+candidates verifies: the longest candidate
//     prefix that matches the base's own greedy tokens is accepted, plus one
//     bonus/correction token read from the same verification pass.
//
// Every emitted token equals the base's greedy choice given its true prefix, so
// the output is EXACTLY plain greedy decoding for ANY head weights — LOSSLESS;
// the head only controls how many tokens each round advances. draftLen < 1
// selects 4. Returns prompt+generated and the draft accept stats (the exact
// first token and the bonus token of a round are not counted as proposals).
// Like MedusaGenerate this reference version recomputes full forwards each
// round (analysis-scale); the KV-cached variant is a perf follow-up.
func EagleGenerate(base *GPT, head *EagleHead, prompt []int, maxNew, draftLen int) ([]int, SpecStats, error) {
	var stats SpecStats
	if base == nil || head == nil {
		return nil, stats, fmt.Errorf("nlp: EagleGenerate needs a base model and a draft head")
	}
	if head.Dim != base.Config.Dim || head.Vocab != base.Config.Vocab {
		return nil, stats, fmt.Errorf("nlp: EagleGenerate head geometry (dim %d, vocab %d) does not match base (dim %d, vocab %d)",
			head.Dim, head.Vocab, base.Config.Dim, base.Config.Vocab)
	}
	if len(prompt) == 0 {
		return nil, stats, fmt.Errorf("nlp: EagleGenerate needs a non-empty prompt")
	}
	if draftLen < 1 {
		draftLen = 4
	}
	ctx := backend.NewContext()
	out := append([]int(nil), prompt...)
	for len(out)-len(prompt) < maxNew && len(out) < base.Config.Ctx {
		room := min(maxNew-(len(out)-len(prompt)), base.Config.Ctx-len(out))
		// 1. base features over the current sequence; exact greedy first candidate.
		hidden, err := base.ForwardHidden(ctx, out)
		if err != nil {
			return nil, stats, err
		}
		f, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: len(out) - 1, End: len(out)}, hidden)
		if err != nil {
			return nil, stats, err
		}
		lg, err := exec1(ctx, backend.OpMatMul, nil, f, base.Head)
		if err != nil {
			return nil, stats, err
		}
		tok := argmax(rowAt(lg, 0))
		cand := []int{tok}
		// 2. feature-level autoregressive draft chain.
		for len(cand) < min(1+draftLen, room) {
			emb, err := eagleEmbed(ctx, base.TokEmb, tok)
			if err != nil {
				return nil, stats, err
			}
			if f, err = head.Predict(ctx, f, emb); err != nil {
				return nil, stats, err
			}
			if lg, err = exec1(ctx, backend.OpMatMul, nil, f, base.Head); err != nil {
				return nil, stats, err
			}
			tok = argmax(rowAt(lg, 0))
			cand = append(cand, tok)
		}
		stats.Proposed += len(cand) - 1
		// 3. one base forward over sequence+candidates verifies the chain.
		ver := append(append([]int(nil), out...), cand...)
		vlog, err := base.Forward(ctx, ver)
		if err != nil {
			return nil, stats, err
		}
		a := 0 // accepted candidates: longest prefix matching the base's greedy walk
		for j := range cand {
			if cand[j] != argmax(rowAt(vlog, len(out)+j-1)) {
				break
			}
			a++
		}
		if a > 0 {
			stats.Accepted += a - 1 // cand[0] is exact, not a speculative accept
		}
		emit := append([]int(nil), cand[:a]...)
		if a < room {
			// bonus/correction: the base's greedy token after the accepted prefix,
			// read from the verification pass — a round always advances ≥ 1 token.
			emit = append(emit, argmax(rowAt(vlog, len(out)+a-1)))
		}
		out = append(out, emit...)
	}
	return out, stats, nil
}
