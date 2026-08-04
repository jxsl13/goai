package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Coconut is a Chain-of-Continuous-Thought reasoner (Hao et al. 2024, "Training
// Large Language Models to Reason in a Continuous Latent Space",
// arXiv:2412.06769): a causal GPT backbone that reasons in LATENT space instead
// of emitting its intermediate steps as text. One "continuous thought" is one
// feedback step — run the backbone (ForwardHidden), take the LAST final hidden
// state, and append that vector DIRECTLY as the next input-embedding row (plus
// its position embedding), then re-run. K such latent steps happen between the
// prompt and the answer, so the sequential computation depth at inference is
// (K+1) forward passes — the answer is decoded from the final state through the
// ordinary LM head. All K+1 forwards run through one recording context, so
// gradients flow through every thought (backpropagation through time across the
// feedback loop; nothing is detached).
//
// Contrast with the two neighboring ideas: TEXT chain-of-thought emits discrete
// tokens — each step is squeezed through an argmax/sampling bottleneck into one
// vocabulary id before it can influence the next step — whereas a continuous
// thought hands the full hidden vector forward, unquantized. PAUSE tokens
// (Goyal et al. 2023, "Think before you speak") also insert K extra positions,
// but they are learned CONSTANTS: they buy parallel width, not sequential
// depth, because no state is fed back between forwards. The pause ablation is
// available here via WithCoconutPause so the two can be compared on equal
// parameter/position budgets.
//
// Training follows the paper's CURRICULUM (§3, adopted from iCoT): the model
// starts from ordinary text chain-of-thought data; at curriculum stage k the
// first k reasoning steps of the ground-truth chain are REPLACED by k
// continuous thoughts (c=1 thought per removed step), the remaining chain text
// and the answer stay, and the loss is masked to those remaining tokens only —
// no loss is ever placed on a thought position (see CurriculumLoss). The paper
// resets the optimizer state when the stage switches; that reset is the
// CALLER's job between stages (build a fresh optimizer per stage). Optional
// learnable <bot>/<eot> marker embeddings bracket the thought block
// (WithCoconutMarkers), the paper's convention for delimiting latent mode.
//
// In plain terms: instead of writing its scratch work down word by word, the
// model "thinks silently" — it keeps its raw mental picture (a vector of
// numbers) and feeds it straight back into itself K times before answering.
// Because nothing is rounded to words in between, each silent step can carry
// much more than one token could, and K silent steps let a shallow model do K
// extra rounds of sequential reasoning it structurally cannot do in a single
// pass.
type Coconut struct {
	Config CoconutConfig // the model hyperparameters
	// GPT is the causal decoder-only backbone; thoughts are rows appended to
	// its input embedding, the answer comes from its LM head.
	GPT *GPT
	// BotEmb is the learnable <bot> (begin-of-thought) marker embedding [1,dim]
	// inserted before the thought block; nil unless WithCoconutMarkers.
	BotEmb *tensor.Tensor
	// EotEmb is the learnable <eot> (end-of-thought) marker embedding [1,dim]
	// appended after the thought block; nil unless WithCoconutMarkers.
	EotEmb *tensor.Tensor
	// PauseEmb is the learned CONSTANT pause embedding [1,dim] used in place of
	// hidden-state feedback (the pause-token ablation of the paper); nil unless
	// WithCoconutPause.
	PauseEmb *tensor.Tensor
}

// CoconutConfig fixes the backbone geometry (mirrors GPTConfig). Ctx must
// cover prompt + markers + thoughts + any supervised suffix.
type CoconutConfig struct {
	Vocab  int     // vocabulary size
	Ctx    int     // max sequence length including thought/marker positions
	Dim    int     // embedding width (must divide by Heads)
	Heads  int     // number of attention heads
	Layers int     // number of transformer blocks
	Eps    float64 // LayerNorm epsilon (≤0 → 1e-5)
}

// CoconutOption customizes NewCoconut (functional options).
type CoconutOption func(*coconutOptions)

// coconutOptions collects the NewCoconut option flags.
type coconutOptions struct {
	markers bool // allocate learnable <bot>/<eot> marker embeddings
	pause   bool // pause-token ablation: constant rows instead of feedback
}

// WithCoconutMarkers allocates learnable <bot>/<eot> marker embeddings that
// bracket the thought block on every forward (the paper's latent-mode
// delimiters). Without markers, a K=0 forward is EXACTLY the plain GPT forward.
//
// Curriculum caveat for tiny from-scratch models: with an <eot> marker, the
// next-step prediction signal at stage k lands on the <eot> row, so the row
// ABOVE the last thought — the very hidden state that becomes thought k+1 at
// stage k+1 — never receives the "encode the next step" cross-entropy signal
// and the stage handoff must be learned from BPTT alone. Marker-free training
// makes the handoff exact (the row that predicts step k+1 at stage k IS the
// row fed back at stage k+1) and trains far more reliably at small scale; the
// paper's large pretrained models absorb the indirection.
func WithCoconutMarkers() CoconutOption {
	return func(o *coconutOptions) { o.markers = true }
}

// WithCoconutPause switches the model to the PAUSE-TOKEN ablation (Goyal et
// al. 2023; the control baseline in Hao et al. 2024): the K extra rows are one
// learned constant embedding instead of the fed-back hidden state, so the
// model gets the same extra positions but NO recurrent feedback — extra
// parallel width without extra sequential depth.
func WithCoconutPause() CoconutOption {
	return func(o *coconutOptions) { o.pause = true }
}

// NewCoconut builds a randomly initialized Coconut model: a causal GPT
// backbone (mirroring NewDiffusionLM/NewCLA: Xavier-uniform weights,
// LayerNorms 1/0, zero FFN biases, F64 — the repo's training/gradcheck
// precision) plus the optional marker/pause embeddings. Deterministic via
// seed.
func NewCoconut(cfg CoconutConfig, seed uint64, opts ...CoconutOption) (*Coconut, error) {
	if cfg.Vocab <= 0 || cfg.Ctx <= 0 || cfg.Dim <= 0 || cfg.Layers <= 0 {
		return nil, fmt.Errorf("nlp: NewCoconut needs Vocab, Ctx, Dim, Layers > 0 (got %d, %d, %d, %d)",
			cfg.Vocab, cfg.Ctx, cfg.Dim, cfg.Layers)
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: NewCoconut dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	if cfg.Eps <= 0 {
		cfg.Eps = 1e-5
	}
	var o coconutOptions
	for _, opt := range opts {
		opt(&o)
	}
	randn := func(s uint64, fanIn, fanOut int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
		nn.XavierUniform(t, fanIn, fanOut, s)
		return t
	}
	ln := func() *nn.LayerNorm {
		l := nn.NewLayerNorm(tensor.F64, cfg.Dim)
		l.Eps = cfg.Eps
		return l
	}
	g := &GPT{
		Config: GPTConfig{Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, Layers: cfg.Layers, Eps: cfg.Eps},
		TokEmb: randn(seed, cfg.Vocab, cfg.Dim),
		PosEmb: randn(seed+1, cfg.Ctx, cfg.Dim),
		Head:   randn(seed+2, cfg.Dim, cfg.Vocab),
		LNf:    ln(),
	}
	for l := range cfg.Layers {
		s := seed + 10 + uint64(l)*8
		attn, err := NewMHA(cfg.Heads,
			randn(s, cfg.Dim, cfg.Dim), randn(s+1, cfg.Dim, cfg.Dim),
			randn(s+2, cfg.Dim, cfg.Dim), randn(s+3, cfg.Dim, cfg.Dim))
		if err != nil {
			return nil, err
		}
		attn.Causal = true // decoder-only: thoughts may only look backwards
		g.Blocks = append(g.Blocks, &Block{
			LN1: ln(), LN2: ln(), Attn: attn,
			//perfscan:ignore PS6016 model-construction block alloc, one-time init
			W1: randn(s+4, cfg.Dim, 4*cfg.Dim), B1: tensor.New(tensor.F64, tensor.Shape{4 * cfg.Dim}),
			//perfscan:ignore PS6016 model-construction block alloc, one-time init
			W2: randn(s+5, 4*cfg.Dim, cfg.Dim), B2: tensor.New(tensor.F64, tensor.Shape{cfg.Dim}),
		})
	}
	c := &Coconut{Config: cfg, GPT: g}
	if o.markers {
		c.BotEmb = randn(seed+3, 1, cfg.Dim)
		c.EotEmb = randn(seed+4, 1, cfg.Dim)
	}
	if o.pause {
		c.PauseEmb = randn(seed+5, 1, cfg.Dim)
	}
	return c, nil
}

// Params returns every trainable tensor: the backbone's parameters plus the
// marker/pause embeddings when allocated.
func (c *Coconut) Params() []*tensor.Tensor {
	ps := c.GPT.Params()
	for _, t := range []*tensor.Tensor{c.BotEmb, c.EotEmb, c.PauseEmb} {
		if t != nil {
			ps = append(ps, t)
		}
	}
	return ps
}

// embedTokensAt gathers token+position embeddings for toks occupying absolute
// positions startPos..startPos+len(toks)−1 — GPT.Embed with a position offset,
// through the dispatch so gradients flow to TokEmb/PosEmb.
func (c *Coconut) embedTokensAt(ctx *backend.Context, toks []int, startPos int) (*tensor.Tensor, error) {
	n := len(toks)
	if n == 0 {
		return nil, fmt.Errorf("nlp: Coconut needs a non-empty token sequence")
	}
	if startPos < 0 || startPos+n > c.Config.Ctx {
		return nil, fmt.Errorf("nlp: Coconut positions [%d,%d) outside context %d", startPos, startPos+n, c.Config.Ctx)
	}
	for _, t := range toks {
		if t < 0 || t >= c.Config.Vocab {
			return nil, fmt.Errorf("nlp: Coconut token %d outside vocab %d", t, c.Config.Vocab)
		}
	}
	tokIdx := tensor.New(c.GPT.TokEmb.Dtype(), tensor.Shape{n})
	posIdx := tensor.New(c.GPT.PosEmb.Dtype(), tensor.Shape{n})
	for i, t := range toks {
		tokIdx.SetF64(float64(t), i)
		posIdx.SetF64(float64(startPos+i), i)
	}
	et, err := exec1(ctx, backend.OpEmbed, nil, c.GPT.TokEmb, tokIdx)
	if err != nil {
		return nil, err
	}
	ep, err := exec1(ctx, backend.OpEmbed, nil, c.GPT.PosEmb, posIdx)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpAdd, nil, et, ep)
}

// appendRow appends one content row [1,dim] (a thought, marker, or pause
// embedding) at absolute position pos: x' = concat(x, row + PosEmb[pos]).
// Differentiable (OpEmbed/OpAdd/OpConcat are all VJP-backed).
func (c *Coconut) appendRow(ctx *backend.Context, x, row *tensor.Tensor, pos int) (*tensor.Tensor, error) {
	if pos < 0 || pos >= c.Config.Ctx {
		return nil, fmt.Errorf("nlp: Coconut position %d outside context %d", pos, c.Config.Ctx)
	}
	posIdx := tensor.New(c.GPT.PosEmb.Dtype(), tensor.Shape{1})
	posIdx.SetF64(float64(pos), 0)
	pe, err := exec1(ctx, backend.OpEmbed, nil, c.GPT.PosEmb, posIdx)
	if err != nil {
		return nil, err
	}
	sum, err := exec1(ctx, backend.OpAdd, nil, row, pe)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, x, sum)
}

// latentPrefix builds the input embedding for prompt + (<bot>) + k latent rows
// + (<eot>) and returns it with its length. This is the continuous-thought
// loop: each latent row is the LAST final hidden state of a full backbone pass
// over everything so far (or the constant PauseEmb in the pause ablation) —
// k feedback steps means k extra backbone forwards, all on the tape.
func (c *Coconut) latentPrefix(ctx *backend.Context, prompt []int, k int) (*tensor.Tensor, int, error) {
	if k < 0 {
		return nil, 0, fmt.Errorf("nlp: Coconut needs k >= 0 thoughts, got %d", k)
	}
	x, err := c.embedTokensAt(ctx, prompt, 0)
	if err != nil {
		return nil, 0, err
	}
	pos := len(prompt)
	if c.BotEmb != nil {
		if x, err = c.appendRow(ctx, x, c.BotEmb, pos); err != nil {
			return nil, 0, err
		}
		pos++
	}
	for range k {
		row := c.PauseEmb // pause ablation: constant, no feedback, no extra forward
		if row == nil {
			h, err := c.GPT.hiddenFromEmbed(ctx, x) // full pass over everything so far
			if err != nil {
				return nil, 0, err
			}
			// the continuous thought: the last row of the final hidden state
			//perfscan:ignore PS6017 OpSlice dwarfed by full transformer forward per thought
			if row, err = exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: pos - 1, End: pos}, h); err != nil {
				return nil, 0, err
			}
		}
		if x, err = c.appendRow(ctx, x, row, pos); err != nil {
			return nil, 0, err
		}
		pos++
	}
	if c.EotEmb != nil {
		if x, err = c.appendRow(ctx, x, c.EotEmb, pos); err != nil {
			return nil, 0, err
		}
		pos++
	}
	return x, pos, nil
}

// ForwardWithThoughts embeds the prompt, runs k continuous-thought feedback
// steps (each: full backbone pass → last hidden row appended as the next input
// embedding), then one final pass through the LM head, returning logits
// [len(prompt)+k+markers, vocab]. The last row's logits predict the answer.
// With k=0 and no markers this IS the plain GPT forward on the prompt (same
// weights, same ops). Differentiable through every thought when ctx records.
func (c *Coconut) ForwardWithThoughts(ctx *backend.Context, prompt []int, k int) (*tensor.Tensor, error) {
	x, _, err := c.latentPrefix(ctx, prompt, k)
	if err != nil {
		return nil, err
	}
	h, err := c.GPT.hiddenFromEmbed(ctx, x)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, h, c.GPT.Head)
}

// CurriculumLoss is one example's loss at curriculum stage `stage` (Hao et al.
// 2024 §3): the first `stage` ground-truth reasoning steps of chain are
// REPLACED by `stage` continuous thoughts (c=1 thought per step), the
// remaining steps chain[stage:] and the answer stay as text, and the loss is
// the mean next-token cross-entropy over those remaining tokens ONLY — no loss
// on the question, markers, or thoughts. stage ranges 0 (full text chain-of-
// thought) to len(chain) (fully latent: question → thoughts → answer). The
// answer must be non-empty; chain steps hold ≥1 token each. Optimizer state
// should be reset by the caller when the stage changes (fresh optimizer per
// stage), per the paper. Fully differentiable — the thoughts receive gradient
// THROUGH the remaining-token predictions (BPTT across the feedback loop).
//
// Two training-loop recommendations from the value-validation experiments
// (coconut_test.go): interleave earlier-stage examples while training stage k
// (e.g. replay a uniform stage < k on half the steps) — with a hard stage
// switch nothing anchors the prefix computation the thoughts are produced by,
// and the curriculum can collapse mid-way; and format the data so the first
// supervised token after the thought block plays the same ROLE at every stage
// (the answer should be one further reasoning step from the last thought, not
// a restatement of it), otherwise the fully-latent stage trains a rule that
// contradicts the step-replacement rule every earlier stage taught.
func (c *Coconut) CurriculumLoss(ctx *backend.Context, question []int, chain [][]int, answer []int, stage int) (*tensor.Tensor, error) {
	if stage < 0 || stage > len(chain) {
		return nil, fmt.Errorf("nlp: CurriculumLoss stage %d outside [0,%d]", stage, len(chain))
	}
	if len(answer) == 0 {
		return nil, fmt.Errorf("nlp: CurriculumLoss needs a non-empty answer")
	}
	var suffix []int // the supervised tokens: remaining chain steps, then the answer
	for _, step := range chain[stage:] {
		if len(step) == 0 {
			return nil, fmt.Errorf("nlp: CurriculumLoss chain steps must be non-empty")
		}
		suffix = append(suffix, step...)
	}
	suffix = append(suffix, answer...)
	x, pos, err := c.latentPrefix(ctx, question, stage)
	if err != nil {
		return nil, err
	}
	// Teacher-forced input: everything but the last supervised token (which is
	// only ever a target). suffix[i] sits at absolute position pos+i.
	if len(suffix) > 1 {
		xs, err := c.embedTokensAt(ctx, suffix[:len(suffix)-1], pos)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, x, xs); err != nil {
			return nil, err
		}
	}
	h, err := c.GPT.hiddenFromEmbed(ctx, x)
	if err != nil {
		return nil, err
	}
	logits, err := exec1(ctx, backend.OpMatMul, nil, h, c.GPT.Head)
	if err != nil {
		return nil, err
	}
	// Gather the rows predicting each supervised token (row pos+i−1 predicts
	// suffix[i]) — differentiable row-gather, as in DiffusionLM.MaskedLoss.
	idx := tensor.New(logits.Dtype(), tensor.Shape{len(suffix)})
	targets := tensor.New(logits.Dtype(), tensor.Shape{len(suffix)})
	for i, tok := range suffix {
		if tok < 0 || tok >= c.Config.Vocab {
			return nil, fmt.Errorf("nlp: CurriculumLoss target token %d outside vocab %d", tok, c.Config.Vocab)
		}
		idx.SetF64(float64(pos+i-1), i)
		targets.SetF64(float64(tok), i)
	}
	rows, err := exec1(ctx, backend.OpEmbed, nil, logits, idx)
	if err != nil {
		return nil, err
	}
	return nn.CrossEntropy(ctx, rows, targets)
}

// Generate decodes maxNew answer tokens greedily after k continuous thoughts:
// prompt → k latent feedback steps → argmax over the last row's logits, with
// each emitted token embedded and appended for the next step (thoughts are
// computed once, before the first token). Deterministic. Inference-only —
// pass a non-recording context (backend.NewContext()).
func (c *Coconut) Generate(ctx *backend.Context, prompt []int, k, maxNew int) ([]int, error) {
	if maxNew <= 0 {
		return nil, fmt.Errorf("nlp: Generate needs maxNew > 0, got %d", maxNew)
	}
	x, pos, err := c.latentPrefix(ctx, prompt, k)
	if err != nil {
		return nil, err
	}
	out := make([]int, 0, maxNew)
	for {
		h, err := c.GPT.hiddenFromEmbed(ctx, x)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 OpSlice last-row trivial vs hiddenFromEmbed forward
		last, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: pos - 1, End: pos}, h)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 single LM-head matvec dispatch, not restructurable
		logits, err := exec1(ctx, backend.OpMatMul, nil, last, c.GPT.Head)
		if err != nil {
			return nil, err
		}
		best := 0
		for v := 1; v < c.Config.Vocab; v++ {
			if logits.AtF64(0, v) > logits.AtF64(0, best) {
				best = v
			}
		}
		out = append(out, best)
		if len(out) == maxNew {
			return out, nil
		}
		xt, err := c.embedTokensAt(ctx, []int{best}, pos)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 OpConcat tiny vs full re-forward it feeds
		if x, err = exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, x, xt); err != nil {
			return nil, err
		}
		pos++
	}
}
