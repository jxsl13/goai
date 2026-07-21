package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// DiffusionLM is a LLaDA-style masked-diffusion language model (Nie et al. 2025,
// "Large Language Diffusion Models", arXiv:2502.09992): a BIDIRECTIONAL
// transformer trained to reconstruct tokens that a forward corruption process
// replaced by an absorbing [MASK] token, and generating by iterative unmasking
// instead of left-to-right decoding.
//
// Training objective (the discrete-diffusion ELBO bound): per sequence draw a
// masking level t ~ U(0,1], mask every position independently with probability t
// to the [MASK] id (absorbing state — ALWAYS [MASK], unlike BERT's 80/10/10
// MLMMask split), forward the corrupted sequence, and score cross-entropy on the
// masked positions only, weighted by 1/t (see MaskedLoss). Generation reverses
// the corruption: start from an all-[MASK] sequence and, over a fixed number of
// steps, fill every masked position in parallel, keeping only the
// highest-confidence predictions and re-masking the rest on a linear schedule
// (see DiffusionGenerate) — a Jacobi-like fixed-iteration loop (compare
// JacobiDecode), not an autoregressive scan.
//
// In plain terms: a GPT reads left-to-right and writes one word at a time; this
// model plays fill-in-the-blanks. It learns by hiding a random fraction of each
// training sentence behind [MASK] tokens and guessing the hidden words from BOTH
// sides at once, and it writes new text by starting from a page of blanks and
// filling them in over a few rounds, keeping the guesses it is most sure of and
// retrying the rest.
//
// Contrast with the two siblings: GPT (gpt.go) is the same block stack but
// CAUSAL and autoregressive — here every block's attention is bidirectional
// (Causal=false) and all positions are predicted in parallel; nn/ddpm.go is
// diffusion over CONTINUOUS data (Gaussian noise on real vectors) — this is
// DISCRETE diffusion whose forward process jumps tokens to an absorbing [MASK]
// state, so the "noise level" is the masked fraction t, not a Gaussian variance.
type DiffusionLM struct {
	Config DiffusionLMConfig // the model hyperparameters (Vocab counts REAL tokens; [MASK] is the extra id Vocab)
	GPT    *GPT              // the bidirectional backbone: the GPT block stack with Causal=false and Vocab+1 token/logit rows
}

// DiffusionLMConfig fixes the diffusion model geometry (mirrors GPTConfig). The
// backbone embedding table and head carry Vocab+1 rows: the extra row is the
// [MASK] token, whose id is Vocab itself (MaskID).
type DiffusionLMConfig struct {
	Vocab  int     // REAL vocabulary size; the [MASK] id is Vocab (the backbone works over Vocab+1 ids)
	Ctx    int     // max sequence length
	Dim    int     // embedding width
	Heads  int     // number of attention heads (must divide Dim)
	Layers int     // number of transformer blocks
	Eps    float64 // LayerNorm epsilon (≤0 → 1e-5)
}

// NewDiffusionLM builds a randomly initialized masked-diffusion LM: the GPT
// block shape with bidirectional (non-causal) attention in every block and a
// Vocab+1 token table/head for the [MASK] row. Deterministic via seed; every
// weight matrix is Xavier-uniform (nn.XavierUniform, the repo's trained-from-
// scratch default — at the small dims this model targets it converges far
// faster than GPT-2's 0.02 normal), LayerNorms 1/0, FFN biases 0; parameters
// are F64 (the repo's training/gradcheck precision).
func NewDiffusionLM(cfg DiffusionLMConfig, seed uint64) (*DiffusionLM, error) {
	if cfg.Vocab <= 0 || cfg.Ctx <= 0 || cfg.Dim <= 0 || cfg.Layers <= 0 {
		return nil, fmt.Errorf("nlp: NewDiffusionLM needs Vocab, Ctx, Dim, Layers > 0 (got %d, %d, %d, %d)",
			cfg.Vocab, cfg.Ctx, cfg.Dim, cfg.Layers)
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: NewDiffusionLM dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	if cfg.Eps <= 0 {
		cfg.Eps = 1e-5
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
	v1 := cfg.Vocab + 1 // +1 row: the [MASK] token
	g := &GPT{
		Config: GPTConfig{Vocab: v1, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, Layers: cfg.Layers, Eps: cfg.Eps},
		TokEmb: randn(seed, v1, cfg.Dim),
		PosEmb: randn(seed+1, cfg.Ctx, cfg.Dim),
		Head:   randn(seed+2, cfg.Dim, v1),
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
		// attn.Causal stays false — bidirectional attention is the whole point:
		// the constructor line attn.Causal=true is the ONLY causal coupling in
		// the GPT backbone, and this model omits it.
		g.Blocks = append(g.Blocks, &Block{
			LN1: ln(), LN2: ln(), Attn: attn,
			W1: randn(s+4, cfg.Dim, 4*cfg.Dim), B1: tensor.New(tensor.F64, tensor.Shape{4 * cfg.Dim}),
			W2: randn(s+5, 4*cfg.Dim, cfg.Dim), B2: tensor.New(tensor.F64, tensor.Shape{cfg.Dim}),
		})
	}
	return &DiffusionLM{Config: cfg, GPT: g}, nil
}

// MaskID returns the [MASK] token id — the absorbing state of the forward
// corruption process. It is Config.Vocab, the one id past the real vocabulary
// (the backbone's extra embedding/head row).
func (m *DiffusionLM) MaskID() int { return m.Config.Vocab }

// Forward computes logits [len(tokens), Vocab+1] for a (possibly corrupted)
// token sequence through the bidirectional backbone. Token ids may include
// MaskID. The pass is mask-agnostic — it is exactly the GPT forward, reused
// unchanged, with non-causal attention.
func (m *DiffusionLM) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	return m.GPT.Forward(ctx, tokens)
}

// Params returns every trainable tensor (token/pos embeddings, all block
// weights, final LN, and the head) for optimizers — the backbone's Params.
func (m *DiffusionLM) Params() []*tensor.Tensor { return m.GPT.Params() }

// DiffusionMask applies the forward (corruption) process of masked diffusion to
// tokens at masking level t ∈ [0,1]: each position is independently replaced by
// maskID with probability t. Unlike BERT's MLMMask 80/10/10 split, the
// replacement is ALWAYS maskID — the absorbing state of the discrete diffusion
// (Nie et al. 2025, arXiv:2502.09992). It returns the corrupted copy and the
// masked position indices (ascending); tokens is not modified. Deterministic
// given the rng state (one draw per position).
func DiffusionMask(tokens []int, t float64, maskID int, rng *rand.Rand) (corrupted, positions []int) {
	corrupted = make([]int, len(tokens))
	copy(corrupted, tokens)
	for i := range tokens {
		if rng.Float64() < t {
			corrupted[i] = maskID
			positions = append(positions, i)
		}
	}
	return corrupted, positions
}

// MaskedLoss is the masked weighted cross-entropy of one corrupted sequence at
// masking level t ∈ (0,1] — the per-sequence term of the LLaDA training bound
// (Nie et al. 2025, arXiv:2502.09992, eq. 3):
//
//	loss = (1/(L·t)) · Σ_{i: corrupted[i]=[MASK]} CE(logits[i], original[i])
//
// computed as OpEmbed row-gather of the masked logit rows (differentiable — the
// embed VJP scatter-adds the row gradients back), nn.CrossEntropy over the
// gathered rows (mean over the m masked positions), then a scale by m/(L·t):
// the m/L restores the sum-over-positions normalized by sequence length and the
// 1/t is the ELBO weight that makes the expectation over t an upper bound on
// negative log-likelihood. Positions with corrupted[i] ≠ [MASK] contribute
// nothing. Errors if no position is masked (an empty term carries no gradient;
// Loss redraws instead). Fully differentiable w.r.t. Params through a recording
// context.
func (m *DiffusionLM) MaskedLoss(ctx *backend.Context, corrupted, original []int, t float64) (*tensor.Tensor, error) {
	seqLen := len(corrupted)
	if seqLen == 0 || len(original) != seqLen {
		return nil, fmt.Errorf("nlp: MaskedLoss needs equal non-empty sequences (corrupted %d, original %d)",
			seqLen, len(original))
	}
	if t <= 0 || t > 1 {
		return nil, fmt.Errorf("nlp: MaskedLoss masking level t=%g outside (0,1]", t)
	}
	maskID := m.MaskID()
	var positions []int
	for i, tok := range corrupted {
		if tok != maskID {
			continue
		}
		if original[i] < 0 || original[i] >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: MaskedLoss original token %d at %d outside vocab %d",
				original[i], i, m.Config.Vocab)
		}
		positions = append(positions, i)
	}
	if len(positions) == 0 {
		return nil, fmt.Errorf("nlp: MaskedLoss needs at least one masked position")
	}
	logits, err := m.Forward(ctx, corrupted)
	if err != nil {
		return nil, err
	}
	// Gather the masked logit rows [m, Vocab+1] (differentiable row-gather) and
	// their true tokens, then mean-CE over the gathered rows.
	idx := tensor.New(logits.Dtype(), tensor.Shape{len(positions)})
	targets := tensor.New(logits.Dtype(), tensor.Shape{len(positions)})
	for k, p := range positions {
		idx.SetF64(float64(p), k)
		targets.SetF64(float64(original[p]), k)
	}
	rows, err := exec1(ctx, backend.OpEmbed, nil, logits, idx)
	if err != nil {
		return nil, err
	}
	ce, err := nn.CrossEntropy(ctx, rows, targets)
	if err != nil {
		return nil, err
	}
	// ce is the mean over m masked rows; m/(L·t) turns it into the 1/t-weighted
	// per-position sum normalized by L. OpAXPY with a zero addend is the
	// differentiable scalar scale (α·x + 0).
	w := float64(len(positions)) / (float64(seqLen) * t)
	return exec1(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: w}, ce, tensor.New(ce.Dtype(), tensor.Shape{}))
}

// Loss is one Monte-Carlo draw of the masked-diffusion training objective for a
// clean sequence: it draws t ~ U(0,1) from rng, corrupts tokens with
// DiffusionMask, and returns MaskedLoss on the result. Draws that mask no
// position are redrawn (they carry zero signal and no gradient), i.e. the
// estimate conditions on ≥1 masked position. Deterministic given the rng state;
// differentiable through a recording context.
func (m *DiffusionLM) Loss(ctx *backend.Context, tokens []int, rng *rand.Rand) (*tensor.Tensor, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("nlp: Loss needs a non-empty sequence")
	}
	if rng == nil {
		return nil, fmt.Errorf("nlp: Loss needs a non-nil rng")
	}
	for {
		t := rng.Float64()
		corrupted, positions := DiffusionMask(tokens, t, m.MaskID(), rng)
		if len(positions) == 0 {
			continue // no masked position → no signal; redraw t and the mask
		}
		return m.MaskedLoss(ctx, corrupted, tokens, t)
	}
}

// DiffusionGenerate produces length tokens by iterative unmasking (the LLaDA
// reverse process, Nie et al. 2025, arXiv:2502.09992 §2.4 low-confidence
// remasking): start from an all-[MASK] sequence; at each of steps rounds,
// forward the whole sequence bidirectionally, sample a token for EVERY masked
// position from its logit row over the real vocabulary (the [MASK] column is
// excluded — the reverse process never writes the absorbing state), score each
// fill by the model's softmax probability of the chosen token, and re-mask the
// lowest-confidence fills so that round(length·(1−(step+1)/steps)) positions
// stay masked — a linear unmasking schedule that reaches zero masks at the last
// step. With steps=1 the whole sequence is filled in a single parallel shot.
// The sampler drives the per-position draw (Sampler.Sample); greedy
// (temperature ≤ 0) is fully deterministic.
// Like JacobiDecode this is a fixed number of full-sequence parallel passes,
// but the fills come from the learned denoiser, not a causal fixed-point.
func (m *DiffusionLM) DiffusionGenerate(length, steps int, s *Sampler) ([]int, error) {
	return m.diffusionGenerate(length, steps, s, nil)
}

// diffusionGenerate is DiffusionGenerate with a per-step observation hook
// (onStep receives the step index and a copy of the sequence after its remask),
// so tests can verify the unmasking schedule without widening the API.
func (m *DiffusionLM) diffusionGenerate(length, steps int, s *Sampler, onStep func(step int, seq []int)) ([]int, error) {
	if length <= 0 || length > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: DiffusionGenerate length %d outside (0,%d]", length, m.Config.Ctx)
	}
	if steps <= 0 {
		return nil, fmt.Errorf("nlp: DiffusionGenerate needs steps > 0, got %d", steps)
	}
	if s == nil {
		return nil, fmt.Errorf("nlp: DiffusionGenerate needs a non-nil sampler")
	}
	maskID := m.MaskID()
	seq := make([]int, length)
	for i := range seq {
		seq[i] = maskID
	}
	ctx := backend.NewContext()
	for step := range steps {
		var masked []int
		for i, tok := range seq {
			if tok == maskID {
				masked = append(masked, i)
			}
		}
		if len(masked) == 0 {
			break
		}
		logits, err := m.Forward(ctx, seq)
		if err != nil {
			return nil, err
		}
		lc := logits.Contiguous()
		stride := logits.Shape()[1] // Vocab+1
		// Fill every masked position; confidence = the model's softmax
		// probability of the chosen token over the raw logit row (the paper's
		// low-confidence remasking score). The sampler's own Dist cannot serve
		// here: under greedy decoding it is a one-hot, which would collapse
		// every confidence to 1 and reduce remasking to position order.
		confs := make([]float64, len(masked))
		for k, i := range masked {
			row := logitsRow(lc, i, stride, m.Config.Vocab)
			seq[i] = s.Sample(row)
			confs[k] = softmaxProb(row)[seq[i]]
		}
		// Linear schedule: after step s+1, round(L·(1−(s+1)/steps)) positions
		// remain masked — re-mask the lowest-confidence fills (ties broken by
		// position for determinism).
		remain := int(math.Round(float64(length) * (1 - float64(step+1)/float64(steps))))
		if remain > len(masked) {
			remain = len(masked)
		}
		if remain > 0 {
			order := diffusionRefillOrder(confs)
			for _, k := range order[:remain] {
				seq[masked[k]] = maskID
			}
		}
		if onStep != nil {
			onStep(step, append([]int(nil), seq...))
		}
	}
	return seq, nil
}

// logitsRow copies the first v entries of row i from a contiguous [L, stride]
// logits tensor — the real-vocabulary slice of one position's logits, excluding
// the trailing [MASK] column. Typed fast paths mirror rowLogits (§base-perf).
// diffusionRefillOrder orders the masked-position indices by ASCENDING
// confidence (lowest first — those get re-masked next). Confidences are softmax
// probabilities (≥0), so the ascending radix (sortIdxAscByScore) orders them
// identically to the comparison sort; equal confidences retain ascending index
// order, which — because the caller's masked list is in ascending position order
// — reproduces the previous masked[order[a]] < masked[order[b]] tie-break.
func diffusionRefillOrder(confs []float64) []int {
	order := make([]int, len(confs))
	for k := range order {
		order[k] = k
	}
	sortIdxAscByScore(order, confs)
	return order
}

func logitsRow(t *tensor.Tensor, i, stride, v int) []float64 {
	out := make([]float64, v)
	switch t.Dtype() {
	case tensor.F64:
		copy(out, t.Storage().F64()[i*stride:i*stride+v])
	case tensor.F32:
		src := t.Storage().F32()[i*stride : i*stride+v]
		for j, x := range src {
			out[j] = float64(x)
		}
	default:
		for j := range v {
			out[j] = t.AtF64(i, j)
		}
	}
	return out
}
