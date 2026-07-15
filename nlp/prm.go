package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// RewardModelConfig fixes the geometry of the reward-model backbones
// (ProcessRewardModel, OutcomeRewardModel) — it mirrors GPTConfig: the
// backbone is the GPT block stack with the [Dim,Vocab] LM head replaced by a
// [Dim,1] scalar scoring head.
type RewardModelConfig struct {
	Vocab  int     // vocabulary size
	Ctx    int     // max sequence length
	Dim    int     // embedding width
	Heads  int     // number of attention heads (must divide Dim)
	Layers int     // number of transformer blocks
	Eps    float64 // LayerNorm epsilon (≤0 → 1e-5)
}

// ProcessRewardModel is a step-level (process) reward model: a causal GPT
// backbone whose scalar head is read at each STEP-END token position, giving a
// per-step score σ(h_pos·W + b) ∈ (0,1) = P(step correct). It is trained with
// binary cross-entropy against per-step correctness labels (Loss) and scored
// with StepScores.
//
// Process vs outcome supervision: an OUTCOME reward model (ORM) scores a whole
// solution with one number learned from the final-answer-correct flag alone —
// the label is silent about WHERE reasoning failed, and a solution that
// reaches the right answer through a wrong step is labelled "good" (the
// lucky-correct contamination). A PROCESS reward model (PRM) instead receives
// a label per reasoning step, so it localizes the earliest error and rejects
// right-answer/wrong-reasoning solutions. Lightman et al. 2023, "Let's Verify
// Step by Step" (arXiv:2305.20050) showed process supervision significantly
// outperforms outcome supervision for math reasoning; Math-Shepherd (Wang et
// al. 2023, arXiv:2312.08935) automates the per-step labels by Monte-Carlo
// completion (its soft labels ∈ [0,1] are accepted by Loss); ProcessBench
// (Zheng et al. 2024, arXiv:2412.06559) evaluates step-error detection as
// earliest-error identification, scored by the F1 of the accuracies on the
// erroneous and correct subsets.
//
// Verification (best-of-N): a solution's score is the MINIMUM step score —
// a chain of reasoning is only as strong as its weakest step (Lightman et al.
// §4.2; Math-Shepherd Eq. 5 uses the same min aggregation) — and the candidate
// with the highest min-step-score wins. Contrast with the repo's outcome-level
// alignment tools: nn.BradleyTerryLoss trains a reward model on whole-response
// preference PAIRS, and nn.GRPO consumes one scalar reward per sampled
// response; both are outcome-level signals that cannot localize errors, which
// is exactly what the per-step head here adds.
//
// In plain terms: an outcome grader only marks the final answer, so a student
// who guesses right gets full marks; this model grades every line of the
// working, catches the first wrong line, and prefers the solution whose worst
// line is still sound.
type ProcessRewardModel struct {
	Config RewardModelConfig // the model hyperparameters
	GPT    *GPT              // causal backbone; its Head is the [Dim,1] step-scoring weight
	Bias   *tensor.Tensor    // [1] scoring-head bias
}

// NewProcessRewardModel builds a randomly initialized step-level reward model:
// the GPT block stack with causal attention and a [Dim,1] scoring head in
// place of the LM head. Deterministic via seed; weight matrices are
// Xavier-uniform (nn.XavierUniform, the repo's trained-from-scratch default),
// LayerNorms 1/0, biases 0; parameters are F64 (the repo's training/gradcheck
// precision).
func NewProcessRewardModel(cfg RewardModelConfig, seed uint64) (*ProcessRewardModel, error) {
	g, bias, cfg, err := newRewardBackbone("NewProcessRewardModel", cfg, seed)
	if err != nil {
		return nil, err
	}
	return &ProcessRewardModel{Config: cfg, GPT: g, Bias: bias}, nil
}

// Params returns every trainable tensor (token/pos embeddings, all block
// weights, final LN, the scoring head and its bias) for optimizers.
func (m *ProcessRewardModel) Params() []*tensor.Tensor {
	return append(m.GPT.Params(), m.Bias)
}

// StepScores runs the backbone over tokens and returns σ(h·W+b) at each
// stepEnds position — one score in (0,1) per step, the model's P(step
// correct). stepEnds are the STEP-END token positions (e.g. the step-separator
// after each reasoning step); they must be strictly increasing and within
// [0,len(tokens)). Under causal attention the hidden state at a step's end has
// seen the whole step including its stated result, which is what the head
// judges.
func (m *ProcessRewardModel) StepScores(ctx *backend.Context, tokens, stepEnds []int) ([]float64, error) {
	z, err := rewardLogits(ctx, m.GPT, m.Bias, tokens, stepEnds)
	if err != nil {
		return nil, err
	}
	s, err := exec1(ctx, backend.OpSigmoid, nil, z)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(stepEnds))
	for k := range out {
		out[k] = s.AtF64(k, 0)
	}
	return out, nil
}

// Loss is the mean binary cross-entropy between the step scores at stepEnds
// and the per-step correctness labels — the PRM training objective (Lightman
// et al. 2023 §2; Math-Shepherd Eq. 4). labels[i] ∈ [0,1] is the target
// P(step i correct): hard 0/1 labels for human/oracle annotation, soft values
// for Math-Shepherd-style Monte-Carlo estimates. Computed stably in logit
// space as y·softplus(−z) + (1−y)·softplus(z) (no σ/log that could
// under/overflow); fully differentiable w.r.t. Params through a recording
// context.
func (m *ProcessRewardModel) Loss(ctx *backend.Context, tokens, stepEnds []int, labels []float64) (*tensor.Tensor, error) {
	if len(labels) != len(stepEnds) {
		return nil, fmt.Errorf("nlp: PRM Loss needs one label per step end, got %d labels for %d steps", len(labels), len(stepEnds))
	}
	z, err := rewardLogits(ctx, m.GPT, m.Bias, tokens, stepEnds)
	if err != nil {
		return nil, err
	}
	return rewardBCE(ctx, z, labels)
}

// OutcomeRewardModel is the outcome-level (ORM) ablation twin of
// ProcessRewardModel: the IDENTICAL backbone and scalar head, but read at ONE
// position — the last token — and trained only on the final-answer-correct
// label (one bit per sequence). This is the classic outcome supervision of
// Cobbe et al. 2021 / the baseline of Lightman et al. 2023 (arXiv:2305.20050):
// it cannot say where a solution went wrong, and its training labels are
// poisoned by lucky-correct solutions (wrong reasoning, right answer), which
// is precisely the gap the PRM closes.
type OutcomeRewardModel struct {
	Config RewardModelConfig // the model hyperparameters
	GPT    *GPT              // causal backbone; its Head is the [Dim,1] scoring weight
	Bias   *tensor.Tensor    // [1] scoring-head bias
}

// NewOutcomeRewardModel builds a randomly initialized outcome-level reward
// model with the same architecture and initialization scheme as
// NewProcessRewardModel (deterministic via seed) — the controlled ablation
// baseline for the PRM: same capacity, coarser supervision.
func NewOutcomeRewardModel(cfg RewardModelConfig, seed uint64) (*OutcomeRewardModel, error) {
	g, bias, cfg, err := newRewardBackbone("NewOutcomeRewardModel", cfg, seed)
	if err != nil {
		return nil, err
	}
	return &OutcomeRewardModel{Config: cfg, GPT: g, Bias: bias}, nil
}

// Params returns every trainable tensor (token/pos embeddings, all block
// weights, final LN, the scoring head and its bias) for optimizers.
func (m *OutcomeRewardModel) Params() []*tensor.Tensor {
	return append(m.GPT.Params(), m.Bias)
}

// Score returns σ(h_last·W+b) ∈ (0,1) — the model's P(final answer correct)
// read at the LAST token position, one scalar for the whole sequence.
func (m *OutcomeRewardModel) Score(ctx *backend.Context, tokens []int) (float64, error) {
	if len(tokens) == 0 {
		return 0, fmt.Errorf("nlp: ORM Score needs a non-empty sequence")
	}
	z, err := rewardLogits(ctx, m.GPT, m.Bias, tokens, []int{len(tokens) - 1})
	if err != nil {
		return 0, err
	}
	s, err := exec1(ctx, backend.OpSigmoid, nil, z)
	if err != nil {
		return 0, err
	}
	return s.AtF64(0, 0), nil
}

// Loss is the binary cross-entropy between the sequence score (Score's logit
// at the last token) and the outcome label ∈ [0,1] (1 = final answer correct)
// — the ORM training objective. Differentiable w.r.t. Params through a
// recording context.
func (m *OutcomeRewardModel) Loss(ctx *backend.Context, tokens []int, label float64) (*tensor.Tensor, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("nlp: ORM Loss needs a non-empty sequence")
	}
	z, err := rewardLogits(ctx, m.GPT, m.Bias, tokens, []int{len(tokens) - 1})
	if err != nil {
		return nil, err
	}
	return rewardBCE(ctx, z, []float64{label})
}

// newRewardBackbone builds the shared reward-model backbone: the GPT block
// stack (causal, like gpt.go's FromSafetensors path) whose Head field holds
// the [Dim,1] scoring weight instead of an LM head, plus the [1] head bias.
// Initialization mirrors NewDiffusionLM/NewCLA (Xavier-uniform matrices,
// LayerNorm 1/0, zero biases, F64). Returns the eps-normalized config.
func newRewardBackbone(fname string, cfg RewardModelConfig, seed uint64) (*GPT, *tensor.Tensor, RewardModelConfig, error) {
	if cfg.Vocab <= 0 || cfg.Ctx <= 0 || cfg.Dim <= 0 || cfg.Layers <= 0 {
		return nil, nil, cfg, fmt.Errorf("nlp: %s needs Vocab, Ctx, Dim, Layers > 0 (got %d, %d, %d, %d)",
			fname, cfg.Vocab, cfg.Ctx, cfg.Dim, cfg.Layers)
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, nil, cfg, fmt.Errorf("nlp: %s dim %d not divisible by heads %d", fname, cfg.Dim, cfg.Heads)
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
	g := &GPT{
		Config: GPTConfig{Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, Layers: cfg.Layers, Eps: cfg.Eps},
		TokEmb: randn(seed, cfg.Vocab, cfg.Dim),
		PosEmb: randn(seed+1, cfg.Ctx, cfg.Dim),
		Head:   randn(seed+2, cfg.Dim, 1), // the scalar scoring head
		LNf:    ln(),
	}
	for l := range cfg.Layers {
		s := seed + 10 + uint64(l)*8
		attn, err := NewMHA(cfg.Heads,
			randn(s, cfg.Dim, cfg.Dim), randn(s+1, cfg.Dim, cfg.Dim),
			randn(s+2, cfg.Dim, cfg.Dim), randn(s+3, cfg.Dim, cfg.Dim))
		if err != nil {
			return nil, nil, cfg, err
		}
		attn.Causal = true // reward models read solutions left-to-right, like their LM backbone
		g.Blocks = append(g.Blocks, &Block{
			LN1: ln(), LN2: ln(), Attn: attn,
			W1: randn(s+4, cfg.Dim, 4*cfg.Dim), B1: tensor.New(tensor.F64, tensor.Shape{4 * cfg.Dim}),
			W2: randn(s+5, 4*cfg.Dim, cfg.Dim), B2: tensor.New(tensor.F64, tensor.Shape{cfg.Dim}),
		})
	}
	return g, tensor.New(tensor.F64, tensor.Shape{1}), cfg, nil
}

// rewardLogits is the shared scoring path: ForwardHidden over tokens
// [seq,Dim], a differentiable row-gather of the positions (OpEmbed — its VJP
// scatter-adds the row gradients back, the MaskedLoss pattern), then the
// scalar head h·W+b → logits [n,1]. Positions must be strictly increasing and
// within [0,len(tokens)).
func rewardLogits(ctx *backend.Context, g *GPT, bias *tensor.Tensor, tokens, positions []int) (*tensor.Tensor, error) {
	if len(positions) == 0 {
		return nil, fmt.Errorf("nlp: reward scoring needs at least one position")
	}
	for i, p := range positions {
		if p < 0 || p >= len(tokens) {
			return nil, fmt.Errorf("nlp: score position %d = %d outside sequence length %d", i, p, len(tokens))
		}
		if i > 0 && p <= positions[i-1] {
			return nil, fmt.Errorf("nlp: score positions must be strictly increasing, got %d after %d", p, positions[i-1])
		}
	}
	hidden, err := g.ForwardHidden(ctx, tokens) // [seq, Dim]
	if err != nil {
		return nil, err
	}
	idx := tensor.New(hidden.Dtype(), tensor.Shape{len(positions)})
	for k, p := range positions {
		idx.SetF64(float64(p), k)
	}
	rows, err := exec1(ctx, backend.OpEmbed, nil, hidden, idx) // [n, Dim]
	if err != nil {
		return nil, err
	}
	z, err := exec1(ctx, backend.OpMatMul, nil, rows, g.Head) // [n, 1]
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpAddBias, nil, z, bias)
}

// rewardBCE is the mean binary cross-entropy of score logits z [n,1] against
// labels y ∈ [0,1], computed stably in softplus space:
//
//	BCE = mean_k [ y_k·softplus(−z_k) + (1−y_k)·softplus(z_k) ]
//
// since −log σ(z) = softplus(−z) and −log(1−σ(z)) = softplus(z). Exact for
// SOFT labels too (unlike the sign-flip softplus((1−2y)z) shortcut, which
// only equals BCE at y ∈ {0,1}). The label weights are non-differentiable
// constants; the loss is differentiable w.r.t. the logits.
func rewardBCE(ctx *backend.Context, logits *tensor.Tensor, labels []float64) (*tensor.Tensor, error) {
	pos := tensor.New(logits.Dtype(), logits.Shape()) // y
	neg := tensor.New(logits.Dtype(), logits.Shape()) // 1−y
	for k, y := range labels {
		if y < 0 || y > 1 {
			return nil, fmt.Errorf("nlp: reward label %d = %g outside [0,1]", k, y)
		}
		pos.SetF64(y, k, 0)
		neg.SetF64(1-y, k, 0)
	}
	negZ, err := exec1(ctx, backend.OpNeg, nil, logits)
	if err != nil {
		return nil, err
	}
	spNegZ, err := exec1(ctx, backend.OpSoftplus, nil, negZ) // −log σ(z)
	if err != nil {
		return nil, err
	}
	spZ, err := exec1(ctx, backend.OpSoftplus, nil, logits) // −log(1−σ(z))
	if err != nil {
		return nil, err
	}
	a, err := exec1(ctx, backend.OpMul, nil, pos, spNegZ)
	if err != nil {
		return nil, err
	}
	b, err := exec1(ctx, backend.OpMul, nil, neg, spZ)
	if err != nil {
		return nil, err
	}
	sum, err := exec1(ctx, backend.OpAdd, nil, a, b)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMean, nil, sum)
}
