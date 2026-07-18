package nn

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Losses (§T16). Both return scalar tensors and are fully differentiable when
// executed through a recording context.

// MSE is the mean squared error mean((pred−target)²), composed from existing
// ops — its gradient comes entirely from the Sub/Mul/Mean VJPs (§T14).
func MSE(ctx *backend.Context, pred, target *tensor.Tensor) (*tensor.Tensor, error) {
	diff, err := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{pred, target}, nil)
	if err != nil {
		return nil, err
	}
	sq, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
	if err != nil {
		return nil, err
	}
	out, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{sq[0]}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// CEOption configures CrossEntropy via the functional-options idiom: the required
// tensors stay positional and every loss variant is opt-in, so existing two-argument
// calls keep their exact previous behaviour (hard labels, mean over the whole batch).
type CEOption func(*backend.CrossEntropyAttrs)

// WithIgnoreIndex masks out every row whose target equals idx: it contributes nothing to
// the loss and receives exactly zero gradient, and (under the default mean reduction) it
// is excluded from the divisor. This is torch's `ignore_index`, and idx=-100 is the
// convention every instruction-tuning and masked-LM pipeline uses to mask prompt or pad
// positions — without it those workflows cannot be expressed at all.
//
// The masking composes with WithLabelSmoothing exactly as torch does: smoothing applies
// only to the surviving rows, and the mean divides by the surviving-row count.
//
// A batch in which EVERY row is ignored follows torch: NaN under Reduction mean (0/0),
// 0 under sum, all-zeros under none — and zero gradients in all three cases.
func WithIgnoreIndex(idx int) CEOption {
	return func(a *backend.CrossEntropyAttrs) {
		a.IgnoreIndex, a.HasIgnoreIndex = idx, true
	}
}

// WithReduction selects how the per-row losses collapse: backend.ReductionMean (the
// default and the historical behaviour), backend.ReductionSum, or backend.ReductionNone,
// which returns a [batch] tensor of per-row losses with 0 in every ignored slot. Per-row
// losses are what you need to weight examples, or to reduce over a custom mask yourself.
func WithReduction(r backend.Reduction) CEOption {
	return func(a *backend.CrossEntropyAttrs) { a.Reduction = r }
}

// WithLabelSmoothing sets ε∈[0,1) for label smoothing (Szegedy et al. 2016; Vaswani et al.
// 2017 §5.4) — the option form of CrossEntropySmooth, so smoothing can be combined with
// masking and a reduction in one call.
func WithLabelSmoothing(epsilon float64) CEOption {
	return func(a *backend.CrossEntropyAttrs) { a.LabelSmoothing = epsilon }
}

// WithZLoss sets the PaLM z-loss coefficient (§R113) — the option form of
// CrossEntropyZLoss, combinable with masking and a reduction.
func WithZLoss(coeff float64) CEOption {
	return func(a *backend.CrossEntropyAttrs) { a.ZLoss = coeff }
}

// CrossEntropy is the fused, numerically stable softmax cross-entropy over
// logits[batch,classes] with class-index targets[batch] (ADR-0007). With no options it is
// the mean over the batch with hard labels, exactly as before; targets are
// non-differentiable throughout.
//
// Options cover torch's cross_entropy surface — WithIgnoreIndex to mask prompt/pad rows,
// WithReduction for mean|sum|none, plus WithLabelSmoothing and WithZLoss:
//
//	loss, err := nn.CrossEntropy(ctx, logits, targets,
//	    nn.WithIgnoreIndex(-100), nn.WithReduction(backend.ReductionSum))
//
// Output shape is a scalar for mean and sum, and [batch] for ReductionNone.
func CrossEntropy(ctx *backend.Context, logits, targets *tensor.Tensor, opts ...CEOption) (*tensor.Tensor, error) {
	var attrs backend.Attrs // nil for the zero-option case: byte-identical to the old call
	if len(opts) > 0 {
		var a backend.CrossEntropyAttrs
		for _, o := range opts {
			o(&a)
		}
		attrs = a
	}
	out, err := backend.Execute(ctx, backend.OpCrossEntropy, []*tensor.Tensor{logits, targets}, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// CrossEntropySmooth is cross-entropy with label smoothing (Szegedy et al. 2016;
// used in the original Transformer, Vaswani et al. 2017 §5.4, with ε=0.1). Instead
// of a one-hot target it regresses toward q'(k)=(1−ε)·δ(k,target)+ε/K, mixing in
// a little of the uniform distribution over all K classes. This discourages the
// network from becoming over-confident (huge logits for the correct class),
// improving calibration and generalization. ε∈[0,1); ε=0 is ordinary
// CrossEntropy. Mean over the batch; targets are non-differentiable (§R52).
func CrossEntropySmooth(ctx *backend.Context, logits, targets *tensor.Tensor, epsilon float64) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpCrossEntropy,
		[]*tensor.Tensor{logits, targets}, backend.CrossEntropyAttrs{LabelSmoothing: epsilon})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// CrossEntropyZLoss is cross-entropy plus the PaLM z-loss (Chowdhery et al. 2022, §R113; origin the
// Mesh-TensorFlow softmax z-loss, Shazeer 2018): it adds zCoeff·(logsumexp(logits))² per token to
// the loss, penalising the softmax log-partition log Z so it stays near 0. This keeps the logits
// from drifting/exploding and markedly improves large-model bf16/fp16 training stability — a
// standard LLM-pretraining stabilizer (PaLM, ST-MoE, Baichuan, OLMo 2), distinct from label
// smoothing and weight decay. Its gradient is 2·zCoeff·log Z·softmax(logits) per token. The paper's
// default zCoeff is 1e-4; 0 recovers ordinary CrossEntropy. Mean over the batch.
func CrossEntropyZLoss(ctx *backend.Context, logits, targets *tensor.Tensor, zCoeff float64) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpCrossEntropy,
		[]*tensor.Tensor{logits, targets}, backend.CrossEntropyAttrs{ZLoss: zCoeff})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// ZLoss is the standalone log-Z softmax regularizer coeff·mean(logsumexp(logits)²) over the rows of
// a [batch,classes] logit matrix (§R113) — the same penalty CrossEntropyZLoss fuses into
// cross-entropy, but on ARBITRARY logits with no targets. Its primary use is the ST-MoE router
// z-loss (Zoph et al. 2022): feed the MoE router gate logits (SparseMoE.Forward's second return) to
// keep the router's log-partition near 0 alongside the load-balancing loss, stabilising sparse
// training. Coefficient coeff (ST-MoE router default 1e-3, PaLM output 1e-4); its gradient is
// 2·coeff·logZ·softmax per row. Mean over the batch; differentiable through the tape.
func ZLoss(ctx *backend.Context, logits *tensor.Tensor, coeff float64) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpZLoss, []*tensor.Tensor{logits}, backend.ZLossAttrs{Coeff: coeff})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// PrefOption configures a preference-alignment loss (DPO, IPO, KTO) via the
// functional-options idiom — the required log-prob tensors stay positional, the
// hyperparameters are optional. Beta applies to all three; ReferencePoint,
// DesirableWeight and UndesirableWeight are KTO-only.
type PrefOption func(*prefConfig)

type prefConfig struct {
	beta     float64
	zRef     float64
	lambdaD  float64
	lambdaU  float64
	gamma    float64 // SimPO target reward margin
	lambdaOR float64 // ORPO odds-ratio weight
	cpoAlpha float64 // CPO NLL/BC regularizer weight (TRL cpo_alpha)
}

func newPrefConfig(opts []PrefOption) prefConfig {
	c := prefConfig{beta: 0.1, zRef: 0, lambdaD: 1, lambdaU: 1} // paper defaults
	for _, o := range opts {
		o(&c)
	}
	return c
}

// Beta sets the KL/regularization strength β (default 0.1) for DPO, IPO and KTO.
// A non-positive value is ignored (keeps the default).
func Beta(b float64) PrefOption {
	return func(c *prefConfig) {
		if b > 0 {
			c.beta = b
		}
	}
}

// ReferencePoint sets KTO's detached reward-unit reference point zRef = β·KL
// (default 0). Any value is accepted (0 is valid).
func ReferencePoint(z float64) PrefOption { return func(c *prefConfig) { c.zRef = z } }

// DesirableWeight sets KTO's λ_D (default 1); non-positive is ignored.
func DesirableWeight(w float64) PrefOption {
	return func(c *prefConfig) {
		if w > 0 {
			c.lambdaD = w
		}
	}
}

// UndesirableWeight sets KTO's λ_U (default 1); non-positive is ignored.
func UndesirableWeight(w float64) PrefOption {
	return func(c *prefConfig) {
		if w > 0 {
			c.lambdaU = w
		}
	}
}

// Gamma sets SimPO's target reward margin γ (default 1). Any value is accepted.
func Gamma(g float64) PrefOption { return func(c *prefConfig) { c.gamma = g } }

// Lambda sets ORPO's odds-ratio weight λ (default 0.1); non-positive is ignored.
func Lambda(l float64) PrefOption {
	return func(c *prefConfig) {
		if l > 0 {
			c.lambdaOR = l
		}
	}
}

// CPOAlpha sets CPO's NLL/behavior-cloning regularizer weight λ (TRL cpo_alpha,
// default 1). 0 is valid (drops the BC term → bare reference-free preference);
// only negative values are ignored.
func CPOAlpha(a float64) PrefOption {
	return func(c *prefConfig) {
		if a >= 0 {
			c.cpoAlpha = a
		}
	}
}

// DPO is the Direct Preference Optimization loss (Rafailov et al. 2023): it aligns
// a policy to human preferences directly from (chosen, rejected) pairs, with NO
// reward model and NO RL (the RLHF-free alternative). Each argument is a [batch]
// tensor of SEQUENCE log-probabilities — the sum of per-token log-probs of a whole
// response — for the policy and the frozen reference model on the chosen (y_w) and
// rejected (y_l) responses:
//
//	loss = mean −log σ( β·( (πθ_w − πref_w) − (πθ_l − πref_l) ) )
//
// β (temperature, default 0.1, set with Beta) controls how hard the KL to the
// reference is enforced. The reference log-probs are treated as constants (frozen
// model), so gradients flow only into the policy log-probs. Fused + stable
// (§R48, §V16). Pair it with SequenceLogProbs to get the inputs from logits.
func DPO(ctx *backend.Context, policyChosen, policyRejected, refChosen, refRejected *tensor.Tensor, opts ...PrefOption) (*tensor.Tensor, error) {
	cfg := newPrefConfig(opts)
	out, err := backend.Execute(ctx, backend.OpDPO,
		[]*tensor.Tensor{policyChosen, policyRejected, refChosen, refRejected},
		backend.DPOAttrs{Beta: cfg.beta})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// IPO is the Identity Preference Optimization loss (Azar et al. 2023) — a DPO
// variant that replaces the logistic loss with a SQUARED loss toward a finite
// target margin, so it regularizes to 1/(2β) instead of pushing the chosen/
// rejected gap to infinity (DPO's overfitting failure mode on near-deterministic
// preferences). Same inputs as DPO — [batch] sequence log-probs for the policy
// and the frozen reference on the chosen (y_w) and rejected (y_l) responses:
//
//	loss = mean( ((πθ_w−πref_w) − (πθ_l−πref_l)) − 1/(2β) )²
//
// β (regularization strength, default 0.1, set with Beta) sets the target margin.
// The reference log-probs are frozen; gradients flow only into the policy (§R58).
func IPO(ctx *backend.Context, policyChosen, policyRejected, refChosen, refRejected *tensor.Tensor, opts ...PrefOption) (*tensor.Tensor, error) {
	cfg := newPrefConfig(opts)
	out, err := backend.Execute(ctx, backend.OpIPO,
		[]*tensor.Tensor{policyChosen, policyRejected, refChosen, refRejected},
		backend.IPOAttrs{Beta: cfg.beta})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// KTO is the Kahneman-Tversky Optimization loss (Ethayarajh et al. 2024). Unlike
// DPO/IPO it aligns from UNPAIRED binary feedback: every example is simply labeled
// desirable or undesirable (no chosen/rejected pairs), which is far cheaper to
// collect. Inputs are [batch]: the policy and frozen-reference sequence log-probs
// and per-example labels (nonzero = desirable). With reward r = β·(logπθ−logπref)
// and a detached reference point zRef in reward units — zRef = β·KL_batch, the
// β-scaled batch KL estimate the caller supplies (matches TRL's σ(β·kl − r)):
//
//	desirable   loss = λ_D·σ(zRef − r)   (pushes the reward up)
//	undesirable loss = λ_U·σ(r − zRef)   (pushes the reward down)
//
// Configure with Beta (default 0.1), ReferencePoint (zRef = β·KL, default 0) and
// DesirableWeight / UndesirableWeight (λ_D, λ_U, default 1) — the prospect-theory
// value's parameters. The reference is frozen and zRef is a constant; gradients
// flow only into the policy (§R59). E.g.:
//
//	nn.KTO(ctx, pl, rl, labels, nn.Beta(0.1), nn.ReferencePoint(z))
func KTO(ctx *backend.Context, policyLogps, refLogps, labels *tensor.Tensor, opts ...PrefOption) (*tensor.Tensor, error) {
	cfg := newPrefConfig(opts)
	out, err := backend.Execute(ctx, backend.OpKTO,
		[]*tensor.Tensor{policyLogps, refLogps, labels},
		backend.KTOAttrs{Beta: cfg.beta, ZRef: cfg.zRef, LambdaD: cfg.lambdaD, LambdaU: cfg.lambdaU})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// SimPO is the Simple Preference Optimization loss (Meng et al. 2024) — a
// reference-free, length-normalized alternative to DPO. The two arguments are
// [batch] tensors of the AVERAGE (length-normalized) token log-probabilities of
// the chosen and rejected responses under the policy — no reference model:
//
//	loss = mean −log σ( β·(avgChosen − avgRejected) − γ )
//
// The length-normalized reward β·avglogp removes DPO's length bias, and the
// target margin γ (subtracted inside the sigmoid) requires the chosen response to
// win by a margin. β (default 2, set with Beta) and γ (default 1, set with Gamma).
// Gradients flow into both log-prob vectors (§C12, §R71).
func SimPO(ctx *backend.Context, policyChosenAvg, policyRejectedAvg *tensor.Tensor, opts ...PrefOption) (*tensor.Tensor, error) {
	cfg := prefConfig{beta: 2, gamma: 1} // SimPO paper defaults
	for _, o := range opts {
		o(&cfg)
	}
	out, err := backend.Execute(ctx, backend.OpSimPO,
		[]*tensor.Tensor{policyChosenAvg, policyRejectedAvg},
		backend.SimPOAttrs{Beta: cfg.beta, Gamma: cfg.gamma})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// ORPO is the Odds Ratio Preference Optimization loss (Hong et al. 2024) — a
// monolithic objective that folds supervised fine-tuning and preference alignment
// into one term, with NO reference model and NO separate SFT phase. The two
// arguments are [batch] tensors of the AVERAGE token log-probabilities of the
// chosen and rejected responses:
//
//	P     = exp(avg log-prob)                       // length-normalized seq prob
//	logOR = (avgChosen − avgRejected) − (log(1−Pc) − log(1−Pr))
//	loss  = mean( −avgChosen + λ·(−log σ(logOR)) )
//
// The first term is the SFT negative-log-likelihood on the chosen response; the
// second rewards higher odds for chosen over rejected. λ (default 0.1, set with
// Lambda) weights the odds-ratio term. Requires average log-probs < 0 (P < 1).
// Gradients flow into both log-prob vectors (§C12, §R71).
func ORPO(ctx *backend.Context, policyChosenAvg, policyRejectedAvg *tensor.Tensor, opts ...PrefOption) (*tensor.Tensor, error) {
	cfg := prefConfig{lambdaOR: 0.1} // ORPO paper default
	for _, o := range opts {
		o(&cfg)
	}
	out, err := backend.Execute(ctx, backend.OpORPO,
		[]*tensor.Tensor{policyChosenAvg, policyRejectedAvg},
		backend.ORPOAttrs{Lambda: cfg.lambdaOR})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// CPO is the Contrastive Preference Optimization loss (Xu et al. 2024,
// arXiv:2401.08417) — a reference-free objective that folds preference alignment
// and SFT into one term, with NO reference model and NO separate SFT phase (its
// origin task was machine translation, but it is a general alignment loss). The
// two arguments are [batch] tensors of the SUMMED sequence log-probabilities
// (DPO-style, not length-normalized) of the chosen and rejected responses under
// the policy:
//
//	loss = mean −log σ( β·(logπ_chosen − logπ_rejected) ) + λ·(−logπ_chosen)
//
// The first term is the reference-free Bradley-Terry preference (DPO with the
// reference dropped as a uniform prior); the second is the negative-log-likelihood
// behavior-cloning regularizer on the chosen response that keeps the policy close
// to the preferred data. β (reward scale, default 0.1, set with Beta) and λ
// (BC weight, TRL cpo_alpha, default 1, set with CPOAlpha; λ=0 leaves the bare
// preference term). Gradients flow into both log-prob vectors (§C12, §R121).
// Pair it with SequenceLogProbs to get the summed log-probs from logits.
func CPO(ctx *backend.Context, policyChosen, policyRejected *tensor.Tensor, opts ...PrefOption) (*tensor.Tensor, error) {
	cfg := prefConfig{beta: 0.1, cpoAlpha: 1} // CPO paper / TRL defaults
	for _, o := range opts {
		o(&cfg)
	}
	out, err := backend.Execute(ctx, backend.OpCPO,
		[]*tensor.Tensor{policyChosen, policyRejected},
		backend.CPOAttrs{Beta: cfg.beta, Alpha: cfg.cpoAlpha})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// PPOClipLoss is the Proximal Policy Optimization clipped surrogate policy loss
// (Schulman et al. 2017) — the core policy objective of RLHF. Given the current
// policy log-probs logπθ(a|s), the frozen rollout log-probs logπ_old, and the
// advantage estimates Â (all [batch]), it returns
//
//	loss = −mean min( r·Â , clip(r, 1−ε, 1+ε)·Â ),   r = exp(logπθ − logπ_old)
//
// (the paper MAXIMIZES the surrogate, so this loss is its negation). The clip
// forms a trust region: once the ratio leaves [1−ε, 1+ε] in the pessimistic
// direction its gradient is zero, so a single batch cannot push the policy too
// far. epsilon defaults to 0.2 (epsilon≤0 → 0.2). logπ_old and Â are constants
// (rollout time); gradients flow only into logπθ. Value-function and entropy
// terms are separate and added by the caller (§R49, §V16).
func PPOClipLoss(ctx *backend.Context, logpNew, logpOld, advantage *tensor.Tensor, epsilon float64) (*tensor.Tensor, error) {
	if epsilon <= 0 {
		epsilon = 0.2
	}
	out, err := backend.Execute(ctx, backend.OpPPOClip,
		[]*tensor.Tensor{logpNew, logpOld, advantage},
		backend.PPOClipAttrs{Epsilon: epsilon})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// DistillLoss is the soft-target knowledge-distillation loss (Hinton et al. 2015)
// — the standard way to compress a large "teacher" model into a small "student".
// It matches the student's softened output distribution to the teacher's over
// logits[batch, classes]:
//
//	loss = mean_b T²·KL( softmax(teacher_b/T) ‖ softmax(student_b/T) )
//
// The temperature T > 1 softens both distributions so the student learns the
// teacher's "dark knowledge" (relative class probabilities), and the T² factor
// keeps the gradient magnitude comparable to a hard-label loss. The teacher is
// frozen (nil gradient). temperature defaults to 1 (temperature≤0 is an error).
// For the full recipe combine it with CrossEntropy on the true labels:
// α·DistillLoss + (1−α)·CrossEntropy (§R51, §V16).
func DistillLoss(ctx *backend.Context, studentLogits, teacherLogits *tensor.Tensor, temperature float64) (*tensor.Tensor, error) {
	if temperature <= 0 {
		temperature = 1
	}
	out, err := backend.Execute(ctx, backend.OpDistill,
		[]*tensor.Tensor{studentLogits, teacherLogits},
		backend.DistillAttrs{Temperature: temperature})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
