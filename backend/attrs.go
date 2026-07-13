package backend

import (
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Attrs is the typed parameter set an op carries through Execute, onto the
// autograd tape, and into its VJP. It is a sealed interface: each op that takes
// parameters has exactly one concrete implementation (e.g. AttnAttrs for OpMHA,
// NormAttrs for OpLayerNorm), and ops without parameters pass a nil Attrs.
//
// This replaces the former map[string]any bag (ADR-0014): construction is now a
// checked struct literal instead of stringly-typed keys, and a kernel reads its
// parameters by type-asserting to its op's struct —
//
//	p, _ := attrs.(backend.AttnAttrs) // zero value when nil or mismatched
//	p = p.WithDefaults()              // fill non-zero defaults (heads=1, scale=1…)
//
// so a wrong field name or type is a compile error, and each struct's fields are
// the op's self-documenting, godoc-rendered contract.
//
// To add parameters for a new op: declare a `FooAttrs` struct here, give it the
// unexported opAttrs() marker method so it satisfies Attrs, and (if any default is
// non-zero) a WithDefaults method that fills them. Kernel and VJP share that one
// WithDefaults so their defaults can never drift apart.
type Attrs interface{ opAttrs() }

// ReduceAll is the ArgMaxAttrs.Axis sentinel meaning "flatten every axis and
// return one global arg-max index" (the behaviour of a nil Attrs). It is the most
// negative int so it can never collide with a real axis.
const ReduceAll = math.MinInt

// AttnAttrs parameterises fused multi-head attention (OpMHA) and its
// online-softmax-tiled twin (OpFlashAttn). The two ops share this struct because
// they share a backward pass (mhaVJP); each op ignores the fields that do not
// apply to it (OpFlashAttn ignores Scale/ALiBi/Window; OpMHA ignores Block).
type AttnAttrs struct {
	Heads   int     // number of query heads; 0 → 1
	KVHeads int     // key/value heads for GQA/MQA; 0 → Heads (standard MHA)
	Causal  bool    // apply the autoregressive j>i mask
	Scale   float64 // extra pre-softmax score multiplier (YaRN attention temperature, see YaRNAttnScale); 0 → 1
	ALiBi   bool    // add the static ALiBi per-head linear distance bias
	Window  int     // sliding-window width (keys within Window of the query); 0 → unlimited
	Block   int     // OpFlashAttn tile size; 0 → a backend default
}

func (AttnAttrs) opAttrs() {}

// WithDefaults returns a copy with the zero-valued knobs replaced by their
// documented defaults (Heads→1, KVHeads→Heads, Scale→1).
func (a AttnAttrs) WithDefaults() AttnAttrs {
	if a.Heads == 0 {
		a.Heads = 1
	}
	if a.KVHeads == 0 {
		a.KVHeads = a.Heads
	}
	if a.Scale == 0 {
		a.Scale = 1
	}
	return a
}

// MLAAttrs parameterises multi-head latent attention (OpMLA).
type MLAAttrs struct {
	Heads    int     // number of attention heads; 0 → 1
	Causal   bool    // apply the autoregressive mask
	RoPEBase float64 // RoPE frequency base θ for the decoupled rotary key; 0 → 10000
}

func (MLAAttrs) opAttrs() {}

// WithDefaults returns a copy with Heads→1 and RoPEBase→10000 applied.
func (a MLAAttrs) WithDefaults() MLAAttrs {
	if a.Heads == 0 {
		a.Heads = 1
	}
	if a.RoPEBase == 0 {
		a.RoPEBase = 10000
	}
	return a
}

// ConvAttrs parameterises 2-D convolution (OpConv2D).
type ConvAttrs struct {
	Stride int // window step in both spatial dims; 0 → 1
	Pad    int // zero-padding added on every spatial edge
}

func (ConvAttrs) opAttrs() {}

// WithDefaults returns a copy with Stride→1 applied.
func (a ConvAttrs) WithDefaults() ConvAttrs {
	if a.Stride == 0 {
		a.Stride = 1
	}
	return a
}

// PoolAttrs parameterises 2-D max/average pooling (OpMaxPool2D, OpAvgPool2D).
type PoolAttrs struct {
	Kernel int // pooling window size
	Stride int // window step; 0 → Kernel (non-overlapping windows)
}

func (PoolAttrs) opAttrs() {}

// WithDefaults returns a copy with Stride→Kernel applied.
func (a PoolAttrs) WithDefaults() PoolAttrs {
	if a.Stride == 0 {
		a.Stride = a.Kernel
	}
	return a
}

// ReduceAttrs parameterises the axis-reductions (OpSum, OpMean, OpMax, OpMin).
// Its zero value (Axes nil, KeepDims false) already means "reduce over every axis
// to a scalar", so no defaulting is needed.
type ReduceAttrs struct {
	Axes     []int // axes to reduce; nil → all axes
	KeepDims bool  // keep reduced axes as length-1 instead of dropping them
}

func (ReduceAttrs) opAttrs() {}

// ArgMaxAttrs parameterises OpArgMax. A nil Attrs (or the ReduceAll sentinel)
// means the global arg-max over the flattened tensor.
type ArgMaxAttrs struct {
	Axis int // axis to take the arg-max along; ReduceAll → flatten everything
}

func (ArgMaxAttrs) opAttrs() {}

// NormAttrs parameterises the normalization layers (OpLayerNorm, OpRMSNorm).
type NormAttrs struct {
	Eps float64 // variance floor added before the reciprocal-sqrt; 0 → 1e-5
}

func (NormAttrs) opAttrs() {}

// WithDefaults returns a copy with Eps→1e-5 applied.
func (a NormAttrs) WithDefaults() NormAttrs {
	if a.Eps == 0 {
		a.Eps = 1e-5
	}
	return a
}

// RoPEAttrs parameterises rotary position embedding (OpRoPE), including linear
// position interpolation (PosScale) and YaRN NTK-by-parts scaling (the YaRN*
// fields). Leaving the YaRN fields zero disables YaRN.
type RoPEAttrs struct {
	Base         float64 // frequency base θ; 0 → 10000
	PosScale     float64 // linear position-interpolation divisor (PI); 0 → 1 (no interpolation)
	YaRNScale    float64 // YaRN context-extension factor s; 0 → YaRN off
	YaRNOrigCtx  float64 // original training context length YaRN extends from; 0 → YaRN off
	YaRNBetaFast float64 // YaRN high-frequency correction boundary; 0 → 32
	YaRNBetaSlow float64 // YaRN low-frequency correction boundary; 0 → 1
	// Heads > 1 treats the input as [seq, Heads·headDim] and rotates each head's
	// headDim slice independently (multi-head attention); 0 or 1 → a single head over
	// the whole width. The rotary frequencies use the per-head dim, not the full width.
	Heads int
	// PosOffset shifts the absolute positions: row p is rotated for position
	// PosOffset+p. 0 = positions 0..seq−1 (full-sequence forward); a KV-cache decode
	// step passes the current cache length so its single new token rotates at its true
	// position.
	PosOffset int
	// XPos enables the length-extrapolatable xPos scaling (Sun et al. 2022, §R125): after
	// the (unchanged) RoPE rotation, each pair i is multiplied by ζ_i^(±n), n the position,
	// ζ_i=(i/(hd/2)+γ)/(1+γ). Queries use +n (XPosDownscale false), keys −n (true), so the
	// attention score picks up an exponential decay ζ^(n−m) in the relative distance.
	XPos bool
	// XPosGamma is the xPos γ constant; 0 → 0.4 (the torchscale/RetNet default). Only used
	// when XPos is set.
	XPosGamma float64
	// XPosDownscale selects the key path (ζ^(−n)); false is the query path (ζ^(+n)).
	XPosDownscale bool
}

func (RoPEAttrs) opAttrs() {}

// WithDefaults returns a copy with Base→10000, PosScale→1, YaRNBetaFast→32 and
// YaRNBetaSlow→1 applied. The YaRNScale/YaRNOrigCtx zero values are meaningful
// (they switch YaRN off) and are left untouched.
func (a RoPEAttrs) WithDefaults() RoPEAttrs {
	if a.Base == 0 {
		a.Base = 10000
	}
	if a.PosScale == 0 {
		a.PosScale = 1
	}
	if a.YaRNBetaFast == 0 {
		a.YaRNBetaFast = 32
	}
	if a.YaRNBetaSlow == 0 {
		a.YaRNBetaSlow = 1
	}
	if a.XPos && a.XPosGamma == 0 {
		a.XPosGamma = 0.4
	}
	return a
}

// AXPYAttrs parameterises the scaled vector add y += α·x (OpAXPY).
type AXPYAttrs struct {
	Alpha float64 // scale applied to x before the add; 0 → 1
}

func (AXPYAttrs) opAttrs() {}

// WithDefaults returns a copy with Alpha→1 applied.
func (a AXPYAttrs) WithDefaults() AXPYAttrs {
	if a.Alpha == 0 {
		a.Alpha = 1
	}
	return a
}

// CrossEntropyAttrs parameterises softmax cross-entropy (OpCrossEntropy). Its zero
// value (no label smoothing, no z-loss) is the default, so no defaulting is needed.
type CrossEntropyAttrs struct {
	LabelSmoothing float64 // ε in (0,1) mixing the target with the uniform distribution; 0 → hard targets
	ZLoss          float64 // z-loss coefficient: adds coeff·(logsumexp(logits))² per token (PaLM, §R113); 0 → off
}

func (CrossEntropyAttrs) opAttrs() {}

// ZLossAttrs parameterises the standalone log-Z regularizer (OpZLoss): loss = Coeff·mean over rows
// of (logsumexp(logits))². Its zero value (Coeff 0) yields a zero loss, so no defaulting is needed.
type ZLossAttrs struct {
	Coeff float64 // z-loss coefficient (ST-MoE router default 1e-3, PaLM output 1e-4, §R113)
}

func (ZLossAttrs) opAttrs() {}

// RetentionAttrs parameterises RetNet retention (OpRetention): Gamma is the per-head decay γ of the
// causal mask D_nm=γ^(n−m) (§R114). Its zero value (Gamma 0) collapses retention to the diagonal.
type RetentionAttrs struct {
	Gamma float64 // decay γ ∈ [0,1] (MSR per-head 1−2^(−5−h); 1 = no decay = causal cumulative)
}

func (RetentionAttrs) opAttrs() {}

// DistillAttrs parameterises knowledge-distillation loss (OpDistill).
type DistillAttrs struct {
	Temperature float64 // softmax temperature applied to both logits; 0 → 1
}

func (DistillAttrs) opAttrs() {}

// WithDefaults returns a copy with Temperature→1 applied.
func (a DistillAttrs) WithDefaults() DistillAttrs {
	if a.Temperature == 0 {
		a.Temperature = 1
	}
	return a
}

// DPOAttrs parameterises Direct Preference Optimization loss (OpDPO).
type DPOAttrs struct {
	Beta float64 // KL strength β trading off reward vs. staying near the reference; 0 → 0.1
}

func (DPOAttrs) opAttrs() {}

// WithDefaults returns a copy with Beta→0.1 applied.
func (a DPOAttrs) WithDefaults() DPOAttrs {
	if a.Beta == 0 {
		a.Beta = 0.1
	}
	return a
}

// IPOAttrs parameterises Identity Preference Optimization loss (OpIPO).
type IPOAttrs struct {
	Beta float64 // regularization strength β (IPO targets a margin of 1/(2β)); 0 → 0.1
}

func (IPOAttrs) opAttrs() {}

// WithDefaults returns a copy with Beta→0.1 applied.
func (a IPOAttrs) WithDefaults() IPOAttrs {
	if a.Beta == 0 {
		a.Beta = 0.1
	}
	return a
}

// KTOAttrs parameterises Kahneman-Tversky Optimization loss (OpKTO).
type KTOAttrs struct {
	Beta    float64 // KL strength β; 0 → 0.1
	ZRef    float64 // reference KL point z_ref; defaults to 0
	LambdaD float64 // desirable-example loss weight λ_D; 0 → 1
	LambdaU float64 // undesirable-example loss weight λ_U; 0 → 1
}

func (KTOAttrs) opAttrs() {}

// WithDefaults returns a copy with Beta→0.1, LambdaD→1 and LambdaU→1 applied.
func (a KTOAttrs) WithDefaults() KTOAttrs {
	if a.Beta == 0 {
		a.Beta = 0.1
	}
	if a.LambdaD == 0 {
		a.LambdaD = 1
	}
	if a.LambdaU == 0 {
		a.LambdaU = 1
	}
	return a
}

// PPOClipAttrs parameterises the PPO clipped surrogate loss (OpPPOClip).
type PPOClipAttrs struct {
	Epsilon float64 // clip range ε around a probability ratio of 1; 0 → 0.2
}

func (PPOClipAttrs) opAttrs() {}

// WithDefaults returns a copy with Epsilon→0.2 applied.
func (a PPOClipAttrs) WithDefaults() PPOClipAttrs {
	if a.Epsilon == 0 {
		a.Epsilon = 0.2
	}
	return a
}

// GSPOAttrs parameterises Group Sequence Policy Optimization loss (OpGSPO, §T549).
type GSPOAttrs struct {
	// Epsilon is the sequence-level clip range ε; 0 → 3e-4. GSPO's ratios are
	// length-normalized sequence likelihoods, which concentrate tightly around 1,
	// so the paper clips orders of magnitude tighter than PPO/GRPO's 0.2.
	Epsilon float64
	// Lengths[i] is the token count of sequence i; the flat [batch] inputs are the
	// concatenation of the G sequences' per-token log-probs (Σ Lengths = batch).
	Lengths []int
}

func (GSPOAttrs) opAttrs() {}

// WithDefaults returns a copy with Epsilon→3e-4 applied.
func (a GSPOAttrs) WithDefaults() GSPOAttrs {
	if a.Epsilon == 0 {
		a.Epsilon = 3e-4
	}
	return a
}

// GRPOAttrs parameterises Group Relative Policy Optimization loss (OpGRPO).
type GRPOAttrs struct {
	Epsilon float64 // PPO-style clip range ε; 0 → 0.2
	Beta    float64 // KL-penalty strength β against the reference policy; 0 → 0.04
}

func (GRPOAttrs) opAttrs() {}

// WithDefaults returns a copy with Epsilon→0.2 and Beta→0.04 applied.
func (a GRPOAttrs) WithDefaults() GRPOAttrs {
	if a.Epsilon == 0 {
		a.Epsilon = 0.2
	}
	if a.Beta == 0 {
		a.Beta = 0.04
	}
	return a
}

// SimPOAttrs parameterises Simple Preference Optimization loss (OpSimPO).
type SimPOAttrs struct {
	Beta  float64 // reward scale β on the length-normalized log-prob margin; 0 → 2
	Gamma float64 // target reward margin γ subtracted before the sigmoid; 0 → 1
}

func (SimPOAttrs) opAttrs() {}

// WithDefaults returns a copy with Beta→2 and Gamma→1 applied.
func (a SimPOAttrs) WithDefaults() SimPOAttrs {
	if a.Beta == 0 {
		a.Beta = 2
	}
	if a.Gamma == 0 {
		a.Gamma = 1
	}
	return a
}

// ORPOAttrs parameterises Odds-Ratio Preference Optimization loss (OpORPO).
type ORPOAttrs struct {
	Lambda float64 // weight λ on the odds-ratio term added to the SFT loss; 0 → 0.1
}

func (ORPOAttrs) opAttrs() {}

// WithDefaults returns a copy with Lambda→0.1 applied.
func (a ORPOAttrs) WithDefaults() ORPOAttrs {
	if a.Lambda == 0 {
		a.Lambda = 0.1
	}
	return a
}

// ConcatAttrs parameterises the concatenate op (OpConcat): the axis to join along.
type ConcatAttrs struct {
	Axis int // axis to concatenate along (0-based; negative counts from the end)
}

func (ConcatAttrs) opAttrs() {}

// SliceAttrs parameterises the slice op (OpSlice): extract the half-open range
// [Start,End) along Axis.
type SliceAttrs struct {
	Axis, Start, End int // axis (negative counts from the end) and the [Start,End) range
}

func (SliceAttrs) opAttrs() {}

// ReshapeAttrs parameterises the reshape op (OpReshape): the target shape, which
// must have the same element count as the input (row-major order preserved).
type ReshapeAttrs struct {
	Shape tensor.Shape // target shape; must have the same element count as the input
}

func (ReshapeAttrs) opAttrs() {}

// ClipAttrs parameterises the clip op (OpClip): clamp each element to [Lo,Hi].
type ClipAttrs struct {
	Lo, Hi float64 // inclusive lower and upper bounds (Lo ≤ Hi)
}

func (ClipAttrs) opAttrs() {}

// SoftCapAttrs parameterises the logit soft-cap op (OpSoftCap): y=Cap·tanh(x/Cap).
type SoftCapAttrs struct {
	Cap float64 // soft-cap magnitude; bounds output to (−Cap, Cap); must be > 0
}

func (SoftCapAttrs) opAttrs() {}

// BroadcastAttrs parameterises the broadcast op (OpBroadcast): the target shape to
// broadcast the input to (numpy.broadcast_to semantics).
type BroadcastAttrs struct {
	Shape tensor.Shape // target shape (right-aligned; each input dim must be 1 or match)
}

func (BroadcastAttrs) opAttrs() {}

// CumsumAttrs parameterises the cumulative-sum op (OpCumsum): the axis to accumulate
// along (negative counts from the end).
type CumsumAttrs struct {
	Axis int // axis to accumulate along (negative counts from the end)
}

func (CumsumAttrs) opAttrs() {}

// CPOAttrs parameterises Contrastive Preference Optimization loss (OpCPO).
type CPOAttrs struct {
	Beta  float64 // reward scale β on the summed log-prob margin; 0 → 0.1
	Alpha float64 // weight λ (TRL cpo_alpha) on the NLL/BC regularizer; <0 → 1
}

func (CPOAttrs) opAttrs() {}

// WithDefaults returns a copy with Beta→0.1 and Alpha→1 applied. A negative
// Alpha maps to 1; Alpha==0 is meaningful (drops the BC term → reference-free
// preference only) and is left untouched.
func (a CPOAttrs) WithDefaults() CPOAttrs {
	if a.Beta == 0 {
		a.Beta = 0.1
	}
	if a.Alpha < 0 {
		a.Alpha = 1
	}
	return a
}

// MoEBalanceAttrs parameterises the mixture-of-experts load-balancing auxiliary
// loss (OpMoEBalance).
type MoEBalanceAttrs struct {
	Alpha float64 // auxiliary-loss weight α; 0 → 0.01
}

func (MoEBalanceAttrs) opAttrs() {}

// WithDefaults returns a copy with Alpha→0.01 applied.
func (a MoEBalanceAttrs) WithDefaults() MoEBalanceAttrs {
	if a.Alpha == 0 {
		a.Alpha = 0.01
	}
	return a
}
