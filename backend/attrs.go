package backend

import (
	"fmt"
	"math"
	"reflect"

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
//	p, _ := attrs.(backend.AttnAttrs) // zero value when attrs is nil
//	p = p.WithDefaults()              // fill non-zero defaults (heads=1, scale=1…)
//
// so a wrong field name or type is a compile error, and each struct's fields are
// the op's self-documenting, godoc-rendered contract.
//
// Discarding the `ok` above is correct ONLY because Execute has already rejected a
// mismatched type (see opAttrsSpec and checkAttrs below). Left unchecked it is a
// silent-wrong-answer bug: Execute is public, so `Execute(ctx, OpSum, in,
// ConcatAttrs{Axis: 1})` used to hand OpSum's kernel the ZERO ReduceAttrs — which
// means "reduce every axis" — and return a scalar with err == nil instead of the
// per-axis sums the caller asked for. A nil Attrs stays legal (it is the convention
// for parameterless ops and for ops whose defaults suffice); only a NON-nil value of
// the wrong concrete type is refused.
//
// To add parameters for a new op: declare a `FooAttrs` struct here, give it the
// unexported opAttrs() marker method so it satisfies Attrs, and (if any default is
// non-zero) a WithDefaults method that fills them. Kernel and VJP share that one
// WithDefaults so their defaults can never drift apart. Then register the op in
// opAttrsSpec — the guard test derives the truth from the reference kernels and
// fails if you forget.
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
	// Alpha is the scale applied to x before the add. Its zero value means
	// "unset" and is defaulted to 1 (a caller who forgot to set Alpha still gets
	// a plain add). FOOTGUN: therefore a *computed* Alpha that happens to be 0
	// does NOT give y += 0·x — it silently becomes y += 1·x. If you need a scale
	// that can legitimately be 0 (e.g. a weight from a loop that may be zero),
	// do NOT use OpAXPY; use OpMul against a scalar/broadcast constant, which
	// zeroes both the value and (via its VJP) the gradient. Cf. CPOAttrs.Alpha,
	// which uses <0→1 precisely because 0 is meaningful there.
	Alpha float64
}

func (AXPYAttrs) opAttrs() {}

// WithDefaults returns a copy with Alpha→1 applied.
func (a AXPYAttrs) WithDefaults() AXPYAttrs {
	if a.Alpha == 0 {
		a.Alpha = 1
	}
	return a
}

// Reduction selects how a per-row loss is collapsed into the op's output. Its zero
// value is ReductionMean, which is what every loss in this package did before the
// field existed — so an unset Reduction never changes existing behaviour.
//
// The names mirror torch's `reduction=` string argument (torch is definitional here,
// §V16): "mean", "sum", "none".
type Reduction uint8

const (
	// ReductionMean averages the per-row losses (the zero value, and the historical
	// behaviour of every loss op). When rows are masked out via CrossEntropyAttrs'
	// IgnoreIndex the divisor is the NON-IGNORED row count, not the batch size —
	// so an all-ignored batch yields 0/0 = NaN, exactly as torch does.
	ReductionMean Reduction = iota
	// ReductionSum adds the per-row losses. An all-ignored batch yields 0.
	ReductionSum
	// ReductionNone skips the reduction entirely: the op returns a [batch] tensor of
	// per-row losses, with exactly 0 at every ignored row (torch's convention — the
	// ignored slots are kept in place rather than filtered out, so the result stays
	// aligned with the input batch).
	ReductionNone
)

// String renders the reduction with torch's spelling ("mean", "sum", "none").
func (r Reduction) String() string {
	switch r {
	case ReductionMean:
		return "mean"
	case ReductionSum:
		return "sum"
	case ReductionNone:
		return "none"
	}
	return fmt.Sprintf("Reduction(%d)", uint8(r))
}

// CrossEntropyAttrs parameterises softmax cross-entropy (OpCrossEntropy). Its zero
// value (no label smoothing, no z-loss, no ignored rows, mean reduction) is the
// default, so no defaulting is needed.
//
// IgnoreIndex/HasIgnoreIndex express torch's `ignore_index` (the -100 convention used
// by every instruction-tuning and masked-LM pipeline to mask prompt or pad positions).
// The flag is a separate bool rather than a magic sentinel value because 0 — and every
// other integer — is a legitimate class index someone may want to mask; and rather than
// a *int because Attrs are copied by value across the backend boundary (and into the
// op recorder), where a shared pointer would alias. Zero value = feature off, which is
// what keeps every existing caller byte-identical.
//
// Semantics when HasIgnoreIndex is set: a row whose target equals IgnoreIndex contributes
// EXACTLY nothing to the loss and receives EXACTLY zero gradient. Label smoothing applies
// only to the surviving rows, and ReductionMean divides by the surviving-row count — the
// intersection where hand-rolled implementations typically drift from torch.
type CrossEntropyAttrs struct {
	LabelSmoothing float64   // ε in (0,1) mixing the target with the uniform distribution; 0 → hard targets
	ZLoss          float64   // z-loss coefficient: adds coeff·(logsumexp(logits))² per token (PaLM, §R113); 0 → off
	IgnoreIndex    int       // target class to mask out; only consulted when HasIgnoreIndex (torch's ignore_index, conventionally -100)
	HasIgnoreIndex bool      // false (zero value) → no row is ignored
	Reduction      Reduction // how per-row losses collapse; zero value ReductionMean = historical behaviour
}

func (CrossEntropyAttrs) opAttrs() {}

// IsBasic reports whether every optional cross-entropy feature is off — plain hard-label
// mean cross-entropy over the whole batch. GPU kernels that implement ONLY that form
// (the metal/vulkan fused forward) gate on this instead of hand-listing fields, so
// adding a field to CrossEntropyAttrs cannot silently escape the fallback and produce a
// wrong GPU answer. The guard is pinned by the table-driven fallback invariant test.
func (a CrossEntropyAttrs) IsBasic() bool {
	return a == CrossEntropyAttrs{}
}

// IsUnmaskedMean reports whether no row is masked and the reduction is the historical
// mean. GPU kernels that DO implement label smoothing and z-loss (the metal/vulkan fused
// backward) gate on this: they handle every pre-existing field but neither of the row-set
// changing ones, which alter the per-row skip and the mean divisor.
func (a CrossEntropyAttrs) IsUnmaskedMean() bool {
	return !a.HasIgnoreIndex && a.Reduction == ReductionMean
}

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

// NumOps is one past the highest defined Op — the length of every per-op table in
// this package. Exported so external tests can sweep every op without guessing at
// the enum's end (Op.String falls back to "op(?)" for an op whose name entry was
// forgotten, which would silently truncate such a sweep).
const NumOps = int(numOps)

// attrsSpec is the Attrs contract of one op: which concrete Attrs types its kernels
// accept. The zero value — no types, not any — means "this op takes no parameters",
// so only a nil Attrs is legal for it.
type attrsSpec struct {
	// types lists the accepted concrete Attrs types. Ops that legitimately accept
	// more than one (none do today) list them all; the check passes on any match.
	types []reflect.Type
	// any waives the check for an op whose kernels genuinely inspect the dynamic
	// type themselves. It is the deliberate escape hatch that keeps the check
	// strict everywhere else instead of being weakened globally for one outlier.
	// Nothing uses it today; it exists so that the first op that needs it does not
	// tempt someone into deleting the check.
	any bool
}

// attrsOf builds an attrsSpec from sample values, so the table below reads as the
// Attrs structs themselves rather than as reflect boilerplate.
func attrsOf(samples ...Attrs) attrsSpec {
	ts := make([]reflect.Type, len(samples))
	for i, s := range samples {
		ts[i] = reflect.TypeOf(s)
	}
	return attrsSpec{types: ts}
}

// opAttrsSpec maps each op to the Attrs type its kernels assert. Execute consults it
// to turn a mismatched Attrs from a silent wrong answer into an error naming the op,
// the type passed, and the type expected.
//
// Entries omitted here are parameterless ops (add, matmul, softmax, embed…): they
// accept a nil Attrs and nothing else.
//
// KEEPING THIS HONEST is the whole problem — a table like this rots the moment
// someone adds an op and forgets it, and a stale entry would restore exactly the
// silent wrongness it exists to prevent. So it is not trusted: TestOpAttrsSpecMatchesKernels
// (backend/attrs_guard_test.go) recovers each op's reference kernel, walks the AST of
// the function that kernel actually is — following the attrs parameter through helper
// calls — and collects the `attrs.(backend.XAttrs)` assertions it really performs. A
// new parameterised op that is missing here, an entry naming a type its kernel does
// not assert, and an entry for an op whose kernel takes no parameters at all are each
// a test failure with the correct line to write.
var opAttrsSpec = [numOps]attrsSpec{
	// reductions
	OpSum:    attrsOf(ReduceAttrs{}),
	OpMean:   attrsOf(ReduceAttrs{}),
	OpMax:    attrsOf(ReduceAttrs{}),
	OpMin:    attrsOf(ReduceAttrs{}),
	OpProd:   attrsOf(ReduceAttrs{}),
	OpArgMax: attrsOf(ArgMaxAttrs{}),

	// blas
	OpAXPY: attrsOf(AXPYAttrs{}),

	// nn
	OpCrossEntropy:         attrsOf(CrossEntropyAttrs{}),
	OpCrossEntropyBackward: attrsOf(CrossEntropyAttrs{}),
	OpLayerNorm:            attrsOf(NormAttrs{}),
	OpLayerNormBackward:    attrsOf(NormAttrs{}),
	OpRMSNorm:              attrsOf(NormAttrs{}),
	OpRMSNormBackward:      attrsOf(NormAttrs{}),
	OpRoPE:                 attrsOf(RoPEAttrs{}),
	OpRoPEBackward:         attrsOf(RoPEAttrs{}),

	// cv
	OpConv2D:         attrsOf(ConvAttrs{}),
	OpConv2DBackward: attrsOf(ConvAttrs{}),
	OpMaxPool2D:      attrsOf(PoolAttrs{}),
	OpAvgPool2D:      attrsOf(PoolAttrs{}),

	// attention (AttnAttrs is shared by the whole family — see its godoc)
	OpMHA:               attrsOf(AttnAttrs{}),
	OpMHABackward:       attrsOf(AttnAttrs{}),
	OpMHAMasked:         attrsOf(AttnAttrs{}),
	OpMHAMaskedBackward: attrsOf(AttnAttrs{}),
	OpMHASelect:         attrsOf(AttnAttrs{}),
	OpFlashAttn:         attrsOf(AttnAttrs{}),
	OpMLA:               attrsOf(MLAAttrs{}),
	OpRetention:         attrsOf(RetentionAttrs{}),
	OpRetentionBackward: attrsOf(RetentionAttrs{}),

	// losses / alignment
	OpDPO:     attrsOf(DPOAttrs{}),
	OpIPO:     attrsOf(IPOAttrs{}),
	OpKTO:     attrsOf(KTOAttrs{}),
	OpCPO:     attrsOf(CPOAttrs{}),
	OpSimPO:   attrsOf(SimPOAttrs{}),
	OpORPO:    attrsOf(ORPOAttrs{}),
	OpPPOClip: attrsOf(PPOClipAttrs{}),
	OpGRPO:    attrsOf(GRPOAttrs{}),
	OpGSPO:    attrsOf(GSPOAttrs{}),
	OpDistill: attrsOf(DistillAttrs{}),
	OpZLoss:   attrsOf(ZLossAttrs{}),

	// moe
	OpMoEBalance: attrsOf(MoEBalanceAttrs{}),

	// shape / elementwise
	OpConcat:    attrsOf(ConcatAttrs{}),
	OpSlice:     attrsOf(SliceAttrs{}),
	OpReshape:   attrsOf(ReshapeAttrs{}),
	OpBroadcast: attrsOf(BroadcastAttrs{}),
	OpCumsum:    attrsOf(CumsumAttrs{}),
	OpClip:      attrsOf(ClipAttrs{}),
	OpSoftCap:   attrsOf(SoftCapAttrs{}),
	OpEinsum:    attrsOf(EinsumAttrs{}),
}

// checkAttrs rejects a non-nil Attrs whose concrete type is not the one op's kernels
// assert. It is the guard behind the `pa, _ := attrs.(backend.FooAttrs)` idiom every
// kernel uses: that discarded `ok` makes a mismatch indistinguishable from the zero
// value, which is a WRONG ANSWER rather than a crash — OpSum handed a ConcatAttrs
// used to reduce every axis and report success. Callers who pass the right type (and
// callers who pass nil, still legal and still the documented default) never see this.
//
// Returns nil for an out-of-range op: that is not this check's failure to report, and
// kernel resolution in Execute names it accurately a few lines later.
func checkAttrs(op Op, attrs Attrs) error {
	if attrs == nil {
		// A nil Attrs is legal for EVERY op (Execute short-circuits before calling
		// this; repeated here so the function is correct in isolation).
		return nil
	}
	if op < 0 || int(op) >= len(opAttrsSpec) {
		return nil
	}
	spec := opAttrsSpec[op]
	if spec.any {
		return nil
	}
	got := reflect.TypeOf(attrs)
	for _, want := range spec.types {
		if got == want {
			return nil
		}
	}
	if len(spec.types) == 0 {
		return fmt.Errorf("backend: op %v takes no parameters, but Attrs of type %s was passed "+
			"(pass a nil Attrs)", op, got)
	}
	if len(spec.types) == 1 {
		return fmt.Errorf("backend: op %v wants Attrs of type %s, got %s", op, spec.types[0], got)
	}
	return fmt.Errorf("backend: op %v wants Attrs of one of %v, got %s", op, spec.types, got)
}
