package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// growKeyScale shrinks the Xavier init of key tokens appended by Grow in the
// DEFAULT (coupled) mode. The paper zero-initializes new key tokens for an exact
// checkpoint resume; a small NON-zero value keeps the perturbation of the shared
// per-row L2 denominator tiny (near-lossless resume) while still giving the new
// tokens a non-zero score so they receive gradient and train — a zero-gradient
// deadlock otherwise freezes both new K and V (see Grow).
const growKeyScale = 0.1

// PattentionLayer is a token-parameter attention ("Pattention") projection
// (Wang, Du, Ge, Li, Chen, Peng, Wang, Song, Yang & Shi 2024, "TokenFormer:
// Rethinking Transformer Scaling with Tokenized Model Parameters",
// arXiv:2410.23168). It is a drop-in replacement for a Linear layer whose weight
// is expressed as attention over LEARNABLE parameter tokens rather than a dense
// matrix, so the layer can be GROWN incrementally (append key/value token pairs)
// to add capacity while approximately preserving the current function — see Grow.
//
// Given input X[batch, dIn] the layer holds two learnable parameter matrices,
// nTokens "key" tokens K[nTokens, dIn] and "value" tokens V[nTokens, dOut] (both
// standalone parameters, NOT derived from X), and computes (paper Eq. 4):
//
//	S = X·Kᵀ            // [batch, nTokens] — score of X against each key token
//	A = Θ(S)            // nonlinear normalization (see below)
//	Y = A·V             // [batch, dOut]
//
// Normalization Θ — DEFAULT, coupled (paper Eq. 5, §C21). Attention's softmax is
// replaced by an L2-normalized GELU over the token axis, exactly as the paper
// writes it:
//
//	Θ(S)_ij = GELU( S_ij · τ / √(Σ_k S_ik² + ε) ),   τ = √n
//
// where the sum Σ_k runs over the n key tokens (the "token axis"), τ = √n is the
// scale from Eq. 5 (n = the token count at construction; see Tau), and ε is a
// tiny denominator floor (see Eps). GELU is the paper's nonlinearity. The row L2
// norm ‖S_i‖ COUPLES all tokens through a shared denominator (like softmax), which
// is what makes growth APPROXIMATE rather than exact: appending a token changes
// ‖S_i‖ and so perturbs every existing score. τ is frozen at the construction
// token count so that, in the limit of zero-scored new keys, the existing tokens'
// normalization is preserved (the basis of the paper's checkpoint-resume claim,
// Eq. 12-13) — Grow documents the exact bound.
//
// Normalization Θ — OPTION, uncoupled (WithPattentionUncoupledNorm, §C21). A
// per-score GELU with a fixed temperature that does NOT sum over the token axis:
//
//	Θ(S)_ij = GELU( S_ij · scale ),   scale = 1/√dIn  (frozen)
//
// This DIVERGES from Eq. 5's coupling: because each score is normalized
// independently, appending value-token rows initialized to ZERO leaves every
// existing score untouched and (since GELU(0)=0 and Y=A·V) contributes exactly
// nothing, so growth is BIT-EXACT. It trades faithfulness to Eq. 5 for an exact
// incremental-growth guarantee; choose it when exact resume matters more than
// matching the paper's normalizer.
//
// The whole forward pass is a composition of backend ops (transpose, matmul,
// square, sum, sqrt, div, GELU, matmul), each with a registered VJP, so gradients
// reach K and V with zero layer-specific autograd code — mirroring nn.Linear
// (§T5/§T13).
type PattentionLayer struct {
	K         *tensor.Tensor // [nTokens, dIn]  — learnable key tokens (grow-able)
	V         *tensor.Tensor // [nTokens, dOut] — learnable value tokens (grow-able)
	Uncoupled bool           // false → coupled Eq. 5 (default); true → per-score GELU
	Tau       float64        // coupled τ = √n frozen at construction (Eq. 5)
	Eps       float64        // coupled denominator floor ε inside √(Σ S² + ε)
	Scale     float64        // uncoupled per-score temperature (default 1/√dIn)
}

// pattentionConfig holds NewPattention's optional settings before the parameter
// tensors are allocated.
type pattentionConfig struct {
	dtype     tensor.Dtype
	uncoupled bool
}

// PattentionOption configures NewPattention (functional-options pattern). See
// WithPattentionDtype and WithPattentionUncoupledNorm.
type PattentionOption func(*pattentionConfig)

// WithPattentionDtype selects the parameter dtype (default tensor.F64, chosen so
// finite-difference gradchecks and the hand-computed forward stay exact).
func WithPattentionDtype(dt tensor.Dtype) PattentionOption {
	return func(c *pattentionConfig) { c.dtype = dt }
}

// WithPattentionUncoupledNorm selects the uncoupled per-score GELU normalization
// (Θ = GELU(S·scale)) instead of the default coupled Eq. 5 form. It diverges from
// the paper's Eq. 5 but makes incremental growth BIT-EXACT (appending zero value
// rows never changes the output). See PattentionLayer for the trade-off.
func WithPattentionUncoupledNorm() PattentionOption {
	return func(c *pattentionConfig) { c.uncoupled = true }
}

// NewPattention builds a Pattention projection dIn→dOut with nTokens learnable
// key/value token pairs, using the default coupled Eq. 5 normalization (pass
// WithPattentionUncoupledNorm for the per-score variant). Key and value tokens
// use Xavier-uniform init (Glorot 2010); deterministic via seed. nTokens, dIn and
// dOut must each be ≥ 1. Grow later appends more token pairs (§C21).
func NewPattention(dIn, dOut, nTokens int, seed uint64, opts ...PattentionOption) *PattentionLayer {
	if dIn < 1 || dOut < 1 || nTokens < 1 {
		panic(fmt.Sprintf("nn: NewPattention needs dIn,dOut,nTokens ≥ 1, got %d,%d,%d", dIn, dOut, nTokens))
	}
	cfg := pattentionConfig{dtype: tensor.F64}
	for _, o := range opts {
		o(&cfg)
	}
	k := tensor.New(cfg.dtype, tensor.Shape{nTokens, dIn})
	XavierUniform(k, dIn, nTokens, seed)
	v := tensor.New(cfg.dtype, tensor.Shape{nTokens, dOut})
	XavierUniform(v, nTokens, dOut, seed+1)
	return &PattentionLayer{
		K:         k,
		V:         v,
		Uncoupled: cfg.uncoupled,
		Tau:       math.Sqrt(float64(nTokens)),
		Eps:       1e-12,
		Scale:     1 / math.Sqrt(float64(dIn)),
	}
}

func (p *PattentionLayer) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes Y = Θ(X·Kᵀ)·V for X[batch, dIn] through ctx (paper Eq. 4). Θ is
// the coupled Eq. 5 normalization by default, or the uncoupled per-score GELU when
// Uncoupled is set.
func (p *PattentionLayer) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: Pattention expects x rank-2 [batch,dIn], got %v", x.Shape())
	}
	dIn := p.K.Shape()[1]
	if x.Shape()[1] != dIn {
		return nil, fmt.Errorf("nn: Pattention dIn %d != x cols %d", dIn, x.Shape()[1])
	}
	dt := p.K.Dtype()
	// S = X·Kᵀ  (K stored [nTokens,dIn]; the transpose keeps the paper's
	// row-per-token layout so Grow can append rows — its VJP still routes
	// gradients into K).
	kt, err := p.exec(ctx, backend.OpTranspose, nil, p.K)
	if err != nil {
		return nil, err
	}
	s, err := p.exec(ctx, backend.OpMatMul, nil, x, kt)
	if err != nil {
		return nil, err
	}

	var arg *tensor.Tensor
	if p.Uncoupled {
		// arg = S·scale (per score; scale is a fractional constant → OpMul, not OpAXPY, B60)
		arg, err = p.exec(ctx, backend.OpMul, nil, s, tensor.Full(dt, tensor.Shape{1}, p.Scale))
		if err != nil {
			return nil, err
		}
	} else {
		// Coupled Eq. 5: arg = S·τ / √(Σ_k S_k² + ε), sum over the token axis.
		sq, err := p.exec(ctx, backend.OpMul, nil, s, s)
		if err != nil {
			return nil, err
		}
		ss, err := p.exec(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, sq)
		if err != nil {
			return nil, err
		}
		floored, err := p.exec(ctx, backend.OpAdd, nil, ss, tensor.Full(dt, tensor.Shape{1}, p.Eps))
		if err != nil {
			return nil, err
		}
		den, err := p.exec(ctx, backend.OpSqrt, nil, floored)
		if err != nil {
			return nil, err
		}
		num, err := p.exec(ctx, backend.OpMul, nil, s, tensor.Full(dt, tensor.Shape{1}, p.Tau))
		if err != nil {
			return nil, err
		}
		arg, err = p.exec(ctx, backend.OpDiv, nil, num, den)
		if err != nil {
			return nil, err
		}
	}

	// A = Θ(S) = GELU(arg); Y = A·V.
	a, err := p.exec(ctx, backend.OpGELU, nil, arg)
	if err != nil {
		return nil, err
	}
	return p.exec(ctx, backend.OpMatMul, nil, a, p.V)
}

// Params returns the trainable tensors (K, V).
func (p *PattentionLayer) Params() []*tensor.Tensor { return []*tensor.Tensor{p.K, p.V} }

// NumTokens reports the current parameter-token count.
func (p *PattentionLayer) NumTokens() int { return p.K.Shape()[0] }

// Grow appends deltaTokens new key/value token pairs, expanding capacity while
// (approximately) preserving the current output — TokenFormer's incremental
// scaling (paper Eq. 12-13). New VALUE rows are initialized to ZERO; new KEY rows
// are small Xavier values (default coupled mode: scaled by growKeyScale; uncoupled
// mode: full Xavier). Growth semantics depend on the normalization mode:
//
//   - Coupled (default, Eq. 5): APPROXIMATE / resume-from-checkpoint. The new keys
//     add small terms to the shared per-row L2 denominator, so existing scores are
//     perturbed by a BOUNDED amount that shrinks toward 0 as the new keys' scores
//     → 0 (exactly zero-scored new keys leave the output bit-exact, given the
//     frozen τ). This is why the paper zero-initializes new keys.
//   - Uncoupled (WithPattentionUncoupledNorm): BIT-EXACT. Per-score normalization
//     means existing scores are untouched and the zero value rows contribute
//     nothing, so the output is unchanged to the last bit.
//
// In both modes the new tokens are trainable: because the new keys are non-zero,
// the new columns of A are non-zero, so the (zero-initialized) new value rows
// receive gradient on the first step and the added capacity engages (§V16.4).
//
// Grow REPLACES the K and V parameter tensors with larger ones, so callers must
// re-read Params() (e.g. rebuild the optimizer) after growing. deltaTokens ≥ 1.
func (p *PattentionLayer) Grow(deltaTokens int, seed uint64) {
	if deltaTokens < 1 {
		panic(fmt.Sprintf("nn: Pattention Grow needs deltaTokens ≥ 1, got %d", deltaTokens))
	}
	old := p.K.Shape()[0]
	dIn := p.K.Shape()[1]
	dOut := p.V.Shape()[1]
	dt := p.K.Dtype()

	deltaK := tensor.New(dt, tensor.Shape{deltaTokens, dIn})
	XavierUniform(deltaK, dIn, deltaTokens, seed)
	if !p.Uncoupled {
		// Near-zero key init keeps coupled-mode resume near-lossless (see growKeyScale).
		for r := 0; r < deltaTokens; r++ {
			for c := 0; c < dIn; c++ {
				deltaK.SetF64(deltaK.AtF64(r, c)*growKeyScale, r, c)
			}
		}
	}

	newK := tensor.New(dt, tensor.Shape{old + deltaTokens, dIn})
	copyRows(newK, p.K, 0)      // existing key tokens, byte-for-byte
	copyRows(newK, deltaK, old) // appended (small) key tokens

	// New value rows are ZERO (tensor.New zero-fills), so the grown V is the old V
	// followed by a zero block.
	newV := tensor.New(dt, tensor.Shape{old + deltaTokens, dOut})
	copyRows(newV, p.V, 0)

	p.K, p.V = newK, newV
}

// copyRows copies every element of src into dst starting at row offset rowOff.
// dst must be wide enough; src and dst share column count and dtype. Runs once
// per Grow (not a hot path).
func copyRows(dst, src *tensor.Tensor, rowOff int) {
	cols := src.Shape()[1]
	for r := 0; r < src.Shape()[0]; r++ {
		for c := 0; c < cols; c++ {
			dst.SetF64(src.AtF64(r, c), rowOff+r, c)
		}
	}
}
