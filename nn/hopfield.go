package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Hopfield is a Modern (continuous) Hopfield Network (Ramsauer, Lehner,
// Fürst, Rumetshofer, Gruber, Holzleitner, Adler, Kreil, Kopp, Klambauer,
// Brandstetter & Hochreiter 2020, "Hopfield Networks is All You Need",
// arXiv:2008.02217). It is an ASSOCIATIVE MEMORY: it stores a set of patterns
// and, given a partial or noisy cue, retrieves the whole stored pattern that
// best matches the cue. The single retrieval step is
//
//	ξ_new = Yᵀ · softmax(β · Y · ξ)
//
// for stored patterns Y ∈ ℝ^{N×d}, a query/state ξ ∈ ℝ^d and an inverse-
// temperature β. The softmax scores ξ against every stored pattern; the output
// is the β-weighted average of the patterns, which — for large β and well-
// separated patterns — collapses onto the single closest stored pattern.
// Iterating the update drives ξ to a FIXED POINT (a stored memory).
//
// In plain terms: it is auto-complete for vectors. Show it a smudged or
// half-erased pattern and it snaps to the clean stored one it most resembles.
// Turn β up and it commits hard to a single memory; turn β down and it returns
// a soft blend of several ("metastable") memories.
//
// The paper's central observation — and the reason this file composes on the
// very same ops as the attention layers — is that ONE Hopfield update IS a
// softmax attention: with queries ξ, keys = values = Y and scale β = 1/√d, the
// update equals Attention(ξ, Y, Y). So Hopfield.Update below is literally
// scaled dot-product attention, and Forward just iterates it toward the memory.
//
// Further reading: arXiv:2008.02217 (the paper); Vaswani et al. 2017
// (arXiv:1706.03762) for the attention view. See nn.SetTransformer for the
// set-encoding reading of the same update.
type Hopfield struct {
	Stored *tensor.Tensor // stored memory patterns Y, [N, Dim] (trainable)
	Beta   float64        // inverse temperature β (default 1/√Dim, the attention scale)
	Steps  int            // number of retrieval iterations toward the fixed point (default 1)
	Dim    int            // pattern width d
}

// hopfieldConfig collects the Hopfield tunables.
type hopfieldConfig struct {
	beta     float64        // <0 sentinel → default 1/√Dim
	steps    int            // retrieval iterations
	patterns *tensor.Tensor // preset stored memory, or nil to build a random one
}

// HopfieldOption configures a Hopfield layer (functional-options idiom, §C12);
// options carry the WithHopfield* prefix.
type HopfieldOption func(*hopfieldConfig)

// WithHopfieldBeta sets the inverse temperature β. Larger β sharpens the
// softmax so retrieval commits to a single stored pattern; smaller β returns a
// soft (metastable) average. The default is β = 1/√Dim, which makes one update
// identical to scaled dot-product attention.
func WithHopfieldBeta(beta float64) HopfieldOption {
	return func(c *hopfieldConfig) { c.beta = beta }
}

// WithHopfieldSteps sets how many times Forward iterates the retrieval update
// toward the fixed point (default 1). More steps drive a cue deeper into the
// basin of the nearest stored pattern.
func WithHopfieldSteps(steps int) HopfieldOption {
	return func(c *hopfieldConfig) { c.steps = steps }
}

// WithHopfieldPatterns presets the stored memory to a specific [N,Dim] tensor
// (a fixed or externally-initialised memory) instead of the random patterns the
// constructor would otherwise build. The tensor is used as-is and is still
// returned by Params, so it can be frozen or fine-tuned at the caller's choice.
func WithHopfieldPatterns(patterns *tensor.Tensor) HopfieldOption {
	return func(c *hopfieldConfig) { c.patterns = patterns }
}

// NewHopfield builds a Modern Hopfield associative memory over pattern width dim
// with numPatterns stored patterns. By default the memory is Xavier-uniform and
// trainable, β = 1/√dim and one retrieval step is taken; use the WithHopfield*
// options to set β, the number of iterations, or supply a preset memory (in
// which case numPatterns must match its row count). Deterministic via seed.
func NewHopfield(dtype tensor.Dtype, dim, numPatterns int, seed uint64, opts ...HopfieldOption) (*Hopfield, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: Hopfield dim %d must be positive", dim)
	}
	if numPatterns <= 0 {
		return nil, fmt.Errorf("nn: Hopfield numPatterns %d must be positive", numPatterns)
	}
	c := hopfieldConfig{beta: -1, steps: 1}
	for _, o := range opts {
		o(&c)
	}
	if c.steps <= 0 {
		c.steps = 1
	}
	beta := c.beta
	if beta < 0 {
		beta = 1 / math.Sqrt(float64(dim))
	}
	stored := c.patterns
	if stored == nil {
		stored = tensor.New(dtype, tensor.Shape{numPatterns, dim})
		XavierUniform(stored, numPatterns, dim, seed)
	} else if stored.Ndim() != 2 || stored.Shape()[1] != dim || stored.Shape()[0] != numPatterns {
		return nil, fmt.Errorf("nn: Hopfield preset patterns %v do not match [%d,%d]", stored.Shape(), numPatterns, dim)
	}
	return &Hopfield{Stored: stored, Beta: beta, Steps: c.steps, Dim: dim}, nil
}

func (h *Hopfield) exec(ctx *backend.Context, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
	o, err := backend.Execute(ctx, op, in, at)
	if err != nil {
		return nil, err
	}
	return o[0], nil
}

// Update applies ONE Hopfield retrieval step to the query state xi [Q,Dim]
// against the stored memory Y:
//
//	Update(ξ) = Yᵀ · softmax(β · Y · ξᵀ)ᵀ = softmax(β·ξ·Yᵀ) · Y
//
// returning [Q,Dim]. This is exactly scaled dot-product attention with queries
// ξ, keys = values = Y and scale β — assert it against a reference attention to
// see the equivalence. Differentiable w.r.t. both xi and the stored patterns.
func (h *Hopfield) Update(ctx *backend.Context, xi *tensor.Tensor) (*tensor.Tensor, error) {
	if xi.Ndim() != 2 || xi.Shape()[1] != h.Dim {
		return nil, fmt.Errorf("nn: Hopfield query wants [Q,%d], got %v", h.Dim, xi.Shape())
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc, no wall-clock win
	yt, err := h.exec(ctx, backend.OpTranspose, nil, h.Stored) // [Dim,N]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc, no wall-clock win
	scores, err := h.exec(ctx, backend.OpMatMul, nil, xi, yt) // [Q,N] = ξ·Yᵀ
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc, no wall-clock win
	scaled, err := h.exec(ctx, backend.OpMul, nil, scores, scalarTensor(xi.Dtype(), h.Beta)) // β·(ξ·Yᵀ)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc, no wall-clock win
	p, err := h.exec(ctx, backend.OpSoftmax, nil, scaled) // softmax over stored patterns
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 resource-only variadic-alloc, no wall-clock win
	return h.exec(ctx, backend.OpMatMul, nil, p, h.Stored) // [Q,Dim] = p·Y
}

// Forward retrieves from the stored memory by iterating the Hopfield update
// Steps times on the query state x [Q,Dim] → [Q,Dim]. With one step it is a
// single softmax-attention read of the memory; with more steps and a large β it
// drives each cue to the fixed point of the nearest stored pattern (associative
// recall). Differentiable through every iteration.
func (h *Hopfield) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != h.Dim {
		return nil, fmt.Errorf("nn: Hopfield wants x [Q,%d], got %v", h.Dim, x.Shape())
	}
	xi := x
	var err error
	for range h.Steps {
		if xi, err = h.Update(ctx, xi); err != nil {
			return nil, err
		}
	}
	return xi, nil
}

// Params returns the stored memory patterns (trainable). Freeze it by excluding
// it from the optimizer if the memory should stay fixed.
func (h *Hopfield) Params() []*tensor.Tensor { return []*tensor.Tensor{h.Stored} }
