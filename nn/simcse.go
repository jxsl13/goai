package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// simCSEDefaultTemperature is the paper's temperature τ=0.05 (Gao et al. 2021 §7,
// tuned on the STS development sets); see WithSimCSETemperature for the τ boundary
// behaviour (§C21).
const simCSEDefaultTemperature = 0.05

// simCSEConfig holds the resolved SimCSELoss knobs; callers set them through the
// functional-option setters, never directly (§C12).
type simCSEConfig struct {
	temperature float64
	normalize   bool
	symmetric   bool
}

// SimCSEOption configures SimCSELoss via the functional-options idiom (§C12);
// construct values with the WithSimCSE* setters and pass them variadically.
type SimCSEOption func(*simCSEConfig)

// WithSimCSETemperature overrides the softmax temperature τ (paper default 0.05).
//
// §C21 τ boundaries: τ>0 is required (a non-positive τ makes SimCSELoss error).
// τ scales the cosine logits S=cos(zᵢ,zⱼ)/τ before the softmax, so it sets how
// sharply the objective separates the positive from the in-batch negatives: as
// τ→0⁺ the logits blow up and the softmax collapses onto the single largest
// similarity (all weight on the hardest negative or the positive — a near
// arg-max, high-variance gradient), while as τ→∞ the logits flatten toward 0 and
// the loss saturates to log(batch) with a vanishing gradient (no contrast). The
// paper's τ=0.05 is the tuned middle ground; values are ignored unless positive so
// a stray τ≤0 leaves the default in place rather than producing a silent error.
func WithSimCSETemperature(tau float64) SimCSEOption {
	return func(c *simCSEConfig) {
		if tau > 0 {
			c.temperature = tau
		}
	}
}

// WithSimCSENormalize toggles per-row L2 normalization of z1, z2 before the
// similarity matrix. It is on by default because SimCSE (and the sentence-embedding
// literature) scores with COSINE similarity; turning it off uses raw dot products,
// which couples the loss to embedding magnitude and is almost never what you want.
func WithSimCSENormalize(normalize bool) SimCSEOption {
	return func(c *simCSEConfig) { c.normalize = normalize }
}

// WithSimCSESymmetric selects the symmetrized objective. The paper's canonical
// unsupervised loss (Gao et al. 2021, Eq. 1) is ONE-DIRECTIONAL — each anchor z1[i]
// classifies its positive z2[i] against the negatives {z2[j] : j≠i}. Enabling this
// averages that with the reverse direction (z2 anchors against z1), i.e. the
// symmetric in-batch InfoNCE, which is a slightly lower-variance but non-canonical
// variant. Off (one-directional) by default.
func WithSimCSESymmetric(symmetric bool) SimCSEOption {
	return func(c *simCSEConfig) { c.symmetric = symmetric }
}

// SimCSELoss is the unsupervised SimCSE contrastive objective (Gao, Yao & Chen 2021,
// "SimCSE: Simple Contrastive Learning of Sentence Embeddings", EMNLP,
// arXiv:2104.08821, Eq. 1). Its one distinctive idea is dropout-as-augmentation: the
// POSITIVE pair for input i is the SAME input encoded TWICE under two INDEPENDENT
// dropout masks, so z1[i] and z2[i] are two noisy views of one sentence with no
// labelled pairs, and every OTHER input in the batch is an in-batch negative. The
// caller must therefore produce z1 and z2 by running the identical batch through the
// SAME encoder twice with dropout ENABLED (Training=true) — see SimCSEViews and the
// ExampleSimCSELoss; feeding two dropout-free (identical) encodings removes the only
// source of the positive signal.
//
// Given z1, z2 ∈ Rᴮˣᵈ it builds the temperature-scaled cosine-similarity matrix
// S[i,j] = cos(z1[i], z2[j]) / τ ∈ Rᴮˣᴮ (per-row L2 normalization on by default) and
// returns the softmax cross-entropy whose target for row i is its own index i (the
// diagonal is the positive):
//
//	L = (1/B) · Σᵢ −log( exp(S[i,i]) / Σⱼ exp(S[i,j]) )
//
// This is exactly the paper's Eq. 1. With WithSimCSESymmetric it instead returns the
// symmetric two-direction average (delegating to InfoNCE). Composed from
// matmul/transpose/softmax-CE, so it is differentiable w.r.t. both z1 and z2.
//
// τ defaults to the paper's 0.05; see WithSimCSETemperature for the τ→0⁺ / τ→∞
// boundary behaviour (§C21). Errors on non-matching or non-[B,d] shapes, or τ≤0.
func SimCSELoss(ctx *backend.Context, z1, z2 *tensor.Tensor, opts ...SimCSEOption) (*tensor.Tensor, error) {
	cfg := simCSEConfig{temperature: simCSEDefaultTemperature, normalize: true}
	for _, o := range opts {
		o(&cfg)
	}
	if z1.Ndim() != 2 || !z1.Shape().Equal(z2.Shape()) {
		return nil, fmt.Errorf("nn: SimCSELoss needs equal-shaped [batch,dim] z1 and z2, got %v and %v", z1.Shape(), z2.Shape())
	}
	if cfg.temperature <= 0 {
		return nil, fmt.Errorf("nn: SimCSELoss temperature must be > 0, got %g", cfg.temperature)
	}
	if cfg.symmetric {
		// Symmetric two-direction average is exactly the in-batch InfoNCE.
		return InfoNCE(ctx, z1, z2, cfg.temperature, cfg.normalize)
	}
	n := z1.Shape()[0]
	a, b := z1, z2
	if cfg.normalize {
		var err error
		if a, err = l2NormalizeRows(ctx, a); err != nil {
			return nil, err
		}
		if b, err = l2NormalizeRows(ctx, b); err != nil {
			return nil, err
		}
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	bt, err := ex(backend.OpTranspose, nil, b) // [d, B]
	if err != nil {
		return nil, err
	}
	sim, err := ex(backend.OpMatMul, nil, a, bt) // z1·z2ᵀ [B, B]
	if err != nil {
		return nil, err
	}
	s, err := ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: 1 / cfg.temperature}, sim, tensor.New(sim.Dtype(), sim.Shape())) // /τ
	if err != nil {
		return nil, err
	}
	// targets = [0,1,…,B−1]: the positive for anchor i is z2[i] on the diagonal.
	targets := tensor.New(a.Dtype(), tensor.Shape{n})
	//perfscan:ignore PS1001 1D target arange fill, n=batch low trip
	for i := range n {
		targets.SetF64(float64(i), i)
	}
	return CrossEntropy(ctx, s, targets) // mean over the B anchors
}

// SimCSEViews produces the two dropout views z1, z2 that SimCSELoss consumes by
// running x through encode TWICE. encode must apply an ACTIVE Dropout (Training=true)
// somewhere in its path: because a Dropout layer advances its own rng on each call,
// the two invocations sample two INDEPENDENT Bernoulli masks, which is precisely the
// SimCSE positive signal (dropout-as-minimal-augmentation) — no explicit rng argument
// is needed, the randomness lives inside encode's Dropout. With dropout disabled the
// two views are identical and the contrastive signal degenerates (see SimCSELoss).
// The context is threaded through both passes so autograd records them; encode's own
// error is propagated.
func SimCSEViews(ctx *backend.Context, encode func(*backend.Context, *tensor.Tensor) (*tensor.Tensor, error), x *tensor.Tensor) (z1, z2 *tensor.Tensor, err error) {
	if z1, err = encode(ctx, x); err != nil {
		return nil, nil, err
	}
	if z2, err = encode(ctx, x); err != nil {
		return nil, nil, err
	}
	return z1, z2, nil
}
