package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// matryoshka2DConfig carries the Matryoshka2DLoss knobs set via
// Matryoshka2DOption (per-layer and per-dim importance weights).
type matryoshka2DConfig struct {
	layerWeights []float64 // one weight per provided layer, or nil for uniform 1
	dimWeights   []float64 // one weight per nesting dim, or nil for uniform 1
}

// Matryoshka2DOption configures Matryoshka2DLoss (functional options; see
// WithMatryoshka2DLayerWeights and WithMatryoshka2DDimWeights).
type Matryoshka2DOption func(*matryoshka2DConfig)

// WithMatryoshka2DLayerWeights sets a per-layer importance weight w_l applied to
// every (layer, dim) term read out at layer l, overriding the default of a
// uniform 1 for each layer. The slice length MUST equal len(layerReps) (the
// number of provided layer representations); weights are matched positionally to
// layerReps[0], layerReps[1], …. The 2D-Matryoshka / ESE paper (Li, Zhang, Zhang,
// Long, Xie & Zhang 2024, arXiv:2402.14776 §3.2, Eq. 3) up-weights SHALLOWER
// layers so that early-exit (shallow-layer) embeddings stay strong — e.g. passing
// weights that decrease with depth. A weight of exactly 0 drops that layer's
// entire contribution (and its gradient) from the sum, which is the clean way to
// disable a layer without changing the dims list; negative weights are permitted
// but unusual. See WithMatryoshka2DDimWeights for the embedding-dimension axis.
func WithMatryoshka2DLayerWeights(weights ...float64) Matryoshka2DOption {
	return func(c *matryoshka2DConfig) { c.layerWeights = weights }
}

// WithMatryoshka2DDimWeights sets a per-dim importance weight w_d applied to every
// (layer, dim) term at truncation width dims[d], overriding the default of a
// uniform 1 for each nesting dim. The slice length MUST equal len(dims); weights
// are matched positionally to dims[0], dims[1], …. This is the embedding-dimension
// axis of the standard Matryoshka objective (Kusupati et al. 2022,
// arXiv:2205.13147 Eq. 1, the c_m importance weights): raising a dim's weight
// biases the shared representation toward being useful at that truncation width. A
// weight of exactly 0 drops that truncation width's contribution (and its
// gradient) at every layer; negative weights are permitted but unusual. See
// WithMatryoshka2DLayerWeights for the transformer-layer axis.
func WithMatryoshka2DDimWeights(weights ...float64) Matryoshka2DOption {
	return func(c *matryoshka2DConfig) { c.dimWeights = weights }
}

// Matryoshka2DLoss is the 2D-Matryoshka / Espresso Sentence Embeddings (ESE)
// objective (Li, Zhang, Zhang, Long, Xie & Zhang 2024, "2D Matryoshka Sentence
// Embeddings", arXiv:2402.14776). It generalizes standard Matryoshka
// Representation Learning (Kusupati et al. 2022, arXiv:2205.13147; see
// MatryoshkaLoss) from ONE axis to TWO: the standard loss nests only the
// embedding-DIMENSION axis (every prefix z[:, :m] of the final layer is trained to
// be useful), while 2D-Matryoshka ALSO nests the transformer-LAYER axis, training
// the embedding read out at each SHALLOW layer to be useful at every truncated
// dimension too. The result is a single model that yields a usable sentence
// embedding at any {shallow-layer, truncated-dim} combination — cutting both depth
// (fewer layers → less compute) and width (fewer dims → less storage) on demand.
//
// The loss sums the base per-dim Matryoshka loss over the full (layer × dim) grid:
//
//	L = Σ_{l} Σ_{d∈dims} w_l · w_d · CE( z_l[:, :d] · W_{1:d}, y )
//
// where z_l = layerReps[l] ∈ ℝ^{batch×D} is the embedding read out at layer l (the
// caller runs its model and passes in the intermediate-layer embeddings), W ∈
// ℝ^{D×C} is a shared classifier tied across BOTH axes, y are class-index targets,
// dims is the strictly-increasing set of nested truncation widths ⊆ [1,D], and w_l,
// w_d are the optional per-layer / per-dim importance weights (both default to 1;
// see WithMatryoshka2DLayerWeights and WithMatryoshka2DDimWeights). Each grid cell
// (l, d) is exactly the standard MatryoshkaLoss evaluated at a single granularity
// {d} on layer l's embedding — this function reuses MatryoshkaLoss as its per-cell
// base loss and never reimplements the truncate-and-classify step. Because the
// earliest coordinates and the shallowest layers appear in the most terms,
// gradient concentrates there, producing a representation that is coarse-to-fine
// along BOTH axes.
//
// In plain terms: MatryoshkaLoss lets you keep the first few numbers of one
// embedding and still search well; Matryoshka2DLoss additionally lets you stop the
// transformer early and use a middle layer's embedding — so one trained model
// serves a whole grid of speed/quality trade-offs, from "all layers, full width"
// down to "few layers, few dims".
//
// Collapse: with a SINGLE layer (len(layerReps)==1), the full dims list and
// default (uniform-1) weights, Matryoshka2DLoss is bit-identical to
// MatryoshkaLoss(ctx, layerReps[0], w, targets, dims) — 2D adds ONLY the layer
// axis. layerReps must be non-empty; every rep must be rank-2 [batch, D] with the
// same shape and match w's D rows and targets' batch (validated by the per-cell
// MatryoshkaLoss). dims must be strictly increasing within [1, D]. The returned
// scalar is differentiable w.r.t. every layer rep (gradient reaches ALL provided
// layers, not just the last) and w.
func Matryoshka2DLoss(ctx *backend.Context, layerReps []*tensor.Tensor, w, targets *tensor.Tensor, dims []int, opts ...Matryoshka2DOption) (*tensor.Tensor, error) {
	if len(layerReps) == 0 {
		return nil, fmt.Errorf("nn: Matryoshka2DLoss needs ≥1 layer representation")
	}
	if len(dims) == 0 {
		return nil, fmt.Errorf("nn: Matryoshka2DLoss needs ≥1 nesting dimension")
	}
	// Validate the whole dims list up front: the per-cell MatryoshkaLoss only sees
	// one singleton dim at a time, so it cannot catch a non-increasing full list.
	if w.Ndim() != 2 {
		return nil, fmt.Errorf("nn: Matryoshka2DLoss wants rank-2 classifier w[D,C], got %v", w.Shape())
	}
	D := w.Shape()[0]
	prev := 0
	for i, m := range dims {
		if m < 1 || m > D {
			return nil, fmt.Errorf("nn: Matryoshka2DLoss nesting dim %d out of [1,%d]", m, D)
		}
		if m <= prev {
			return nil, fmt.Errorf("nn: Matryoshka2DLoss nesting dims must be strictly increasing (index %d: %d after %d)", i, m, prev)
		}
		prev = m
	}
	cfg := matryoshka2DConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.layerWeights != nil && len(cfg.layerWeights) != len(layerReps) {
		return nil, fmt.Errorf("nn: Matryoshka2DLoss layer weights (%d) must match layer count (%d)", len(cfg.layerWeights), len(layerReps))
	}
	if cfg.dimWeights != nil && len(cfg.dimWeights) != len(dims) {
		return nil, fmt.Errorf("nn: Matryoshka2DLoss dim weights (%d) must match dims count (%d)", len(cfg.dimWeights), len(dims))
	}
	layerWeight := func(l int) float64 {
		if cfg.layerWeights == nil {
			return 1
		}
		return cfg.layerWeights[l]
	}
	dimWeight := func(d int) float64 {
		if cfg.dimWeights == nil {
			return 1
		}
		return cfg.dimWeights[d]
	}

	var total *tensor.Tensor // running Σ over the (layer × dim) grid of weighted base losses
	//perfscan:ignore PS4011 layer×dim architectural grid; per-cell CrossEntropy matmul dominates dispatch
	for l, rep := range layerReps {
		wl := layerWeight(l)
		for di, d := range dims {
			// Each grid cell is the standard MatryoshkaLoss at a single granularity {d}
			// on this layer's embedding — this is CE(z_l[:, :d]·W_{1:d}, y). Reusing the
			// 1D loss as the per-cell base loss keeps the truncate-and-classify step in
			// exactly one place; passing every dim at once instead would forfeit per-dim
			// weighting, so we invoke it per cell.
			cell, err := MatryoshkaLoss(ctx, rep, w, targets, []int{d})
			if err != nil {
				return nil, fmt.Errorf("nn: Matryoshka2DLoss layer %d dim %d: %w", l, d, err)
			}
			if weight := wl * dimWeight(di); weight != 1 {
				// Scale the scalar cell loss by w_l·w_d via an elementwise multiply
				// against a same-shaped constant. (OpAXPY is unusable here: its
				// Alpha==0 is defaulted to 1, so a zero weight would NOT drop the
				// term — OpMul zeroes both the value and, via its VJP, the gradient.)
				wt := tensor.New(cell.Dtype(), cell.Shape())
				wt.SetF64(weight)
				//perfscan:ignore PS3024,PS6017 resource-only variadic-alloc, only fires when weight!=1 | resource-only variadic-alloc pool sibling, no wall-c
				if cell, err = rdropExec(ctx, backend.OpMul, nil, cell, wt); err != nil {
					return nil, err
				}
			}
			if total == nil {
				total = cell
				//perfscan:ignore PS3024,PS6017 resource-only variadic-alloc, no wall-clock win | resource-only variadic-alloc pool sibling, no wall-clock win
			} else if total, err = rdropExec(ctx, backend.OpAdd, nil, total, cell); err != nil {
				return nil, err
			}
		}
	}
	return total, nil
}
