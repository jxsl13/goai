package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GroupNorm is Group Normalization (Wu & He 2018, "Group Normalization", ECCV,
// arXiv:1803.08494). It divides the C feature channels into G groups and normalizes
// each group PER SAMPLE — computing the mean and (biased) variance over the C/G
// channels of the group, independent of the batch — then applies a learnable
// per-channel affine y = γ⊙x̂ + β. Because the statistics never cross the batch,
// GroupNorm is stable at any batch size (unlike BatchNorm), which is why it is used
// in vision/diffusion transformers and detectors.
//
// Two limits: G=1 makes it Layer Normalization (all channels in one group) and G=C
// makes it Instance Normalization (each channel alone). γ, β are per-channel [C].
type GroupNorm struct {
	Gamma  *tensor.Tensor // [C] per-channel scale, init 1
	Beta   *tensor.Tensor // [C] per-channel shift, init 0
	Groups int            // number of groups G (must divide C)
	Eps    float64        // variance-floor ε (default 1e-5)
}

// NewGroupNorm builds a GroupNorm over C channels split into groups (γ=1, β=0,
// ε=1e-5). Panics if groups does not divide C or is non-positive.
func NewGroupNorm(dtype tensor.Dtype, groups, c int) *GroupNorm {
	if groups <= 0 || c%groups != 0 {
		panic(fmt.Sprintf("nn: GroupNorm groups=%d must be positive and divide C=%d", groups, c))
	}
	g := tensor.New(dtype, tensor.Shape{c})
	b := tensor.New(dtype, tensor.Shape{c})
	for i := range c {
		g.SetF64(1, i)
	}
	return &GroupNorm{Gamma: g, Beta: b, Groups: groups, Eps: 1e-5}
}

// Forward normalizes x [batch, C] per sample and per group, then applies the
// per-channel affine. Differentiable w.r.t. x, γ and β.
func (gn *GroupNorm) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: GroupNorm.Forward wants rank-2 [batch,C], got %v", x.Shape())
	}
	batch, c := x.Shape()[0], x.Shape()[1]
	if c != gn.Gamma.Shape()[0] {
		return nil, fmt.Errorf("nn: GroupNorm C=%d != γ dim %d", c, gn.Gamma.Shape()[0])
	}
	gs := c / gn.Groups // group size (channels per group)
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	reshape := func(t *tensor.Tensor, shape tensor.Shape) (*tensor.Tensor, error) {
		return ex(backend.OpReshape, backend.ReshapeAttrs{Shape: shape}, t)
	}
	// [batch, C] → [batch, G, C/G]; normalize over the last axis (the group's channels).
	xr, err := reshape(x, tensor.Shape{batch, gn.Groups, gs})
	if err != nil {
		return nil, err
	}
	mu, err := ex(backend.OpMean, backend.ReduceAttrs{Axes: []int{2}, KeepDims: true}, xr) // [batch,G,1]
	if err != nil {
		return nil, err
	}
	centered, err := ex(backend.OpSub, nil, xr, mu)
	if err != nil {
		return nil, err
	}
	sq, err := ex(backend.OpMul, nil, centered, centered)
	if err != nil {
		return nil, err
	}
	variance, err := ex(backend.OpMean, backend.ReduceAttrs{Axes: []int{2}, KeepDims: true}, sq) // biased var
	if err != nil {
		return nil, err
	}
	epsT := tensor.New(x.Dtype(), tensor.Shape{1, 1, 1})
	epsT.SetF64(gn.Eps, 0, 0, 0)
	varEps, err := ex(backend.OpAdd, nil, variance, epsT)
	if err != nil {
		return nil, err
	}
	std, err := ex(backend.OpSqrt, nil, varEps)
	if err != nil {
		return nil, err
	}
	normed, err := ex(backend.OpDiv, nil, centered, std) // [batch,G,C/G]
	if err != nil {
		return nil, err
	}
	nback, err := reshape(normed, tensor.Shape{batch, c})
	if err != nil {
		return nil, err
	}
	// per-channel affine: γ,β [C] → [1,C] to broadcast over the batch.
	gRow, err := reshape(gn.Gamma, tensor.Shape{1, c})
	if err != nil {
		return nil, err
	}
	bRow, err := reshape(gn.Beta, tensor.Shape{1, c})
	if err != nil {
		return nil, err
	}
	scaled, err := ex(backend.OpMul, nil, nback, gRow)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, nil, scaled, bRow)
}

// Params returns the per-channel scale and shift.
func (gn *GroupNorm) Params() []*tensor.Tensor { return []*tensor.Tensor{gn.Gamma, gn.Beta} }
