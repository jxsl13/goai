package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GumbelSoftmax implements the Gumbel-Softmax / Concrete relaxation (Jang, Gu &
// Poole 2016, "Categorical Reparameterization with Gumbel-Softmax", arXiv:1611.01144;
// Maddison, Mnih & Teh 2016, "The Concrete Distribution", arXiv:1611.00712). It draws
// a DIFFERENTIABLE approximate sample from a categorical whose class logits are the
// rows of logits, using the reparameterization (Eq. 1-2):
//
//	y_i = softmax((logit_i + g_i)/τ) ,   g_i = −log(−log(u_i)),  u_i ~ Uniform(0,1)
//
// The Gumbel noise g makes the softmax a stochastic sample (the Gumbel-Max trick:
// argmax_i(logit_i + g_i) is an exact categorical draw), while the temperature τ
// controls the relaxation: τ→0 sharpens y toward a one-hot vector (a discrete
// sample), τ→∞ flattens it toward uniform. Because the randomness is pushed into the
// externally-supplied noise, y is a smooth, differentiable function of the logits —
// the reparameterization trick for discrete variables (MoE routing, discrete latents).
//
// logits and noise are [batch, C]; τ > 0. The noise is passed in (draw it with
// SampleGumbelNoise) so the op is deterministic and differentiable w.r.t. logits.
// Returns the [batch, C] relaxed sample (each row a distribution over C classes).
func GumbelSoftmax(ctx *backend.Context, logits, noise *tensor.Tensor, tau float64) (*tensor.Tensor, error) {
	if logits.Ndim() != 2 || noise.Ndim() != 2 {
		return nil, fmt.Errorf("nn: GumbelSoftmax wants rank-2 [batch,C], got %v, %v", logits.Shape(), noise.Shape())
	}
	if !logits.Shape().Equal(noise.Shape()) {
		return nil, fmt.Errorf("nn: GumbelSoftmax logits/noise shape mismatch: %v vs %v", logits.Shape(), noise.Shape())
	}
	if tau <= 0 {
		return nil, fmt.Errorf("nn: GumbelSoftmax temperature must be > 0, got %g", tau)
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	perturbed, err := ex(backend.OpAdd, nil, logits, noise) // logit + gumbel
	if err != nil {
		return nil, err
	}
	invTau := scalarTensor(logits.Dtype(), 1/tau)
	scaled, err := ex(backend.OpMul, nil, perturbed, invTau) // (logit+gumbel)/τ, broadcast [batch,C]*[1,1]
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSoftmax, nil, scaled) // softmax over the last (class) axis
}

// GumbelSoftmaxHard is the Straight-Through Gumbel-Softmax estimator (ST-GS, Jang
// et al. 2016 §2.2): the forward pass returns the HARD one-hot sample
// onehot(argmax_i y_i), but gradients flow as if the soft GumbelSoftmax were used
// (∂y_hard/∂logits ≈ ∂y_soft/∂logits). It is built as y = y_hard − y_soft + y_soft
// with the first two terms detached to constants, so the forward value is the
// one-hot y_hard while the backward path is the differentiable y_soft. Use it when a
// discrete choice is needed on the forward pass but the model must still train.
func GumbelSoftmaxHard(ctx *backend.Context, logits, noise *tensor.Tensor, tau float64) (*tensor.Tensor, error) {
	ySoft, err := GumbelSoftmax(ctx, logits, noise, tau)
	if err != nil {
		return nil, err
	}
	batch, c := ySoft.Shape()[0], ySoft.Shape()[1]
	// y_hard = argmax one-hot (a non-differentiable constant).
	yHard := tensor.New(ySoft.Dtype(), ySoft.Shape())
	//perfscan:ignore PS1001 only batch stores (one per row), negligible
	for b := range batch {
		best, bv := 0, math.Inf(-1)
		//perfscan:ignore PS1001,PS3068 argmax dominated by surrounding softmax/add/mul ops | niche ST-GS one-hot argmax, backend-op dominated
		for j := range c {
			if v := ySoft.AtF64(b, j); v > bv {
				best, bv = j, v
			}
		}
		yHard.SetF64(1, b, best)
	}
	ex := func(op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, nil)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	// Straight-through (Jang §2.2): y = y_hard + (y_soft − stop_gradient(y_soft)).
	// Forward = y_hard; backward = the soft gradient (the detached term drops out).
	sg, err := ex(backend.OpStopGradient, ySoft)
	if err != nil {
		return nil, err
	}
	diff, err := ex(backend.OpSub, ySoft, sg)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, diff, yHard)
}

// SampleGumbelNoise draws a [shape] tensor of i.i.d. standard Gumbel(0,1) noise,
// g = −log(−log(u)) with u ~ Uniform(0,1), for GumbelSoftmax. The seed makes it
// reproducible; u is clamped away from 0 so the double log stays finite.
func SampleGumbelNoise(seed uint64, shape tensor.Shape) *tensor.Tensor {
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	t := tensor.New(tensor.F64, shape)
	// fresh contiguous F64 tensor: flat index i == storage index, so write directly
	// (byte-identical to the Unravel+SetF64 loop, same RNG draw order).
	d := t.Storage().F64()
	for i := range d {
		u := rng.Float64()
		if u < 1e-20 {
			u = 1e-20
		}
		d[i] = -math.Log(-math.Log(u))
	}
	return t
}
