package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SpectralNorm is a linear layer whose weight is spectrally normalized,
// y = x·(W/σ(W)) (+b), where σ(W) is the largest singular value of W — Miyato,
// Kataoka, Koyama & Yoshida 2018 "Spectral Normalization for GANs" (arXiv:1802.05957,
// ICLR'18). Dividing by σ(W) bounds the layer's Lipschitz constant to 1, which
// stabilizes training (originally for GAN discriminators, also used for Lipschitz
// control in transformers/RL).
//
// σ(W) is estimated by ONE power iteration per forward on persistent left/right
// singular-vector buffers u,v (v ← Wᵀu/‖·‖, u ← Wv/‖·‖, σ ≈ uᵀWv); the buffers are
// updated WITHOUT gradient (paper §2.3), so the backward pass differentiates only the
// explicit W/σ with u,v held constant — which reproduces the paper's Eq. 9 gradient
// exactly via autograd (no hand-coded VJP). The estimate converges to the true σ_max
// (= the top singular value from SVD) as W changes slowly across steps.
type SpectralNorm struct {
	W     *tensor.Tensor // [in, out] the raw (trainable) weight
	B     *tensor.Tensor // [out]; nil → no bias
	u     *tensor.Tensor // [1, in] persistent left singular-vector estimate (buffer, not trained)
	v     *tensor.Tensor // [out, 1] persistent right singular-vector estimate (buffer)
	iters int            // power iterations per forward (default 1)
}

// SpectralNormOption configures a SpectralNorm (functional options, §C12).
type SpectralNormOption func(*SpectralNorm)

// WithSpectralNormIters sets the number of power-iteration steps run per forward pass to
// estimate the weight matrix's largest singular value (spectral norm).
//
// In plain terms: spectral normalization keeps a layer from amplifying its input too much by
// dividing the weights by their largest singular value; that value is estimated cheaply with
// "power iteration", and this is how many refinement steps run each forward pass. Boundary
// behavior — 1 refines the running estimate by one step per pass (the estimate stays accurate
// because it carries over between steps); more iterations converge faster on a changing weight
// but cost more. SPECIAL VALUE: 0 freezes the u,v estimate buffers — useful for a deterministic
// gradient check or for inference on a fixed estimate.
//
// Default 1 (research-grounded: Miyato et al. 2018's choice of one power iteration per step,
// §R150 — sufficient because the estimate is amortized across training steps).
func WithSpectralNormIters(n int) SpectralNormOption { return func(s *SpectralNorm) { s.iters = n } }

// NewSpectralNorm builds a spectrally-normalized linear layer with Xavier-uniform W,
// zero bias, a random left singular-vector buffer, and 15 warm-up power iterations at
// init (paper Appendix A). Deterministic via seed.
func NewSpectralNorm(dtype tensor.Dtype, in, out int, seed uint64, opts ...SpectralNormOption) *SpectralNorm {
	w := tensor.New(dtype, tensor.Shape{in, out})
	XavierUniform(w, in, out, seed)
	b := tensor.New(dtype, tensor.Shape{out})
	Zeros(b)
	u := tensor.New(dtype, tensor.Shape{1, in})
	rng := rand.New(rand.NewPCG(seed, 0x59b1c0ffee5eed01))
	var nrm float64
	for i := range in { // u ~ N(0,1), normalized
		val := rng.NormFloat64()
		u.SetF64(val, 0, i)
		nrm += val * val
	}
	nrm = math.Sqrt(nrm)
	for i := range in {
		u.SetF64(u.AtF64(0, i)/nrm, 0, i)
	}
	s := &SpectralNorm{W: w, B: b, u: u, v: tensor.New(dtype, tensor.Shape{out, 1}), iters: 1}
	for _, o := range opts {
		o(s)
	}
	s.powerIterate(15) // warm up the estimate at init
	return s
}

// powerIterate refines the u,v buffers by `iters` rounds of power iteration on the
// CURRENT W, read directly (no tape) so the singular vectors are stop-gradient.
func (s *SpectralNorm) powerIterate(iters int) {
	in, out := s.W.Shape()[0], s.W.Shape()[1]
	// The two matvecs read W[i,j] and u[i]/v[j] through AtF64 on every (i,j) — 2·in·out
	// dispatches per iteration. Walk the contiguous W/u/v storage directly (u is [1,in],
	// v is [out,1], W is [in,out] row-major). Bit-identical: same values, same ascending-i /
	// ascending-j accumulation. AtF64 fallback for exotic dtypes.
	if ws, us, vs := flatF64(s.W), flatF64(s.u), flatF64(s.v); ws != nil && us != nil && vs != nil {
		tv := make([]float64, out)
		tu := make([]float64, in)
		for range iters {
			var nv float64
			for j := range out {
				var acc float64
				for i := range in {
					acc += ws[i*out+j] * us[i]
				}
				tv[j] = acc
				nv += acc * acc
			}
			if nv = math.Sqrt(nv); nv > 0 {
				for j := range out {
					vs[j] = tv[j] / nv
				}
			}
			var nu float64
			for i := range in {
				wrow := ws[i*out : i*out+out : i*out+out]
				var acc float64
				for j := range out {
					acc += wrow[j] * vs[j] // contiguous W row · v
				}
				tu[i] = acc
				nu += acc * acc
			}
			if nu = math.Sqrt(nu); nu > 0 {
				for i := range in {
					us[i] = tu[i] / nu
				}
			}
		}
		return
	}
	for range iters {
		// v = normalize(Wᵀu): v[j] = Σ_i W[i,j]·u[i]
		tv := make([]float64, out)
		var nv float64
		for j := range out {
			var acc float64
			for i := range in {
				acc += s.W.AtF64(i, j) * s.u.AtF64(0, i)
			}
			tv[j] = acc
			nv += acc * acc
		}
		if nv = math.Sqrt(nv); nv > 0 {
			for j := range out {
				s.v.SetF64(tv[j]/nv, j, 0)
			}
		}
		// u = normalize(Wv): u[i] = Σ_j W[i,j]·v[j]
		tu := make([]float64, in)
		var nu float64
		for i := range in {
			var acc float64
			for j := range out {
				acc += s.W.AtF64(i, j) * s.v.AtF64(j, 0)
			}
			tu[i] = acc
			nu += acc * acc
		}
		if nu = math.Sqrt(nu); nu > 0 {
			for i := range in {
				s.u.SetF64(tu[i]/nu, 0, i)
			}
		}
	}
}

// SigmaEst returns the current spectral-norm estimate σ ≈ uᵀWv from the buffers
// (detached; introspection/verification).
func (s *SpectralNorm) SigmaEst() float64 {
	in, out := s.W.Shape()[0], s.W.Shape()[1]
	var sig float64
	for i := range in {
		var wv float64
		for j := range out {
			wv += s.W.AtF64(i, j) * s.v.AtF64(j, 0)
		}
		sig += s.u.AtF64(0, i) * wv
	}
	return sig
}

// Forward computes x·(W/σ(W)) (+b) through ctx. The u,v buffers are refined by
// `iters` power iterations first (stop-gradient); σ = uᵀWv and W/σ are built on ctx
// so gradients flow to W (the paper's Eq. 9 gradient, via autograd).
func (s *SpectralNorm) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != s.W.Shape()[0] {
		return nil, fmt.Errorf("nn: SpectralNorm expects x [batch,%d], got %v", s.W.Shape()[0], x.Shape())
	}
	s.powerIterate(s.iters)

	// σ = uᵀ·W·v as a [1,1] scalar, differentiable in W (u,v constant).
	wv, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{s.W, s.v}, nil) // [in,1]
	if err != nil {
		return nil, err
	}
	sig, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{s.u, wv[0]}, nil) // [1,1]
	if err != nil {
		return nil, err
	}
	// W_SN = W / σ  (broadcast [in,out] / [1,1]).
	wsn, err := backend.Execute(ctx, backend.OpDiv, []*tensor.Tensor{s.W, sig[0]}, nil)
	if err != nil {
		return nil, err
	}
	y, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{x, wsn[0]}, nil)
	if err != nil {
		return nil, err
	}
	if s.B != nil {
		yb, err := backend.Execute(ctx, backend.OpAddBias, []*tensor.Tensor{y[0], s.B}, nil)
		if err != nil {
			return nil, err
		}
		return yb[0], nil
	}
	return y[0], nil
}

// Params returns the trainable tensors (W, and B if present). The u,v power-iteration
// buffers are NOT parameters (stop-gradient state).
func (s *SpectralNorm) Params() []*tensor.Tensor {
	if s.B == nil {
		return []*tensor.Tensor{s.W}
	}
	return []*tensor.Tensor{s.W, s.B}
}
