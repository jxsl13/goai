package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// AWQ — Activation-aware Weight Quantization (Lin, Tang, Tang, Yang, Dang, Gan & Han
// 2023, "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration",
// arXiv:2306.00978, MLSys'24). AWQ's insight: the weights that matter most are the ones
// multiplied by LARGE activations, not the ones with large magnitude. It protects those
// salient input channels by scaling their weight column UP by s_j = act_j^α (so their
// relative rounding error shrinks) before quantizing, and folds the inverse scale back so
// the layer output W·X is preserved: Ŵ = Quant(W·diag(s))·diag(1/s). The exponent α∈[0,1]
// is chosen by a grid search minimizing the output error ‖W·X − Ŵ·X‖; α=0 (no scaling)
// recovers plain round-to-nearest.

// AWQOption configures AWQuantize (functional-options idiom, §C12).
type AWQOption func(*awqCfg)

type awqCfg struct {
	alpha float64 // pinned α; NaN = grid-search
	grid  int     // grid-search resolution over α∈[0,1]
}

// WithAWQAlpha pins the per-channel scaling exponent α, skipping AWQ's grid search.
//
// In plain terms: AWQ protects the most important weight channels by scaling them up before
// quantizing (and down after), so their precision is preserved; α controls how aggressively
// that scaling follows the activation magnitudes. Normally AWQ searches for the best α — this
// pins it. Boundary behavior — SPECIAL VALUE α=0 disables scaling entirely (plain
// round-to-nearest quantization); larger α scales salient channels harder, which helps until
// it starts distorting the non-salient ones.
//
// Default: unset (the grid search picks α); pin it only to reproduce a specific configuration.
// Research-grounded: the salience-aware scaling of Lin et al. 2023 (AWQ, §R172).
func WithAWQAlpha(a float64) AWQOption { return func(c *awqCfg) { c.alpha = a } }

// WithAWQGrid sets the resolution of AWQ's α grid search over [0, 1].
//
// In plain terms: how many candidate scaling exponents AWQ tries before keeping the one with
// the lowest quantization error — more points = a finer search at more calibration compute.
// Boundary behavior — small grids may miss the best α; large grids cost more with diminishing
// returns. SPECIAL VALUE: n ≤ 1 is ignored (keeps the current value).
//
// Default 20 (research-grounded: the AWQ reference grid of 20 points over [0,1], §R172).
func WithAWQGrid(n int) AWQOption {
	return func(c *awqCfg) {
		if n > 1 {
			c.grid = n
		}
	}
}

// AWQuantize returns the AWQ-quantized weight Ŵ[out,in] for weight w[out,in], calibration
// inputs x[in,samples], and a `levels`-point symmetric grid (per output row). It searches
// α over [0,1] (unless pinned) and returns the reconstruction with the lowest output error.
func AWQuantize(w, x *tensor.Tensor, levels int, opts ...AWQOption) (*tensor.Tensor, error) {
	if w.Ndim() != 2 || x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: AWQuantize needs rank-2 w and x, got %v and %v", w.Shape(), x.Shape())
	}
	out, in := w.Shape()[0], w.Shape()[1]
	if x.Shape()[0] != in {
		return nil, fmt.Errorf("nn: AWQuantize x must be [%d,samples], got %v", in, x.Shape())
	}
	if levels < 2 {
		return nil, fmt.Errorf("nn: AWQuantize needs levels ≥ 2, got %d", levels)
	}
	samples := x.Shape()[1]
	cfg := awqCfg{alpha: math.NaN(), grid: 20}
	for _, o := range opts {
		o(&cfg)
	}

	// per-input-channel activation magnitude: act_j = mean_s |X[j,s]|
	act := make([]float64, in)
	for j := range in {
		var s float64
		for k := range samples {
			s += math.Abs(x.AtF64(j, k))
		}
		act[j] = s/float64(samples) + 1e-12 // ε floor so act^α is well-defined for dead channels
	}
	wm := matAt(w) // [out][in]

	// candidate α values (pinned, or a grid over [0,1]).
	var alphas []float64
	if !math.IsNaN(cfg.alpha) {
		alphas = []float64{cfg.alpha}
	} else {
		alphas = make([]float64, cfg.grid) // ratio = i/n_grid ∈ [0,1) (llm-awq)
		for i := range cfg.grid {
			alphas[i] = float64(i) / float64(cfg.grid)
		}
	}

	var best *tensor.Tensor
	bestErr := math.Inf(1)
	for _, alpha := range alphas {
		// scale_j = act_j^α, normalized so its geometric mean is ~1 (scales/√(max·min)).
		scale := make([]float64, in)
		smin, smax := math.Inf(1), math.Inf(-1)
		for j := range in {
			scale[j] = math.Max(math.Pow(act[j], alpha), 1e-4) // clamp (llm-awq)
			smin = math.Min(smin, scale[j])
			smax = math.Max(smax, scale[j])
		}
		norm := math.Sqrt(smax * smin)
		for j := range in {
			scale[j] /= norm
		}
		cand := quantizeScaled(wm, scale, out, in, levels, w.Dtype())
		if e := reconErrMat(wm, cand, x, out, in, samples); e < bestErr {
			bestErr, best = e, cand
		}
	}
	return best, nil
}

// quantizeScaled returns Ŵ = Quant(W·diag(scale))·diag(1/scale) with per-output-row
// symmetric `levels`-point quantization computed from the scaled row's max-abs.
func quantizeScaled(w [][]float64, scale []float64, out, in, levels int, dt tensor.Dtype) *tensor.Tensor {
	q := tensor.New(dt, tensor.Shape{out, in})
	// Write the quantized row through the contiguous typed slice (was a SetF64 dispatch per
	// element, out·in calls; quantizeScaled runs once per candidate scale in AWQ's search).
	// Bit-identical: the same grid(w·scale)/scale value stored to the same [r,j] slot.
	switch dt {
	case tensor.F64:
		qs := q.Storage().F64()
		for r := range out {
			var maxabs float64
			for j := range in {
				maxabs = math.Max(maxabs, math.Abs(w[r][j]*scale[j]))
			}
			if maxabs == 0 {
				continue
			}
			grid := UniformQuantizer(levels, -maxabs, maxabs)
			row := qs[r*in : r*in+in : r*in+in]
			for j := range in {
				row[j] = grid(w[r][j]*scale[j]) / scale[j]
			}
		}
	case tensor.F32:
		qs := q.Storage().F32()
		for r := range out {
			var maxabs float64
			for j := range in {
				maxabs = math.Max(maxabs, math.Abs(w[r][j]*scale[j]))
			}
			if maxabs == 0 {
				continue
			}
			grid := UniformQuantizer(levels, -maxabs, maxabs)
			row := qs[r*in : r*in+in : r*in+in]
			for j := range in {
				row[j] = float32(grid(w[r][j]*scale[j]) / scale[j])
			}
		}
	default:
		for r := range out { // exotic dtype: per-element dispatch
			var maxabs float64
			for j := range in {
				maxabs = math.Max(maxabs, math.Abs(w[r][j]*scale[j]))
			}
			if maxabs == 0 {
				continue
			}
			grid := UniformQuantizer(levels, -maxabs, maxabs)
			//perfscan:ignore PS1005 intentional exotic-dtype fallback; F32/F64 take the typed path above
			for j := range in {
				q.SetF64(grid(w[r][j]*scale[j])/scale[j], r, j)
			}
		}
	}
	return q
}

// reconErrMat returns ‖(W − Ŵ)·X‖_F for W as a dense matrix and Ŵ as a tensor.
func reconErrMat(w [][]float64, wq *tensor.Tensor, x *tensor.Tensor, out, in, samples int) float64 {
	// The old loop re-dispatched wq.AtF64(r,i) for every sample s (it is invariant across s)
	// and x.AtF64(i,s) per element — out·samples·in interface calls over a GEMM-shaped kernel.
	// Grab contiguous typed rows once, hoist the per-row residual diff[i]=w[r][i]−ŵ[r][i]
	// (invariant across s), then the inner sum is a plain diff·xcol dot. Bit-identical: the
	// same residual and the same ascending-i f64 accumulation as before.
	xc, wqc := x.Contiguous(), wq.Contiguous()
	var ss float64
	diff := make([]float64, in)
	switch wq.Dtype() {
	case tensor.F64:
		if xf, ok := f64Storage(xc); ok {
			wqf := wqc.Storage().F64()
			for r := range out {
				wr, wqr := w[r], wqf[r*in:r*in+in:r*in+in]
				for i := range diff {
					diff[i] = wr[i] - wqr[i]
				}
				for s := range samples {
					var v float64
					for i := 0; i < in; i++ {
						v += diff[i] * xf[i*samples+s]
					}
					ss += v * v
				}
			}
			return math.Sqrt(ss)
		}
	case tensor.F32:
		if xf, ok := f32Storage(xc); ok {
			wqf := wqc.Storage().F32()
			for r := range out {
				wr := w[r]
				for i := range diff {
					diff[i] = wr[i] - float64(wqf[r*in+i])
				}
				for s := range samples {
					var v float64
					for i := 0; i < in; i++ {
						v += diff[i] * float64(xf[i*samples+s])
					}
					ss += v * v
				}
			}
			return math.Sqrt(ss)
		}
	}
	// Exotic-dtype fallback (verbatim per-element dispatch).
	for r := range out {
		for s := range samples {
			var v float64
			//perfscan:ignore PS1005 intentional exotic-dtype fallback; F32/F64 take the typed path above
			for i := range in {
				v += (w[r][i] - wq.AtF64(r, i)) * x.AtF64(i, s)
			}
			ss += v * v
		}
	}
	return math.Sqrt(ss)
}

// f64Storage/f32Storage return the contiguous typed backing slice of a tensor whose dtype
// matches, or ok=false to route the caller to its dispatch fallback.
func f64Storage(t *tensor.Tensor) ([]float64, bool) {
	if t.Dtype() != tensor.F64 {
		return nil, false
	}
	return t.Storage().F64(), true
}

func f32Storage(t *tensor.Tensor) ([]float32, bool) {
	if t.Dtype() != tensor.F32 {
		return nil, false
	}
	return t.Storage().F32(), true
}
