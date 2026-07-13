package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// GPTQ post-training weight quantization (Frantar, Ashkboos, Hoefler & Alistarh 2022,
// "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
// arXiv:2210.17323, ICLR'23). Unlike data-free round-to-nearest (RTN) or NF4, GPTQ is
// CALIBRATION-AWARE: it quantizes a linear layer's weight W[out,in] column by column
// and, after rounding each input column, updates the not-yet-quantized columns to
// COMPENSATE for the rounding error, using second-order (Hessian) information from a
// batch of calibration activations X[in,samples]. This minimizes the layer's output
// error ‖W·X − Ŵ·X‖²_F far better than RTN at the same bit-width.

// GPTQOption configures GPTQuantize (functional-options idiom, §C12).
type GPTQOption func(*gptqCfg)

type gptqCfg struct{ damp float64 }

// WithGPTQDamp sets the Hessian dampening fraction (default 0.01): a ridge of
// damp·mean(diag H) is added to the Hessian diagonal for a stable inverse.
func WithGPTQDamp(d float64) GPTQOption { return func(c *gptqCfg) { c.damp = d } }

// UniformQuantizer returns a quantizer that snaps a value to the nearest of `levels`
// evenly spaced points in [lo, hi] (clamping out-of-range values) — a simple symmetric
// grid suitable for GPTQuantize's per-value quant callback.
func UniformQuantizer(levels int, lo, hi float64) func(float64) float64 {
	step := (hi - lo) / float64(levels-1)
	return func(w float64) float64 {
		w = math.Max(lo, math.Min(hi, w))
		return lo + math.Round((w-lo)/step)*step
	}
}

// GPTQuantize returns the GPTQ-quantized version of weight w[out,in] given calibration
// inputs x[in,samples] and a per-value quantizer quant (which snaps a weight to the
// target grid — e.g. from UniformQuantizer). w and x must be rank-2 with matching input
// dimension. The result has w's shape and dtype.
func GPTQuantize(w, x *tensor.Tensor, quant func(float64) float64, opts ...GPTQOption) (*tensor.Tensor, error) {
	if w.Ndim() != 2 || x.Ndim() != 2 {
		return nil, fmt.Errorf("nn: GPTQuantize needs rank-2 w and x, got %v and %v", w.Shape(), x.Shape())
	}
	out, in := w.Shape()[0], w.Shape()[1]
	if x.Shape()[0] != in {
		return nil, fmt.Errorf("nn: GPTQuantize x must be [%d,samples], got %v", in, x.Shape())
	}
	samples := x.Shape()[1]
	cfg := gptqCfg{damp: 0.01}
	for _, o := range opts {
		o(&cfg)
	}

	// H = 2·X·Xᵀ (the layer-wise reconstruction Hessian), in×in.
	h := make([][]float64, in)
	for i := range in {
		h[i] = make([]float64, in)
		for j := range in {
			var s float64
			for k := range samples {
				s += x.AtF64(i, k) * x.AtF64(j, k)
			}
			h[i][j] = 2 * s
		}
	}
	wm := matAt(w) // working weight [out][in]

	// dead columns (no calibration signal) → quantize to 0; then dampen the diagonal.
	var meanDiag float64
	for i := range in {
		meanDiag += h[i][i]
	}
	meanDiag /= float64(in)
	for i := range in {
		if h[i][i] == 0 {
			h[i][i] = 1
			for r := range out {
				wm[r][i] = 0
			}
		}
	}
	damp := cfg.damp * meanDiag
	for i := range in {
		h[i][i] += damp
	}

	// Hinv = upper Cholesky of H⁻¹: H⁻¹ = L·Lᵀ (L lower) ⇒ the upper factor U = Lᵀ,
	// so hinv[i][j] = L[j][i] for j ≥ i.
	flat := make([]float64, in*in)
	for i := range in {
		copy(flat[i*in:(i+1)*in], h[i])
	}
	hinvT, err := linalg.Inverse(tensor.FromFloat64(tensor.Shape{in, in}, flat))
	if err != nil {
		return nil, fmt.Errorf("nn: GPTQuantize inverse Hessian: %w", err)
	}
	lo, err := linalg.Cholesky(hinvT)
	if err != nil {
		return nil, fmt.Errorf("nn: GPTQuantize Cholesky: %w", err)
	}
	hinv := make([][]float64, in)
	for i := range in {
		hinv[i] = make([]float64, in)
		for j := i; j < in; j++ {
			hinv[i][j] = lo.AtF64(j, i) // upper = Lᵀ
		}
	}

	// Greedy column quantization with error propagation to the remaining columns.
	q := tensor.New(w.Dtype(), tensor.Shape{out, in})
	for i := range in {
		d := hinv[i][i]
		for r := range out {
			wi := wm[r][i]
			qv := quant(wi)
			q.SetF64(qv, r, i)
			e := (wi - qv) / d // per-row error, scaled by the inverse-Hessian diagonal
			for j := i + 1; j < in; j++ {
				wm[r][j] -= e * hinv[i][j] // compensate the not-yet-quantized columns
			}
		}
	}
	return q, nil
}
