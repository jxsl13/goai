package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// SmoothQuant implements the SmoothQuant activation-weight smoothing transform
// (Xiao, Lin, Seznec, Wu, Demouth & Han 2022/2023, "SmoothQuant: Accurate and
// Efficient Post-Training Quantization for Large Language Models", arXiv:2211.10438,
// ICML'23). LLM activations have a few systematic per-channel OUTLIER dimensions
// that make them far harder to quantize than the weights. SmoothQuant migrates that
// difficulty into the weights with a per-input-channel smoothing factor s, applied
// as a mathematically-equivalent rescaling of a linear layer Y = X·W:
//
//	X̂ = X·diag(s)⁻¹ ,  Ŵ = diag(s)·W   ⇒   X̂·Ŵ = X·W   (exactly)
//	s_j = max(|X_j|)^α / max(|W_j|)^(1−α)                 (Eq. 4)
//
// so the outliers move from X into W and BOTH X̂ (per-tensor) and Ŵ (per-channel)
// become easy to quantize to INT8 (W8A8). α ∈ [0,1] is the migration strength (0.5
// balances the two; the paper's default). The scale is normally fused into the
// preceding LayerNorm at deploy time so no extra runtime op is needed.
//
// X is [tokens, C_in] calibration activations, W is [C_in, C_out]. Returns the
// smoothed X̂ and Ŵ and the per-channel scale s. This is a post-training transform
// (a pure-f64 utility, like AWQuantize/GPTQ), not a differentiable op.
func SmoothQuant(x, w *tensor.Tensor, alpha float64) (xHat, wHat *tensor.Tensor, scale []float64, err error) {
	if x.Ndim() != 2 || w.Ndim() != 2 {
		return nil, nil, nil, fmt.Errorf("nn: SmoothQuant wants rank-2 X[tokens,C_in], W[C_in,C_out], got %v, %v", x.Shape(), w.Shape())
	}
	cin := x.Shape()[1]
	if w.Shape()[0] != cin {
		return nil, nil, nil, fmt.Errorf("nn: SmoothQuant C_in mismatch: X has %d, W has %d", cin, w.Shape()[0])
	}
	if alpha < 0 || alpha > 1 {
		return nil, nil, nil, fmt.Errorf("nn: SmoothQuant alpha=%g out of [0,1]", alpha)
	}
	tokens, cout := x.Shape()[0], w.Shape()[1]
	scale = SmoothQuantScale(actAbsMax(x), weightAbsMax(w), alpha)

	xHat = tensor.New(x.Dtype(), x.Shape())
	if xs, xhs := flatF64(x), flatF64(xHat); xs != nil && xhs != nil {
		for t := range tokens {
			base := t * cin
			for j := range cin {
				xhs[base+j] = xs[base+j] / scale[j]
			}
		}
	} else {
		for t := range tokens {
			for j := range cin {
				xHat.SetF64(x.AtF64(t, j)/scale[j], t, j) // X̂_j = X_j / s_j
			}
		}
	}
	wHat = tensor.New(w.Dtype(), w.Shape())
	if ws, whs := flatF64(w), flatF64(wHat); ws != nil && whs != nil {
		for j := range cin {
			sj, base := scale[j], j*cout
			for o := range cout {
				whs[base+o] = sj * ws[base+o]
			}
		}
	} else {
		for j := range cin {
			for o := range cout {
				wHat.SetF64(scale[j]*w.AtF64(j, o), j, o) // Ŵ_j = s_j · W_j (scale input-channel row)
			}
		}
	}
	return xHat, wHat, scale, nil
}

// SmoothQuantScale returns the per-input-channel smoothing factor s_j =
// max(|X_j|)^α / max(|W_j|)^(1−α) (Eq. 4) from the per-channel activation and weight
// absolute maxima. Degenerate channels (a zero max) fall back to s_j = 1 so the
// transform stays a no-op there rather than dividing by zero.
func SmoothQuantScale(actAbsMax, weightAbsMax []float64, alpha float64) []float64 {
	s := make([]float64, len(actAbsMax))
	for j := range s {
		a, wv := actAbsMax[j], weightAbsMax[j]
		if a <= 0 || wv <= 0 {
			s[j] = 1
			continue
		}
		s[j] = math.Pow(a, alpha) / math.Pow(wv, 1-alpha)
	}
	return s
}

// actAbsMax returns the per-channel (per-column) maximum absolute activation over
// all tokens (rows) of X[tokens, C_in].
func actAbsMax(x *tensor.Tensor) []float64 {
	tokens, cin := x.Shape()[0], x.Shape()[1]
	m := make([]float64, cin)
	if xs := flatF64(x); xs != nil {
		for t := range tokens {
			base := t * cin
			for j := range cin {
				if a := math.Abs(xs[base+j]); a > m[j] {
					m[j] = a
				}
			}
		}
		return m
	}
	for t := range tokens {
		for j := range cin {
			if a := math.Abs(x.AtF64(t, j)); a > m[j] {
				m[j] = a
			}
		}
	}
	return m
}

// weightAbsMax returns the per-input-channel (per-row) maximum absolute weight over
// all output channels of W[C_in, C_out].
func weightAbsMax(w *tensor.Tensor) []float64 {
	cin, cout := w.Shape()[0], w.Shape()[1]
	m := make([]float64, cin)
	if ws := flatF64(w); ws != nil {
		for j := range cin {
			base := j * cout
			for o := range cout {
				if a := math.Abs(ws[base+o]); a > m[j] {
					m[j] = a
				}
			}
		}
		return m
	}
	for j := range cin {
		for o := range cout {
			if a := math.Abs(w.AtF64(j, o)); a > m[j] {
				m[j] = a
			}
		}
	}
	return m
}
