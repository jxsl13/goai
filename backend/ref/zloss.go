package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// zLossKernel is the standalone log-Z softmax regularizer (§R113; ST-MoE router z-loss, Zoph et al.
// 2022; PaLM output z-loss, Chowdhery et al. 2022): for logits[b,n], per row lse = logsumexp =
// m + log(Σ exp(z−m)); loss = Coeff · mean over rows of lse². Penalising the log-partition toward 0
// keeps the softmax normalizer from drifting (numerical stability). Unlike OpCrossEntropy it takes
// ONLY logits (no targets) — for the MoE router logits or any auxiliary use. Accumulation is f64
// (§V10); the max-shift keeps exp in range (§V12).
func zLossKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: zloss wants (logits), got %d inputs", len(in))
	}
	z := in[0]
	if z.Ndim() != 2 {
		return nil, fmt.Errorf("ref: zloss needs logits rank-2, got %dD", z.Ndim())
	}
	b, c := z.Shape()[0], z.Shape()[1]
	if b == 0 {
		return nil, fmt.Errorf("ref: zloss on empty batch")
	}
	pa, _ := attrs.(backend.ZLossAttrs)
	if pa.Coeff < 0 {
		return nil, fmt.Errorf("ref: zloss coefficient %g must be ≥ 0", pa.Coeff)
	}

	var total float64
	for i := range b {
		m := math.Inf(-1)
		for j := range c {
			if v := z.AtF64(i, j); v > m {
				m = v
			}
		}
		var sum float64
		for j := range c {
			sum += math.Exp(z.AtF64(i, j) - m)
		}
		lse := m + math.Log(sum)
		total += lse * lse
	}
	out := tensor.NewOn(ctx.Device(), z.Dtype(), tensor.Shape{})
	out.SetF64(pa.Coeff * total / float64(b))
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpZLoss, tensor.F32, zLossKernel)
	std.add(backend.OpZLoss, tensor.F64, zLossKernel)
}
