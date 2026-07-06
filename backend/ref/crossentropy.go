package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// crossEntropyKernel is the fused, numerically stable cross-entropy (ADR-0007):
// per row m = max(z); lse = m + log(Σ exp(z−m)); lossᵢ = lse − z[targetᵢ];
// output = mean over the batch. Accumulation in f64 (§V10); the max-shift keeps
// exp in range for arbitrarily large logits (§V12). targets[b] holds class
// indices as floats (§B12); out-of-range indices error.
func crossEntropyKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: crossentropy wants (logits, targets), got %d inputs", len(in))
	}
	z, tg := in[0], in[1]
	if z.Ndim() != 2 || tg.Ndim() != 1 {
		return nil, fmt.Errorf("ref: crossentropy needs logits rank-2 and targets rank-1, got %dD, %dD", z.Ndim(), tg.Ndim())
	}
	b, c := z.Shape()[0], z.Shape()[1]
	if tg.Shape()[0] != b {
		return nil, fmt.Errorf("ref: crossentropy targets len %d != batch %d", tg.Shape()[0], b)
	}
	if b == 0 {
		return nil, fmt.Errorf("ref: crossentropy on empty batch")
	}

	var total float64
	for i := range b {
		ti := int(tg.AtF64(i))
		if ti < 0 || ti >= c {
			return nil, fmt.Errorf("ref: crossentropy target %d out of range [0,%d)", ti, c)
		}
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
		total += lse - z.AtF64(i, ti)
	}
	out := tensor.NewOn(ctx.Device(), z.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpCrossEntropy, tensor.F32, crossEntropyKernel)
	std.add(backend.OpCrossEntropy, tensor.F64, crossEntropyKernel)
}
