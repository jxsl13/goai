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
//
// attrs["label_smoothing"]=ε∈[0,1) (default 0) smooths the target toward the
// uniform distribution (Szegedy et al. 2016; Vaswani et al. 2017 §5.4): the target
// becomes q'(k)=(1−ε)·δ(k,target)+ε/c, giving lossᵢ = lse − (1−ε)·z[targetᵢ] −
// (ε/c)·Σₖ z[k]. ε=0 is exactly the hard-label loss above (§R52).
//
// attrs ZLoss=coeff≥0 (default 0) adds the PaLM z-loss coeff·lseᵢ² per row (§R113,
// Chowdhery et al. 2022): penalising the log-partition lse=log Z keeps the softmax
// normalizer ≈0, preventing logit drift/explosion and improving bf16/fp16 stability.
// Added to the mean loss; its gradient is 2·coeff·lse·softmax (see the VJP).
func crossEntropyKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
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
	pa, _ := attrs.(backend.CrossEntropyAttrs)
	eps := pa.LabelSmoothing
	if eps < 0 || eps >= 1 {
		return nil, fmt.Errorf("ref: crossentropy label_smoothing %g out of [0,1)", eps)
	}
	if pa.ZLoss < 0 {
		return nil, fmt.Errorf("ref: crossentropy z_loss coefficient %g must be ≥ 0", pa.ZLoss)
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
		var sum, rowSum float64
		for j := range c {
			sum += math.Exp(z.AtF64(i, j) - m)
			rowSum += z.AtF64(i, j)
		}
		lse := m + math.Log(sum)
		if eps == 0 {
			total += lse - z.AtF64(i, ti)
		} else {
			total += lse - (1-eps)*z.AtF64(i, ti) - (eps/float64(c))*rowSum
		}
		if pa.ZLoss != 0 {
			total += pa.ZLoss * lse * lse // PaLM z-loss: coeff·(logsumexp)² keeps log Z ≈ 0 (§R113)
		}
	}
	out := tensor.NewOn(ctx.Device(), z.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

// crossEntropyBackwardKernel is the fused cross-entropy gradient (ADR-0007, §R52/§R113).
// Inputs (z[b,c] logits, targets[b] as float indices, g = scalar upstream); output dz[b,c].
// Per row i, with stable softmax p, smoothed target q'(k)=(1−ε)δ(k,tᵢ)+ε/c and the z-loss
// term 2·zl·lse·p: dz[i,k] = g·(p_k − q'_k + 2·zl·lse·p_k)/b. Moved out of the autograd VJP
// so the loss gradient (which seeds the whole backward pass) dispatches on the active backend.
func crossEntropyBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: crossentropy-backward wants (z, targets, g), got %d", len(in))
	}
	z, tg, g := in[0], in[1], in[2]
	if z.Ndim() != 2 || tg.Ndim() != 1 {
		return nil, fmt.Errorf("ref: crossentropy-backward needs z rank-2 and targets rank-1, got %dD/%dD", z.Ndim(), tg.Ndim())
	}
	b, c := z.Shape()[0], z.Shape()[1]
	if tg.Shape()[0] != b {
		return nil, fmt.Errorf("ref: crossentropy-backward targets len %d != batch %d", tg.Shape()[0], b)
	}
	pX, _ := attrs.(backend.CrossEntropyAttrs)
	eps := pX.LabelSmoothing
	zl := pX.ZLoss
	gv := g.AtF64()
	dz := tensor.NewOn(ctx.Device(), z.Dtype(), z.Shape())
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
		ti := int(tg.AtF64(i))
		var lseTerm float64
		if zl != 0 {
			lseTerm = 2 * zl * (m + math.Log(sum))
		}
		for j := range c {
			p := math.Exp(z.AtF64(i, j)-m) / sum
			q := eps / float64(c)
			if j == ti {
				q += 1 - eps
			}
			dz.SetF64(gv*(p-q+lseTerm*p)/float64(b), i, j)
		}
	}
	return []*tensor.Tensor{dz}, nil
}

func init() {
	std.add(backend.OpCrossEntropy, tensor.F32, crossEntropyKernel)
	std.add(backend.OpCrossEntropy, tensor.F64, crossEntropyKernel)
	std.add(backend.OpCrossEntropyBackward, tensor.F32, crossEntropyBackwardKernel)
	std.add(backend.OpCrossEntropyBackward, tensor.F64, crossEntropyBackwardKernel)
}
