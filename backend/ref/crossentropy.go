package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// ceAccum collects per-row cross-entropy losses under a backend.Reduction, and is the
// single place the ignore-mask interacts with the reduction — the intersection where
// hand-rolled implementations drift from torch (§V16).
//
// Ignored rows never reach row(), so they contribute nothing to total, nothing to n,
// and leave their ReductionNone slot at its zero-initialised 0. n is therefore the
// NON-IGNORED row count, which is exactly what ReductionMean divides by; with no
// ignore mask n == batch, so the arithmetic is byte-identical to the pre-mask kernel.
type ceAccum struct {
	red   backend.Reduction
	total float64
	n     int
	out   *tensor.Tensor // ReductionNone only: the [batch] per-row loss tensor
}

// row records one surviving row: base is its (possibly smoothed) NLL and zl·lse² its
// PaLM z-loss term. The z-loss term is added as a SEPARATE accumulation step (not folded
// into base) so the mean/sum float64 accumulation order matches the historical kernel bit
// for bit.
func (a *ceAccum) row(i int, base, lse, zl float64) {
	a.n++
	if a.red == backend.ReductionNone {
		v := base
		if zl != 0 {
			v += zl * lse * lse
		}
		a.out.SetF64(v, i)
		return
	}
	a.total += base
	if zl != 0 {
		a.total += zl * lse * lse // PaLM z-loss: coeff·(logsumexp)² keeps log Z ≈ 0 (§R113)
	}
}

// finish writes the reduced result. ReductionMean divides by the non-ignored count, so an
// all-ignored batch gives 0/0 = NaN and ReductionSum gives 0 — both confirmed against torch.
// ReductionNone already wrote every slot (ignored ones stay 0), so there is nothing to do.
func (a *ceAccum) finish(out *tensor.Tensor) {
	switch a.red {
	case backend.ReductionNone:
	case backend.ReductionSum:
		out.SetF64(a.total)
	default:
		out.SetF64(a.total / float64(a.n))
	}
}

// ceOutput allocates the op's output for the given reduction: a scalar for mean/sum,
// a zero-initialised [batch] vector for ReductionNone.
func ceOutput(ctx *backend.Context, dt tensor.Dtype, red backend.Reduction, b int) *tensor.Tensor {
	if red == backend.ReductionNone {
		return tensor.NewOn(ctx.Device(), dt, tensor.Shape{b})
	}
	return tensor.NewOn(ctx.Device(), dt, tensor.Shape{})
}

// ceIgnored reports whether row target ti is masked out by the attrs' ignore index.
func ceIgnored(pa backend.CrossEntropyAttrs, ti int) bool {
	return pa.HasIgnoreIndex && ti == pa.IgnoreIndex
}

// crossEntropyKernel is the fused, numerically stable cross-entropy (ADR-0007):
// per row m = max(z); lse = m + log(Σ exp(z−m)); lossᵢ = lse − z[targetᵢ];
// output = mean over the batch. Accumulation in f64 (§V10); the max-shift keeps
// exp in range for arbitrarily large logits (§V12). targets[b] holds class
// indices as floats (§B12); out-of-range indices error.
//
// attrs LabelSmoothing=ε∈[0,1) (default 0) smooths the target toward the
// uniform distribution (Szegedy et al. 2016; Vaswani et al. 2017 §5.4): the target
// becomes q'(k)=(1−ε)·δ(k,target)+ε/c, giving lossᵢ = lse − (1−ε)·z[targetᵢ] −
// (ε/c)·Σₖ z[k]. ε=0 is exactly the hard-label loss above (§R52).
//
// attrs ZLoss=coeff≥0 (default 0) adds the PaLM z-loss coeff·lseᵢ² per row (§R113,
// Chowdhery et al. 2022): penalising the log-partition lse=log Z keeps the softmax
// normalizer ≈0, preventing logit drift/explosion and improving bf16/fp16 stability.
// Its gradient is 2·coeff·lse·softmax (see the VJP).
//
// attrs IgnoreIndex/HasIgnoreIndex (torch's ignore_index, conventionally -100) masks
// rows whose target equals it: they are skipped entirely — not range-checked, not
// smoothed, no loss, and zero gradient (see the backward kernel). attrs Reduction
// selects mean (default; divides by the NON-IGNORED count), sum, or none (a [batch]
// tensor with 0 in every ignored slot). Both match torch exactly, including the
// all-ignored batch: NaN for mean, 0 for sum, all-zeros for none (§V16).
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
	if pa.Reduction > backend.ReductionNone {
		return nil, fmt.Errorf("ref: crossentropy unknown reduction %d", uint8(pa.Reduction))
	}

	out := ceOutput(ctx, z.Dtype(), pa.Reduction, b)
	acc := &ceAccum{red: pa.Reduction, out: out}

	// Devirtualised fast paths (§T646): this is a hot cpu→ref fallback on the CPU
	// training path (the loss, every step). The generic AtF64 loop below pays a
	// dtype dispatch + flat-offset per logit; here we grab the raw [b,c] slice once
	// (row-major: (i,j) at i*c+j) and index directly. The batch accumulator, per-row
	// softmax sum and row-sum stay float64 on ALL paths and the F32 path reads logits
	// as float64 — so the arithmetic and per-row iteration order are byte-for-byte
	// identical to the generic path (§V9). targets[b] holds class indices as floats
	// (§B12) and is read generically once per row (cheap, dtype-agnostic). The scalar
	// output store (SetF64) rounds to z.Dtype() exactly as before. Contiguous() runs
	// once (returns self when already contiguous).
	switch z.Dtype() {
	case tensor.F64:
		zs := z.Contiguous().Storage().F64()
		for i := range b {
			ti := int(tg.AtF64(i))
			if ceIgnored(pa, ti) {
				continue // masked row: no loss, no count, no range check (torch ignores it wholesale)
			}
			if ti < 0 || ti >= c {
				return nil, fmt.Errorf("ref: crossentropy target %d out of range [0,%d)", ti, c)
			}
			base := i * c
			m := math.Inf(-1)
			for j := range c {
				if v := zs[base+j]; v > m {
					m = v
				}
			}
			var sum, rowSum float64
			for j := range c {
				sum += math.Exp(zs[base+j] - m)
				rowSum += zs[base+j]
			}
			lse := m + math.Log(sum)
			loss := lse - zs[base+ti]
			if eps != 0 {
				loss = lse - (1-eps)*zs[base+ti] - (eps/float64(c))*rowSum
			}
			acc.row(i, loss, lse, pa.ZLoss)
		}
		acc.finish(out)
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		zs := z.Contiguous().Storage().F32()
		for i := range b {
			ti := int(tg.AtF64(i))
			if ceIgnored(pa, ti) {
				continue
			}
			if ti < 0 || ti >= c {
				return nil, fmt.Errorf("ref: crossentropy target %d out of range [0,%d)", ti, c)
			}
			base := i * c
			m := math.Inf(-1)
			for j := range c {
				if v := float64(zs[base+j]); v > m {
					m = v
				}
			}
			var sum, rowSum float64
			for j := range c {
				sum += math.Exp(float64(zs[base+j]) - m)
				rowSum += float64(zs[base+j])
			}
			lse := m + math.Log(sum)
			loss := lse - float64(zs[base+ti])
			if eps != 0 {
				loss = lse - (1-eps)*float64(zs[base+ti]) - (eps/float64(c))*rowSum
			}
			acc.row(i, loss, lse, pa.ZLoss)
		}
		acc.finish(out)
		return []*tensor.Tensor{out}, nil
	}

	// Generic fallback for exotic dtypes.
	for i := range b {
		ti := int(tg.AtF64(i))
		if ceIgnored(pa, ti) {
			continue
		}
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
		loss := lse - z.AtF64(i, ti)
		if eps != 0 {
			loss = lse - (1-eps)*z.AtF64(i, ti) - (eps/float64(c))*rowSum
		}
		acc.row(i, loss, lse, pa.ZLoss)
	}
	acc.finish(out)
	return []*tensor.Tensor{out}, nil
}

// ceGrad resolves the upstream cotangent into a per-row scale and the reduction divisor.
// For mean/sum the cotangent g is a scalar; for ReductionNone it is a [batch] vector, one
// cotangent per row (that is what the tape hands back when the loss is a vector). The
// divisor is the NON-IGNORED row count under ReductionMean and 1 otherwise — mirroring
// the forward, so grad(mean-loss) stays the exact derivative of the forward's mean.
func ceGrad(g *tensor.Tensor, red backend.Reduction, b, nKeep int) (perRow func(int) float64, div float64) {
	div = 1
	if red == backend.ReductionMean {
		div = float64(nKeep)
	}
	if red == backend.ReductionNone && g.Ndim() == 1 && g.Shape()[0] == b {
		return func(i int) float64 { return g.AtF64(i) }, div
	}
	gv := g.AtF64()
	return func(int) float64 { return gv }, div
}

// crossEntropyBackwardKernel is the fused cross-entropy gradient (ADR-0007, §R52/§R113).
// Inputs (z[b,c] logits, targets[b] as float indices, g = the upstream cotangent — a scalar
// for mean/sum, a [batch] vector for ReductionNone); output dz[b,c].
// Per row i, with stable softmax p, smoothed target q'(k)=(1−ε)δ(k,tᵢ)+ε/c and the z-loss
// term 2·zl·lse·p: dz[i,k] = gᵢ·(p_k − q'_k + 2·zl·lse·p_k)/D, where D is the NON-IGNORED
// row count under ReductionMean and 1 under sum/none. Rows masked by IgnoreIndex are left
// at exactly 0 (torch's behaviour, and the reason an all-ignored mean gives NaN loss but
// finite zero gradients). Moved out of the autograd VJP so the loss gradient (which seeds
// the whole backward pass) dispatches on the active backend.
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

	// The mean divisor must count the SAME rows the forward counted, so resolve the
	// non-ignored count up front (it is b when no mask is set — the historical divisor).
	nKeep := b
	if pX.HasIgnoreIndex {
		nKeep = 0
		for i := range b {
			if !ceIgnored(pX, int(tg.AtF64(i))) {
				nKeep++
			}
		}
	}
	gAt, div := ceGrad(g, pX.Reduction, b, nKeep)
	dz := tensor.NewOn(ctx.Device(), z.Dtype(), z.Shape())

	// Devirtualised fast paths (§T646): the loss gradient seeds the whole backward
	// pass and is a hot cpu→ref fallback every training step. The generic AtF64/SetF64
	// loop pays a dtype dispatch + flat-offset per logit; here we grab the raw [b,c]
	// slices once (row-major: (i,j) at i*c+j) and index directly. The per-row max,
	// softmax sum and lse stay float64 on ALL paths and the F32 path reads logits as
	// float64 and only rounds the STORED gradient once — so the math and per-row
	// iteration order are byte-for-byte identical to the generic path (§V9). targets[b]
	// (float class indices, §B12) and the cotangent g are read generically. Contiguous()
	// runs once (returns self when already contiguous). dz is zero-initialised, so an
	// ignored row's `continue` leaves EXACTLY zero gradient with no extra writes.
	switch z.Dtype() {
	case tensor.F64:
		zs := z.Contiguous().Storage().F64()
		dzs := dz.Storage().F64()
		for i := range b {
			ti := int(tg.AtF64(i))
			if ceIgnored(pX, ti) {
				continue
			}
			base := i * c
			m := math.Inf(-1)
			for j := range c {
				if v := zs[base+j]; v > m {
					m = v
				}
			}
			var sum float64
			for j := range c {
				e := math.Exp(zs[base+j] - m)
				dzs[base+j] = e // stash exp(z-m) in the grad slot; reused then overwritten below
				sum += e
			}
			var lseTerm float64
			if zl != 0 {
				lseTerm = 2 * zl * (m + math.Log(sum))
			}
			gv := gAt(i)
			for j := range c {
				p := dzs[base+j] / sum // cached exp(z-m) from the sum pass — identical double, no 2nd Exp
				q := eps / float64(c)
				if j == ti {
					q += 1 - eps
				}
				dzs[base+j] = gv * (p - q + lseTerm*p) / div
			}
		}
		return []*tensor.Tensor{dz}, nil
	case tensor.F32:
		zs := z.Contiguous().Storage().F32()
		dzs := dz.Storage().F32()
		expScr := make([]float64, c) // f64 exp cache (can't stash in the f32 grad slot without rounding)
		for i := range b {
			ti := int(tg.AtF64(i))
			if ceIgnored(pX, ti) {
				continue
			}
			base := i * c
			m := math.Inf(-1)
			for j := range c {
				if v := float64(zs[base+j]); v > m {
					m = v
				}
			}
			var sum float64
			for j := range c {
				e := math.Exp(float64(zs[base+j]) - m)
				expScr[j] = e
				sum += e
			}
			var lseTerm float64
			if zl != 0 {
				lseTerm = 2 * zl * (m + math.Log(sum))
			}
			gv := gAt(i)
			for j := range c {
				p := expScr[j] / sum // cached exp(z-m) — identical double, no 2nd Exp
				q := eps / float64(c)
				if j == ti {
					q += 1 - eps
				}
				dzs[base+j] = float32(gv * (p - q + lseTerm*p) / div)
			}
		}
		return []*tensor.Tensor{dz}, nil
	}

	// Generic fallback for exotic dtypes.
	for i := range b {
		ti := int(tg.AtF64(i))
		if ceIgnored(pX, ti) {
			continue
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
		var lseTerm float64
		if zl != 0 {
			lseTerm = 2 * zl * (m + math.Log(sum))
		}
		gv := gAt(i)
		for j := range c {
			p := math.Exp(z.AtF64(i, j)-m) / sum
			q := eps / float64(c)
			if j == ti {
				q += 1 - eps
			}
			dz.SetF64(gv*(p-q+lseTerm*p)/div, i, j)
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
