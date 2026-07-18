package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// F32 cross-entropy on the vexp leaf (§T664): the fused loss + its gradient
// were hot cpu→ref fallbacks on the f32 GPT training step (fwd 3.5% + bwd
// 7.2% of the §V22 op-profile) — serial rows with scalar f64 math.Exp per
// logit, and the backward evaluating every exp TWICE (once for the row sum,
// once for p). These kernels keep ref's exact row math (ADR-0007: stable
// max-shift softmax, label smoothing §R52, PaLM z-loss §R113 — same formulas,
// same errors) but evaluate f32-native: the per-row exp+sum runs through
// vexpRowF32 (the shared 4-wide NEON exp band, vexp.go), the backward reuses
// that single exp pass for p (dz doubles as the scratch row), and rows split
// across the worker pool. Per-row results are deterministic (the batch mean
// sums a per-row loss array serially, in order). Gated exactly like the other
// vexp paths: registered ONLY when vexpNeon and ONLY for F32 — the default
// build, amd64, and F64 everywhere keep the untouched ref fallback
// bit-for-bit. Numerics ride the ADR-0021 tolerant-parity budget (rtol 2e-3):
// the vexp poly adds ~3e-7 relative, the 4-lane f32 row sum ≤ c/4·2⁻²⁴
// (~6e-5 at c=4096), both orders inside budget (TestCPUCrossEntropyParity).

// crossEntropyKernelCPU: loss = mean over rows of lse − Σₖ q'(k)·z[k]
// (+ zl·lse²), q'(k) = (1−ε)·δ(k,tᵢ) + ε/c. Same guards and error text
// shape as ref's crossEntropyKernel.
func crossEntropyKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: crossentropy wants (logits, targets), got %d inputs", len(in))
	}
	z, tg := in[0], in[1]
	if z.Ndim() != 2 || tg.Ndim() != 1 {
		return nil, fmt.Errorf("cpu: crossentropy needs logits rank-2 and targets rank-1, got %dD, %dD", z.Ndim(), tg.Ndim())
	}
	b, c := z.Shape()[0], z.Shape()[1]
	if tg.Shape()[0] != b {
		return nil, fmt.Errorf("cpu: crossentropy targets len %d != batch %d", tg.Shape()[0], b)
	}
	if b == 0 {
		return nil, fmt.Errorf("cpu: crossentropy on empty batch")
	}
	pa, _ := attrs.(backend.CrossEntropyAttrs)
	eps := pa.LabelSmoothing
	if eps < 0 || eps >= 1 {
		return nil, fmt.Errorf("cpu: crossentropy label_smoothing %g out of [0,1)", eps)
	}
	if pa.ZLoss < 0 {
		return nil, fmt.Errorf("cpu: crossentropy z_loss coefficient %g must be ≥ 0", pa.ZLoss)
	}
	if pa.HasIgnoreIndex || pa.Reduction != backend.ReductionMean {
		// Masked rows / non-mean reductions stay on the reference (§I4): this vexp path
		// exists purely for the hot unmasked mean training loop, and a partial
		// implementation here would silently diverge from ref (TestCEGuardFallback).
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpCrossEntropy, in, attrs)
	}

	// Targets validated serially up front (cheap, b reads) so the parallel row
	// loop below is error-free; first bad index reported like ref's serial loop.
	tis := make([]int, b)
	for i := range b {
		ti := int(tg.AtF64(i))
		if ti < 0 || ti >= c {
			return nil, fmt.Errorf("cpu: crossentropy target %d out of range [0,%d)", ti, c)
		}
		tis[i] = ti
	}

	zs := z.Contiguous().Storage().F32()
	losses := make([]float64, b) // per-row losses: parallel fill, serial sum (deterministic)
	parallelWork(b, 8*c, func(lo, hi int) {
		scratch := make([]float32, c)
		for i := lo; i < hi; i++ {
			row := zs[i*c : (i+1)*c : (i+1)*c]
			m := float32(math.Inf(-1))
			for _, v := range row {
				if v > m {
					m = v
				}
			}
			copy(scratch, row)
			sum := vexpRowF32(scratch, m) // exp(z−m) + 4-lane f32 sum, NEON
			lse := float64(m) + math.Log(float64(sum))
			loss := lse - float64(row[tis[i]])
			if eps != 0 {
				var rowSum float64
				for _, v := range row {
					rowSum += float64(v)
				}
				loss = lse - (1-eps)*float64(row[tis[i]]) - (eps/float64(c))*rowSum
			}
			if pa.ZLoss != 0 {
				loss += pa.ZLoss * lse * lse // PaLM z-loss (§R113)
			}
			losses[i] = loss
		}
	})
	var total float64
	for _, l := range losses {
		total += l
	}
	out := tensor.NewOn(ctx.Device(), z.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

// crossEntropyBackwardKernelCPU: dz[i,k] = g·(p_k − q'_k + 2·zl·lse·p_k)/b per
// row (ADR-0007) — ONE vexp pass per row: dz's row doubles as the exp scratch
// (copy z row in, exp in place, then scale to the gradient), where ref's
// scalar loop paid math.Exp twice per logit. Same (lack of) target-range
// guarding as ref: an out-of-range index just never matches j.
func crossEntropyBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: crossentropy-backward wants (z, targets, g), got %d", len(in))
	}
	z, tg, g := in[0], in[1], in[2]
	if z.Ndim() != 2 || tg.Ndim() != 1 {
		return nil, fmt.Errorf("cpu: crossentropy-backward needs z rank-2 and targets rank-1, got %dD/%dD", z.Ndim(), tg.Ndim())
	}
	b, c := z.Shape()[0], z.Shape()[1]
	if tg.Shape()[0] != b {
		return nil, fmt.Errorf("cpu: crossentropy-backward targets len %d != batch %d", tg.Shape()[0], b)
	}
	pX, _ := attrs.(backend.CrossEntropyAttrs)
	if pX.HasIgnoreIndex || pX.Reduction != backend.ReductionMean {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpCrossEntropyBackward, in, attrs)
	}
	eps := float32(pX.LabelSmoothing)
	zl := pX.ZLoss
	gv := g.AtF64()
	dz := tensor.NewOn(ctx.Device(), z.Dtype(), z.Shape())

	zs := z.Contiguous().Storage().F32()
	dzs := dz.Storage().F32()
	tis := make([]int, b)
	for i := range b {
		tis[i] = int(tg.AtF64(i))
	}
	scale := float32(gv / float64(b))
	q0 := eps / float32(c)
	parallelWork(b, 8*c, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			zr := zs[i*c : (i+1)*c : (i+1)*c]
			dr := dzs[i*c : (i+1)*c : (i+1)*c]
			m := float32(math.Inf(-1))
			for _, v := range zr {
				if v > m {
					m = v
				}
			}
			copy(dr, zr)
			sum := vexpRowF32(dr, m) // dr[j] = exp(z−m), reused as p below
			inv := 1 / sum
			var lseTerm float32
			if zl != 0 {
				lseTerm = float32(2 * zl * (float64(m) + math.Log(float64(sum))))
			}
			ti := tis[i]
			for j, e := range dr {
				p := e * inv
				q := q0
				if j == ti {
					q += 1 - eps
				}
				dr[j] = scale * (p - q + lseTerm*p)
			}
		}
	})
	return []*tensor.Tensor{dz}, nil
}

func init() {
	if vexpNeon {
		// arm64 perf build only, F32 only (§T664): vexp-routed cross-entropy.
		// vexpNeon is a compile-time const, so every other build registers
		// nothing here and both ops keep their ref fallback bit-for-bit (as
		// does F64 on this build).
		std.add(backend.OpCrossEntropy, tensor.F32, crossEntropyKernelCPU)
		std.add(backend.OpCrossEntropyBackward, tensor.F32, crossEntropyBackwardKernelCPU)
	}
}
