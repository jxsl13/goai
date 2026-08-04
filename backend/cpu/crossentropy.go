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
		//perfscan:ignore PS6008 already per-band scratch (once per chunk), not per-row
		scratch := make([]float32, c)
		//perfscan:ignore PS4002 math.Log one-per-row not per-element; c-wide exp already vexp'd
		for i := lo; i < hi; i++ {
			row := zs[i*c : (i+1)*c : (i+1)*c]
			m := rowMaxF32(row) // AVX2 max (amd64-SIMD) / scalar −Inf-start reduction elsewhere
			copy(scratch, row)
			sum := vexpRowF32(scratch, m) // exp(z−m) + 8-lane f32 sum (AVX) / NEON / scalar
			lse := float64(m) + math.Log(float64(sum))
			loss := lse - float64(row[tis[i]])
			if eps != 0 {
				var rowSum float64
				for _, v := range row {
					rowSum += float64(v)
				}
				//perfscan:ignore PS3020 per-row scalar label-smoothing, not a per-element indexed loop
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
		//perfscan:ignore PS3044,PS4002 rows write disjoint dz rows; no shared-slot reduction, already parallel | math.Log one-per-row (zl branch); ex
		for i := lo; i < hi; i++ {
			zr := zs[i*c : (i+1)*c : (i+1)*c]
			dr := dzs[i*c : (i+1)*c : (i+1)*c]
			m := rowMaxF32(zr) // AVX2 max (amd64-SIMD) / scalar −Inf-start reduction elsewhere
			copy(dr, zr)
			sum := vexpRowF32(dr, m) // dr[j] = exp(z−m), reused as p below
			inv := 1 / sum
			var lseTerm float32
			if zl != 0 {
				lseTerm = float32(2 * zl * (float64(m) + math.Log(float64(sum))))
			}
			ti := tis[i]
			// dr holds e=exp(z−m). Common gradient scale·(p·(1+lseTerm) − q0), p=e·inv, is the
			// affine e·k1 − k2 (k1=inv·scale·(1+lseTerm), k2=scale·q0) applied 8-wide; the target
			// column then subtracts scale·(1−eps) (its q carries the extra 1−eps).
			k1 := inv * scale * (1 + lseTerm)
			k2 := scale * q0
			axpbRowF32(dr, k1, -k2)
			dr[ti] -= scale * (1 - eps)
		}
	})
	return []*tensor.Tensor{dz}, nil
}

// crossEntropyF64CPU is the bit-exact parallel F64 twin of crossEntropyKernelCPU: F64
// cross-entropy was a cpu dtype-gap (cpu registered CE for F32 only, so F64 fell to the
// serial ref kernel — PS6006). It reproduces ref's F64 arithmetic exactly (scalar max,
// Σexp(z−m) and Σz in one pass with scalar math.Exp, lse = m+log(sum), the same
// label-smoothed loss), fills the per-row loss and lse in parallel, then replays ref's
// EXACT serial accumulation order — total += loss; if zl≠0 total += zl·lse² as a SEPARATE
// step, interleaved per row (ceAccum.row) — so the mean is byte-identical to ref, not
// merely tolerant. Mean + no-ignore-index only (the hot training case); every other
// reduction / masked row defers to ref, matching the F32 kernel's guard.
func crossEntropyF64CPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
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
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpCrossEntropy, in, attrs)
	}
	tis := make([]int, b)
	for i := range b {
		ti := int(tg.AtF64(i))
		if ti < 0 || ti >= c {
			return nil, fmt.Errorf("cpu: crossentropy target %d out of range [0,%d)", ti, c)
		}
		tis[i] = ti
	}
	zs := z.Contiguous().Storage().F64()
	losses := make([]float64, b)
	lses := make([]float64, b)
	parallelWork(b, 8*c, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			base := i * c
			m := math.Inf(-1)
			for j := 0; j < c; j++ {
				if v := zs[base+j]; v > m {
					m = v
				}
			}
			var sum, rowSum float64
			//perfscan:ignore PS3010 F64 byte-identical to ref; add follows Exp (compute-bound not latency)
			for j := 0; j < c; j++ {
				sum += math.Exp(zs[base+j] - m)
				rowSum += zs[base+j]
			}
			lse := m + math.Log(sum)
			loss := lse - zs[base+tis[i]]
			if eps != 0 {
				//perfscan:ignore PS3020 per-row scalar label-smoothing, not per-element loop
				loss = lse - (1-eps)*zs[base+tis[i]] - (eps/float64(c))*rowSum
			}
			losses[i] = loss
			lses[i] = lse
		}
	})
	zl := pa.ZLoss
	var total float64 // serial, in ref's ceAccum.row order (base then zl·lse² separately)
	for i := 0; i < b; i++ {
		total += losses[i]
		if zl != 0 {
			total += zl * lses[i] * lses[i]
		}
	}
	out := tensor.NewOn(ctx.Device(), z.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

// crossEntropyBackwardF64CPU is the bit-exact parallel F64 twin of
// crossEntropyBackwardKernelCPU: F64 CE backward was a cpu dtype-gap (cpu registered it
// for F32 only → F64 fell to serial ref, PS6006). Rows are independent (each reads its own
// z row, writes its own dz row), so they run across the worker pool; each row reproduces
// ref's exact F64 sequence — stash e=exp(z−m) in the grad slot during the sum pass, then
// dz = gv·(p − q + lseTerm·p)/div with p=e/sum (the SAME per-element expression and order
// as ref, not the rearranged scale form the tol-gated F32 kernel uses) → BYTE-IDENTICAL.
// Mean + no-ignore-index only; every other case defers to ref (the F32 kernel's guard).
func crossEntropyBackwardF64CPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
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
	eps := pX.LabelSmoothing
	zl := pX.ZLoss
	gv := g.AtF64()
	div := float64(b)
	dz := tensor.NewOn(ctx.Device(), z.Dtype(), z.Shape())
	zs := z.Contiguous().Storage().F64()
	dzs := dz.Storage().F64()
	tis := make([]int, b)
	for i := range b {
		tis[i] = int(tg.AtF64(i))
	}
	parallelWork(b, 8*c, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			base := i * c
			m := math.Inf(-1)
			for j := 0; j < c; j++ {
				if v := zs[base+j]; v > m {
					m = v
				}
			}
			var sum float64
			for j := 0; j < c; j++ {
				e := math.Exp(zs[base+j] - m)
				dzs[base+j] = e // stash exp(z-m) in the grad slot; overwritten below
				sum += e
			}
			var lseTerm float64
			if zl != 0 {
				lseTerm = 2 * zl * (m + math.Log(sum))
			}
			ti := tis[i]
			//perfscan:ignore PS5001 F64 keeps /sum for byte-identity to ref (F32 twin uses inv under tolerance)
			for j := 0; j < c; j++ {
				p := dzs[base+j] / sum
				q := eps / float64(c)
				if j == ti {
					q += 1 - eps
				}
				dzs[base+j] = gv * (p - q + lseTerm*p) / div
			}
		}
	})
	return []*tensor.Tensor{dz}, nil
}

func init() {
	// F64 cross-entropy fwd+bwd: scalar-math (no vexp), bit-exact to ref on every build →
	// registered unconditionally, closing the F64→serial-ref dtype-gap (PS6006).
	std.add(backend.OpCrossEntropy, tensor.F64, crossEntropyF64CPU)
	std.add(backend.OpCrossEntropyBackward, tensor.F64, crossEntropyBackwardF64CPU)
	if vexpF32Fast {
		// SIMD perf build, F32 only (§T664): vexp-routed cross-entropy — the LSE
		// exp+sum runs through vexpRowF32 (4-wide NEON on arm64, 8-wide AVX2 on
		// amd64; the single per-row math.Log stays scalar). Compile-time const: the
		// plain (no-simd) build registers nothing and both ops keep their ref f64
		// fallback bit-for-bit. Rides the ADR-0021 f32 tolerance.
		std.add(backend.OpCrossEntropy, tensor.F32, crossEntropyKernelCPU)
		std.add(backend.OpCrossEntropyBackward, tensor.F32, crossEntropyBackwardKernelCPU)
	}
}
