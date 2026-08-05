package cpu

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// F64 GELU/SiLU backward (VJP) CPU kernels. The F32 VJPs ride vectorized SIMD kernels (registered
// under vexpF32Fast), but the F64 variants had NO CPU kernel and fell to the single-threaded reference
// — a hot cpu→ref fallback on the F64 training path (GELU-backward was ~19% of the GPT step; SiLU-
// backward is the SwiGLU-FFN VJP on every Llama/Qwen/Mistral layer). The per-element work is a real
// erf/exp (GELU) or a stable sigmoid (SiLU), i.e. compute-bound, so fanning the flat range across cores
// scales well. These replicate the reference formulas op-for-op — same constants, same math.Erf/Exp,
// same stable-sigmoid branch, same evaluation order — so the result is BIT-IDENTICAL to the ref, and
// each ds[i] is a disjoint write (no reduction).

const cpuInvSqrt2Pi = 0.3989422804014327 // 1/√(2π), matches the reference's refInvSqrt2Pi

// geluGradF64 = g·(Φ(x)+x·φ(x)), Φ=0.5(1+erf(x/√2)), φ=(1/√2π)exp(−x²/2) — the exact reference form.
func geluGradF64(x, g float64) float64 {
	phi := 0.5 * (1 + math.Erf(x/math.Sqrt2))
	pdf := cpuInvSqrt2Pi * math.Exp(-0.5*x*x)
	return g * (phi + x*pdf)
}

// stableSigmoidF64 mirrors the reference's branchy overflow-safe sigmoid exactly.
func stableSigmoidF64(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

// siluGradF64 = g·σ(x)(1+x(1−σ(x))) — the exact reference SiLU derivative.
func siluGradF64(x, g float64) float64 {
	s := stableSigmoidF64(x)
	return g * s * (1 + x*(1-s))
}

func activationBackwardF64(ctx *backend.Context, op backend.Op, in []*tensor.Tensor, attrs backend.Attrs, grad func(x, g float64) float64) ([]*tensor.Tensor, error) {
	if len(in) == 2 && in[0].Dtype() == tensor.F64 && in[1].Dtype() == tensor.F64 && in[1].Shape().Equal(in[0].Shape()) {
		xc, gc := in[0].Contiguous(), in[1].Contiguous()
		dx := tensor.NewOn(ctx.Device(), tensor.F64, in[0].Shape())
		xs, gs, ds := xc.Storage().F64(), gc.Storage().F64(), dx.Storage().F64()
		parallel(len(ds), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				ds[i] = grad(xs[i], gs[i])
			}
		})
		return []*tensor.Tensor{dx}, nil
	}
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), op, in, attrs)
}

func geluBackwardF64KernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	return activationBackwardF64(ctx, backend.OpGELUBackward, in, attrs, geluGradF64)
}

func siluBackwardF64KernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	return activationBackwardF64(ctx, backend.OpSiLUBackward, in, attrs, siluGradF64)
}

func init() {
	//perfscan:ignore PS3052 false-positive: kernel registration init, one-time
	std.add(backend.OpGELUBackward, tensor.F64, geluBackwardF64KernelCPU)
	std.add(backend.OpSiLUBackward, tensor.F64, siluBackwardF64KernelCPU)
}
