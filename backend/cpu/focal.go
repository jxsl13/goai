package cpu

import (
	"errors"
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func sigmoidFocalKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: sigmoid focal core wants (logits, targets), got %d inputs", len(in))
	}
	logits, targets := in[0], in[1]
	if logits.Dtype() != targets.Dtype() || !logits.Shape().Equal(targets.Shape()) {
		return nil, errors.New("cpu: sigmoid focal core inputs must share dtype and shape")
	}
	pa, _ := attrs.(backend.SigmoidFocalAttrs)
	logits, targets = logits.Contiguous(), targets.Contiguous()
	out := tensor.NewOn(ctx.Device(), logits.Dtype(), logits.Shape())
	switch logits.Dtype() {
	case tensor.F32:
		xs, ys, os := logits.Storage().F32(), targets.Storage().F32(), out.Storage().F32()
		parallel(len(os), func(lo, hi int) { sigmoidFocalF32(os[lo:hi], xs[lo:hi], ys[lo:hi], pa.Gamma, pa.Alpha) })
	case tensor.F64:
		xs, ys, os := logits.Storage().F64(), targets.Storage().F64(), out.Storage().F64()
		parallel(len(os), func(lo, hi int) { sigmoidFocalF64(os[lo:hi], xs[lo:hi], ys[lo:hi], pa.Gamma, pa.Alpha) })
	default:
		return nil, fmt.Errorf("cpu: sigmoid focal core unsupported dtype %v", logits.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func sigmoidFocalBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: sigmoid focal core backward wants (logits, targets, grad), got %d inputs", len(in))
	}
	logits, targets, grad := in[0], in[1], in[2]
	if logits.Dtype() != targets.Dtype() || logits.Dtype() != grad.Dtype() ||
		!logits.Shape().Equal(targets.Shape()) || !logits.Shape().Equal(grad.Shape()) {
		return nil, errors.New("cpu: sigmoid focal core backward inputs must share dtype and shape")
	}
	pa, _ := attrs.(backend.SigmoidFocalAttrs)
	logits, targets, grad = logits.Contiguous(), targets.Contiguous(), grad.Contiguous()
	dLogits := tensor.NewOn(ctx.Device(), logits.Dtype(), logits.Shape())
	switch logits.Dtype() {
	case tensor.F32:
		xs, ys, gs, ds := logits.Storage().F32(), targets.Storage().F32(), grad.Storage().F32(), dLogits.Storage().F32()
		parallel(len(ds), func(lo, hi int) {
			sigmoidFocalBackwardF32(ds[lo:hi], xs[lo:hi], ys[lo:hi], gs[lo:hi], pa.Gamma, pa.Alpha)
		})
	case tensor.F64:
		xs, ys, gs, ds := logits.Storage().F64(), targets.Storage().F64(), grad.Storage().F64(), dLogits.Storage().F64()
		parallel(len(ds), func(lo, hi int) {
			sigmoidFocalBackwardF64(ds[lo:hi], xs[lo:hi], ys[lo:hi], gs[lo:hi], pa.Gamma, pa.Alpha)
		})
	default:
		return nil, fmt.Errorf("cpu: sigmoid focal core backward unsupported dtype %v", logits.Dtype())
	}
	return []*tensor.Tensor{dLogits}, nil
}

func stableSoftplus64(x float64) float64 {
	if x > 0 {
		return x + math.Log1p(math.Exp(-x))
	}
	return math.Log1p(math.Exp(x))
}

func stableSigmoid64(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

// focalSoftplusPair computes softplus(z) and softplus(-z) from their shared
// log1p(exp(-abs(z))) base. This is algebraically the same branch structure as
// two standalone stable-softplus calls, including their store-rounding points,
// but removes one exponential and one logarithm per focal element.
func focalSoftplusPairF32(z float32) (float32, float32) {
	negZ := math.Float32frombits(math.Float32bits(z) ^ 1<<31)
	if vexpF32Fast {
		a := math.Float32frombits(math.Float32bits(z) &^ (1 << 31))
		base := logF32(1 + expF32(-a))
		if z > 0 {
			return float32(z + base), base
		}
		if z < 0 {
			return base, float32(negZ + base)
		}
		return base, base
	}
	base := math.Log1p(math.Exp(-math.Abs(float64(z))))
	if z > 0 {
		return float32(float64(z) + base), float32(base)
	}
	if z < 0 {
		return float32(base), float32(float64(negZ) + base)
	}
	b := float32(base)
	return b, b
}

func focalSoftplusPairF64(z float64) (float64, float64) {
	base := math.Log1p(math.Exp(-math.Abs(z)))
	if z > 0 {
		return z + base, base
	}
	if z < 0 {
		return base, -z + base
	}
	return base, base
}

func focalAlphaF32(y float32, alpha float64) float32 {
	if alpha < 0 {
		return 1
	}
	if y == 1 {
		return float32(alpha)
	}
	return float32(1 - alpha)
}

func focalAlphaF64(y, alpha float64) float64 {
	if alpha < 0 {
		return 1
	}
	if y == 1 {
		return alpha
	}
	return 1 - alpha
}

// focalSoftplusGradF32 keeps the SIMD sigmoid leaf on its normal domain, but
// falls back to the exact scalar evaluation when expF32's underflow clamp can
// be magnified by focal's upstream product. The fallback is a cold extreme
// tail; it also preserves Inf*0 = NaN from the reference VJP.
func focalSoftplusGradF32(x, g float32) float32 {
	if x < expLoClamp {
		return float32(float64(g) * stableSigmoid64(float64(x)))
	}
	return softplusGradF32(x, g)
}

func sigmoidFocalF32(dst, logits, targets []float32, gamma, alpha float64) {
	logits, targets = logits[:len(dst)], targets[:len(dst)]
	if gamma == 0 {
		for i := range dst {
			sign := float32(1 - 2*targets[i])
			z := float32(logits[i] * sign)
			var spz float32
			if vexpF32Fast {
				spz = softplusF32(z)
			} else {
				spz = float32(stableSoftplus64(float64(z)))
			}
			dst[i] = float32(spz * focalAlphaF32(targets[i], alpha))
		}
		return
	}
	c := float32(-gamma)
	for i := range dst {
		sign := float32(1 - 2*targets[i])
		z := float32(logits[i] * sign)
		spz, spNegZ := focalSoftplusPairF32(z)
		var mod float32
		if vexpF32Fast {
			mod = expFullF32(float32(spNegZ * c))
		} else {
			mod = float32(math.Exp(float64(float32(spNegZ * c))))
		}
		term := float32(spz * mod)
		dst[i] = float32(term * focalAlphaF32(targets[i], alpha))
	}
}

func sigmoidFocalF64(dst, logits, targets []float64, gamma, alpha float64) {
	logits, targets = logits[:len(dst)], targets[:len(dst)]
	if gamma == 0 {
		for i := range dst {
			sign := float64(1 - 2*targets[i])
			z := float64(logits[i] * sign)
			dst[i] = float64(stableSoftplus64(z) * focalAlphaF64(targets[i], alpha))
		}
		return
	}
	c := -gamma
	for i := range dst {
		sign := float64(1 - 2*targets[i])
		z := float64(logits[i] * sign)
		spz, spNegZ := focalSoftplusPairF64(z)
		mod := math.Exp(float64(spNegZ * c))
		term := float64(spz * mod)
		dst[i] = float64(term * focalAlphaF64(targets[i], alpha))
	}
}

func sigmoidFocalBackwardF32(dst, logits, targets, grad []float32, gamma, alpha float64) {
	logits, targets, grad = logits[:len(dst)], targets[:len(dst)], grad[:len(dst)]
	if gamma == 0 {
		for i := range dst {
			sign := float32(1 - 2*targets[i])
			z := float32(logits[i] * sign)
			gTerm := float32(grad[i] * focalAlphaF32(targets[i], alpha))
			var gz float32
			if vexpF32Fast {
				gz = softplusGradF32(z, gTerm)
			} else {
				gz = float32(float64(gTerm) * stableSigmoid64(float64(z)))
			}
			dst[i] = float32(gz * sign)
		}
		return
	}
	c := float32(-gamma)
	for i := range dst {
		sign := float32(1 - 2*targets[i])
		z := float32(logits[i] * sign)
		alphaT := focalAlphaF32(targets[i], alpha)
		gTerm := float32(grad[i] * alphaT)
		negZ := math.Float32frombits(math.Float32bits(z) ^ 1<<31)
		spz, spNegZ := focalSoftplusPairF32(z)
		var mod float32
		if vexpF32Fast {
			mod = expFullF32(float32(spNegZ * c))
		} else {
			mod = float32(math.Exp(float64(float32(spNegZ * c))))
		}
		gu := float32(gTerm * mod)
		gm := float32(gTerm * spz)
		gq := float32(gm * mod)
		gv := float32(gq * c)
		var gzU, gNegZ float32
		if vexpF32Fast {
			gzU = focalSoftplusGradF32(z, gu)
			gNegZ = focalSoftplusGradF32(negZ, gv)
		} else {
			gzU = float32(float64(gu) * stableSigmoid64(float64(z)))
			gNegZ = float32(float64(gv) * stableSigmoid64(float64(negZ)))
		}
		gzV := math.Float32frombits(math.Float32bits(gNegZ) ^ 1<<31)
		gz := float32(gzV + gzU)
		dst[i] = float32(gz * sign)
	}
}

func sigmoidFocalBackwardF64(dst, logits, targets, grad []float64, gamma, alpha float64) {
	logits, targets, grad = logits[:len(dst)], targets[:len(dst)], grad[:len(dst)]
	if gamma == 0 {
		for i := range dst {
			sign := float64(1 - 2*targets[i])
			z := float64(logits[i] * sign)
			gTerm := float64(grad[i] * focalAlphaF64(targets[i], alpha))
			gz := float64(gTerm * stableSigmoid64(z))
			dst[i] = float64(gz * sign)
		}
		return
	}
	c := -gamma
	for i := range dst {
		sign := float64(1 - 2*targets[i])
		z := float64(logits[i] * sign)
		gTerm := float64(grad[i] * focalAlphaF64(targets[i], alpha))
		spz, spNegZ := focalSoftplusPairF64(z)
		mod := math.Exp(float64(spNegZ * c))
		gu := float64(gTerm * mod)
		gm := float64(gTerm * spz)
		gq := float64(gm * mod)
		gv := float64(gq * c)
		gzU := float64(gu * stableSigmoid64(z))
		gNegZ := float64(gv * stableSigmoid64(-z))
		gzV := -gNegZ
		gz := float64(gzV + gzU)
		dst[i] = float64(gz * sign)
	}
}

func init() {
	std.add(backend.OpSigmoidFocalCore, tensor.F32, sigmoidFocalKernelCPU)
	std.add(backend.OpSigmoidFocalCore, tensor.F64, sigmoidFocalKernelCPU)
	std.add(backend.OpSigmoidFocalCoreBackward, tensor.F32, sigmoidFocalBackwardKernelCPU)
	std.add(backend.OpSigmoidFocalCoreBackward, tensor.F64, sigmoidFocalBackwardKernelCPU)
}
