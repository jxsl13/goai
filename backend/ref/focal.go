package ref

import (
	"errors"
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func sigmoidFocalKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: sigmoid focal core wants (logits, targets), got %d inputs", len(in))
	}
	logits, targets := in[0], in[1]
	if logits.Dtype() != targets.Dtype() || !logits.Shape().Equal(targets.Shape()) {
		return nil, errors.New("ref: sigmoid focal core inputs must share dtype and shape")
	}
	pa, _ := attrs.(backend.SigmoidFocalAttrs)
	logits, targets = logits.Contiguous(), targets.Contiguous()
	out := tensor.NewOn(ctx.Device(), logits.Dtype(), logits.Shape())
	switch logits.Dtype() {
	case tensor.F32:
		xs, ys, os := logits.Storage().F32(), targets.Storage().F32(), out.Storage().F32()
		c := float32(-pa.Gamma)
		if pa.Gamma == 0 {
			for i := range os {
				sign := float32(1 - 2*ys[i])
				z := float32(xs[i] * sign)
				spz := float32(sigmoidFocalSoftplus(float64(z)))
				os[i] = float32(spz * sigmoidFocalAlphaF32(ys[i], pa.Alpha))
			}
			break
		}
		for i := range os {
			sign := float32(1 - 2*ys[i])
			z := float32(xs[i] * sign)
			spz, spNegZ := sigmoidFocalSoftplusPairF32(z)
			mod := float32(math.Exp(float64(float32(spNegZ * c))))
			term := float32(spz * mod)
			os[i] = float32(term * sigmoidFocalAlphaF32(ys[i], pa.Alpha))
		}
	case tensor.F64:
		xs, ys, os := logits.Storage().F64(), targets.Storage().F64(), out.Storage().F64()
		c := -pa.Gamma
		if pa.Gamma == 0 {
			for i := range os {
				sign := float64(1 - 2*ys[i])
				z := float64(xs[i] * sign)
				spz := sigmoidFocalSoftplus(z)
				os[i] = float64(spz * sigmoidFocalAlphaF64(ys[i], pa.Alpha))
			}
			break
		}
		for i := range os {
			sign := float64(1 - 2*ys[i])
			z := float64(xs[i] * sign)
			spz, spNegZ := sigmoidFocalSoftplusPairF64(z)
			mod := math.Exp(float64(spNegZ * c))
			term := float64(spz * mod)
			os[i] = float64(term * sigmoidFocalAlphaF64(ys[i], pa.Alpha))
		}
	default:
		return nil, fmt.Errorf("ref: sigmoid focal core unsupported dtype %v", logits.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func sigmoidFocalBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: sigmoid focal core backward wants (logits, targets, grad), got %d inputs", len(in))
	}
	logits, targets, grad := in[0], in[1], in[2]
	if logits.Dtype() != targets.Dtype() || logits.Dtype() != grad.Dtype() ||
		!logits.Shape().Equal(targets.Shape()) || !logits.Shape().Equal(grad.Shape()) {
		return nil, errors.New("ref: sigmoid focal core backward inputs must share dtype and shape")
	}
	pa, _ := attrs.(backend.SigmoidFocalAttrs)
	logits, targets, grad = logits.Contiguous(), targets.Contiguous(), grad.Contiguous()
	dLogits := tensor.NewOn(ctx.Device(), logits.Dtype(), logits.Shape())
	switch logits.Dtype() {
	case tensor.F32:
		xs, ys, gs, ds := logits.Storage().F32(), targets.Storage().F32(), grad.Storage().F32(), dLogits.Storage().F32()
		c := float32(-pa.Gamma)
		for i := range ds {
			sign := float32(1 - 2*ys[i])
			z := float32(xs[i] * sign)
			gTerm := float32(gs[i] * sigmoidFocalAlphaF32(ys[i], pa.Alpha))
			if pa.Gamma == 0 {
				gz := float32(float64(gTerm) * sigmoidFocalSigmoid(float64(z)))
				ds[i] = float32(gz * sign)
				continue
			}
			negZ := math.Float32frombits(math.Float32bits(z) ^ 1<<31)
			spz, spNegZ := sigmoidFocalSoftplusPairF32(z)
			mod := float32(math.Exp(float64(float32(spNegZ * c))))
			gu := float32(gTerm * mod)
			gm := float32(gTerm * spz)
			gq := float32(gm * mod)
			gv := float32(gq * c)
			gzU := float32(float64(gu) * sigmoidFocalSigmoid(float64(z)))
			gNegZ := float32(float64(gv) * sigmoidFocalSigmoid(float64(negZ)))
			gzV := math.Float32frombits(math.Float32bits(gNegZ) ^ 1<<31)
			gz := float32(gzV + gzU)
			ds[i] = float32(gz * sign)
		}
	case tensor.F64:
		xs, ys, gs, ds := logits.Storage().F64(), targets.Storage().F64(), grad.Storage().F64(), dLogits.Storage().F64()
		c := -pa.Gamma
		for i := range ds {
			sign := float64(1 - 2*ys[i])
			z := float64(xs[i] * sign)
			gTerm := float64(gs[i] * sigmoidFocalAlphaF64(ys[i], pa.Alpha))
			if pa.Gamma == 0 {
				gz := float64(gTerm * sigmoidFocalSigmoid(z))
				ds[i] = float64(gz * sign)
				continue
			}
			spz, spNegZ := sigmoidFocalSoftplusPairF64(z)
			mod := math.Exp(float64(spNegZ * c))
			gu := float64(gTerm * mod)
			gm := float64(gTerm * spz)
			gq := float64(gm * mod)
			gv := float64(gq * c)
			gzU := float64(gu * sigmoidFocalSigmoid(z))
			gNegZ := float64(gv * sigmoidFocalSigmoid(-z))
			gz := float64(-gNegZ + gzU)
			ds[i] = float64(gz * sign)
		}
	default:
		return nil, fmt.Errorf("ref: sigmoid focal core backward unsupported dtype %v", logits.Dtype())
	}
	return []*tensor.Tensor{dLogits}, nil
}

func sigmoidFocalSoftplus(x float64) float64 {
	if x > 0 {
		return x + math.Log1p(math.Exp(-x))
	}
	return math.Log1p(math.Exp(x))
}

func sigmoidFocalSigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func sigmoidFocalSoftplusPairF32(z float32) (float32, float32) {
	negZ := math.Float32frombits(math.Float32bits(z) ^ 1<<31)
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

func sigmoidFocalSoftplusPairF64(z float64) (float64, float64) {
	base := math.Log1p(math.Exp(-math.Abs(z)))
	if z > 0 {
		return z + base, base
	}
	if z < 0 {
		return base, -z + base
	}
	return base, base
}

func sigmoidFocalAlphaF32(y float32, alpha float64) float32 {
	if alpha < 0 {
		return 1
	}
	if y == 1 {
		return float32(alpha)
	}
	return float32(1 - alpha)
}

func sigmoidFocalAlphaF64(y, alpha float64) float64 {
	if alpha < 0 {
		return 1
	}
	if y == 1 {
		return alpha
	}
	return 1 - alpha
}

func init() {
	std.add(backend.OpSigmoidFocalCore, tensor.F32, sigmoidFocalKernel)
	std.add(backend.OpSigmoidFocalCore, tensor.F64, sigmoidFocalKernel)
	std.add(backend.OpSigmoidFocalCoreBackward, tensor.F32, sigmoidFocalBackwardKernel)
	std.add(backend.OpSigmoidFocalCoreBackward, tensor.F64, sigmoidFocalBackwardKernel)
}
