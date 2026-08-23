package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type preNormFFNGeometry struct {
	rows, dim, hidden int
}

func preNormFFNShape(in []*tensor.Tensor, backward bool) (preNormFFNGeometry, error) {
	want := 7
	if backward {
		want = 8
	}
	if len(in) != want {
		return preNormFFNGeometry{}, fmt.Errorf("ref: prenorm-ffn wants %d inputs, got %d", want, len(in))
	}
	x, gamma, beta, w1, b1, w2, b2 := in[0], in[1], in[2], in[3], in[4], in[5], in[6]
	if x.Ndim() != 2 {
		return preNormFFNGeometry{}, fmt.Errorf("ref: prenorm-ffn x must be rank-2, got %v", x.Shape())
	}
	rows, dim := x.Shape()[0], x.Shape()[1]
	if rows == 0 || dim == 0 {
		return preNormFFNGeometry{}, fmt.Errorf("ref: prenorm-ffn needs non-empty x, got %v", x.Shape())
	}
	if gamma.Ndim() != 1 || gamma.Shape()[0] != dim || beta.Ndim() != 1 || beta.Shape()[0] != dim {
		return preNormFFNGeometry{}, fmt.Errorf("ref: prenorm-ffn gamma/beta must be [%d], got %v/%v", dim, gamma.Shape(), beta.Shape())
	}
	if w1.Ndim() != 2 || w1.Shape()[0] != dim {
		return preNormFFNGeometry{}, fmt.Errorf("ref: prenorm-ffn w1 must be [%d,hidden], got %v", dim, w1.Shape())
	}
	hidden := w1.Shape()[1]
	if hidden == 0 || b1.Ndim() != 1 || b1.Shape()[0] != hidden || w2.Ndim() != 2 || w2.Shape()[0] != hidden || w2.Shape()[1] != dim || b2.Ndim() != 1 || b2.Shape()[0] != dim {
		return preNormFFNGeometry{}, fmt.Errorf("ref: prenorm-ffn hidden tensors mismatch dim=%d hidden=%d: b1=%v w2=%v b2=%v", dim, hidden, b1.Shape(), w2.Shape(), b2.Shape())
	}
	for _, t := range in {
		if t.Dtype() != x.Dtype() {
			return preNormFFNGeometry{}, fmt.Errorf("ref: prenorm-ffn inputs must share dtype %v", x.Dtype())
		}
	}
	if backward && !in[7].Shape().Equal(x.Shape()) {
		return preNormFFNGeometry{}, fmt.Errorf("ref: prenorm-ffn upstream %v must match x %v", in[7].Shape(), x.Shape())
	}
	return preNormFFNGeometry{rows: rows, dim: dim, hidden: hidden}, nil
}

func preNormFFNForwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	g, err := preNormFFNShape(in, false)
	if err != nil {
		return nil, err
	}
	pa, _ := attrs.(backend.NormAttrs)
	eps := pa.WithDefaults().Eps
	x, gamma, beta, w1, b1, w2, b2 := in[0], in[1], in[2], in[3], in[4], in[5], in[6]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	xs, _ := f64Data(x)
	gammas, _ := f64Data(gamma)
	betas, _ := f64Data(beta)
	w1s, _ := f64Data(w1)
	b1s, _ := f64Data(b1)
	w2s, _ := f64Data(w2)
	b2s, _ := f64Data(b2)
	outs, flush, _ := outF64(out)
	norm := make([]float64, g.dim)
	act := make([]float64, g.hidden)
	for r := range g.rows {
		rbase := r * g.dim
		var mean float64
		for j := range g.dim {
			mean += xs[rbase+j]
		}
		mean /= float64(g.dim)
		var variance float64
		for j := range g.dim {
			d := xs[rbase+j] - mean
			variance += d * d
		}
		inv := 1 / math.Sqrt(variance/float64(g.dim)+eps)
		for j := range g.dim {
			norm[j] = (xs[rbase+j]-mean)*inv*gammas[j] + betas[j]
		}
		for h := range g.hidden {
			z := b1s[h]
			//perfscan:ignore PS4008 reference oracle: preserve the scalar definition's accumulation order
			for j := range g.dim {
				//perfscan:ignore PS6010,PS1006 reference oracle: preserve definition order and matrix layout
				z += norm[j] * w1s[j*g.hidden+h]
			}
			act[h] = 0.5 * z * (1 + math.Erf(z/math.Sqrt2))
		}
		for j := range g.dim {
			v := b2s[j] + xs[rbase+j]
			//perfscan:ignore PS4008 reference oracle: preserve the scalar definition's accumulation order
			for h := range g.hidden {
				//perfscan:ignore PS6010,PS1006 reference oracle: preserve definition order and matrix layout
				v += act[h] * w2s[h*g.dim+j]
			}
			outs[rbase+j] = v
		}
	}
	flush()
	return []*tensor.Tensor{out}, nil
}

func preNormFFNBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	g, err := preNormFFNShape(in, true)
	if err != nil {
		return nil, err
	}
	pa, _ := attrs.(backend.NormAttrs)
	eps := pa.WithDefaults().Eps
	x, gamma, beta, w1, b1, w2, dOut := in[0], in[1], in[2], in[3], in[4], in[5], in[7]
	dx := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	dgamma := tensor.NewOn(ctx.Device(), x.Dtype(), gamma.Shape())
	dbeta := tensor.NewOn(ctx.Device(), x.Dtype(), beta.Shape())
	dw1 := tensor.NewOn(ctx.Device(), x.Dtype(), w1.Shape())
	db1 := tensor.NewOn(ctx.Device(), x.Dtype(), b1.Shape())
	dw2 := tensor.NewOn(ctx.Device(), x.Dtype(), w2.Shape())
	db2 := tensor.NewOn(ctx.Device(), x.Dtype(), in[6].Shape())
	xs, _ := f64Data(x)
	gammas, _ := f64Data(gamma)
	betas, _ := f64Data(beta)
	w1Values, _ := f64Data(w1)
	b1Values, _ := f64Data(b1)
	w2Values, _ := f64Data(w2)
	dOutValues, _ := f64Data(dOut)
	dxValues, flushDX, _ := outF64(dx)
	dgammaValues, flushDGamma, _ := outF64(dgamma)
	dbetaValues, flushDBeta, _ := outF64(dbeta)
	dw1Values, flushDW1, _ := outF64(dw1)
	db1Values, flushDB1, _ := outF64(db1)
	dw2GradValues, flushDW2, _ := outF64(dw2)
	db2Values, flushDB2, _ := outF64(db2)
	xhat := make([]float64, g.dim)
	norm := make([]float64, g.dim)
	z := make([]float64, g.hidden)
	act := make([]float64, g.hidden)
	dz := make([]float64, g.hidden)
	dnorm := make([]float64, g.dim)
	const invSqrt2Pi = 0.39894228040143267794
	for r := range g.rows {
		rbase := r * g.dim
		var mean float64
		for j := range g.dim {
			mean += xs[rbase+j]
		}
		mean /= float64(g.dim)
		var variance float64
		for j := range g.dim {
			d := xs[rbase+j] - mean
			variance += d * d
		}
		inv := 1 / math.Sqrt(variance/float64(g.dim)+eps)
		for j := range g.dim {
			xhat[j] = (xs[rbase+j] - mean) * inv
			norm[j] = xhat[j]*gammas[j] + betas[j]
		}
		for h := range g.hidden {
			v := b1Values[h]
			//perfscan:ignore PS4008 reference oracle: preserve the scalar definition's accumulation order
			for j := range g.dim {
				//perfscan:ignore PS6010,PS1006 reference oracle: preserve definition order and matrix layout
				v += norm[j] * w1Values[j*g.hidden+h]
			}
			z[h] = v
			cdf := 0.5 * (1 + math.Erf(v/math.Sqrt2))
			act[h] = v * cdf
			var dAct float64
			for j := range g.dim {
				do := dOutValues[rbase+j]
				dAct += do * w2Values[h*g.dim+j]
				dw2GradValues[h*g.dim+j] += act[h] * do
			}
			dz[h] = dAct * (cdf + v*math.Exp(-0.5*v*v)*invSqrt2Pi)
			db1Values[h] += dz[h]
		}
		for j := range g.dim {
			dnorm[j] = 0
			for h := range g.hidden {
				dnorm[j] += dz[h] * w1Values[j*g.hidden+h]
				dw1Values[j*g.hidden+h] += norm[j] * dz[h]
			}
			dgammaValues[j] += dnorm[j] * xhat[j]
			dbetaValues[j] += dnorm[j]
			db2Values[j] += dOutValues[rbase+j]
		}
		var meanA, meanAX float64
		for j := range g.dim {
			a := dnorm[j] * gammas[j]
			meanA += a
			meanAX += a * xhat[j]
		}
		meanA /= float64(g.dim)
		meanAX /= float64(g.dim)
		for j := range g.dim {
			a := dnorm[j] * gammas[j]
			dxValues[rbase+j] = dOutValues[rbase+j] + inv*(a-meanA-xhat[j]*meanAX)
		}
	}
	flushDX()
	flushDGamma()
	flushDBeta()
	flushDW1()
	flushDB1()
	flushDW2()
	flushDB2()
	return []*tensor.Tensor{dx, dgamma, dbeta, dw1, db1, dw2, db2}, nil
}

func init() {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		std.add(backend.OpPreNormFFN, dt, preNormFFNForwardKernel)
		std.add(backend.OpPreNormFFNBackward, dt, preNormFFNBackwardKernel)
	}
}
