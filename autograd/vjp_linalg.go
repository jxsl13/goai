package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Linear-algebra VJPs (§T14). MatMul composes backend ops (transpose views cost
// nothing; the optimized cpu GEMM does the work). dot/nrm2/axpy are small scalar
// loops.

func init() {
	// C = A·B → gA = g·Bᵀ, gB = Aᵀ·g
	RegisterVJP(backend.OpMatMul, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		a, b := in[0], in[1]
		bt, err := b.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		ga, err := exec1(ctx, backend.OpMatMul, g, bt)
		if err != nil {
			return nil, err
		}
		at, err := a.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		gb, err := exec1(ctx, backend.OpMatMul, at, g)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{ga, gb}, nil
	})

	// s = Σaᵢbᵢ → (g·b, g·a) with scalar g
	RegisterVJP(backend.OpDot, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		a, b := in[0], in[1]
		gv := g.AtF64()
		ga := tensor.New(a.Dtype(), a.Shape())
		gb := tensor.New(b.Dtype(), b.Shape())
		for i := range a.Numel() {
			idx := tensor.Unravel(i, a.Shape())
			ga.SetF64(gv*b.AtF64(idx...), idx...)
			gb.SetF64(gv*a.AtF64(idx...), idx...)
		}
		return []*tensor.Tensor{ga, gb}, nil
	})

	// n = ‖x‖ → g·x/n ; subgradient 0 at x = 0
	RegisterVJP(backend.OpNrm2, func(_ *backend.Context, in, out []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		norm := out[0].AtF64()
		gv := g.AtF64()
		gin := tensor.New(x.Dtype(), x.Shape())
		if norm != 0 {
			for i := range x.Numel() {
				idx := tensor.Unravel(i, x.Shape())
				gin.SetF64(gv*x.AtF64(idx...)/norm, idx...)
			}
		}
		return []*tensor.Tensor{gin}, nil
	})

	// y = x + b (row-broadcast) → (g, Σᵢ g[i,·])
	RegisterVJP(backend.OpAddBias, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		b := in[1]
		m, n := g.Shape()[0], g.Shape()[1]
		gb := tensor.New(b.Dtype(), b.Shape())
		for j := range n {
			var s float64 // f64 accumulation (§V10)
			for i := range m {
				s += g.AtF64(i, j)
			}
			gb.SetF64(s, j)
		}
		return []*tensor.Tensor{g, gb}, nil
	})

	// CE: ∂L/∂z = g·(softmax(z) − onehot(t))/b, targets non-diff (ADR-0007)
	RegisterVJP(backend.OpCrossEntropy, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		z, tg := in[0], in[1]
		b, c := z.Shape()[0], z.Shape()[1]
		gv := g.AtF64()
		gz := tensor.New(z.Dtype(), z.Shape())
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
			for j := range c {
				p := math.Exp(z.AtF64(i, j)-m) / sum // stable softmax
				if j == ti {
					p -= 1
				}
				gz.SetF64(gv*p/float64(b), i, j)
			}
		}
		return []*tensor.Tensor{gz, nil}, nil
	})

	// z = αx + y → (α·g, g)
	RegisterVJP(backend.OpAXPY, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		alpha := attrs.Float("alpha", 1)
		x := in[0]
		gx := tensor.New(x.Dtype(), x.Shape())
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			gx.SetF64(alpha*g.AtF64(idx...), idx...)
		}
		return []*tensor.Tensor{gx, g}, nil
	})
}
