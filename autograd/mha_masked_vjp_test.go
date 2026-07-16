package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/tensor"
)

// TestMHAMaskedVJP gradchecks OpMHAMasked's VJP (dQ/dK/dV/dmask) against central
// finite differences of sum(out) — the correctness gate for T5-attention training.
func TestMHAMaskedVJP(t *testing.T) {
	const sq, sk, heads, dk = 3, 4, 2, 2
	const dm = heads * dk
	mk := func(shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape(shape))
		s := x.Storage().F64()
		for i := range s {
			s[i] = math.Sin(float64(i*7+3)) * 0.5
		}
		return x
	}
	q, k, v := mk(sq, dm), mk(sk, dm), mk(sk, dm)
	mask := mk(heads, sq, sk) // per-head additive bias
	attrs := backend.AttnAttrs{Heads: heads, Scale: 1.3}

	// analytic grads via the tape
	tape := autograd.NewTape()
	ctx := tape.Context()
	outs, err := backend.Execute(ctx, backend.OpMHAMasked, []*tensor.Tensor{q, k, v, mask}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(outs[0]); err != nil {
		t.Fatal(err)
	}

	sumOut := func(qq, kk, vv, mm *tensor.Tensor) float64 {
		o, _ := backend.Execute(backend.NewContext(), backend.OpMHAMasked, []*tensor.Tensor{qq, kk, vv, mm}, attrs)
		s := o[0].Contiguous().Storage().F64()
		var t float64
		for _, x := range s {
			t += x
		}
		return t
	}
	const eps = 1e-6
	check := func(name string, x *tensor.Tensor) {
		grad := tape.Grad(x)
		if grad == nil {
			t.Fatalf("%s: nil grad", name)
		}
		gs := grad.Contiguous().Storage().F64()
		xs := x.Storage().F64()
		var maxAbs float64
		for i := range xs {
			orig := xs[i]
			xs[i] = orig + eps
			lp := sumOut(q, k, v, mask)
			xs[i] = orig - eps
			lm := sumOut(q, k, v, mask)
			xs[i] = orig
			num := (lp - lm) / (2 * eps)
			if d := math.Abs(num - gs[i]); d > maxAbs {
				maxAbs = d
			}
		}
		t.Logf("%s: max |analytic − numeric| = %.2e", name, maxAbs)
		if maxAbs > 1e-5 {
			t.Fatalf("%s VJP wrong: %.2e", name, maxAbs)
		}
	}
	check("dQ", q)
	check("dK", k)
	check("dV", v)
	check("dmask", mask)
}
