package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: Apply prepends the p-token prefix to the seq keys/values, giving
// [p+seq, d] — the first p rows are the computed prefix P_K/P_V and the remaining
// seq rows are exactly the original keys/values (§R133).
func TestPrefixTuningApply(t *testing.T) {
	const p, d, seq = 2, 4, 3
	pt, err := nn.NewPrefixTuning(tensor.F64, p, d, 3, 5, 7)
	if err != nil {
		t.Fatal(err)
	}
	kd, vd := make([]float64, seq*d), make([]float64, seq*d)
	for i := range kd {
		kd[i] = float64(i) - 5
		vd[i] = float64(i) + 0.5
	}
	k := tensor.FromFloat64(tensor.Shape{seq, d}, kd)
	v := tensor.FromFloat64(tensor.Shape{seq, d}, vd)
	ctx := backend.NewContext()
	kp, vp, err := pt.Apply(ctx, k, v)
	if err != nil {
		t.Fatal(err)
	}
	if !kp.Shape().Equal(tensor.Shape{p + seq, d}) || !vp.Shape().Equal(tensor.Shape{p + seq, d}) {
		t.Fatalf("shapes kp%v vp%v", kp.Shape(), vp.Shape())
	}
	pk, pv, _ := pt.KV(ctx)
	if pt.PrefixLen() != p {
		t.Errorf("PrefixLen %d, want %d", pt.PrefixLen(), p)
	}
	for r := range p { // first p rows == the computed prefix
		for c := range d {
			if math.Abs(kp.AtF64(r, c)-pk.AtF64(r, c)) > 1e-12 || math.Abs(vp.AtF64(r, c)-pv.AtF64(r, c)) > 1e-12 {
				t.Errorf("prefix row %d col %d mismatch", r, c)
			}
		}
	}
	for r := range seq { // remaining rows == original k,v (unchanged)
		for c := range d {
			if kp.AtF64(p+r, c) != k.AtF64(r, c) || vp.AtF64(p+r, c) != v.AtF64(r, c) {
				t.Errorf("original row %d col %d changed", r, c)
			}
		}
	}
	if got := len(pt.Params()); got != 5 { // Pemb, Trans1.W/B, Trans2.W/B
		t.Errorf("Params len %d, want 5", got)
	}
}

// §V2: the end-to-end gradient of a loss over the prefixed K/V flows into the prefix
// parameters (P' and the reparameterization MLP), matching finite differences — so
// the prefix trains while the base stays frozen.
func TestPrefixTuningGradcheck(t *testing.T) {
	const p, d, seq, h, rtol = 2, 3, 2, 1e-6, 2e-3
	pt, _ := nn.NewPrefixTuning(tensor.F64, p, d, 2, 4, 3)
	k := tensor.FromFloat64(tensor.Shape{seq, d}, []float64{0.5, -1, 2, 0.3, -0.7, 1.1})
	v := tensor.FromFloat64(tensor.Shape{seq, d}, []float64{1, 0.2, -0.4, 0.8, -1.2, 0.6})

	loss := func(ctx *backend.Context) (*tensor.Tensor, error) {
		kp, vp, err := pt.Apply(ctx, k, v)
		if err != nil {
			return nil, err
		}
		sk, _ := mustOp(ctx, backend.OpSum, kp)
		sv, _ := mustOp(ctx, backend.OpSum, vp)
		return mustOp(ctx, backend.OpAdd, sk, sv)
	}
	tape := autograd.NewTape()
	l, err := loss(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		l, err := loss(backend.NewContext())
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	check := func(name string, w, grad *tensor.Tensor) {
		for i := range w.Numel() {
			idx := tensor.Unravel(i, w.Shape())
			o := w.AtF64(idx...)
			w.SetF64(o+h, idx...)
			lp := sumLoss()
			w.SetF64(o-h, idx...)
			lm := sumLoss()
			w.SetF64(o, idx...)
			if num, ana := (lp-lm)/(2*h), grad.AtF64(idx...); math.Abs(num-ana) > rtol*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%v]: numeric %.8g vs analytic %.8g", name, idx, num, ana)
			}
		}
	}
	check("Pemb", pt.Pemb, tape.Grad(pt.Pemb))
	check("Trans1.W", pt.Trans1.W, tape.Grad(pt.Trans1.W))
	check("Trans2.W", pt.Trans2.W, tape.Grad(pt.Trans2.W))
}

func TestPrefixTuningErrors(t *testing.T) {
	if _, err := nn.NewPrefixTuning(tensor.F64, 0, 4, 2, 4, 1); err == nil {
		t.Error("non-positive prefixLen must error")
	}
	pt, _ := nn.NewPrefixTuning(tensor.F64, 2, 4, 2, 4, 1)
	k := tensor.New(tensor.F64, tensor.Shape{3, 4})
	if _, _, err := pt.Apply(backend.NewContext(), k, tensor.New(tensor.F64, tensor.Shape{3, 5})); err == nil {
		t.Error("k/v shape mismatch must error")
	}
	if _, _, err := pt.Apply(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 5}), tensor.New(tensor.F64, tensor.Shape{3, 5})); err == nil {
		t.Error("wrong d_model must error")
	}
}

// Prefix tuning (Li & Liang 2021) prepends learned key/value vectors to an
// attention layer. Here a 2-token prefix extends a length-3, width-4 K/V to length 5.
func ExamplePrefixTuning() {
	pt, _ := nn.NewPrefixTuning(tensor.F64, 2, 4, 3, 8, 1)
	k := tensor.New(tensor.F64, tensor.Shape{3, 4})
	v := tensor.New(tensor.F64, tensor.Shape{3, 4})
	kp, _, _ := pt.Apply(backend.NewContext(), k, v)
	fmt.Println(kp.Shape())
	// Output: (5, 4)
}
