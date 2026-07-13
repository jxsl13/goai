package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func wkvInputs() []*tensor.Tensor {
	const seq, d = 5, 3
	mk := func(seed int, shape tensor.Shape) *tensor.Tensor {
		x := tensor.New(tensor.F64, shape)
		for i := range x.Numel() {
			x.SetF64(math.Sin(float64(seed*100+i)*1.7)*0.8, tensor.Unravel(i, x.Shape())...)
		}
		return x
	}
	k := mk(1, tensor.Shape{seq, d})
	v := mk(2, tensor.Shape{seq, d})
	w := tensor.New(tensor.F64, tensor.Shape{d})
	u := mk(4, tensor.Shape{d})
	for c := range d {
		w.SetF64(0.3+0.4*float64(c), c) // decay must be > 0
	}
	return []*tensor.Tensor{k, v, w, u}
}

// §T516: the dispatched OpWKV is the same stabilized recurrence as the nn.WKV
// host utility — bit-identical outputs.
func TestWKVOpMatchesHostWKV(t *testing.T) {
	ins := wkvInputs()
	out, err := backend.Execute(backend.NewContext(), backend.OpWKV, ins, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := nn.WKV(ins[0], ins[1], ins[2], ins[3])
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Numel() {
		idx := tensor.Unravel(i, want.Shape())
		if out[0].AtF64(idx...) != want.AtF64(idx...) {
			t.Fatalf("diverges at %v", idx)
		}
	}
}

// §V2: central finite differences of Σ(wkv) w.r.t. k, v, w, u match the
// analytic O(T²) reverse VJP.
func TestWKVGradCheck(t *testing.T) {
	ins := wkvInputs()
	sumOut := func(xs []*tensor.Tensor) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpWKV, xs, nil)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out[0].Numel() {
			s += out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
		}
		return s
	}
	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), backend.OpWKV, ins, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	names := []string{"k", "v", "w", "u"}
	for n, in := range ins {
		grad := tape.Grad(in)
		if grad == nil {
			t.Fatalf("input %s received no gradient", names[n])
		}
		for i := range in.Numel() {
			idx := tensor.Unravel(i, in.Shape())
			o := in.AtF64(idx...)
			in.SetF64(o+h, idx...)
			lp := sumOut(ins)
			in.SetF64(o-h, idx...)
			lm := sumOut(ins)
			in.SetF64(o, idx...)
			num, ana := (lp-lm)/(2*h), grad.AtF64(idx...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", names[n], i, num, ana)
			}
		}
	}
}
