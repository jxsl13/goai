package autograd_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §R62: sliding-window attention restricts each causal query to the W most recent
// keys. With equal q·k scores and V a positional ramp, a query's output is the
// uniform mean over its attended window — so query 5 with window 3 averages only
// keys 3,4,5 (mean 4), while full causal averages 0..5 (mean 2.5).
func TestSlidingWindowMasksDistantKeys(t *testing.T) {
	seq, dk := 6, 2
	q := tensor.New(tensor.F64, tensor.Shape{seq, dk})
	k := tensor.New(tensor.F64, tensor.Shape{seq, dk})
	v := tensor.New(tensor.F64, tensor.Shape{seq, dk})
	for i := range seq {
		q.SetF64(1, i, 0) // all q·k equal → uniform scores
		k.SetF64(1, i, 0)
		v.SetF64(float64(i), i, 0) // ramp 0..5
	}
	run := func(window int) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpMHA,
			[]*tensor.Tensor{q, k, v},
			backend.AttnAttrs{Heads: 1, Causal: true, Window: window})
		if err != nil {
			t.Fatal(err)
		}
		return out[0].AtF64(seq-1, 0) // last query, first dim
	}
	if full := run(0); math.Abs(full-2.5) > 1e-12 { // mean(0..5)
		t.Errorf("full causal output = %.6f, want 2.5", full)
	}
	if win := run(3); math.Abs(win-4.0) > 1e-12 { // mean(3,4,5)
		t.Errorf("window-3 output = %.6f, want 4.0 (only last 3 keys)", win)
	}
	if win5 := run(5); math.Abs(win5-3.0) > 1e-12 { // mean(1..5)
		t.Errorf("window-5 output = %.6f, want 3.0", win5)
	}
}

// §R62: a window ≥ sequence length is a no-op — identical to full causal attention.
func TestSlidingWindowFullEqualsCausal(t *testing.T) {
	seq, dm := 5, 4
	data := func(s float64) []float64 {
		d := make([]float64, seq*dm)
		for i := range d {
			d[i] = math.Sin(s + float64(i)*0.6)
		}
		return d
	}
	q := tensor.FromFloat64(tensor.Shape{seq, dm}, data(1))
	k := tensor.FromFloat64(tensor.Shape{seq, dm}, data(2))
	v := tensor.FromFloat64(tensor.Shape{seq, dm}, data(3))
	run := func(window int) *tensor.Tensor {
		out, err := backend.Execute(backend.NewContext(), backend.OpMHA,
			[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: 2, Causal: true, Window: window})
		if err != nil {
			t.Fatal(err)
		}
		return out[0]
	}
	full, wide := run(0), run(seq) // window==seq covers all causal keys
	for i := range full.Numel() {
		idx := tensor.Unravel(i, full.Shape())
		if math.Abs(full.AtF64(idx...)-wide.AtF64(idx...)) > 1e-12 {
			t.Errorf("window≥seq must equal full causal at %v", idx)
		}
	}
}

// §V2: the fused MHA backward respects the window when it recomputes the softmax,
// so the gradient matches central finite differences of the windowed attention.
func TestSlidingWindowGradCheck(t *testing.T) {
	seq, dm, heads, window := 5, 4, 2, 2
	attrs := backend.AttnAttrs{Heads: heads, Causal: true, Window: window}
	data := func(s float64) []float64 {
		d := make([]float64, seq*dm)
		for i := range d {
			d[i] = math.Sin(s + float64(i)*0.7)
		}
		return d
	}
	qd, kd, vd := data(1), data(2), data(3)

	loss := func(qd, kd, vd []float64) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpMHA,
			[]*tensor.Tensor{
				tensor.FromFloat64(tensor.Shape{seq, dm}, qd),
				tensor.FromFloat64(tensor.Shape{seq, dm}, kd),
				tensor.FromFloat64(tensor.Shape{seq, dm}, vd),
			}, attrs)
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
	qT := tensor.FromFloat64(tensor.Shape{seq, dm}, qd)
	kT := tensor.FromFloat64(tensor.Shape{seq, dm}, kd)
	vT := tensor.FromFloat64(tensor.Shape{seq, dm}, vd)
	out, err := backend.Execute(tape.Context(), backend.OpMHA, []*tensor.Tensor{qT, kT, vT}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}

	const h = 1e-6
	check := func(name string, base []float64, which int, grad *tensor.Tensor) {
		work := append([]float64(nil), base...)
		for idx := range base {
			o := work[idx]
			work[idx] = o + h
			var lp float64
			switch which {
			case 0:
				lp = loss(work, kd, vd)
			case 1:
				lp = loss(qd, work, vd)
			default:
				lp = loss(qd, kd, work)
			}
			work[idx] = o - h
			var lm float64
			switch which {
			case 0:
				lm = loss(work, kd, vd)
			case 1:
				lm = loss(qd, work, vd)
			default:
				lm = loss(qd, kd, work)
			}
			work[idx] = o
			if num, ana := (lp-lm)/(2*h), grad.AtF64(tensor.Unravel(idx, grad.Shape())...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", name, idx, num, ana)
			}
		}
	}
	check("Q", qd, 0, tape.Grad(qT))
	check("K", kd, 1, tape.Grad(kT))
	check("V", vd, 2, tape.Grad(vT))
}

// Sliding-window attention (Mistral) limits each causal query to the W most
// recent keys. With equal scores and a positional-ramp V, the last of 6 tokens
// with window 3 averages only keys 3,4,5 (mean 4) instead of all six (mean 2.5).
func Example_slidingWindowAttention() {
	seq := 6
	q := tensor.New(tensor.F64, tensor.Shape{seq, 2})
	k := tensor.New(tensor.F64, tensor.Shape{seq, 2})
	v := tensor.New(tensor.F64, tensor.Shape{seq, 2})
	for i := range seq {
		q.SetF64(1, i, 0)
		k.SetF64(1, i, 0)
		v.SetF64(float64(i), i, 0)
	}
	out, _ := backend.Execute(backend.NewContext(), backend.OpMHA,
		[]*tensor.Tensor{q, k, v}, backend.AttnAttrs{Heads: 1, Causal: true, Window: 3})
	fmt.Printf("%.1f\n", out[0].AtF64(seq-1, 0))
	// Output: 4.0
}
