package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §R60: ALiBi's linear bias makes a causal query attend MORE to recent keys. With
// equal q·k scores everywhere and V a positional ramp, plain attention is the
// uniform mean over the causal window; ALiBi shifts the weight toward the latest
// (highest-index) keys, so the last query's output rises above the uniform mean.
func TestALiBiForwardPrefersRecent(t *testing.T) {
	seq, dk := 4, 2
	q := tensor.New(tensor.F64, tensor.Shape{seq, dk})
	k := tensor.New(tensor.F64, tensor.Shape{seq, dk})
	v := tensor.New(tensor.F64, tensor.Shape{seq, dk})
	for i := range seq {
		q.SetF64(1, i, 0) // all q·k equal → uniform base scores
		k.SetF64(1, i, 0)
		v.SetF64(float64(i), i, 0) // V is a ramp: 0,1,2,3
	}
	run := func(alibi bool) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpMHA,
			[]*tensor.Tensor{q, k, v},
			backend.AttnAttrs{Heads: 1, Causal: true, ALiBi: alibi})
		if err != nil {
			t.Fatal(err)
		}
		return out[0].AtF64(seq-1, 0) // last query, first dim
	}
	plain := run(false) // uniform mean of 0..3 = 1.5
	biased := run(true) // weighted toward recent (higher V) ⇒ > 1.5
	if math.Abs(plain-1.5) > 1e-12 {
		t.Errorf("plain attention should be the uniform mean 1.5, got %.6f", plain)
	}
	if biased <= plain {
		t.Errorf("ALiBi must shift weight to recent keys: %.6f (alibi) vs %.6f (plain)", biased, plain)
	}
}

// §V2: the fused MHA backward re-adds the ALiBi bias when recomputing the softmax
// weights, so the analytic gradient matches central finite differences of the
// biased attention (a wrong/missing bias in the VJP would fail here).
func TestALiBiGradCheck(t *testing.T) {
	seq, dm, heads := 3, 4, 2
	attrs := backend.AttnAttrs{Heads: heads, Causal: true, ALiBi: true}
	data := func(seed float64) []float64 {
		d := make([]float64, seq*dm)
		for i := range d {
			d[i] = math.Sin(seed + float64(i)*0.7) // deterministic, varied
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
		return s // loss = Σ output
	}

	// analytic grads via the tape
	tape := autograd.NewTape()
	qT := tensor.FromFloat64(tensor.Shape{seq, dm}, qd)
	kT := tensor.FromFloat64(tensor.Shape{seq, dm}, kd)
	vT := tensor.FromFloat64(tensor.Shape{seq, dm}, vd)
	out, err := backend.Execute(tape.Context(), backend.OpMHA, []*tensor.Tensor{qT, kT, vT}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil { // seed = ones ⇒ d(Σout)
		t.Fatal(err)
	}

	const h = 1e-6
	checkOne := func(name string, base []float64, which int, grad *tensor.Tensor) {
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
			num := (lp - lm) / (2 * h)
			ana := grad.AtF64(tensor.Unravel(idx, grad.Shape())...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", name, idx, num, ana)
			}
		}
	}
	checkOne("Q", qd, 0, tape.Grad(qT))
	checkOne("K", kd, 1, tape.Grad(kT))
	checkOne("V", vd, 2, tape.Grad(vT))
}
