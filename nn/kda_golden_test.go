package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// kdaRef transcribes the unoptimized recurrence to pin the arithmetic the shipped version
// must reproduce EXACTLY. Every transform applied to KimiDeltaAttention is order-preserving
// per accumulator — loop interchange over a pure elementwise scale, and 4-way blocking of
// output loops whose accumulators still walk dk ascending — so raw-bit equality is the
// right assertion, not a tolerance.
func kdaRef(q, k, v, a, beta *tensor.Tensor) *tensor.Tensor {
	seq, dk := q.Shape()[0], q.Shape()[1]
	dv := v.Shape()[1]
	out := tensor.New(q.Dtype(), tensor.Shape{seq, dv})
	S := make([]float64, dv*dk)
	sk := make([]float64, dv)
	qt := make([]float64, dk)
	kt := make([]float64, dk)
	for t := range seq {
		bt := beta.AtF64(t, 0)
		var qn, kn float64
		for c := range dk {
			qv, kv := q.AtF64(t, c), k.AtF64(t, c)
			qt[c], kt[c] = qv, kv
			qn += qv * qv
			kn += kv * kv
		}
		if qn > 0 {
			qn = 1 / math.Sqrt(qn)
			for c := range dk {
				qt[c] *= qn
			}
		}
		if kn > 0 {
			kn = 1 / math.Sqrt(kn)
			for c := range dk {
				kt[c] *= kn
			}
		}
		for c := range dk {
			ac := a.AtF64(t, c)
			for r := range dv {
				S[r*dk+c] *= ac
			}
		}
		for r := range dv {
			var s float64
			for c := range dk {
				s += S[r*dk+c] * kt[c]
			}
			sk[r] = s
		}
		for r := range dv {
			delta := bt * (v.AtF64(t, r) - sk[r])
			for c := range dk {
				S[r*dk+c] += delta * kt[c]
			}
		}
		for r := range dv {
			var o float64
			for c := range dk {
				o += S[r*dk+c] * qt[c]
			}
			out.SetF64(o, t, r)
		}
	}
	return out
}

// TestKDABitIdenticalToReference sweeps dv and dk across every remainder class of a 4-way
// unroll, including dimensions BELOW 4 where the blocked loop never runs and only the tail
// executes — the case a size divisible by 4 would never reach.
func TestKDABitIdenticalToReference(t *testing.T) {
	for _, sz := range [][3]int{{5, 4, 4}, {6, 5, 7}, {7, 6, 6}, {4, 7, 5}, {9, 8, 8}, {3, 1, 1}, {5, 2, 3}, {6, 3, 2}, {8, 9, 13}} {
		seq, dk, dv := sz[0], sz[1], sz[2]
		mk := func(seed float64, cols int) *tensor.Tensor {
			x := tensor.New(tensor.F64, tensor.Shape{seq, cols})
			for i := range seq {
				for j := range cols {
					x.SetF64(math.Sin(seed+1.3*float64(i*cols+j)), i, j)
				}
			}
			return x
		}
		q, k, v := mk(1, dk), mk(2, dk), mk(3, dv)
		a := tensor.New(tensor.F64, tensor.Shape{seq, dk})
		beta := tensor.New(tensor.F64, tensor.Shape{seq, 1})
		for i := range seq {
			beta.SetF64(0.3+0.4*math.Abs(math.Cos(float64(i))), i, 0)
			for c := range dk {
				a.SetF64(0.4+0.5*math.Abs(math.Sin(float64(i)*1.7+float64(c))), i, c)
			}
		}
		got, err := nn.KimiDeltaAttention(q, k, v, a, beta)
		if err != nil {
			t.Fatalf("seq=%d dk=%d dv=%d: %v", seq, dk, dv, err)
		}
		want := kdaRef(q, k, v, a, beta)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, w := got.AtF64(idx...), want.AtF64(idx...)
			if math.Float64bits(g) != math.Float64bits(w) {
				t.Fatalf("seq=%d dk=%d dv=%d at %v: got %g want %g (not bit-identical)", seq, dk, dv, idx, g, w)
			}
		}
	}
}
