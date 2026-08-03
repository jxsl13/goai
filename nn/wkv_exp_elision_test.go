package nn_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestWKVExpElisionIsBitIdentical locks the exp(0) elision in nn.WKV to the arithmetic it replaced.
// q is the max of its two arguments, so whichever argument it equals gives exp(x-x), and math.Exp
// of an exact zero returns exactly 1 — the elision substitutes a value the call would have produced,
// not an approximation of it. The reference below is the unelided recurrence written out in full,
// and the comparison is on raw bits, both dtypes.
//
// Two regimes are covered because they exercise opposite sides of each max: strongly negative decay
// makes the carried state's exponent lose, and near-zero decay makes it win. A fixture that only
// ever took one branch would leave half the elision untested.
func TestWKVExpElisionIsBitIdentical(t *testing.T) {
	const seq, d = 37, 53
	for _, decay := range []float64{-2.5, -0.02} {
		rng := rand.New(rand.NewSource(7))
		kd := make([]float64, seq*d)
		vd := make([]float64, seq*d)
		wd := make([]float64, d)
		ud := make([]float64, d)
		for i := range kd {
			kd[i] = rng.NormFloat64() * 2
			vd[i] = rng.NormFloat64()
		}
		for i := range wd {
			wd[i] = decay - 0.1*rng.Float64()
			ud[i] = rng.NormFloat64()
		}

		// The reference: the recurrence as it was written before the elision.
		want := make([]float64, seq*d)
		for c := range d {
			wc, uc := wd[c], ud[c]
			aa, bb, pp := 0.0, 0.0, -1e38
			for tt := range seq {
				o := tt*d + c
				kk, vv := kd[o], vd[o]
				ww := uc + kk
				q := math.Max(pp, ww)
				e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
				want[o] = (e1*aa + e2*vv) / (e1*bb + e2)
				q = math.Max(pp-wc, kk)
				e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
				aa = e1*aa + e2*vv
				bb = e1*bb + e2
				pp = q
			}
		}

		mk := func(data []float64, shape ...int) *tensor.Tensor {
			x := tensor.New(tensor.F64, tensor.Shape(shape))
			copy(x.Storage().F64(), data)
			return x
		}
		got, err := nn.WKV(mk(kd, seq, d), mk(vd, seq, d), mk(wd, d), mk(ud, d))
		if err != nil {
			t.Fatal(err)
		}
		gs := got.Storage().F64()
		for i := range want {
			if math.Float64bits(gs[i]) != math.Float64bits(want[i]) {
				t.Fatalf("decay %v, f64 element %d: elided %v, unelided %v — not bit-identical",
					decay, i, gs[i], want[i])
			}
		}

		// F32 carries its running state in F64 and rounds only on the store, so the same reference
		// applies with a conversion at the end.
		mk32 := func(data []float64, shape ...int) *tensor.Tensor {
			x := tensor.New(tensor.F32, tensor.Shape(shape))
			s := x.Storage().F32()
			for i := range data {
				s[i] = float32(data[i])
			}
			return x
		}
		k32, v32 := mk32(kd, seq, d), mk32(vd, seq, d)
		w32, u32 := mk32(wd, d), mk32(ud, d)
		got32, err := nn.WKV(k32, v32, w32, u32)
		if err != nil {
			t.Fatal(err)
		}
		ks, vs := k32.Storage().F32(), v32.Storage().F32()
		wsx, usx := w32.Storage().F32(), u32.Storage().F32()
		gs32 := got32.Storage().F32()
		for c := range d {
			wc, uc := float64(wsx[c]), float64(usx[c])
			aa, bb, pp := 0.0, 0.0, -1e38
			for tt := range seq {
				o := tt*d + c
				kk, vv := float64(ks[o]), float64(vs[o])
				ww := uc + kk
				q := math.Max(pp, ww)
				e1, e2 := math.Exp(pp-q), math.Exp(ww-q)
				w32want := float32((e1*aa + e2*vv) / (e1*bb + e2))
				if math.Float32bits(gs32[o]) != math.Float32bits(w32want) {
					t.Fatalf("decay %v, f32 element %d: elided %v, unelided %v — not bit-identical",
						decay, o, gs32[o], w32want)
				}
				q = math.Max(pp-wc, kk)
				e1, e2 = math.Exp(pp-wc-q), math.Exp(kk-q)
				aa = e1*aa + e2*vv
				bb = e1*bb + e2
				pp = q
			}
		}
	}
}
