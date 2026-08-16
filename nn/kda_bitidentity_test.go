package nn

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestKDAIsBitIdentical freezes the Kimi delta-attention output. Merging the four per-step
// passes over the state claims to change no value — each row's operations keep their order and
// operands, and only WHEN the row is visited moves — and an attention variant is somewhere a
// small drift would never be noticed.
//
// The shapes are chosen around the state size, because that is what the merge acts on: at
// dk=dv=64 the state is 32 KB and already L1-resident, and at 128 it is 131 KB and is not.
func TestKDAIsBitIdentical(t *testing.T) {
	cases := []struct {
		seq, dk, dv int
		dt          tensor.Dtype
		want        uint64
	}{
		{16, 8, 8, tensor.F64, archgold.Pick(4297825276869110848, 6446602709617325988)},
		{64, 64, 64, tensor.F64, archgold.Pick(1344577449931651826, 758658149952110689)},
		{48, 128, 128, tensor.F64, archgold.Pick(13733883409037120111, 6960860736063445236)},
		{48, 32, 32, tensor.F32, archgold.Pick(14432043542469650367, 14432043542469650367)},
	}
	for _, c := range cases {
		mk := func(fn func(i int) float64, r, cc int) *tensor.Tensor {
			tn := tensor.New(c.dt, tensor.Shape{r, cc})
			for i := range r * cc {
				tn.SetF64(fn(i), i/cc, i%cc)
			}
			return tn
		}
		q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) }, c.seq, c.dk)
		k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) }, c.seq, c.dk)
		v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) }, c.seq, c.dv)
		a := mk(func(i int) float64 { return 0.9 + 0.05*math.Sin(float64(i)) }, c.seq, c.dk)
		beta := mk(func(i int) float64 { return 0.5 + 0.25*math.Cos(float64(i)) }, c.seq, 1)
		out, err := KimiDeltaAttention(q, k, v, a, beta)
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		for i := range c.seq {
			for j := range c.dv {
				b := math.Float64bits(out.AtF64(i, j))
				for s := 0; s < 64; s += 8 {
					h = (h ^ (b>>s)&0xff) * 1099511628211
				}
			}
		}
		if h != c.want {
			t.Fatalf("%v %dx%dx%d digest %d, want %d", c.dt, c.seq, c.dk, c.dv, h, c.want)
		}
	}
}
