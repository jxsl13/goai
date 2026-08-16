package autograd

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The MLA backward accumulates the query gradient into a slot that does not vary with the
// attended key, so it can be jammed over keys — and the claim is that every dqC[i][d] still
// takes the same products in the same ascending key order. Attention gradients are checked
// elsewhere against finite differences, which is a tolerance and would absorb a reassociation
// without a word; this freezes the bits.
//
// Every returned gradient is hashed, not just the query one: the same loop writes dkC and
// dvC, and a jam that got the tail wrong would leave one of those short.
func mlaVJPDigest(t *testing.T, seq, heads, dh, dR int, causal bool, dt tensor.Dtype) uint64 {
	t.Helper()
	vjp := vjps[backend.OpMLA]
	if vjp == nil {
		t.Fatal("no VJP registered for OpMLA")
	}
	mk := func(shape tensor.Shape, seed int) *tensor.Tensor {
		x := tensor.New(dt, shape)
		n := x.Numel()
		f := func(i int) float64 { return math.Sin(float64(i*7+seed*13)) * 0.4 }
		if dt == tensor.F64 {
			s := x.Storage().F64()
			for i := range n {
				s[i] = f(i)
			}
		} else {
			s := x.Storage().F32()
			for i := range n {
				s[i] = float32(f(i))
			}
		}
		return x
	}
	hdh := heads * dh
	in := []*tensor.Tensor{
		mk(tensor.Shape{seq, hdh}, 1), mk(tensor.Shape{seq, hdh}, 2), mk(tensor.Shape{seq, hdh}, 3),
		mk(tensor.Shape{seq, heads * dR}, 4), mk(tensor.Shape{seq, dR}, 5),
	}
	g := mk(tensor.Shape{seq, hdh}, 6)
	gs, err := vjp(nil, in, nil, backend.MLAAttrs{Heads: heads, Causal: causal, RoPEBase: 10000}, g)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	mix := func(u uint64) {
		for s := 0; s < 64; s += 8 {
			h = (h ^ (u>>s)&0xff) * 1099511628211
		}
	}
	for _, grad := range gs {
		if grad == nil {
			mix(0xdead)
			continue
		}
		n := grad.Numel()
		if grad.Dtype() == tensor.F64 {
			for _, v := range grad.Storage().F64()[:n] {
				mix(math.Float64bits(v))
			}
			continue
		}
		for _, v := range grad.Storage().F32()[:n] {
			mix(uint64(math.Float32bits(v)))
		}
	}
	return h
}

func TestMLAVJPIsBitIdentical(t *testing.T) {
	// Sequence lengths straddle every remainder a jam of 2, 4 or 8 can leave. Under a CAUSAL
	// mask the key count is i+1 and runs over every value from 1 to seq, so one causal case
	// exercises all of them at once — including the lengths where the jammed loop never runs.
	cases := []struct {
		seq, heads, dh, dR int
		causal             bool
		dt                 tensor.Dtype
		want               uint64
	}{
		{13, 2, 8, 4, true, tensor.F64, archgold.Pick(5259501665213688223, 2081554234887433254)},
		{13, 2, 8, 4, false, tensor.F64, archgold.Pick(12131797958127341408, 14843990582745093933)},
		{22, 3, 6, 2, true, tensor.F64, archgold.Pick(14812911774429625573, 16287650737661751141)},
		{16, 2, 8, 4, false, tensor.F64, archgold.Pick(5559681268579107254, 3739404349636064010)},
		{13, 2, 8, 4, true, tensor.F32, archgold.Pick(14442385213026910017, 14442385213026910017)},
	}
	for _, c := range cases {
		got := mlaVJPDigest(t, c.seq, c.heads, c.dh, c.dR, c.causal, c.dt)
		if got != c.want {
			t.Errorf("seq=%d heads=%d dh=%d dR=%d causal=%v %v: digest %d",
				c.seq, c.heads, c.dh, c.dR, c.causal, c.dt, got)
		}
	}
}
