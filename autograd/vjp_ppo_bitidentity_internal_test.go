package autograd

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func ppoInputs(n int, dt tensor.Dtype) []*tensor.Tensor {
	mk := func(off float64) *tensor.Tensor {
		t := tensor.New(dt, tensor.Shape{n})
		for i := range n {
			// A spread wide enough that the ratio lands on BOTH sides of each clip bound and on
			// both sides of the surr1/surr2 comparison; a narrow one exercises one branch only.
			t.SetF64(math.Sin(float64(i)*0.31+off)*0.9, i)
		}
		return t
	}
	return []*tensor.Tensor{mk(0), mk(1.3), mk(2.7)}
}

func ppoDigest(t *testing.T, n int, dt tensor.Dtype, eps float64) uint64 {
	t.Helper()
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.Storage().F64()[0] = 1.5
	out, err := vjps[backend.OpPPOClip](nil, ppoInputs(n, dt), nil,
		backend.PPOClipAttrs{Epsilon: eps}, g)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	for i := range n {
		u := math.Float64bits(out[0].AtF64(i))
		for s := 0; s < 64; s += 8 {
			h = (h ^ (u>>s)&0xff) * 1099511628211
		}
	}
	return h
}

// TestPPOVJPIsBitIdentical freezes the rule's gradient. Taking the typed storage path and
// rewriting the clamp as a comparison chain both claim to change no value, and this rule had NO
// test of any kind before, so nothing would have noticed either way.
func TestPPOVJPIsBitIdentical(t *testing.T) {
	for _, c := range []struct {
		n    int
		eps  float64
		want uint64
	}{
		{1000, 0.2, archgold.Pick(5299779950205309273, 1141810445622132973)},
		{257, 0.05, archgold.Pick(2663088398810480165, 6111902220461594643)},
		{64, 1, archgold.Pick(17185744957343320499, 2411941273833116206)}, // eps=1 puts the low bound at exactly zero, where Max(0,-0) differs from <
	} {
		got := ppoDigest(t, c.n, tensor.F64, c.eps)
		if got != c.want {
			t.Fatalf("n=%d eps=%g digest %d, want %d", c.n, c.eps, got, c.want)
		}
	}
}

// TestPPOVJPArmsAgree pins the two arms against each other. The typed path runs only when every
// operand is already F64; an F32 input takes the accessor path.
//
// THEY CANNOT BE COMPARED AS EQUAL BITS, and asserting that was the first mistake here: the
// output tensor takes the dtype of its input, so the accessor arm stores float32 where the typed
// arm stores float64. What must hold is that they are the SAME float64 computation with one
// rounding at the end — so the accessor result must equal the typed result rounded once.
func TestPPOVJPArmsAgree(t *testing.T) {
	const n = 512
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.Storage().F64()[0] = 1.5
	f64, err := vjps[backend.OpPPOClip](nil, ppoInputs(n, tensor.F64), nil,
		backend.PPOClipAttrs{Epsilon: 0.2}, g)
	if err != nil {
		t.Fatal(err)
	}
	// F32 inputs round on construction, so compare against a F64 set built from those same
	// rounded values: the arms must agree on identical inputs, not on differently-rounded ones.
	in32 := ppoInputs(n, tensor.F32)
	in64 := make([]*tensor.Tensor, 3)
	for k, s := range in32 {
		t64 := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			t64.SetF64(s.AtF64(i), i)
		}
		in64[k] = t64
	}
	viaTyped, err := vjps[backend.OpPPOClip](nil, in64, nil, backend.PPOClipAttrs{Epsilon: 0.2}, g)
	if err != nil {
		t.Fatal(err)
	}
	viaAccessor, err := vjps[backend.OpPPOClip](nil, in32, nil, backend.PPOClipAttrs{Epsilon: 0.2}, g)
	if err != nil {
		t.Fatal(err)
	}
	_ = f64
	for i := range n {
		want := float32(viaTyped[0].AtF64(i))
		got := float32(viaAccessor[0].AtF64(i))
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("idx=%d: accessor %v, typed rounded %v", i, got, want)
		}
	}
}

// TestPPOVJPClipBoundaries pins the branch boundaries against the formula the rule is written
// from. A digest over ordinary inputs cannot: the ratio is math.Exp of a difference, so it is
// always finite and positive — the negative-zero and NaN cases the clamp rewrite guards against
// never arise here — and sine-generated data never lands EXACTLY on a clip bound, which is the
// only place the trust-region comparison changes meaning. These inputs are constructed to sit on
// the bounds, just inside and just outside.
func TestPPOVJPClipBoundaries(t *testing.T) {
	for _, eps := range []float64{0.2, 0.05, 1} {
		lo, hi := 1-eps, 1+eps
		ratios := []float64{
			lo, hi, // exactly on each bound
			math.Nextafter(lo, 0), math.Nextafter(lo, 2), // straddling the low bound
			math.Nextafter(hi, 0), math.Nextafter(hi, 2), // straddling the high bound
			1, lo / 2, hi * 2,
		}
		n := len(ratios) * 2 // each ratio with a positive and a negative advantage
		lpNew := tensor.New(tensor.F64, tensor.Shape{n})
		lpOld := tensor.New(tensor.F64, tensor.Shape{n})
		adv := tensor.New(tensor.F64, tensor.Shape{n})
		for i, rr := range ratios {
			for s := range 2 {
				k := i*2 + s
				lpNew.SetF64(math.Log(rr), k) // lpOld stays 0, so r == exp(log(rr)) == rr
				adv.SetF64(map[int]float64{0: 0.75, 1: -0.75}[s], k)
			}
		}
		g := tensor.New(tensor.F64, tensor.Shape{})
		g.Storage().F64()[0] = 1.5
		out, err := vjps[backend.OpPPOClip](nil,
			[]*tensor.Tensor{lpNew, lpOld, adv}, nil, backend.PPOClipAttrs{Epsilon: eps}, g)
		if err != nil {
			t.Fatal(err)
		}
		scale := -1.5 / float64(n)
		for k := range n {
			r := math.Exp(lpNew.AtF64(k) - lpOld.AtF64(k))
			a := adv.AtF64(k)
			// The rule as originally written, kept verbatim as the oracle.
			want := 0.0
			if r*a <= math.Max(lo, math.Min(hi, r))*a {
				want = a * r
			} else if r > lo && r < hi {
				want = a * r
			}
			want *= scale
			if got := out[0].AtF64(k); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("eps=%g idx=%d r=%v a=%v: got %v, want %v", eps, k, r, a, got, want)
			}
		}
	}
}
