package nn

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// WKV's running maximum is a math.Max per token per channel, in the innermost position there
// is. Moving it onto the min/max builtins through internal/fmath claims to change no value —
// fmath restores math's infinity rule, which the builtins do not have — and the surrounding
// code DEPENDS on the exact result: it tests `q != pp` and `q != ww` to skip the exps that are
// provably one, so a max that returned a different bit pattern would silently take a different
// branch and evaluate a different number of exps.
//
// Two cases plant +Inf, -Inf and NaN in the same channel. WHAT THAT DOES AND DOES NOT PROVE
// is worth stating, because it is not what it looks like: it does NOT distinguish math.Max
// from the raw builtin here. WKV's recurrence carries the running maximum forward, so the one
// pairing the two disagree on — NaN against +Inf — turns the channel's state to NaN either
// way within the same step, and the outputs agree on NaN. A raw-builtin mutation of all six
// sites leaves every digest below unchanged.
//
// The rewrite therefore rests on internal/fmath's own exhaustive pair test for equivalence,
// and these digests are here to catch a MIS-EDIT: substituting Min for Max at a single site
// reddens three of the five cases, and the infinities make sure the branch-skipping equality
// tests around the max are exercised rather than only its ordinary path.
func wkvDigest(t *testing.T, seq, d int, dt tensor.Dtype, hostile bool) uint64 {
	t.Helper()
	mk := func(shape tensor.Shape, fn func(i int) float64) *tensor.Tensor {
		x := tensor.New(dt, shape)
		n := x.Numel()
		switch dt {
		case tensor.F64:
			s := x.Storage().F64()
			for i := range n {
				s[i] = fn(i)
			}
		case tensor.F32:
			s := x.Storage().F32()
			for i := range n {
				s[i] = float32(fn(i))
			}
		}
		return x
	}
	kf := func(i int) float64 {
		v := math.Sin(float64(i)*0.37) * 3
		if hostile {
			switch i % 17 {
			case 0:
				return math.Inf(1)
			case 5:
				return math.NaN()
			case 9:
				return math.Inf(-1)
			}
		}
		return v
	}
	k := mk(tensor.Shape{seq, d}, kf)
	v := mk(tensor.Shape{seq, d}, func(i int) float64 { return math.Cos(float64(i) * 0.21) })
	w := mk(tensor.Shape{d}, func(i int) float64 { return 0.5 + 0.01*float64(i%7) })
	u := mk(tensor.Shape{d}, func(i int) float64 { return -0.25 + 0.02*float64(i%5) })
	out, err := WKV(k, v, w, u)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	mix := func(b uint64) {
		for s := 0; s < 64; s += 8 {
			h = (h ^ (b>>s)&0xff) * 1099511628211
		}
	}
	n := out.Numel()
	switch dt {
	case tensor.F64:
		for _, x := range out.Storage().F64()[:n] {
			mix(math.Float64bits(x))
		}
	case tensor.F32:
		for _, x := range out.Storage().F32()[:n] {
			mix(uint64(math.Float32bits(x)))
		}
	}
	return h
}

func TestWKVIsBitIdentical(t *testing.T) {
	cases := []struct {
		seq, d  int
		dt      tensor.Dtype
		hostile bool
		want    uint64
	}{
		{16, 8, tensor.F64, false, archgold.Pick(8704977994969926434, 6850958381347604755)},
		{16, 8, tensor.F32, false, archgold.Pick(3639629632577827843, 3639629632577827843)},
		{33, 12, tensor.F64, false, archgold.Pick(5138939865296147482, 10587339890939798671)},
		{33, 12, tensor.F64, true, archgold.Pick(16919174266983207772, 1454813574854328081)}, // +Inf and NaN in the same channel
		{33, 12, tensor.F32, true, archgold.Pick(2897106097270140966, 3906677676722939046)},
	}
	for _, c := range cases {
		got := wkvDigest(t, c.seq, c.d, c.dt, c.hostile)
		if got != c.want {
			t.Errorf("seq=%d d=%d dt=%v hostile=%v: digest %d", c.seq, c.d, c.dt, c.hostile, got)
		}
	}
}
