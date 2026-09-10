package nn

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/archgold"
	"github.com/jxsl13/goai/tensor"
)

// TestKANForwardIsBitIdentical freezes the layer's output. Banding the fused spline over the
// batch claims to change no value — a band never splits a row, so each row's accumulation over
// the input dimension keeps the same ascending order — and a KAN is a smooth approximator whose
// outputs would absorb a reassociation without any test noticing.
//
// The batch sizes straddle the fan-out gate: the 3-row case runs serially in both arms, the
// 96-row case bands, and 13 is deliberately not a multiple of the worker count so the last
// band is short.
func TestKANForwardIsBitIdentical(t *testing.T) {
	cases := []struct {
		name           string
		batch, in, out int
		want           uint64
	}{
		// F64 SIMD SiLU intentionally differs from scalar math.Exp; these exact goldens are per architecture and experiment, sourced from pinned native baseline evidence.
		{"3x5x7", 3, 5, 7, archgold.PickSIMD(5936029728971432568, 14272068029666688409, 17265271475585544907, 16662584054408946177)},
		{"13x8x6", 13, 8, 6, archgold.PickSIMD(15159748691548848689, 6609257596807823200, 5035091549113534389, 3106186755478235033)},
		{"96x24x32", 96, 24, 32, archgold.PickSIMD(515177776064738749, 9025949438388873583, 12048638696559957597, 611919531391070369)},
	}
	cpuBE, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend not registered")
	}
	t.Logf("compiler=%s os=%s arch=%s backend=cpu", runtime.Version(), runtime.GOOS, runtime.GOARCH)

	// Independent subtests retain every fixture failure for feature and architecture qualification.
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			l, err := NewKAN(c.in, c.out, 1)
			if err != nil {
				t.Fatal(err)
			}
			x := tensor.New(tensor.F64, tensor.Shape{c.batch, c.in})
			xs := x.Storage().F64()
			for i := range xs {
				xs[i] = math.Sin(float64(i*11+5)) * 0.6
			}
			y, err := l.Forward(backend.NewContext().WithBackend(cpuBE), x)
			if err != nil {
				t.Fatal(err)
			}
			h := uint64(14695981039346656037)
			for _, v := range y.Storage().F64() {
				b := math.Float64bits(v)
				for s := 0; s < 64; s += 8 {
					h = (h ^ (b>>s)&0xff) * 1099511628211
				}
			}
			if h != c.want {
				t.Fatalf("%dx%dx%d digest %d, want %d", c.batch, c.in, c.out, h, c.want)
			}
		})
	}
}
