package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestUnscaleGradsParallel verifies the parallelized MixedPrecision.UnscaleGrads (parStep-threshold
// fans the unscale + inf/nan count across cores for large grads) produces BIT-IDENTICAL unscaled output
// AND the same foundInf as the serial path — F32/F64 grads, with and without an injected inf. The
// unscale is disjoint per-element (bit-exact) and the inf-check is an order-independent count.
func TestUnscaleGradsParallel(t *testing.T) {
	const n = 1<<18 + 5555
	orig := parStepMinElems
	defer func() { parStepMinElems = orig }()
	W := tensor.New(tensor.F32, tensor.Shape{n})
	mp := NewMixedPrecision([]*tensor.Tensor{W}, tensor.F16)
	const scale = 65536.0

	for _, tc := range []struct {
		name   string
		dt     tensor.Dtype
		injInf bool
	}{
		{"F32-clean", tensor.F32, false},
		{"F64-clean", tensor.F64, false},
		{"F32-inf", tensor.F32, true},
		{"F64-inf", tensor.F64, true},
	} {
		g := tensor.New(tc.dt, tensor.Shape{n})
		if tc.dt == tensor.F32 {
			gf := g.Storage().F32()
			for i := range gf {
				gf[i] = float32(math.Sin(float64(i)*0.0011) * 1000)
			}
			if tc.injInf {
				gf[123456] = float32(math.Inf(1))
			}
		} else {
			gf := g.Storage().F64()
			for i := range gf {
				gf[i] = math.Sin(float64(i)*0.0011) * 1000
			}
			if tc.injInf {
				gf[123456] = math.Inf(1)
			}
		}
		run := func(threshold int) ([]float32, bool) {
			parStepMinElems = threshold
			var found bool
			out := mp.UnscaleGrads(func(*tensor.Tensor) *tensor.Tensor { return g }, scale, &found)(mp.Masters[0])
			return append([]float32(nil), out.Storage().F32()...), found
		}
		so, sf := run(1 << 40)
		po, pf := run(1 << 4)
		if sf != pf {
			t.Fatalf("%s: foundInf serial=%v parallel=%v", tc.name, sf, pf)
		}
		if sf != tc.injInf {
			t.Fatalf("%s: foundInf=%v, want %v", tc.name, sf, tc.injInf)
		}
		for i := range so {
			if so[i] != po[i] {
				t.Fatalf("%s idx %d: serial %v != parallel %v (bit)", tc.name, i, so[i], po[i])
			}
		}
		t.Logf("%s: parallel UnscaleGrads bit-identical to serial (foundInf=%v)", tc.name, sf)
	}
}

func benchUnscale(b *testing.B, threshold int) {
	orig := parStepMinElems
	parStepMinElems = threshold
	defer func() { parStepMinElems = orig }()
	const n = 8192 * 4096 // 33.5M
	W := tensor.New(tensor.F32, tensor.Shape{n})
	mp := NewMixedPrecision([]*tensor.Tensor{W}, tensor.F16)
	g := tensor.New(tensor.F32, tensor.Shape{n})
	gf := g.Storage().F32()
	for i := range gf {
		gf[i] = float32(i%17) * 1e-2
	}
	master := mp.Masters[0]
	gfn := func(*tensor.Tensor) *tensor.Tensor { return g }
	b.ResetTimer()
	for b.Loop() {
		var found bool
		_ = mp.UnscaleGrads(gfn, 65536, &found)(master)
	}
}

func BenchmarkUnscaleGradsSerial(b *testing.B)   { benchUnscale(b, 1<<40) }
func BenchmarkUnscaleGradsParallel(b *testing.B) { benchUnscale(b, 1<<4) }
