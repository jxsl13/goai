//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func softplusF64Arm64ScalarControl(dst, src []float64) {
	for i, x := range src {
		if x > 0 {
			dst[i] = x + math.Log1p(math.Exp(-x))
		} else {
			dst[i] = math.Log1p(math.Exp(x))
		}
	}
}

func TestVsoftplusF64Arm64Accuracy(t *testing.T) {
	const n = 1<<18 + 1
	src := make([]float64, n)
	for i := range src {
		src[i] = -800 + 1600*float64(i)/float64(n-1)
	}
	wantInput := append([]float64(nil), src...)
	dst := make([]float64, n)
	vsoftplusF64(dst, src)
	var maxAbs float64
	for i, x := range src {
		if math.Float64bits(x) != math.Float64bits(wantInput[i]) {
			t.Fatalf("input changed at %d", i)
		}
		var want float64
		if x > 0 {
			want = x + math.Log1p(math.Exp(-x))
		} else {
			want = math.Log1p(math.Exp(x))
		}
		err := math.Abs(dst[i] - want)
		if err > maxAbs {
			maxAbs = err
		}
	}
	t.Logf("ARM64 NEON F64 Softplus max absolute error %.3e over %d values", maxAbs, n)
	if maxAbs > 1e-13 {
		t.Fatalf("max absolute error %.3e exceeds 1e-13", maxAbs)
	}
}

func TestVsoftplusF64Arm64VectorTailBitIdentity(t *testing.T) {
	for n := 1; n <= 17; n++ {
		src := make([]float64, n)
		dst := make([]float64, n)
		for i := range src {
			src[i] = -96 + float64(i*37+n)/3
		}
		vsoftplusF64(dst, src)
		for i, x := range src {
			want := softplusF64Arm64Poly(x)
			if math.Float64bits(dst[i]) != math.Float64bits(want) {
				t.Fatalf("n=%d i=%d x=%g: vector/tail=%#x scalar=%#x", n, i, x,
					math.Float64bits(dst[i]), math.Float64bits(want))
			}
		}
	}
}

func TestVsoftplusF64Arm64EdgesAliasAndNaNPayload(t *testing.T) {
	negZero := math.Copysign(0, -1)
	qnanPos := math.Float64frombits(0x7ff8000000000042)
	qnanNeg := math.Float64frombits(0xfff8000000000084)
	src := []float64{0, negZero, math.Inf(1), math.Inf(-1), qnanPos, qnanNeg,
		math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, 708, -708, 709, -709, 1e-16}
	wantInput := append([]float64(nil), src...)
	got := make([]float64, len(src))
	vsoftplusF64(got, src)
	alias := append([]float64(nil), src...)
	vsoftplusF64(alias, alias)
	for i, x := range src {
		want := softplusF64Arm64Poly(x)
		if math.Float64bits(got[i]) != math.Float64bits(want) {
			t.Errorf("i=%d x=%v: vector=%v (%#x), scalar=%v (%#x)", i, x, got[i],
				math.Float64bits(got[i]), want, math.Float64bits(want))
		}
		if math.Float64bits(alias[i]) != math.Float64bits(got[i]) {
			t.Errorf("i=%d: alias=%#x separate=%#x", i,
				math.Float64bits(alias[i]), math.Float64bits(got[i]))
		}
		if math.Float64bits(src[i]) != math.Float64bits(wantInput[i]) {
			t.Errorf("input changed at %d", i)
		}
	}
	if math.Float64bits(got[4]) != math.Float64bits(qnanPos) ||
		math.Float64bits(got[5]) != math.Float64bits(qnanNeg) {
		t.Fatalf("NaN payloads changed: got %#x %#x want %#x %#x",
			math.Float64bits(got[4]), math.Float64bits(got[5]),
			math.Float64bits(qnanPos), math.Float64bits(qnanNeg))
	}
}

func TestSoftplusF64Arm64MatchesReferenceBackend(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend not registered")
	}
	const n = 200003
	x := tensor.New(tensor.F64, tensor.Shape{n})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Sin(float64(i)*0.017) * 120
	}
	wantInput := append([]float64(nil), xs...)
	got, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpSoftplus,
		[]*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpSoftplus,
		[]*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got[0].Storage().F64() {
		w := want[0].Storage().F64()[i]
		if err := math.Abs(v - w); err > 1e-13 {
			t.Fatalf("i=%d x=%v: cpu %.17g ref %.17g absolute error %.3e", i, xs[i], v, w, err)
		}
		if math.Float64bits(xs[i]) != math.Float64bits(wantInput[i]) {
			t.Fatalf("input changed at %d", i)
		}
	}
}

func BenchmarkVsoftplusF64Arm64(b *testing.B) {
	for _, n := range []int{1 << 16, 1 << 18} {
		src := make([]float64, n)
		for i := range src {
			src[i] = -120 + 240*float64(i)/float64(n-1)
		}
		dst := make([]float64, n)
		b.Run("neon/"+softplusBenchSize(n), func(b *testing.B) {
			b.SetBytes(int64(n * 8 * 2))
			for b.Loop() {
				vsoftplusF64(dst, src)
			}
		})
		b.Run("scalar/"+softplusBenchSize(n), func(b *testing.B) {
			b.SetBytes(int64(n * 8 * 2))
			for b.Loop() {
				softplusF64Arm64ScalarControl(dst, src)
			}
		})
	}
}

func softplusBenchSize(n int) string {
	if n == 1<<16 {
		return "64K"
	}
	return "256K"
}
