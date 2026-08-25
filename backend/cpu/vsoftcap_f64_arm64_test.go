//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func softcapF64Arm64ScalarControl(dst, src []float64, cap float64) {
	for i, x := range src {
		dst[i] = cap * math.Tanh(x/cap)
	}
}

func TestVsoftcapF64Arm64Accuracy(t *testing.T) {
	const n = 1<<18 + 1
	for _, cap := range []float64{30, 50} {
		src := make([]float64, n)
		for i := range src {
			src[i] = -80*cap + 160*cap*float64(i)/float64(n-1)
		}
		wantInput := append([]float64(nil), src...)
		dst := make([]float64, n)
		vsoftcapF64(dst, src, cap)
		var maxAbs float64
		for i, x := range src {
			if math.Float64bits(x) != math.Float64bits(wantInput[i]) {
				t.Fatalf("cap=%g: input changed at %d", cap, i)
			}
			err := math.Abs(dst[i] - cap*math.Tanh(x/cap))
			if err > maxAbs {
				maxAbs = err
			}
		}
		t.Logf("ARM64 NEON F64 soft-cap cap=%g max absolute error %.3e over %d values", cap, maxAbs, n)
		if maxAbs > 1e-13 {
			t.Fatalf("cap=%g: max absolute error %.3e exceeds 1e-13", cap, maxAbs)
		}
	}
}

func TestVsoftcapF64Arm64VectorTailBitIdentity(t *testing.T) {
	for _, cap := range []float64{30, 50} {
		for n := 1; n <= 17; n++ {
			src := make([]float64, n)
			dst := make([]float64, n)
			for i := range src {
				src[i] = -4*cap + cap*float64(i*37+n)/11
			}
			vsoftcapF64(dst, src, cap)
			for i, x := range src {
				want := softcapF64Arm64Poly(x, cap)
				if math.Float64bits(dst[i]) != math.Float64bits(want) {
					t.Fatalf("cap=%g n=%d i=%d x=%g: vector/tail=%#x scalar=%#x", cap, n, i, x,
						math.Float64bits(dst[i]), math.Float64bits(want))
				}
			}
		}
	}
}

func TestVsoftcapF64Arm64EdgesAliasAndNaNPayload(t *testing.T) {
	negZero := math.Copysign(0, -1)
	qnanPos := math.Float64frombits(0x7ff8000000000042)
	qnanNeg := math.Float64frombits(0xfff8000000000084)
	for _, cap := range []float64{30, 50} {
		src := []float64{0, negZero, math.Inf(1), math.Inf(-1), qnanPos, qnanNeg,
			math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, cap, -cap, 708 * cap / 2, -708 * cap / 2, 1e-16}
		wantInput := append([]float64(nil), src...)
		got := make([]float64, len(src))
		vsoftcapF64(got, src, cap)
		alias := append([]float64(nil), src...)
		vsoftcapF64(alias, alias, cap)
		for i, x := range src {
			want := softcapF64Arm64Poly(x, cap)
			if math.Float64bits(got[i]) != math.Float64bits(want) {
				t.Errorf("cap=%g i=%d x=%v: vector=%v (%#x), scalar=%v (%#x)", cap, i, x, got[i],
					math.Float64bits(got[i]), want, math.Float64bits(want))
			}
			if math.Float64bits(alias[i]) != math.Float64bits(got[i]) {
				t.Errorf("cap=%g i=%d: alias=%#x separate=%#x", cap, i,
					math.Float64bits(alias[i]), math.Float64bits(got[i]))
			}
			if math.Float64bits(src[i]) != math.Float64bits(wantInput[i]) {
				t.Fatalf("cap=%g: input changed at %d", cap, i)
			}
			if !math.IsNaN(x) {
				if err := math.Abs(got[i] - cap*math.Tanh(x/cap)); err > 1e-13 {
					t.Errorf("cap=%g i=%d x=%v: absolute reference error %.3e", cap, i, x, err)
				}
			}
		}
		if math.Float64bits(got[0]) != 0 || math.Float64bits(got[1]) != 1<<63 {
			t.Errorf("cap=%g: softcap(+0,-0) bits=(%#x,%#x)", cap,
				math.Float64bits(got[0]), math.Float64bits(got[1]))
		}
		if got[2] != cap || got[3] != -cap {
			t.Errorf("cap=%g: softcap(+Inf,-Inf)=(%v,%v)", cap, got[2], got[3])
		}
		if math.Float64bits(got[4]) != math.Float64bits(qnanPos) ||
			math.Float64bits(got[5]) != math.Float64bits(qnanNeg) {
			t.Errorf("cap=%g: NaN payloads=(%#x,%#x), want (%#x,%#x)", cap,
				math.Float64bits(got[4]), math.Float64bits(got[5]),
				math.Float64bits(qnanPos), math.Float64bits(qnanNeg))
		}
	}
}

func TestSoftCapF64Arm64MatchesReferenceBackend(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend not registered")
	}
	const n = 200003
	for _, cap := range []float64{30, 50} {
		x := tensor.New(tensor.F64, tensor.Shape{n})
		xs := x.Storage().F64()
		for i := range xs {
			xs[i] = math.Sin(float64(i)*0.017) * 4 * cap
		}
		wantInput := append([]float64(nil), xs...)
		attrs := backend.SoftCapAttrs{Cap: cap}
		got, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpSoftCap,
			[]*tensor.Tensor{x}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		want, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpSoftCap,
			[]*tensor.Tensor{x}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got[0].Storage().F64() {
			w := want[0].Storage().F64()[i]
			if err := math.Abs(v - w); err > 1e-13 {
				t.Fatalf("cap=%g i=%d x=%v: cpu %.17g ref %.17g absolute error %.3e", cap, i, xs[i], v, w, err)
			}
			if math.Float64bits(xs[i]) != math.Float64bits(wantInput[i]) {
				t.Fatalf("cap=%g: input changed at %d", cap, i)
			}
		}
	}
}

func BenchmarkVsoftcapF64Arm64(b *testing.B) {
	for _, n := range []int{1 << 16, 1 << 18} {
		src := make([]float64, n)
		for i := range src {
			src[i] = -120 + 240*float64(i)/float64(n-1)
		}
		dst := make([]float64, n)
		b.Run("neon/"+softcapBenchSize(n), func(b *testing.B) {
			b.SetBytes(int64(n * 8 * 2))
			for b.Loop() {
				vsoftcapF64(dst, src, 30)
			}
		})
		b.Run("scalar/"+softcapBenchSize(n), func(b *testing.B) {
			b.SetBytes(int64(n * 8 * 2))
			for b.Loop() {
				softcapF64Arm64ScalarControl(dst, src, 30)
			}
		})
	}
}

func softcapBenchSize(n int) string {
	if n == 1<<16 {
		return "64K"
	}
	return "256K"
}
