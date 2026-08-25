//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func tanhF64Arm64ScalarControl(dst, src []float64) {
	for i, x := range src {
		dst[i] = math.Tanh(x)
	}
}

func TestTanhF64Arm64Accuracy(t *testing.T) {
	const n = 1<<18 + 1
	src := make([]float64, n)
	for i := range src {
		src[i] = -20 + 40*float64(i)/float64(n-1)
	}
	wantInput := append([]float64(nil), src...)
	dst := make([]float64, n)
	vtanhF64(dst, src)
	var maxAbs float64
	for i, x := range src {
		if math.Float64bits(x) != math.Float64bits(wantInput[i]) {
			t.Fatalf("input changed at %d", i)
		}
		err := math.Abs(dst[i] - math.Tanh(x))
		if err > maxAbs {
			maxAbs = err
		}
	}
	t.Logf("ARM64 NEON-composed F64 tanh max absolute error %.3e over %d values", maxAbs, n)
	if maxAbs > 1e-13 {
		t.Fatalf("max absolute error %.3e exceeds 1e-13", maxAbs)
	}
}

func TestTanhF64Arm64VectorTailBitIdentity(t *testing.T) {
	for n := 1; n <= 17; n++ {
		src := make([]float64, n)
		dst := make([]float64, n)
		for i := range src {
			src[i] = -20 + float64(i*37+n)/11
		}
		vtanhF64(dst, src)
		for i, x := range src {
			want := tanhF64LogisticPoly(x)
			if math.Float64bits(dst[i]) != math.Float64bits(want) {
				t.Fatalf("n=%d i=%d x=%g: vector/tail=%#x scalar=%#x", n, i, x,
					math.Float64bits(dst[i]), math.Float64bits(want))
			}
		}
	}
}

func TestTanhF64Arm64Edges(t *testing.T) {
	negZero := math.Copysign(0, -1)
	nan := math.Float64frombits(0x7ff8000000000042)
	src := []float64{0, negZero, math.Inf(1), math.Inf(-1), nan,
		math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, 1e-16, -1e-16}
	wantInput := append([]float64(nil), src...)
	got := make([]float64, len(src))
	vtanhF64(got, src)
	for i, x := range src {
		want := tanhF64LogisticPoly(x)
		if math.Float64bits(got[i]) != math.Float64bits(want) {
			t.Errorf("x=%v: vector=%v (%#x), scalar=%v (%#x)", x, got[i],
				math.Float64bits(got[i]), want, math.Float64bits(want))
		}
		if math.Float64bits(x) != math.Float64bits(wantInput[i]) {
			t.Fatalf("input changed at %d", i)
		}
	}
	if math.Float64bits(got[0]) != 0 || math.Float64bits(got[1]) != 1<<63 {
		t.Errorf("tanh(+0,-0) bits=(%#x,%#x), want (0,%#x)",
			math.Float64bits(got[0]), math.Float64bits(got[1]), uint64(1)<<63)
	}
	if got[2] != 1 || got[3] != -1 || !math.IsNaN(got[4]) {
		t.Errorf("tanh(+Inf,-Inf,NaN)=(%v,%v,%v), want (1,-1,NaN)", got[2], got[3], got[4])
	}
	for _, i := range []int{5, 6, 7, 8} {
		if err := math.Abs(got[i] - math.Tanh(src[i])); err > 1e-13 {
			t.Errorf("tiny x=%g: absolute error %.3e", src[i], err)
		}
	}
}

func TestTanhF64Arm64MatchesRef(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend not registered")
	}
	const n = 200003
	x := tensor.New(tensor.F64, tensor.Shape{n})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Sin(float64(i)*0.017) * 20
	}
	wantInput := append([]float64(nil), xs...)
	got, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpTanh, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpTanh,
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

func BenchmarkVtanhF64Arm64(b *testing.B) {
	const n = 1 << 16
	src := make([]float64, n)
	for i := range src {
		src[i] = -8 + 16*float64(i)/float64(n-1)
	}
	dst := make([]float64, n)
	b.Run("neon-composed", func(b *testing.B) {
		b.SetBytes(n * 8 * 2)
		for b.Loop() {
			vtanhF64(dst, src)
		}
	})
	b.Run("scalar", func(b *testing.B) {
		b.SetBytes(n * 8 * 2)
		for b.Loop() {
			tanhF64Arm64ScalarControl(dst, src)
		}
	})
}
