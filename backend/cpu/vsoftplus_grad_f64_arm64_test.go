//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

func softplusGradF64Arm64ScalarControl(dst, x, g []float64) {
	for i, v := range x {
		z := math.Exp(-math.Abs(v))
		num := z
		if v >= 0 {
			num = 1
		}
		dst[i] = g[i] * (num / (1 + z))
	}
}

func TestSoftplusGradF64Arm64Accuracy(t *testing.T) {
	const n = 1<<18 + 1
	x := make([]float64, n)
	g := make([]float64, n)
	for i := range x {
		x[i] = -80 + 160*float64(i)/float64(n-1)
		g[i] = math.Sin(float64(i)*0.017) * 3
	}
	dst := make([]float64, n)
	vsoftplusGradF64(dst, x, g)
	var maxRel float64
	for i := range x {
		want := g[i] * sigmoidF64ScalarReference(x[i])
		rel := math.Abs(dst[i]-want) / math.Max(1, math.Abs(want))
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("ARM64 NEON F64 softplus gradient max relative error %.3e over %d values", maxRel, n)
	if maxRel > 1e-13 {
		t.Fatalf("max relative error %.3e exceeds 1e-13", maxRel)
	}
}

func TestSoftplusGradF64Arm64VectorTailBitIdentity(t *testing.T) {
	for x := -80.0; x <= 80.0; x += 0.00031 {
		g := math.Sin(x*0.73) * 3
		body := []float64{0, 0}
		vsoftplusGradF64(body, []float64{x, x}, []float64{g, g})
		want := g * sigmoidF64poly(x)
		if math.Float64bits(body[0]) != math.Float64bits(want) ||
			math.Float64bits(body[1]) != math.Float64bits(want) {
			t.Fatalf("x=%g g=%g: vector=(%#x,%#x) tail=%#x", x, g,
				math.Float64bits(body[0]), math.Float64bits(body[1]), math.Float64bits(want))
		}
	}
}

func TestSoftplusGradF64Arm64Edges(t *testing.T) {
	negZero := math.Copysign(0, -1)
	nanX := math.Float64frombits(0x7ff8000000000042)
	nanG := math.Float64frombits(0x7ff8000000000084)
	x := []float64{0, negZero, 1, -1, math.Inf(1), math.Inf(-1), nanX, 2, -2}
	g := []float64{1, negZero, math.Inf(1), math.Inf(-1), 0, math.Inf(1), 3, nanG, -3}
	got := make([]float64, len(x))
	vsoftplusGradF64(got, x, g)
	for i := range x {
		want := g[i] * sigmoidF64poly(x[i])
		if math.Float64bits(got[i]) != math.Float64bits(want) {
			t.Errorf("i=%d x=%v g=%v: vector=%v (%#x), scalar=%v (%#x)", i, x[i], g[i],
				got[i], math.Float64bits(got[i]), want, math.Float64bits(want))
		}
	}
}

func TestSoftplusBackwardF64Arm64MatchesRef(t *testing.T) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend not registered")
	}
	const n = 200003
	x := tensor.New(tensor.F64, tensor.Shape{n})
	g := tensor.New(tensor.F64, tensor.Shape{n})
	xs, gs := x.Storage().F64(), g.Storage().F64()
	for i := range xs {
		xs[i] = math.Sin(float64(i)*0.017)*12 - 3
		gs[i] = math.Cos(float64(i)*0.011) * 2
	}
	got, err := backend.Execute(backend.NewContext().WithBackend(be), backend.OpSoftplusBackward,
		[]*tensor.Tensor{x, g}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpSoftplusBackward,
		[]*tensor.Tensor{x, g}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got[0].Storage().F64() {
		w := want[0].Storage().F64()[i]
		if rel := math.Abs(v-w) / math.Max(1, math.Abs(w)); rel > 1e-13 {
			t.Fatalf("i=%d x=%v g=%v: cpu %.17g ref %.17g rel %.3e", i, xs[i], gs[i], v, w, rel)
		}
	}
}

func BenchmarkSoftplusBackwardF64_256K_M2(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("cpu backend not registered")
	}
	const n = 1 << 18
	x := tensor.New(tensor.F64, tensor.Shape{n})
	g := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range n {
		x.SetF64(-8+16*float64(i)/float64(n-1), i)
		g.SetF64(math.Sin(float64(i)*0.017)*3, i)
	}
	ctx := backend.NewContext().WithBackend(be)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := backend.Execute(ctx, backend.OpSoftplusBackward, []*tensor.Tensor{x, g}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVsoftplusGradF64Arm64(b *testing.B) {
	const n = 1 << 18
	x := make([]float64, n)
	g := make([]float64, n)
	for i := range x {
		x[i] = -8 + 16*float64(i)/float64(n-1)
		g[i] = math.Sin(float64(i)*0.017) * 3
	}
	dst := make([]float64, n)
	b.Run("neon", func(b *testing.B) {
		b.SetBytes(n * 8 * 3)
		for b.Loop() {
			vsoftplusGradF64(dst, x, g)
		}
	})
	b.Run("scalar", func(b *testing.B) {
		b.SetBytes(n * 8 * 3)
		for b.Loop() {
			softplusGradF64Arm64ScalarControl(dst, x, g)
		}
	})
}
