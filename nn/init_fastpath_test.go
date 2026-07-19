package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// slowFillUniform is the verbatim pre-T897 per-element path, kept here as the
// bit-identity oracle: the fast path must reproduce it exactly (seeded, §V13).
func slowFillUniform(t *tensor.Tensor, lo, hi float64, seed uint64) {
	rng := rand.New(rand.NewPCG(seed, 0x6b79a2c3d4e5f601))
	for i := range t.Numel() {
		t.SetF64(lo+rng.Float64()*(hi-lo), tensor.Unravel(i, t.Shape())...)
	}
}

func TestFillGenBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{1}, {7}, {4, 5}, {2, 3, 4}, {512, 512}} {
			fast := tensor.Zeros(dt, shape)
			slow := tensor.Zeros(dt, shape)
			fillUniform(fast, -0.3, 0.7, 42)
			slowFillUniform(slow, -0.3, 0.7, 42)
			n := fast.Numel()
			for i := range n {
				idx := tensor.Unravel(i, shape)
				a, b := fast.AtF64(idx...), slow.AtF64(idx...)
				if math.Float64bits(a) != math.Float64bits(b) {
					t.Fatalf("dt=%v shape=%v elem %d: fast %v != slow %v (not bit-identical)", dt, shape, i, a, b)
				}
			}
		}
	}
}

// A non-contiguous view must still fill correctly via the fallback.
func TestFillGenNonContiguousView(t *testing.T) {
	base := tensor.Zeros(tensor.F64, tensor.Shape{4, 3})
	view, err := base.Transpose(0, 1) // [3,4], non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	fillUniform(view, 1.0, 1.0, 7) // constant 1.0 (lo==hi): every elem must be 1
	n := view.Numel()
	for i := range n {
		if got := view.AtF64(tensor.Unravel(i, view.Shape())...); got != 1.0 {
			t.Fatalf("non-contiguous fill elem %d = %v, want 1.0", i, got)
		}
	}
}

func BenchmarkFillUniformFast(b *testing.B) {
	t := tensor.Zeros(tensor.F32, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fillUniform(t, -0.1, 0.1, uint64(i))
	}
}

func BenchmarkFillUniformSlow(b *testing.B) {
	t := tensor.Zeros(tensor.F32, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		slowFillUniform(t, -0.1, 0.1, uint64(i))
	}
}

func slowZeros(t *tensor.Tensor) {
	for i := range t.Numel() {
		t.SetF64(0, tensor.Unravel(i, t.Shape())...)
	}
}

func BenchmarkZerosFast(b *testing.B) {
	t := tensor.Zeros(tensor.F32, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Zeros(t)
	}
}

func BenchmarkZerosSlow(b *testing.B) {
	t := tensor.Zeros(tensor.F32, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		slowZeros(t)
	}
}
