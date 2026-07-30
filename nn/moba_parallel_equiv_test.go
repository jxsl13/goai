package nn_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MoBAAttention fans its per-head computation over GOMAXPROCS. Each head writes disjoint output
// columns and runs its exact serial code, so the output must be BYTE-FOR-BYTE identical to the
// single-worker result. Locked at GOMAXPROCS=1 vs N; covers a shape that exercises the F64 fast
// path (dm%heads==0, all flatF64) at several head counts.
func TestMoBAAttentionParallelBitExact(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	for _, cfg := range []struct{ seq, dm, heads, block, topK int }{
		{96, 128, 4, 16, 3},
		{128, 192, 6, 32, 2},
		{200, 128, 8, 24, 4},
	} {
		mk := func(f func(i int) float64) *tensor.Tensor {
			tt := tensor.New(tensor.F64, tensor.Shape{cfg.seq, cfg.dm})
			s := tt.Storage().F64()
			for i := range s {
				s[i] = f(i)
			}
			return tt
		}
		q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.021) })
		k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.017) })
		v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.013) })

		runtime.GOMAXPROCS(1)
		o1, err := nn.MoBAAttention(q, k, v, cfg.heads, cfg.block, cfg.topK, 0)
		if err != nil {
			t.Fatal(err)
		}
		runtime.GOMAXPROCS(prev)
		o2, err := nn.MoBAAttention(q, k, v, cfg.heads, cfg.block, cfg.topK, 0)
		if err != nil {
			t.Fatal(err)
		}
		a, b := o1.Storage().F64(), o2.Storage().F64()
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("cfg=%+v idx=%d: serial %v != parallel %v", cfg, i, a[i], b[i])
			}
		}
	}
}
