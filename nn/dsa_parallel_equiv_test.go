package nn_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// DSAAttention fans its per-query computation over GOMAXPROCS. Each query writes only its own
// output row and runs its exact serial code, so the output must be BYTE-FOR-BYTE identical to
// the single-worker result. Locked at GOMAXPROCS=1 vs N over several shapes.
func TestDSAAttentionParallelBitExact(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	for _, cfg := range []struct{ seq, dm, heads, idxHeads, idxDim, topK int }{
		{80, 96, 4, 3, 16, 8},
		{128, 128, 8, 4, 32, 16},
		{200, 64, 4, 2, 24, 24},
	} {
		mk := func(f func(i int) float64, cols int) *tensor.Tensor {
			tt := tensor.New(tensor.F64, tensor.Shape{cfg.seq, cols})
			s := tt.Storage().F64()
			for i := range s {
				s[i] = f(i)
			}
			return tt
		}
		q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.021) }, cfg.dm)
		k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.017) }, cfg.dm)
		v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.013) }, cfg.dm)
		qIdx := mk(func(i int) float64 { return math.Cos(float64(i) * 0.007) }, cfg.idxHeads*cfg.idxDim)
		kIdx := mk(func(i int) float64 { return math.Sin(float64(i) * 0.009) }, cfg.idxHeads*cfg.idxDim)
		w := make([]float64, cfg.idxHeads)
		for h := range w {
			w[h] = 1.0 / float64(h+1)
		}
		runtime.GOMAXPROCS(1)
		o1, err := nn.DSAAttention(q, k, v, qIdx, kIdx, w, cfg.heads, cfg.topK, 0)
		if err != nil {
			t.Fatal(err)
		}
		runtime.GOMAXPROCS(prev)
		o2, err := nn.DSAAttention(q, k, v, qIdx, kIdx, w, cfg.heads, cfg.topK, 0)
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
