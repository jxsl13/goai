package nn_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MaxAttentionLogits fans its per-head max over GOMAXPROCS. Each head writes only out[h] and
// runs its exact serial code (no reassociation across the parallelization), so the result must
// be BYTE-FOR-BYTE identical to the single-worker result. Locked at GOMAXPROCS=1 vs N.
func TestMaxAttentionLogitsParallelBitExact(t *testing.T) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	for _, cfg := range []struct {
		seq, dm, heads int
		causal         bool
	}{
		{96, 128, 4, true},
		{128, 256, 8, false},
		{200, 192, 6, true},
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

		runtime.GOMAXPROCS(1)
		s1, err := nn.MaxAttentionLogits(q, k, cfg.heads, 0.125, cfg.causal)
		if err != nil {
			t.Fatal(err)
		}
		runtime.GOMAXPROCS(prev)
		s2, err := nn.MaxAttentionLogits(q, k, cfg.heads, 0.125, cfg.causal)
		if err != nil {
			t.Fatal(err)
		}
		for h := range s1 {
			if s1[h] != s2[h] {
				t.Fatalf("cfg=%+v head=%d: serial %v != parallel %v", cfg, h, s1[h], s2[h])
			}
		}
	}
}
