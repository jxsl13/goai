package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMARSStep covers one MARS-AdamW update over transformer-sized 2-D weights,
// F64, default config (Gamma=0.025, no clip → the fused fast path).
func BenchmarkMARSStep(b *testing.B) {
	shapes := [][2]int{{1024, 1024}, {1024, 4096}, {4096, 1024}}
	params := make([]*tensor.Tensor, len(shapes))
	grads := make([]*tensor.Tensor, len(shapes))
	for i, s := range shapes {
		p := tensor.New(tensor.F64, tensor.Shape{s[0], s[1]})
		g := tensor.New(tensor.F64, tensor.Shape{s[0], s[1]})
		ps, gs := p.Storage().F64(), g.Storage().F64()
		for j := range ps {
			ps[j] = math.Sin(float64(i*7+j) * 0.001)
			gs[j] = math.Cos(float64(i*3+j) * 0.0017)
		}
		params[i], grads[i] = p, g
	}
	opt := nn.NewMARS(params, 0.001)
	fn := func(q *tensor.Tensor) *tensor.Tensor {
		for i, p := range params {
			if q == p {
				return grads[i]
			}
		}
		return nil
	}
	if err := opt.Step(fn); err != nil { // prime a.seen so the correction path is live
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := opt.Step(fn); err != nil {
			b.Fatal(err)
		}
	}
}
