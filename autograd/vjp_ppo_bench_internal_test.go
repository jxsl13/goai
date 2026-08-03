package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchPPOVJP times the PPO clipped-surrogate backward at batch sizes a rollout produces. The
// rule had NO benchmark, so neither its per-element tensor dispatch nor its clamp could be
// measured before this existed.
func benchPPOVJP(b *testing.B, n int) {
	mk := func(off float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{n})
		s := t.Storage().F64()
		for i := range s {
			s[i] = math.Sin(float64(i)*0.017+off) * 0.4
		}
		return t
	}
	in := []*tensor.Tensor{mk(0), mk(1), mk(2)}
	g := tensor.New(tensor.F64, tensor.Shape{})
	g.Storage().F64()[0] = 1
	vjp := vjps[backend.OpPPOClip]
	if vjp == nil {
		b.Skip("PPO VJP not registered")
	}
	attrs := backend.PPOClipAttrs{Epsilon: 0.2}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := vjp(nil, in, nil, attrs, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPPOVJP_65536(b *testing.B) { benchPPOVJP(b, 65536) }
func BenchmarkPPOVJP_4096(b *testing.B)  { benchPPOVJP(b, 4096) }
