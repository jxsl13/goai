package nlp

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkDeepSeekV2MoEFFN measures the DeepSeekMoE FFN sublayer at realistic routing (160 routed
// experts, top-6 — DeepSeek-V2's n_routed_experts/num_experts_per_tok) in the decode (seq=1) and small-
// prefill (seq=32) regimes. In decode only topK experts have nonzero gate weight, so the dense-vs-
// selected difference is the whole point: the skip evaluates 6 expert SwiGLUs instead of 160.
func BenchmarkDeepSeekV2MoEFFN(b *testing.B) {
	const dim, hidden, E, topK = 512, 256, 160, 6
	moe, err := nn.NewDeepSeekMoE(tensor.F64, dim, hidden, 1, E, topK, 42)
	if err != nil {
		b.Fatal(err)
	}
	m := &DeepSeekV2{Config: DeepSeekV2Config{RoutedScale: 1.0}}
	run := func(name string, seq int) {
		x := tensor.New(tensor.F64, tensor.Shape{seq, dim})
		xs := x.Storage().F64()
		rng := rand.New(rand.NewSource(1))
		for i := range xs {
			xs[i] = rng.NormFloat64()
		}
		b.Run(name, func(b *testing.B) {
			ctx := backend.NewContext()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := m.moeFFN(ctx, moe, x); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
	run("decode_seq1", 1)
	run("prefill_seq32", 32)
}
