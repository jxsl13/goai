package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// BenchmarkTokenLogProbs at a realistic RLHF shape (Llama-3 vocab 128256): TokenLogProbs runs per
// training step for every DPO/GRPO/PPO objective. The gather path avoids the [seq,vocab] one-hot
// alloc + the logp/mul/sum full-vocab passes the old one-hot·mul·sum incurred.
func benchTokenLogProbs(b *testing.B, seq, vocab int) {
	logits := tensor.New(tensor.F32, tensor.Shape{seq, vocab})
	lf := logits.Storage().F32()
	for i := range lf {
		lf[i] = float32((i*131+7)%1000) / 500.0
	}
	targets := make([]int, seq)
	for i := range targets {
		targets[i] = (i * 977) % vocab
	}
	ctx := backend.NewContext()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := nn.TokenLogProbs(ctx, logits, targets); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTokenLogProbs_seq512_v128k(b *testing.B) { benchTokenLogProbs(b, 512, 128256) }
func BenchmarkTokenLogProbs_seq2048_v32k(b *testing.B) { benchTokenLogProbs(b, 2048, 32000) }
