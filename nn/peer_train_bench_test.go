package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// benchPEERForwardTrain times PEER on a RECORDING context (autograd tape) — the
// training/fine-tuning path — at product-key scale. This exercises the gate-score
// gather's recording branch (the one-hot·mul·sum vs the flat OpEmbed take-along),
// which the inference bench (benchPEERForward, Recorder==nil) never touches.
func benchPEERForwardTrain(b *testing.B, tks, dModel, n, topK, heads int) {
	p := nn.NewPEER(tensor.F64, dModel, n, topK, 7, nn.WithPEERHeads(heads))
	x := tensor.New(tensor.F64, tensor.Shape{tks, dModel})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = float64((i*2654435761)&0xffff)/32768.0 - 1.0
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tape := autograd.NewTape()
		if _, _, _, err := p.Forward(tape.Context(), x); err != nil {
			b.Fatal(err)
		}
	}
}

// N ≈ 262k experts, product-key retrieval — the "million experts" regime (training).
func BenchmarkPEERForwardTrain_T64_d512_n512_k16_h4(b *testing.B) {
	benchPEERForwardTrain(b, 64, 512, 512, 16, 4)
}

// Smaller: N ≈ 16k, single head (training).
func BenchmarkPEERForwardTrain_T32_d256_n128_k8_h1(b *testing.B) {
	benchPEERForwardTrain(b, 32, 256, 128, 8, 1)
}
