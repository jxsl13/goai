package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// benchPEERForward times PEER inference (ctx.Recorder == nil) at product-key
// scale: N = n² experts retrieved topK per head. The gate-score gather is the
// hot dispatch loop — Heads·topK slots each build two [T,n] one-hot matrices.
func benchPEERForward(b *testing.B, tks, dModel, n, topK, heads int) {
	p := nn.NewPEER(tensor.F64, dModel, n, topK, 7, nn.WithPEERHeads(heads))
	x := tensor.New(tensor.F64, tensor.Shape{tks, dModel})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = float64((i*2654435761)&0xffff)/32768.0 - 1.0
	}
	ctx := backend.NewContext() // no recorder → inference path
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := p.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

// N ≈ 262k experts, product-key retrieval — the "million experts" regime.
func BenchmarkPEERForward_T64_d512_n512_k16_h4(b *testing.B) {
	benchPEERForward(b, 64, 512, 512, 16, 4)
}

// Smaller: N ≈ 16k, single head.
func BenchmarkPEERForward_T32_d256_n128_k8_h1(b *testing.B) {
	benchPEERForward(b, 32, 256, 128, 8, 1)
}
