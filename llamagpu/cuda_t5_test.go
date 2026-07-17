//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestCUDAT5MatchesReference is the parity anchor for the GPU T5 encoder — the second non-decoder GPU
// model. Its [seq, dim] hidden states must match the reference nlp.T5.Forward for BOTH FFN variants:
// v1.1 gated-GELU and v1.0 ReLU. T5 exercises the new per-head relative-position-bias attention
// (cu_attn_softmax_bias / MHABias, unscaled), PRE-LN residuals, RMSNorm and a d_kv (head width)
// independent of dim/heads — all on the shared recorder ops.
func TestCUDAT5MatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cases := []struct{ name, weights string }{
		{"v1.1-gated-gelu", "../nlp/testdata/t5_hf.safetensors"},
		{"v1.0-relu", "../nlp/testdata/t5v10_hf.safetensors"},
	}
	tokens := []int{3, 7, 1, 9, 4, 2, 8}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ts, _, err := safetensors.LoadFile(c.weights)
			if err != nil {
				t.Skipf("t5 testdata unavailable (run make golden): %v", err)
			}
			m, err := nlp.T5FromHF(ts, nlp.T5Config{Heads: 2, HeadDim: 8, Eps: 1e-6})
			if err != nil {
				t.Fatalf("T5FromHF: %v", err)
			}
			enc, err := llamagpu.NewT5CUDA(m)
			if err != nil {
				t.Fatalf("NewT5CUDA: %v", err)
			}
			defer enc.Release()

			refT, err := m.Forward(backend.NewContext().WithBackend(backend.Reference()), tokens)
			if err != nil {
				t.Fatalf("reference Forward: %v", err)
			}
			seq, dim := refT.Shape()[0], refT.Shape()[1]
			got, err := enc.Forward(tokens)
			if err != nil {
				t.Fatalf("cuda Forward: %v", err)
			}
			if len(got) != seq*dim {
				t.Fatalf("got %d values, want seq·dim %d", len(got), seq*dim)
			}
			var maxAbs float64
			for i := 0; i < seq; i++ {
				for j := 0; j < dim; j++ {
					d := math.Abs(float64(got[i*dim+j]) - refT.AtF64(i, j))
					if math.IsNaN(float64(got[i*dim+j])) || d > maxAbs {
						maxAbs = d
					}
				}
			}
			if maxAbs > 2e-3 {
				t.Fatalf("%s: GPU T5 diverges from reference: max abs %.3e", c.name, maxAbs)
			}
			t.Logf("llamagpu NewT5CUDA (%s) matches reference nlp.T5.Forward (max abs %.2e); relpos-bias GPU encoder", c.name, maxAbs)
		})
	}
}
