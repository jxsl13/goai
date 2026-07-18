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
			maxAbs, err := encoderMaxAbs(got, seq, dim, 2e-3, refT.AtF64)
			if err != nil {
				t.Fatalf("%s: GPU T5 diverges from reference: %v", c.name, err)
			}
			t.Logf("llamagpu NewT5CUDA (%s) matches reference nlp.T5.Forward (max abs %.2e); relpos-bias GPU encoder", c.name, maxAbs)
		})
	}
}

// TestCUDAT5Q8CloseToF32 validates Q8 on the T5 encoder (NewT5Q8CUDA) for both FFN variants: the
// attention q/k/v/o and FFN wi0/wi1/wOut projections go resident Q8_0, running on the int8 tensor-core
// MMQ GEMM (T5 encodes the whole sequence, M=seq>1). Compared to the f32 encoder by per-token cosine of
// the [seq,dim] hidden states (RMSNorm + relative-position bias stay f32).
func TestCUDAT5Q8CloseToF32(t *testing.T) {
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
			f32Enc, err := llamagpu.NewT5CUDA(m)
			if err != nil {
				t.Fatalf("NewT5CUDA: %v", err)
			}
			defer f32Enc.Release()
			q8Enc, err := llamagpu.NewT5Q8CUDA(m)
			if err != nil {
				t.Fatalf("NewT5Q8CUDA: %v", err)
			}
			defer q8Enc.Release()

			fOut, err := f32Enc.Forward(tokens)
			if err != nil {
				t.Fatalf("f32 Forward: %v", err)
			}
			qOut, err := q8Enc.Forward(tokens)
			if err != nil {
				t.Fatalf("q8 Forward: %v", err)
			}
			dim := len(fOut) / len(tokens)
			minCos := 1.0
			for i := range tokens {
				var dot, nf, nq float64
				for j := 0; j < dim; j++ {
					f, q := float64(fOut[i*dim+j]), float64(qOut[i*dim+j])
					if math.IsNaN(q) || math.IsInf(q, 0) {
						t.Fatalf("q8 token %d dim %d non-finite", i, j)
					}
					dot += f * q
					nf += f * f
					nq += q * q
				}
				if cos := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30); cos < minCos {
					minCos = cos
				}
			}
			if minCos < 0.999 {
				t.Fatalf("T5 Q8 (%s) min per-token cosine %.5f < 0.999 vs f32", c.name, minCos)
			}
			t.Logf("NewT5Q8CUDA (%s) tracks f32 encoder: min per-token cosine %.6f", c.name, minCos)
		})
	}
}
