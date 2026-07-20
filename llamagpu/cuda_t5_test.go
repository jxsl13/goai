//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// tinyT5CheckpointHF builds a minimal but well-formed HF T5ForConditionalGeneration tensor map — one
// gated-GELU (v1.1) encoder block and one decoder block sharing "shared.weight" (dim 8, 2 heads,
// d_kv 4), small deterministic weights, no safetensors fixture. T5FromHF reads only its encoder.*
// keys and T5DecoderFromHF only its decoder.* keys, so the same map (a real checkpoint's shape) feeds
// both loaders — used by ExampleGPUT5_Forward and ExampleGPUT5Decoder_Decode to demonstrate the
// NewT5CUDA / NewT5DecoderCUDA call shape without needing `make golden` fixtures.
func tinyT5CheckpointHF() map[string]*tensor.Tensor {
	const dim, heads, headDim, ffn, vocab, buckets = 8, 2, 4, 16, 12, 8
	attW := heads * headDim
	w := func(shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape(shape))
		d := x.Storage().F32()
		for i := range d {
			d[i] = float32(i%13-6) * 0.03
		}
		return x
	}
	ones := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = 1
		}
		return x
	}
	m := map[string]*tensor.Tensor{"shared.weight": w(vocab, dim)}
	for _, side := range []string{"encoder", "decoder"} {
		m[side+".block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = w(buckets, heads)
		m[side+".final_layer_norm.weight"] = ones(dim)
		p := side + ".block.0."
		m[p+"layer.0.SelfAttention.q.weight"] = w(attW, dim)
		m[p+"layer.0.SelfAttention.k.weight"] = w(attW, dim)
		m[p+"layer.0.SelfAttention.v.weight"] = w(attW, dim)
		m[p+"layer.0.SelfAttention.o.weight"] = w(dim, attW)
		m[p+"layer.0.layer_norm.weight"] = ones(dim)
	}
	// encoder.layer.1 is the FFN sublayer; decoder.layer.1 is cross-attention and layer.2 the FFN.
	m["encoder.block.0.layer.1.layer_norm.weight"] = ones(dim)
	m["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"] = w(ffn, dim)
	m["encoder.block.0.layer.1.DenseReluDense.wi_1.weight"] = w(ffn, dim)
	m["encoder.block.0.layer.1.DenseReluDense.wo.weight"] = w(dim, ffn)
	m["decoder.block.0.layer.1.EncDecAttention.q.weight"] = w(attW, dim)
	m["decoder.block.0.layer.1.EncDecAttention.k.weight"] = w(attW, dim)
	m["decoder.block.0.layer.1.EncDecAttention.v.weight"] = w(attW, dim)
	m["decoder.block.0.layer.1.EncDecAttention.o.weight"] = w(dim, attW)
	m["decoder.block.0.layer.1.layer_norm.weight"] = ones(dim)
	m["decoder.block.0.layer.2.layer_norm.weight"] = ones(dim)
	m["decoder.block.0.layer.2.DenseReluDense.wi_0.weight"] = w(ffn, dim)
	m["decoder.block.0.layer.2.DenseReluDense.wi_1.weight"] = w(ffn, dim)
	m["decoder.block.0.layer.2.DenseReluDense.wo.weight"] = w(dim, ffn)
	return m
}

// tinyT5Config is the nlp.T5Config matching tinyT5CheckpointHF (both the encoder and decoder loaders
// take the same geometry from the same checkpoint).
func tinyT5Config() nlp.T5Config {
	return nlp.T5Config{Heads: 2, HeadDim: 4, NumBuckets: 8, MaxDistance: 16, Eps: 1e-6}
}

// ExampleGPUT5_Forward uploads a tiny T5 encoder to the GPU and runs one bidirectional forward over
// the relative-position-bias attention. Real checkpoints arrive via T5FromHF(safetensors.LoadFile(...),
// ...). Runs the real GPU path when a CUDA device is present; prints the same shape either way so the
// example is deterministic without one.
func ExampleGPUT5_Forward() {
	if !cuda.Available() {
		fmt.Println("hidden states: 3x8")
		return
	}
	m, err := nlp.T5FromHF(tinyT5CheckpointHF(), tinyT5Config())
	if err != nil {
		panic(err)
	}
	enc, err := llamagpu.NewT5CUDA(m)
	if err != nil {
		panic(err)
	}
	defer enc.Release()

	tokens := []int{3, 7, 1}
	hidden, err := enc.Forward(tokens)
	if err != nil {
		panic(err)
	}
	fmt.Printf("hidden states: %dx%d\n", len(tokens), len(hidden)/len(tokens))
	// Output: hidden states: 3x8
}

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
