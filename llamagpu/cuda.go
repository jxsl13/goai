//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// cuda adapter: thin assertions from the backend-agnostic buffer/recorder interfaces to the
// concrete cuda.DeviceF32/cuda.Recorder, the same shape as the metal (§T408) and vulkan (§T409)
// adapters. The one difference is submit semantics: the CUDA recorder enqueues eagerly on the work
// stream (Commit is a no-op, Wait syncs), so asyncEncode stays false — a second recorder must not
// pre-encode onto the shared stream while one is in flight.

type cBuf struct{ *cuda.DeviceF32 }

type cRec struct{ r *cuda.Recorder }

func (c cRec) RMSNorm(x, g, o buffer, rows, dim int, eps float32) error {
	return c.r.RMSNorm(cb(x), cb(g), cb(o), rows, dim, eps)
}
func (c cRec) LayerNorm(x, g, b, o buffer, rows, dim int, eps float32) error {
	return c.r.LayerNorm(cb(x), cb(g), cb(b), cb(o), rows, dim, eps)
}
func (c cRec) AddBias(x, b, o buffer, rows, n int) error {
	return c.r.AddBias(cb(x), cb(b), cb(o), rows, n)
}
func (c cRec) MatMul(a, b, cc buffer, m, k, n int) error {
	return c.r.MatMul(cb(a), cb(b), cb(cc), m, k, n)
}
func (c cRec) MatMulAcc(a, b, cc buffer, m, k, n int) error {
	return c.r.MatMulAcc(cb(a), cb(b), cb(cc), m, k, n)
}
func (c cRec) RoPE(q, inv, o buffer, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return c.r.RoPE(cb(q), cb(inv), cb(o), seq, width, heads, hd, half, pos, posDiv)
}
func (c cRec) RoPEAt(q, inv, o buffer, off, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return c.r.RoPEAt(cb(q), cb(inv), cb(o), off, seq, width, heads, hd, half, pos, posDiv)
}
func (c cRec) RoPEPair(qkv, inv buffer, seq, stride, headsQ, offQ, headsK, offK, hd, half, pos int, posDiv float32) error {
	return c.r.RoPEPair(cb(qkv), cb(inv), seq, stride, headsQ, offQ, headsK, offK, hd, half, pos, posDiv)
}
func (c cRec) RoPEPartialPair(qkv, inv buffer, seq, stride, headsQ, offQ, headsK, offK, hd, rotaryDim, pos int, posDiv float32) error {
	return c.r.RoPEPartialPair(cb(qkv), cb(inv), seq, stride, headsQ, offQ, headsK, offK, hd, rotaryDim, pos, posDiv)
}
func (c cRec) Blit(src buffer, srcOff int, dst buffer, dstOff, n int) error {
	return c.r.Blit(cb(src), srcOff, cb(dst), dstOff, n)
}
func (c cRec) Copy2D(src buffer, srcOff, srcStride int, dst buffer, dstOff, dstStride, rows, rowFloats int) error {
	return c.r.Copy2D(cb(src), srcOff, srcStride, cb(dst), dstOff, dstStride, rows, rowFloats)
}
func (c cRec) MHA(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return c.r.MHA(cb(q), cb(k), cb(v), cb(o), sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
}
func (c cRec) Unary(x, o buffer, op int) error { return c.r.Unary(cb(x), cb(o), op) }
func (c cRec) Binary(a, b, o buffer, op int) error {
	return c.r.Binary(cb(a), cb(b), cb(o), op)
}
func (c cRec) QMatMulResident(x buffer, w qweight, o buffer, m int) error {
	return c.r.QMatMulResident(cb(x), w.(*cuda.ResidentBQ8), cb(o), m)
}
func (c cRec) Commit() error { return c.r.Commit() }
func (c cRec) Wait() error   { return c.r.Wait() }
func (c cRec) Finish() error { return c.r.Finish() }
func (c cRec) Free()         { c.r.Free() }

func cb(b buffer) *cuda.DeviceF32 { return b.(cBuf).DeviceF32 }

// NewCUDA uploads an nlp.Llama's weights into CUDA device buffers and prepares a KV cache up to
// m.Config.Ctx tokens — the NVIDIA variant of New/NewVulkan. It reuses the backend-agnostic batched
// Decoder core (§T409) via the cuda.Recorder, so Generate/Step/StepN and every sampler work
// unchanged. f32 path (quantized NewQuantCUDA is a follow-up: it needs cuda.Recorder.QMatMulResident).
func NewCUDA(m *nlp.Llama) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false, // one shared work stream — pre-encoding a second recorder would interleave
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewQwen2CUDA uploads a Qwen2 / Qwen2.5 model onto the batched Decoder core. Qwen2 shares
// nlp.Llama (SwiGLU MLP, RMSNorm, GQA, full rope) and departs only in carrying q/k/v projection
// biases (o_proj has none) — the newDecoder core adds them via qkvBias when b.Bq/Bk/Bv are set, so
// this is exactly NewCUDA with a Qwen2-typed entry point. Load with nlp.LlamaFromHF; the biases are
// picked up automatically from the checkpoint. (Qwen3's per-head QK-norm is a separate follow-up.)
func NewQwen2CUDA(m *nlp.Llama) (*Decoder, error) { return NewCUDA(m) }

// NewQwen3CUDA uploads a Qwen3 model onto the batched Decoder core. Qwen3 shares nlp.Llama and adds
// per-head RMSNorm on Q and K before RoPE (b.QNorm/b.KNorm), which newDecoder wires into recordQKNorm
// — it drops Qwen2's q/k/v projection bias, so this is again NewCUDA with a Qwen3-typed entry point.
// Load with nlp.LlamaFromHF; the QK-norm gains are picked up automatically from the checkpoint.
func NewQwen3CUDA(m *nlp.Llama) (*Decoder, error) { return NewCUDA(m) }

// NewPhi3CUDA uploads a Phi-3 model onto the batched Decoder core. Phi-3 is structurally a plain
// Llama (RMSNorm, SwiGLU, GQA, full rope, no biases) — nlp.Phi3FromHF just unpacks its row-packed
// qkv_proj / gate_up_proj into the standard projections and returns an *nlp.Llama — so this is
// NewCUDA with a Phi-3-typed entry point. Distinct from NewPhiCUDA (the older Phi-1/1.5/2 family,
// which is parallel-residual with LayerNorm and partial rotary).
func NewPhi3CUDA(m *nlp.Llama) (*Decoder, error) { return NewCUDA(m) }

// NewGraniteCUDA uploads an IBM Granite model onto the batched Decoder core. Granite is a plain
// Llama plus four learned-at-config scalars — embedding_multiplier, attention_multiplier,
// residual_multiplier and logits_scaling — which newDecoder folds into the upload (embedding gather
// scale, softmax scale override, Wo/Wdown pre-scale, and 1/logits_scaling into the lm_head), so this
// is NewCUDA with a Granite-typed entry point. Load with nlp.GraniteFromHF.
func NewGraniteCUDA(m *nlp.Llama) (*Decoder, error) { return NewCUDA(m) }

// NewStableLMCUDA uploads an nlp.StableLM into CUDA device buffers and runs it through the same
// batched Decoder core as NewCUDA — but with LayerNorm-with-bias norms and PARTIAL rotary (the
// StableLM/Phi/StarCoder2-class departures from Llama). The first of the new-architecture GPU
// graph decoders enabled by the cu_rope_partial* kernels. cuda-only for now.
func NewStableLMCUDA(m *nlp.StableLM) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newStableLMDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewStarCoder2CUDA uploads an nlp.StarCoder2 onto the batched Decoder core: LayerNorm-with-bias,
// biased q/k/v/o projections, a biased 2-layer GELU MLP, full rope and GQA. The second new-arch GPU
// graph decoder, exercising every Decoder-core generalization except partial rotary. cuda-only.
func NewStarCoder2CUDA(m *nlp.StarCoder2) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newStarCoder2Decoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewPhiCUDA uploads an nlp.Phi onto the batched Decoder core: one-norm parallel residual, biased
// q/k/v/dense projections, a biased GELU MLP, partial rotary, a biased untied lm_head and a final
// LayerNorm — every Decoder-core generalization at once. The third new-arch GPU graph decoder.
func NewPhiCUDA(m *nlp.Phi) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newPhiDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// NewGPTNeoXCUDA uploads an nlp.GPTNeoX onto the batched Decoder core: TWO-norm parallel residual
// (input_layernorm → attn and post_attention_layernorm → mlp, both over the raw residual),
// LayerNorm-with-bias, biased q/k/v/dense projections, a biased GELU MLP, (partial or full) rotary
// and an untied unbiased lm_head. The fourth new-arch GPU graph decoder — the two-norm sibling of
// Phi's one-norm parallel residual. cuda-only.
func NewGPTNeoXCUDA(m *nlp.GPTNeoX) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newGPTNeoXDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
	})
}

// cudaUploadQWeight makes a ggml quantized [Out,In] weight resident on the GPU as Q8.
// The CUDA backend has ONE quant kernel (Q8 GEMV), so — unlike metal/vulkan, which keep
// each native ggml type and dequantize in-kernel — any source type (Q4_K, Q6_K, …) is
// dequantized to f32 and re-quantized to a uniform resident Q8_0. That's a faithful Q8
// inference path (Q8→f32→Q8 is near-lossless; lower-bit sources lose only what they
// already lost). NewResidentBQ8 wants [In,Out], so the dequantized [Out,In] is transposed.
func cudaUploadQWeight(weight []byte, qt uint32, n, k int) (qweight, error) {
	f32, err := gguf.QuantTensor{Data: weight, GGType: qt, Shape: tensor.Shape{n, k}}.Dequantize()
	if err != nil {
		return nil, fmt.Errorf("llamagpu: CUDA dequant qt=%d [%d,%d]: %w", qt, n, k, err)
	}
	tin, err := f32.Transpose(0, 1) // [Out,In] → [In,Out]
	if err != nil {
		return nil, fmt.Errorf("llamagpu: CUDA weight transpose: %w", err)
	}
	rw, err := cuda.NewResidentBQ8(tin)
	if err != nil {
		return nil, fmt.Errorf("llamagpu: CUDA Q8 upload: %w", err)
	}
	return rw, nil
}

// NewQuantCUDA uploads a quantized Llama with every projection as a resident Q8 weight
// (source types re-quantized to Q8) — the NVIDIA variant of NewQuant/NewQuantVulkan.
// Same batched Decoder core, so Generate/Step and every sampler work unchanged.
func NewQuantCUDA(m *nlp.QuantLlama) (*Decoder, error) {
	if !cuda.Available() {
		return nil, fmt.Errorf("llamagpu: no CUDA GPU")
	}
	return newQuantDecoder(m, backendOps{
		name:        string(backend.CUDA),
		asyncEncode: false,
		newBuffer: func(data []float32) (buffer, error) {
			b, err := cuda.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return cBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := cuda.NewRecorder()
			if err != nil {
				return nil, err
			}
			return cRec{r}, nil
		},
		uploadQWeight: cudaUploadQWeight,
	})
}
