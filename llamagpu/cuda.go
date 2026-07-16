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
