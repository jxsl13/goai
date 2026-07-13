//go:build darwin && cgo

// Further reading: Pope et al. 2022, "Efficiently Scaling Transformer Inference", for the batched-decode systems view; Dao et al. 2022, "FlashAttention", for the memory-aware attention lineage behind the cooperative kernels.
package llamagpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
)

// metal adapter (§T409): thin assertions from the backend-agnostic buffer/recorder interfaces to
// the concrete metal.DeviceBuffer/metal.Recorder (whose API vulkan mirrors method-for-method).

type mBuf struct{ *metal.DeviceBuffer }

type mRec struct{ r *metal.Recorder }

func (m mRec) RMSNorm(x, g, o buffer, rows, dim int, eps float32) error {
	return m.r.RMSNorm(mb(x), mb(g), mb(o), rows, dim, eps)
}
func (m mRec) LayerNorm(x, g, bb, o buffer, rows, dim int, eps float32) error {
	return m.r.LayerNorm(mb(x), mb(g), mb(bb), mb(o), rows, dim, eps)
}
func (m mRec) AddBias(x, bb, o buffer, rows, n int) error {
	return m.r.AddBias(mb(x), mb(bb), mb(o), rows, n)
}
func (m mRec) MatMul(a, b, c buffer, mm, k, n int) error {
	return m.r.MatMul(mb(a), mb(b), mb(c), mm, k, n)
}
func (m mRec) RoPE(q, inv, o buffer, seq, width, heads, hd, half, pos int, posDiv float32) error {
	return m.r.RoPE(mb(q), mb(inv), mb(o), seq, width, heads, hd, half, pos, posDiv)
}
func (m mRec) Blit(src buffer, srcOff int, dst buffer, dstOff, n int) error {
	return m.r.Blit(mb(src), srcOff, mb(dst), dstOff, n)
}
func (m mRec) MHA(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error {
	return m.r.MHA(mb(q), mb(k), mb(v), mb(o), sq, sk, dm, heads, kvHeads, dk, causal, window, scale)
}
func (m mRec) Unary(x, o buffer, op int) error { return m.r.Unary(mb(x), mb(o), op) }
func (m mRec) Binary(a, b, o buffer, op int) error {
	return m.r.Binary(mb(a), mb(b), mb(o), op)
}
func (m mRec) QMatMulResident(x buffer, w qweight, o buffer, mm int) error {
	return m.r.QMatMulResident(mb(x), w.(*metal.ResidentQWeight), mb(o), mm)
}
func (m mRec) Finish() error { return m.r.Finish() }
func (m mRec) Free()         { m.r.Free() }

func mb(b buffer) *metal.DeviceBuffer { return b.(mBuf).DeviceBuffer }

// New uploads m's weights into metal device buffers and prepares a KV cache up to m.Config.Ctx
// tokens (the 24× batched decode, §T404).
func New(m *nlp.Llama) (*Decoder, error) {
	if !metal.Available() {
		return nil, fmt.Errorf("llamagpu: no metal GPU")
	}
	return newDecoder(m, backendOps{
		name: string(backend.Metal),
		newBuffer: func(data []float32) (buffer, error) {
			b, err := metal.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return mBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := metal.NewRecorder()
			if err != nil {
				return nil, err
			}
			return mRec{r}, nil
		},
		uploadQWeight: metalUploadQWeight,
	})
}

// NewGPT uploads an nlp.GPT's weights into metal device buffers for batched decoding — the
// GPT-2-style sibling of New (§T422).
func NewGPT(m *nlp.GPT) (*GPTDecoder, error) {
	if !metal.Available() {
		return nil, fmt.Errorf("llamagpu: no metal GPU")
	}
	return newGPTDecoder(m, backendOps{
		name: string(backend.Metal),
		newBuffer: func(data []float32) (buffer, error) {
			b, err := metal.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return mBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := metal.NewRecorder()
			if err != nil {
				return nil, err
			}
			return mRec{r}, nil
		},
	})
}

func metalUploadQWeight(weight []byte, qt uint32, n, k int) (qweight, error) {
	rw, err := metal.Backend{}.UploadQuant(weight, qt, n, k)
	if err != nil {
		return nil, err
	}
	return rw.(*metal.ResidentQWeight), nil
}

// NewQuant uploads a quantized Llama (nlp.QuantizeLlama / nlp.QuantLlamaFromGGUF) with every
// projection as a device-resident quantized weight — the 4-8× smaller weights of quantization
// combined with the batched-decode speedup (§T415). Metal.
func NewQuant(m *nlp.QuantLlama) (*Decoder, error) {
	if !metal.Available() {
		return nil, fmt.Errorf("llamagpu: no metal GPU")
	}
	return newQuantDecoder(m, backendOps{
		name: string(backend.Metal),
		newBuffer: func(data []float32) (buffer, error) {
			b, err := metal.NewDeviceBufferF32(data)
			if err != nil {
				return nil, err
			}
			return mBuf{b}, nil
		},
		newRecorder: func() (recorder, error) {
			r, err := metal.NewRecorder()
			if err != nil {
				return nil, err
			}
			return mRec{r}, nil
		},
		uploadQWeight: metalUploadQWeight,
	})
}
